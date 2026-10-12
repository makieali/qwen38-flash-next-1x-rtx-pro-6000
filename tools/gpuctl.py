"""GPU control daemon (runs as root). The only thing that touches fan speed and power limit.

Listens on 127.0.0.1:8766 (localhost only, no authentication: put your own authenticated front end in front of it)
for JSON commands, applies them through NVML, and runs a 2-second loop that drives the quiet fan curve and the safety override:
at SAFETY_TEMP_C or above the fans always go back to NVIDIA's automatic control, whatever mode was chosen.
Settings persist in /etc/gpuctl.json and are re-applied at start.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE_FILE = "/etc/gpuctl.json"
PORT = 8766
SAFETY_TEMP_C = 80            # at or above this, fans are always automatic
FAN_MIN, FAN_MAX = 30, 100    # what the card accepts
POWER_MIN, POWER_MAX = 150, 600
# Quiet curve: (temperature °C, fan %). Below the first point the minimum; linear in between.
QUIET_CURVE = [(50, 30), (60, 35), (68, 45), (75, 60), (79, 75)]
DEFAULTS = {"fan_mode": "auto", "fan_percent": 40, "power_limit_w": 600}
RESTARTABLE = {"engine": "flashnext-sglang"}  # nothing else


def restart_unit(name: str) -> dict:
    """Restart one of the approved services (asynchronously, so the caller gets an answer)."""
    unit = RESTARTABLE.get(name)
    if not unit:
        raise ValueError(f"restart must be one of {sorted(RESTARTABLE)}")
    import subprocess
    subprocess.Popen(["systemctl", "restart", "--no-block", unit])
    return {"restarting": unit}


def quiet_fan(temp_c: float) -> int:
    """Fan % for the quiet curve."""
    if temp_c <= QUIET_CURVE[0][0]:
        return QUIET_CURVE[0][1]
    for (t0, f0), (t1, f1) in zip(QUIET_CURVE, QUIET_CURVE[1:]):
        if temp_c <= t1:
            return round(f0 + (f1 - f0) * (temp_c - t0) / (t1 - t0))
    return QUIET_CURVE[-1][1]


def decide_fan(mode: str, manual_percent: int, temp_c: float):
    """Return ("auto", None) or ("set", percent). The safety override wins over every mode."""
    if temp_c is None or temp_c >= SAFETY_TEMP_C or mode == "auto":
        return ("auto", None)
    if mode == "quiet":
        return ("set", quiet_fan(temp_c))
    return ("set", max(FAN_MIN, min(FAN_MAX, int(manual_percent))))


def validate(cmd: dict, state: dict) -> dict:
    """Merge a command into a copy of the state, rejecting anything out of range."""
    new = dict(state)
    if "fan_mode" in cmd:
        if cmd["fan_mode"] not in ("auto", "quiet", "manual"):
            raise ValueError("fan_mode must be auto, quiet or manual")
        new["fan_mode"] = cmd["fan_mode"]
    if "fan_percent" in cmd:
        p = int(cmd["fan_percent"])
        if not FAN_MIN <= p <= FAN_MAX:
            raise ValueError(f"fan_percent must be {FAN_MIN}-{FAN_MAX}")
        new["fan_percent"] = p
    if "power_limit_w" in cmd:
        w = int(cmd["power_limit_w"])
        if not POWER_MIN <= w <= POWER_MAX:
            raise ValueError(f"power_limit_w must be {POWER_MIN}-{POWER_MAX}")
        new["power_limit_w"] = w
    return new


class Gpu:
    def __init__(self):
        import pynvml as n
        self.n = n
        n.nvmlInit()
        self.h = n.nvmlDeviceGetHandleByIndex(0)
        self.fans = n.nvmlDeviceGetNumFans(self.h)
        self.applied_fan = None

    def temp(self):
        return self.n.nvmlDeviceGetTemperature(self.h, self.n.NVML_TEMPERATURE_GPU)

    def set_fan(self, action, percent):
        if (action, percent) == self.applied_fan:
            return
        for i in range(self.fans):
            if action == "auto":
                self.n.nvmlDeviceSetDefaultFanSpeed_v2(self.h, i)
            else:
                self.n.nvmlDeviceSetFanSpeed_v2(self.h, i, percent)
        self.applied_fan = (action, percent)

    def set_power(self, watts):
        self.n.nvmlDeviceSetPowerManagementLimit(self.h, int(watts) * 1000)

    def power_limit(self):
        return self.n.nvmlDeviceGetPowerManagementLimit(self.h) / 1000

    def fan_status(self):
        return [self.n.nvmlDeviceGetFanSpeed_v2(self.h, i) for i in range(self.fans)]


class Controller:
    def __init__(self, gpu):
        self.gpu = gpu
        self.lock = threading.Lock()
        self.state = dict(DEFAULTS)
        if os.path.exists(STATE_FILE):
            try:
                self.state = validate(json.load(open(STATE_FILE)), DEFAULTS)
            except (ValueError, OSError, json.JSONDecodeError):
                pass
        self.override = False
        self.gpu.set_power(self.state["power_limit_w"])

    def save(self):
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh)
        os.replace(tmp, STATE_FILE)

    def apply(self, cmd):
        with self.lock:
            new = validate(cmd, self.state)
            if new["power_limit_w"] != self.state["power_limit_w"]:
                self.gpu.set_power(new["power_limit_w"])
            self.state = new
            self.save()
        self.tick()
        return self.status()

    def tick(self):
        with self.lock:
            t = self.gpu.temp()
            action, pct = decide_fan(self.state["fan_mode"], self.state["fan_percent"], t)
            self.override = self.state["fan_mode"] != "auto" and action == "auto"
            self.gpu.set_fan(action, pct)

    def status(self):
        return {**self.state, "temp_c": self.gpu.temp(), "fans_percent": self.gpu.fan_status(),
                "power_limit_now_w": self.gpu.power_limit(), "safety_override": self.override,
                "safety_temp_c": SAFETY_TEMP_C, "fan_range": [FAN_MIN, FAN_MAX], "power_range": [POWER_MIN, POWER_MAX],
                "quiet_curve": QUIET_CURVE}

    def loop(self):
        while True:
            try:
                self.tick()
            except Exception:
                try:
                    self.gpu.set_fan("auto", None)  # anything odd: hand fans back to the card
                except Exception:
                    pass
            time.sleep(2)


def serve(ctl):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._send(200, ctl.status())

        def do_POST(self):  # noqa: N802
            try:
                cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)) or b"{}")
                if "restart" in cmd:
                    self._send(200, restart_unit(str(cmd["restart"])))
                    return
                self._send(200, ctl.apply(cmd))
            except (ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    controller = Controller(Gpu())
    threading.Thread(target=controller.loop, daemon=True).start()
    serve(controller)
