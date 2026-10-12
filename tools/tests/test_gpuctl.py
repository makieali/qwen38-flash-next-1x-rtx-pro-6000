"""GPU control logic: quiet curve, safety override, range checks (no GPU needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpuctl as g  # noqa: E402


def test_quiet_curve_is_monotonic_and_bounded():
    vals = [g.quiet_fan(t) for t in range(20, 80)]
    assert vals == sorted(vals)
    assert min(vals) == g.FAN_MIN and max(vals) <= 75


def test_safety_override_beats_every_mode():
    for mode in ("auto", "quiet", "manual"):
        assert g.decide_fan(mode, 30, g.SAFETY_TEMP_C) == ("auto", None)
        assert g.decide_fan(mode, 30, 92) == ("auto", None)
    assert g.decide_fan("manual", 30, None) == ("auto", None)  # unknown temperature: never pin the fans low


def test_modes_below_safety_temp():
    assert g.decide_fan("auto", 50, 40) == ("auto", None)
    assert g.decide_fan("quiet", 50, 40) == ("set", 30)
    assert g.decide_fan("manual", 55, 60) == ("set", 55)
    assert g.decide_fan("manual", 5, 60) == ("set", g.FAN_MIN)


def test_validate_rejects_out_of_range():
    s = dict(g.DEFAULTS)
    for bad in ({"fan_mode": "turbo"}, {"fan_percent": 10}, {"fan_percent": 101},
                {"power_limit_w": 100}, {"power_limit_w": 700}):
        with pytest.raises(ValueError):
            g.validate(bad, s)
    ok = g.validate({"fan_mode": "quiet", "power_limit_w": 450}, s)
    assert ok["fan_mode"] == "quiet" and ok["power_limit_w"] == 450 and s == g.DEFAULTS  # original untouched


def test_restart_only_approved_units(monkeypatch):
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda args: calls.append(args))
    assert g.restart_unit("engine") == {"restarting": "flashnext-sglang"}
    assert calls == [["systemctl", "restart", "--no-block", "flashnext-sglang"]]
    for bad in ("ssh", "flashnext-sglang", "../../bin/sh", ""):
        with pytest.raises(ValueError):
            g.restart_unit(bad)
