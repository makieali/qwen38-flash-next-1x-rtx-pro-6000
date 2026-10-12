"""Assemble one ModelScope file from reusable parts + parallel range downloads, then verify SHA-256."""
import hashlib, json, os, re, sys, urllib.request, concurrent.futures as cf, threading
REPO, NAME, DIR = "trohrbaugh/Qwen3.8-Flash-Next-heretic", sys.argv[1], "/srv/ai/models/flash-next-heretic-bf16"
meta = next(f for f in json.load(open("/tmp/ms-files.json"))["Data"]["Files"] if f["Path"] == NAME)
SIZE, SHA, CH = meta["Size"], meta["Sha256"], 167772160
URL = f"https://www.modelscope.cn/models/{REPO}/resolve/master/{NAME}"
out = f"{DIR}/{NAME}.assembling"
fd = os.open(out, os.O_RDWR | os.O_CREAT, 0o644); os.ftruncate(fd, SIZE)
lock, done = threading.Lock(), [0]

def chunk(a):
    b = min(a + CH, SIZE) - 1; part = f"{DIR}/{NAME}_{a}_{b}"
    if os.path.exists(part) and os.path.getsize(part) == b - a + 1:
        data = open(part, "rb").read()
    else:
        for i in range(8):
            try:
                data = urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read()
                if len(data) == b - a + 1:
                    break
            except Exception:
                if i == 7:
                    raise
    os.pwrite(fd, data, a)
    with lock:
        done[0] += len(data)
        if done[0] // (10 << 30) != (done[0] - len(data)) // (10 << 30):
            print(f"{done[0]/1e9:.0f} / {SIZE/1e9:.0f} GB", flush=True)

with cf.ThreadPoolExecutor(16) as ex:
    list(ex.map(chunk, range(0, SIZE, CH)))
os.fsync(fd); os.close(fd)
h = hashlib.sha256()
with open(out, "rb") as f:
    while b := f.read(64 << 20):
        h.update(b)
ok = h.hexdigest() == SHA
print("SHA256", "OK" if ok else "MISMATCH", flush=True)
if ok:
    os.replace(out, f"{DIR}/{NAME}")
    for fn in os.listdir(DIR):
        if re.match(re.escape(NAME) + r"(_\d+_\d+|\.incomplete)$", fn):
            os.remove(f"{DIR}/{fn}")
    print("ASSEMBLED", NAME, flush=True)
