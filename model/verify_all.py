import json, os, hashlib, concurrent.futures as cf
D = "/srv/ai/models/flash-next-heretic-bf16"
fs = [f for f in json.load(open("/tmp/ms-files.json"))["Data"]["Files"] if f["Type"] == "blob" and f["Path"] != "model-00002-of-00033.safetensors"]
def chk(f):
    p = os.path.join(D, f["Path"])
    if not os.path.exists(p): return (f["Path"], "missing")
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(64 << 20): h.update(b)
    return (f["Path"], "ok" if h.hexdigest() == f["Sha256"] else "SHA MISMATCH")
with cf.ThreadPoolExecutor(12) as ex: res = list(ex.map(chk, fs))
bad = [r for r in res if r[1] != "ok"]
print(len(res), "files checked:", "ALL SHA256 OK" if not bad else bad, flush=True)
