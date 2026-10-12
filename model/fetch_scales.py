"""Collect RadixArk calibrated per-expert input_scale (and weight_scale_2 for comparison) via HTTP range reads."""
import json, struct, urllib.request, concurrent.futures as cf, re
BASE = "https://www.modelscope.cn/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/resolve/master/"
idx = json.load(open("/srv/ai/models/ref-radixark/model.safetensors.index.json"))["weight_map"]
files = sorted({f for k, f in idx.items() if ".mlp.experts." in k and k.startswith("model.")})
want = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.(input_scale|weight_scale_2)$")
def get(url, a, b, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=60).read()
        except Exception:
            if i == tries - 1: raise
def one(fname):
    url = BASE + fname; n = struct.unpack("<Q", get(url, 0, 7))[0]; hdr = json.loads(get(url, 8, 8 + n - 1))
    items = sorted((m["data_offsets"][0], k) for k, m in hdr.items() if k != "__metadata__" and want.search(k))
    out, i = {}, 0
    while i < len(items):                       # merge neighbours less than 64 KB apart into one request
        j = i
        while j + 1 < len(items) and items[j + 1][0] - items[j][0] < 65536: j += 1
        a, b = items[i][0], items[j][0] + 4
        raw = get(url, 8 + n + a, 8 + n + b - 1)
        for off, k in items[i:j + 1]: out[k] = struct.unpack("<f", raw[off - a:off - a + 4])[0]
        i = j + 1
    return out
res = {}
with cf.ThreadPoolExecutor(16) as ex:
    for d in ex.map(one, files): res.update(d)
json.dump(res, open("/srv/ai/quant/radixark_scales.json", "w"))
n_in = sum(1 for k in res if k.endswith("input_scale"))
print(len(files), "files,", len(res), "scalars,", n_in, "input_scales (expected", 48 * 512 * 3, ")")
