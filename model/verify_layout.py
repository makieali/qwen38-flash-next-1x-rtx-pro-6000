import json, struct, urllib.request, torch, sys
sys.path.insert(0, "/srv/ai/quant"); from nvfp4 import dequant, quant
from safetensors import safe_open
BASE = "https://www.modelscope.cn/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/resolve/master/"
DT = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}
def get(url, a, b):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read()
def remote_tensors(fname, keys):
    url = BASE + fname; n = struct.unpack("<Q", get(url, 0, 7))[0]; hdr = json.loads(get(url, 8, 8 + n - 1)); out = {}
    for k in keys:
        m = hdr[k]; a, b = m["data_offsets"]; raw = get(url, 8 + n + a, 8 + n + b - 1)
        out[k] = torch.frombuffer(bytearray(raw), dtype=DT[m["dtype"]]).reshape(m["shape"]) if m["shape"] else torch.frombuffer(bytearray(raw), dtype=DT[m["dtype"]])[0]
    return out
L, E = 3, 0
p = f"model.language_model.layers.{L}.mlp.experts.{E}."
keys = [p + f"{w}.{t}" for w in ("gate_proj", "up_proj", "down_proj") for t in ("weight", "weight_scale", "weight_scale_2", "input_scale")]
idx = json.load(open("/srv/ai/models/ref-radixark/model.safetensors.index.json"))["weight_map"]
R = remote_tensors(idx[keys[0]], keys)
H = json.load(open("/srv/ai/models/flash-next-heretic-bf16/model.safetensors.index.json"))["weight_map"]
hk = f"model.language_model.layers.{L}.mlp.experts."
d = "/srv/ai/models/flash-next-heretic-bf16/"
gu = safe_open(d + H[hk + "gate_up_proj"], "pt").get_slice(hk + "gate_up_proj")[E].float()
dn = safe_open(d + H[hk + "down_proj"], "pt").get_slice(hk + "down_proj")[E].float()
print("heretic gate_up", tuple(gu.shape), "down", tuple(dn.shape))
def cos(a, b): return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
rg = dequant(R[p + "gate_proj.weight"], R[p + "gate_proj.weight_scale"], R[p + "gate_proj.weight_scale_2"])
ru = dequant(R[p + "up_proj.weight"], R[p + "up_proj.weight_scale"], R[p + "up_proj.weight_scale_2"])
rd = dequant(R[p + "down_proj.weight"], R[p + "down_proj.weight_scale"], R[p + "down_proj.weight_scale_2"])
cands = {"[2I,H] first half": gu[:640], "[2I,H] second half": gu[640:], "[H,2I]^T first": gu.T[:640] if gu.shape[0] == 2560 else None,
         "[H,2I]^T second": gu.T[640:] if gu.shape[0] == 2560 else None}
for name, c in cands.items():
    if c is not None and c.shape == rg.shape: print(f"gate vs {name}: {cos(rg, c):.4f}   up vs {name}: {cos(ru, c):.4f}")
for name, c in {"down as is": dn, "down ^T": dn.T}.items():
    if c.shape == rd.shape: print(f"down vs {name}: {cos(rd, c):.4f}")
print("radixark scales: gate s2", R[p + "gate_proj.weight_scale_2"].item(), "up s2", R[p + "up_proj.weight_scale_2"].item(),
      "| input_scale gate/up/down", R[p + "gate_proj.input_scale"].item(), R[p + "up_proj.input_scale"].item(), R[p + "down_proj.input_scale"].item())
gate = gu[:640] if gu.shape[0] == 1280 else gu.T[:640]
pk, s, s2 = quant(gate, R[p + "gate_proj.weight_scale_2"])
print("my quant vs radixark: packed bytes equal", f"{(pk == R[p + 'gate_proj.weight']).float().mean().item():.3f}",
      "| scales equal", f"{(s.float() == R[p + 'gate_proj.weight_scale'].float()).float().mean().item():.3f}",
      "| my dequant cos to bf16", f"{cos(dequant(pk, s, s2), gate):.4f}", "| radixark dequant cos to bf16", f"{cos(rg, gate):.4f}")
