"""Heretic BF16 -> NVFP4 (ModelOpt/RadixArk layout): routed experts only; everything else copied as-is.
Usage: convert.py [shard-name ...]   (no args = all shards)"""
import json, os, re, shutil, sys, time, torch
from safetensors import safe_open
from safetensors.torch import save_file
SRC, OUT, REF = "/srv/ai/models/flash-next-heretic-bf16", "/srv/ai/models/flash-next-heretic-nvfp4", "/srv/ai/models/ref-radixark"
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
SC = json.load(open("/srv/ai/quant/radixark_scales.json"))
EXP = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts)\.(gate_up_proj|down_proj)$")

def quant_batch(w, scale2):
    """w [E, out, in] float32 cuda, scale2 [E] -> packed U8 [E,out,in/2], scale F8 [E,out,in/16]."""
    E_, out, inn = w.shape
    s2 = scale2.view(E_, 1, 1, 1)
    blocks = w.view(E_, out, inn // 16, 16)
    s = (blocks.abs().amax(-1, keepdim=True) / 6.0 / s2).clamp(max=448.0).to(torch.float8_e4m3fn)
    sd = s.float() * s2
    x = blocks / torch.where(sd == 0, torch.ones_like(sd), sd)
    mag = x.abs()
    idx = torch.zeros_like(mag, dtype=torch.uint8)
    for i in range(1, 8):                           # nearest E2M1 magnitude without a big temp tensor
        idx += (mag > (E2M1[i - 1] + E2M1[i]) / 2).to(torch.uint8)
    code = (idx + 8 * (x < 0).to(torch.uint8)).view(E_, out, inn)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous(), s.view(E_, out, inn // 16).contiguous()

def convert_experts(prefix, kind, t, out):
    E_ = t.shape[0]
    if kind == "gate_up_proj":
        w = t.cuda().float(); g, u = w[:, :640], w[:, 640:]
        s2 = torch.maximum(g.abs().amax((1, 2)), u.abs().amax((1, 2))) / (6.0 * 448.0)
        parts = (("gate_proj", g), ("up_proj", u))
    else:
        w = t.cuda().float(); s2 = w.abs().amax((1, 2)) / (6.0 * 448.0); parts = (("down_proj", w),)
    s2 = s2.clamp(min=1e-12)
    for name, ww in parts:
        pk, sc = quant_batch(ww.contiguous(), s2)
        pk, sc, s2c = pk.cpu(), sc.cpu(), s2.cpu()
        for e in range(E_):
            k = f"{prefix}.{e}.{name}"
            out[k + ".weight"] = pk[e].clone(); out[k + ".weight_scale"] = sc[e].clone()
            out[k + ".weight_scale_2"] = s2c[e].clone().reshape(())
            out[k + ".input_scale"] = torch.tensor(SC[k + ".input_scale"], dtype=torch.float32)
    del w; torch.cuda.empty_cache()

def main(only):
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(idx.values()))
    for sh in shards:
        if only and sh not in only: continue
        dst = f"{OUT}/{sh}"
        if os.path.exists(dst + ".done"): continue
        t0, out = time.time(), {}
        with safe_open(f"{SRC}/{sh}", "pt") as f:
            for k in f.keys():
                m = EXP.match(k)
                if m: convert_experts(m.group(1), m.group(3), f.get_tensor(k), out)
                else: out[k] = f.get_tensor(k)
        save_file(out, dst + ".tmp", metadata={"format": "pt"}); os.replace(dst + ".tmp", dst)
        json.dump(sorted(out), open(dst + ".done", "w"))
        print(f"{sh}: {len(out)} tensors, {os.path.getsize(dst)/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)
    if only: return
    wm = {}
    for sh in shards: wm.update({k: sh for k in json.load(open(f"{OUT}/{sh}.done"))})
    total = sum(os.path.getsize(f"{OUT}/{sh}") for sh in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": wm}, open(f"{OUT}/model.safetensors.index.json", "w"))
    for fn in os.listdir(SRC):
        if not fn.endswith(".safetensors") and fn != "model.safetensors.index.json" and os.path.isfile(f"{SRC}/{fn}") and "safetensors_" not in fn:
            shutil.copy(f"{SRC}/{fn}", f"{OUT}/{fn}")
    cfg = json.load(open(f"{OUT}/config.json")); cfg["quantization_config"] = json.load(open(f"{REF}/config.json"))["quantization_config"]
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2); shutil.copy(f"{REF}/hf_quant_config.json", f"{OUT}/hf_quant_config.json")
    print(f"DONE: {len(wm)} tensors, {total/1e9:.1f} GB")

if __name__ == "__main__":
    main(set(sys.argv[1:]))
