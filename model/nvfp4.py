"""NVFP4 (ModelOpt layout) helpers: E2M1 values, block-16 FP8-E4M3 scales, one FP32 global scale per tensor."""
import torch
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])

def dequant(packed, scale_fp8, scale2):
    """packed U8 [out, in/2], scale F8 [out, in/16], scale2 F32 scalar -> float32 [out, in] (low nibble = even element)."""
    lut = E2M1.to(packed.device)
    lo, hi = lut[(packed & 0x0F).long()], lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    s = scale_fp8.float().repeat_interleave(16, dim=1) * float(scale2)
    return vals * s

def quant(w, scale2=None):
    """float [out, in] -> (packed U8, scale F8_E4M3, scale2 F32). Round-to-nearest on the E2M1 grid."""
    w = w.float()
    out, inn = w.shape
    if scale2 is None:
        scale2 = w.abs().max() / (6.0 * 448.0)
    scale2 = torch.as_tensor(scale2, dtype=torch.float32, device=w.device).clamp(min=1e-12)
    blocks = w.view(out, inn // 16, 16)
    s = (blocks.abs().amax(-1) / 6.0 / scale2).clamp(max=448.0).to(torch.float8_e4m3fn)
    sd = s.float() * scale2
    x = blocks / torch.where(sd == 0, torch.ones_like(sd), sd)[..., None]
    grid = E2M1[:8].to(w.device)
    idx = (x.abs()[..., None] - grid).abs().argmin(-1)          # 0..7 magnitude code
    code = (idx + 8 * (x < 0).long()).to(torch.uint8).view(out, inn)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, s, scale2.float()
