"""Draw the README images (SVG) from results/*.json. Run from the repo root: python tools/make_images.py"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "images"
FONT = "font-family=\"-apple-system,Segoe UI,Helvetica,Arial,sans-serif\""
C = {"blue": "#2563eb", "green": "#16a34a", "orange": "#ea580c", "purple": "#7c3aed", "grey": "#64748b",
     "teal": "#0d9488", "red": "#dc2626", "ink": "#0f172a", "muted": "#475569", "line": "#e2e8f0", "bg": "#ffffff"}


def hi(v):
    """Upper end of a value or [low, high] range."""
    return max(v) if isinstance(v, list) else v


def label(v):
    return f"{v[0]}–{v[1]}" if isinstance(v, list) else f"{v:g}"


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{escape(title)}"><title>{escape(title)}</title>'
            f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="14" fill="{C["bg"]}" stroke="{C["line"]}"/>'
            f'<g {FONT}>{body}</g></svg>\n')


def text(x, y, s, size=14, color=C["ink"], weight="normal", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(str(s))}</text>')


def box(x, y, w, h, fill, stroke=None, rx=8, opacity=1):
    st = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" fill-opacity="{opacity}"{st}/>'


def arrow(x1, y1, x2, y2, color=C["muted"]):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


MARKER = (f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
          f'<path d="M0,0 L10,5 L0,10 z" fill="{C["muted"]}"/></marker></defs>')


def speed_journey():
    d = json.load(open(RES / "engine-steps.json"))["steps"]
    names = ["llama.cpp\n(Windows, GGUF)", "vLLM\n+ our NVFP4", "SGLang\n+ our NVFP4", "SGLang\n+ online MXFP8"]
    series = [("chat_1_user", "Chat, 1 user", C["blue"]), ("code_1_user", "Code, 1 user", C["green"]),
              ("users_4_total", "4 users, total", C["orange"])]
    W, H, L, B, T = 960, 470, 70, 380, 70
    top = 500; scale = (B - T) / top
    body = text(30, 40, "Same GPU, same model: 45 → 185 tokens/s for one user, 481 for four", 20, weight="bold")
    for v in range(0, top + 1, 100):
        y = B - v * scale
        body += f'<line x1="{L}" y1="{y}" x2="{W-30}" y2="{y}" stroke="{C["line"]}"/>' + text(L - 10, y + 5, v, 12, C["muted"], anchor="end")
    group = (W - L - 40) / 4; bw = 52
    for i, step in enumerate(d):
        gx = L + 20 + i * group
        for j, (k, _, col) in enumerate(series):
            v = step[k]; x = gx + j * (bw + 8); h = hi(v) * scale
            body += box(x, B - h, bw, h, col, rx=4) + text(x + bw / 2, B - h - 7, label(v), 12, C["ink"], "bold", "middle")
        for n, line in enumerate(names[i].split("\n")):
            body += text(gx + 1.5 * bw + 8, B + 22 + n * 17, line, 13, C["ink"] if n == 0 else C["muted"], "bold" if n == 0 else "normal", "middle")
    for j, (_, name, col) in enumerate(series):
        x = L + 20 + j * 170
        body += box(x, 52, 14, 14, col, rx=3) + text(x + 20, 64, name, 13, C["muted"])
    body += text(W - 30, H - 12, "output tokens/s, streaming, unique prompts · RTX PRO 6000 96 GB · results/engine-steps.json", 11, C["muted"], anchor="end")
    return svg(W, H, body, "Speed per engine step")


def concurrency():
    d = json.load(open(RES / "concurrency-2026-10-12.json"))["runs"]
    runs = [r for r in d if "warm-up" not in r.get("note", "")]
    W, H, L, B, T = 960, 400, 70, 300, 80
    top = 500; scale = (B - T) / top
    body = text(30, 40, "Concurrency: total throughput and wait for the first word", 20, weight="bold")
    body += text(30, 62, "max-running-requests 8, but the state cache (24 slots, 5 per request) caps it at 4 → users 5–8 queue", 13, C["red"])
    for v in range(0, top + 1, 100):
        y = B - v * scale
        body += f'<line x1="{L}" y1="{y}" x2="{W-330}" y2="{y}" stroke="{C["line"]}"/>' + text(L - 10, y + 5, v, 12, C["muted"], anchor="end")
    bw = 110
    for i, r in enumerate(runs):
        x = L + 40 + i * 190; h = hi(r["total_tok_s"]) * scale
        body += box(x, B - h, bw, h, C["orange"], rx=4) + text(x + bw / 2, B - h - 8, f'{label(r["total_tok_s"])} tok/s', 13, C["ink"], "bold", "middle")
        body += text(x + bw / 2, B + 22, f'{r["users"]} user' + ("s" if r["users"] > 1 else ""), 14, C["ink"], "bold", "middle")
        body += text(x + bw / 2, B + 42, f'first word {label(r["ttft_s"])} s', 12, C["muted"], anchor="middle")
        body += text(x + bw / 2, B + 60, f'running {r["running_peak"]} · queued {r["queued_peak"]}', 12, C["red"] if r["queued_peak"] else C["muted"], anchor="middle")
    px = W - 300
    body += box(px, 95, 270, 190, "#f8fafc", C["line"]) + text(px + 16, 122, "Proposed fix (not yet measured)", 15, weight="bold")
    for n, s in enumerate(["--max-mamba-cache-size 40", "  (5 slots × 8 requests)", "--max-running-requests 8",
                           "--mem-fraction-static 0.95", "  (825k KV tokens, not 471k)"]):
        body += text(px + 16, 150 + n * 24, s, 13, C["ink"] if not s.startswith("  ") else C["muted"])
    body += text(W - 30, H - 12, "SGLang + MXFP8, 2026-10-12 · results/concurrency-2026-10-12.json", 11, C["muted"], anchor="end")
    return svg(W, H, body, "Concurrency results")


def mtp_depth():
    d = json.load(open(RES / "mtp-depth-vllm.json"))["runs"]
    W, H, L, B, T = 640, 340, 60, 250, 70
    top = 400; scale = (B - T) / top
    body = text(30, 38, "MTP guess-ahead depth (vLLM)", 18, weight="bold") + text(30, 58, "k = 3 kept: best code and 4-user speed", 13, C["muted"])
    for v in range(0, top + 1, 100):
        y = B - v * scale
        body += f'<line x1="{L}" y1="{y}" x2="{W-30}" y2="{y}" stroke="{C["line"]}"/>' + text(L - 8, y + 5, v, 11, C["muted"], anchor="end")
    bw = 38
    for i, r in enumerate(d):
        gx = L + 30 + i * 180
        for j, (k, col) in enumerate((("chat", C["blue"]), ("code", C["green"]), ("users_4_total", C["orange"]))):
            x = gx + j * (bw + 6); h = r[k] * scale
            body += box(x, B - h, bw, h, col, rx=3, opacity=1 if r.get("kept") else 0.55) + text(x + bw / 2, B - h - 5, f'{r[k]:g}', 11, C["ink"], anchor="middle")
        body += text(gx + 1.5 * bw + 6, B + 22, f'k = {r["k"]}' + ("  ✓" if r.get("kept") else ""), 14, C["ink"], "bold", "middle")
        body += text(gx + 1.5 * bw + 6, B + 40, f'accept {r["acceptance_length"]}', 12, C["muted"], anchor="middle")
    body += text(W - 30, H - 10, "blue chat · green code · orange 4 users total", 11, C["muted"], anchor="end")
    return svg(W, H, body, "MTP depth")


def memory_map():
    W, H = 960, 445
    body = text(30, 40, "What lives where (production build)", 20, weight="bold")
    # VRAM bar (96 GB)
    x0, y0, bw = 30, 90, 900; s = bw / 96
    body += text(x0, y0 - 12, "GPU memory (VRAM) — 96 GB, ~90 GB used at --mem-fraction-static 0.95", 14, C["ink"], "bold")
    parts = [(68, "NVFP4 experts", "~68 GB", C["purple"]), (12, "other weights (FP8)", "~8–16 GB", C["blue"]),
             (10, "KV cache (FP8)", "~10 GB = 825k tokens", C["green"]), (4, "state cache + graphs", "", C["teal"]), (2, "", "", C["line"])]
    x = x0
    for gb, name, size, col in parts:
        w = gb * s; body += box(x, y0, w, 56, col, rx=0)
        if gb >= 30:  # only the big segment has room for its label
            body += text(x + 8, y0 + 24, name, 13, "#ffffff", "bold") + text(x + 8, y0 + 44, size, 12, "#ffffff")
        x += w
    lx = x0  # legend for every segment under the bar
    for gb, name, size, col in parts:
        if not name:
            continue
        body += box(lx, y0 + 68, 12, 12, col, rx=2) + text(lx + 18, y0 + 79, f"{name}  {size}".strip(), 12, C["muted"])
        lx += 230
    body += f'<rect x="{x0}" y="{y0}" width="{bw}" height="56" rx="8" fill="none" stroke="{C["muted"]}"/>'
    # DRAM bar (256 GB)
    y1 = 250; s2 = bw / 256
    body += text(x0, y1 - 12, "System RAM (DRAM) — 256 GB", 14, C["ink"], "bold")
    w = 102 * s2; body += box(x0, y1, w, 56, C["orange"], rx=0) + text(x0 + 8, y1 + 24, "PLE n-gram table", 13, "#ffffff", "bold") + text(x0 + 8, y1 + 44, "102 GB, pinned", 12, "#ffffff")
    body += box(x0 + w, y1, bw - w, 56, "#f1f5f9", rx=0) + text(x0 + w + 12, y1 + 34, "OS, page cache, other services, free (~140 GB)", 13, C["muted"])
    body += f'<rect x="{x0}" y="{y1}" width="{bw}" height="56" rx="8" fill="none" stroke="{C["muted"]}"/>'
    notes = ["• KV cache is small because only 12 of 48 layers use full attention; 36 Gated DeltaNet layers keep a fixed-size state.",
             "• Nothing is offloaded to DRAM except the PLE table: each token reads a few PLE rows over PCIe, cheap compared with the experts.",
             "• At --mem-fraction-static 0.90 the KV cache shrinks to 471,680 tokens (5.7 GB) with no speed gain."]
    for n, l in enumerate(notes):
        body += text(30, 345 + n * 24, l, 13, C["muted"])
    body += text(W - 30, H - 12, "nvidia-smi, SGLang /metrics, /proc · results/memory.json", 11, C["muted"], anchor="end")
    return svg(W, H, MARKER + body, "Memory map")


def conversion():
    W, H = 960, 330
    body = MARKER + text(30, 40, "Building our own NVFP4 checkpoint (experts only)", 20, weight="bold")
    def node(x, y, w, h, title, lines, col):
        b = box(x, y, w, h, col, rx=10, opacity=0.12) + f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" stroke="{col}" stroke-width="1.5"/>'
        b += text(x + 12, y + 24, title, 14, C["ink"], "bold")
        for n, l in enumerate(lines):
            b += text(x + 12, y + 46 + n * 19, l, 12, C["muted"])
        return b
    body += node(30, 70, 200, 120, "BF16 source", ["Qwen3.8 Flash-Next", "Heretic (uncensored)", "360 GB · 53 files", "SHA-256 verified"], C["grey"])
    body += node(280, 70, 240, 120, "Routed experts → NVFP4", ["gate / up / down only", "FP4 E2M1, group 16, FP8 scales", "RadixArk calibrated input_scale", "gate+up share weight_scale_2"], C["purple"])
    body += node(280, 200, 240, 100, "Kept 16-bit", ["attention (GDN + QSA), shared", "experts, router, embeddings,", "lm_head, vision, MTP head, PLE"], C["blue"])
    body += node(570, 70, 170, 120, "Checkpoint", ["186 GB", "ModelOpt layout", "rel. error 0.00893", "(reference 0.00892)"], C["green"])
    body += node(780, 70, 150, 230, "Load (SGLang)", ["experts: NVFP4", "CUTLASS MoE", "", "16-bit layers →", "MXFP8 online", "", "PLE → DRAM", "KV → FP8"], C["orange"])
    body += arrow(230, 130, 278, 130) + arrow(230, 160, 278, 240) + arrow(520, 130, 568, 130) + arrow(520, 250, 568, 170) + arrow(740, 130, 778, 130)
    return svg(W, H, body, "NVFP4 conversion pipeline")


def request_path():
    W, H = 960, 380
    body = MARKER + text(30, 40, "One decode step on one card", 20, weight="bold")
    steps = [("Next token", ["+ 3 drafted by MTP", "(FR-Spec: 64k vocab)"], C["grey"]),
             ("PLE lookup", ["n-gram rows from", "pinned DRAM (PCIe)"], C["orange"]),
             ("48 layers", ["36 × Gated DeltaNet", "(fixed state)", "12 × full attention", "(FP8 KV cache)"], C["blue"]),
             ("MoE per layer", ["router picks 10 of 512", "experts + shared expert", "NVFP4, CUTLASS kernel"], C["purple"]),
             ("Verify drafts", ["accept ~2.1 tokens", "per step on average", "→ ~185 tok/s"], C["green"])]
    x = 30
    for n, (title, lines, col) in enumerate(steps):
        w = 170
        body += box(x, 80, w, 170, col, rx=10, opacity=0.12) + f'<rect x="{x}" y="80" width="{w}" height="170" rx="10" fill="none" stroke="{col}" stroke-width="1.5"/>'
        body += text(x + 12, 106, title, 14, C["ink"], "bold")
        for k, l in enumerate(lines):
            body += text(x + 12, 132 + k * 20, l, 12, C["muted"])
        if n < len(steps) - 1:
            body += arrow(x + w, 165, x + w + 18, 165)
        x += w + 18
    for n, l in enumerate(["Why it's fast: only ~6B parameters are active per token, the active experts are 4-bit, the rest is FP8,",
                           "KV is FP8 and small, and speculative decoding turns one forward pass into ~2 tokens."]):
        body += text(30, 290 + n * 22, l, 13, C["muted"])
    return svg(W, H, body, "Decode step")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in (("speed-journey", speed_journey), ("concurrency", concurrency), ("mtp-depth", mtp_depth),
                     ("memory-map", memory_map), ("nvfp4-conversion", conversion), ("decode-step", request_path)):
        (OUT / f"{name}.svg").write_text(fn())
        print("wrote", f"images/{name}.svg")
