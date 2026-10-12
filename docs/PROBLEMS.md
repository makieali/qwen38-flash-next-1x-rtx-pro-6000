# Problems we hit, and the fixes

On a chip this new, every fix tends to uncover the next problem. Listed in the order we met them.

## Getting the model

| Problem | Fix |
|---|---|
| The downloader reported "done" but the 102 GB PLE shard was missing (stored as 402 pieces of 160 MB; it stalled) | [`model/fetch_shard.py`](../model/fetch_shard.py): parallel HTTP range downloads with resume, then a SHA-256 check. Always verify every file ([`model/verify_all.py`](../model/verify_all.py)) |
| Some "mirror" sites just forward large files from the origin, slowly | Use providers that host the files themselves (e.g. ModelScope), and check the download speed in the first minute |

## vLLM stage

| # | Problem | Fix |
|---|---|---|
| 1 | `FLASHINFER_B12X does not support the current device` | vLLM pins `b12x` 1.3.0, which needs an older CUTLASS; install `b12x` 1.5.0 (matches vLLM 0.31's CUTLASS) |
| 2 | Same error, different cause: vLLM thought FlashInfer wasn't installed | vLLM looks for FlashInfer's prebuilt kernels *or* `nvcc`; put the venv's bundled `.../site-packages/nvidia/cu13/bin` on `PATH` |
| 3 | FlashInfer 0.7.0 crash with 512 experts (CUDA-graph padding rows = −1 read as an expert id) | FlashInfer 0.7.1rc4 (PR #5451). Never 0.7.0 with this model |
| 4 | `flashinfer_b12x is not supported for unquantized MoE` (the MTP layer has BF16 experts) | Give the speculative config its own backend: `"moe_backend": "triton"` |
| 5 | Out of memory at warm-up (+14 GB): one b12x workspace **per layer** | [`serving/vllm/patch_b12x.py`](../serving/vllm/patch_b12x.py): all 48 layers share the first layer's workspace (FlashInfer says identical layers may) |
| 6 | Out of memory again: "static" b12x pads expert width 640 → 768 and caches a padded copy of every layer | Force "dynamic" mode (multiples of 128): [`serving/vllm/patch_flashinfer_force_backend.py`](../serving/vllm/patch_flashinfer_force_backend.py) + `B12X_FORCE_BACKEND=dynamic`. The proper fix is an unmerged FlashInfer change; it's why vLLM's FP4 path stays slower than SGLang's here |
| 7 | `No such file or directory: ninja` at the end of startup | `ninja` is in the venv; add the venv's `bin` to `PATH` |
| 8 | `CUDA compiler and CUDA toolkit headers are incompatible` (pip gave nvcc 13.4 + headers 13.0) | Install the exact 13.0.88 compiler, crt and nvvm that PyTorch cu130 expects |
| 9 | `cannot find -lcudart` | pip CUDA ships only `libcudart.so.13`; add `libcudart.so` symlinks in a `lib64` folder. Prebuild FlashInfer's two runtime kernels once ([`serving/vllm/prebuild.py`](../serving/vllm/prebuild.py), 1.5 + 16 min) so restarts don't pay for it |
| 10 | Tool calls repeated about a third of the time (vLLM parser issue #58147) | Thinking on for tool requests: 40/40 |
| 11 | Stopping vLLM left a worker holding 83 GB of VRAM | `KillMode=control-group` + `ExecStopPost=pkill -9 -f VLLM::` in [`flashnext.service`](../deploy/systemd/flashnext.service) |

## SGLang stage

| Problem | Fix |
|---|---|
| Our config names the 12 sparse-attention layers `qwen_sparse_attention`; SGLang expects `full_attention` | Model view (symlinks) with a patched `config.json` ([`model/sglang-view-config.json`](../model/sglang-view-config.json)); same layers, older label |
| FR-Spec's 64k token map checks the tokenizer file's checksum | Proved the vocab identical (248,044 tokens, same ids on English/Urdu/Chinese/code/emoji) and used RadixArk's byte-identical `tokenizer.json` |
| JIT compile failed: Ubuntu 26.04's glibc declares `rsqrt` differently from the CUDA 13.0 headers | Separate CUDA 13.3 pip toolkit for SGLang, plus links to the cuBLAS headers PyTorch ships |
| The first fused-MoE request took ~10 minutes (JIT build) | Prebuild it once: [`serving/sglang/prebuild_moe.sh`](../serving/sglang/prebuild_moe.sh); it's cached afterwards |
| Two engines on one card | `Conflicts=flashnext.service` in [`flashnext-sglang.service`](../deploy/systemd/flashnext-sglang.service), so only one runs |
| `--max-running-requests 8` ran only 4 | Capped by the state cache (5 slots per request, 24 slots). See [TUNING.md](TUNING.md#concurrency) |

## Keeping it up

| Problem | Fix |
|---|---|
| The engine restarted itself one morning and was down for minutes | Ubuntu's `unattended-upgrades` installed a library update and `needrestart` restarted every service using it, including the engine (a 4–5 minute cold start). Fixes: `needrestart` in list-only mode (`/etc/needrestart/conf.d/` with `$nrconf{restart} = 'l';`); NVIDIA, CUDA and kernel packages blacklisted from unattended upgrades; the apt timers moved to a quiet hour. Update those by hand |
| GPU at 88–89 °C after an hour at 600 W with the default fan curve | [`tools/gpuctl.py`](../tools/gpuctl.py): fan modes with a safety override (always automatic at ≥ 80 °C) and a power cap |

## Measuring

| Problem | Fix |
|---|---|
| Benchmark showing 72,000 tok/s | Prefix cache: the same prompt was sent twice. Use unique prompts (our scripts add a UUID) |
| First multi-user run after a restart much slower (210 vs 461–476 tok/s, 7.9 s first token) | One-time CUDA-graph warm-up at each new batch size. Warm up, then measure |
