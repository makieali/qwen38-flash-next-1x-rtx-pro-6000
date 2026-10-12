# Build guide

Everything we ran, in order. Paths match our machine (`/srv/ai/...`); change them for yours. Versions are exact.
Where a command is reconstructed from notes rather than copied from a script, it says so.

## 0. Machine and OS

| | Ours | Notes |
|---|---|---|
| GPU | 1 × RTX PRO 6000 Blackwell Workstation Edition, 96 GB | SM120. Turn on persistence mode (`nvidia-smi -pm 1`) |
| CPU | AMD Threadripper 3990X (64 cores) | the CPU barely matters for decode |
| RAM | 256 GB DDR4-2400 | the PLE table alone is **102 GB** in pinned RAM; 192 GB is the practical minimum |
| PCIe | Gen 4 x16 (board limit; the card supports Gen 5) | PLE rows cross PCIe every token; Gen 5 helps a little |
| OS | Ubuntu 26.04 LTS, kernel 7.0, NVIDIA driver 610.57 (open) | glibc in 26.04 clashes with CUDA 13.0 headers → step 3 uses a CUDA 13.3 pip toolkit |
| Disk | ~550 GB free during the build | BF16 source 360 GB + NVFP4 186 GB; delete the source afterwards |

## 1. Download the source model

`trohrbaugh/Qwen3.8-Flash-Next-heretic`: BF16, 53 files, 360 GB (the Heretic uncensored variant of Qwen3.8
Flash-Next). We used ModelScope:

```sh
pip install modelscope
modelscope download --model trohrbaugh/Qwen3.8-Flash-Next-heretic --local_dir /srv/ai/models/flash-next-heretic-bf16
```

**Verify every file.** The downloader once reported "done" while the 102 GB PLE shard was missing.

```sh
# file list with sizes and SHA-256 from ModelScope
curl -s "https://www.modelscope.cn/api/v1/models/trohrbaugh/Qwen3.8-Flash-Next-heretic/repo/files?Recursive=true" > /tmp/ms-files.json
python model/verify_all.py                                   # every file: ok / missing / SHA MISMATCH
python model/fetch_shard.py model-00002-of-00033.safetensors # re-fetch one file: parallel ranges, resume, SHA check
```

## 2. Convert to NVFP4 (experts only)

![conversion](../images/nvfp4-conversion.svg)

Only the routed experts' `gate`, `up` and `down` projections become NVFP4 (FP4 E2M1 values, one FP8 scale per 16
values, one FP32 scale per tensor). Everything else stays BF16 in the checkpoint: attention (Gated DeltaNet + the
sparse-attention layers), shared experts, router, embeddings, `lm_head`, vision tower, MTP head and the PLE table.

Instead of calibrating ourselves we reuse **RadixArk's calibrated activation scales** (`input_scale`) from their
NVFP4 release, read with HTTP range requests (no need to download their weights):

```sh
# REF only needs RadixArk's model.safetensors.index.json + config.json
python model/fetch_scales.py    # → /srv/ai/quant/radixark_scales.json  (per-expert input_scale + weight_scale_2)
python model/convert.py         # all shards; or name shards as arguments. Uses the GPU for packing
python model/verify_layout.py   # quantises sample experts and compares with RadixArk's tensors
```

Paths are at the top of each script (`SRC`, `OUT`, `REF`). Gate and up of each expert share `weight_scale_2`
(the fused kernels require it). Check: relative error vs BF16 **0.00893** (RadixArk's own: 0.00892). Output: 186 GB.

## 3. SGLang Pennyroyal (production engine)

Pennyroyal is [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000), an SGLang build
tuned for the RTX PRO 6000. Outline of our install (versions exact; follow its `BUILD.md`/`RUN.md` for the precise
wheel sources, which change between releases):

```sh
git clone https://github.com/jpezzulli/sglang-rtxpro6000 /srv/ai/sglang/pennyroyal
cd /srv/ai/sglang/pennyroyal && git checkout pennyroyal-v2.5.3-setup1        # commit 468e63a
uv venv -p 3.12 .venv && . .venv/bin/activate                                 # Python 3.12.13
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install sglang-kernel==0.4.6.post1 flashinfer-python==0.6.17          # NOT FlashInfer 0.7.0 (512-expert bug)
SGLANG_BUILD_RUST_EXTS=none uv pip install -e "python[all]"
```

**CUDA 13.3 toolkit (pip).** Ubuntu 26.04's glibc declares `rsqrt` differently from the CUDA 13.0 headers, so
SGLang's JIT kernels fail to compile. Install a 13.3 toolkit into its own folder and point the launcher at it
(`CU=` in `serve-sglang.sh`); link PyTorch's cuBLAS headers/libs into it if the JIT build can't find them:

```sh
uv pip install --target /srv/ai/sglang/cuda133 "nvidia-cuda-nvcc-cu13==13.3.*" "nvidia-cuda-runtime-cu13==13.3.*"   # outline
```

### Model view (3 fixes, without touching the checkpoint)

```sh
V=/srv/ai/models/flash-next-heretic-nvfp4-sgl; mkdir -p $V
for f in /srv/ai/models/flash-next-heretic-nvfp4/*; do ln -s "$f" $V/; done
rm $V/config.json $V/tokenizer.json
cp model/sglang-view-config.json $V/config.json        # sparse-attention layer type renamed for SGLang (see below)
cp <RadixArk NVFP4 release>/tokenizer.json $V/          # byte-identical vocab: FR-Spec checks the tokenizer checksum
```

1. Our config calls the 12 sparse-attention layers `qwen_sparse_attention` (newer name); SGLang expects
   `full_attention`. Same layers, older label.
2. FR-Spec ships a token map for the 64k most common tokens and verifies the tokenizer file. We checked that our
   vocabulary is identical (248,044 tokens; identical ids on English, Urdu, Chinese, code and emoji samples) and used
   RadixArk's byte-identical `tokenizer.json`.
3. The PLE table is offloaded to pinned RAM by a launcher flag (`--ple-offload-embedding`), not by the view.

### Prebuild the MoE kernel once (~9–10 minutes, then cached)

```sh
serving/sglang/prebuild_moe.sh     # FlashInfer SM120 CUTLASS fused-MoE JIT module
```

## 4. Serve

```sh
serving/sglang/serve-sglang.sh true      # argument = online MXFP8 for the 16-bit layers
# or as a service: deploy/systemd/flashnext-sglang.service
```

Cold start takes about 4–5 minutes (loading 186 GB, pinning the PLE table, capturing CUDA graphs). The first
request at each new batch size is slow once (graph warm-up); exclude it from benchmarks.

Every flag is explained in [TUNING.md](TUNING.md). The important ones:

| Flag | Why |
|---|---|
| `--quantization modelopt_fp4` + `SGLANG_SM120_ONLINE_MXFP8=true` | NVFP4 experts; the remaining 16-bit layers are converted to MXFP8 at load (+20–25 % speed, ~2× less memory for those layers) |
| `--kv-cache-dtype fp8_e4m3`, `--mem-fraction-static 0.95` | ~825k tokens of KV room; 256k context |
| `--ple-offload-embedding` | 102 GB PLE table in pinned RAM |
| `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-token-map …/flash-next-64k.pt` | MTP guess-ahead with FR-Spec |
| `--max-running-requests 4 --max-mamba-cache-size 24` | 4 concurrent requests; the state cache needs 5 slots per request |
| `--linear-attn-*-backend flashinfer`, `--mamba-radix-cache-strategy extra_buffer` | fast kernels and prefix caching for the Gated DeltaNet layers |
| `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` | OpenAI-style reasoning and tool calls |
| `--preferred-sampling-params` 0.7 / 0.8 / 20 / presence 1.5 | Qwen's non-thinking defaults (the model files only carry thinking defaults) |
| `--default-chat-template-kwargs '{"enable_thinking": false}'` | fast replies by default; turn thinking on per request for tool use ([RESULTS.md](RESULTS.md#tool-calls)) |

## 5. Optional: the vLLM path (our fallback)

vLLM 0.31.0 + FlashInfer 0.7.1rc4 + b12x 1.5.0 + cutlass-dsl 4.7.1 (Python 3.14 venv). It reached 127 tok/s chat;
see [PROBLEMS.md](PROBLEMS.md) for the eleven fixes it needed. Two patches are in `serving/vllm/`:

```sh
python serving/vllm/patch_b12x.py <venv>/lib/python3.14/site-packages/vllm/<path to the b12x MoE layer>.py   # share workspaces
python serving/vllm/patch_flashinfer_force_backend.py <venv>/lib/python3.14/site-packages                       # B12X_FORCE_BACKEND
serving/vllm/serve-flashnext.sh          # sets B12X_FORCE_BACKEND=dynamic, MTP with a separate Triton MoE backend
```

Only one engine can hold the card: `deploy/systemd/flashnext-sglang.service` has `Conflicts=flashnext.service`.

## 6. Optional: fan and power control

`tools/gpuctl.py` (root daemon, NVML) sets fan mode (auto / quiet curve / manual) and the power cap
(150–600 W). Safety rule: at ≥ 80 °C or an unreadable temperature the fans always go back to automatic.
`deploy/systemd/gpuctl.service` runs it. Requires `pynvml`.

## 7. Benchmark

```sh
python bench/bench.py http://127.0.0.1:8080 qwen3.8-flash-next-uncensored   # 1 and 4 users, chat and code
python bench/concurrency.py                                                 # 1 / 4 / 8 parallel users (+ SGLang /metrics)
python bench/tooltest.py                                                    # two-step tool calls, 20 runs per setting
python bench/longtest.py                                                    # long context answer
python bench/eval_models.py flashnext out.json                              # tools, refusals, coding (executed), long answer, vision
```

Edit the endpoint/paths at the top of each script. **Use unique prompts**: the same prompt twice hits the prefix
cache and reports nonsense (we once saw "72,000 tok/s").
