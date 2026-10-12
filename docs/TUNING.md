# Tuning: every flag in the production launcher

The launcher: [`serving/sglang/serve-sglang.sh`](../serving/sglang/serve-sglang.sh). It is adapted from
Pennyroyal's `configs/pennyroyal/serve-flash-next-frspec.sh`: without NIXL/HiCache storage, native 256k context
(no YaRN), PLE table in RAM.

"Measured" means we have numbers in [`results/`](../results/); "upstream default" means we kept Pennyroyal's value
without testing alternatives.

## Weights and precision

| Flag / env | Value | Effect |
|---|---|---|
| `--quantization` | `modelopt_fp4` | Loads our NVFP4 experts (ModelOpt layout) |
| `SGLANG_SM120_ONLINE_MXFP8` | `true` (script argument) | 16-bit layers → MXFP8 at load. **Measured: 150 → 179–188 chat, 215–240 → 281–287 code** |
| `--dtype` | `bfloat16` | Compute type for what isn't quantised |
| `--speculative-draft-model-quantization` | `unquant` | MTP head stays 16-bit (its experts are BF16) |
| `SGLANG_MAMBA_CONV_DTYPE`, `--mamba-ssm-dtype` | `bfloat16` | Gated DeltaNet state precision (upstream default) |

## Memory and context

| Flag | Value | Effect |
|---|---|---|
| `--kv-cache-dtype` | `fp8_e4m3` | Half the KV bytes of BF16 |
| `--mem-fraction-static` | `0.95` | **Measured: 825,472 KV tokens vs 471,680 at 0.90**, same speed |
| `--context-length` | `262144` | Native 256k. For 512k see [CONTEXT-512K.md](CONTEXT-512K.md) |
| `--page-size` | `64` | KV page size (upstream default) |
| `--ple-offload-embedding` | | 102 GB PLE table in pinned RAM. Required on one card |
| `--chunked-prefill-size` | `4096` | Long prompts are processed in 4k chunks, so other users' decoding isn't frozen during a big prefill |

## Concurrency

| Flag | Value | Effect |
|---|---|---|
| `--max-running-requests` | `4` | Requests decoded together; more wait in a queue |
| `--max-mamba-cache-size` | `24` | Gated DeltaNet state slots. **Each running request needs 5** (with MTP and prefix caching), so 24 → 4 requests |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Prefix caching for the Gated DeltaNet layers (repeated system prompts/conversations skip prefill) |
| `--mamba-track-interval` | `64` | How often state is checkpointed for that cache (upstream default) |
| `--sleep-on-idle` | | The scheduler sleeps when idle instead of spinning a CPU core |

**Measured** ([`results/concurrency-2026-10-12.json`](../results/concurrency-2026-10-12.json)): with
`--max-running-requests 8` and `--max-mamba-cache-size 24`, SGLang logged *"max_running_requests is capped to 4 by
the mamba state cache"*. 8 users gave 449–459 tok/s total (same as 4) with 4 of them queued and waiting up to 7 s for
their first token.

**Proposed, not measured:** `--max-mamba-cache-size 40 --max-running-requests 8`. The state slots take VRAM from
the KV room; the exact cost and the 8-user speed still need measuring. Expect total throughput to rise only
somewhat (one user already uses much of the memory bandwidth), but no queueing for users 5–8.

## Speculative decoding

| Flag | Value | Effect |
|---|---|---|
| `--speculative-algorithm` | `NEXTN` | Use the model's MTP head |
| `--speculative-num-steps` | `3` | Guess depth. **Measured on vLLM: 3 beats 2 and 4** ([`mtp-depth-vllm.json`](../results/mtp-depth-vllm.json)) |
| `--speculative-num-draft-tokens` | `4` | 3 guesses + 1 |
| `--speculative-eagle-topk` | `1` | One guess per position (a chain, not a tree) |
| `--speculative-token-map` | `flash-next-64k.pt` | FR-Spec: the draft scores only the 64k most frequent tokens |
| `--gdn-mtp-cache-mode` | `none` | (upstream default) |

## Kernels

| Flag / env | Value | Effect |
|---|---|---|
| `--linear-attn-decode-backend`, `--linear-attn-prefill-backend` | `flashinfer` | FlashInfer's Gated DeltaNet kernels |
| `CUDA_HOME`, `CUDACXX` | CUDA 13.3 pip toolkit | JIT kernels compile on Ubuntu 26.04 |
| `CC`, `CXX` | gcc-15 | Host compiler for the JIT |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | Build only for SM120 |
| `MAX_JOBS`, `FLASHINFER_NINJA_JOBS` | `8` | Parallel JIT builds |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Less fragmentation |

## Chat behaviour

| Flag | Value | Effect |
|---|---|---|
| `--reasoning-parser` | `qwen3` | Thinking text goes to `reasoning_content`, not the answer |
| `--tool-call-parser` | `qwen3_coder` | OpenAI-style `tool_calls` |
| `--chat-template` | Pennyroyal's `froggeric-v22.5.jinja` | Fixed chat template for tools and thinking |
| `--default-chat-template-kwargs` | `{"enable_thinking": false}` | Fast replies by default |
| `--preferred-sampling-params` | temperature 0.7, top_p 0.8, top_k 20, presence 1.5 | Qwen's recommended non-thinking defaults; the model files only carry the thinking ones |

**Measured tool-call note:** with thinking off, presence penalty 1.5 made the model ignore tool results in about
half the runs. Turning thinking on for requests that carry `tools` gave 40/40. Clients can send
`"chat_template_kwargs": {"enable_thinking": true}`. See [RESULTS.md](RESULTS.md#tool-calls).

## Power

The card's default limit is 600 W. [`tools/gpuctl.py`](../tools/gpuctl.py) can cap it (150–600 W) and run a
quieter fan curve, with a hard rule: at ≥ 80 °C or an unreadable temperature, fans go back to automatic. A rough
observation suggested 450 W keeps about 90 % of the speed; we haven't measured the full curve yet, so treat that as
a hint, not a result.
