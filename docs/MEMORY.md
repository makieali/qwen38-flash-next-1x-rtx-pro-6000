# Memory: what lives where

![Memory map](../images/memory-map.svg)

Measured on the production build (12 Oct 2026) with `nvidia-smi`, SGLang's `/metrics` and `/proc`.
Raw data: [`results/memory.json`](../results/memory.json).

## GPU (97,887 MiB)

| Item | Size | Notes |
|---|---:|---|
| Routed experts, NVFP4 | ~68 GB | 48 layers × 512 experts × gate/up/down; 4-bit values + FP8 scale per 16 + FP32 per tensor ≈ 4.5 bits/weight |
| Other weights, MXFP8 (converted at load) | ~8–16 GB | attention, shared experts, router, embeddings, `lm_head`, vision, MTP head |
| KV cache, FP8 E4M3 | ~10 GB at 0.95 | 825,472 tokens; 5.74 GB = 471,680 tokens at 0.90 |
| Gated DeltaNet state cache | 24 slots | 5 slots per running request → 4 requests |
| CUDA graphs, workspaces, activations | rest | |
| **Total in use** | 87–91 GB | at `--mem-fraction-static` 0.90–0.95 |

`--mem-fraction-static` is the share of the card SGLang takes for weights + KV + state; what's left after the
weights becomes KV room. Going from 0.90 to 0.95 adds ~4.9 GB of the card; the KV cache grew by ~4.3 GB of it.

## System RAM (256 GB)

| Item | Size | Notes |
|---|---:|---|
| PLE n-gram table | 102 GB | pinned (page-locked) so the GPU can read rows directly; `--ple-offload-embedding` |
| Engine process total | ~99 GB RSS | |
| OS, page cache, everything else | the rest | 192 GB RAM is the practical minimum |

## Why the KV cache is so small

Only **12 of the 48 layers** are full attention and need a KV cache that grows with the context. The other 36 are
Gated DeltaNet layers that keep a **fixed-size state** per request, however long the context.

From the measurement: 5.74 GB ÷ 471,680 tokens ≈ **12.2 KB per token** in FP8 for all 12 layers together. A
model of this size with 48 full-attention layers would need about 4× that, and BF16 another 2×.

| KV setting | Bytes per token | Tokens at 0.95 |
|---|---:|---:|
| FP8 (ours) | ~12 KB | 825,472 |
| BF16 (calculated, not run) | ~24 KB | ~412,000 |

The KV room is **shared** by all running requests: one 256k request plus several short ones fit at once.

## Why the PLE table can live in RAM

Per token the model reads a few rows of the table (indexed by recent tokens), a few KB, not the whole 102 GB.
Over PCIe Gen 4 x16 (~25 GB/s in practice) that is microseconds, small next to the ~5 ms each token takes on the
GPU. Keeping it in VRAM is impossible anyway (102 GB > 96 GB).

## What is *not* in RAM

The KV cache, the state cache and all the weights the GPU computes with are in VRAM. Offloading KV to RAM
(SGLang's HiCache) exists but we don't use it: our KV room is already larger than one request's maximum context.
