# Results

Every number below is in a JSON file in [`results/`](../results/), with hardware, date and method. "Reported by
others" numbers are kept separate and were not measured by us.

**Hardware for all runs:** 1× RTX PRO 6000 Blackwell Workstation Edition 96 GB (SM120, PCIe Gen4 x16), Threadripper
3990X, 256 GB DDR4-2400, Ubuntu 26.04 LTS, kernel 7.0, driver 610.57, power limit 600 W.

## Speed per engine step

[`engine-steps.json`](../results/engine-steps.json): output tokens/s, streaming, unique prompts, warm engine.

| Step | Date | Chat, 1 user | Code, 1 user | 4 users total | Prefill | KV room |
|---|---|---:|---:|---:|---:|---:|
| llama.cpp, Windows, GGUF Q5_K_M (20 expert layers in RAM) | 2026-10-06 | ~45 | ~45 | ~45 | ~380 | n/a |
| vLLM 0.31 + our NVFP4 (b12x dynamic, MTP k=3) | 2026-10-07 | 127 | 195 | 353 | ~14,000 | 296,000 |
| SGLang Pennyroyal v2.5.3 + our NVFP4 | 2026-10-08 | 150 | 215–240 | 395–432 | ~14,450 | 565,120 |
| SGLang + online MXFP8 (production) | 2026-10-08 | 179–188 | 281–287 | 481 | 13,500–14,500 | 825,472 |

The production KV room is at `--mem-fraction-static 0.95` (12 Oct); at 0.90 it was 471,680.

## Concurrency

[`concurrency-2026-10-12.json`](../results/concurrency-2026-10-12.json), [`bench/concurrency.py`](../bench/concurrency.py):
500-word story prompts with a unique UUID, `max_tokens` 700, streaming.

| Users | Total tok/s | Per user | First token | Running / queued (peak) |
|---:|---:|---:|---:|---|
| 1 | 175 | 192 | 0.37 s | 1 / 0 |
| 4 (first after restart, excluded) | 210 | 129–137 | 7.87 s | 4 / 0 |
| 4 | 461–476 | 124–142 | 0.48–0.49 s | 4 / 0 |
| 8 | 449–459 | 111–134 | 0.48–7.08 s | 4 / 4 |

The 1-user "total" (175) is lower than "per user" (192) because total includes the time to first token.

## MTP depth (vLLM)

[`mtp-depth-vllm.json`](../results/mtp-depth-vllm.json)

| k | Chat | Code | 4 users | Acceptance length |
|---:|---:|---:|---:|---:|
| 2 | 128 | 170 | 350 | 2.37 |
| 3 (kept) | 126.5 | 201 | 355 | 2.15 |
| 4 | 118 | 189 | 321 | 3.10 |

SGLang production (NEXTN 3 steps, 4 draft tokens, FR-Spec 64k): acceptance length 2.06–2.13 (12 Oct).

## Memory

[`memory.json`](../results/memory.json): see [MEMORY.md](MEMORY.md).

## Conversion

[`conversion.json`](../results/conversion.json): relative error vs BF16 0.00893 (ours) vs 0.00892 (RadixArk);
186 GB.

## Quality

[`quality-2026-10-08.json`](../results/quality-2026-10-08.json), on SGLang + MXFP8. Spot checks, not a benchmark.

| Check | Result |
|---|---|
| Module + its own tests written by the model, tests executed | 10/10 and 9/9 passed |
| Long answer | 2,934 tokens in 13.5 s, no repetition |
| Vision | correct description |
| Borderline prompts | 0 of 5 refused |

## Tool calls

Two-step test ([`bench/tooltest.py`](../bench/tooltest.py)): the model calls `get_exchange_rate`, receives a
result, and must use it in the final answer.

| Sampling | Thinking | Answer used the result |
|---|---|---|
| presence penalty 1.5 | off | 10/20 (vLLM), 15/40 (SGLang) |
| presence penalty 0–1.0 | off | ~26–27/40 |
| any | **on** | **40/40** (both engines) |

Practical rule: thinking off for plain chat (~0.3 s to first token), on for requests that carry `tools`
(~0.5 s extra per tool round).

## Reported by others (not measured by us)

| Source | Setup | Numbers |
|---|---|---|
| SGLang Pennyroyal (jpezzulli/sglang-rtxpro6000) | 1× RTX PRO 6000 | 171–207 tok/s, 1 user |
| SirTificate/qwen38-flash-next-2x-rtx6000-sm120 | 2× RTX PRO 6000, vLLM | 161 tok/s, 1 user; 1,056 tok/s total, 32 users |

## Method

Unique prompts (prefix caching otherwise inflates results: we once saw "72,000 tok/s"), warm engine, first run
after a restart excluded, streaming, ranges = spread over repeated runs. Scripts in [`bench/`](../bench/).
