# FAQ

**Does this run on a 32 GB / 48 GB card?**
No. The NVFP4 experts alone are ~68 GB. You need 96 GB on one card (or two cards with tensor parallelism).

**Does it run on an RTX 5090 or B200?**
We only tested the RTX PRO 6000 (SM120). The RTX 5090 is also SM120 but has 32 GB. Datacenter Blackwell (SM100)
uses different kernels; the flags here are SM120-specific.

**Why not GGUF / llama.cpp?**
At Q5 the model doesn't fit on one 96 GB card with its 102 GB PLE table, so part of it runs on the CPU (~45 tok/s
for us).

**Is the KV cache in system RAM?**
No. KV cache and state cache are on the GPU. Only the 102 GB PLE lookup table is in RAM; a few rows are read per
token. [MEMORY.md](MEMORY.md).

**Why FP8 KV instead of BF16?**
Half the bytes: 825k tokens of room instead of ~412k. FP8 KV is Pennyroyal's default for this model; we haven't
run a side-by-side quality comparison with BF16 KV.

**Can I get 1M context?**
Not on one card: the KV room is 825k tokens, and ×4 position stretching is beyond what the model was validated
for. 512k is possible but untested: [CONTEXT-512K.md](CONTEXT-512K.md).

**Would a second card make it faster?**
For one user, not much: decode is bound by how fast one card reads the weights, and splitting across cards adds
communication. For many users, yes: others report 1,056 tok/s total at 32 users on two cards (not measured by us).

**Why is code faster than chat?**
Speculative decoding: the MTP head guesses 3 tokens ahead. Code is more predictable, so more guesses are accepted
per step.

**Why is one user's total (175) lower than their speed (192)?**
"Total" includes the time to first token; "per user" counts from the first token to the last.

**Why do 8 users get the same total as 4?**
Only 4 run at once; the state cache caps it. [TUNING.md](TUNING.md#concurrency).

**Is your NVFP4 as good as the reference?**
Same relative error vs BF16 on sampled experts (0.00893 vs 0.00892) and the same layout. We haven't run a full
benchmark suite on both; that's on the list.

**Where are the weights?**
Not published. The repo has the scripts to build them from the public BF16 weights.

**Why the uncensored variant?**
Our own use. All the speed work applies equally to the original model. See the responsible-use note in the
[README](../README.md#17-responsible-use).
