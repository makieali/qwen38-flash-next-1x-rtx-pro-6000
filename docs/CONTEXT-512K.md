# 512k context (not tested yet)

**Status: not tried on our build.** This page says how we would do it and what to measure. We'll add results when
we have them.

## How

The model is trained for 256k (262,144) positions. YaRN stretches the position encoding; ×2 gives 512k. This is the
setting Pennyroyal's reference config uses:

```json
"rope_scaling": {
  "rope_type": "yarn",
  "factor": 2.0,
  "original_max_position_embeddings": 262144
}
```

Add it to the model view's `config.json` (inside `text_config`/`rope_parameters` depending on the config layout;
compare with Pennyroyal's config), then in the launcher:

```sh
--context-length 524288
```

`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` is already set in the launcher.

## Will it fit?

KV room at `--mem-fraction-static 0.95` is 825,472 tokens (~12 KB per token in FP8), so one 512k request fits, with
~300k tokens left for other users. The Gated DeltaNet state doesn't grow with context.

## What to measure

1. **Recall**: hide a fact at 10 %, 50 % and 90 % depth of 128k, 256k and 512k-token prompts, ask for it
   (a "needle in a haystack" test; [`bench/longtest.py`](../bench/longtest.py) is a simple counting variant).
2. **Short-context quality**: YaRN can slightly change short-prompt behaviour; re-run the spot checks.
3. **Prefill time** for a 512k prompt (at ~14,000 tok/s that's at least ~40 s; attention cost grows with length,
   so expect more).
4. **Decode speed** at long context.

## 1M?

Not realistic: one 1M request needs more KV room than we have (825k tokens), and ×4 stretching goes beyond what the
model authors validated.
