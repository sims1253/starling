# Wave D — parakeet long-audio decode closure (Phase 3, items 1-2 + stretch)

## Why

After Wave C the in-tree engine is at 13/33/89 ms (short/medium/long) — short
beats the PyTorch peak (17 ms), but long trails it (89 vs 62 ms). Breakdown on
long (from `docs/ggml-parakeet-perf-analysis.md`, still structurally valid):
mel ~10 ms (host), encoder ~3 ms (captured graph), decode ~75 ms (K=64
multistep, ~8 replays).

The peak engine wins long because its whole TDT loop is one captured CUDA
graph — zero host interaction between steps. Our K-step multistep still pays,
per replay: H2D uploads of LSTM h/c + frame + last_token, the graph launch,
one sync, and D2H readback of the full K-step state.

## Task (in order; each is independently gated)

### D1. Device-resident decode state between replays

The K-step graph's chained state (LSTM h/c, frame counter, last_token,
committed-count) currently round-trips through host between replays. Keep it
in persistent device buffers (the Wave A `DeviceCache` pattern in
`cpp/moss/llm.cpp`: fixed backend-buffer tensors referenced as graph leaves,
written in-graph at the end of each replay, read in-graph at the start of the
next). Host reads back per replay ONLY what the termination check needs (the
emitted ids + final frame index — one small readback).

Gate: `scripts/verify_parakeet_decode.py` OK + parity 9/9 exit 0.

### D2. K/replay tuning after D1

With per-replay overhead reduced, re-sweep K on long (64/96/128) and re-check
the T-aware thresholds. Keep whatever the bench says; document the sweep.

### D3 (stretch, attempt only if D1+D2 leave >=10 ms on the table).
Device-terminated replay chaining: enqueue N replays back-to-back without
host sync (the state chaining from D1 makes replays composable), syncing only
every N replays for the termination check. The TDT termination condition
(frame >= T) is monotone, so overshooting replays past termination is safe if
the graph makes post-termination steps no-ops (mask emissions once frame >= T
— an in-graph predicate). Emitted-id readback then happens once at the end.
This approximates the peak's single-graph loop without a megakernel. If the
no-op-step masking cannot be made byte-exact (the id stream must stay
IDENTICAL, including blanks), stop and report instead of forcing it.

### D4. Mel overlap (only if trivially safe)

Long-audio mel is ~10 ms of host DFT work that runs strictly before the
encoder. If the mel module already processes independent frames, split the
mel compute and the H2D upload so upload overlaps compute (double-buffered
chunks). Do NOT change any mel numerics (host double-precision path is the
byte-exactness anchor). Skip if it turns invasive.

## Requirements

Same hard rules: parity 9/9 exit 0 per milestone; verify script OK; no
`third_party/ggml/` edits; no git commands; CPU serial fallback and
`STARLING_GGML_TDT_SERIAL` env override keep working; teardown stays exit-0.

## Verification

```bash
cmake --build build -j
uv run python scripts/verify_parakeet_decode.py
uv run python -m pytest tests/test_ggml_parity.py -x -q
uv run python benchmarks/bench_all.py --models parakeet \
  --engines starling-ggml --lengths short,medium,long --batches 1 \
  --reps 20 --warmup 3
```

Success bar: long <= 70 ms with short/medium not regressing (short <= 14,
medium <= 34). Stretch: long <= 62 ms (peak parity). WER 0.00 everywhere.

Report: files changed, per-milestone bench deltas, K-sweep table, parity.
