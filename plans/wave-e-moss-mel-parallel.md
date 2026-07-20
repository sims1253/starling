# Wave E — parallelize the MOSS host mel frontend (bit-identical)

## Why

Wave B's stage timing (`STARLING_MOSS_TIMING`) shows mel is now the largest
non-decode cost: **38 / 111 / 466 ms** (short/medium/long) — a serial,
single-threaded per-frame pocketfft loop in
`cpp/moss/mel.cpp::compute_log_mel`. Everything else non-decode is captured
(enc+adapt 7/18/45, prefill 28/41/96).

The loop nest is embarrassingly parallel with **zero numerics risk**:
- the frame loop (`for t in 0..fullT`): reflect-pad + window + r2c FFT + power
  per frame — frames are fully independent;
- the mel filterbank loop (`for m, for t`, inner sum over B): each (m,t)
  output is an independent dot product whose inner accumulation order must
  stay exactly `b = 0..B-1` (it does under per-(m,t) parallelism);
- the log10/normalize loops.

Parallelizing over frames / (m,t) cells does not change any per-element
floating-point operation or accumulation order, so the output is
**bit-identical** to the serial path. (The one global reduction — the `mx`
max — is order-insensitive for max, but keep a deterministic reduction anyway:
per-chunk maxes combined in chunk order.)

## Task

1. Thread the three loop nests in `cpp/moss/mel.cpp` with `std::thread`
   (chunked ranges, `std::thread::hardware_concurrency()` capped at, say, 16;
   respect an env override `STARLING_MEL_THREADS`, and fall back to the serial
   path for `STARLING_MEL_THREADS=1`). Per-thread scratch for `frame` and `z`
   (they are currently shared across iterations). No OpenMP (keep the build
   dependency-free), no new libraries.
2. While there: `cpp/parakeet/mel.cpp` has the same host-side structure for
   its double-precision stages (~10 ms on long). Apply the same treatment ONLY
   if it is the same trivially-parallel shape — parakeet's mel is
   byte-exactness-anchored, so if anything about its loop structure makes
   per-chunk parallelism non-order-preserving, leave parakeet alone and say so.

## Requirements

1. Bit-identical output: assert by running the mel goldens —
   `uv run python -m pytest tests/test_ggml_parity.py -x -q` must stay 9/9
   exit 0 (the moss mel gate is per-bin one-ULP, but the implementation must
   be exactly bit-identical anyway; verify with `STARLING_MEL_DUMP` before/
   after on the long fixture — byte-compare the dumps).
   Note: `test_ggml_moss_near_exact[long]` drives an EXTERNAL CrispASR server
   with a documented flake; if it fails, rerun it in isolation and with
   `GGML_MOSS_SERVER=0` — in-tree gates must be deterministic-green.
2. No git state-changing commands; no `third_party/ggml/` edits.
3. Report the new `STARLING_MOSS_TIMING` mel numbers (short/medium/long) and
   the e2e bench:

```bash
cmake --build build -j
uv run python -m pytest tests/test_ggml_parity.py -x -q
uv run python benchmarks/bench_all.py --models moss \
  --engines starling,starling-ggml --lengths short,medium,long --batches 1 \
  --reps 10 --warmup 2
```

Success bar: mel long <= 60 ms; e2e moss <= 210 / 560 / 1200 ms
(short/medium/long). If parakeet mel was also threaded: parakeet long <= 85 ms
with WER 0.00 and `scripts/verify_parakeet_decode.py` OK.
