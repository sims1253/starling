# Wave F — parakeet in-tree encoder cost on long audio (investigate, then fix)

## Why

After Wave D, parakeet in-tree is 13/27/82 ms (short/medium/long) vs the
PyTorch peak 17/28/62. Wave D's instrumentation showed the long-audio encoder
phase is **~34 ms** — the dominant reducible cost (decode is ~41 ms and near
its serial floor; mel ~7 ms).

The external parakeet.cpp (`~/Documents/parakeet.cpp`, branch `dev`) claimed
~2.8 ms encoder GPU time (`docs/ggml-parakeet-perf-analysis.md`), and
back-of-envelope FLOPs (24 conformer layers, d=1024, T~929 after ÷8
subsampling on 74 s audio ≈ 0.5 TFLOP) say ~2-3 ms at good tensor-core
efficiency. 34 ms means the in-tree encoder is either not on the intended
kernels/dtypes, not actually capture-replaying, or carrying node bloat —
OR the external 2.8 ms claim was mismeasured. Find out which. Do not guess.

## Task

### F1. Measure (before touching anything)

Instrument the in-tree encoder phase (env-gated, keep it in the tree —
follow Wave D's timing-hook style): host wall around the encoder call, and
GPU time via cudaEvent or repeated-replay amortization. Break down: mel->H2D,
graph replay (GPU), readback. Also count graph nodes and verify: does the
long-shape encoder ReplayGraph actually reach CUDA-graph capture steady state
(patch-0008 gate: persistent + stable uid), or is it re-warming/rebuilding
per call? Compare the in-tree encoder graph vs the external one on the same
long fixture: node count, which ops dominate (the external repo has
`PARAKEET_ENC_TIMING` / debug prints to crib from), matmul dtypes (bf16/f16
tensor-core path vs f32 fallback), whether the NORM+MUL+ADD fusion and
depthwise-direct conv actually engage (both patches are applied in-tree —
check the ops are USED by our graph builder, e.g. we emit the fusable
NORM->MUL->ADD sequence and `ggml_conv_2d_dw_direct` rather than im2col).

Write the breakdown into the report BEFORE fixing. If the external 2.8 ms
claim turns out to be wrong and the in-tree encoder is already at the
hardware floor, STOP after F1 and report that conclusion with evidence.

### F2. Fix what F1 found

Likely suspects, in order of prior probability (address only what F1
implicates):
- encoder weights or activations routed through f32 matmuls (missing the
  f16/bf16 tensor-core cuBLAS path — the external repo casts conv pointwise
  weights to f16 at load);
- fusable sequences not emitted in the fusion-friendly op order;
- conv module using im2col instead of depthwise-direct;
- per-call rebuild instead of steady-state replay (cache keyed wrong for the
  long shape, or capture silently disabled);
- avoidable f32 casts / cont nodes bloating the graph (the external repo
  dropped 48 redundant cont nodes in its flash-attn path).

## Requirements

Same hard rules: byte-exactness gates after every change —
`scripts/verify_parakeet_decode.py` OK AND parity 9/9 exit 0 (CrispASR moss
long flake: retry in isolation / `GGML_MOSS_SERVER=0`); no
`third_party/ggml/` edits; no git state-changing commands; CPU path and env
fallbacks intact; teardown exit-0.

## Verification

```bash
cmake --build build -j
uv run python scripts/verify_parakeet_decode.py
uv run python -m pytest tests/test_ggml_parity.py -x -q
uv run python benchmarks/bench_all.py --models parakeet \
  --engines starling-ggml --lengths short,medium,long --batches 1 \
  --reps 20 --warmup 3
```

Success bar: long <= 65 ms (stretch <= 62 = peak parity) with short <= 13 and
medium <= 28 not regressing, WER 0.00 everywhere. If F1 proves the encoder is
already at its floor, the honest measured report IS the deliverable.

Report: F1 breakdown table (in-tree vs external), what was wrong, per-fix
deltas, final bench, parity.
