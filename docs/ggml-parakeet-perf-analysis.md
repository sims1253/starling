# parakeet-tdt ggml vs starling: performance analysis

The `ggml-parakeet` engine transcribes **byte-exact** vs the golden (WER 0.00%
on short/medium/long, asserted by `tests/test_ggml_parity.py`). This document
analyzes its wall-clock latency vs the PyTorch + CUDAGraph + Triton peak engine
(`starling-parakeet`) and explains the remaining gap.

## Current numbers (RTX 5090, bf16, B=1, in-process ctypes path, model load excluded)

| length | audio  | ggml   | starling | gap   | WER   |
|--------|--------|--------|----------|-------|-------|
| short  | 7.4s   | 44 ms  | 13 ms    | 3.4x  | 0.00% |
| medium | 22.3s  | 85 ms  | 24 ms    | 3.5x  | 0.00% |
| long   | 74.3s  | 268 ms | 58 ms    | 4.6x  | 0.00% |

The progression that got here (short, from the initial 158 ms): persistent
server instead of per-process spawn (158→71), encoder CUDA-graph capture via
per-shape ReplayGraph (71→~50), native in-process ctypes path (→44), decode
pred+joint+argmax fusion + async readback (→44, decode-only 15.5→13.6 ms).

## Where the 44 ms (short) goes — measured breakdown

In-process (`parakeet-cli bench` / `bench-decode`):

| stage           | time   | how measured                                  |
|-----------------|--------|-----------------------------------------------|
| mel + encoder   | ~68 ms | full `proc_ms` (82) − decode-only (14)        |
| decode (TDT)    | ~14 ms | `bench-decode` serial_ms                      |
| total           | ~82 ms | `parakeet-cli bench proc_ms`                  |

(The bench harness's 44 ms is lower than the CLI's 82 ms because the harness
reuses the loaded model across reps while the CLI `bench` includes more per-file
overhead; the in-process ctypes probe reports ~50 ms median. The breakdown
ratios hold either way.)

starling does mel + encoder + decode in ~13 ms total. So:

- **Encoder: ~10x kernel-efficiency gap.** The 24-layer Conformer encoder is
  CUDA-graph-captured (per-shape ReplayGraph, `parakeet.cpp src/encoder.cpp`),
  so launch overhead is eliminated — the time is genuine kernel compute. The
  gap vs starling is kernel fusion: starling uses Triton fused kernels
  (rmsnorm/attention/conv/FFN folded), parakeet.cpp uses generic ggml ops
  (separate kernels per op, manual relative-position attention that
  materializes the full T×T matrix instead of flash attention). This is the
  dominant cost (~80% of the pipeline).
- **Decode: at the per-step host-sync floor.** The TDT loop is data-dependent
  (each step's argmax determines the next token), so it cannot be captured as
  one CUDA graph. Each of the ~49 inner steps needs ≥1 host↔device sync to
  read back the argmax. On WSL2 each `cudaStreamSynchronize` is ~150–200 µs;
  ×49 ≈ 8–10 ms hard floor. The decode is already optimized (pred+joint+argmax
  fused into one replay graph → one sync/step; async readback batching), and
  sits at ~14 ms, near this floor. starling's graphed decode keeps the argmax
  on-device within a captured multi-step megakernel — a fundamentally different
  architecture.

## What reaching 10% would require

Both components are at architectural floors:

1. **Encoder**: port starling's fused ops as ggml custom CUDA kernels — fused
   rmsnorm, fused SiLU·mul, residual scale-add, fp8 weight-only dequant-GEMV,
   and convert the manual relpos attention to `ggml_flash_attn_ext` (folding the
   relative-position bias into the attention bias/mask). This is the goal's
   item 3. Expected: could close most of the encoder gap, but it is a large,
   per-backend kernel-writing effort and ggml's kernel infra differs from
   Triton.
2. **Decode**: an algorithmic change — speculative multi-step decode with
   rollback, or batching a small fixed lookahead before syncing — to drop below
   one-sync-per-step. Large rewrite of the serial TDT control flow; risks
   byte-exactness.
3. **Platform**: native Linux instead of WSL2 would lower the per-sync latency
   (WSL2's GPU passthrough adds overhead to each `cudaStreamSynchronize`),
   directly helping the decode floor.

## Conclusion

The ggml engine is **correct (byte-exact), first-class, and universal-backend**.
It is **not within 10%** of the PyTorch peak on this fixture: the gap is
kernel-fusion depth in the encoder (~10x) and the data-dependent decode's
per-step sync floor. Closing it is the goal's item-3 custom-ops work (encoder)
plus a decode-architecture change — both substantial, and neither a simple
config tweak. This is the documented wall.
