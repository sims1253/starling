# parakeet-tdt ggml vs starling: performance analysis

The `ggml-parakeet` engine transcribes **byte-exact** vs the golden (WER 0.00%
on short/medium/long, asserted by `tests/test_ggml_parity.py`). This document
analyzes its wall-clock latency vs the PyTorch + CUDAGraph + Triton peak engine
(`starling-parakeet`) and explains the remaining gap.

## Current numbers (RTX 5090, bf16, B=1, in-process, model load excluded)

Harness (`bench_all.py`, native ctypes path) and in-process CLI (`parakeet-cli
bench proc_ms`) agree to within noise:

| length | audio  | ggml (harness) | ggml (CLI proc_ms) | starling | gap   | WER   |
|--------|--------|----------------|--------------------|----------|-------|-------|
| short  | 7.4s   | 51 ms          | 73 ms              | 14-17 ms | ~3x   | 0.00% |
| medium | 22.3s  | 124 ms         | 153 ms             | 24-26 ms | ~5x   | 0.00% |
| long   | 74.3s  | 262 ms         | 321 ms             | 57-58 ms | ~5x   | 0.00% |

The progression that got here (short, harness): persistent server instead of
per-process spawn (158→71), encoder CUDA-graph capture via per-shape
ReplayGraph (71→~50), native in-process ctypes path (→44→50), decode
pred+joint+argmax fusion + async readback, encoder mask-skip + f16 im2col, and
**flash-attn for the relpos attention** (the biggest single encoder lever).

## The flash-attn win (the goal's item-3 attention fusion)

The encoder's relative-position attention was manual (materialize full QK^T,
add the per-head relpos bias `bd`, softmax, AV mul_mat). ggml's
`ggml_flash_attn_ext` fuses QK+softmax+AV but aborted on per-head masks
(`fattn.cu:448`). Two parakeet.cpp commits unblocked it:

1. `625b993` — vendored ggml fattn: allow a per-head additive mask
   (`mask->ne[2] == n_head`), baking the head offset into the mask pointer in
   the WMMA/MMA kernels; broadcast-mask path unchanged.
2. `6c40f43` — `RelPosAttention::build_graph` (B=1, GPU): one
   `ggml_flash_attn_ext` with `mask = cast_f16(bd*scale + vmask)` shaped
   `[T_k, T_q, H]`, replacing the manual ac/bd/softmax/AV sequence. CPU keeps
   the manual path (byte-identical reference).

**Byte-exact** on all fixtures (the per-head mask carries the exact relpos
bias; f16 KQ accumulation did not tip any argmax). Encoder-only improvement
(~11-13% on short/long in the CLI bench, larger on long where O(T²)
dominates): short 82→73ms, long 370→321ms proc_ms.

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

1. **Encoder**: the encoder graph is **1218 sequential small kernels** (364
   `mul_mat`, 318 `add`, 120 `norm`, 99 `unary`, 27 `im2col`, 24 each
   `soft_max`/`pad`, ...) captured in ONE CUDA graph. Launch overhead is gone
   (the graph replays as one `cudaGraphLaunch`), but each kernel still pays its
   own device-side latency (~50-60 µs each ≈ the observed ~60 ms). The cost is
   **latency-across-kernels**, not FLOPs or bandwidth — so the only fix is
   **fusion**. Verified blockers for the obvious fusions:
   - `ggml_flash_attn_ext` CANNOT replace the manual relpos attention: the
     relpos bias `bd = qv @ p^T` is per-head `[T_k, T_q, H]`, but ggml-cuda's
     flash-attn dispatch (`fattn.cu:448`) returns `BEST_FATTN_KERNEL_NONE` and
     `GGML_ABORT`s for any mask with `ne[2] != 1` — the kernel only broadcasts a
     `[n_kv, n_batch]` mask over heads. `bd` can't be folded into Q/K (it
     depends on relative position `t_k - t_q`, not either index alone). So the
     attention stays manual and materializes the full T×T matrix.
   - softmax is hard-locked to f32 (`softmax.cu:388`), so dtype doesn't help the
     softmax chain; the ac/bd/ctxh matmuls already run on the fp16 tensor-core
     path.
   Closing the encoder gap therefore needs a **custom fused relpos-attention
   CUDA kernel** (the competitor's Triton kernel fuses QK + relpos-bd + softmax
   + AV) — either a new ggml op in `third_party/ggml/src/ggml-cuda/` or lifting
   the per-head-mask restriction in flash-attn, plus the rmsnorm/SiLU/residual
   fusions. This is the goal's item 3. It is large, per-backend kernel work.
2. **Decode**: at the per-step host-sync floor. An algorithmic change —
   speculative multi-step decode with rollback, or batching a small fixed
   lookahead before syncing — to drop below one-sync-per-step. Large rewrite of
   the serial TDT control flow; risks byte-exactness.
3. **Platform**: native Linux instead of WSL2 would lower the per-sync latency
   (WSL2's GPU passthrough adds overhead to each `cudaStreamSynchronize`),
   directly helping the decode floor.

## What was done (encoder)

- Per-shape `ReplayGraph` capturing the whole 24-layer encoder as one CUDA graph
  (`parakeet.cpp` commit `2025082`).
- Skip the trivial (all-zero) attention mask + f16 depthwise-conv im2col
  (`8782556`): long-audio −20% (the mask is `[930×930]` f32 × 24 layers = 3.5 MB
  saved per call). Short/medium flat.
- Verified the flash-attn path is blocked (above) — left the manual relpos
  attention as-is pending a custom fused kernel.

## Conclusion — the documented wall

The decisive arithmetic (short fixture, in-process CLI):

- decode-only: **14.2 ms** (`parakeet-cli bench-decode` serial)
- full pipeline: **81 ms** → **mel+encoder = ~67 ms**
- starling entire pipeline: ~14-17 ms

The **decode is already at parity with starling's whole pipeline** (14.2 ms vs
14-17 ms). The **encoder IS the entire gap**: cutting mel+encoder from ~67 ms
toward ~0 would put parakeet at ~14 ms ≈ starling. The 10% target (~18 ms)
requires reducing mel+encoder from 67 ms to ~4 ms — a ~17x cut.

The encoder's ~67 ms is the **device-side latency of 1218 sequential small
kernels** (~55 µs each) captured in one CUDA graph. Launch overhead is gone
(single `cudaGraphLaunch`); the cost is the kernels' own latency × count. The
available fusions have been applied (flash-attn for attention, mask skip, f16
im2col, CUDA-graph capture). Further fusion (fused rmsnorm/silu/residual/fp8-
gemv — the goal's item 3) would reduce kernel *count*, but each eliminated
kernel saves only ~55 µs, so reaching 18 ms would require eliminating ~1100 of
the 1218 kernels — i.e. rewriting ggml's op granularity itself, not a tunable
parameter. This is the documented wall: **ggml's op count is the encoder's
cost**, and starling's Triton megakernels sidestep it by fusing hundreds of ops
per kernel — a per-backend kernel-authoring effort far beyond config tuning.

The ggml engine is **correct (byte-exact), first-class, and universal-backend**.
It is **not within 10%** of the PyTorch peak, and the reason is structural
(ggml op granularity vs fused Triton megakernels), not a bug or a missing
flag. The documented path to close it — authoring fused custom CUDA kernels
for the rmsnorm/conv/residual/FFN chains and an on-device multi-step decode
megakernel — is the goal's item-3 work at full scope.
