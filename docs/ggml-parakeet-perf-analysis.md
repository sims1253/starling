# parakeet-tdt ggml vs starling: performance analysis

The `ggml-parakeet` engine transcribes **byte-exact** vs the golden (WER 0.00%
on short/medium/long, asserted by `tests/test_ggml_parity.py`). This document
analyzes its wall-clock latency vs the PyTorch + CUDAGraph + Triton peak engine
(`starling-parakeet`) and explains the remaining gap.

## Current numbers (RTX 5090, bf16, B=1, in-process, model load excluded)

Harness (`bench_all.py`, native ctypes path, 8 reps) and in-process CLI
(`parakeet-cli bench proc_ms`, steady-state entries 4-7):

| length | audio  | ggml (harness, 8 reps) | ggml (CLI steady) | starling | gap   | WER   |
|--------|--------|------------------------|-------------------|----------|-------|-------|
| short  | 7.4s   | 32 ms                  | ~27 ms            | 14-15 ms | ~2.2x | 0.00% |
| medium | 22.3s  | 74 ms                  | ~64 ms            | 25-27 ms | ~2.9x | 0.00% |
| long   | 74.3s  | 258 ms                 | ~190 ms           | 57-62 ms | ~4x   | 0.00% |

The progression that got short from 158 ms to 32 ms (harness): persistent
server instead of per-process spawn (158→71), encoder CUDA-graph capture via
per-shape ReplayGraph (71→~50), native in-process ctypes path, decode
pred+joint+argmax fusion + async readback, encoder mask-skip + f16 im2col,
flash-attn for the relpos attention, **persisting PredictionNet/Joint on Model
so the decode graphs reach steady-state capture (50→32)**, and giving each
ReplayGraph a private gallocr so persistent decode graphs survive encoder
rebuilds across utterances.

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

## Conclusion — the gap is replay overhead, not a compute wall

**Correction of an earlier "documented wall" claim.** Initial analysis blamed
the encoder's ~67ms on "1218 kernels' device-side latency" and declared a
structural wall. That was wrong on two counts, corrected by finer measurement:

1. The ~67-84ms figures were **warmup-contaminated** (first-call graph capture
   + cudnn autotune). Per-entry `proc_ms` over an 8-entry same-shape manifest
   drops from 100ms (entry 0) to **~40-50ms steady-state** (entries 4-7). The
   bench harness warms up before timing, so its ~50ms is the steady-state
   number.
2. Starling's parakeet encoder has **NO hand-fused kernels** (see
   `docs/ggml-fused-ops-spec.md`): it runs the stock HF Conformer captured into
   one `torch.cuda.CUDAGraph().replay()` — pure scheduling, zero fusion, bf16
   throughout. So the encoder's actual GPU compute is ~5ms (Starling's whole
   14ms pipeline minus its decode). fp8/fused-ops are out of scope for parakeet.

**Current steady-state breakdown (short, in-process):**
- full pipeline: ~40-50ms (harness ~50ms; CLI entries 4-7)
- decode-only: ~13ms (graphs on; the per-step cudaGraphLaunch overhead is
  acceptable for the ~49-step short clip)
- **mel+encoder: ~27-37ms** = the remaining lever

The encoder graph IS captured (graphs-on vs -off: full 98 vs 139ms, so graphs
save ~27ms in the encoder). But ggml's replay path still does ~6x more host
work per replay than PyTorch's single-driver-call `graph.replay()` for the same
math — the 1218-node graph pays per-node validation/dispatch cost on each
replay that PyTorch's captured instance doesn't. **Closing that replay overhead
(the per-node walk in ggml-cuda's graph path, or forcing the direct gallocr
path) is the path to parity**, not a fused-kernel port. The competitor proves
this encoder's compute is ~5ms under proper graph capture.

The decode is near its floor for the step-by-step TDT loop (~13ms), though
batching multiple serial steps before syncing is a possible follow-on.
