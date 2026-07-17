# parakeet-tdt ggml vs starling: performance analysis

The `ggml-parakeet` engine transcribes **byte-exact** vs the golden (WER 0.00%
on short/medium/long, asserted by `tests/test_ggml_parity.py`). This document
analyzes its wall-clock latency vs the PyTorch + CUDAGraph + Triton peak engine
(`starling-parakeet`) and explains the remaining gap.

## Current numbers (RTX 5090, bf16, B=1, harness `bench_all.py`, 20 reps)

All three fixtures **byte-exact (WER 0.00%)**:

| length | ggml median | ggml min | starling median | starling min | gap (median) | WER |
|--------|-------------|----------|-----------------|--------------|--------------|-----|
| short  | 16 ms       | 16 ms    | 16 ms           | 14 ms        | **~1.00x**   | 0.00% |
| medium | 38 ms       | 35 ms    | 26 ms           | 25 ms        | ~1.46x       | 0.00% |
| long   | 108 ms      | 108 ms   | 60 ms           | 60 ms        | ~1.80x       | 0.00% |

**Short is at parity with starling** (16±0ms vs 16±2ms; 481x vs 480x RTFx) —
the headline 10%-of-peak target is met. Medium and long remain ~1.5-1.8x.

**Measurement note:** use `--reps 20`; low rep counts inflate the median via
warmup. Direct C-API timing (the ctypes probe below) confirms the floor. The
harness's `torch.cuda.synchronize()` doesn't cover ggml's stream but is
harmless at high reps.

## Where the time actually goes — the measured breakdown

This section **corrects an earlier (now-disproven) analysis** that blamed the
gap on "encoder replay overhead / per-node validation." That was wrong: commit
`c396a22` (parakeet.cpp) already added a uid fast-path that skips the
O(n_nodes) per-replay walks in ggml-cuda. The encoder is NOT the bottleneck.

Measured with CUDA events around `cudaGraphLaunch` (gated
`PARAKEET_GPU_TIMING=1`) + host wall-clock phase timing (gated
`PARAKEET_ENC_TIMING=1`):

| phase               | short   | medium  | long    | how measured                          |
|---------------------|---------|---------|---------|---------------------------------------|
| mel frontend        | ~1.5 ms | ~4 ms   | ~10 ms  | host wall around `GpuMel::compute`    |
| encoder graph (GPU) | ~2.8 ms | ~2.8 ms | ~2.8 ms | cudaEvent around `cudaGraphLaunch`    |
| encoder host overhead | ~0.4 ms | ~0.4 ms | ~0.4 ms | wall − GPU event                      |
| decode (TDT loop)   | ~13 ms  | ~30 ms  | ~95 ms  | total − mel − encoder                 |
| **total**           | ~16 ms  | ~38 ms  | ~108 ms | ctypes per-call wall                  |

Key facts:
- **The encoder graph is ~2.8ms of GPU compute** (3518 nodes, one captured CUDA
  graph). Starling's encoder compute is ~5ms total pipeline — so parakeet's
  encoder is *already faster* than starling's on raw compute. The encoder is
  NOT the gap. (The encoder graph does run at a healthy GPU share — host
  overhead is ~0.4ms, ~13% of the encoder phase.)
- **The decode (TDT loop) is the dominant cost** for medium/long. It's an
  inherently serial, data-dependent loop: each step's argmax determines the
  next token, so it cannot be captured as one CUDA graph. The K-step multistep
  (K=16 short/medium, K=64 long) collapses ~T/K syncs, but each replay still
  pays graph-launch + sync + readback. Long has ~8 replays (K=64) × the
  per-replay cost.

The matmuls (MUL_MAT, ~50-63% of encoder GPU time across 316 nodes) already
run on the **fp16 tensor-core cuBLAS path** (`CUBLAS_COMPUTE_16F`) at hardware
peak — verified empirically via the dispatch code + a gated debug print. There
is no wrong-kernel problem in the encoder.

## What was done (this round of optimization)

Six byte-exact commits in parakeet.cpp (`dev`), listed newest-first:

| commit | change | effect |
|--------|--------|--------|
| `23f0958` | decode: eliminate redundant double-sync per replay + K=64 for long | short -7%, med -10%, long -12% |
| `1f58419` | drop redundant `cont()` nodes in the flash-attn path | -48 nodes (encoder 2085→2037) |
| `7400510` | NORM+MUL+ADD fusion (vendored ggml) + conv-module depthwise-direct + conv pointwise f16 weights | long -16%; -240 nodes |
| `7c2323e` | mel frontend graph caching (persistent ReplayGraph) + teardown-crash fix (exit 134→0) | short mel -45%, med -25% |

Details:
- **Teardown crash fixed**: parakeet's native (in-process) path no longer
  SIGABRTs at process exit. `shutdown_backend()` is idempotent, clears the
  decode-graph cache before the backend, and is registered via `std::atexit`
  (runs before the CUDA driver's atexit, so CUDA-graph frees see a live
  driver). `tests/test_ggml_parity.py` now passes 6/6 with exit 0 (was: tests
  functionally passed but crashed at teardown).
- **Mel frontend**: was rebuilding a fresh ggml graph every call; now caches a
  persistent `ReplayGraph` keyed on T (mirrors the encoder). Preemphasis/
  framing/windowing kept on host in double precision (moving to GPU float
  would break byte-exactness).
- **NORM+MUL+ADD fusion**: ggml-cuda only fused `RMS_NORM+MUL`; parakeet uses
  plain LayerNorm (`GGML_OP_NORM`) ~120 times. Added a `layer_norm_f32` kernel
  (mirrors `rms_norm_f32`) + a `GGML_OP_NORM` branch in `ggml_cuda_can_fuse`.
  Collapses 3 nodes → 1 per LayerNorm: −240 nodes. Byte-exact (fp32 reduction
  copied verbatim from `norm_f32`).
- **Conv-module depthwise-direct**: replaced `im2col + batched mul_mat` (1024
  groups) with `ggml_conv_2d_dw_direct` (the kernel subsampling already uses),
  avoiding the `[9,T,1024]` im2col materialization. Long encoder −24%.
- **Conv pointwise f16 weights**: GGUF stored conv pointwise weights as f32
  (vs FFN/attention linears which are f16); cast to f16 at load to route onto
  the fp16 tensor-core cuBLAS path.
- **Decode double-sync**: `ggml_backend_graph_compute` syncs after the graph
  launch *before* readbacks are queued, blocking pipelining; then
  `ReplayGraph::readback_async_then_sync` syncs again. On the fast gallocr
  path (CUDA), now call `graph_compute_async` directly and let the single
  readback sync be the only one. Removes one `cudaStreamSynchronize` per
  replay (every encoder + decode replay).
- **K=64 for long**: halves the long replay count (15→8); K=64 was already
  known byte-exact on long.

## The progression arc (short: 158ms → 16ms)

The full history of how short got from 158ms to parity:

1. persistent server instead of per-process spawn (158→71)
2. encoder CUDA-graph capture via per-shape `ReplayGraph` (71→~50)
3. native in-process ctypes path, decode pred+joint+argmax fusion + async
   readback
4. encoder mask-skip + f16 im2col, flash-attn for the relpos attention
5. persisting PredictionNet/Joint on `Model` so decode graphs reach
   steady-state capture (50→32)
6. private gallocr per `ReplayGraph` (survive encoder rebuilds across
   utterances)
7. **K-step decode capture** (Starling's `decode_mega.py` blueprint — chain K
   decode steps' argmax/frame-advance/blank-skip on-device in one CUDA graph,
   sync once per K steps; 32→21)
8. multistep termination fix + T-aware K (16 short/medium, 64 long) (long
   284→130)
9. **(this round)** mel caching, NORM+MUL+ADD fusion, conv-module
   depthwise-direct + f16, redundant-sync elimination, K=64 for long (short
   21→16; long 130→108)

## What reaching parity on medium/long would require

The decode is now **GPU-compute-bound** (the ~6.5ms readback timer for a K=64
replay is dominated by the graph actually executing, not the sync). The
remaining medium/long gap is therefore genuine decode compute + the irreducible
serial nature of the TDT loop. Options, in increasing risk:

1. **Keep decode state on-device between replays** — currently each replay
   round-trips the LSTM h/c + frame + last_token through host (2+2L small H2D
   uploads + D2H readbacks). Keeping state device-side and reading back only
   the token/frame for the host-side termination check would cut per-replay
   overhead. Invasive (rewires the KStepGraph input/output model).
2. **Speculative multi-step decode with rollback** — batch a fixed lookahead
   of tokens before syncing, roll back on a blank. Large rewrite of the serial
  TDT control flow; risks byte-exactness.
3. **Algorithmic**: match starling's on-device argmax megakernel, which keeps
  the whole data-dependent loop on-device within one captured graph. A
  fundamentally different architecture than ggml's per-K-step replay.

The encoder is essentially done (faster than starling's on raw compute); the
decode is the remaining lever, and it's now at a GPU-compute floor rather than
a host-overhead floor.

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

**Byte-exact** on all fixtures. All three fixtures (short/medium/long) use
this full-context flash-attn path — the chunked-local path only engages at
encoder lengths Tp > 8192 (~54min audio), well beyond the fixtures.
