# Fast engines (model-specialized Vulkan)

Starling ships two kinds of engines behind the same C API:

* the **ggml engines** — general, every model, every backend
  ([engine notes](ggml-engine.md));
* the **fast engines** in `cpp/fast/` — hand-written Vulkan compute kernels
  for exactly two model families, **Parakeet-TDT** and **MOSS-Transcribe**,
  built around their shapes the way [Splash](https://inco.ai/blog/splash/)
  builds an engine around one model.

A fast engine loads the same GGUF files. It is chosen per model load and
falls back to ggml whenever it cannot run, so enabling it never removes a
working path.

## Selecting the engine

| `STARLING_ENGINE` | Behavior |
| --- | --- |
| `auto` (default) | fast engine when the build has it, a Vulkan GPU is present and the model's tensors are supported; otherwise ggml |
| `fast` | fast engine or a load error (no silent fallback; for benchmarking) |
| `ggml` | always the ggml engine |

Build with `-DSTARLING_FAST=ON` (needs `glslc` and the Vulkan headers; the
Android build takes both from the NDK and enables the fast engine for
`arm64-v8a`). `libvulkan` is opened at run time, so a fast-enabled binary
still runs, on ggml, on machines without a Vulkan driver. CPU/software Vulkan
devices (SwiftShader, llvmpipe) are ignored.

| Variable | Effect |
| --- | --- |
| `STARLING_FAST_F16=0` | f32 products in the GEMMs (default: packed-f16 products with f32 accumulation per 32-deep slice where the device has `shaderFloat16`) |
| `STARLING_FAST_CACHE_DIR` | where the Vulkan pipeline cache and autotuning results persist (the Android app sets it to its model directory) |
| `STARLING_FAST_TUNE=1` | re-run the tile autotuner now (also on PowerVR, which otherwise ships measured defaults) |
| `STARLING_FAST_W4U=0/1` | W4 decode GEMVs through `unpackUnorm4x8` (default on for PowerVR only) |
| `STARLING_FAST_MICRO=...` | run one isolated kernel probe at load (`bits,N,K[,reps[,rows]]` GEMV, `norm,...`, `alu,...`, `idot,...`; see `Kernels::micro`) |
| `STARLING_FAST_TILE=BM,BN,TM,TN` / `STARLING_FAST_GEMV_ROWS=n` | override the tuned GEMM tile / GEMV rows |
| `STARLING_FAST_KSTEP=n` | MOSS decode tokens per submission (default 16) |
| `STARLING_FAST_DEVICE=n` | Vulkan physical device index |
| `STARLING_FAST_TIMING=1` / `STARLING_FAST_PROFILE=1` | per-stage wall times / per-kernel GPU times |
| `STARLING_FAST_VERBOSE=1` | device, weights and tuning choices at load |

## Design

**Weights are repacked at load.** GGUF stays the distribution format; every
matrix is rewritten once into one of three device layouts: *W4* (4-bit, a
16-bit scale and offset per 32 weights, nibbles stored apart from scales so a
tile is a few contiguous 16-byte loads — lossless for Q4_0, and Q4_K keeps
its sub-block scales), *W8* (int8 with a 16-bit scale per 16 weights —
lossless for Q8_0 and Q6_K), and *F16*. Other types are dequantized and
requantized to W8. Rows of paired projections are interleaved so one
workgroup owns both halves of a gated unit (Conformer GLU, SwiGLU). A
format like EXL3 would buy quality per bit at 2–3 bits, but its trellis
decode costs ALU work that mobile GPUs do not have to spare; at 4 bits the
cheap affine format is the right trade.

**Whole passes are recorded once and replayed.** Each forward pass is a
recorded Vulkan command buffer, cached per input shape; replaying it costs the
host nothing per operation. Recordings are split into one submission per
layer, which keeps each GPU job short (driver watchdogs, compositor
interleaving).

**Fusion.** GEMM epilogues apply bias, activation (SiLU / ReLU / exact
GELU), residual accumulation, GLU / SwiGLU pairs, and Parakeet's two
positional-bias query projections. A layer's closing LayerNorm is fused with
the next layer's first. Decode GEMVs fuse the preceding RMSNorm. MOSS decode
attention fuses q/k RMSNorm, RoPE, the KV-cache append and the attention
itself into one dispatch per head.

**No host round trips.** The Parakeet mel runs on the CPU (bit-identical to
the reference, 4–6× faster than the ggml CPU frontend). Everything from mel
to the joint encoder projection is one replay. MOSS writes the adapter
output straight into the LLM's input rows, and its greedy loop runs on the
GPU: argmax, token history, EOS/budget test and next-token embedding live in
a device state buffer, so the host waits on one fence per 16 tokens and the
KV cache never leaves GPU memory.

**Parakeet's transducer decoder stays on the CPU** — sequential, a few small
matrix-vector products per step — with int8 dot-product kernels (ARMv8.2
SDOT / x86 AVX2, chosen at run time). The prediction network runs only after
an emission, and its first-layer input projection is memoized per token.

**Autotuning.** Tile sizes that suit one GPU starve another, so with a cache
directory set the engine times candidate GEMM tiles and GEMV row counts on
synthetic weights at the real shapes the first time it meets a device, and
stores the choice.

## Kernels

| Shader | Role |
| --- | --- |
| `gemm.comp` | tiled GEMM, B = W4/W8/F16/F16ᵀ, f16-packed shared tiles, batched heads/windows, implicit-GEMM 3×3/s2 convs (MOSS), fused epilogues |
| `gemv.comp` | decode matrix-vector, x kept in registers per K slice, fused RMSNorm, residual / SwiGLU / argmax-partial epilogues |
| `attn_decode.comp` | fused qk-norm + RoPE + KV append + attention (one query head per workgroup) |
| `rope_kv.comp` | prefill q/k norm + RoPE + KV-cache write |
| `softmax.comp` | attention softmax with Parakeet's relative-position skew folded into the read index, causal and key-length masks |
| `norm.comp` | LayerNorm / RMSNorm, including the fused layer-boundary pair |
| `pk_conv.comp` | subsampling convolutions, Conformer depthwise conv + batch-norm + SiLU, MOSS stage-1 conv |
| `embed_rows.comp`, `dec_next.comp` | embedding lookup; on-device greedy step (argmax, EOS, next embedding) |

Shaders use only Vulkan 1.1 core features (16-bit values are packed in 32-bit
words); the packed-f16 variants additionally need `shaderFloat16`.

## Validation

* `fast_weights_test` (CPU-only, runs in CI): every repacked format against
  ggml's dequantizers, and the CPU GEMV against an f64 reference.
* `benchmarks/fast_engine/parity_parakeet.py`: ggml vs fast encoder output,
  token streams and transcripts on the same library build.
* `benchmarks/fast_engine/wer_engines.py` with `export_fleurs.py`: corpus WER
  and latency for several engine variants on the same clips.
* `benchmarks/fast_engine/android_bench.sh`: cross-builds `starling-bench`,
  pushes it to a phone with the models and compares engines there.
* `benchmarks/fast_engine/pixel_measure.sh`: the tuning campaign's phone
  metric (Parakeet medium + MOSS short) with a fixture-transcript gate;
  `phone_ab.sh` interleaves several binaries in one thermal window.
* `benchmarks/fast_engine/phone_gates.sh`: A/B of the current Android build
  against a sha256-pinned base binary (alternating rounds, screen off) with
  the G1 transcript gate and the optional Parakeet canary; prints `METRIC`
  lines. `phone_energy.sh` measures energy per MOSS-short transcription
  from the battery charge counter with a duration-scaled idle control (an
  estimate, not a rail measurement).

The fast engines are **not** byte-exact with ggml: activations enter the
matrix products as f16 (ggml's CPU path quantizes them to 8 bits), so
near-tie tokens occasionally differ. They are held to corpus quality, not to
the ggml engines' exact-text gates.

## Measured results

Development machine: AMD Ryzen 5 PRO 5650U with its Radeon (Vega 7, RADV)
iGPU. **These are not Pixel numbers** — measure the phone with
`benchmarks/fast_engine/android_bench.sh`, where the first load also runs the
autotuner. Wall time per transcription after one warm-up, same build and
model files.

| Model / audio | ggml CPU | ggml Vulkan | fast (Vulkan, f16) |
| --- | --- | --- | --- |
| Parakeet q4_k_m-shrink16, 7.4 s | 485 ms | — | 230 ms |
| Parakeet, 22.3 s | — | 640 ms | 522 ms |
| Parakeet, 74.4 s | — | 2 331 ms | 1 743 ms |
| MOSS q4e8, 7.4 s | 3 129 ms | — | 1 820 ms |
| MOSS, 22.3 s | 9 596 ms | — | 5 150 ms |

Encoder-only (Parakeet, 22.3 s): 345 ms fast vs 535 ms ggml Vulkan. MOSS
decode runs at ~33 ms/token, close to this iGPU's memory-bandwidth floor
(~1.1 GB of weights per token, 28 % of it the Q8 tied lm_head).

FLEURS en_us test, first 100 clips (`wer_engines.py`):

| Model | ggml CPU WER | fast f32 WER | fast f16 WER | time ggml / fast f16 |
| --- | --- | --- | --- | --- |
| Parakeet q4_k_m-shrink16 | 5.47 % | 5.33 % | 5.33 % | 72 s / 25 s |
| MOSS q4e8 | 7.92 % | 7.87 % | 7.92 % | 441 s / 186 s |

10 (Parakeet) and 16 (MOSS) of 100 transcripts differ from ggml, all in
near-tie words and punctuation, in both directions.

### Pixel 10 Pro (Tensor G5, PowerVR DXT-48-1536)

Tuned on-device (see `benchmarks/fast_engine/RESEARCH_LOG.md` for the
experiment log and `AUTORESEARCH.md` for the method). Vendor-keyed defaults
for PowerVR: GEMM tile 32,128,4,8 (a full 128-thread subgroup with BN = 128),
f32 GEMM products, GEMV rows 16 pinned (measured best-or-tied at every MOSS
decode shape — rows=8 cost 15–31 % on the small-N GEMVs whose per-workgroup
x re-read rivals the weights; RESEARCH_LOG P1-8) with RSPLIT row slots
padding workgroups to a whole subgroup; the CPU transducer decoder splits its
GEMV rows across a second thread (held spinning for one transcription,
parked in between); W4 decode GEMVs unpack nibbles with `unpackUnorm4x8`.
ggml runs with the app's 6 threads. Transcripts match ggml on the fixtures;
FLEURS en_us 100: Parakeet 5.33 % (ggml 5.47 %), MOSS q4e8 7.87 %
(ggml 7.92 %).

Two further PowerVR notes from the layout research (#317, same log):
`gemv_w4um` processes two tokens' GEMVs in one weight pass (1.76–2.11× the
per-token rate; the verify primitive for #311's speculative decoding — not
dispatched by the engine yet), and the GPU driver on the Pixel 10 Pro can
enter a degraded state under sustained load where fences time out (#325):
the engine marks itself wedged, leaves a 15-minute marker file
(`starling-fast-gpu-wedged`, in `STARLING_FAST_CACHE_DIR`) so fresh
processes refuse with a restart hint instead of joining a retry storm, and
checks `VK_EXT_memory_budget` before the big allocations where the driver
exposes it.

| Model / audio | ggml CPU | fast (start of tuning) | fast (tuned) |
| --- | --- | --- | --- |
| Parakeet, 22.3 s | 7.2 s (enc 1.30 s, dec 5.9 s) | 2.79 s (enc 2.56 s, dec 0.20 s) | **2.22 s** (enc 1.98 s, dec 0.14–0.20 s) |
| Parakeet, 74.4 s | — | 9.0 s | **7.6 s** |
| MOSS, 7.4 s | 7.3 s | 7.0 s (decode 100 ms/tok) | **5.6 s** (mel 0.13 s, enc+prefill 2.95 s, decode 76 ms/tok) |

Energy per transcription (batterystats power model, 40/20-run averages,
measured before the W4 unpack change, i.e. MOSS at 88.7 ms/token):
Parakeet medium fast ≈ 1.33 mWh vs ggml ≈ 5.0 mWh (3.8×), MOSS short fast
≈ 2.05 mWh vs ggml ≈ 5.2 mWh (2.5×).

The desktop-tuned GPU kernels now reach ~180 GFLOPS on the encoder GEMMs
(peak f32 ≈ 500 GFLOPS); the remaining gap is shared-memory traffic, not
barriers or tail waves. Decode GEMVs, in contrast, are instruction-issue
bound on this GPU, not bandwidth bound: W4 and W8 both stream ~33 G
weights/s, so saving ALU per weight (the `unpackUnorm4x8` path, −13 % MOSS
decode) pays and saving bytes alone does not. `VK_KHR_cooperative_matrix`
is advertised (f16 and int8 shapes) but the driver's shader compiler faults
on any coopmat instruction; an unfinished coopmat GEMM was removed (commit
b1182f2 has it). Integer dot products (`OpSDot`) do work
(`STARLING_FAST_MICRO=idot`, needs a glslc with `GL_EXT_integer_dot_product`;
the NDK's does not have it) — the obvious next lever for decode, at the cost
of int8 activations. The q4e4 model file
saves a further ~7 % decode time at +0.14 % WER (8.06 vs 7.92, inside the
0.2-pt gate) if maximum battery life matters more than the last word error.
