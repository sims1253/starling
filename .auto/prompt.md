# Autoresearch: parakeet-tdt on 5650U — fastest + most memory-efficient native inference

## Objective
Make `parakeet-tdt-0.6b-v3` transcription run as fast and memory-efficient as possible
on THIS machine (AMD Ryzen 5 5650U, 6C/12T Zen3 AVX2+FMA+F16C, Renoir Vega iGPU via
RADV Vulkan, 14GB RAM). Primary target: **Vulkan/iGPU**. Secondary target: **CPU**.
Paths allowed: (a) quantized GGUF choice / custom quantization recipes via
`starling-quantize` + imatrix, (b) engine graph/kernel improvements under `cpp/`,
(c) custom patches to the ggml submodule (`third_party/ggml` — the user explicitly
blessed this), (d) build flags. Do not assume prebuilt anything is optimal.

## Metrics
- **Primary**: `rtf_med` (unitless RTF = wall_s / audio_s on the 22.3s medium fixture,
  Vulkan device, median of 5) — lower is better.
- **Secondary**: `rtf_cpu_med` (same on CPU build), `short_ms`, `long_ms`,
  `peak_rss_mb` (VmHWM of the bench process), `sha1_12` (transcript hash — drift
  detector), `wer_real` (quality gate, mean per-clip WER% on 32-utterance real
  LibriSpeech corpus, must stay ≤ baseline + 1.0).

## How to Run
`./.auto/measure.sh` — emits `METRIC name=value` lines (Vulkan first, then CPU).
`./.auto/checks.sh` — WER quality gate (runs bench_wer.py on Vulkan; exit 1 on fail).
Env knobs honored by measure.sh if pre-exported: `PK_MODEL` (default
`models/parakeet-tdt-0.6b-v3-q8_0.gguf`), `PK_VK_LIB` (default
`build-ar-vk/libstarling_ggml.so`), `PK_CPU_LIB` (default
`build-ar-cpu/libstarling_ggml.so`), `PK_REPS`.

Build commands (incremental, fast):
```
cmake --build build-ar-vk  -j12 --target starling_ggml   # Vulkan shared lib
cmake --build build-ar-cpu -j12 --target starling_ggml   # CPU shared lib
cmake -B build-ar-vk  -DSTARLING_SERVE=ON -DSTARLING_GGML_VULKAN=ON \
  -DSTARLING_GGML_SHARED=ON -DCMAKE_BUILD_TYPE=Release \
  -DVulkan_INCLUDE_DIR=/home/m0hawk/local-prefix/include \
  -DSPIRV-Headers_DIR=/home/m0hawk/local-prefix/share/cmake/SPIRV-Headers
cmake -B build-ar-cpu -DSTARLING_SERVE=ON -DSTARLING_GGML_SHARED=ON \
  -DCMAKE_BUILD_TYPE=Release
```
NOTE: engine C++ changes need BOTH libs rebuilt before measuring (CPU is the
secondary metric). ggml-submodule changes need both too. Rebuild inside the
experiment command, not in measure.sh.

## Files in Scope
- `cpp/**` — the starling ggml engine (parakeet graphs, runtime, backend)
- `third_party/ggml/**` + `third_party/ggml-patches/*.patch` — ggml itself; new
  patches go in the series (numbered, must apply cleanly via
  `scripts/apply_ggml_patches.sh`; the script snapshots+validates, so `git add`
  new patch files under third_party/ggml-patches/ and re-run cmake configure)
- `models/*.gguf` + `benchmarks/recipes/*` + quantization via `build-*/starling-quantize`
- `.auto/*` — session files
- CMakeLists.txt (build flags)

## Off Limits
- `benchmarks/wer.py` reference transcripts, fixture generation logic
  (`tests/fixtures/make_fixtures.py`), `tests/fixtures/real_corpus/reference.json`
- The WER gate itself (`.auto/quality.json` baseline may only be re-captured
  deliberately with justification, never to make a failing run pass)
- Nothing may weaken/deceive the benchmark: no caching transcripts, no
  short-circuiting decode, no per-fixture special-casing.

## Constraints
- Quality gate: `wer_real` ≤ 3.94 + 1.0 on the real corpus (q8_0 baseline).
  For pure kernel/numerics changes to the SAME model, prefer also checking
  `sha1_12` / `wer_fix` for drift; small WER movement within gate is acceptable.
- ggml patch series must apply cleanly (`cmake` configure validates it).
- Do not break `uv run python .auto/bench_wer.py` or the load path.

## Machine facts (measured 2026-09-21)
- Vulkan device: AMD Radeon Graphics (RADV RENOIR), uma=1, fp16=1, bf16=0, int8 dot=0,
  no matrix cores, warp 64, 64KB shared. ~1.7 TFLOPS fp32 theoretical.
- Stage split (q8_0, medium 22.3s): encoder 657ms (85%), TDT decode 106ms (14%),
  mel 7ms. Encoder graph: 2037 nodes, ~1440 dispatched (MUL_MAT=293, ADD=295,
  CONT=246, MUL=168, NORM=120, UNARY=99, CPY=72, SCALE=72, CONV_2D_DW=26,
  FLASH_ATTN_EXT=24, PAD=24, IM2COL=3; views free).
- Fixed overhead ≈ 76ms/enc-pass (dispatch-bound), marginal ≈ 2.47ms/Tp-frame.
- Quant landscape (Vulkan, medium): bf16 1612ms / q8_0 769ms / q5_0 826ms /
  q4_0 786ms; WERs all ≈3.4-3.9 (within noise, 32 clips). q8_0 fastest — the
  q5/q4 Vulkan dequant paths are NOT faster than q8_0 here.
- CPU (q8_0, medium): 1684ms. CPU build uses LLAMAFILE ON, NATIVE ON.

## What's Been Tried
- **#2 KEEP** Vulkan fused LayerNorm+affine (norm_affine.comp, NORM_MUL_ADD): -26ms, byte-identical. (patch 0012)
- **#3 KEEP** per-T' pos-bias cache (persistent ReplayGraph inputs): med -30ms, long -112ms, RSS -4MB.
- **#4 KEEP** cwhn depthwise conv (GPU; CPU keeps whcn — CPU cwhn kernel asserts different strides) + GLU cont removal.
- **#5 KEEP** decode gate CONT removal + argmax BLOCK=512.
- **#8 KEEP** F16->F16 cast gating on conv pw weights: med -78ms (was copying 150MB/pass), short -29%. THE big win.
- **#12 KEEP** STARLING_GGML_THREADS env (CPU leg pins 6 physical): cpu_med 1877->1582ms. SMT hurts ~25%.
- **#19 KEEP** DFT-mel (GpuMel) on the CPU backend: mel 80->10.8ms, CPU med -70ms (interleaved cold A/B 1536/1540 vs 1611/1604), byte-identical both paths (sha equal in 6 runs). Kill switch STARLING_MEL_CPU_FFT=1.
- **#18 KEEP** persistent BN scale/shift inputs: encoder h2d 2595->285us/pass (only mel is volatile now). Byte-identical.
- **DEAD (iter 2)**: forced split-K (parity at 2, worse at 4 — m-tiles saturate; reduce overhead eats gains); KSTEP sweep (K=16 optimal for T<=512 — bigger K wastes unconditional LSTM recompute vs saved syncs; all byte-exact).
- **bd matmul identified**: the 24x[557x279x128,b8] f32 = per-head pos-scores (k=dk=128 short-k staging inefficiency, 15.2ms at 500 GFLOPS); merging heads into one GEMM is mathematically invalid (per-head operands differ); fix = custom kernel or FA fusion.
- **Remaining profile anomalies** (need numerics drift or new ops, parked): mel DFT matmul writes 73MB/pass (6.4ms; f16 would halve but breaks byte-exactness); subsampling f32 im2col-matmuls 15.2ms (q8_0 = drift); FA at 366 GFLOPS (21.6ms, near shader ceiling); bd rel-shift chain ~7.5MB copies x24 (fused rel_shift needs a custom op).
- **DEAD ENDS**: FLOPS_PER_SUBMIT, NO_BARRIERS/EXEC_ONLY (nondeterministic races), coherent-loads+fast-sync (L1-bypass cost > flush saving; but PROVEN byte-exact-stable — the technique works, just not profitable here), warptile sweep (base optimal), FORCE_PIPE all variants, m_align 32 (no effect), q4_0 joint/LSTM quant (decode GEMV latency-bound: weight bytes don't matter), q8_0 conv pw quant (Vulkan flat, CPU +5%), exact ×0.5 fold (bit-exact! but flat speed + 408MB RSS from weight copies).
- **MEASUREMENT CAVEAT**: GGML_VK_PERF_LOGGER per-node times are inflated by its own timestamp+barrier — don't trust elementwise numbers from it; only trust the big GEMM lines.
- **Final state**: Vulkan med 770.7->624ms (rtf 0.0346->0.0280, -19%), CPU 1754->1582 (-10%), RSS 978->971, byte-identical transcripts, WER 3.94.
- Remaining big lever: custom FFN GEMM shader (~250ms at ~61% of fp32 peak; needs int24 dot or better dequant pipelining — days of work, see ideas.md).


## Final state (end of session)
- **Vulkan (primary)**: 770.7 → 618-636ms formal band (rtf 0.0346 → 0.0278-0.0285, **−19.7%**)
- **CPU (secondary)**: 1753.8 → 1576-1615ms (**−10%**; pin STARLING_GGML_THREADS=6)
- **Memory**: peak RSS 978 → 973MB; pos-cache adds device-resident ph per cached T (bounded by LRU)
- **Quality**: byte-identical transcripts on every keep (sha1_12=2746b80b32fe throughout); WER 3.94 == baseline
- Shipped: ggml patch 0012 (LN-affine fusion, argmax 512, debug envs) + engine commits through dac718b
- To resume: rebuild build-ar-vk/build-ar-cpu (config in "How to Run"), ./.auto/measure.sh
