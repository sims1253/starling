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
- **#2 KEEP** Vulkan fused LayerNorm+affine (norm_affine.comp, NORM_MUL_ADD, 120 norms): -26ms, byte-identical. (patch 0012)
- **#3 KEEP** per-T' pos-bias cache (persistent ReplayGraph inputs): med -30ms, long -112ms, RSS -4MB. One-time 24 small computes per T (~65ms for new T).
- **#4 KEEP** cwhn depthwise conv (GPU) + GLU cont removal: -192 nodes, -3ms. CPU keeps whcn path (CPU cwhn kernel asserts different strides).
- **#5 KEEP** decode gate CONT removal + argmax BLOCK=512: -6ms formal.
- **#7 KEEP** patch-series hygiene: ggml edits -> 0012 patch.
- **#8 KEEP** F16->F16 cast gating on conv pw weights: med -78ms (150MB/pass of CPY copies!), short -29%. THE big single win.
- **DEAD**: GGML_VK_FLOPS_PER_SUBMIT (worse), NO_BARRIERS/EXEC_ONLY (races), coherent-loads+fast-sync (slower: L1-bypass), warptile sweep (base optimal), FORCE_PIPE variants, q4_0 joint/LSTM quant (GEMV latency-bound not bandwidth — weight bytes don't matter), m_align 64->32 (no effect).
- **Post-#8 encoder profile (medium, 521ms GPU-busy)**: FFN GEMMs 249ms (48%), attn qkvo 64ms, conv f16 52ms, FA 19ms, SILU 15ms, decode ~95ms, elementwise ~25ms, mel ~10ms. GPU ~fully busy (dispatch gaps hidden).
- CPU (secondary): 1706ms after rebuild (cast gating + GLU cont help slightly); pos-cache/cwhn/argmax are GPU-only.
