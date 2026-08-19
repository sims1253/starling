# ggml engine — universal backend ASR for Starling

Starling's in-tree `starling-ggml-parakeet` and `starling-ggml-moss` engines
bring a second, **universal-backend** transcription path alongside the PyTorch
+ CUDA-graph peak engine. Both are model-tagged entry points in the shared
`libstarling_ggml` C API and reach any backend ggml supports (NVIDIA CUDA, AMD
via HIP, Apple Metal, Intel/AMD/ARM via Vulkan, CPU) from one codebase. Legacy
external `GgmlParakeet` and `GgmlMoss` wrappers remain only for temporary A/B
comparison and are deprecated pending Phase-4 removal.

The PyTorch engine remains the NVIDIA peak path (CUDAGraph + Triton fused
kernels, tuned on sm_120). The ggml engines are dispatched alongside it: same
fixtures, same golden contract, portable. **Correctness:** parakeet-tdt and the
in-tree MOSS engine return the exact canonical eager token IDs/text on all
fixtures. MOSS logits retain documented bf16 ULP differences, so its component
gates use max-abs tolerances while the end-to-end contract is token/text exact.
The legacy CrispASR MOSS engine remains only a near-exact comparison path.
**Speed:** the current in-tree Parakeet path is faster than the PyTorch peak on
short and within ~1.25-1.5x on medium/long. The current in-tree MOSS path is
within ~1.3x on short/medium and faster on the long synthetic fixture. See the
maintained README tables; older external-engine measurements are historical
context only.

## Model registry (adding an engine)

Every engine built into `libstarling_ggml` is registered in one place:
`cpp/lib/model_registry.hpp/.cpp` (`ModelDescriptor`: public enum kind, serve
slug, the load/free/decode entry points, and the error-message shape for the
shared 16 kHz guard). The C API's `load`/`free`/`transcribe_pcm` dispatch
(`cpp/capi.cpp`) and the serve slug mapping, supported-model check, and
`--version` model list (`cpp/serve/server.cpp`) are all table lookups; the
public `starling_ggml.h` stays flat and unchanged (ABI 5).

Adding a model = its `cpp/<model>/` implementation (with the three
`capi_<model>.cpp` entry points) + one contiguous addition in
`cpp/lib/model_registry.cpp`: the three entry-point declarations and the
`kRegistry` row, side by side. The public `starling_ggml_model` enum kind is
added to `starling_ggml.h` with the usual ABI bump (the Python binding's
`_EXPECTED_ABI_VERSION` follows). No other per-model edits anywhere in the
tree.

## Correctness contract

Byte-exact transcripts vs the golden references (`golden/parakeet_tdt_*.txt`
for parakeet, `golden/moss_*.txt` for moss), asserted by
`tests/test_ggml_parity.py` (skipped if the ggml binaries aren't built).

- **parakeet-tdt (in-tree `StarlingGgmlParakeet`)**: byte-exact text on
  short/medium/long and exact non-blank content-token streams. Blank cadence
  differs on short/long but blanks are not linguistic tokens and are discarded
  by detokenization. Both gates live in `tests/test_ggml_parity.py`.
- **moss (in-tree `StarlingGgmlMoss`)**: exact eager greedy token IDs and text
  on short/medium/long. The reference explicitly propagates eager attention to
  nested model configs and uses exact-width `DynamicCache`; padded StaticCache
  reduction-order noise is not a golden contract. Component ULP tolerances are
  documented alongside the component tests.
- **moss (legacy external CrispASR)**: short is byte-exact; medium/long retain
  the historical normalized-CER gate and single-chunk workaround. This engine
  is deprecated and does not define Starling's in-tree correctness.
- **granite (in-tree `StarlingGgmlGranite`)**: greedy path only (the
  self-speculative CTC-drafting path stays Python-side and is byte-identical
  to greedy by construction). Exact text on short/medium/long against
  `golden/granite_reference.json`, captured from the stock-numerics Python
  path by `scripts/make_granite_golden.py` (staged component tensors via
  `scripts/granite_golden_components.py`). The engine mirrors the Python
  server's chunk policy for long audio (30 s zero-padded chunks, per-chunk
  budget clamped to the 640-token cache, whitespace-collapsed join).
- **qwen3 (in-tree `StarlingGgmlQwen3`)**: greedy path. Exact text on
  short/medium/long against `golden/qwen3_reference.json`, captured from the
  stock-numerics Python path by `scripts/make_qwen3_golden.py` (staged
  component tensors via `scripts/qwen3_golden_components.py`). The engine
  mirrors the Python server's chunk policy for long audio (contiguous 30 s
  chunks, the last chunk passed through SHORT, per-chunk budget
  min(200, ceil(dur*5)+32), whitespace-collapsed join) and the
  `transcription_only` text extraction (the `<asr_text>` marker split with the
  Qwen3-ASR library's repetition fix, ported in `capi_qwen3.cpp`).
- **audex (in-tree `StarlingGgmlAudex`)**: greedy path. Exact text on
  short/medium/long against `golden/audex_reference.json`, captured from the
  stock-numerics Python path by `scripts/make_audex_golden.py` (staged
  component tensors via `scripts/audex_golden_components.py`). The engine
  mirrors the Python server's chunk policy for long audio (contiguous 30 s
  chunks — exactly one 750-token clip each, the tail zero-padded to a full
  clip at the mel level, per-chunk budget min(200, ceil(dur*5)+32),
  whitespace-collapsed join) and the `_decode_response` quote extraction
  (first-to-last single-quote span, ported in `capi_audex.cpp`).

### granite engine notes

- **Mel**: the shared `lib/whisper_mel` frontend with the `T_FULLT` rule
  (torchaudio `center=True` keeps every `S/hop + 1` frame); the odd-frame drop
  and 80→160 pair-stack are engine-side (`cpp/granite/mel.cpp`).
- **Encoder**: CTC conformer with block-local Shaw attention. The per-layer
  `(200, 200, 128)` rel-pos bias is precomputed by the converter (an exact
  embedding gather) — the bias term lands as ONE extra batched matmul per
  layer by making the query's within-window position the batch dim. The
  depthwise conv is a 15-tap shift-multiply-accumulate (ggml's im2col is
  unvalidated under CUDA-graph capture in this build); the eval BatchNorm
  recomputes `1/sqrt(var+eps)` in-graph per channel.
- **Projector**: BLIP2 Q-Former (2 BERT-style layers, erf GELU, LN eps 1e-12);
  layer 0's self-attention runs once on the shared 3 queries and the
  cross-attention broadcasts them over the windows.
- **Decoder**: the shared `lib/qwen_decode` stack via a third trunk variant —
  `QwenDecodeSpec` with `qkv_bias=false, qk_norm=false`, an UNTIED
  `llm.lm_head`, and the Granite multipliers (attention scale 0.0078125,
  embedding ×12.0 applied to the whole merged inputs_embeds at prefill and the
  embed lookup at decode, residual ×0.22, logits ÷8.0). All spec extensions
  default to the historical moss/ark op sequence, so the older engines' graphs
  are unchanged.

### qwen3 engine notes

- **Mel**: the shared `lib/whisper_mel` frontend with the `T_FULLT_MINUS_1`
  rule (the extractor computes the mel over `stft[..., :-1]`, i.e. drops the
  trailing frame), slaney filterbank baked by the converter; engine-side,
  clips under 8000 samples are zero-padded first and the mel axis is then
  zero-padded (mel value 0.0, NOT silence-mel — those frames leak into valid
  conv outputs through the 3-wide kernels) to a multiple of 100 frames
  (`cpp/qwen3/mel.cpp`).
- **Encoder**: per 100-frame chunk three GELU 3x3/stride-2 conv2d layers
  (480 channels) + a bias-free Linear(7680 -> 1024) + a converter-baked
  sinusoidal position table; the valid post-CNN rows (triple ceil-halving,
  13 per full chunk) are gathered into a packed sequence and padded to whole
  104-row attention windows (n_window_infer 800 = 8 chunks), where 24 layers
  run full (non-causal) batched attention — biased MHA 16 heads x 64 —
  masked + trimmed on the tail. The convs are an explicit F32 im2col + F32
  GEMM (`conv_step`): `ggml_conv_2d`'s F16 im2col lands the GEMM on the
  F16-accumulating cuBLAS path. The window-pad tail duplicates row 0 — its values never reach a
  valid row (masked as keys, row-local ops) — which avoids a concat
  entirely; the valid-row gather runs on an F32 copy because this ggml
  build's CPU get_rows bf16 kernel writes f32 rows into the bf16 destination.
- **Projector**: Linear(1024 -> 1024) + erf GELU + Linear(1024 -> 2048), all
  biased.
- **Decoder**: the shared `lib/qwen_decode` stack in its stock Qwen3 variant
  (`qkv_bias=false, qk_norm=true`, TIED lm_head, no multipliers) — the same
  spec shape as moss — plus the `argmax_low_ties` extension: torch reads the
  lm_head output stored as bf16 and keeps the FIRST index on exact ties,
  while raw f32 logits and ggml's CUDA argmax (warp-order ties) can pick the
  other side of a tie; the extension bf16-rounds the greedy logits, the host
  picks keep-first-index on the exact ties, and the K-step graph masks the
  rounded logits by equality with their max (ggml_argmax's VALUE is
  order-independent) and weights the masked columns by a descending column
  iota (`vocab - col`, exact integers < 2^24), making the lowest tied column
  a unique argmax. Skip-when-default, so the moss/ark/granite graphs stay
  byte-identical.

### audex engine notes

- **Mel**: the shared `lib/whisper_mel` frontend with the `T_FULLT_MINUS_1`
  rule and the `MAX_KEPT_FRAMES` max scope — the eager WhisperFeatureExtractor
  drops the trailing STFT frame (`stft[..., :-1]`) BEFORE the global
  max-clamp, so the normalization max runs over the kept 3000 frames only.
  Every clip is zero-padded to the full 30 s / 480000 samples
  (padding="max_length") BEFORE the mel, so the frame count (3000) — and with
  it every downstream encoder shape — is fixed (`cpp/audex/mel.cpp`).
- **Encoder**: the stock Qwen2AudioEncoder (whisper-large-v3 shaped) with
  fixed shapes: two GELU Conv1d k3/p1 layers over time (stride 1 then 2:
  3000 -> 1500), the LEARNED (1500, 1280) positional table, 32 pre-norm
  layers of FULL bidirectional attention (no mask — the reference attends
  padded tail frames like any other; 20 heads x 64, biased q/v/out with a
  bias-free k, the query pre-scaled by 0.125 at projection), an avg-pooler
  halving 1500 -> 750 (even/odd strided views, f32 pair average, one bf16
  round), and the final biased LayerNorm. The convs are an explicit F32
  im2col + F32 GEMM (the qwen3 `conv_step` pattern with a degenerate H axis);
  as with qwen3, a GEMM formulation cannot bitwise-match cuDNN conv in
  general — parity holds on the gated fixtures.
- **Projector**: single-round RMSNorm(1280, eps 1e-5) -> bias-free fc1
  (-> 4096) -> relu(x)^2 (relu exact, one bf16 round after the square) ->
  bias-free fc2 (-> 2048).
- **Decoder**: the shared `lib/qwen_decode` stack in a new Nemotron-Dense
  variant (`qkv_bias=false, qk_norm=false`, UNTIED lm_head, no multipliers,
  `argmax_low_ties`) plus TWO skip-when-default spec extensions: the
  `mlp_activation=relu2_plain` MLP (up -> relu^2 -> down, no gate tensor) and
  `rms_norm_single_round` (Nemotron normalizes with `F.rms_norm` — normalize
  AND affine in f32, ONE bf16 round at the end — vs the stack's historical
  Llama-style round-after-rsqrt; the two disciplines differ on ~25% of
  elements, verified empirically against `torch.nn.functional.rms_norm`).
  Defaults keep the moss/ark/granite/qwen3 graphs byte-identical.

## Backends

The in-tree runtime selects compute through **ggml's device registry**
(`cpp/runtime/backend.cpp`), not model-specific backend code. It picks the first
GPU/IGPU or a device named by `STARLING_GGML_DEVICE` (`CUDA0`, `Vulkan0`,
`Metal`, `cpu`, ...). The ggml build controls which backends are compiled.

### NVIDIA CUDA (primary, verified)
The default. Built with `-DGGML_CUDA=ON`. Verified byte-exact and benchmarked
on RTX 5090 (Blackwell, sm_120). See the perf table below.

### CPU (verified — the non-NVIDIA backend)
`STARLING_GGML_DEVICE=cpu` forces the CPU ggml backend. **Verified byte-exact** vs
the golden on all fixtures (the eager greedy-TDT path is deterministic, and the
CPU backend runs the identical model math). This satisfies the project's "at
least one non-NVIDIA backend compiles + runs correctly" requirement: the CPU
backend is a distinct ggml backend, compiled in every build, and reproduces the
golden transcript bit-for-bit. It is ~10-20x slower than CUDA (no graph
capture, CPU kernels) — a correctness/fallback path, not a perf path.

### Apple Metal (gate + document)
Runs on Apple Silicon with a ggml built `-DGGML_METAL=ON` (the Metal kernels
ship in `third_party/ggml/src/ggml-metal/`). Select with
`STARLING_GGML_DEVICE=Metal`. **Not verified in CI here** — the development machine
is x86/WSL2 with no Apple hardware. The path is architectural: the encoder is
captured in a ggml compute graph (the portable CUDAGraph equivalent) and the
decode uses ggml's `ReplayGraph`, both of which replay on Metal the same way
they replay on CUDA. Per the project's OUT-OF-SCOPE note, Apple/mobile perf
tuning beyond "it runs" is a follow-up; bf16 (not fp8) is the portable
numerics contract on non-NVIDIA.

### Vulkan (universal: Intel / AMD / ARM)
Built with `-DGGML_VULKAN=ON` (`third_party/ggml/src/ggml-vulkan/`). Select
with `STARLING_GGML_DEVICE=Vulkan0`. Targets the Intel/AMD/ARM GPUs CUDA can't
reach. Same graph-replay path as CUDA/Metal.

### HIP (AMD) / SYCL (Intel)
Supported by ggml's registry; selected the same way when ggml is built with
`-DGGML_HIP=ON` / `-DGGML_SYCL=ON`.

## How the launch-folding works (the portable CUDAGraph)

Starling's PyTorch peak wins by capturing the decode/encoder loop into a
CUDA graph, eliminating host launch overhead (the README's "hundreds of tiny
kernels, GPU ~10% busy" problem). The ggml equivalent is ggml's compute graph:
each model component (24-layer Conformer encoder, per-step TDT joint/prediction)
is built as ONE `ggml_cgraph` and replayed, so the backend folds all its ops
into minimal device submissions. On CUDA, ggml captures the replayed graph as a
CUDA graph itself (keyed on the graph's first node pointer, which is why the
encoder is routed through a per-shape `ReplayGraph` that keeps that pointer
stable across calls — `src/encoder.cpp`).

## One-shot graph safety

`run_graph` graphs are transient (`cgraph->uid == 0`); only persistent
`ReplayGraph` instances receive stable nonzero UIDs and may use ggml-CUDA graph
capture (patch 0008). This avoids pointer-key collisions across recycled
one-shot metadata contexts. Intermediate captures must be expanded explicitly,
but diagnostic capture branches are not a numerical oracle: changing which
tensors are marked output changes gallocr reuse. In particular, never mark a
graph-input leaf as an output merely to inspect it. MOSS's former
`ggml_set_output(mask_input)` experiment changed the allocation layout and
produced non-deterministic mask/softmax garbage; the durable LLM parity probes
instead select an intermediate as the graph's normal output via
`STARLING_MOSS_L0_STAGE`.

## Performance (RTX 5090, bf16, B=1, model load excluded)

The maintained synthetic-fixture table is in `README.md` and is generated by
`benchmarks/bench_all.py`. At the current verified baseline, in-tree Parakeet is
14 / 30 / 86 ms (short / medium / long) versus 16 / 24 / 58 ms for the PyTorch
peak; in-tree MOSS is 214 / 535 / 1180 ms versus 166 / 397 / 1499 ms. Use the
real-corpus README table for workload throughput and WER.

Parakeet's remaining medium/long gap is the serial, data-dependent TDT decode.
K-step capture and device-resident state are already implemented; long K>16
still uses host state round-trips because the persistent-device writeback path
hit a ggml CUDA-graph topology defect. The encoder is already at its measured
hardware floor. MOSS decode is near the PyTorch per-token floor after whole-model
capture and device-resident KV; remaining work is dominated by mel/encoder and
prefill rather than another host-controlled per-layer decode rewrite.

## Build

Parakeet and MOSS are both built into Starling's shared `libstarling_ggml`.
Build the in-tree library from the repository root:
```
flock /tmp/starling-cpp-build.lock bash -c \
  'cmake -B build -DSTARLING_GGML_CUDA=ON -DSTARLING_GGML_SHARED=ON && cmake --build build -j'
```

Place Starling's exact BF16 GGUF at
`models/moss-transcribe-preview-2b-bf16-exact.gguf`, or override it with
`STARLING_GGML_MOSS_MODEL=/path/to/model.gguf`. The benchmark key is
`starling-ggml-moss`; it loads once and calls the in-tree C API directly. For
example, the Python binding is `GgmlModel(MOSS, path)` from
`starling._ggml`.

The legacy external `GgmlMoss` CrispASR engine remains available temporarily
for A/B comparisons, but is **deprecated** and will be removed in Phase 4.
It is not required to build or run the Starling-owned MOSS path.
