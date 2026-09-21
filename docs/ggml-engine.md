# Native ggml engines

The native engines load GGUF models through the shared `libstarling_ggml` C API.
They support Parakeet, MOSS, ARK, Higgs, Hojo, Granite, Qwen3, S1, and Audex.
Backend availability depends on the build; see the [benchmarks](benchmarks.md) for measured
performance and the model-specific parity notes below. To run an HTTP server,
use the [native serving guide](native-serving.md). To embed the C API, start
with [build](#build) and [model lifetime](#model-and-backend-lifetime).

This page also contains implementation notes for engine contributors. Fixture
parity results describe the tested inputs and backends, not all possible audio.

## ggml version and local patches

The submodule is pinned to ggml **v0.23.0**,
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`, the upstream release and default-branch
head checked on September 6, 2026. The upgrade from v0.22.0 retains all nine
local patches: upstream has not replaced their behavior.

| Patch | Why it remains |
| --- | --- |
| 0001 | CUDA flash attention still needs per-head additive-mask addressing and dispatch. |
| 0002 | Stable graph UIDs still need the shortcut around per-node replay validation. |
| 0003 | CPU llamafile dispatch still loops over broadcast batches separately. |
| 0006 | CUDA PAD still needs bounded grid dimensions with grid-stride iteration. |
| 0007 | CUDA still lacks the plain LayerNorm + affine fusion used here. |
| 0008 | Transient UID-zero graphs still need exclusion from CUDA graph capture. |
| 0009 | Vulkan GEMV still rejects BF16 activations without the F32 staging conversion. |
| 0010 | Vulkan matmul still assumes contiguous batch strides for in-place operands. |
| 0011 | The allocator still permits reuse of input storage needed by repeated graph execution. |

Patches 0009 and 0010 use the renamed upstream staging flag
`prealloc_y_last_k_padded`; their behavior is unchanged. The other seven apply
without changes. Apply patches in filename order: 0008 depends on 0002, and
0010 depends on 0009. Successful application alone does not establish runtime
parity; CUDA and other accelerator changes require tests on those backends.

Upgrade checks on Linux: CPU and Vulkan engines, C API libraries, and native
servers build. Native unit tests and 45 HTTP/WebSocket contract checks pass.
Vulkan BF16 GEMV, batched/cache-view matmul, op, and mask probes pass on RADV
RENOIR. Parakeet Q8_0 reproduces the short fixture transcript on two consecutive
calls on both CPU and Vulkan; CPU also passes with llamafile enabled. CUDA
compilation and runtime parity remain unverified on this machine, which has
no CUDA toolkit or GPU.

## Model and backend lifetime

Each loaded model owns its realized weights, encoder caches, and decoder replay
state. Two instances of the same model family have separate caches even when
their tensor shapes match. Freeing a context releases its graphs before its
weight buffers; the shared backend remains available to other contexts.

Public C API load, inference, free, and shutdown operations are serialized.
Callers must keep a context alive until its pending calls finish.
`starling_ggml_shutdown()` releases remaining model resources before the backend
and is terminal: subsequent loads or inference return an error. Ordinary
`starling_ggml_free()` allows another model to be loaded. An exit handler performs
shutdown if the caller does not.

`model_lifetime_test` uses distinct synthetic weights to check instance
isolation, unload/reload, failure recovery, and teardown. It runs in CPU CI
with address, undefined-behavior, and leak sanitizers; accelerator replay also
needs testing on the target backend.

## Model registry (adding an engine)

Every engine built into `libstarling_ggml` is registered in one place:
`cpp/lib/model_registry.hpp/.cpp` (`ModelDescriptor`: public enum kind, serve
slug, the load/free/decode entry points, and the error-message shape for the
shared 16 kHz guard). The C API's `load`/`free`/`transcribe_pcm` dispatch
(`cpp/capi.cpp`) and the serve slug mapping, supported-model check, and
`--version` model list (`cpp/serve/server.cpp`) are all table lookups; the
public `starling_ggml.h` defines the C interface (ABI 6).

Adding a model = its `cpp/<model>/` implementation (with the three
`capi_<model>.cpp` entry points) + one contiguous addition in
`cpp/lib/model_registry.cpp`: the three entry-point declarations and the
`kRegistry` row, side by side. The public `starling_ggml_model` enum kind is
added to `starling_ggml.h` with the usual ABI bump (the Python binding's
`_EXPECTED_ABI_VERSION` follows). The dispatch code needs no per-model conditionals.

## Correctness contract

The executable contract is the versioned manifest
`tests/native_parity_manifest.json` (loader/validator:
`tests/parity_contract.py`, self-tests: `tests/test_native_parity_contract.py`).
It keys every gate by model, backend, GGUF sha256 and pinned reference
revision, records the token-suppression / tie-breaking / normalization /
stop-and-truncation policies, and the table below is GENERATED from it
(`uv run python tests/parity_contract.py --write-docs`; a test fails when the
table and the manifest disagree). The engine gates live in
`tests/test_ggml_parity.py`; every gate there is claimed by exactly one
manifest target.

<!-- parity-manifest:v1 begin -->
This table is GENERATED from `tests/native_parity_manifest.json` (schema v1, revision `r1-167-parity-contract`) — edit the manifest, then run `uv run python tests/parity_contract.py --write-docs`. `tests/test_native_parity_contract.py` fails if this table and the manifest disagree. V = validated backend, - = unvalidated.

| Target | Model | Engine | Contract class | Fixtures | Backends (V/-) | Required |
| --- | --- | --- | --- | --- | --- | --- |
| `moss.intree.text` | MOSS-Transcribe-preview-2B | starling-ggml-moss | exact-text | short, medium, long | cpu:V, cuda:- | yes |
| `moss.intree.mel` | MOSS-Transcribe-preview-2B | cpp-test:moss_mel_test | component-tol | short, medium, long | cpu:V, cuda:- | yes |
| `moss.intree.encoder` | MOSS-Transcribe-preview-2B | cpp-test:moss_encoder_test | component-tol | short, medium, long | cpu:V, cuda:- | yes |
| `qwen_decode.shared_fixture` | MOSS-Transcribe-preview-2B LLM trunk (shared lib/qwen_decode stack) | cpp-test:moss_llm_test | component-tol | short, medium, long | cpu:V, cuda:V | yes |
| `qwen_decode.shared_fixture.token_stream` | MOSS-Transcribe-preview-2B LLM trunk (shared lib/qwen_decode stack) | cpp-test:moss_llm_test | exact-tokens | short, medium, long | cpu:-, cuda:V | yes |
| `parakeet.intree.text` | parakeet-tdt-0.6b-v3 | starling-ggml-parakeet | exact-text + long:corpus-quality | short, medium, long | cpu:V, cuda:V | yes |
| `parakeet.intree.ids` | parakeet-tdt-0.6b-v3 | starling-ggml-parakeet | exact-tokens + long:corpus-quality | short, medium, long | cpu:V, cuda:V | yes |
| `moss.crispasr.external` | MOSS-Transcribe-preview-2B (CrispASR f16 build) | crispasr-moss-transcribe (external, DEPRECATED) | exact-text + medium:corpus-quality + long:corpus-quality | short, medium, long | cpu:-, cuda:V | no |
| `parakeet.external` | parakeet-tdt-0.6b-v3 (f16 GGUF) | parakeet.cpp-server (external; renamed SMOKE-QUALITY gate) | exact-text + long:corpus-quality | short, medium, long | cpu:-, cuda:V | no |
| `moss.intree.kstep_regression` | MOSS-Transcribe-preview-2B | cpp-test:moss_kstep_oob_test | smoke (non-certifying) | synthetic_maxcache_boundary | cpu:-, cuda:V | no |
| `ark.intree.text` | ark-asr-3b (BAAI/ARK-ASR-3B) | starling-ggml-ark | exact-text | short, medium, long | cpu:-, cuda:V | no |
| `higgs.intree.text` | bosonai/higgs-audio-v3-stt | starling-ggml-higgs | exact-text | short, medium, long | cpu:-, cuda:V | no |
| `hojo.intree.text` | HojoAI/Hojo-ASR-V1 | starling-ggml-hojo | exact-text | short, medium, long | cpu:-, cuda:V | no |
| `granite.intree.text` | ibm-granite/granite-speech-4.1-2b | starling-ggml-granite | exact-text | short, medium, long | cpu:-, cuda:V | no |
| `qwen3.intree.text` | Qwen/Qwen3-ASR-1.7B | starling-ggml-qwen3 | exact-text | short, medium, long | cpu:-, cuda:V | no |
| `audex.intree.text` | nvidia/nemotron-labs-audex-2b | starling-ggml-audex | exact-text | short, medium, long | cpu:-, cuda:V | no |
| `s1.intree.text` | superwhisper/s1-mini | starling-ggml-s1 | exact-text | short, medium, long | cpu:-, cuda:V | no |
| `s1.intree.control_matrix_smoke` | superwhisper/s1-mini | starling-ggml-s1 | smoke (non-certifying) | control_matrix_16_cells | cpu:-, cuda:V | no |
<!-- parity-manifest:v1 end -->

Semantics:

- **Contract classes.** `exact_token_equality` / `exact_text_equality` compare
  the ENTIRE applicable output — a matching prefix, a single incorrect token,
  and a shifted EOS boundary are all failures (proven by the negative tests in
  `tests/test_native_parity_contract.py`). `component_numerical_tolerance`
  names metric, aggregation, normalization and margin per component.
  `corpus_quality_acceptance` is the explicitly documented approximate path:
  its tolerance block is recorded in the manifest BEFORE any candidate is
  evaluated, never invented at failure time. `smoke_non_certifying` marks
  renamed smoke checks that certify nothing.
- **Required vs unavailable coverage.** A REQUIRED target whose assets are all
  present but which executes zero tests FAILS (a skip-vacuum is a broken gate,
  not a green one). Genuinely unavailable coverage — gitignored GGUFs/goldens
  on a CPU CI runner — is reported distinctly by the accounting tests and only
  fails under `STARLING_PARITY_STRICT=1`. A present GGUF or golden whose
  sha256 differs from the manifest pin FAILS outright: never measure — or
  silently regenerate — the wrong reference (`scripts/moss_golden.py` refuses
  to overwrite existing goldens without `--force`).
- **Reference updates are reviewed changes.** Every tolerance or reference
  change must appear as a justified `changelog` entry in the manifest; the
  MOSS reference record (device, library versions, per-fixture output hashes)
  lives in `golden/moss_reference_provenance.json`, produced by the capture
  script.

Model notes:

- **parakeet-tdt (in-tree `StarlingGgmlParakeet`)**: short/medium are
  byte-exact text AND exact non-blank content-token streams. Blank cadence
  differs on short/long but blanks are not linguistic tokens and are discarded
  by detokenization. LONG is the documented approximate path (manifest targets
  `parakeet.intree.text` / `parakeet.intree.ids`): transformers 5.14 SDPA
  kernel-path drift on the 74 s decode — the golden was captured via HF
  `model.generate` whose SDPA reduction order differs from the in-tree eager
  loop — gated by raw-transcript difflib ratio >= 0.90 plus a >= 0.65
  content-token match-rate floor, WER-verified benign (3.18% ==
  starling-vs-stock).
- **moss (in-tree `StarlingGgmlMoss`)**: exact text on short/medium/long —
  no tolerance. Manifest target `moss.intree.text` pins the reference
  (fresh capture: eager greedy, exact-width `DynamicCache`, device/library
  versions/output hashes recorded) and the GGUF sha256; the historical
  normalized-CER < 0.10 escape hatch on the long fixture was guarding a stale
  reference, not engine drift, and is removed (issue #167). Component ULP and
  max-abs tolerances (mel, encoder/adapter, shared-Qwen decoder fixture) are
  recorded per component in the manifest and enforced by the
  `build/moss_*_test` binaries. The decoder's exact-token fixture
  (`qwen_decode.shared_fixture.token_stream`) is CUDA-scoped for medium/long:
  `cpp/moss/llm.cpp` documents that CPU bf16 GEMMs are not bit-identical to
  cuBLAS (fallback only), and on CPU the medium/long id streams flip near-tie
  argmax positions vs the eager reference — reported, not gated, there; short
  and all component gates are enforced everywhere, and end-to-end CPU text
  exactness is carried by `moss.intree.text`.
- **moss (legacy external CrispASR)**: short is byte-exact; medium/long carry
  the historical normalized-CER smoke gate (renamed
  `test_ggml_moss_crispasr_approx_smoke` — a smoke gate, not certification)
  and single-chunk workaround. This engine is deprecated and does not define
  Starling's in-tree correctness.
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
  chunks with exactly one 750-token clip each, the tail zero-padded to a full
  clip at the mel level, per-chunk budget min(200, ceil(dur*5)+32),
  whitespace-collapsed join) and the `_decode_response` quote extraction
  (first-to-last single-quote span, ported in `capi_audex.cpp`).

Entries whose golden references are gitignored assets are marked `required: no`
in the manifest and carry an UNPINNED reference note until a recapture pins
their hashes; their backend statuses are the recorded historical validations,
not new measurements.

### granite engine notes

- **Mel**: the shared `lib/whisper_mel` frontend with the `T_FULLT` rule
  (torchaudio `center=True` keeps every `S/hop + 1` frame); the odd-frame drop
  and 80→160 pair-stack are engine-side (`cpp/granite/mel.cpp`).
- **Encoder**: CTC conformer with block-local Shaw attention. The per-layer
  `(200, 200, 128)` rel-pos bias is precomputed by the converter (an exact
  embedding gather); the bias term lands as ONE extra batched matmul per
  layer by making the query's within-window position the batch dim. The
  depthwise conv is a 15-tap shift-multiply-accumulate (ggml's im2col is
  unvalidated under CUDA-graph capture in this build); the eval BatchNorm
  recomputes `1/sqrt(var+eps)` in-graph per channel.
- **Projector**: BLIP2 Q-Former (2 BERT-style layers, erf GELU, LN eps 1e-12);
  layer 0's self-attention runs once on the shared 3 queries and the
  cross-attention broadcasts them over the windows.
- **Decoder**: the shared `lib/qwen_decode` stack via a third trunk variant:
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
  zero-padded (mel value 0.0, NOT silence-mel; those frames leak into valid
  conv outputs through the 3-wide kernels) to a multiple of 100 frames
  (`cpp/qwen3/mel.cpp`).
- **Encoder**: per 100-frame chunk three GELU 3x3/stride-2 conv2d layers
  (480 channels) + a bias-free Linear(7680 -> 1024) + a converter-baked
  sinusoidal position table; the valid post-CNN rows (triple ceil-halving,
  13 per full chunk) are gathered into a packed sequence and padded to whole
  104-row attention windows (n_window_infer 800 = 8 chunks), where 24 layers
  run full (non-causal) batched attention (biased MHA, 16 heads x 64),
  masked + trimmed on the tail. The convs are an explicit F32 im2col + F32
  GEMM (`conv_step`): `ggml_conv_2d`'s F16 im2col lands the GEMM on the
  F16-accumulating cuBLAS path. The window-pad tail duplicates row 0 to avoid a concat. Its values never
  reach a valid row because the keys are masked and the other operations are
  row-local; the valid-row gather runs on an F32 copy because this ggml
  build's CPU get_rows bf16 kernel writes f32 rows into the bf16 destination.
- **Projector**: Linear(1024 -> 1024) + erf GELU + Linear(1024 -> 2048), all
  biased.
- **Decoder**: the shared `lib/qwen_decode` stack in its stock Qwen3 variant
  (`qkv_bias=false, qk_norm=true`, TIED lm_head, no multipliers), the same
  spec shape as moss, plus the `argmax_low_ties` extension: torch reads the
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
  rule and the `MAX_KEPT_FRAMES` max scope; the eager WhisperFeatureExtractor
  drops the trailing STFT frame (`stft[..., :-1]`) BEFORE the global
  max-clamp, so the normalization max runs over the kept 3000 frames only.
  Every clip is zero-padded to the full 30 s / 480000 samples
  (padding="max_length") BEFORE the mel, so the frame count (3000) and
  every downstream encoder shape are fixed (`cpp/audex/mel.cpp`).
- **Encoder**: the stock Qwen2AudioEncoder (whisper-large-v3 shaped) with
  fixed shapes: two GELU Conv1d k3/p1 layers over time (stride 1 then 2:
  3000 -> 1500), the LEARNED (1500, 1280) positional table, 32 pre-norm
  layers of FULL bidirectional attention (no mask; the reference attends
  padded tail frames like any other; 20 heads x 64, biased q/v/out with a
  bias-free k, the query pre-scaled by 0.125 at projection), an avg-pooler
  halving 1500 -> 750 (even/odd strided views, f32 pair average, one bf16
  round), and the final biased LayerNorm. The convs are an explicit F32
  im2col + F32 GEMM (the qwen3 `conv_step` pattern with a degenerate H axis);
  as with qwen3, a GEMM formulation cannot bitwise-match cuDNN conv in
  general; parity holds on the gated fixtures.
- **Projector**: single-round RMSNorm(1280, eps 1e-5) -> bias-free fc1
  (-> 4096) -> relu(x)^2 (relu exact, one bf16 round after the square) ->
  bias-free fc2 (-> 2048).
- **Decoder**: the shared `lib/qwen_decode` stack in a new Nemotron-Dense
  variant (`qkv_bias=false, qk_norm=false`, UNTIED lm_head, no multipliers,
  `argmax_low_ties`) plus TWO skip-when-default spec extensions: the
  `mlp_activation=relu2_plain` MLP (up -> relu^2 -> down, no gate tensor) and
  `rms_norm_single_round`. Nemotron uses `F.rms_norm`: normalization and
  affine operations run in f32 with one bf16 round at the end. The stack's
  default Llama-style path rounds after rsqrt. These rounding rules differ on
  ~25% of elements, verified against `torch.nn.functional.rms_norm`.
  Defaults keep the moss/ark/granite/qwen3 graphs byte-identical.

### s1 engine notes (first text-to-text engine)

S1-mini (`superwhisper/s1-mini`, 0.6B Qwen3 decoder-only) has **no audio
front-end**; `cpp/s1/` is loader + LLM binding + C-API shell only, and its
registry row is the first with a `normalize_fn` (the PCM `decode_fn` is a
stub that points callers at `starling_ggml_normalize_text` / `POST
/normalize`; `starling_ggml.h` ABI 6 adds the enum kind + the text entry
point).

- **Tokenizer**: the first engine that ENCODES. `lib/bpe_tokenizer` gained a
  byte-level BPE encoder over the GGUF merge table: special-token longest
  match first, then the Qwen pre-tokenizer regex hand-rolled over codepoints
  (`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`; note Qwen splits
  digits individually), then lowest-rank pair merging. ASCII input is exact;
  non-ASCII is classified L except a small explicit punctuation set (see
  `bpe_tokenizer.cpp`). The decoder half is unchanged for every engine.
- **Prompt**: the chat template is baked in the GGUF as prefix/suffix id
  arrays around the runtime-encoded user content (`[Styling: s] [Structure:
  t] [Context: c]\n<transcript>`), including the `enable_thinking=False`
  assistant prefix `<think>\n\n</think>\n\n`. Both boundaries are pre-token
  splits, so prefix + encode(content) + suffix is token-identical to encoding
  the rendered string (asserted at conversion time).
- **Decoder**: the shared `lib/qwen_decode` stock Qwen3 spec (identical to
  qwen3's modulo env/label/dims: 28 layers, hidden 1024, GQA 16Q/8KV,
  head_dim 128) with `argmax_low_ties` on, plus the `eos2_token_id`
  extension: s1's generation_config stops on BOTH `<|im_end|>` (151645) and
  `<|endoftext|>` (151643), so `GenerateParams` carries an optional second
  stop id (default -1; the older engines' stop behavior is unchanged).
- **Correctness**: byte-exact vs stock transformers on the short/medium/long
  fixture tiers and 15/16 control-matrix cells; the one divergent cell
  (semi-casual/lists/general) hits an EXACT bf16 tie between `' so'` and
  `','` (stock-eager top-2 f32 gap 0.0000); the winner is argmax tie-break
  order, not numerics (both fast paths pick the same side; the diff is one
  comma, CER < 1%). Documented + gated in `benchmarks/s1/bench_normalize.py`.
- **OOD inputs and the limits of byte-exactness**: an 80-case synthetic fuzz
  (random word-salad transcripts, fillers, numbers, tabs/caps edges,
  near-cap lengths) diverges from stock greedy on ~40-50% of cases; every
  sampled divergence was verified tie-class: at the first differing token,
  the fast path picked stock's literal #2 and the top-2 f32 gap was <= 0.125
  at logit magnitude ~20, at or below one bf16 ULP (ULP(20) = 0.125),
  unresolvable in bf16. One flip cascades into a wholly different
  continuation on OOD input (the model is uncertain there; realistic ASR
  transcripts like the fixtures have decisive argmax and stay
  byte-exact). Byte-exactness is therefore a property of in-distribution
  prompts, not arbitrary text; the parity gates run on the fixtures.

## Backends

The in-tree runtime selects compute through **ggml's device registry**
(`cpp/runtime/backend.cpp`), not model-specific backend code. It picks the first
GPU/IGPU or a device named by `STARLING_GGML_DEVICE` (`CUDA0`, `Vulkan0`,
`Metal`, `cpu`, ...). The ggml build controls which backends are compiled.

### NVIDIA CUDA (primary, verified)

Build from the repository root with `-DSTARLING_GGML_CUDA=ON`. Verified byte-exact and benchmarked
on RTX 5090 (Blackwell, sm_120). See [performance](#performance-rtx-5090-bf16-b1-model-load-excluded).

`ggml_backend_cuda_device_supports_op` rejects `GGML_OP_UNARY` whose `src0` is
not fully contiguous (`ggml-cuda.cu`, UNARY case: `ggml_is_contiguous(op->src[0])`;
the plain unary kernel is flat-indexed and asserts the same). The check is
architecture-independent — a layout gate, not an sm_120 kernel-coverage gap —
so graph builders must materialize strided views with `ggml_cont` before a
unary op (the conformer GLU gate half and the TDT LSTM gates do; MUL consumes
row-strided operands fine). A captured `ReplayGraph` whose primary accelerator
rejects any node fails at graph build with an engine error enumerating every
rejected node (`STARLING_SCHED_DEBUG=1` echoes them to stderr): captured
replays upload inputs through the primary backend, which cannot address
sched-allocated buffers. One-shot graphs fall back to `ggml_backend_sched`
safely and are unaffected.

### CPU

`STARLING_GGML_DEVICE=cpu` forces the CPU backend, which is compiled into every
build. The recorded CPU checks reproduce the golden fixture transcripts.
CPU inference is about 10-20 times slower than CUDA in the recorded benchmarks
and is useful for correctness checks and machines without a supported GPU.

### Apple Metal

Build for Apple Silicon with `-DSTARLING_GGML_METAL=ON` and select the device
with `STARLING_GGML_DEVICE=Metal`. The kernels are in
`third_party/ggml/src/ggml-metal/`. Metal uses the shared compute graphs and
`ReplayGraph` abstraction. Runtime parity is not verified by the checks recorded
here; it requires Apple hardware. BF16 is the intended numerical format for
non-NVIDIA backends. Apple/mobile performance tuning remains follow-up work.

### Vulkan

Built with `-DSTARLING_GGML_VULKAN=ON` (`third_party/ggml/src/ggml-vulkan/`). Select
with `STARLING_GGML_DEVICE=Vulkan0`. Targets the Intel/AMD/ARM GPUs CUDA can't
reach. Same graph-replay path as CUDA/Metal.

In-tree ggml patches 0009/0010/0011 (GEMV bf16 upcast, batch-strided
mul_mat, galloc INPUT-storage reuse) carry the Vulkan-correctness story;
the per-finding root-cause record is `scripts/diagnostics/vulkan/README.md`.
Known upstream gap, deliberately NOT patched here: the `mul_mat_id`
(MoE dispatch) variants `ggml_vk_mul_mat_id_q_f16` (wide) and
`ggml_vk_mul_mat_vec_id_q_f16` (GEMV) still hardcode contiguous batch
strides (`ne00*ne01` / `ne10*ne11`) for in-place operands, and the wide
variant also falls back to `nb[0]`, which is not a batch stride.
Patch 0010 removed these two defects from `mul_mat`. (The GEMV
variant's y-stride fallback is already `nb[2]`-derived; its x batch
stride is hardcoded inline in the push constants with no fallback.)
Starling's engines never emit `mul_mat_id` (no MoE models), so it is
untriggered; a MoE engine would need the same nb[2]-derived-stride
treatment before its KV-cache views are trustworthy on Vulkan.

### HIP (AMD) / SYCL (Intel)

Build HIP with `-DSTARLING_GGML_HIP=ON`; see the compiler flags in the
[native serving guide](native-serving.md#build). Upstream ggml also provides
SYCL through `-DGGML_SYCL=ON`. Starling does not publish a SYCL release variant
or claim runtime validation for it here.

## Graph replay

Starling builds model components as reusable `ggml_cgraph` objects to reduce
host launch overhead. `ReplayGraph` keeps each graph's storage and identity
stable across calls. On CUDA, ggml can capture these persistent graphs as CUDA
graphs. Other backends execute the same compute graphs through their own
implementations; this does not imply equal performance or numerical results.

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

The maintained synthetic-fixture table is in [benchmarks](benchmarks.md) and is generated by
`benchmarks/bench_all.py`. At the current verified baseline, in-tree Parakeet is
14 / 30 / 86 ms (short / medium / long) versus 16 / 24 / 58 ms for the PyTorch
peak; in-tree MOSS is 214 / 535 / 1180 ms versus 166 / 397 / 1499 ms. Use the
real-corpus benchmark table for workload throughput and WER.

Parakeet's remaining medium/long gap is the serial, data-dependent TDT decode.
K-step capture and device-resident state are already implemented; long K>16
still uses host state round-trips because the persistent-device writeback path
hit a ggml CUDA-graph topology defect. The encoder is already at its measured
hardware floor. MOSS decode is near the PyTorch per-token floor after whole-model
capture and device-resident KV; remaining work is dominated by mel/encoder and
prefill rather than another host-controlled per-layer decode rewrite.

## Build

All nine engines listed above are built into `libstarling_ggml`.
Build the shared library from the repository root. Initialize the submodules
first; CMake applies the local ggml patches. The build needs CMake 3.18 or later,
a C++17 compiler, Git, Bash, and the development toolkit for the chosen GPU
backend. See [runtime prerequisites](native-serving.md#release-artifacts) when
distributing binaries.

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

The legacy external `GgmlMoss` CrispASR engine is deprecated and remains
available for A/B comparisons.
It is not required to build or run the Starling-owned MOSS path.
