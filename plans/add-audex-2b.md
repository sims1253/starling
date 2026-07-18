# Plan: add nvidia/Nemotron-Labs-Audex-2B (ASR path) to starling

Goal: a new `audex` backend that transcribes speech with CUDA-graph capture,
byte-identical to the eager `transformers` reference at bf16, registered in the
server, both benchmarks, tests, and the README — same bar as every other model
in this repo.

Scope: **ASR only.** Audex is a unified audio-text LLM (ASR, audio
understanding, TTS, speech-to-speech). We implement only the
audio-in → text-out transcription path. Everything related to audio
*generation* is out of scope: `checkpoint_folder_audiogen`,
`audex_causal_speech_decoder/`, `enhancement_VAE/`, XCodec1/XCodec2, and the
`nemotron_dense_vllm_plugin/` (useful only as a reference implementation).

> Provenance note: the architecture facts below were read from the HF model
> card and `checkpoint_folder_full/config.json` via web fetch. Treat them as
> INFERRED until you re-verify each one against the downloaded files in
> Phase 0. Do not build kernels on an unverified dim.

## License flag (surface to the user, do not silently proceed)

The model is under the **NVIDIA Oneway Noncommercial License**. Every other
model in the README is Apache/MIT-ish. Mention the license in the README entry.

## Architecture (as fetched, re-verify in Phase 0)

The HF repo root `config.json` is only a **manifest** (`model_type:
"nemotron_labs_audex"`) pointing at sub-checkpoints. ASR loads
`checkpoint_folder_full/` as the model path — i.e.
`AutoModel.from_pretrained("nvidia/Nemotron-Labs-Audex-2B", subfolder="checkpoint_folder_full", ...)`
or a local snapshot path into that folder. `checkpoint_folder_textonly` has the
audio parts stripped; not usable here.

- Architecture class: `NemotronDenseAudexForConditionalGeneration`,
  `model_type: "nemotron_dense_audex"`, `trust_remote_code` via `auto_map`
  (`configuration_nemotron_h_audio.py` / `modeling_nemotron_h_audio.py`).
  Despite the `nemotron_h` file naming, the config has **no Mamba/SSM/hybrid
  keys** — the 2B decoder is dense attention. Confirm by reading the modeling
  file anyway.
- Audio encoder ("NV-Whisper", registered as `qwen2_audio_encoder`):
  Whisper-large-v3-shaped — 128 mel bins, 32 layers, d_model 1280, 20 heads,
  FFN 5120, max_source_positions 1500. 16 kHz input, 30.0 s clip window,
  emits **750 sound embeddings per 30 s clip** (`sound_embedding_size: 750`,
  so 1500 frames pooled ×2 — verify where the pooling happens).
- Projector: encoder 1280 → LLM 2048, intermediate 4096, activation `relu2`.
- LLM decoder (Nemotron dense 2B): 28 layers, hidden 2048, 16 attn heads /
  8 KV heads (GQA), head_dim 128, intermediate 9216, activation **`relu2`
  (squared ReLU, non-gated MLP — NOT SwiGLU)**, vocab **205,312**,
  rope_theta 1e8, max_position_embeddings 131072, norm eps 1e-5.
- Multimodal tokens: `<so_start>`=30, `<so_embedding>`=29, `<so_end>`=31.
  Audio embeddings are scattered into the `<so_embedding>` slots (same
  masked_scatter pattern as qwen3/ark).
- Prompting: ChatML. ASR prompt from the model card:
  `"<sound>\nTranscribe the speech in the input audio.\n<sound>"` with
  **greedy** decoding. The model has thinking/instruct modes (`<think>` tags) —
  find how the chat template disables thinking and pin the ASR path to
  non-thinking, otherwise the decode loop burns tokens on reasoning.
- Checkpoint dtype is **float32** (~8 GB for 2B). Load and cast to bf16; the
  golden reference is the bf16 eager model, matching repo policy.

## Phase 0 — recon + eager baseline + goldens

1. Download the repo (`hf download`, or let `from_pretrained` cache it). Only
   `checkpoint_folder_full/` + tokenizer/processor files + the remote-code
   `.py` files are needed; skip audiogen/VAE folders if using selective
   download (12.3 GB total repo).
2. Read `inference_scripts_hf/` in the HF repo — it is the ground truth for
   how audio is featurized (is there an `AutoProcessor`? a hand-rolled
   `WhisperFeatureExtractor`?), how the ChatML prompt is assembled for ASR,
   and what generation kwargs are used.
3. Model card says transformers >= 4.53, "compatible with >= 5.0". Verify the
   remote code actually imports and runs under this repo's pinned
   transformers (5.x). If it breaks, follow the higgs precedent: isolated
   venv + `UV_NOTES.md` (see `src/starling/higgs/UV_NOTES.md`). Prefer
   vendoring + patching the remote code (`src/starling/audex/vendor/`, like
   `moss/vendor` and `higgs/vendor`) over a second venv if the breakage is
   small.
4. Run eager bf16 transcription on the repo's standard sample audio and on a
   LibriSpeech fixture (see `tests/fixtures/`). Confirm sane transcripts.
5. Dump goldens to `golden/` (gitignored), following `qwen3/golden.py`:
   mel/input features, encoder last_hidden, projector output (audio_embeds),
   merged `inputs_embeds`, per-step logits for a short decode, final token
   ids + text. These are the byte-exact targets for every later phase.
6. Re-verify every dim in the Architecture section against the downloaded
   `config.json` + modeling code and correct this plan / the new
   `audex/config.py` accordingly.

## Phase 1 — package scaffold `src/starling/audex/`

Mirror the qwen3/ark layout (encoder + LLM-decoder family):

- `config.py` — all verified dims/token-ids as module constants, docstringed,
  like `src/starling/config.py` and `qwen3/config.py`.
- `loader.py` — `load_model_and_processor(attn_impl="eager", dtype=bf16)` +
  `get_components()` resolving (audio encoder, projector, decoder trunk,
  lm_head, token embeddings). `attn_implementation="eager"` for the decoder so
  StaticCache + 4D-mask decode matches golden (see `ark/loader.py` docstring
  for the rationale).
- `audio.py` — `build_inputs(...)`: waveform → 128-mel features (GPU mel
  path exists for 128-bin Whisper in higgs — reuse), ChatML prompt with
  `<so_start> <so_embedding>*N <so_end>` expansion, 30 s windowing hooked
  into `stream_chunk.py` for long audio like qwen3.

Gate: eager pipeline through these helpers reproduces Phase-0 goldens exactly.

## Phase 2 — graphed encoder `encoder_mega.py`

Port the `GraphedEncoder` pattern from `qwen3/encoder_mega.py` (both are
Whisper-style: conv frontend + transformer stack). Fixed 30 s window means a
fixed encoder shape — ideal graph capture, likely one shape bucket.
Include the ×2 pooling to 750 embeds wherever the reference puts it.

Gate: encoder output byte-exact vs golden (`ENCODER_ATOL` style comparison at
bf16; the repo standard is byte-identical for graphed vs eager — hold that).

## Phase 3 — fused decoder `llm_mega.py` (+ `multistep.py`)

Port `FusedLLMMega`/`LLMMega` from qwen3. Deltas vs qwen3 to handle:

- **MLP is relu2, not SwiGLU.** The fused SwiGLU kernels don't apply. Either
  add a relu2 variant of the fused MLP (squared-ReLU is elementwise —
  straightforward) or leave the MLP as plain linears + `relu()**2` inside the
  graph first, and fuse later. Correctness first.
- GQA 16/8, head_dim 128, 28 layers — within what the attention path already
  supports (`attention.py`); confirm.
- rope_theta 1e8 — just a constant, but verify the rope implementation the
  remote code uses (standard vs scaled) before reusing the shared rope.
- Vocab 205,312 → the lm_head GEMV is ~2.6× granite's. It will work unfused;
  note `fp8_gemv.py` (granite/moss precedent) as a later optimization, not
  part of this plan.
- Prefill: keep eager, decode graphed — repo default policy (see
  `qwen3/pipeline.py` docstring about prefill graph allocator churn).
  Prefill length is ~750 audio embeds + prompt per 30 s window.

Gate: per-step decode logits byte-exact vs golden over a full sample decode.

## Phase 4 — `pipeline.py`

`MegaPipeline` wiring mel → graphed encoder → projector (eager is fine, it's
tiny) → masked_scatter merge into `inputs_embeds` (replicate the reference
`forward` exactly — see the qwen3 pipeline docstring for the pattern) →
`FusedLLMMega.generate` greedy → decode text. Strip/forbid `<think>` content.

Gate: end-to-end transcript identical to eager reference on fixtures, single
and multi-window audio.

## Phase 5 — registration

All the places a model touches (grep any existing slug, e.g. `higgs`, to
catch stragglers):

- `src/starling/server.py`: `MODEL_SLUGS` (line ~102), label map (~111),
  a `AudexBackend(ModelBackend)` class, `BACKENDS` dict (~607), argparse
  description + any per-model gates (fp8_models etc. — audex is not fp8).
  Check whether the adaptive CUDA-graph policy list (ark/qwen3/cohere) should
  include audex — it should, same shape-cache characteristics.
- `benchmarks/bench_all.py`: `MODEL_LABELS`, `--models` help text,
  availability gating like qwen3/higgs.
- `benchmarks/bench_leaderboard.py`: `MODEL_LABELS`.
- `tests/test_audex_pipeline.py` modeled on `tests/test_qwen3_pipeline.py`:
  golden comparisons for encoder, decode steps, end-to-end, multi-utterance
  byte-exactness.
- `README.md`: model list entry (architecture one-liner + **noncommercial
  license note**).

## Phase 6 — benchmarks

- `uv run python benchmarks/bench_all.py --models audex --update-readme`
  (starling vs stock engines).
- `uv run python benchmarks/bench_leaderboard.py --models audex` capped first,
  then full if WER looks sane. ASR-leaderboard WER for Audex-class models
  should be competitive; a wildly high WER means a prompting/template bug
  (thinking mode leaking, wrong normalization), not a kernel bug — check the
  transcript text before blaming numerics.

## Known risks

1. **transformers version drift** in the remote code — mitigations: vendor +
   patch, or isolated venv (higgs precedent). Decide in Phase 0, not later.
2. **Thinking mode leakage** into ASR output — pin the template to instruct
   mode; add a test asserting no `<think>` in output.
3. **fp32 checkpoint** — cast to bf16 once at load; goldens must be produced
   from the *same* bf16 cast, or nothing downstream will match.
4. **Processor availability** — if there's no `AutoProcessor`, `audio.py`
   must reproduce the featurization from `inference_scripts_hf/` exactly
   (window/hop/mel filterbank, padding to 30 s).
5. **`<sound>` placeholder expansion** — the JSON prompt format wraps the
   audio in two `<sound>` markers; verify precisely how the reference expands
   these into `<so_start>/<so_embedding>×750/<so_end>` before writing
   `build_inputs`.
