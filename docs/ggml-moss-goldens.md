# MOSS staged ggml goldens

This document inventories the golden artefacts for
`OpenMOSS-Team/MOSS-Transcribe-preview-2B`.  The staged files are captured by
[`scripts/moss_golden_components.py`](../scripts/moss_golden_components.py)
using the eager reference helpers in
[`src/starling/moss/reference.py`](../src/starling/moss/reference.py):
`audio_features()` for the stock audio encoder and `build_inputs_embeds()` for
the stock adapter/embed merge. `load_model_and_processor()` explicitly
propagates `attn_implementation="eager"` to the nested audio and language-model
configs; Transformers otherwise silently leaves both on SDPA despite the
loader argument. Decode uses an exact-width `DynamicCache`, not a padded
`StaticCache`: padded attention changes f32 softmax reduction order and flipped
a bf16 near-tie at generated token 21 on medium/long. The component script runs
the same canonical eager prefill as `greedy_generate()` and asserts its complete
greedy output equals `moss_{fixture}_ids.pt`.

All staged floating tensors are CPU `float32`; this is intentional so the C++
stage gates do not depend on PyTorch's bf16 serialization.  The model's actual
processor output is bf16 and is converted to fp32 only while saving: every
saved mel value is therefore an exact fp32 representation of the bf16 value
fed to the eager encoder.  `prompt_ids` is CPU `int64`.

## Existing decode goldens

These are produced by [`scripts/moss_golden.py`](../scripts/moss_golden.py)
from the same reference helpers and are retained unchanged.  They are the
end-to-end greedy-decode gate.

| fixture | `moss_{fixture}_ids.pt` | `moss_{fixture}_text.txt` | role |
|---|---:|---|---|
| short | `(1, 31)` `int64` | decoded greedy transcript | Exact complete token-stream/decode gate |
| medium | `(1, 89)` `int64` | decoded greedy transcript | Exact complete token-stream/decode gate |
| long | `(1, 187)` `int64` | decoded greedy transcript | Exact complete token-stream/decode gate (eager EOS) |

## Staged component goldens

Each fixture has the following six tensor files plus one self-describing JSON
sidecar.  Shapes below are the persisted tensor shapes.

| file suffix | short | medium | long | capture and exactness gate |
|---|---:|---:|---:|---|
| `_mel.pt` | `(128, 743)` `float32` | `(128, 2230)` `float32` | `(128, 7435)` `float32` | BF16-value gate for `cpp/moss/mel`; at most 64 one-ULP boundary bins are accepted (observed 5/13/40). |
| `_encoder_hidden.pt` | `(97, 2048)` `float32` | `(290, 2048)` `float32` | `(967, 2048)` `float32` | Stock eager audio encoder `last_hidden_state`, pre-adapter; max-abs gate `<= 0.02`. |
| `_audio_embeds.pt` | `(97, 2048)` `float32` | `(290, 2048)` `float32` | `(967, 2048)` `float32` | `audio_adapter(encoder_hidden)`; max-abs gate `<= 0.001`. This is exactly the source passed to `masked_scatter_`. |
| `_prompt_ids.pt` | `(1, 107)` `int64` | `(1, 300)` `int64` | `(1, 977)` `int64` | Processor chat-template IDs; exact prompt/layout and audio-slot gate. |
| `_inputs_embeds.pt` | `(1, 107, 2048)` `float32` | `(1, 300, 2048)` `float32` | `(1, 977, 2048)` `float32` | Output of reference `build_inputs_embeds`; gate for token embedding plus audio `masked_scatter_` merge. |
| `_prefill_logits.pt` | `(1, 1, 151936)` `float32` | `(1, 1, 151936)` `float32` | `(1, 1, 151936)` `float32` | `lm_head` logits at the final prefill position; exact argmax plus max-abs `<= 8` sanity gate. Bitwise percentage/top-5 ordering are diagnostic because bf16 ties reorder after ULP accumulation. |
| `_meta.json` | JSON | JSON | JSON | Self-description: model, fixture, seconds, frontend parameters, tensor shapes/dtypes, `audio_data_seqlens`, audio-mask sum, audio-token count, and first generated ID. |

Thus the concrete staged inventory is `moss_{short,medium,long}_{mel,
encoder_hidden,audio_embeds,prompt_ids,inputs_embeds,prefill_logits}.pt` (18
files), plus `moss_{short,medium,long}_meta.json` (3 files): **21 files**.
Together with the six existing decode files above, MOSS has **27** prefixed
artifacts.

### Fixture metadata

| fixture | seconds | mel frames (`audio_data_seqlens`) | audio-mask sum / `n_audio_tokens` | first generated token |
|---|---:|---:|---:|---:|
| short | 7.435 | 743 | 97 | 9157 |
| medium | 22.305 | 2230 | 290 | 9157 |
| long | 74.355 | 7435 | 967 | 11395 |

The observed frontend is 16 kHz, 128 mel bins, `n_fft=640`, and hop length
160.  The non-default-looking `n_fft=640` is from MOSS's vendored
`MelConfig`, not Whisper's usual default.

## Regeneration workflow (canonical)

`golden/` is intentionally `.gitignore`d: this document plus the scripts below
are the source of truth, and any checkout regenerates the artifacts locally.
Run the full sequence in order (each stage depends on the previous output):

1. `uv run scripts/moss_golden.py` — end-to-end decode goldens
   (`moss_{fixture}_ids.pt`, `moss_{fixture}_text.txt`).
2. `uv run scripts/moss_golden_components.py` — the 21 staged component
   artifacts above.
3. `uv run scripts/golden_to_raw.py` — raw F32 dumps consumed by the C++ tests.
4. `uv run scripts/probe_moss_llm_layers_any.py short|medium|long` — per-layer
   LLM stage probes for each fixture.
5. `uv run scripts/probe_moss_llm_layer0_true.py short` — authoritative layer-0
   true-eager-path probe.

All stages use the eager + exact-width `DynamicCache` reference path described
at the top of this document; never regenerate from a padded-`StaticCache` or
SDPA configuration. After regenerating, re-run the gates: `moss_mel_test`,
`moss_encoder_test`, `moss_llm_test`, and `uv run pytest tests/test_moss.py
tests/test_ggml_parity.py`.

## Unprefixed files are not MOSS

The unprefixed `golden/audio_embeds.pt`, `encoder_last_hidden.pt`,
`inputs_embeds.pt`, `llm_prefill_logits.pt`, `projector_out.pt`,
`greedy_ids.pt`, and `greedy_text.txt` belong to the historical
**Granite-Speech-4.1-2b** single downloaded sample fixture.  Their capture API
is [`src/starling/granite/golden.py`](../src/starling/granite/golden.py), which
writes precisely those names.  Their shapes/dtypes are incompatible with every
MOSS staged fixture: respectively `(1,252,2048) bf16`, `(1,1247,1024) bf16`,
`(1,271,2048) bf16`, `(1,1,100353) bf16`, `(1,252,2048) bf16`, and `(1,371)
int64` (the text is its corresponding transcript).  In particular, Granite's
100353-vocabulary logits and 1024-wide encoder state cannot be values from
MOSS, whose staged logits have vocabulary 151936 and encoder state width 2048.
They are retained in place for Granite consumers.

## Mel exactness note

The C++ frontend now mirrors NumPy's spectral-power operation order explicitly:
the double-precision FFT result is rounded to complex64 components, magnitude is
computed with float32 `hypot`, and that float32 magnitude is squared in float32
before the float64 mel-bank accumulation. On the current fixtures this remains
99.9947–99.9954% bitwise equal after the required BF16 conversion: short has
5/95,104 mismatches, medium 13/285,440, and long 40/951,680. Every residual is
exactly one BF16 ULP (`max_abs = 0.00390625`). The operation-order correction is
kept because it is the direct implementation of the NumPy contract. The exact
source of the remaining 58 one-ULP values has not been isolated; they remain
within the frontend's documented timeboxed tolerance, but are not claimed to
be bitwise exact.

## Prompt / Qwen3 / detokenizer acceptance (current)

The prompt merge, one-shot Qwen3 prefill/decode graphs, BF16 KV state, and GGUF
GPT-2/Qwen2 detokenizer are accepted. The merged-embedding gate is bitwise exact
when fed the independent adapter golden. CUDA and CPU produce the canonical
eager token stream; the in-tree C API returns the exact golden text on all three
fixtures (`tests/test_ggml_parity.py`: 9/9 passed).

The LLM follows the explicit eager cast contract:

- BF16 storage at every projection, residual, SiLU/product, probability, and
  RoPE product/add boundary.
- RMSNorm performs F32 reduction, casts normalized values to BF16, then applies
  the BF16 norm weight.
- Q/K per-head normalization precedes transpose/RoPE.
- RoPE uses non-interleaved rotate-half and host F32 cos/sin generation rounded
  to BF16.
- Attention matmul and scalar scaling are rounded to BF16; causal masks are F32;
  softmax input is F32 and probabilities are rounded back to BF16.
- Argmax is performed on the host with strict `>` iteration, preserving the
  lowest vocabulary index on ties.

Trustworthy real-output stage probes (`STARLING_MOSS_L0_STAGE`) match the true
eager layer-0 path bitwise through input norm, Q/K norms, and Q/K RoPE. On the
short fixture the post-o-projection attention is 99.818% bitwise
(`max_abs=0.000976562`), the FFN down projection 99.040%
(`max_abs=0.00390625`), and layer output 99.280%
(`max_abs=0.0078125`). These GEMM-order ULPs compound over 28 layers, so logits
are not a bitwise contract: current prefill bitwise/max-abs is 94.306%/0.5,
77.641%/0.5, and 0%/3.75 for short/medium/long, while argmax, all generated IDs,
and decoded text are exact.

`scripts/probe_moss_llm_layer0_true.py` captures the authoritative real model
path. `scripts/probe_moss_llm_layer0.py` remains a manual operation-by-operation
probe and can differ from the true eager layer output by bf16 ULPs; it must not
replace the true-path oracle.

## Rescue-stage probes

`scripts/moss_stage_probe.py` captures the short/medium/long encoder boundaries
used to diagnose the C++ port. It acquires the shared GPU lock with
`session="ggml-rescue"` and writes raw F32 tensors plus a shape JSON under
`golden/raw/`: `conv1_raw`, `conv2_raw`, `conv3_raw`, `conv3_flat`,
`conv_out_padded`, `post_conv`, `encL0`, `encL31`, `ln_post`, and
`encoder_hidden`.

The probe exposed two independent layout defects in the initial conversion and
encoder graph. The Conv output must be permuted from ggml `[time,freq,channel,
chunk]` to contiguous `[freq,channel,time,chunk]`, expressed by ggml's
*destination-axis* permutation API as `ggml_permute(x, 2, 0, 1, 3)`. Also,
`enc.positional_embedding` must be passed to `GGUFWriter` as the native
`[position,hidden]` NumPy array. Passing `pos.T` made the loaded ggml tensor
encode `sin(position)` down the hidden dimension (the first row began
`0,sin(1),sin(2),...`) instead of the required all-zero position-0 row.
