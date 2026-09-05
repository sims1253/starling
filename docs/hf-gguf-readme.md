# Starling GGUF Model Files

Pre-converted GGUF files for the native `starling-serve` binary. One repo per
model, mirroring the `models/` directory layout in the starling repository.

## Naming convention

```text
<model-slug>-<quant>.gguf
```

| File | Model | Quantization | Size (approx) |
|------|-------|-------------|----------------|
| `parakeet-tdt-0.6b-v3-bf16-exact.gguf` | nvidia/parakeet-tdt-0.6b-v3 | bf16-exact | ~1.2 GB |
| `parakeet-tdt-0.6b-v3-q8_0.gguf` | nvidia/parakeet-tdt-0.6b-v3 | q8_0 | ~0.6 GB |
| `moss-transcribe-preview-2b-bf16-exact.gguf` | OpenMOSS-Team/MOSS-Transcribe-preview-2B | bf16-exact | ~4.5 GB |
| `ark-asr-3b-bf16-exact.gguf` | AutoArk-AI/ARK-ASR-3B | bf16-exact | ~7.0 GB |
| `ark-asr-3b-q8_0.gguf` | AutoArk-AI/ARK-ASR-3B | q8_0 | ~4.3 GB |
| `higgs-audio-v3-bf16-exact.gguf` | bosonai/higgs-audio-v3-stt | bf16-exact | ~5.0 GB |
| `hojo-asr-v1-bf16-exact.gguf` | HojoAI/Hojo-ASR-V1 | bf16-exact | ~11.2 GB |
| `granite-speech-4.1-2b-bf16-exact.gguf` | ibm-granite/granite-speech-4.1-2b | bf16-exact | ~4.8 GB |
| `qwen3-asr-1.7b-bf16-exact.gguf` | Qwen/Qwen3-ASR-1.7B-hf | bf16-exact | ~4.1 GB |
| `audex-2b-bf16-exact.gguf` | nvidia/Nemotron-Labs-Audex-2B | bf16-exact | ~5.8 GB |

## Quantization strategy

- **q4_k_m** (calibrated, recommended): no measurable WER loss vs f32 on
  clean/noisy English, 300-clip English/German, and 48-clip-per-language
  evaluations across the other languages — at 28% of the f32 size.
- **q8_0**: conservative default for byte-parity worriers; also lossless in
  every measurement.
- **q2_k / iq2_xxs+shrink16**: size-constrained tiers; see
  `docs/quantization.md` for the measured multilingual trade-offs
  (iq2_xxs is an English-first trade, q2_k is the even option).
- All calibrated quants share one importance matrix collected over 24 of
  the 25 supported languages (FLEURS train, 48 clips/language); the matrix
  itself ships in the repo for downstream re-quantization and recipes.
- **bf16-exact**: Byte-exact parity with the Python reference. For users with
  VRAM to spare who want maximum accuracy.

Released at [`scholzmx/parakeet-tdt-0.6b-v3-gguf`](https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf)
(move to the `starling` org once it exists).

## Usage with starling-serve

```bash
# Download a GGUF file (the calibrated quant ladder lives at
# scholzmx/parakeet-tdt-0.6b-v3-gguf — q8_0, q6_k, q5_k_m, q4_k_m,
# q4_k_s, q4_k_m-shrink16, q3_k_m, q2_k, iq2_xxs-imx-shrink16, plus the
# importance matrix for custom recipes):
hf download scholzmx/parakeet-tdt-0.6b-v3-gguf \
  parakeet-tdt-0.6b-v3-q8_0.gguf \
  --local-dir ./models

# Serve it:
starling-serve --model parakeet --gguf ./models/parakeet-tdt-0.6b-v3-q8_0.gguf --port 8181
```

## Verification

Every converted file records `starling.format_version` (currently `1`) and
`starling.numeric_profile` metadata. On load, `starling-serve` validates these
fields when they are present:

- `starling.numeric_profile` — accepted values: `bf16_exact`, `f16`
  (parakeet, moss, ark, higgs); `mixed_f32_bf16_exact`, `bf16_exact`, `f16`
  (hojo); `bf16_exact` (granite, qwen3, audex).
- `starling.format_version` — must be `1`.

The parakeet loader does not validate these fields; the other model loaders
reject unsupported values at load time with a clear error.

`starling-serve --abi-version` prints the binary's compiled-in GGML C-API ABI
version. It describes the binary only — it does not inspect the GGUF file.
