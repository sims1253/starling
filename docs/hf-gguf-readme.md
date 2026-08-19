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

## Quantization strategy

- **q8_0** (recommended default): Halves VRAM with negligible WER delta.
- **bf16-exact**: Byte-exact parity with the Python reference. For users with
  VRAM to spare who want maximum accuracy.

## Usage with starling-serve

```bash
# Download a GGUF file:
hf download starling/parakeet-tdt-0.6b-v3-gguf \
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
  (hojo); `bf16_exact` (granite, qwen3).
- `starling.format_version` — must be `1`.

The parakeet loader does not validate these fields; the other model loaders
reject unsupported values at load time with a clear error.

`starling-serve --abi-version` prints the binary's compiled-in GGML C-API ABI
version. It describes the binary only — it does not inspect the GGUF file.
