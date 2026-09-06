# GGUF model files

The native `starling-serve` binary loads GGUF model files. The published
Parakeet files are at
[`scholzmx/parakeet-tdt-0.6b-v3-gguf`](https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf).
See [native serving](native-serving.md) for build and conversion instructions.

## Naming convention

```text
<model-slug>-<quant>.gguf
```

These naming examples are not a download catalog. Sizes are approximate and
vary with conversion settings; check the release files for available names
and sizes. A `bf16-exact` filename does not guarantee identical output on
every backend or input.

| File | Model | Quantization | Size (approx) |
|------|-------|-------------|----------------|
| `parakeet-tdt-0.6b-v3-bf16-exact.gguf` | nvidia/parakeet-tdt-0.6b-v3 | bf16-exact | ~1.2 GB |
| `parakeet-tdt-0.6b-v3-q8_0.gguf` | nvidia/parakeet-tdt-0.6b-v3 | q8_0 | ~0.91 GB |
| `moss-transcribe-preview-2b-bf16-exact.gguf` | OpenMOSS-Team/MOSS-Transcribe-preview-2B | bf16-exact | ~4.5 GB |
| `ark-asr-3b-bf16-exact.gguf` | AutoArk-AI/ARK-ASR-3B | bf16-exact | ~7.0 GB |
| `ark-asr-3b-q8_0.gguf` | AutoArk-AI/ARK-ASR-3B | q8_0 | ~4.3 GB |
| `higgs-audio-v3-bf16-exact.gguf` | bosonai/higgs-audio-v3-stt | bf16-exact | ~5.0 GB |
| `hojo-asr-v1-bf16-exact.gguf` | HojoAI/Hojo-ASR-V1 | bf16-exact | ~11.2 GB |
| `granite-speech-4.1-2b-bf16-exact.gguf` | ibm-granite/granite-speech-4.1-2b | bf16-exact | ~4.8 GB |
| `qwen3-asr-1.7b-bf16-exact.gguf` | Qwen/Qwen3-ASR-1.7B-hf | bf16-exact | ~4.1 GB |
| `audex-2b-bf16-exact.gguf` | nvidia/Nemotron-Labs-Audex-2B | bf16-exact | ~5.8 GB |

## Parakeet quantization results

The results here apply to Parakeet TDT 0.6B v3. On the recorded 300-clip
English and German evaluations, q4_k_m uses 704 MB (28% of F32) and differs
from F32 by at most 0.07 WER percentage points. These samples do not establish
lossless quantization across all languages or recordings.

Q8_0 matched the F32 WER in the recorded clean and noisy English tests.
Q2_K and IQ2_XXS with `shrink16` save more memory but can lose accuracy,
especially outside English. The calibrated Parakeet variants use an
importance matrix collected from 24 of the model's 25 supported languages,
with 48 FLEURS training clips per language. The release includes the matrix
for custom quantization recipes.

See [quantization results](quantization.md) for per-language measurements,
confidence intervals, calibration details, and lower-bit tradeoffs.
Compare a candidate file on your own audio before choosing it. Neither a
quantization label nor a `bf16-exact` filename establishes maximum accuracy.

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

Starling converters write `starling.format_version` and
`starling.numeric_profile` metadata. The Parakeet loader does not validate
these fields. The other model loaders check them when present and reject
unsupported values:

| Model | Accepted `starling.numeric_profile` values |
| --- | --- |
| MOSS, ARK, Higgs | `bf16_exact`, `f16` |
| Hojo | `mixed_f32_bf16_exact`, `bf16_exact`, `f16` |
| Granite, Qwen3, S1 | `bf16_exact` |
| Audex | `bf16_exact`, `quantized` |

For these loaders, `starling.format_version` must be `1` when present.
These checks establish format compatibility, not transcript accuracy or
output parity. Use the fixture and corpus tests described in
[benchmarks](benchmarks.md) to measure output differences.

`starling-serve --abi-version` prints the binary's compiled-in GGML C API ABI
version. It does not inspect a GGUF file.
