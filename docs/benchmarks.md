# Benchmarks

Starling measures the native engine (`libstarling_ggml`, `starling-serve`)
directly. The tools:

- **`starling-bench`** (built with the native tree, `cpp/tools/starling_bench.cpp`):
  transcribes WAV files through the C API and prints wall time, RTFx and the
  transcript per run. `STARLING_ENGINE=ggml|fast` picks the engine family, so
  one build compares the ggml and fast engines on the same files.
- **`benchmarks/experiments/`** (issue #168): the shared
  baseline-versus-candidate experiment record and comparison command.
  Preregisters the objective, workload pin, protocol, and acceptance rules
  in a sealed spec; runs both arms interleaved in fresh `starling-serve`
  processes with cold/warm separation; compares under a deterministic paired
  bootstrap and refuses invalid pairs (changed corpus, mismatched metric
  identities, models, or hardware). Inconclusive results cannot be promoted
  as wins. See [experiments/README.md](../benchmarks/experiments/README.md).
- **`benchmarks/fast_engine/wer_engines.py`** with `export_fleurs.py`: corpus
  WER, total transcription time and transcript diffs for engine variants of
  one library (`--lib build/libstarling_ggml.so`). The scorer and normalizer
  are `benchmarks/fast_engine/wer.py`.
- **`benchmarks/fast_engine/android_bench.sh`** and the phone scripts next to
  it: cross-build `starling-bench`, push it to a phone with the models, and
  compare engines there ([fast engines](fast-engine.md)).
- **`tests/test_ggml_parity.py`**: exactness against the reference goldens
  (see [engine development](ggml-engine.md#correctness-contract)).

```bash
uv sync --locked --group eval                       # corpus export + WER scorer
cmake --build build -j --target starling-bench
build/starling-bench --model parakeet --gguf models/parakeet-tdt-0.6b-v3-bf16-exact.gguf \
  --warmup tests/fixtures/medium.wav
```

RTFx = audio_seconds / transcribe_seconds (higher is faster).

## Per-model speed check in the app (#475)

`starling-bench` answers "how fast is this model on this machine?" on the
command line. Settings → Engine asks the same question from inside the app:
**Speed check** on any installed model measures it offline, in about the
time the model needs to load.

- Two probes, both local. The *synthetic tone* is generated on the spot (a
  modulated tone, the same speech stand-in `tests/fidelity_audio.py`
  builds) and is a smoke test: silence and short outputs decode fast, so its
  number is an **upper bound**, never real-speech speed. The *speech clip*
  is the bundled LibriSpeech fixture and is the representative figure.
- The engine backend is named with every result, so a comparison never
  silently mixes Vulkan and CPU.
- Read-only. A check spawns and stops its own sidecar; it never touches the
  active engine, an in-flight take, history audio, or the insight-event
  counts. Its transcript is discarded — this is a throughput probe, not a
  quality check.
- Seconds per minute of audio is `60 / RTFx`, the way a user asks the
  question. Unmeasurable probes report unknown, never zero.
- Warm before measured. The check drives its sidecar to warm first, so a
  model's load time is excluded from the reported figure — which is what
  makes two models comparable on one machine. A cold model is therefore
  slower to answer the *first* time than the number implies, and faster
  the second; compare runs of the same model by their later numbers.

Related: the energy-measurement spike (#476) decides whether an "energy per
minute" card can be answered honestly; wall time and peak RSS
(`starling-bench`) already are.

## Experiment records and comparisons

A candidate may only be called a win when the paired-improvement CI clears
the preregistered bar; inconclusive results cannot be promoted, failed runs
are refused with diagnostics, and records with different corpus hashes,
metric identities, normalizers, model/config, or hardware claims are rejected
as incomparable. The one-command paired CPU reproduction runs against the
contract-fixture binary with no models:

```bash
cmake -B build-exp -DSTARLING_SERVE=ON -DSTARLING_GGML_TESTS=ON \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF
cmake --build build-exp -j --target starling-serve-contract-fixture
python benchmarks/experiments/run_experiment.py demo \
  --binary build-exp/starling-serve-contract-fixture --out-dir build-exp/experiments/demo
```

Identical arms never compare `pass` (normally **inconclusive**): the
no-false-win contract demonstrated end to end.

## Recorded measurements

The tables below were the last ones recorded with the removed Python harness
(`benchmarks/bench_all.py`, `bench_leaderboard.py` and
`s1/bench_normalize.py` at tag `python-research-final`) on a single RTX 5090,
bf16, model load excluded. They keep the native `starling-ggml` engine and
the stock transformers reference; the rows of the removed Python/CUDA
pipeline are at the tag.

### Synthetic fixture latency / RTFx

Tiled single-utterance fixtures, batch 1, median ± standard deviation. A
deterministic regression gate, not a representative workload.

| model | length | stock transformers | starling-ggml |
| --- | --- | --- | --- |
| parakeet-tdt-0.6b-v3 | short | 163±40ms (46x) | 14±0ms (550x) |
| parakeet-tdt-0.6b-v3 | medium | 472±80ms (47x) | 30±1ms (753x) |
| parakeet-tdt-0.6b-v3 | long | 1411±273ms (53x) | 86±2ms (861x) |
| moss-transcribe-preview-2b | short | 1774±389ms (4x) | 214±4ms (35x) |
| moss-transcribe-preview-2b | medium | 5630±851ms (4x) | 535±3ms (42x) |
| moss-transcribe-preview-2b | long | 12307±1126ms (6x) | 1180±10ms (63x) |
| ark-asr-3b | short | 2367±353ms (3x) | 310±20ms (24x) |
| ark-asr-3b | medium | 6531±659ms (3x) | 836±26ms (27x) |
| ark-asr-3b | long | 6467±433ms (12x) | 941±40ms (79x) |

### Accuracy (Open ASR Leaderboard reproduction)

English short-form methodology of the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
(Whisper `EnglishTextNormalizer` + `kaldialign` WER, unweighted mean across 7
datasets), 50 clips per dataset. Fixture parity does not establish corpus-wide
parity; the table records the differences.

| model | engine | voxpopuli | ami | earnings22 | gigaspeech | librispeech_clean | librispeech_other | spgispeech | avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parakeet-tdt-0.6b-v3 | stock transformers | 6.28% | 7.21% | 7.71% | 4.36% | 1.68% | 3.31% | 3.56% | 4.87% |
| parakeet-tdt-0.6b-v3 | starling-ggml | 7.22% | 8.30% | 8.41% | 5.55% | 1.83% | 3.60% | 4.01% | 5.56% |
| moss-transcribe-preview-2b | stock transformers | 3.81% | 6.17% | 6.68% | 4.28% | 1.62% | 2.66% | 2.10% | 3.90% |
| moss-transcribe-preview-2b | starling-ggml | 3.81% | 6.21% | 6.75% | 4.32% | 1.56% | 2.56% | 2.00% | 3.89% |
| ark-asr-3b | stock transformers | 11.38% | 6.21% | 8.15% | 3.77% | 2.63% | 3.81% | 2.25% | 5.46% |
| ark-asr-3b | starling-ggml | 11.53% | 6.45% | 8.26% | 3.86% | 2.60% | 4.00% | 2.35% | 5.58% |

Real-audio RTFx on the same clips:

| model | engine | voxpopuli | ami | earnings22 | gigaspeech | librispeech_clean | librispeech_other | spgispeech |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parakeet-tdt-0.6b-v3 | stock transformers | 54x | 54x | 66x | 48x | 56x | 52x | 48x |
| parakeet-tdt-0.6b-v3 | starling-ggml | 260x | 178x | 265x | 234x | 301x | 266x | 682x |
| moss-transcribe-preview-2b | stock transformers | 6x | 6x | 6x | 5x | 6x | 5x | 5x |
| moss-transcribe-preview-2b | starling-ggml | 52x | 43x | 47x | 40x | 47x | 42x | 49x |
| ark-asr-3b | stock transformers | 7x | 6x | 6x | 5x | 6x | 6x | 5x |
| ark-asr-3b | starling-ggml | 12x | 12x | 15x | 14x | 17x | 16x | 30x |

### s1-mini normalization

Text in, text out: fixture tiers are raw transcripts; words/s = input words
normalized per second. Model load and graph capture excluded.

| tier | engine | ms | words/s | vs stock greedy |
| --- | --- | --- | --- | --- |
| short | stock transformers | 533±15ms | 34 | reference |
| short | starling-ggml | 102±16ms | 176 | byte-exact |
| medium | stock transformers | 4065±388ms | 19 | reference |
| medium | starling-ggml | 411±8ms | 185 | byte-exact |
| long | stock transformers | 17017±1376ms | 15 | reference |
| long | starling-ggml | 1253±22ms | 198 | byte-exact |

Control matrix (4 styling x 2 structure x 2 context): 15/16 cells exact
against stock greedy (the remaining cell is an exact bf16 tie, see
[engine development](ggml-engine.md)). Curated quality cases: 5/5 matched.

## Quantization

The native engines support GGUF block quantization with importance matrices
collected from audio. Use `build/starling-quantize` to quantize weights,
`benchmarks/imatrix_collect.py` to collect activation importance, and
`benchmarks/fast_engine/wer_engines.py` to score each GGUF on a corpus.

On the recorded 300-clip English and German Parakeet evaluations, q4_k_m uses
704 MB (28% of F32) and differs from F32 by at most 0.07 WER percentage points.
Lower bit widths save memory but can lose accuracy. See
[quantization results](quantization.md) for calibration, per-language results,
and confidence intervals.
