# Benchmarks


These scripts support `--update-readme` to refresh the tables on this page.

- **`benchmarks/bench_all.py`**: latency/RTFx grid. Sweeps model × engine ×
  audio length × batch size on tiled-LibriSpeech fixtures. Engines: `starling`,
  `stock`, `crispasr`, `parakeet.cpp`, `starling-ggml`, `starling-batched`
  (granite/qwen3), `starling-spec` (granite). s1 runs here too (`--models s1`):
  fixture tiers select transcripts, RTFx reads normalized words/s.
- **`benchmarks/bench_leaderboard.py`**: accuracy grid. Reproduces the
  [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
  English short-form methodology (Whisper `EnglishTextNormalizer` +
  `kaldialign` WER, `merge_compounds=True`, unweighted mean across 7 datasets).
  Use WER to check optimizations that change numerical results.
- **`benchmarks/s1/bench_normalize.py`**: s1-mini's dedicated suite:
  latency/throughput per engine, byte-exact parity vs stock, curated quality
  cases, and the full 16-combination control matrix (styling × structure ×
  context). Splices the `BENCH:S1` block below.

```
uv run python benchmarks/bench_all.py --update-readme
uv run python benchmarks/bench_leaderboard.py                  # capped, fast
uv run python benchmarks/bench_leaderboard.py --num-samples 0  # full splits
```

RTFx = audio_seconds / transcribe_seconds (higher is faster). bf16, model load
excluded, single RTX 5090.

### Synthetic fixture latency / RTFx

These tiled single-utterance numbers are a deterministic regression gate, not
a representative workload distribution. Use the real-corpus leaderboard RTFx
table below for headline cross-model throughput. Newly generated tables report
median ± standard deviation across repetitions.

<!-- BENCH:START -->
**granite-speech-4.1-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling       | stock transformers   |
|----------|---------|----------------|----------------------|
| short    |       1 | 171±2ms (44x)  | 2657±357ms (3x)      |
| medium   |       1 | 324±2ms (69x)  | 4935±749ms (4x)      |
| long     |       1 | 1237±2ms (60x) | 16402±1377ms (4x)    |

**parakeet-tdt-0.6b-v3** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling       | stock transformers   | starling-ggml   |
|----------|---------|----------------|----------------------|-----------------|
| short    |       1 | 16±2ms (452x)  | 163±40ms (46x)       | 14±0ms (550x)   |
| short    |       8 | 27±1ms (279x)  | —                    | —               |
| medium   |       1 | 24±1ms (936x)  | 472±80ms (47x)       | 30±1ms (753x)   |
| medium   |       8 | 61±1ms (366x)  | —                    | —               |
| long     |       1 | 58±1ms (1279x) | 1411±273ms (53x)     | 86±2ms (861x)   |
| long     |       8 | 181±2ms (410x) | —                    | —               |

**moss-transcribe-preview-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling        | stock transformers   | starling-ggml   |
|----------|---------|-----------------|----------------------|-----------------|
| short    |       1 | 166±9ms (45x)   | 1774±389ms (4x)      | 214±4ms (35x)   |
| medium   |       1 | 397±19ms (56x)  | 5630±851ms (4x)      | 535±3ms (42x)   |
| long     |       1 | 1499±50ms (50x) | 12307±1126ms (6x)    | 1180±10ms (63x) |

**ark-asr-3b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling       | stock transformers   | starling-ggml   |
|----------|---------|----------------|----------------------|-----------------|
| short    |       1 | 217±8ms (34x)  | 2367±353ms (3x)      | 310±20ms (24x)  |
| medium   |       1 | 649±60ms (34x) | 6531±659ms (3x)      | 836±26ms (27x)  |
| long     |       1 | 703±9ms (106x) | 6467±433ms (12x)     | 941±40ms (79x)  |

**cohere-transcribe-03-2026** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling       | stock transformers   |
|----------|---------|----------------|----------------------|
| short    |       1 | 61±8ms (122x)  | 462±108ms (16x)      |
| medium   |       1 | 164±2ms (136x) | 1279±205ms (17x)     |
| long     |       1 | 334±5ms (222x) | 1641±379ms (45x)     |

**nemotron-labs-audex-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling       | stock transformers   |
|----------|---------|----------------|----------------------|
| short    |       1 | 244±1ms (30x)  | 1515±269ms (5x)      |
| medium   |       1 | 464±1ms (48x)  | 3393±488ms (7x)      |
| long     |       1 | 1656±5ms (45x) | 13051±1354ms (6x)    |

**qwen3-asr-1.7b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling        | stock transformers   |
|----------|---------|-----------------|----------------------|
| short    |       1 | 192±36ms (39x)  | 2037±408ms (4x)      |
| medium   |       1 | 380±35ms (59x)  | 5292±886ms (4x)      |
| long     |       1 | 1156±17ms (64x) | 18788±1454ms (4x)    |
<!-- BENCH:END -->

**granite-speech-4.1-2b-nar**: latency / RTFx (ms, RTFx×)

| length | batch | starling | stock transformers |
|--------|------:|----------|--------------------|
| short  |     1 | 14ms (531x) | 75ms (99x) |
| medium |     1 | 30ms (744x) | 95ms (235x) |
| long   |     1 | 104ms (715x) | 139ms (535x) |

### Accuracy (Open ASR Leaderboard reproduction)

<!-- BENCH:WER:START -->
**Open ASR Leaderboard — WER %** (per dataset, unweighted mean avg)

| model                      | engine             | voxpopuli   | ami    | earnings22   | gigaspeech   | librispeech_clean   | librispeech_other   | spgispeech   | avg   |
|----------------------------|--------------------|-------------|--------|--------------|--------------|---------------------|---------------------|--------------|-------|
| granite-speech-4.1-2b      | starling           | 7.47%       | 8.02%  | 8.48%        | 5.21%        | 1.77%               | 2.35%               | 2.80%        | 5.16% |
| granite-speech-4.1-2b      | stock transformers | 7.47%       | 8.02%  | 8.44%        | 5.13%        | 1.77%               | 2.25%               | 2.90%        | 5.14% |
| parakeet-tdt-0.6b-v3       | starling           | 6.35%       | 7.21%  | 7.71%        | 4.36%        | 1.71%               | 3.28%               | 3.56%        | 4.88% |
| parakeet-tdt-0.6b-v3       | stock transformers | 6.28%       | 7.21%  | 7.71%        | 4.36%        | 1.68%               | 3.31%               | 3.56%        | 4.87% |
| parakeet-tdt-0.6b-v3       | starling-ggml      | 7.22%       | 8.30%  | 8.41%        | 5.55%        | 1.83%               | 3.60%               | 4.01%        | 5.56% |
| moss-transcribe-preview-2b | starling           | 3.81%       | 6.31%  | 6.72%        | 4.24%        | 1.62%               | 2.66%               | 2.15%        | 3.93% |
| moss-transcribe-preview-2b | stock transformers | 3.81%       | 6.17%  | 6.68%        | 4.28%        | 1.62%               | 2.66%               | 2.10%        | 3.90% |
| moss-transcribe-preview-2b | starling-ggml      | 3.81%       | 6.21%  | 6.75%        | 4.32%        | 1.56%               | 2.56%               | 2.00%        | 3.89% |
| qwen3-asr-1.7b             | starling           | 6.91%       | 7.31%  | 8.19%        | 4.07%        | 1.80%               | 2.88%               | 2.80%        | 4.85% |
| qwen3-asr-1.7b             | stock transformers | 6.94%       | 7.45%  | 8.30%        | 3.98%        | 1.80%               | 2.91%               | 2.75%        | 4.88% |
| ark-asr-3b                 | starling           | 11.35%      | 6.31%  | 8.04%        | 3.77%        | 2.60%               | 3.97%               | 2.35%        | 5.48% |
| ark-asr-3b                 | stock transformers | 11.38%      | 6.21%  | 8.15%        | 3.77%        | 2.63%               | 3.81%               | 2.25%        | 5.46% |
| ark-asr-3b                 | starling-ggml      | 11.53%      | 6.45%  | 8.26%        | 3.86%        | 2.60%               | 4.00%               | 2.35%        | 5.58% |
| cohere-transcribe-03-2026  | starling           | 10.32%      | 6.31%  | 8.59%        | 5.47%        | 1.47%               | 1.78%               | 2.45%        | 5.20% |
| cohere-transcribe-03-2026  | stock transformers | 10.28%      | 6.31%  | 8.59%        | 5.51%        | 1.47%               | 1.81%               | 2.45%        | 5.20% |
| nemotron-labs-audex-2b     | starling           | 9.80%       | 11.90% | 6.09%        | 4.14%        | 1.57%               | 2.01%               | 2.65%        | 5.45% |
| nemotron-labs-audex-2b     | stock transformers | 9.80%       | 11.90% | 6.09%        | 4.14%        | 1.57%               | 2.01%               | 2.65%        | 5.45% |

**Open ASR Leaderboard — RTFx** (real audio_s / inference_s)

| model                      | engine             | voxpopuli   | ami   | earnings22   | gigaspeech   | librispeech_clean   | librispeech_other   | spgispeech   |
|----------------------------|--------------------|-------------|-------|--------------|--------------|---------------------|---------------------|--------------|
| granite-speech-4.1-2b      | starling           | 78x         | 74x   | 78x          | 64x          | 69x                 | 63x                 | 66x          |
| granite-speech-4.1-2b      | stock transformers | 5x          | 5x    | 5x           | 4x           | 5x                  | 4x                  | 5x           |
| parakeet-tdt-0.6b-v3       | starling           | 600x        | 533x  | 1083x        | 841x         | 1104x               | 998x                | 833x         |
| parakeet-tdt-0.6b-v3       | stock transformers | 54x         | 54x   | 66x          | 48x          | 56x                 | 52x                 | 48x          |
| parakeet-tdt-0.6b-v3       | starling-ggml      | 260x        | 178x  | 265x         | 234x         | 301x                | 266x                | 682x         |
| moss-transcribe-preview-2b | starling           | 64x         | 54x   | 63x          | 51x          | 65x                 | 58x                 | 53x          |
| moss-transcribe-preview-2b | stock transformers | 6x          | 6x    | 6x           | 5x           | 6x                  | 5x                  | 5x           |
| moss-transcribe-preview-2b | starling-ggml      | 52x         | 43x   | 47x          | 40x          | 47x                 | 42x                 | 49x          |
| qwen3-asr-1.7b             | starling           | 55x         | 50x   | 65x          | 48x          | 59x                 | 53x                 | 57x          |
| qwen3-asr-1.7b             | stock transformers | 6x          | 5x    | 6x           | 4x           | 5x                  | 4x                  | 5x           |
| ark-asr-3b                 | starling           | 53x         | 46x   | 46x          | 40x          | 50x                 | 47x                 | 42x          |
| ark-asr-3b                 | stock transformers | 7x          | 6x    | 6x           | 5x           | 6x                  | 6x                  | 5x           |
| ark-asr-3b                 | starling-ggml      | 12x         | 12x   | 15x          | 14x          | 17x                 | 16x                 | 30x          |
| cohere-transcribe-03-2026  | starling           | 97x         | 83x   | 102x         | 75x          | 110x                | 97x                 | 82x          |
| cohere-transcribe-03-2026  | stock transformers | 32x         | 29x   | 36x          | 29x          | 29x                 | 26x                 | 26x          |
| nemotron-labs-audex-2b     | starling           | 60x         | 57x   | 66x          | 49x          | 60x                 | 55x                 | 48x          |
| nemotron-labs-audex-2b     | stock transformers | 9x          | 8x    | 10x          | 7x           | 9x                  | 8x                  | 8x           |
<!-- BENCH:WER:END -->

<!-- BENCH:S1:START -->
**s1-mini** — normalization latency / throughput (ms, words/s)

Text-in/text-out: fixture tiers are raw transcripts;
words/s = input words normalized per second (higher is faster).
bf16, model load + graph capture excluded, single RTX 5090.

| tier   | engine             | ms           |   words/s |
|--------|--------------------|--------------|-----------|
| short  | starling           | 120±5ms      |       149 |
| short  | stock transformers | 533±15ms     |        34 |
| short  | starling-ggml      | 102±16ms     |       176 |
| medium | starling           | 239±28ms     |       318 |
| medium | stock transformers | 4065±388ms   |        19 |
| medium | starling-ggml      | 411±8ms      |       185 |
| long   | starling           | 595±47ms     |       417 |
| long   | stock transformers | 17017±1376ms |        15 |
| long   | starling-ggml      | 1253±22ms    |       198 |

**s1-mini** — accuracy (vs stock transformers greedy)

| tier   | starling   | starling-ggml   |
|--------|------------|-----------------|
| short  | byte-exact | byte-exact      |
| medium | byte-exact | byte-exact      |
| long   | byte-exact | byte-exact      |

Control matrix (4 styling x 2 structure x 2 context): starling_exact: 15/16, starling-ggml_exact: 15/16
Curated quality cases: 5/5 expected outputs matched
<!-- BENCH:S1:END -->

The leaderboard results use 50 clips per dataset. Parakeet, ARK, and Qwen3
bucket mel lengths to share encoder graphs and run prompt prefill eagerly.
Their decode loops stay graphed. This limits graph memory use when clips have
many different lengths. Fixture parity does not establish corpus-wide parity;
the WER table records differences between engines.

## What did not work

- **INT8 weight-only quant** is slower: decode is launch-bound, not bandwidth-bound.
- **FP8 `_scaled_mm`** is slower for M=1 decode and proved unsafe across many
  captured graphs. The shipped FP8 path uses a fused weight-only Triton GEMV.
- **`torch.compile` on the encoder** is not byte-exact: inductor upcasts attention to fp32 and the conformer's BatchNorm amplifies the difference.
- **Batched spec decoding at B≥16** is slower than non-spec (0.76x): lock-step cache rewind wastes verify work when streams differ in acceptance.

## Corpus and optional engines

The leaderboard uses the `hf-audio/open-asr-leaderboard` dataset. Set `HF_TOKEN`
if downloads are rate-limited. Clips cache under
`tests/fixtures/leaderboard_corpus/`. The external CrispASR and parakeet.cpp
engines use a sibling `~/asr-bench` checkout; the harness skips them if absent.

## Quantization

The native engines support GGUF block quantization with importance matrices
collected from audio. Use `build/starling-quantize` to quantize weights,
`benchmarks/imatrix_collect.py` to collect activation importance, and
`benchmarks/wer_quant.py` to compare WER against F32.

On the recorded 300-clip English and German Parakeet evaluations, q4_k_m uses
704 MB (28% of F32) and differs from F32 by at most 0.07 WER percentage points.
Lower bit widths save memory but can lose accuracy. See
[quantization results](quantization.md) for calibration, per-language results,
and confidence intervals.
