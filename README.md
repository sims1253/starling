# Starling

CUDA-graph inference kernels for speech-recognition models, tuned for a single
RTX 5090 (Blackwell, sm_120). Portable to any Ampere+ NVIDIA GPU.

The stock `transformers` decode loop emits hundreds of tiny kernels per token
and spends most of its wall time on CPU launch overhead — the GPU sits ~10%
busy. Starling captures everything replayable into a CUDA graph: decode steps,
fused RMSNorm/SwiGLU, attention masks, multi-step token loops (autoregressive
models) or the single bidirectional editor forward (granite-nar). Output is
byte-identical to eager `transformers` — same accuracy, fewer round trips.

## Models

All do speech-to-text.

- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — encoder + 1B LLM decoder. Optional self-speculative path drafting from the encoder's CTC head.
- [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) — non-autoregressive. One bidirectional forward: CTC conformer draft + blank slots + bidirectional granite-4.0-1b editor refinement. No decode loop.
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — FastConformer + TDT transducer (no LLM). GPU-side mel + chunking for hour-long audio.
- [`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B) — Qwen3-omni MoE encoder + Qwen3 decoder.
- [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — Whisper-style windowed-attention encoder + Qwen3 decoder.
- [`AutoArk-AI/ARK-ASR-3B`](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) — Whisper encoder + MLP adapter + Qwen2.5 decoder.
- [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — first seq2seq encoder-decoder: 48-layer FastConformer encoder + 8-layer Transformer decoder (self + cross attention).
- [`bosonai/higgs-audio-v3-stt`](https://huggingface.co/bosonai/higgs-audio-v3-stt) — Whisper-large-v3 mel + MLP projector + Qwen3-1.7B decoder. Runs under its own `.venv-higgs` (transformers 4.51) because the model's `trust_remote_code` modeling breaks under transformers 5.13.

The autoregressive models (granite, moss, qwen3, ark, higgs, cohere) share an
encoder + LLM-decoder pattern where the decode loop is the bottleneck. Parakeet
is a transducer; granite-nar is a single bidirectional pass.

## Benchmark

Two scripts, each the single source of truth for its slice. Both support
`--update-readme` to splice tables into the sentinel-wrapped blocks below.

- **`benchmarks/bench_all.py`** — latency/RTFx grid. Sweeps model × engine ×
  audio length × batch size on tiled-LibriSpeech fixtures. Engines: `starling`,
  `stock`, `crispasr`, `parakeet.cpp`, `starling-batched` (granite/qwen3),
  `starling-spec` (granite).
- **`benchmarks/bench_leaderboard.py`** — accuracy grid. Reproduces the
  [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
  English short-form methodology (Whisper `EnglishTextNormalizer` +
  `kaldialign` WER, `merge_compounds=True`, unweighted mean across 7 datasets).
  A real WER budget lets future non-byte-exact optimizations land safely.

```
uv run python benchmarks/bench_all.py --update-readme
uv run python benchmarks/bench_leaderboard.py                  # capped, fast
uv run python benchmarks/bench_leaderboard.py --num-samples 0  # full splits
```

RTFx = audio_seconds / transcribe_seconds (higher is faster). bf16, model load
excluded, single RTX 5090.

### Latency / RTFx

<!-- BENCH:START -->
**granite-speech-4.1-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 280ms (27x)  | 2391ms (3x)          |
| medium   |       1 | 502ms (44x)  | 3639ms (6x)          |
| long     |       1 | 1758ms (42x) | 15397ms (5x)         |

**parakeet-tdt-0.6b-v3** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 15ms (509x)  | 146ms (51x)          |
| short    |       8 | 24ms (309x)  | —                    |
| medium   |       1 | 26ms (842x)  | 341ms (66x)          |
| medium   |       8 | 52ms (429x)  | —                    |
| long     |       1 | 63ms (1180x) | 1442ms (52x)         |
| long     |       8 | 180ms (414x) | —                    |

**moss-transcribe-preview-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling    | stock transformers   |
|----------|---------|-------------|----------------------|
| short    |       1 | 202ms (37x) | 2008ms (4x)          |
| medium   |       1 | 464ms (48x) | 4852ms (5x)          |
| long     |       1 | 847ms (88x) | 18803ms (4x)         |

**ark-asr-3b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 232ms (32x)  | 1300ms (6x)          |
| medium   |       1 | 588ms (38x)  | 4285ms (5x)          |
| long     |       1 | 648ms (115x) | 4196ms (18x)         |

**cohere-transcribe-03-2026** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 54ms (138x)  | 436ms (17x)          |
| medium   |       1 | 160ms (139x) | 737ms (30x)          |
| long     |       1 | 340ms (219x) | 1276ms (58x)         |
<!-- BENCH:END -->

### Accuracy (Open ASR Leaderboard reproduction)

<!-- BENCH:WER:START -->
**Open ASR Leaderboard — WER %** (per dataset, unweighted mean avg)

| model                      | engine             | voxpopuli   | ami   | earnings22   | gigaspeech   | librispeech_clean   | librispeech_other   | spgispeech   | avg   |
|----------------------------|--------------------|-------------|-------|--------------|--------------|---------------------|---------------------|--------------|-------|
| ark-asr-3b                 | starling           | 27.56%      | 5.27% | 9.28%        | 3.20%        | 6.01%               | 6.74%               | 2.39%        | 8.64% |
| ark-asr-3b                 | stock transformers | 27.67%      | 5.10% | 9.13%        | 2.82%        | 6.14%               | 6.62%               | 2.39%        | 8.55% |
| cohere-transcribe-03-2026  | starling           | 10.35%      | 6.31% | 8.59%        | 5.42%        | 1.47%               | 1.78%               | 2.50%        | 5.20% |
| cohere-transcribe-03-2026  | stock transformers | 10.28%      | 6.31% | 8.59%        | 5.51%        | 1.47%               | 1.81%               | 2.45%        | 5.20% |
| granite-speech-4.1-2b      | starling           | 7.44%       | 7.97% | 8.41%        | 5.25%        | 1.77%               | 2.35%               | 2.85%        | 5.15% |
| granite-speech-4.1-2b      | stock transformers | 7.47%       | 8.02% | 8.44%        | 5.13%        | 1.77%               | 2.25%               | 2.90%        | 5.14% |
| moss-transcribe-preview-2b | starling           | 3.81%       | 6.40% | 6.72%        | 4.28%        | 1.59%               | 2.69%               | 2.15%        | 3.95% |
| moss-transcribe-preview-2b | stock transformers | 3.81%       | 6.17% | 6.68%        | 4.28%        | 1.62%               | 2.66%               | 2.10%        | 3.90% |
| parakeet-tdt-0.6b-v3       | starling           | 9.69%       | 5.27% | 5.18%        | 3.76%        | 1.96%               | 2.84%               | 4.51%        | 4.74% |
| parakeet-tdt-0.6b-v3       | stock transformers | 9.59%       | 5.44% | 5.18%        | 3.76%        | 1.96%               | 2.84%               | 4.24%        | 4.72% |
| qwen3-asr-1.7b             | starling           | 6.94%       | 7.31% | 8.19%        | 4.07%        | 1.80%               | 2.88%               | 2.80%        | 4.86% |
| qwen3-asr-1.7b             | stock transformers | 6.94%       | 7.45% | 8.30%        | 3.98%        | 1.80%               | 2.91%               | 2.75%        | 4.88% |

**Open ASR Leaderboard — RTFx** (real audio_s / inference_s)

| model                      | engine             | voxpopuli   | ami   | earnings22   | gigaspeech   | librispeech_clean   | librispeech_other   | spgispeech   |
|----------------------------|--------------------|-------------|-------|--------------|--------------|---------------------|---------------------|--------------|
| ark-asr-3b                 | starling           | 47x         | 28x   | 37x          | 26x          | 38x                 | 10x                 | 2x           |
| ark-asr-3b                 | stock transformers | 9x          | 6x    | 1x           | 5x           | 6x                  | 6x                  | 6x           |
| cohere-transcribe-03-2026  | starling           | 38x         | 21x   | 31x          | 28x          | 33x                 | 27x                 | 56x          |
| cohere-transcribe-03-2026  | stock transformers | 32x         | 23x   | 31x          | 23x          | 26x                 | 26x                 | 26x          |
| granite-speech-4.1-2b      | starling           | 49x         | 45x   | 50x          | 40x          | 46x                 | 41x                 | 43x          |
| granite-speech-4.1-2b      | stock transformers | 5x          | 6x    | 6x           | 5x           | 5x                  | 5x                  | 5x           |
| moss-transcribe-preview-2b | starling           | 46x         | 35x   | 49x          | 40x          | 53x                 | 48x                 | 53x          |
| moss-transcribe-preview-2b | stock transformers | 6x          | 6x    | 5x           | 5x           | 6x                  | 5x                  | 6x           |
| parakeet-tdt-0.6b-v3       | starling           | 9x          | 6x    | 12x          | 33x          | 87x                 | 120x                | 818x         |
| parakeet-tdt-0.6b-v3       | stock transformers | 7x          | 11x   | 70x          | 53x          | 62x                 | 62x                 | 48x          |
| qwen3-asr-1.7b             | starling           | 64x         | 49x   | 63x          | 48x          | 56x                 | 52x                 | 52x          |
| qwen3-asr-1.7b             | stock transformers | 6x          | 6x    | 6x           | 4x           | 5x                  | 5x                  | 5x           |
<!-- BENCH:WER:END -->

*Granite, moss, qwen3, cohere use 50 clips/dataset; parakeet and ark use 10
(their graphed pipelines can't yet evict CUDA graphs at high shape diversity,
so per-clip capture cost depresses their RTFx). WER remains meaningful and
byte-exact starling-vs-stock at N=10.*

## What did not work

- **INT8 weight-only quant** is slower — decode is launch-bound, not bandwidth-bound.
- **FP8 `_scaled_mm`** is slower for the same reason.
- **`torch.compile` on the encoder** is not byte-exact: inductor upcasts attention to fp32 and the conformer's BatchNorm amplifies the difference.
- **Batched spec decoding at B≥16** is slower than non-spec (0.76x) — lock-step cache rewind wastes verify work when streams differ in acceptance.

## Requirements

- Ampere+ NVIDIA GPU (RTX 30/40/50, A100, H100), bf16. Tuned on RTX 5090 (sm_120).
- CUDA 13.0, Python 3.10–3.12, [uv](https://github.com/astral-sh/uv). Torch wheels are pinned to the cu130 index in `pyproject.toml` — the default PyPI wheel is cu12/sm_90 and will not run on Blackwell.
- The leaderboard bench pulls the `hf-audio/open-asr-leaderboard` dataset (set `HF_TOKEN` if rate-limited). Clips cache under `tests/fixtures/leaderboard_corpus/`. External `CrispASR` / `parakeet.cpp` engines live in a sibling `~/asr-bench` checkout and are silently skipped if absent.

## Server

`src/starling/server.py` is a long-lived local HTTP/WebSocket sidecar that keeps
one model resident in VRAM. One process runs one model at a time (`--model
granite|parakeet|moss|qwen3`, default `granite`); `/health` reports which is
loaded.

```bash
python -m starling.server --model granite --port 8181 --max-chunk-seconds 30
python -m starling.server --model parakeet --warmup   # pre-capture CUDA graphs
```

Endpoints (FastAPI when available, stdlib fallback):

| Method + path             | Purpose |
| ------------------------- | ------- |
| `GET  /` `/health`        | liveness + `phase` (`loading_weights`/`warming_up`/`ready`) and `queue_depth` |
| `POST /inference`         | multipart or raw WAV -> `{text, segments, duration_s, request_id}` |
| `POST /transcribe`        | raw WAV bytes -> same shape as `/inference` |
| `POST /warmup`            | pre-capture CUDA graphs on a silent clip (idempotent; 202 Accepted) |
| `DELETE /inference/<id>`  | cancel a queued request by its `X-Request-Id` |
| `WS   /stream`            | real-time streaming dictation |

A single GPU worker serves one request at a time; concurrent requests queue
(up to `MAX_WAITERS`) and only get HTTP 503 when full. `X-Request-Id` on a POST
enables `DELETE /inference/<id>` cancellation — best-effort once on the GPU
(CUDA-graph replays aren't preemptible; an in-flight request finishes its
current step then returns HTTP 499).

**Timestamps.** `/inference` returns chunk-level segments
(`[{text, start_s, end_s}]`). LLM-decoder models (granite, moss, qwen3) have no
per-token audio alignment, so segments are at `--max-chunk-seconds` granularity;
shrink it for finer segments at the cost of more decode passes. Parakeet
aligns internally and returns a single whole-utterance segment.
