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
- [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) — Unified FastConformer-RNN-T. NeMo-free port: the `.nemo` checkpoint is loaded directly (no `nemo_toolkit`), the encoder/prediction-net/joint are hand-built in PyTorch, and the encoder + greedy RNN-T decode are captured into CUDA graphs.
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
| short    |       1 | 171ms (43x)  | 2543ms (3x)          |
| medium   |       1 | 326ms (68x)  | 5238ms (4x)          |
| long     |       1 | 1226ms (61x) | 18831ms (4x)         |

**parakeet-tdt-0.6b-v3** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 14ms (531x)  | 167ms (45x)          |
| short    |       8 | 24ms (313x)  | —                    |
| medium   |       1 | 22ms (1004x) | 419ms (53x)          |
| medium   |       8 | 49ms (458x)  | —                    |
| long     |       1 | 56ms (1338x) | 1425ms (52x)         |
| long     |       8 | 163ms (455x) | —                    |

**moss-transcribe-preview-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling    | stock transformers   |
|----------|---------|-------------|----------------------|
| short    |       1 | 157ms (47x) | 1591ms (5x)          |
| medium   |       1 | 380ms (59x) | 5224ms (4x)          |
| long     |       1 | 756ms (98x) | 23719ms (3x)         |

**ark-asr-3b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 186ms (40x)  | 1484ms (5x)          |
| medium   |       1 | 538ms (42x)  | 4687ms (5x)          |
| long     |       1 | 600ms (124x) | 4853ms (15x)         |

**cohere-transcribe-03-2026** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 55ms (134x)  | 323ms (23x)          |
| medium   |       1 | 171ms (131x) | 918ms (24x)          |
| long     |       1 | 324ms (230x) | 1825ms (41x)         |
<!-- BENCH:END -->

### Accuracy (Open ASR Leaderboard reproduction)

<!-- BENCH:WER:START -->
**Open ASR Leaderboard — WER %** (per dataset, unweighted mean avg)

| model                      | engine             | voxpopuli   | ami   | earnings22   | gigaspeech   | librispeech_clean   | librispeech_other   | spgispeech   | avg   |
|----------------------------|--------------------|-------------|-------|--------------|--------------|---------------------|---------------------|--------------|-------|
| ark-asr-3b                 | starling           | 11.35%      | 6.31% | 8.04%        | 3.77%        | 2.60%               | 3.97%               | 2.35%        | 5.48% |
| ark-asr-3b                 | stock transformers | 11.38%      | 6.21% | 8.15%        | 3.77%        | 2.63%               | 3.81%               | 2.25%        | 5.46% |
| cohere-transcribe-03-2026  | starling           | 10.32%      | 6.31% | 8.59%        | 5.47%        | 1.47%               | 1.78%               | 2.45%        | 5.20% |
| cohere-transcribe-03-2026  | stock transformers | 10.28%      | 6.31% | 8.59%        | 5.51%        | 1.47%               | 1.81%               | 2.45%        | 5.20% |
| granite-speech-4.1-2b      | starling           | 7.47%       | 8.02% | 8.48%        | 5.21%        | 1.77%               | 2.35%               | 2.80%        | 5.16% |
| granite-speech-4.1-2b      | stock transformers | 7.47%       | 8.02% | 8.44%        | 5.13%        | 1.77%               | 2.25%               | 2.90%        | 5.14% |
| moss-transcribe-preview-2b | starling           | 3.81%       | 6.31% | 6.72%        | 4.24%        | 1.62%               | 2.66%               | 2.15%        | 3.93% |
| moss-transcribe-preview-2b | stock transformers | 3.81%       | 6.17% | 6.68%        | 4.28%        | 1.62%               | 2.66%               | 2.10%        | 3.90% |
| parakeet-tdt-0.6b-v3       | starling           | 6.35%       | 7.21% | 7.71%        | 4.36%        | 1.71%               | 3.28%               | 3.56%        | 4.88% |
| parakeet-tdt-0.6b-v3       | stock transformers | 6.28%       | 7.21% | 7.71%        | 4.36%        | 1.68%               | 3.31%               | 3.56%        | 4.87% |
| qwen3-asr-1.7b             | starling           | 6.91%       | 7.31% | 8.19%        | 4.07%        | 1.80%               | 2.88%               | 2.80%        | 4.85% |
| qwen3-asr-1.7b             | stock transformers | 6.94%       | 7.45% | 8.30%        | 3.98%        | 1.80%               | 2.91%               | 2.75%        | 4.88% |

**Open ASR Leaderboard — RTFx** (real audio_s / inference_s)

| model                      | engine             | voxpopuli   | ami   | earnings22   | gigaspeech   | librispeech_clean   | librispeech_other   | spgispeech   |
|----------------------------|--------------------|-------------|-------|--------------|--------------|---------------------|---------------------|--------------|
| ark-asr-3b                 | starling           | 53x         | 46x   | 46x          | 40x          | 50x                 | 47x                 | 42x          |
| ark-asr-3b                 | stock transformers | 7x          | 6x    | 6x           | 5x           | 6x                  | 6x                  | 5x           |
| cohere-transcribe-03-2026  | starling           | 97x         | 83x   | 102x         | 75x          | 110x                | 97x                 | 82x          |
| cohere-transcribe-03-2026  | stock transformers | 32x         | 29x   | 36x          | 29x          | 29x                 | 26x                 | 26x          |
| granite-speech-4.1-2b      | starling           | 78x         | 74x   | 78x          | 64x          | 69x                 | 63x                 | 66x          |
| granite-speech-4.1-2b      | stock transformers | 5x          | 5x    | 5x           | 4x           | 5x                  | 4x                  | 5x           |
| moss-transcribe-preview-2b | starling           | 64x         | 54x   | 63x          | 51x          | 65x                 | 58x                 | 53x          |
| moss-transcribe-preview-2b | stock transformers | 6x          | 6x    | 6x           | 5x           | 6x                  | 5x                  | 5x           |
| parakeet-tdt-0.6b-v3       | starling           | 600x        | 533x  | 1083x        | 841x         | 1104x               | 998x                | 833x         |
| parakeet-tdt-0.6b-v3       | stock transformers | 54x         | 54x   | 66x          | 48x          | 56x                 | 52x                 | 48x          |
| qwen3-asr-1.7b             | starling           | 55x         | 50x   | 65x          | 48x          | 59x                 | 53x                 | 57x          |
| qwen3-asr-1.7b             | stock transformers | 6x          | 5x    | 6x           | 4x           | 5x                  | 4x                  | 5x           |
<!-- BENCH:WER:END -->

*All models use 50 clips/dataset. Parakeet, ark, and qwen3 previously capped at
10 because their graphed pipelines accumulated one CUDA graph per distinct clip
length at high shape diversity — saturating VRAM (ark OOM) or corrupting the
graph allocator (qwen3 illegal memory access) a few datasets into the sweep.
Two fixes, both byte-exact, cleared this: **shape-bucketing** the mel (pad up to
a canonical frame count so diverse lengths share one captured encoder graph) and
running the prompt **prefill eager** (the per-prompt-length prefill graph was
the dominant accumulator; the decode loop stays graphed). All three now run the
full 50 with flat VRAM and RTFx no longer capture-bound (parakeet 526–1104×, ark
40–53×, qwen3 47–65×). WER remains meaningful and byte-exact starling-vs-stock.*

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
