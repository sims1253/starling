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
- [`bosonai/higgs-audio-v3-stt`](https://huggingface.co/bosonai/higgs-audio-v3-stt) — Whisper-large-v3 mel + MLP projector + Qwen3-1.7B decoder. The CUDA megakernel (encoder kept eager) runs under its own `.venv-higgs` (transformers 4.51) because the model's `trust_remote_code` modeling breaks under transformers 5.13; the in-tree ggml engine (`starling-ggml-higgs`, full Whisper encoder + avg pool + MLP projector + Qwen3 decoder with qk_norm) runs from the main environment and is byte-exact against the golden on short/medium/long.
- [`nvidia/Nemotron-Labs-Audex-2B`](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B) — Whisper-large-v3 encoder (with avg-pooler) + relu2 projector + Nemotron-Dense 2B decoder (squared-ReLU MLP, not SwiGLU). ASR path only. **NVIDIA Oneway Noncommercial License** — non-commercial use only, unlike the Apache/MIT-licensed models above.
- [`HojoAI/Hojo-ASR-V1`](https://huggingface.co/HojoAI/Hojo-ASR-V1) — Whisper-large-v3 mel + Qwen3-Omni audio tower (3× conv2d + 32 transformer layers) + WeNet Conformer bottleneck (2 blocks, rel-pos MHA + BatchNorm conv module) + Qwen3-4B decoder. First **beam-4** decode model in the repo (all others are greedy). The CUDA megakernel runs under `.venv-hojo` (transformers 4.57); the in-tree ggml engine (`starling-ggml-hojo`) runs from the main environment and is byte-exact against the golden on short/medium/long.

The autoregressive models (granite, moss, qwen3, ark, higgs, audex, cohere, hojo) share an
encoder + LLM-decoder pattern where the decode loop is the bottleneck. Parakeet
is a transducer; granite-nar is a single bidirectional pass.

## Benchmark

Two scripts, each the single source of truth for its slice. Both support
`--update-readme` to splice tables into the sentinel-wrapped blocks below.

- **`benchmarks/bench_all.py`** — latency/RTFx grid. Sweeps model × engine ×
  audio length × batch size on tiled-LibriSpeech fixtures. Engines: `starling`,
  `stock`, `crispasr`, `parakeet.cpp`, `starling-ggml`, `starling-batched`
  (granite/qwen3), `starling-spec` (granite).
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

**granite-speech-4.1-2b-nar** — latency / RTFx (ms, RTFx×)

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
- **FP8 `_scaled_mm`** is slower for M=1 decode and proved unsafe across many
  captured graphs. The shipped FP8 path uses a fused weight-only Triton GEMV.
- **`torch.compile` on the encoder** is not byte-exact: inductor upcasts attention to fp32 and the conformer's BatchNorm amplifies the difference.
- **Batched spec decoding at B≥16** is slower than non-spec (0.76x) — lock-step cache rewind wastes verify work when streams differ in acceptance.

## Requirements

- Ampere+ NVIDIA GPU (RTX 30/40/50, A100, H100), bf16. Tuned on RTX 5090 (sm_120).
- CUDA 13.0, Python 3.10–3.12, [uv](https://github.com/astral-sh/uv). Torch wheels are pinned to the cu130 index in `pyproject.toml` — the default PyPI wheel is cu12/sm_90 and will not run on Blackwell.
- The leaderboard bench pulls the `hf-audio/open-asr-leaderboard` dataset (set `HF_TOKEN` if rate-limited). Clips cache under `tests/fixtures/leaderboard_corpus/`. External `CrispASR` / `parakeet.cpp` engines live in a sibling `~/asr-bench` checkout and are silently skipped if absent.

## Platforms (Linux + Windows)

Starling runs on **Linux** and **native Windows** (no WSL2 needed). The fused
decode kernels live behind a backend dispatch in `src/starling/_kernels/`, so
the same model code runs unchanged on both OSes — it just picks a different
kernel backend. All three backends are **byte-exact** on the default decode
path (verified by `tests/test_kernel_backends.py` and the per-model golden
tests run under each backend).

The dispatch (`auto`) selects the fastest backend available, in this order:

- **Triton backend** (`src/starling/_kernels/triton_backend.py`) — the
  hand-tuned, autotuned kernels the benchmark tables were measured on. Default
  on Linux, where the `triton` wheel installs cleanly. **Fastest.**
- **CUDA C++ backend** (`src/starling/_kernels/cuda_backend.py` +
  `cuda/backend.cu`) — selected on Windows when a CUDA toolkit + compiler are
  present (Triton has no official Windows wheel). The kernels are JIT-compiled
  from CUDA C++ via `torch.utils.cpp_extension.load_inline` on first use (one
  ~30–60 s compile, then cached under `~/.cache/starling`). Delivers
  Linux/Triton-class performance: near-parity on the elementwise kernels
  (1–2 µs each) and 2.6–8× faster fp8 than the torch backend, closing the gap
  that otherwise made the fp8-weights path pointless on Windows.
- **Torch backend** (`src/starling/_kernels/torch_backend.py`) — stock-PyTorch
  fused ops, selected as the last resort when neither Triton nor a CUDA
  toolchain is available. Byte-exact for the elementwise kernels; its fp8 GEMV
  materializes the full bf16 weight (correct but not bandwidth-optimal), so on
  a torch-only install leave `fp8_weights` off (the default).

- **macOS / Apple Silicon is not supported** — the architecture is built on
  `torch.cuda.CUDAGraph`, which is NVIDIA-only.

Select the backend explicitly with the `STARLING_KERNEL_BACKEND` env var
(`auto` | `triton` | `cuda` | `torch`) or the `OptFlags.kernel_backend` field.
`auto` resolves to `triton` (if importable) → `cuda` (if a CUDA GPU is
visible) → `torch`. On Windows + CUDA toolkit that means full speed with no
code changes.

Set up either platform with the same command (a cross-platform Python entry
point):

```bash
python scripts/setup_env.py
```

To A/B/C compare the Triton, torch, and CUDA backends head-to-head (on a box
that has all three), use the dedicated harness:

```bash
uv run python benchmarks/bench_kernels.py
```


## Server

`src/starling/server.py` is a long-lived local HTTP/WebSocket sidecar that keeps
one model resident in VRAM. One process runs one model at a time (`--model
granite|parakeet|parakeet_unified|moss|qwen3|ark|cohere|higgs|audex`, default
`granite`); `/health` reports which is loaded. Higgs must be run from the
isolated `.venv-higgs` environment documented in `src/starling/higgs/UV_NOTES.md`.

```bash
python -m starling.server --model granite --port 8181 --max-chunk-seconds 30
python -m starling.server --model parakeet --profile realtime --warmup
python -m starling.server --model moss --profile batch  # SDPA + fused fp8
```

Endpoints (FastAPI when available, stdlib fallback):

| Method + path             | Purpose |
| ------------------------- | ------- |
| `GET  /` `/health`        | liveness + `phase` (`loading_weights`/`warming_up`/`ready`) and `queue_depth` |
| `POST /inference`         | multipart or raw WAV -> `{text, segments, duration_s, request_id}` |
| `POST /transcribe`        | raw WAV bytes -> same shape as `/inference` |
| `POST /warmup`            | pre-capture CUDA graphs on a silent clip (idempotent; 202 Accepted) |
| `DELETE /inference/<id>`  | cancel a queued or running request by its `X-Request-Id` |
| `WS   /stream`            | real-time streaming dictation |

A single GPU worker serves one request at a time; concurrent requests queue
(up to `MAX_WAITERS`) and only get HTTP 503 when full. `X-Request-Id` on a POST
enables `DELETE /inference/<id>` cancellation — best-effort once on the GPU
(CUDA-graph replays aren't preemptible; an in-flight request finishes its
current step then returns HTTP 499).

Requests without an `X-Request-Id` receive a generated ID in the response.
Uploads are capped at 256 MiB and requests have a 10-minute wall-clock deadline
by default; tune these with `--max-upload-mb` and `--request-timeout-seconds`
(pass `0` or a negative value to disable the deadline entirely). A single GPU
worker serves one request at a time, so disabling the deadline can occupy it
indefinitely — only use a non-positive value for trusted local use, or ensure
an upstream proxy enforces its own timeout. The API has no authentication, so
binding a non-loopback `--host` emits a warning and should only be done behind
an authenticated proxy.

Profiles provide supported defaults for the main workloads:

| profile | intended workload | graph/optimization policy |
| ------- | ----------------- | ------------------------- |
| `file` (default) | one-shot files | adaptive graphs, strict flags |
| `realtime` | low-latency dictation | graphed recurring windows + SDPA |
| `batch` | long-form offline throughput | graphed chunks + tolerance-mode SDPA, plus graph-safe fused fp8 weights on granite/moss |
| `accuracy` | strict/reference output | adaptive graphs, strict byte-exact flags |

Model selection is workload-dependent: parakeet has the lowest realtime
latency, moss has the best measured leaderboard WER, qwen3 is a strong
speed/accuracy compromise, and granite is useful when its self-speculative
path is desired. The HTTP server intentionally serializes requests; the
granite/qwen3 batched pipelines are exposed by `bench_all.py` for offline batch
jobs rather than silently changing per-request latency semantics.

**Timestamps.** `/inference` returns chunk-level segments
(`[{text, start_s, end_s}]`). LLM-decoder models (granite, moss, qwen3, ark,
higgs, audex) have no
per-token audio alignment, so segments are at `--max-chunk-seconds` granularity;
shrink it for finer segments at the cost of more decode passes. Parakeet
and cohere chunk internally and return a single whole-utterance segment.

## Native serving: starling-serve

`starling-serve` is a self-contained native binary that serves the GGML
engines behind the same HTTP/WS API — no Python, no torch, one static
executable per platform (CUDA, ROCm, Vulkan, Metal, CPU). It covers the five
GGML-ported models (`parakeet`, `moss`, `ark`, `higgs`, `hojo`); the Python
server above remains the path for the CUDA-megakernel-only models (granite,
parakeet_unified, qwen3, cohere, audex).

```bash
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON   # or VULKAN/METAL/HIP
cmake --build build -j --target starling-serve
./build/starling-serve --model parakeet --gguf model.gguf --port 8181
```

Note: the native server requires 16 kHz audio (no C++ resampler; non-16 kHz
WAVs get a 400). Build variants, the full CLI, the endpoint contract, and
pre-converted GGUF downloads are documented in
[`docs/native-serving.md`](docs/native-serving.md).
