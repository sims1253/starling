# Starling

CUDA-graph inference kernels for speech-recognition models, tuned to run as
fast as possible on a single RTX 5090 (Blackwell, sm_120). Can probably be 
adapted to other GPUs pretty easy.

The core idea is the same across the models. The stock `transformers` decode
loop emits a few hundred tiny kernels per output token and spends most of its
wall time on CPU launch overhead, with the GPU sitting around 10% busy.
Everything that can be captured into a CUDA-graph replay gets captured: decode
steps, fused RMSNorm/SwiGLU, the attention mask, whole multi-step token loops
(autoregressive models), or the single bidirectional editor forward
(granite-speech-4.1-2b-nar). Output is byte-identical to the eager
`transformers` reference, so there is no accuracy trade-off, just fewer round
trips to the GPU.

Most props go to GLM 5.2

## Models

All do speech-to-text; the autoregressive ones (granite, moss, qwen3) share an
encoder + LLM-decoder pattern where the decode loop is the bottleneck, while
parakeet is a transducer and the NAR model is a single bidirectional pass.

- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) (encoder + 1B LLM decoder). The LLM decode is the bottleneck. Includes an optional self-speculative path that drafts tokens from the encoder's CTC head.
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer + TDT transducer, no LLM). Tuned for batched offline throughput, with GPU-side mel extraction and chunking for hour-long audio.
- [`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B) (Qwen3-omni MoE audio encoder + Qwen3 LLM decoder). The same encoder+LLM-decoder pattern as granite: the decode loop is the bottleneck, so a `torch.compile`d hand-iterated layer loop with fused Triton elementwise kernels is captured into a K-step CUDA graph, plus a per-prompt-length graphed prefill. Output is byte-identical to the eager reference.
- [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (Whisper-style windowed-attention audio encoder + Qwen3 LLM decoder). Three byte-exact CUDA-graph layers: a graphed encoder (custom static-shape windowed-attention kernel, since the stock layer does host ops on `cu_seqlens`), a graphed greedy Qwen3 decode over a static KV cache with fused Triton RMSNorm/SwiGLU/residual/QK-norm kernels, and a K-step (K=8) multi-step decode graph. Output is byte-identical to the eager reference.
- [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) (non-autoregressive ASR). Unlike every other model here, there is **no decode loop**: ASR is one bidirectional forward pass — a CTC conformer encoder produces a rough token draft, blank "edit slots" are interleaved, and a *bidirectional* granite-4.0-1b LLM editor refines the whole sequence in a single forward. The win is CUDA-graph capture of the encoder trunk plus a `torch.compile`d (then graph-captured) LLM editor forward, removing host launch overhead across the 16 encoder + 40 LLM layers. Output is byte-identical to the eager reference.
- [`AutoArk-AI/ARK-ASR-3B`](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) (Whisper encoder + MLP adapter + Qwen2.5 decoder). Same encoder+LLM-decoder pattern as granite. The Whisper+adapter forward and the prefill are each captured into shape-keyed CUDA graphs, and the Qwen2.5 decode loop runs as a K-step graph with fused Triton elementwise glue reused from the granite kernels. Output is byte-identical to the eager reference.
- [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (~2B params, the repo's first **seq2seq encoder-decoder**, Whisper-style). A Parakeet FastConformer encoder (48 layers) + an 8-layer Transformer decoder with **self-attention AND cross-attention**. A graphed encoder + a K-step graphed greedy decode over an `EncoderDecoderCache` (StaticCache for both halves, so the K/V tensors are fixed-shape and capture-safe). The decode step has two attention blocks per layer; the self-attn causal mask is built dynamically in-graph from an advancing position counter (no per-step baked masks). Output is byte-identical to the eager reference.
- [`bosonai/higgs-audio-v3-stt`](https://huggingface.co/bosonai/higgs-audio-v3-stt) (Whisper-large-v3 mel encoder + MLP projector + Qwen3-1.7B decoder). Same encoder+LLM-decoder pattern as granite: the Qwen3 decode loop is the bottleneck, so the model's own layers are captured into a CUDA graph over a static KV cache (single- and K-step variants); the Whisper tower + projector prefill run eager. Output is byte-identical to the eager reference. Runs under its own isolated venv (`.venv-higgs`, transformers 4.51) because the model's `trust_remote_code` modeling breaks under the repo's transformers 5.13.

## Numbers

Single RTX 5090, bf16, model load excluded. RTFx (realtime factor) is
audio_seconds / transcribe_seconds, so 100x means 100 seconds of audio
transcribed in 1 second — higher is faster.

Every model is benchmarked on the same audio-length tiers (short ~7s, medium
~22s, long ~45-74s), same weights. The full grid — `starling` vs the
unmodified HuggingFace `generate()` reference (`stock transformers`), the
external `CrispASR` ggml binary, and `parakeet.cpp` — is regenerated by the
benchmark script below and spliced into the **Benchmark** section. Numbers
live in exactly one place; this narrative only covers what the tables can't.

**Granite speculative decoding.** `starling (spec)` drafts tokens from the
encoder's CTC head and verifies them with the LLM. Spec is slower on short
audio (draft extraction has fixed overhead) but pulls ahead on longer audio
where accepted drafts save LLM forward passes. At B=1 it is ~1.65x over
non-spec greedy; at B≥16 the GEMMs are large enough that speculation wastes
more compute than it saves (0.76x regression at B=32), so batched decode
always uses the non-spec path.

**Batching.** Not every model batch-decodes, and the tables reflect that.
Parakeet batches natively (one fused call for B clips). Granite and Qwen3
have batched LLM-decode paths (`starling (batched)`) that turn the
launch-bound B=1 GEMVs into saturating B-wide GEMMs — granite via its
chunked long-audio path, qwen3 single-shot (bounded by its 4096-token static
KV cache). MOSS hard-rejects B>1 (no batched decoder), and the non-AR NAR
model is a single forward pass (B=1 today; structurally batchable but not
wired). Where a model has no batched engine, its B>1 cells stay at B=1.

**Long audio (30-90 min).** Beyond the tier tables, both batched winners
transcribe hour-long tiled audio: granite peaks at B=48 (B=16 216x → B=48
275x → B=64 241x at 60min), parakeet peaks at B=32 (~3800x, 3.3x less VRAM
than granite). 30s chunks, 2s overlap; granite uses text-level overlap
dedup, parakeet uses frame-aligned TDT-duration stitching.

## Benchmark

Two re-runnable scripts, each a single source of truth for its slice:

- **`benchmarks/bench_all.py`** — the latency/RTFx grid. Sweeps every
  supported model × engine × audio length × batch size on deterministic
  tiled-LibriSpeech fixtures, producing the RTFx tables below. Engines:
  `starling`, `stock`, `crispasr`, `parakeet.cpp`, `starling-batched`
  (granite/qwen3), `starling-spec` (granite).
- **`benchmarks/bench_leaderboard.py`** — the accuracy/quality grid.
  Reproduces the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
  English short-form methodology (Whisper `EnglishTextNormalizer` +
  `kaldialign` WER with `merge_compounds=True`, unweighted mean across the 7
  datasets) on real diverse audio — a genuine quality metric and the
  degradation detector: a real WER budget lets future non-byte-exact
  optimizations land safely.

```
  uv run python benchmarks/bench_leaderboard.py                  # capped, fast
  uv run python benchmarks/bench_leaderboard.py --num-samples 0   # full splits
```

### ark-asr-3b (3B params)

B=1 single-stream, fused Triton decode (K=8 multistep graph) with a shape-keyed
graphed prefill and graphed Whisper+adapter encoder. `starling` is byte-identical
to the eager `transformers` reference. The Qwen2.5 decode loop is the bottleneck;
the per-layer QKV and gate+up GEMVs are each fused into one GEMM and RoPE runs as
one Triton kernel, so decode hits ~6.2ms/tok (161 tok/s). The encoder is ~7-21ms
and the prefill is captured so it adds only tens of ms regardless of prompt length.

No external C++ engine runs ARK-ASR-3B yet (CrispASR/whisper.cpp have no arkasr
backend and no GGUF conversion exists), so the only comparison is stock
`transformers`.

| audio | starling | stock transformers |
| ----- | -------- | ------------------ |
| 7s    | 248ms (30x) | 1657ms (4x) |
| 22s   | 608ms (37x) | 5467ms (4x) |
| 74s   | 680ms (109x) | 4661ms (16x) |

### cohere-transcribe-03-2026 (2B params, seq2seq encoder-decoder)

`starling` is byte-identical to the eager `transformers` reference. The repo's
first **seq2seq encoder-decoder** (Whisper-style): a Parakeet FastConformer
encoder (48 layers) + an 8-layer Transformer decoder with self-attention AND
cross-attention. Two byte-exact CUDA-graph layers:

- **Graphed encoder** — the 48-layer FastConformer forward captured per mel
  shape (collapses hundreds of per-layer launches into one replay).
- **K-step graphed decode** — K=16 consecutive decode steps captured into one
  CUDA graph (argmax chained in-graph, 1 host sync per 16 tokens). The decode
  drives the model's own decoder layers over an `EncoderDecoderCache` built on
  two `StaticCache`s (self-attn + cross-attn), so the K/V tensors are
  fixed-shape and capture-safe. The per-step self-attn causal mask is built
  dynamically in-graph from an advancing position counter (the captured template
  runs at any base position, so it serves the whole decode in K-step blocks). A
  precomputed 4D additive bidirectional mask makes `create_bidirectional_mask`
  early-exit (no CPU-scalar capture abortors). Prefill is eager (fills the
  cross-attn cache once); steps 1+ are graphed.

Decode steady-state is ~3.1ms/tok (318 tok/s) at B=1. Audio longer than
`max_audio_clip_s` (35s) is auto-chunked by the processor; each chunk decodes
independently (long tier is B=3).

| audio | starling | stock transformers |
| ----- | -------- | ------------------ |
| 7s    | 54ms (139x)  | 992ms (7x) |
| 22s   | 376ms (59x)  | 2560ms (9x) |
| 74s   | 610ms (122x) | 3567ms (21x) |

### higgs-audio-v3-stt (2.68B params)

B=1 single-stream. `starling` is a CUDA-graph-captured Qwen3-1.7B decode over a
static KV cache. Three byte-exact layers stack: (1) graph-capture of the model's
own layers over `StaticCache` (removes launch overhead), (2) **fused Triton
elementwise kernels** (RMSNorm, SwiGLU, residual add, per-head QK-norm) replacing
the layer glue — unlike granite, higgs benefits substantially (the decode is
compute/memory-bound at the glue once launch overhead is gone, not
pure-GEMV-launch-bound like granite), and (3) **`torch.compile(mode="max-autotune-
no-cudagraphs")`** on the fused decode so inductor fuses the remaining PyTorch
elementwise glue (RoPE cat+mul+add, attention softmax prep, GQA repeats).
Steady-state decode: **3.05ms/tok / 327 tok/s** (byte-exact — the "compile not
byte-exact" finding was the *encoder*'s BatchNorm, not the LLM decode). A K-step
multi-step variant (subclassing the fused path) stacks for host-sync
amortisation. The Whisper-large-v3 audio tower (32 layers) + MLP projector run
eager once per clip (the prefill); the Qwen3 decode loop is the bottleneck.
Identical transcripts to stock. Runs under its own isolated venv (`.venv-higgs`,
transformers 4.51) because the model's `trust_remote_code` modeling breaks under
the repo's transformers 5.13.

| audio | starling | stock transformers | [CrispASR](https://github.com/CrispStrobe/CrispASR) |
| ----- | -------- | ------------------ | -------- |
| 7s    | 98ms (76x)  | 5321ms (1.4x)  | 4043ms (1.8x) |
| 22s   | 304ms (73x) | 8422ms (2.7x)  | 13197ms (1.7x) |
| 74s   | 496ms (150x)| 13451ms (5.5x) | — |

starling is **~54–111x faster than stock `generate()` and ~25–130x faster than
CrispASR's q4_k ggml path** on the same model. (CrispASR wall includes model load;
its `higgs-stt` backend loads the real higgs weights via a whisper-tiny VAD
front-end. Long-audio CrispASR timed out the harness.)

### Long audio (30-90 min)

Run either any time; `--update-readme` splices its tables into the
sentinel-wrapped blocks here. Scope is tunable (`--models`, `--engines`,
`--lengths`/`--datasets`, `--batches`, `--reps`).

```
uv run python benchmarks/bench_all.py --update-readme
uv run python benchmarks/bench_leaderboard.py --update-readme
```

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

The WER column above is the **byte-exact drift gate** (tiled fixtures, `jiwer`
against the repeated transcript) — it confirms `starling` matches `stock`
exactly and is meaningless as an absolute quality metric (repeating one
utterance makes the models' WER reflect tiling artifacts, not recognition
quality). Absolute quality is measured separately below.

### Accuracy (Open ASR Leaderboard reproduction)

Real WER on diverse audio, reproducing the Open ASR Leaderboard's English
short-form methodology: the 7 datasets (voxpopuli, ami, earnings22,
gigaspeech, librispeech clean+other, spgispeech) from
`hf-audio/open-asr-leaderboard`, scored with the Whisper `EnglishTextNormalizer`
applied to both reference and hypothesis plus `kaldialign` WER
(`merge_compounds=True`); the headline is the unweighted mean across datasets.
This is the degradation detector — a real WER budget is what lets non-byte-exact
optimizations land safely. (`--num-samples 0` runs the full splits; the default
caps each dataset for a tractable local run.)

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

*Granite, moss, qwen3, and cohere use 50 clips/dataset; parakeet and ark use
10 clips/dataset (their graphed pipelines can't safely evict CUDA graphs at
high shape diversity yet, so the smaller sample keeps VRAM bounded). WER is
still meaningful and byte-exact starling-vs-stock at N=10; parakeet/ark RTFx
is depressed by per-clip graph-capture overhead (no eviction = each new clip
length pays a one-time capture cost).*

## What did not work

- INT8 weight-only quant is slower. Decode is launch-bound, not bandwidth-bound, so halving weight traffic does not help.
- FP8 `_scaled_mm` is also slower, for the same reason.
- `torch.compile` on the encoder is not byte-exact: inductor upcasts attention to fp32 and the conformer's BatchNorm amplifies the difference.
- Batched spec decoding at B>=16 is slower than non-spec (0.76x). The lock-step cache rewind wastes verify work when streams have differing acceptance.

## Requirements

- Tuned on an RTX 5090 (Blackwell, sm_120). Runs on any Ampere+ NVIDIA GPU
  (RTX 30/40/50, A100, H100) as bf16 is required. The torch wheels are pinned to the
  CUDA 13.0 (cu130) index in `pyproject.toml`. The default PyPI torch wheel is
  cu12 / sm_90 and will not run on Blackwell.
- CUDA 13.0, Python 3.10-3.12, and [uv](https://github.com/astral-sh/uv).
- The leaderboard accuracy bench pulls the `hf-audio/open-asr-leaderboard`
  dataset (set `HF_TOKEN` if you hit rate limits); it adds `datasets`,
  `torchcodec`, `kaldialign`, and `whisper-normalizer` (already in
  `pyproject.toml`). Clips are cached under `tests/fixtures/leaderboard_corpus/`
  after the first run. The external `CrispASR` / `parakeet.cpp` engines live in
  a sibling `~/asr-bench` checkout and are silently skipped if absent.

## Layout

```
src/starling/           shared toolkit (config dims, optimisation flags)
  config.py             Granite-Speech architecture constants (single source of truth)
  flags.py              runtime optimisation flags (byte-exact vs tolerance mode)
  server.py             unified HTTP/WebSocket ASR sidecar (all models; --model flag)
  granite/              granite-speech-4.1-2b megakernel
    encoder_mega.py     fused (cudagraph) conformer encoder
    llm_mega.py         graphed greedy decode over a static KV cache
    multistep.py        K-step graphed decode (multi-step per replay)
    pipeline.py         encoder + projector + LLM wiring
    batched.py          batched (B>1) LLM decode + pipeline
    long_audio.py       chunked long-audio transcription (sequential + batched)
    speculative.py      self-speculative decoding via the CTC draft head
  parakeet/             parakeet-tdt-0.6b-v3 megakernel
    decode_mega.py      multi-step graphed TDT decode
    encoder_graph.py    graphed FastConformer encoder
    mel_gpu.py          GPU-side mel filterbank
    chunking.py         bounded-VRAM long-audio chunking
  moss/                 moss-transcribe-preview-2b megakernel
    llm_mega.py         graphed greedy Qwen3 decode over a static KV cache
                         + per-prompt-length graphed prefill
    fused_decode.py     hand-iterated layer loop + fused Triton elementwise kernels
                         (torch.compile'd for the decode path)
    multistep.py        K-step graphed decode (multi-step per replay)
    encoder_graph.py    eager Qwen3-omni MoE audio encoder + adapter
    pipeline.py         encoder + adapter + LLM wiring
  qwen3/                qwen3-asr-1.7b megakernel
    encoder_mega.py     graphed windowed-attention audio encoder
    llm_mega.py         graphed greedy Qwen3 decode over a static KV cache
    multistep.py        K-step graphed decode (multi-step per replay)
    pipeline.py         encoder + projector + LLM wiring
  nar/                  granite-speech-4.1-2b-nar megakernel (non-autoregressive)
    mega.py             single-pass pipeline: graphed encoder trunk +
                         torch.compiled + graphed bidirectional LLM editor
    fused_llm.py        hand-iterated LLM forward (documented negative result —
                         diverges from stock cuBLAS on long packed sequences)
  ark/                  ARK-ASR-3B megakernel
    encoder_mega.py     graphed Whisper+MLP-adapter audio encoder
    llm_mega.py         graphed greedy decode over a static KV cache + graphed prefill
    multistep.py        K-step graphed decode (multi-step per replay)
    pipeline.py         encoder + audio-embedding injection + LLM wiring
  cohere/               cohere-transcribe-03-2026 megakernel (seq2seq enc-dec)
    encoder_graph.py    graphed FastConformer encoder (48 layers)
    decode_mega.py      K-step graphed seq2seq decode over an EncoderDecoderCache
                         (StaticCache self-attn + cross-attn; dynamic in-graph mask)
    reference.py        eager golden greedy decode (byte-exact vs HF generate)
    pipeline.py         graphed encoder + graphed decode wiring
  higgs/                higgs-audio-v3-stt megakernel (runs under .venv-higgs, tf 4.51)
    llm_mega.py         graphed greedy Qwen3 decode over a static KV cache
    multistep.py        K-step graphed decode (multi-step per replay)
    pipeline.py         collator + eager prefill + graphed decode wiring
    loader.py           model/tokenizer/collator loading (isolated venv notes)
    UV_NOTES.md         how to run higgs via uv (isolated .venv-higgs, tf 4.51)
    vendor/             vendored modeling + collator (tf-version-independent)
benchmarks/             RTF and cross-engine benchmarks
scripts/                bench and probe scripts
tests/                  correctness checks vs. golden references
```

## Server

`src/starling/server.py` is a long-lived local HTTP/WebSocket sidecar that keeps
one model resident in VRAM. It serves every supported model behind a `--model`
flag (`granite` | `parakeet` | `moss` | `qwen3`, default `granite`); one process
runs one model at a time, and `/health` reports which model is loaded so clients
can match per-row readiness. It is the integration surface for external clients
(e.g. the freestyle Electron app's overlapping-window STT provider). Run it with:

```bash
python -m starling.server --model granite --port 8181 --max-chunk-seconds 30
python -m starling.server --model parakeet --warmup   # pre-capture CUDA graphs
```

Endpoints (FastAPI when available, else a stdlib fallback):

| Method + path             | Purpose |
| ------------------------- | ------- |
| `GET  /` `/health`        | liveness + `phase` (`loading_weights`/`warming_up`/`ready`) and `queue_depth` |
| `POST /inference`         | multipart or raw WAV -> `{text, segments, duration_s, request_id}` |
| `POST /transcribe`        | raw WAV bytes -> same shape as `/inference` |
| `POST /warmup`            | pre-capture CUDA graphs on a silent clip (idempotent; 202 Accepted) |
| `DELETE /inference/<id>`  | cancel a queued request by its `X-Request-Id` |
| `WS   /stream`            | real-time streaming dictation |

**Timestamps.** `/inference` returns chunk-level segment timestamps
(`segments: [{text, start_s, end_s}]`). The LLM-decoder models (granite, moss,
qwen3) have no per-token audio alignment (the LLM decodes text with no duration
head), so for those models segments are at `--max-chunk-seconds` granularity —
one per chunk window — rather than per-word; shrink `--max-chunk-seconds` for
finer segments at the cost of more decode passes. Parakeet handles long audio
and alignment internally, so it returns a single whole-utterance segment.

**Queueing.** A single GPU worker serves one request at a time; concurrent
requests queue (up to `MAX_WAITERS`) instead of being rejected, and only get
HTTP 503 once the queue is full. A client that supplies `X-Request-Id` on a
POST may `DELETE /inference/<id>` to drop it from the queue. Cancellation is
best-effort for a request already on the GPU: CUDA-graph replays are not
preemptible, so an in-flight request still finishes its current decode step but
is returned as HTTP 499.
