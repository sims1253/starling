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

Both do speech-to-text.

- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) (encoder + 1B LLM decoder). The LLM decode is the bottleneck. Includes an optional self-speculative path that drafts tokens from the encoder's CTC head.
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer + TDT transducer, no LLM). Tuned for batched offline throughput, with GPU-side mel extraction and chunking for hour-long audio.
- [`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B) (Qwen3-omni MoE audio encoder + Qwen3 LLM decoder). The same encoder+LLM-decoder pattern as granite: the decode loop is the bottleneck, so a `torch.compile`d hand-iterated layer loop with fused Triton elementwise kernels is captured into a K-step CUDA graph, plus a per-prompt-length graphed prefill. Output is byte-identical to the eager reference.
- [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (Whisper-style windowed-attention audio encoder + Qwen3 LLM decoder). Three byte-exact CUDA-graph layers: a graphed encoder (custom static-shape windowed-attention kernel, since the stock layer does host ops on `cu_seqlens`), a graphed greedy Qwen3 decode over a static KV cache with fused Triton RMSNorm/SwiGLU/residual/QK-norm kernels, and a K-step (K=8) multi-step decode graph. Output is byte-identical to the eager reference.
- [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) (non-autoregressive ASR). Unlike every other model here, there is **no decode loop**: ASR is one bidirectional forward pass — a CTC conformer encoder produces a rough token draft, blank "edit slots" are interleaved, and a *bidirectional* granite-4.0-1b LLM editor refines the whole sequence in a single forward. The win is CUDA-graph capture of the encoder trunk plus a `torch.compile`d (then graph-captured) LLM editor forward, removing host launch overhead across the 16 encoder + 40 LLM layers. Output is byte-identical to the eager reference.

## Numbers

Single RTX 5090, bf16, model load excluded. RTFx (realtime factor) means
audio_seconds / transcribe_seconds, so 100x means 100 seconds of audio
transcribed in 1 second. Higher is faster. The `stock transformers` column is
the unmodified HuggingFace `generate()` reference.

Both models were benchmarked on the same audio-length tiers (short ~7s, medium
~22s, long ~45-74s), same weights, producing identical transcripts.

### granite-speech-4.1-2b (2.3B params)

B=1 single-stream. `starling` is standard greedy decode. `starling (spec)` adds
self-speculative decoding (drafts tokens from the encoder's CTC head, verifies
them with the LLM). Spec is slower on short audio because the draft extraction
has fixed overhead, but pulls ahead on longer audio where the accepted drafts
save more LLM forward passes.

| audio | starling | starling (spec) | [stock transformers](https://github.com/huggingface/transformers) | [CrispASR](https://github.com/CrispStrobe/CrispASR) |
| ----- | -------- | --------------- | ------------------ | -------- |
| 7s    | 212ms (35x) | 245ms (30x) | 1363ms (5x) | 1185ms (6x) |
| 25s   | 570ms (44x) | 326ms (77x) | 6329ms (4x) | 2290ms (11x) |
| 45s   | 569ms (79x) | 334ms (135x) | 7026ms (6x) | 4060ms (11x) |

### parakeet-tdt-0.6b-v3 (0.6B params)

B=1 is single-stream. B=8 processes 8 clips at
once.

| audio | starling B=1 | starling B=8 | [stock transformers](https://github.com/huggingface/transformers) | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) B=1 | [CrispASR](https://github.com/CrispStrobe/CrispASR) |
| ----- | ------------ | ------------- | ------------------ | -------------------------- | -------- |
| 7s    | 17ms (446x)  | 27ms (2184x)  | 214ms (35x)        | 30ms (251x)               | 580ms (13x) |
| 22s   | 26ms (863x)  | 57ms (3119x)  | 465ms (48x)        | 76ms (294x)               | 1440ms (16x) |
| 74s   | 67ms (1111x) | 174ms (3416x) | 1325ms (56x)       | 223ms (333x)              | 4505ms (16x) |

### moss-transcribe-preview-2b (2B params)

B=1 single-stream. `starling` is byte-identical to the eager `transformers`
reference. Three layers of CUDA-graph capture, all byte-exact:

- **Graphed prefill** — the eager prefill forward is captured per prompt
  length (68ms → 15ms at T=300, ~4.5x).
- **Compiled K-step decode** — the fused decode forward (hand-iterated 28
  Qwen3 layers + Triton RMSNorm/SwiGLU kernels) is `torch.compile`d then
  captured into a K=16-step CUDA graph (argmax-in-graph, 1 sync/chunk).
  `torch.compile` fuses the elementwise glue the hand loop still emits in
  PyTorch, taking decode 4.85 → 2.95ms/tok (338 tok/s).
- **Eager audio encoder** (Qwen3-omni MoE, 32 layers) runs once per utterance
  (~30-60ms, flash-attention backed, not the bottleneck).

Mel extraction (Whisper log-mel, CPU) is excluded from these numbers; it adds
~40-190ms per utterance and is the main remaining opportunity (GPU mel).

| audio | starling | stock transformers |
| ----- | -------- | ------------------ |
| 7s    | 165ms (45x) | ~3300ms (2x) |
| 22s   | 386ms (58x) | ~6400ms (3x) |
| 74s   | 815ms (91x) | ~13500ms (6x) |

### qwen3-asr-1.7b (1.7B params)

`starling` is byte-identical to the eager `transformers` reference. Four
layers of optimization, all producing byte-identical decoded tokens: a graphed
audio encoder (custom static-shape windowed-attention kernel, since the stock
layer does host ops on `cu_seqlens` that break capture), a graphed greedy
Qwen3 decode over a static KV cache with fused Triton elementwise kernels
(RMSNorm, SwiGLU, residual, QK-norm), a `torch.compile`d fused decode forward
(Inductor fuses the RoPE/softmax-prep/GQA glue), and a K-step (K=8) multi-step
decode graph that syncs with the host once per 8 tokens instead of once per
token. Batched mode turns the launch-bound batch=1 GEMVs into saturating GEMMs.

B=1 single-stream.

| audio | starling | [stock transformers](https://github.com/huggingface/transformers) | [CrispASR](https://github.com/CrispStrobe/CrispASR) |
| ----- | -------- | ------------------ | -------- |
| 7s    | 240ms (31x) | 2962ms (2.5x) | 4190ms (1.8x) |
| 22s   | 437ms (51x) | 6030ms (3.7x) | 5445ms (4.1x) |
| 74s   | 1382ms (54x) | 24993ms (3.0x) | 7812ms (9.5x) |

Batched offline throughput (medium ~22s clip tiled B times, same transcript):

| B | starling | RTFx |
| - | -------- | ---- |
| 1 | 825ms | 27x |
| 8 | 2332ms (8 streams) | 76x |
| 16 | 3920ms (16 streams) | 91x |

### granite-speech-4.1-2b-nar (2.3B params)

Non-autoregressive: a single bidirectional forward, no token-by-token decode.
`starling` is byte-identical to the eager `transformers` reference. Two layers
of CUDA-graph capture, both byte-exact:

- **Graphed encoder + projector** — the graph-safe conformer stack (input → 16
  blocks with self-conditioning) *plus* the multilayer-cat + Q-Former projector
  + BPE CTC head are captured together in one graph per mel-frame count. The
  host-side lengths (`audio_lengths`, `bpe_lengths`) are input-shape constants
  for batch=1, so they are computed once at capture time and cached — the hot
  path never does a `.cpu()` sync. (The BPE head's `lengths.tolist()` would
  break capture, but with cached lengths we slice statically.)
- **Compiled + graphed LLM editor** — the `torch.compile`d stock bidirectional
  granite-4.0-1b forward is captured per edit-sequence length. Compiling in
  eager first makes Inductor emit deterministic Triton kernels, so graph
  capture doesn't perturb the numerics the way capturing raw cuBLAS does.

B=1 single-stream. NAR is already extremely fast eager (it does one forward, no
loop), so the speedups are smaller than the autoregressive tracks — the win is
removing host launch overhead across the 16 encoder + 40 LLM layers. The encoder
trunk + projector + BPE head are captured in one graph (the projector alone was
~6ms eager); the host-side lengths are cached per input shape so the hot path
never does a `.cpu()` sync.

| audio | starling | [stock transformers](https://github.com/huggingface/transformers) |
| ----- | -------- | ------------------ |
| 7s    | 15ms (487x) | 87ms (84x) |
| 22s   | 28ms (796x) | 77ms (289x) |
| 74s   | 67ms (1110x) | 103ms (722x) |

### Long audio (30-90 min)

Both models transcribing the same tiled audio at each duration, using their
strongest batched config (found via a B-size sweep). 5 repeats per cell, mean
reported. Granite uses text-level overlap dedup; parakeet uses frame-aligned
TDT-duration stitching. 30s chunks, 2s overlap.

Granite peaks at B=48 (batch sweep winner: B=16 216x, B=32 248x, B=48 275x,
B=64 241x at 60min). Parakeet peaks at B=32 (B=48 is 10% slower, B=64 OOMs).

| model | config | 30 min | 60 min | 90 min | VRAM |
| ----- | ------ | ------ | ------ | ------ | ---- |
| granite-speech-4.1-2b | B=48 | 8.98s (200x) | 13.4s (268x) | 21.3s (253x) | 9.7 GB |
| parakeet-tdt-0.6b-v3 | B=32 | 0.47s (3742x) | 0.95s (3808x) | 1.42s (3817x) | 2.9 GB |

Parakeet steady-state numbers (graph warmup excluded). Parakeet is ~14x faster
than granite on long-form while using 3.3x less VRAM.

### Speculative decoding

Granite's self-speculative path drafts tokens from the encoder's CTC head and
verifies them with the LLM in multi-token forwards. At B=1 it gives a 1.65x
speedup over non-spec greedy (292 vs 177 tok/s). At B>=16 the GEMMs are large
enough that speculation wastes more compute than it saves (measured 0.76x
regression at B=32), so batched decoding always uses the non-spec path.

## Benchmark

A single re-runnable script sweeps every supported model × engine × audio
length × batch size on the same fixtures and the same ground-truth transcript,
so RTFx (speed) and WER (accuracy) are directly comparable across engines.

Engines:

- `starling` — the fused megakernel pipeline in this repo.
- `stock transformers` — the unmodified HuggingFace `generate()` reference.
  (MOSS's HF `generate()` is broken on this transformers build, so its stock
  reference is the documented byte-exact eager greedy loop over the same model
  modules.)
- `CrispASR` — the external ggml binary. Only granite and qwen3 have a CrispASR
  backend; moss and parakeet have no column there.

Fixtures are deterministic: a single LibriSpeech sample (2086-149220-0033)
tiled 1×/3×/10× into `short` (~7s) / `medium` (~22s) / `long` (~74s). RTFx and
VRAM are medians with model load + CUDA-graph capture excluded. `starling` is
byte-exact with `stock transformers`, so the two report the same WER; any drift
is a bug and the bench prints a warning. WER is computed with `jiwer` against
the tiled LibriSpeech transcript, with identical case/punctuation normalization
applied to every engine (so CrispASR's lowercased output is not penalized).

Re-run it any time:

```
uv run python benchmarks/bench_all.py --update-readme
```

That regenerates `outputs/bench_all.json` and splices the two tables below into
this README between the `BENCH:START`/`BENCH:END` sentinels. Scope is tunable
(`--models`, `--engines`, `--lengths`, `--batches`, `--reps`). `qwen3` rows
appear automatically once that model lands on `master`; until then it lives on
the `qwen3-asr` branch.

<!-- BENCH:START -->
**granite-speech-4.1-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   | CrispASR     |
|----------|---------|--------------|----------------------|--------------|
| short    |       1 | 290ms (26x)  | 2497ms (3x)          | 3860ms (2x)  |
| medium   |       1 | 513ms (44x)  | 4112ms (5x)          | 4422ms (5x)  |
| long     |       1 | 1783ms (42x) | 17719ms (4x)         | 10450ms (7x) |

**granite-speech-4.1-2b** — WER % vs LibriSpeech reference

| length   |   batch | starling   | stock transformers   | CrispASR   |
|----------|---------|------------|----------------------|------------|
| short    |       1 | 0.00%      | 0.00%                | 0.00%      |
| medium   |       1 | 33.33%     | 33.33%               | 33.33%     |
| long     |       1 | 21.30%     | 21.30%               | 41.74%     |

**parakeet-tdt-0.6b-v3** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 15ms (493x)  | 249ms (30x)          |
| short    |       8 | 24ms (307x)  | —                    |
| medium   |       1 | 23ms (956x)  | 500ms (45x)          |
| medium   |       8 | 51ms (437x)  | —                    |
| long     |       1 | 56ms (1315x) | 1252ms (59x)         |
| long     |       8 | 163ms (456x) | —                    |

**parakeet-tdt-0.6b-v3** — WER % vs LibriSpeech reference

| length   |   batch | starling   | stock transformers   |
|----------|---------|------------|----------------------|
| short    |       1 | 0.00%      | 0.00%                |
| short    |       8 | 0.00%      | —                    |
| medium   |       1 | 0.00%      | 0.00%                |
| medium   |       8 | 0.00%      | —                    |
| long     |       1 | 0.00%      | 0.00%                |
| long     |       8 | 0.00%      | —                    |

**moss-transcribe-preview-2b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   |
|----------|---------|--------------|----------------------|
| short    |       1 | 187ms (40x)  | 1745ms (4x)          |
| medium   |       1 | 397ms (56x)  | 4648ms (5x)          |
| long     |       1 | 1301ms (57x) | 21151ms (4x)         |

**moss-transcribe-preview-2b** — WER % vs LibriSpeech reference

| length   |   batch | starling   | stock transformers   |
|----------|---------|------------|----------------------|
| short    |       1 | 0.00%      | 0.00%                |
| medium   |       1 | 0.00%      | 0.00%                |
| long     |       1 | 33.48%     | 33.48%               |

**qwen3-asr-1.7b** — latency / RTFx (ms, RTFx×)

| length   |   batch | starling     | stock transformers   | CrispASR     |
|----------|---------|--------------|----------------------|--------------|
| short    |       1 | 238ms (31x)  | 1780ms (4x)          | 3482ms (2x)  |
| medium   |       1 | 399ms (56x)  | 5642ms (4x)          | 4698ms (5x)  |
| long     |       1 | 1201ms (62x) | 18306ms (4x)         | 7863ms (10x) |

**qwen3-asr-1.7b** — WER % vs LibriSpeech reference

| length   |   batch | starling   | stock transformers   | CrispASR   |
|----------|---------|------------|----------------------|------------|
| short    |       1 | 0.00%      | 0.00%                | 0.00%      |
| medium   |       1 | 0.00%      | 0.00%                | 0.00%      |
| long     |       1 | 0.00%      | 0.00%                | 0.00%      |
<!-- BENCH:END -->

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
