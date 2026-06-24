# Starling

CUDA-graph inference kernels for speech-recognition models, tuned to run as
fast as possible on a single RTX 5090 (Blackwell, sm_120). Can probably be 
adapted to other GPUs pretty easy.

The core idea is the same for both models. The stock `transformers` decode loop
emits a few hundred tiny kernels per output token and spends most of its wall
time on CPU launch overhead, with the GPU sitting around 10% busy. Everything
that can be captured into a CUDA-graph replay gets captured: decode steps, fused
RMSNorm/SwiGLU, the attention mask, and (for the LLM model) whole multi-step
token loops. Output is byte-identical to the eager `transformers` reference, so
there is no accuracy trade-off, just fewer round trips to the GPU.

Most props go to GLM 5.2

## Models

Both do speech-to-text.

- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) (encoder + 1B LLM decoder). The LLM decode is the bottleneck. Includes an optional self-speculative path that drafts tokens from the encoder's CTC head.
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer + TDT transducer, no LLM). Tuned for batched offline throughput, with GPU-side mel extraction and chunking for hour-long audio.

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
benchmarks/             RTF and cross-engine benchmarks
scripts/                bench and probe scripts
tests/                  correctness checks vs. golden references
```
