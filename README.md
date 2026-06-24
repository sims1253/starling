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

## Models

Both do speech-to-text.

- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) (encoder + 1B LLM decoder). The LLM decode is the bottleneck. Includes an optional self-speculative path that drafts tokens from the encoder's CTC head.
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer + TDT transducer, no LLM). Tuned for batched offline throughput, with GPU-side mel extraction and chunking for hour-long audio.

## Numbers

Single RTX 5090, bf16, model load excluded. RTFx (realtime factor) means
audio_seconds / transcribe_seconds, so 100x means 100 seconds of audio
transcribed in 1 second. Higher is faster. Every RTFx number is absolute
(audio seconds per second of compute, not a speedup over another engine); the
`stock transformers` column is the unmodified HuggingFace `generate()` reference
the others replace.

Both models were benchmarked on the same audio-length tiers (short ~7s, medium
~22s, long ~45-74s), same weights, producing identical transcripts.

### granite-speech-4.1-2b (2.3B params)

B=1 single-stream. "starling" is standard greedy decode. "starling (spec)" adds
self-speculative decoding (drafts tokens from the encoder's CTC head, verifies
them with the LLM). Spec is slower on short audio because the draft extraction
has fixed overhead, but pulls ahead on longer audio where the accepted drafts
save more LLM forward passes. "stock transformers" is the unmodified HuggingFace
eager `model.generate()` path (no CUDA graphs), the slow reference these kernels
replace.

| audio | starling | starling (spec) | [stock transformers](https://github.com/huggingface/transformers) | [CrispASR](https://github.com/CrispStrobe/CrispASR) |
| ----- | -------- | --------------- | ------------------ | -------- |
| 7s    | 212ms (35x) | 245ms (30x) | 1363ms (5x) | 1185ms (6x) |
| 25s   | 570ms (44x) | 326ms (77x) | 6329ms (4x) | 2290ms (11x) |
| 45s   | 569ms (79x) | 334ms (135x) | 7026ms (6x) | 4060ms (11x) |

### parakeet-tdt-0.6b-v3 (0.6B params)

B=1 is single-stream latency (one clip at a time). B=8 processes 8 clips at
once: total time goes up, but throughput goes up much more because the GPU does
8x the work in only ~1.6x the time. "stock transformers" is the unmodified
HuggingFace `AutoModelForTDT.generate()` path (CPU mel extraction + eager
encoder + stock TDT decode).

| audio | starling B=1 | starling B=8 | [stock transformers](https://github.com/huggingface/transformers) | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) B=1 | [CrispASR](https://github.com/CrispStrobe/CrispASR) |
| ----- | ------------ | ------------- | ------------------ | -------------------------- | -------- |
| 7s    | 17ms (446x)  | 27ms (2184x)  | 214ms (35x)        | 30ms (251x)               | 580ms (13x) |
| 22s   | 26ms (863x)  | 57ms (3119x)  | 465ms (48x)        | 76ms (294x)               | 1440ms (16x) |
| 74s   | 67ms (1111x) | 174ms (3416x) | 1325ms (56x)       | 223ms (333x)              | 4505ms (16x) |

### Long audio

For audio longer than the KV cache (granite) or encoder attention window
(parakeet), both models support chunked transcription. On granite-speech,
batching the chunked decode (B=16) is about 4x faster than sequential chunking
(~124x RTFx on 5 min audio, ~174x on 10 min). On parakeet, 1h of audio
transcribes at ~293x RTFx using ~1.5 GB VRAM via bounded-VRAM chunking.

## What did not work

- INT8 weight-only quant is slower. Decode is launch-bound, not bandwidth-bound, so halving weight traffic does not help.
- FP8 `_scaled_mm` is also slower, for the same reason.
- `torch.compile` on the encoder is not byte-exact: inductor upcasts attention to fp32 and the conformer's BatchNorm amplifies the difference.

## Requirements

- Tuned on an RTX 5090 (Blackwell, sm_120). Runs on any Ampere+ NVIDIA GPU
  (RTX 30/40/50, A100, H100); bf16 required. The torch wheels are pinned to the
  CUDA 13.0 (cu130) index in `pyproject.toml`; the default PyPI torch wheel is
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
