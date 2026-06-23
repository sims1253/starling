# starling

CUDA-graph inference kernels for two speech-recognition models, tuned to run as
fast as possible on a single RTX 5090 (Blackwell, sm_120). This is not a serving
framework or a drop-in replacement for anything. It is the opposite of general
purpose: model-specific, hardware-specific, and only useful if you have the same
GPU and care about single-machine throughput.

The core idea is the same for both models. The stock `transformers` decode loop
emits a few hundred tiny kernels per output token and spends most of its wall
time on CPU launch overhead, with the GPU sitting around 10% busy. Everything
that can be captured into a CUDA-graph replay gets captured: decode steps, fused
RMSNorm/SwiGLU, the attention mask, and (for the LLM model) whole multi-step
token loops. Output is byte-identical to the eager `transformers` reference, so
there is no accuracy trade-off, just fewer round trips to the GPU.

## Models

Both do speech-to-text.

- `ibm-granite/granite-speech-4.1-2b` (encoder + 1B LLM decoder). The LLM decode
  is the bottleneck. Includes an optional self-speculative path that drafts
  tokens from the encoder's CTC head.
- `nvidia/parakeet-tdt-0.6b-v3` (FastConformer + TDT transducer, no LLM). Tuned
  for batched offline throughput, with GPU-side mel extraction and chunking for
  hour-long audio.

## Numbers

Single RTX 5090, bf16, model load excluded, best-of-N, transcripts byte-identical
to `transformers`.

| model                  | mode                       | vs. stock | realtime       |
| ---------------------- | -------------------------- | --------- | -------------- |
| granite-speech-4.1-2b  | single stream              | ~11x      | ~45x RTF       |
| granite-speech-4.1-2b  | single stream, speculative | ~17x      | ~70x RTF       |
| granite-speech-4.1-2b  | long audio, batched (B=16) | ~4x vs sequential chunking | ~124x RTF @ 5min, ~174x @ 10min |
| parakeet-tdt-0.6b-v3   | batch 8, offline           | ~10x      | ~3100x RTF     |
| parakeet-tdt-0.6b-v3   | 1h audio, chunked          | n/a       | ~293x RTF, ~1.5 GB VRAM |

For context against other engines on the same parakeet weights, this is roughly
1.4-4.4x faster than parakeet.cpp and 31-67x faster than CrispASR, with
identical transcripts.

On the granite-speech-4.1-2b model, starling is ~7x faster than CrispASR (the
ggml/whisper.cpp engine) on identical weights.

For long audio on granite-speech, batching the chunked decode (B=16) gives
about 4x over the sequential chunking path (~124x RTF at 5 min). The encoder
stays per-stream (byte-exact); only the LLM decode is batched.

## What did not work

Kept here because they are the more interesting findings:

- **INT8 weight-only quant** is slower, not faster. Decode is launch-bound, not
  bandwidth-bound, so halving weight traffic does not help and the dequant
  overhead hurts. Kept behind a flag.
- **FP8 `_scaled_mm`** is also slower, for the same reason.
- **`torch.compile` on the encoder** is not byte-exact: inductor upcasts
  attention to fp32 and the conformer's BatchNorm (running variance around 4e-10)
  amplifies the difference ~316x per block.

## Requirements

- Tuned on an RTX 5090 (Blackwell, sm_120). Runs on any Ampere+ NVIDIA GPU
  (RTX 30/40/50, A100, H100); bf16 required. The torch wheels are pinned to the
  CUDA 13.0 (cu130) index in `pyproject.toml`; the default PyPI torch wheel is
  cu12 / sm_90 and will not run on Blackwell.
- CUDA 13.0, Python 3.10-3.12, and [uv](https://github.com/astral-sh/uv).

## Layout

```
src/starling/            granite-speech megakernel
  encoder_mega.py       fused (cudagraph) conformer encoder
  llm_mega.py           graphed greedy decode over a static KV cache
  multistep.py          K-step graphed decode (multi-step per replay)
  pipeline.py           encoder + projector + LLM wiring
  batched.py            batched (B>1) LLM decode + pipeline
  long_audio.py         chunked long-audio transcription (sequential + batched)
  speculative.py        self-speculative decoding via the CTC draft head
  parakeet/             parakeet-tdt megakernel
    decode_mega.py      multi-step graphed TDT decode
    encoder_graph.py    graphed FastConformer encoder
    mel_gpu.py          GPU-side mel filterbank
    chunking.py         bounded-VRAM long-audio chunking
benchmarks/             RTF and cross-engine benchmarks
scripts/                bench and probe scripts
tests/                  correctness checks vs. golden references
```
