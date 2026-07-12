"""starling.audex — megakernel components for nvidia/Nemotron-Labs-Audex-2B (ASR).

High-performance inference pipeline for Audex-2B ASR: a CUDA-graphed
Whisper-style audio encoder (Qwen2AudioEncoder with avg-pooler), a graphed
greedy Nemotron-Dense LLM decoder over a static KV cache (relu2 MLP, GQA,
RoPE), and chunked long-audio transcription. Byte-identical to the eager
``transformers`` reference.

The decoder is a Nemotron-Dense 2B (RMSNorm, GQA, relu2 MLP, RoPE, untied
embeddings) — structurally similar to the Qwen3 decoder but with a squared-ReLU
MLP instead of SwiGLU and no QK-norm.

Shared architecture constants live in :mod:`starling.audex.config`.
"""

from __future__ import annotations
