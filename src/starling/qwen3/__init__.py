"""starling.qwen3 — megakernel components for Qwen/Qwen3-ASR-1.7B.

High-performance inference pipeline for Qwen3-ASR: a fused CUDA-graphed
windowed-attention audio encoder, a graphed greedy Qwen3 LLM decoder over a
static KV cache (single-step + K-step multi-step + batched), and chunked
long-audio transcription. Byte-identical to the eager ``transformers``
reference.

The decoder is a stock Qwen3 LLM (RMSNorm, GQA, SwiGLU, RoPE, tied
embeddings) — structurally the same as the granite decoder, so the graphed
decode + fused Triton elementwise kernels follow the granite design.

Shared architecture constants live in :mod:`starling.qwen3.config`.
"""

# Ensure the shared transformers install exposes qwen3_asr BEFORE any caller
# does `from transformers import Qwen3ASRForConditionalGeneration`. Importing
# this package triggers the restore; run it eagerly here so a bare
# `import starling.qwen3` is enough. (See _tf_bootstrap for why this is needed.)
from . import _tf_bootstrap as _tf_bootstrap  # noqa: F401

