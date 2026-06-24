"""starling.granite — megakernel components for ibm-granite/granite-speech-4.1-2b.

High-performance inference pipeline for the granite-speech-4.1-2b model: a fused
CUDA-graphed conformer encoder, a graphed greedy LLM decoder over a static KV
cache (single-step, K-step multi-step, and batched), chunked long-audio
transcription, and optional self-speculative decoding via the encoder's CTC head.

Shared architecture constants live in :mod:`starling.config` and the runtime
optimisation flags in :mod:`starling.flags` (both top-level).
"""
