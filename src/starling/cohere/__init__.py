"""Starling megakernel pipeline for CohereLabs/cohere-transcribe-03-2026.

Parakeet FastConformer encoder (48 layers) + 8-layer Transformer decoder with
self-attention AND cross-attention (the repo's first seq2seq encoder-decoder,
Whisper-style). The per-token autoregressive decode loop is the launch-bound
bottleneck, so it is captured into a CUDA-graph (K steps per replay), driving
the model's own decoder layers over an ``EncoderDecoderCache`` with
precomputed 4D causal + bidirectional masks.
"""

from __future__ import annotations
