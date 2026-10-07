"""starling.qwen3_06 — megakernel components for Qwen3-ASR-0.6B-hf.

Smaller sibling of the Qwen3-ASR-1.7B track (windowed-attention conv encoder +
2-layer MLP projector + Qwen3 decoder), reusing the 1.7B track's pipelines and
kernels. The hub checkpoint ships a byte-identical tokenizer, chat template,
processor config and generation config — only the architecture dims below
differ.
"""
