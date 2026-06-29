"""Starling megakernel pipeline for OpenMOSS-Team/MOSS-Transcribe-preview-2B.

Audio encoder (Qwen3-omni MoE, 32L) + gated-MLP adapter + Qwen3 LLM decoder
(28L, GQA).  Same encoder+LLM-decoder pattern as granite: the LLM decode loop
is the bottleneck (launch-bound, ~10% GPU busy), so everything that can be
captured into a CUDA-graph replay gets captured.
"""

from __future__ import annotations
