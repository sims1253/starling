"""End-to-end fused ASR megakernel pipeline for Qwen3-ASR-0.6B.

Thin subclass of the 1.7B track's :class:`starling.qwen3.pipeline.MegaPipeline`:
the mel frontend, chat-template prompt layout, windowed-attention encoder
topology, MLP projector and Qwen3 trunk are identical, and every dim the
pipelines use is derived from the loaded model / tensors — so only model
loading differs (the 0.6B hub id). The 0.6B's 2048-wide attention over a
1024-wide trunk needs no special handling: the fused decoders read
``head_dim`` / head counts from the model config (never
``hidden_size // num_heads``).

Public API (fully inherited)
--------------------------------
``MegaPipeline(model, processor, *, max_cache_len=4096, use_fused_llm=True, ...)``
``MegaPipeline.from_pretrained(...)`` (inherited; selects the 0.6B loader via
``_load_model_and_processor`` and accepts an optional ``model_id`` override)
``MegaPipeline.transcribe(input_features, input_ids, mask=None, max_new_tokens=200) -> (text, ids)``
"""

from __future__ import annotations

from ..qwen3.pipeline import MegaPipeline as _Qwen3MegaPipeline
from .loader import load_model_and_processor

__all__ = ["MegaPipeline", "load_model_and_processor"]


class MegaPipeline(_Qwen3MegaPipeline):
    """End-to-end fused Qwen3-ASR-0.6B pipeline (encoder + fused LLM).

    Inherits EVERYTHING from the 1.7B track — transcribe / prewarm /
    graph-mode toggles / chunk decode budgets and :meth:`from_pretrained`
    (including its optional ``model_id`` override and nullable ``dtype``) —
    with only the model loader swapped, so the inherited entry point cannot
    drift from the parent's implementation.
    """

    #: The 0.6B loader (defaults to the Qwen3-ASR-0.6B-hf hub id) used by
    #: the inherited from_pretrained.
    _load_model_and_processor = staticmethod(load_model_and_processor)
