"""End-to-end fused ASR megakernel pipeline for Qwen3-ASR-0.6B.

Thin subclass of the 1.7B track's :class:`starling.qwen3.pipeline.MegaPipeline`:
the mel frontend, chat-template prompt layout, windowed-attention encoder
topology, MLP projector and Qwen3 trunk are identical, and every dim the
pipelines use is derived from the loaded model / tensors — so only model
loading differs (the 0.6B hub id). The 0.6B's 2048-wide attention over a
1024-wide trunk needs no special handling: the fused decoders read
``head_dim`` / head counts from the model config (never
``hidden_size // num_heads``).

Public API (inherited unless noted)
--------------------------------
``MegaPipeline(model, processor, *, max_cache_len=4096, use_fused_llm=True, ...)``
``MegaPipeline.from_pretrained(...)`` (overridden here to point at the 0.6B
loader; accepts an optional ``model_id`` override and a nullable ``dtype``)
``MegaPipeline.transcribe(input_features, input_ids, mask=None, max_new_tokens=200) -> (text, ids)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..qwen3.pipeline import MegaPipeline as _Qwen3MegaPipeline
from .loader import load_model_and_processor

if TYPE_CHECKING:
    import torch

__all__ = ["MegaPipeline", "load_model_and_processor"]


class MegaPipeline(_Qwen3MegaPipeline):
    """End-to-end fused Qwen3-ASR-0.6B pipeline (encoder + fused LLM).

    Inherits transcribe / prewarm / graph-mode toggles / chunk decode budgets
    unchanged from the 1.7B track; only :meth:`from_pretrained` points at the
    0.6B loader.
    """

    @classmethod
    def from_pretrained(
        cls,
        *,
        attn_impl: str = "eager",
        dtype: "torch.dtype | None" = None,
        device: str = "cuda",
        max_cache_len: int = 4096,
        use_fused_llm: bool = True,
        steps_per_replay: int | None = None,
        encoder_mode: str = "cudagraph",
        prefill_use_graph: bool = False,
        model_id: str | None = None,
    ) -> "MegaPipeline":
        import torch

        dt = torch.bfloat16 if dtype is None else dtype  # noqa: RUF046
        model, processor = load_model_and_processor(
            attn_impl=attn_impl,
            dtype=dt,
            device=device,
            model_id=model_id,
        )
        return cls(
            model,
            processor,
            max_cache_len=max_cache_len,
            use_fused_llm=use_fused_llm,
            steps_per_replay=steps_per_replay,
            encoder_mode=encoder_mode,
            prefill_use_graph=prefill_use_graph,
        )
