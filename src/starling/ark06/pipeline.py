"""End-to-end fused ASR megakernel pipeline for Audio8/ARK-ASR-0.6B.

Thin subclass of the 3B track's :class:`starling.ark.pipeline.MegaPipeline`:
the encoder dims, adapter layout, token ids, and audio front-end are identical,
and the fused LLM decoder derives head_dim / layer count / head counts from the
loaded tensors, so only model loading differs (the 0.6B hub id).

Public API
----------
``MegaPipeline(model, processor, *, steps_per_replay=None, max_cache_len=4096)``
``MegaPipeline.from_pretrained(...)``
``MegaPipeline.transcribe(audio_path_or_array, instruction=..., max_new_tokens=200) -> (text, token_ids)``
"""

from __future__ import annotations

from typing import Optional

import torch

from ..ark.audio import build_inputs_embeds, build_prompt_ids, extract_mel, read_wav
from ..ark.pipeline import MegaPipeline as _ArkMegaPipeline
from .loader import load_model_and_processor

__all__ = [
    "MegaPipeline",
    "build_inputs_embeds",
    "build_prompt_ids",
    "extract_mel",
    "read_wav",
]


class MegaPipeline(_ArkMegaPipeline):
    """End-to-end fused ARK-ASR-0.6B pipeline (encoder + fused LLM).

    Inherits transcribe / prewarm / bucketing / graph-mode toggles unchanged
    from the 3B track; only :meth:`from_pretrained` points at the 0.6B loader.
    """

    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder_mode: str = "cudagraph",
        steps_per_replay: Optional[int] = None,
        max_cache_len: int = 4096,
        attn_impl: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        shape_bucketing: bool = True,
        mel_bucket_frames: int = 512,
        prefill_use_graph: bool = False,
    ) -> "MegaPipeline":
        """Load the 0.6B model + processor and wrap them in a MegaPipeline."""
        model, processor = load_model_and_processor(
            attn_impl=attn_impl, dtype=dtype, device=device
        )
        return cls(
            model,
            processor,
            encoder_mode=encoder_mode,
            steps_per_replay=steps_per_replay,
            max_cache_len=max_cache_len,
            shape_bucketing=shape_bucketing,
            mel_bucket_frames=mel_bucket_frames,
            prefill_use_graph=prefill_use_graph,
        )
