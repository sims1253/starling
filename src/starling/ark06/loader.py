"""Model + processor loading helpers for Audio8/ARK-ASR-0.6B.

Thin wrapper over the 3B track's parameterized loader: the 0.6B ships
byte-identical remote modeling code with the same submodule layout
(``model.audio_encoder`` / ``model.model`` / ``model.lm_head`` /
``model.model.embed_tokens``), so ``get_components`` is reused directly and
only the hub repo id differs.
"""

from __future__ import annotations

from typing import Any

import torch

from ..ark.loader import get_components, load_model_and_processor as _load
from .config import MODEL_ID

__all__ = ["MODEL_ID", "get_components", "load_model_and_processor"]


def load_model_and_processor(
    attn_impl: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    model_id: str | None = None,
) -> tuple[Any, Any]:
    """Load the ARK-ASR-0.6B model and processor.

    Args:
        attn_impl: Attention implementation for the Qwen2.5-0.5B-class decoder.
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        model_id: HF hub repo id override (defaults to the 0.6B MODEL_ID).

    Returns:
        ``(model, processor)`` with the model in eval mode.
    """
    return _load(attn_impl=attn_impl, dtype=dtype, device=device, model_id=model_id or MODEL_ID)
