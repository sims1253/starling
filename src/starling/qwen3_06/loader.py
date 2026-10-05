"""Model + processor loading helpers for Qwen3-ASR-0.6B.

Thin wrapper over the 1.7B track's parameterized loader: the 0.6B ships the
same transformers-native layout (``Qwen3ASRForConditionalGeneration`` with
``model.audio_tower`` / ``model.multi_modal_projector`` /
``model.language_model``), so ``get_components`` is reused directly and only
the hub repo id differs.
"""

from __future__ import annotations

from typing import Any

import torch

from ..qwen3.loader import get_components, load_model_and_processor as _load
from .config import MODEL_ID, MODEL_REVISION

__all__ = ["MODEL_ID", "MODEL_REVISION", "get_components", "load_model_and_processor"]


def load_model_and_processor(
    attn_impl: str = "eager",
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    model_id: str | None = None,
    revision: str | None = None,
) -> tuple[Any, Any]:
    """Load the Qwen3-ASR-0.6B model and processor.

    Args:
        attn_impl: Attention implementation for the Qwen3 text decoder
            (``"eager"`` is the byte-exact reference).
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        model_id: HF hub repo id override (defaults to the 0.6B MODEL_ID).
        revision: HF hub revision override (defaults to the pinned 0.6B
            MODEL_REVISION so golden captures stay on the converted
            checkpoint).

    Returns:
        ``(model, processor)`` with the model in eval mode.
    """
    return _load(
        attn_impl=attn_impl, dtype=dtype, device=device,
        model_id=model_id or MODEL_ID,
        revision=MODEL_REVISION if revision is None else revision,
    )
