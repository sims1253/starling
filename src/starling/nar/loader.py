"""Model + processor loading for Granite-Speech-4.1-2b-NAR.

The NAR model ships as ``trust_remote_code`` (no ``granite_speech_nar`` module
in the installed transformers yet). It loads fine under transformers
5.13.0.dev0 with SDPA attention (no flash-attn package required) — verified
end-to-end on the fixture tiers.
"""

from __future__ import annotations

from typing import Any

import torch

from . import config as cfg


def load_model_and_processor(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any]:
    """Load the Granite-Speech-4.1-2b-NAR model + processor.

    Args:
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        attn_implementation: Attention backend. ``"sdpa"`` is the verified
            byte-exact default (the model is bidirectional, ``is_causal=False``).
            ``"flash_attention_2"`` is also supported by the remote code but the
            flash-attn package is not installed in the shared venv.

    Returns:
        ``(model, processor)`` with the model on ``device`` in eval mode.
    """
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.MODEL_ID,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor


def get_components(model: Any) -> dict[str, Any]:
    """Return the three NAR submodules: encoder, projector, language_model."""
    encoder = getattr(model, "encoder", None)
    projector = getattr(model, "projector", None)
    language_model = getattr(model, "language_model", None)
    if encoder is None or projector is None or language_model is None:
        raise AttributeError(
            f"Could not resolve encoder/projector/language_model on "
            f"{type(model).__name__}"
        )
    return {
        "encoder": encoder,
        "projector": projector,
        "language_model": language_model,
    }
