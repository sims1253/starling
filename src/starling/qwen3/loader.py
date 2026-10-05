"""Model + processor loading helpers for Qwen3-ASR-1.7B.

``load_model_and_processor`` returns the eager reference; ``get_components``
resolves the three submodules (audio_tower / multi_modal_projector /
language_model) that the megakernel will accelerate.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import MODEL_ID


def load_model_and_processor(
    attn_impl: str = "eager",
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    model_id: str | None = None,
) -> tuple[Any, Any]:
    """Load a Qwen3-ASR model and processor.

    Args:
        attn_impl: Attention implementation. ``"eager"`` is the byte-exact
            reference (matches the golden capture). The audio tower uses its own
            windowed eager attention regardless; this flag controls the Qwen3
            text decoder.
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        model_id: HF hub repo id. Defaults to the 1.7B MODEL_ID; the 0.6B
            track passes its own id (same processor/tokenizer/template, only
            architecture dims differ).

    Returns:
        ``(model, processor)`` with the model in eval mode.
    """
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    repo = model_id or MODEL_ID
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(repo)
    return model, processor


def get_components(model: Any) -> dict[str, Any]:
    """Return the three starling-relevant submodules.

    ``Qwen3ASRForConditionalGeneration`` wraps ``Qwen3ASRModel`` (on
    ``model.model``), which holds ``audio_tower`` (the encoder),
    ``multi_modal_projector``, and ``language_model`` (the Qwen3 decoder).
    ``lm_head`` lives on the top-level model and is tied to the decoder's
    ``embed_tokens``.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise AttributeError(f"No inner Qwen3ASRModel on {type(model).__name__}")
    encoder = getattr(inner, "audio_tower", None)
    projector = getattr(inner, "multi_modal_projector", None)
    language_model = getattr(inner, "language_model", None)
    if encoder is None or projector is None or language_model is None:
        raise AttributeError(
            f"Could not resolve audio_tower/multi_modal_projector/language_model on "
            f"{type(inner).__name__}"
        )
    return {
        "encoder": encoder,
        "projector": projector,
        "language_model": language_model,
    }
