"""Model + processor loading helpers for AutoArk-AI/ARK-ASR-3B.

``load_model_and_processor`` returns the eager-mode reference model + processor,
and ``get_components`` resolves the four submodules the megakernel pipeline wires
together: the Whisper+MLP audio adapter, the Qwen2.5 decoder trunk, the tied
lm_head, and the decoder's token embeddings.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import MODEL_ID


def load_model_and_processor(
    attn_impl: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    model_id: str | None = None,
) -> tuple[Any, Any]:
    """Load an ARK-ASR model and processor.

    The model ships custom modeling code, so ``trust_remote_code=True`` is
    required for both the model and the processor. The decoder is loaded with
    ``attn_implementation="eager"`` so the StaticCache + 4D-mask decode path
    matches the golden reference exactly (sdpa would fuse away the mask plumbing
    the CUDA-graph decoder relies on).

    Args:
        attn_impl: Attention implementation for the Qwen2.5 decoder.
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        model_id: HF hub repo id. Defaults to the 3B MODEL_ID; the 0.6B track
            passes its own id (same remote modeling code, same submodule layout).

    Returns:
        ``(model, processor)`` with the model in eval mode.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    resolved = model_id or MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(resolved, trust_remote_code=True)
    return model, processor


def get_components(model: Any) -> dict[str, Any]:
    """Return the four starling-relevant submodules of the ARK-ASR-3B model.

    The top-level model exposes:
      * ``model.audio_encoder`` -- the ``AudioMLPAdapter`` (Whisper + MLP).
      * ``model.model``         -- the ``Qwen2Model`` decoder trunk.
      * ``model.lm_head``       -- the ``nn.Linear`` lm_head (tied to embeddings).
      * ``model.model.embed_tokens`` -- the decoder token embeddings.

    Returns:
        Dict with keys ``"audio_encoder"``, ``"language_model"``, ``"lm_head"``,
        ``"embed_tokens"``.
    """
    audio_encoder = getattr(model, "audio_encoder", None)
    language_model = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if (
        audio_encoder is None
        or language_model is None
        or lm_head is None
        or embed_tokens is None
    ):
        raise AttributeError(
            f"Could not resolve audio_encoder/language_model/lm_head/embed_tokens "
            f"on {type(model).__name__}"
        )
    return {
        "audio_encoder": audio_encoder,
        "language_model": language_model,
        "lm_head": lm_head,
        "embed_tokens": embed_tokens,
    }
