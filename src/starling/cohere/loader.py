"""Model + processor loading helpers for CohereLabs/cohere-transcribe-03-2026.

Unlike moss/higgs/ark, cohere-transcribe is **natively registered** in the
shared ``transformers`` (``transformers.models.cohere_asr``), so there is no
vendored modeling, no isolated venv, no transformers bump — it loads directly.

Public API
----------
``load_model_and_processor(dtype, device, attn_impl) -> (model, processor)``
``get_components(model) -> dict``  (encoder / decoder / proj_out)
"""

from __future__ import annotations

from typing import Any

import torch

from .config import MODEL_ID


def load_model_and_processor(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_impl: str = "sdpa",
) -> tuple[Any, Any]:
    """Load cohere-transcribe + its processor.

    Args:
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        attn_impl: attention implementation. ``"sdpa"`` is the byte-exact path
            used by the golden; the decoder attention is small enough that SDPA
            and eager agree at the decoded-token level.
    """
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def get_components(model: Any) -> dict[str, Any]:
    """Resolve the starling-relevant submodules of a ``CohereAsrForConditionalGeneration``.

    * ``encoder``  -- ``model.model.encoder`` (Parakeet FastConformer, 48 layers)
    * ``decoder``  -- ``model.model.decoder`` (CohereAsrDecoder, 8 layers)
    * ``proj_out`` -- the (untied) LM head mapping decoder hidden -> vocab logits
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise AttributeError(f"no .model on {type(model).__name__}")
    return {
        "encoder": inner.encoder,
        "decoder": inner.decoder,
        "proj_out": model.proj_out,
    }
