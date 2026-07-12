"""Model + tokenizer + feature-extractor loading helpers for Audex-2B.

``load_model_and_processor`` returns the eager reference (model + tokenizer +
Whisper feature extractor). ``get_components`` resolves the submodules the
megakernel pipeline wires together: the Qwen2AudioEncoder, the projector, the
NemotronDense decoder trunk, the separate lm_head, and the token embeddings.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import MODEL_ID, MODEL_SUBFOLDER


def _resolve_model_path() -> str:
    """Resolve the local path to ``checkpoint_folder_full``.

    Prefers a local snapshot under ``.hf-cache/``; falls back to the HF hub id
    (``from_pretrained`` will download / cache on first use).
    """
    from .config import REPO_ROOT

    local = REPO_ROOT / ".hf-cache" / "audex-2b" / MODEL_SUBFOLDER
    if local.exists():
        return str(local)
    return MODEL_ID


def load_model_and_processor(
    attn_impl: str = "eager",
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[Any, Any, Any]:
    """Load the Audex-2B model, tokenizer, and Whisper feature extractor.

    The checkpoint is fp32 (~5.8 GB); ``dtype`` controls the on-device cast
    (bf16 is the starling convention).

    Args:
        attn_impl: Attention implementation for the NemotronDense decoder.
            ``"eager"`` is the byte-exact reference (matches the golden capture
            and the StaticCache + 4D-mask decode path).
        dtype: Model dtype (bf16).
        device: Target device.

    Returns:
        ``(model, tokenizer, feature_extractor)`` with the model in eval mode.
    """
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    model_path = _resolve_model_path()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    )
    # The Whisper feature extractor lives under audio_preprocessor/.
    preprocessor_path = config.audio_preprocessor_path or "audio_preprocessor"
    from pathlib import Path

    fe_path = Path(preprocessor_path)
    if not fe_path.is_absolute():
        fe_path = Path(model_path) / preprocessor_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(fe_path))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    model.eval()
    return model, tokenizer, feature_extractor


def get_components(model: Any) -> dict[str, Any]:
    """Return the starling-relevant submodules of the loaded Audex model.

    ``NemotronDenseAudexForConditionalGeneration`` (subclass of
    ``NemotronDenseForCausalLM``) exposes:
      * ``model.audio_encoder``  -- the ``Qwen2AudioEncoder`` (Whisper + avg-pooler).
      * ``model.audio_projector`` -- the ``NemotronDenseAudexProjector``.
      * ``model.model``          -- the ``NemotronDenseModel`` decoder trunk.
      * ``model.lm_head``        -- ``nn.Linear`` (untied from embeddings).
      * ``model.model.embed_tokens`` -- token embeddings.

    Returns:
        Dict with keys ``"encoder"``, ``"projector"``, ``"language_model"``,
        ``"lm_head"``, ``"embed_tokens"``.
    """
    encoder = getattr(model, "audio_encoder", None)
    projector = getattr(model, "audio_projector", None)
    language_model = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if any(
        v is None
        for v in (encoder, projector, language_model, lm_head, embed_tokens)
    ):
        raise AttributeError(
            f"Could not resolve encoder/projector/language_model/lm_head/"
            f"embed_tokens on {type(model).__name__}"
        )
    return {
        "encoder": encoder,
        "projector": projector,
        "language_model": language_model,
        "lm_head": lm_head,
        "embed_tokens": embed_tokens,
    }
