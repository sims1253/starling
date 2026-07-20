"""Model + processor loading helpers for MOSS-Transcribe-preview-2B.

The model ships custom ``modeling_Moss.py`` / ``processing_Moss.py`` via
``auto_map``.  Rather than bumping the shared ``transformers`` (which would
affect the granite / parakeet / qwen3 worktrees), we *vendor* the remote code
under ``starling/moss/vendor/`` and import it directly.

Public API
----------
``load_model_and_processor(dtype, device) -> (model, processor)``
``get_components(model) -> dict``  (audio_model / audio_adapter / language_model / lm_head)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .config import MODEL_ID

_VENDOR = Path(__file__).resolve().parent / "vendor"


def _ensure_vendor_on_path() -> None:
    """Make the vendored ``modeling_Moss`` / ``processing_Moss`` importable."""
    p = str(_VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model_and_processor(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_impl: str = "eager",
) -> tuple[Any, Any]:
    """Load MOSS-Transcribe + its processor.

    Args:
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        attn_impl: attention implementation for BOTH the audio encoder and the
            LLM. ``"eager"`` is the byte-exact golden path.
    """
    _ensure_vendor_on_path()
    from modeling_Moss import MossForCausalLM  # type: ignore[import-not-found]
    from processing_Moss import MossProcessor, MelConfig  # type: ignore[import-not-found]
    from transformers import AutoTokenizer

    model = MossForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    # from_pretrained(attn_implementation=...) only reaches the top-level
    # config; the LLM/audio sub-configs keep their default ("sdpa"), which
    # silently puts the "eager" golden path on the mem-efficient SDPA kernel
    # (bf16-nondeterministic across attention kernels). Propagate explicitly.
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = attn_impl
    model.eval()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    template = _VENDOR / "chat_template_default.py"
    proc = MossProcessor(tok, config=MelConfig(mel_dim=128), template_path=str(template))
    return model, proc


def get_components(model: Any) -> dict[str, Any]:
    """Resolve the starling-relevant submodules.

    ``MossForCausalLM`` has ``self.model`` (a :class:`MossModel`) holding
    ``audio_model``, ``audio_adapter`` and ``language_model``; the ``lm_head``
    lives on the top-level CausalLM (tied to the LLM embed_tokens).
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise AttributeError(f"no .model on {type(model).__name__}")
    return {
        "audio_model": inner.audio_model,
        "audio_adapter": inner.audio_adapter,
        "language_model": inner.language_model,
        "lm_head": model.lm_head,
        "embed_tokens": inner.language_model.get_input_embeddings(),
    }
