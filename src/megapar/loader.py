"""Model + processor loading helpers for Granite-Speech-4.1-2b.

Later phases will use `load_model_and_processor` and `get_components` to get
deterministic, eager-mode references and the three submodules they will replace
with Triton kernels.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from .config import MODEL_ID


def load_model_and_processor(
    attn_impl: str = "eager",
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load the Granite-Speech-4.1-2b model and processor.

    Args:
        attn_impl: Global attention implementation. The q-former (BLIP2) inside
            the projector does NOT support sdpa, so the GOLDEN reference path
            must use ``"eager"``. For the LLM-only sdpa baseline, load eager
            here then call :func:`set_llm_attn_implementation`.
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.

    Returns:
        ``(model, processor)`` with the model in eval mode.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def set_llm_attn_implementation(model: Any, impl: str) -> str:
    """Best-effort switch of the LLM's attention implementation in-place.

    The q-former projector only supports eager, so this targets ONLY the
    Granite LLM decoder. Returns the implementation that actually took effect
    (which may differ from ``impl`` if the LLM rejected it).

    Later phases can use this to produce the realistic "stock-optimized"
    baseline (LLM on sdpa) that they need to beat.
    """
    components = get_components(model)
    llm = components["language_model"]
    cfg = llm.config
    # The transformers Pattern-Runtime stores the resolved impl on the config.
    setattr(cfg, "_attn_implementation", impl)
    # Some transformers builds key internal attention classes off this attribute
    # at module-construction time; to be safe we also flip the per-layer config
    # so the next forward picks it up.
    layers = getattr(llm, "layers", None)
    if layers is not None:
        for layer in layers:
            layer_cfg = getattr(layer, "config", None)
            if layer_cfg is not None:
                setattr(layer_cfg, "_attn_implementation", impl)
            # self_attn may keep a cached implementation
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "_attn_implementation"):
                self_attn._attn_implementation = impl
    return getattr(cfg, "_attn_implementation", impl)


def get_components(model: Any) -> dict[str, Any]:
    """Return the three megapar-relevant submodules.

    The Granite-Speech top-level model is a CausalLM wrapper around an inner
    :class:`GraniteSpeechModel`. The encoder / projector / language_model live
    on that inner model, accessible as ``model.model.<x>`` (also as
    ``model.base_model.<x>``). This helper resolves whichever path works on the
    current transformers version.
    """
    inner = getattr(model, "model", None) or getattr(model, "base_model", None)
    if inner is None:
        # Some versions flatten the wrapper.
        inner = model
    encoder = getattr(inner, "encoder", None) or getattr(model, "encoder", None)
    projector = getattr(inner, "projector", None) or getattr(model, "projector", None)
    language_model = (
        getattr(inner, "language_model", None)
        or getattr(model, "language_model", None)
    )
    if encoder is None or projector is None or language_model is None:
        raise AttributeError(
            f"Could not resolve encoder/projector/language_model on "
            f"{type(model).__name__}; inner={type(inner).__name__}"
        )
    return {
        "encoder": encoder,
        "projector": projector,
        "language_model": language_model,
    }


@contextmanager
def inference_mode() -> Iterator[None]:
    """Thin wrapper around ``torch.inference_mode`` so callers can avoid
    importing torch directly just for the context manager."""
    with torch.inference_mode():
        yield
