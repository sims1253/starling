"""Load S1-mini (a plain ``Qwen3ForCausalLM``) + its tokenizer."""

from __future__ import annotations

from typing import Any

import torch

from .config import MODEL_ID


def load_model_and_tokenizer(
    *,
    attn_impl: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load S1-mini bf16 eager on ``device`` (repo convention: eager attn for
    byte-exactness against the stock golden path)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def get_components(model: Any) -> dict[str, Any]:
    """Split the causal LM into the trunk + lm_head the mega decoder wants."""
    return {
        "language_model": model.model,
        "lm_head": model.lm_head,
        "embed_tokens": model.model.embed_tokens,
    }
