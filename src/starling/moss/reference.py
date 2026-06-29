"""Reference (stock-style) eager greedy decoder for MOSS-Transcribe.

This is the byte-exact golden path.  It runs the audio encoder + adapter with
the model's own modules, merges audio embeddings exactly as
``MossModel.forward`` does, then greedy-decodes with the Qwen3 ``language_model``
over a ``transformers.StaticCache`` using a precomputed 4D attention mask (the
same trick as ``starling.granite.llm_mega``: a 4D mask makes
``create_causal_mask`` early-exit so no CPU scalar is allocated during capture).

The decoded token sequence from here IS the golden reference the megakernel
must reproduce bit-for-bit.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import LLM_EOS_TOKEN_ID


def build_inputs_embeds(
    model: Any,
    input_ids: torch.Tensor,
    audio_features: torch.Tensor,
    audio_input_mask: torch.Tensor,
) -> torch.Tensor:
    """Merge audio features into the LLM token embeddings.

    Replicates ``MossModel.forward`` (the audio branch) exactly:
      1. inputs_embeds = embed_tokens(input_ids)
      2. audio_embeds = audio_adapter(audio_features)
      3. masked_scatter_ audio_embeds into the audio slots.
    """
    comps = _comps(model)
    inputs_embeds = comps["embed_tokens"](input_ids)
    audio_embeds = comps["audio_adapter"](audio_features)
    mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(mask_expanded, audio_embeds)


def audio_features(model: Any, audio_data: torch.Tensor, seqlens: torch.Tensor) -> torch.Tensor:
    """Run the stock audio encoder -> last_hidden_state (the adapter input)."""
    comps = _comps(model)
    return comps["audio_model"](input_features=audio_data, feature_lens=seqlens).last_hidden_state


def greedy_generate(
    model: Any,
    inputs_embeds: torch.Tensor,
    *,
    max_new_tokens: int = 200,
    eos_token_id: int = LLM_EOS_TOKEN_ID,
    max_cache_len: int = 1024,
) -> torch.Tensor:
    """Eager greedy decode over a StaticCache.  Returns (1, n_new) int64 on CPU."""
    from transformers.cache_utils import StaticCache

    comps = _comps(model)
    lm = comps["language_model"]
    lm_head = comps["lm_head"]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    cfg = lm.config

    T = inputs_embeds.shape[1]
    assert T + max_new_tokens <= max_cache_len, f"cache overflow: {T}+{max_new_tokens}>{max_cache_len}"

    cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
    neg = torch.finfo(dtype).min
    ar = torch.arange(max_cache_len, device=device)

    # ---- prefill ----
    pos = torch.arange(T, device=device).unsqueeze(0)
    cp = torch.arange(T, device=device)
    # 4D causal mask (1,1,T,max_cache_len): key j valid iff j <= query i.
    q = torch.arange(T, device=device).unsqueeze(1)
    mask4 = torch.where(ar[None, None, None, :] <= q[None, None, :, :], 0.0, neg).to(dtype)
    out = lm(
        inputs_embeds=inputs_embeds,
        attention_mask=mask4,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        cache_position=cp,
    )
    last_hidden = out.last_hidden_state[:, -1:, :]
    next_token = lm_head(last_hidden).argmax(dim=-1)  # (1,1)
    gen = [int(next_token.item())]

    static_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
    for i in range(max_new_tokens - 1):
        cur = T + i
        static_ids.copy_(next_token)
        pos = torch.tensor([[cur]], device=device)
        cp = torch.tensor([cur], device=device)
        mask4 = torch.where(ar[None, None, None, :] <= cur, 0.0, neg).to(dtype)
        out = lm(
            input_ids=static_ids,
            attention_mask=mask4,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            cache_position=cp,
        )
        last_hidden = out.last_hidden_state[:, -1:, :]
        next_token = lm_head(last_hidden).argmax(dim=-1)
        gen.append(int(next_token.item()))
        if int(next_token.item()) == eos_token_id:
            break
    return torch.tensor(gen, dtype=torch.int64).unsqueeze(0)


def _comps(model: Any) -> dict[str, Any]:
    """Cached component lookup (re-resolve is cheap; kept simple)."""
    inner = model.model
    return {
        "audio_model": inner.audio_model,
        "audio_adapter": inner.audio_adapter,
        "language_model": inner.language_model,
        "lm_head": model.lm_head,
        "embed_tokens": inner.language_model.get_input_embeddings(),
    }
