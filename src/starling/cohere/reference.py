"""Reference (stock-style) eager greedy decoder for cohere-transcribe-03-2026.

This is the byte-exact golden path. It runs the Parakeet encoder with the
model's own module, then greedy-decodes with the ``CohereAsrDecoder`` over an
``EncoderDecoderCache`` (a self-attention ``DynamicCache`` that grows one slot
per step + a cross-attention ``DynamicCache`` that is filled once at prefill and
reused every step), using a precomputed 4D causal mask (self-attn) and a 4D
bidirectional mask (cross-attn).

The decoded token sequence from here IS the golden reference the megakernel must
reproduce bit-for-bit. It is verified byte-exact against ``model.generate()``
(see ``tests/test_cohere.py::test_reference_matches_generate``).

Why precomputed 4D masks (same family of trick as ``starling.granite.llm_mega``):
``create_causal_mask`` / ``create_bidirectional_mask`` allocate CPU scalars and
do host-side shape branching that aborts CUDA-graph capture. Passing a ready 4D
additive mask makes both early-exit, so the whole decoder forward becomes
capture-safe (used by the megakernel in ``decode_mega.py``).
"""

from __future__ import annotations

from typing import Any

import torch

from .config import EOS_TOKEN_ID


def encode(
    model: Any, input_features: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the stock Parakeet encoder.

    Returns ``(encoder_hidden_states (B,S,1280), bidirectional_mask (B,1,1,S))``.
    For a single unpadded utterance the cross-attn mask is all-zero (additive).
    """
    comps = _comps(model)
    enc = comps["encoder"](input_features=input_features, attention_mask=attention_mask)
    enc_h = enc.last_hidden_state
    B, S, _ = enc_h.shape
    # bidirectional: all keys valid. (B,1,1,S) additive 0 mask.
    enc_mask = torch.zeros(B, 1, 1, S, device=enc_h.device, dtype=enc_h.dtype)
    return enc_h, enc_mask


def greedy_generate(
    model: Any,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 300,
    eos_token_id: int = EOS_TOKEN_ID,
    max_cache_len: int = 1024,
) -> torch.Tensor:
    """Eager greedy decode over an EncoderDecoderCache.

    Handles batched input (B>=1) with per-element EOS: finished elements feed
    ``pad_token_id`` and their (ignored) KV slots keep growing lock-step with
    the batch (the cache is batch-major and shared across the batch, so all
    elements advance one step per iteration regardless of individual finish).

    Args:
        encoder_hidden_states: ``(B, S, 1280)`` from :func:`encode`.
        encoder_attention_mask: ``(B, 1, 1, S)`` additive bidirectional mask.
        decoder_input_ids: ``(B, T_prompt)`` chat-format prompt from the processor.

    Returns ``(B, n_gen)`` int64 on CPU — the *generated* tokens (prompt
    excluded). Each row ends with ``eos_token_id``; finished-early rows are
    padded with ``pad_token_id`` so all rows share the same length.
    """
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    from .config import PAD_TOKEN_ID

    comps = _comps(model)
    decoder = comps["decoder"]
    proj_out = comps["proj_out"]
    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype

    B, T = decoder_input_ids.shape
    assert T + max_new_tokens <= max_cache_len, (
        f"cache overflow: {T}+{max_new_tokens}>{max_cache_len}"
    )

    cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
    neg = torch.finfo(dtype).min

    # ---- prefill ----
    pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    q = torch.arange(T, device=device).unsqueeze(1)
    cmask = torch.where(q >= torch.arange(T, device=device), 0.0, neg).to(dtype)
    cmask = cmask.view(1, 1, T, T).expand(B, 1, T, T)
    out = decoder(
        input_ids=decoder_input_ids,
        attention_mask=cmask,
        position_ids=pos,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    last_hidden = out.last_hidden_state[:, -1:, :]
    next_token = proj_out(last_hidden).argmax(dim=-1)  # (B,1)
    gen = [next_token.squeeze(1)]
    finished = (next_token.squeeze(1) == eos_token_id)

    static_ids = torch.zeros((B, 1), dtype=torch.int64, device=device)
    for i in range(max_new_tokens - 1):
        cur = T + i
        nkeys = T + i + 1
        # finished elements feed pad (KV ignored downstream); others feed their token
        static_ids.copy_(torch.where(finished.unsqueeze(1), PAD_TOKEN_ID, next_token))
        pos1 = torch.full((B, 1), cur, device=device, dtype=torch.long)
        keys_ar = torch.arange(nkeys, device=device)
        cmask1 = torch.where(keys_ar <= cur, 0.0, neg).to(dtype).view(1, 1, 1, nkeys)
        out = decoder(
            input_ids=static_ids,
            attention_mask=cmask1,
            position_ids=pos1,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        last_hidden = out.last_hidden_state[:, -1:, :]
        next_token = proj_out(last_hidden).argmax(dim=-1)
        gen.append(next_token.squeeze(1))
        finished = finished | (next_token.squeeze(1) == eos_token_id)
        if bool(finished.all()):
            break
    return torch.stack(gen, dim=1).cpu()  # (B, n_gen)


def _comps(model: Any) -> dict[str, Any]:
    inner = model.model
    return {
        "encoder": inner.encoder,
        "decoder": inner.decoder,
        "proj_out": model.proj_out,
    }
