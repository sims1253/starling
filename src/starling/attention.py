"""Shared attention dispatch for the fused decode paths.

All five fused decoders (granite / moss / qwen3 / higgs / ark) compute decode
attention the same way:

    scores = Q @ K^T * scale + mask      # (1, n_q, 1, L)
    attn   = softmax(scores, fp32).to(dtype)
    out    = attn @ V                     # (1, n_q, 1, hd)

with the K/V cache repeated across GQA groups first.  That recipe has three
degrees of freedom that the optimisation flags expose:

* **manual** (flag off)  -- the original 4-launch PyTorch path with an
  explicit ``_repeat_kv`` materialisation (a full K/V copy per layer).
* **sdpa math**          -- :func:`scaled_dot_product_attention` with the
  ``MATH`` backend and ``enable_gqa=True``.  Byte-exact with the manual path
  (same fp32 softmax over the same scores) but one launch instead of four
  and no K/V repeat copy.
* **flash / fp8**        -- the flash/efficient backend (optionally casting
  Q/K/V to fp8e4m3 first).  Not byte-exact; gated by ``tolerance_mode``.

This module folds the choice into one helper so each fused decoder stays a
single-line call.  The dispatch key (the active :class:`OptFlags`) is read
once per decode forward, not per layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .flags import OptFlags


def gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
    flags: OptFlags,
) -> torch.Tensor:
    """Compute grouped-query attention for one decode step.

    Args:
        q: ``(1, n_q, 1, hd)`` query.
        k: ``(1, n_kv, L, hd)`` key cache (NOT repeated across groups).
        v: ``(1, n_kv, L, hd)`` value cache (NOT repeated).
        attn_mask: ``(1, 1, 1, L)`` additive mask (0 where attended,
            ``-inf`` where masked).  Broadcasts over heads.
        scale: ``1/sqrt(head_dim)`` (Qwen3 absorbs this into q_proj for some
            ports; pass the effective scale here).
        dtype: working dtype (bfloat16); softmax is always fp32.
        flags: active :class:`OptFlags` selecting the backend.

    Returns:
        ``(1, n_q, 1, hd)`` attention output.

    The ``enable_gqa=True`` path (math / flash / fp8) avoids the
    ``(1, n_kv, L, hd) -> (1, n_q, L, hd)`` repeat materialisation that the
    manual path performs.
    """
    if flags.fp8_attention:
        # fp8 Q@K^T and attn@V with fp32 accumulation.  Keep softmax in fp32;
        # only the two matmuls drop to fp8e4m3.  SDPA's flash backend handles
        # the fp8 path natively on Blackwell (sm_120+).
        with torch.nn.attention.sdpa_kernel([
            _sdpa_backend("FLASH_ATTENTION"), _sdpa_backend("EFFICIENT_ATTENTION")
        ]):
            q_f = q.to(torch.float8_e4m3fn)
            k_f = k.to(torch.float8_e4m3fn)
            v_f = v.to(torch.float8_e4m3fn)
            return F.scaled_dot_product_attention(
                q_f, k_f, v_f, attn_mask=attn_mask, is_causal=False,
                scale=scale, enable_gqa=True,
            ).to(dtype)

    if flags.flash_attention:
        # Flash / efficient backend.  Not byte-exact (fp32 register-tiled
        # softmax rather than materialised), but typically no argmax flips.
        with torch.nn.attention.sdpa_kernel([
            _sdpa_backend("FLASH_ATTENTION"),
            _sdpa_backend("EFFICIENT_ATTENTION"),
            _sdpa_backend("MATH"),
        ]):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False,
                scale=scale, enable_gqa=True,
            ).to(dtype)

    if flags.sdpa_attention:
        # Math backend: fp32 softmax over the full score matrix, identical to
        # the reference recipe.  enable_gqa avoids the K/V repeat copy.
        with torch.nn.attention.sdpa_kernel([_sdpa_backend("MATH")]):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False,
                scale=scale, enable_gqa=True,
            ).to(dtype)

    # ---- manual fallback: the original 4-launch PyTorch recipe ------------
    n_kv_groups = q.shape[1] // k.shape[1]
    k_r = _repeat_kv(k, n_kv_groups)
    v_r = _repeat_kv(v, n_kv_groups)
    scores = torch.matmul(q, k_r.transpose(2, 3)) * scale
    scores = scores + attn_mask
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
    return torch.matmul(attn, v_r)


def _sdpa_backend(name: str):
    """Resolve an SDPBackend by name, tolerant to torch version drift."""
    from torch.backends.cuda import SDPBackend
    return getattr(SDPBackend, name)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA repeat (manual fallback only).  (B, n_kv, S, D) -> (B, n_q, S, D)."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(
        B, n_kv * n_rep, S, D
    )


__all__ = ["gqa_attention"]
