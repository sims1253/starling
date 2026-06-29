"""Fused decode path for the MOSS Qwen3 LLM with Triton elementwise kernels.

Subclasses :class:`starling.moss.llm_mega.MossLLMMega` and overrides
:meth:`_decode_step_eager` with a hand-iterated layer loop that replaces the
small elementwise glue (RMSNorm, SwiGLU silu*mul, residual add) with
single-launch Triton kernels (:mod:`starling.moss.llm_kernels`).  GEMMs stay as
cuBLAS bf16 matmuls; the attention softmax/matmul and RoPE stay as stock
PyTorch ops (matching the reference's bf16 arithmetic).

This mirrors ``starling.granite.llm_mega.FusedLLMMega``.  Byte-exactness is
re-verified against the golden transcript in tests.

Qwen3-specific notes
--------------------
* Two RMSNorms per layer (input_layernorm, post_attention_layernorm) PLUS
  per-head ``q_norm`` / ``k_norm`` (RMSNorm over head_dim on the reshaped
  Q/K).  All four use the fused kernel.
* GQA: 16 query heads, 8 KV heads (n_kv_groups = 2).
* Residuals are plain ``x + y`` (no Granite residual_multiplier).
* RoPE is applied to Q and K via the model's ``rotary_emb`` module
  (``position_embeddings``); kept in PyTorch because Triton bf16 multiply
  rounds differently than ATen for large Q values (granite finding).
"""

from __future__ import annotations

from typing import Any

import torch

from .llm_mega import MossLLMMega


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: repeat KV heads to match Q heads.  (B, n_kv, S, D) -> (B, n_q, S, D)."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(
        B, n_kv * n_rep, S, D
    )


class FusedMossLLMMega(MossLLMMega):
    """CUDA-graph greedy decoder with fused Triton elementwise kernels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from . import llm_kernels as _k

        self._k = _k
        self._layers = list(self.lm.layers)
        self._embed = self.lm.embed_tokens
        self._final_norm = self.lm.norm
        self._rotary = self.lm.rotary_emb
        cfg = self.config
        self._n_q_heads = int(cfg.num_attention_heads)
        self._n_kv_heads = int(cfg.num_key_value_heads)
        self._head_dim = int(cfg.head_dim)
        self._n_kv_groups = self._n_q_heads // self._n_kv_heads
        self._attn_scale = float(self._head_dim ** -0.5)
        self._rms_eps = float(cfg.rms_norm_eps)

    def _decode_step_eager(self) -> None:
        """Custom single-token decode forward with fused Triton kernels."""
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads

        # (1) embedding lookup
        hidden = self._embed(self.static_input_ids)  # (1, 1, 2048)

        # (2) rotary cos/sin for this position
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        cos4 = cos.unsqueeze(1)  # (1, 1, 1, hd)
        sin4 = sin.unsqueeze(1)

        # (3) iterate layers
        for idx, layer in enumerate(self._layers):
            sa = layer.self_attn
            mlp = layer.mlp

            # --- attention block ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)

            # Q/K/V projections (cuBLAS bf16 GEMM)
            q = sa.q_proj(normed).view(1, 1, n_q, hd).transpose(1, 2)   # (1,n_q,1,hd)
            kv = sa.k_proj(normed).view(1, 1, n_kv, hd).transpose(1, 2)
            v = sa.v_proj(normed).view(1, 1, n_kv, hd).transpose(1, 2)

            # per-head Q/K RMSNorm (Qwen3 q_norm / k_norm) -- fused, over head_dim
            q = k.fused_rmsnorm(q, sa.q_norm.weight, self._rms_eps)
            kv = k.fused_rmsnorm(kv, sa.k_norm.weight, self._rms_eps)

            # RoPE (PyTorch -- matches the reference's bf16 arithmetic exactly)
            half = hd // 2
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            # cache update (in-place on static-address K/V tensors)
            kv, v = self.cache.update(kv, v, idx)

            # GQA repeat
            kv_r = _repeat_kv(kv, self._n_kv_groups)
            v_r = _repeat_kv(v, self._n_kv_groups)

            # attention
            scores = torch.matmul(q, kv_r.transpose(2, 3)) * self._attn_scale
            scores = scores + self.static_attn_mask
            attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
                self.dtype
            )
            attn_out = torch.matmul(attn, v_r)  # (1, n_q, 1, hd)

            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            attn_out = sa.o_proj(attn_out)

            hidden = k.fused_residual(residual, attn_out)

            # --- MLP block ---
            residual = hidden
            normed = k.fused_rmsnorm(
                hidden, layer.post_attention_layernorm.weight, self._rms_eps
            )
            gate = mlp.gate_proj(normed)
            up = mlp.up_proj(normed)
            act = k.fused_silu_mul(gate, up)
            mlp_out = mlp.down_proj(act)
            hidden = k.fused_residual(residual, mlp_out)

        # (4) final fused RMSNorm
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)

        # (5) lm_head
        logits = self.lm_head(hidden)
        self.static_logits.copy_(logits)
