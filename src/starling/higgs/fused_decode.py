"""Fused Triton elementwise decode path for the Higgs-Audio Qwen3 decoder.

Subclasses :class:`starling.higgs.llm_mega.LLMMega` and overrides
:meth:`_decode_step_eager` with a hand-iterated Qwen3 layer loop that replaces
the memory-bound elementwise glue (RMSNorm, SwiGLU, residual add, QK-norm) with
single-launch Triton kernels reusing ``starling.granite.llm_kernels``.

Qwen3 specifics vs granite:
* **No embedding multiplier**, **no logit scaling** (Qwen3 has neither).
* **QK-norm**: ``q_norm`` / ``k_norm`` (per-head RMSNorm) applied to the reshaped
  Q/K *before* RoPE. We fuse these via :func:`fused_rmsnorm` over the head_dim.
* **Residual is plain ``x + y``** (alpha = 1.0; Qwen3 has no ``residual_multiplier``).
* RoPE stays in PyTorch (matching the reference's bf16 arithmetic exactly;
  granite found Triton RoPE diverges on large Q/K magnitudes).

All GEMMs (q/k/v/o_proj, gate/up/down_proj, lm_head) and the attention
softmax/matmul stay as stock cuBLAS / PyTorch ops -- only the elementwise glue
is fused.  fp32 internal accumulation matches the reference; the decoded
transcript is byte-identical to the golden oracle.
"""

from __future__ import annotations

from typing import Any

import torch

from . import llm_kernels as _k
from .llm_mega import LLMMega


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: repeat KV heads to match Q heads. x (B, n_kv, S, D) -> (B, n_q, S, D)."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(B, n_kv * n_rep, S, D)


class FusedLLMMega(LLMMega):
    """CUDA-graph-captured greedy decoder with **fused Triton elementwise kernels**.

    Inherits all graph-capture / generate / bench machinery from :class:`LLMMega`
    and overrides only :meth:`_decode_step_eager` with a custom Qwen3 forward.
    """

    def __init__(self, *args, compile_decode: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._k = _k
        # Pre-extract per-layer references + Qwen3 dims for the hot decode loop.
        cfg = self.text_config
        self._n_q_heads = int(cfg.num_attention_heads)
        self._n_kv_heads = int(cfg.num_key_value_heads)
        self._head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self._n_q_heads))
        self._n_kv_groups = self._n_q_heads // self._n_kv_heads
        self._rms_eps = float(cfg.rms_norm_eps)
        self._compile_decode = bool(compile_decode)
        if compile_decode:
            # Wrap the fused decode forward in inductor. ``max-autotune-no-cudagraphs``
            # fuses the PyTorch elementwise glue the hand loop still emits (RoPE
            # cat+mul+add, attention softmax prep, GQA repeats) while leaving
            # cudagraph capture to us. Byte-exact for the LLM decode (granite's
            # "compile not byte-exact" finding was the *encoder*'s BatchNorm, not
            # the LLM). Method-assign (not a self-calling wrapper) to avoid dynamo
            # recursion. Credit: Instance D (moss) validated this on the same
            # Qwen3-decode pattern.
            self._decode_step_eager = torch.compile(  # type: ignore[method-assign]
                self._decode_step_eager, mode="max-autotune-no-cudagraphs", dynamic=False
            )

    def _decode_step_eager(self) -> None:
        """Custom single-token Qwen3 decode forward with fused Triton kernels.

        Replicates ``_forward_core`` + ``Qwen3DecoderLayer.forward`` exactly but
        replaces the elementwise glue with fused kernels. Writes the final logits
        into ``self.static_logits``. No embedding multiplier / no logit scaling.
        """
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads

        # (1) embedding lookup (NO multiplier for Qwen3)
        hidden = self._embed(self.static_input_ids)  # (1, 1, 2048)

        # (2) rotary cos/sin for this position (computed once, shared by layers)
        cos, sin = self._rotary(hidden, self.static_position_ids)
        cos4 = cos.unsqueeze(1)  # (1, 1, 1, hd) for broadcast with (B, H, 1, hd)
        sin4 = sin.unsqueeze(1)

        # (3) iterate the 28 Qwen3 decoder layers
        for idx, layer in enumerate(self._layers):
            sa = layer.self_attn
            mlp = layer.mlp

            # --- attention block ---
            residual = hidden

            # fused input RMSNorm
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)

            # Q/K/V projections (cuBLAS bf16 GEMM) -> reshape to heads
            q = sa.q_proj(normed).view(1, 1, n_q, hd)   # (1, 1, n_q, hd) pre-norm
            kv = sa.k_proj(normed).view(1, 1, n_kv, hd)
            v = sa.v_proj(normed).view(1, 1, n_kv, hd)

            # QK-norm (per-head RMSNorm over head_dim) -- fused.
            # q_norm weight shape is (hd,); apply per (head) row.
            q = k.fused_rmsnorm(q, sa.q_norm.weight, self._rms_eps)
            kv = k.fused_rmsnorm(kv, sa.k_norm.weight, self._rms_eps)

            # -> (1, n_heads, 1, hd)
            q = q.transpose(1, 2)
            kv = kv.transpose(1, 2)
            v = v.transpose(1, 2)

            # RoPE (PyTorch, matching the reference's bf16 arithmetic exactly)
            half = hd // 2
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            # cache update (in-place on static-address K/V tensors at cache_position)
            kv_full, v_full = self.cache.update(
                kv, v, idx, {"cache_position": self.static_cache_position}
            )

            # GQA: repeat KV heads to match Q heads
            kv_r = _repeat_kv(kv_full, self._n_kv_groups)  # (1, n_q, max_len, hd)
            v_r = _repeat_kv(v_full, self._n_kv_groups)

            # attention: scores = Q @ K^T * scale + mask, softmax, @ V
            # Qwen3 attention scaling = 1/sqrt(head_dim); Q proj already absorbed
            # it in the reference (q_proj * scaling in Qwen3Attention), so here we
            # apply it to the scores to match.
            scale = 1.0 / (hd ** 0.5)
            scores = torch.matmul(q, kv_r.transpose(2, 3)) * scale  # (1, n_q, 1, max_len)
            scores = scores + self.static_attn_mask  # broadcast (1,1,1,max_len)
            attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
            attn_out = torch.matmul(attn, v_r)  # (1, n_q, 1, hd)

            # reshape + output projection
            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            attn_out = sa.o_proj(attn_out)

            # fused residual add (alpha = 1.0 for Qwen3)
            hidden = k.fused_residual_scale(residual, attn_out, 1.0)

            # --- MLP block ---
            residual = hidden

            # fused post-attention RMSNorm
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)

            # gate/up projections (cuBLAS bf16 GEMM)
            gate = mlp.gate_proj(normed)
            up = mlp.up_proj(normed)

            # fused SwiGLU: silu(gate) * up
            act = k.fused_silu_mul(gate, up)

            # down projection (cuBLAS bf16 GEMM)
            mlp_out = mlp.down_proj(act)

            # fused residual add
            hidden = k.fused_residual_scale(residual, mlp_out, 1.0)

        # (4) final fused RMSNorm + text lm_head (NO logit scaling)
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self._lm_head(hidden)
        self.static_logits.copy_(logits)
