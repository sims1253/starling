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

from typing import Optional

import torch

from ..attention import gqa_attention as _gqa_attention
from ..flags import get_default_flags
from .._kernels._compile import torch_compile
from . import llm_kernels as _k
from .llm_mega import LLMMega


class FusedLLMMega(LLMMega):
    """CUDA-graph-captured greedy decoder with **fused Triton elementwise kernels**.

    Inherits all graph-capture / generate / bench machinery from :class:`LLMMega`
    and overrides only :meth:`_decode_step_eager` with a custom Qwen3 forward.
    """

    def __init__(self, *args, compile_decode: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._k = _k
        self._gqa_attention = _gqa_attention
        # Pre-extract per-layer references + Qwen3 dims for the hot decode loop.
        cfg = self.text_config
        self._n_q_heads = int(cfg.num_attention_heads)
        self._n_kv_heads = int(cfg.num_key_value_heads)
        self._head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self._n_q_heads))
        self._n_kv_groups = self._n_q_heads // self._n_kv_heads
        self._rms_eps = float(cfg.rms_norm_eps)
        self._intermediate = int(cfg.intermediate_size)
        self._attn_scale = float(self._head_dim ** -0.5)
        self._compile_decode = bool(compile_decode)
        self._flags = get_default_flags()
        self._fused_weights: Optional[list[dict]] = None
        if self._flags.fused_qkv:
            self._fused_weights = self._fuse_layer_weights()
        if compile_decode:
            # Wrap the fused decode forward in inductor. ``max-autotune-no-cudagraphs``
            # fuses the PyTorch elementwise glue the hand loop still emits (RoPE
            # cat+mul+add, attention softmax prep, GQA repeats) while leaving
            # cudagraph capture to us. Byte-exact for the LLM decode (granite's
            # "compile not byte-exact" finding was the *encoder*'s BatchNorm, not
            # the LLM). Method-assign (not a self-calling wrapper) to avoid dynamo
            # recursion. Credit: Instance D (moss) validated this on the same
            # Qwen3-decode pattern.
            self._decode_step_eager = torch_compile(  # type: ignore[method-assign]
                self._decode_step_eager, mode="max-autotune-no-cudagraphs", dynamic=False
            )

    def _fuse_layer_weights(self) -> list[dict]:
        """Pre-concatenate QKV and gate/up weights per layer (byte-exact)."""
        fused = []
        for layer in self._layers:
            sa = layer.self_attn
            mlp = layer.mlp
            qkv_w = torch.cat(
                [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], dim=0
            )
            gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
            fused.append({
                "qkv_w": qkv_w.contiguous(),
                "gu_w": gu_w.contiguous(),
                "o_proj": sa.o_proj,
                "down_proj": mlp.down_proj,
            })
        return fused

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
        half = hd // 2
        qkv_split = [n_q * hd, n_kv * hd, n_kv * hd]
        inter = self._intermediate
        flags = self._flags
        fused = self._fused_weights

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

            # Q/K/V projections: fused GEMM (byte-exact) or the model's own.
            if fused is not None:
                f = fused[idx]
                x2 = normed.view(1, -1)
                qkv = torch.nn.functional.linear(x2, f["qkv_w"], None).view(-1)
                q, kv, v = qkv.split(qkv_split, dim=0)
                q = q.view(1, 1, n_q, hd)
                kv = kv.view(1, 1, n_kv, hd)
                v = v.view(1, 1, n_kv, hd)
                o_proj = f["o_proj"]
            else:
                q = sa.q_proj(normed).view(1, 1, n_q, hd)
                kv = sa.k_proj(normed).view(1, 1, n_kv, hd)
                v = sa.v_proj(normed).view(1, 1, n_kv, hd)
                o_proj = sa.o_proj

            # QK-norm (per-head RMSNorm over head_dim) -- fused.
            q = k.fused_rmsnorm(q, sa.q_norm.weight, self._rms_eps)
            kv = k.fused_rmsnorm(kv, sa.k_norm.weight, self._rms_eps)

            # -> (1, n_heads, 1, hd)
            q = q.transpose(1, 2)
            kv = kv.transpose(1, 2)
            v = v.transpose(1, 2)

            # RoPE (PyTorch, matching the reference's bf16 arithmetic exactly)
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            # cache update (in-place on static-address K/V tensors at cache_position)
            kv_full, v_full = self.cache.update(
                kv, v, idx, {"cache_position": self.static_cache_position}
            )

            # attention (SDPA math + enable_gqa, or manual reference path).
            # Qwen3 attention scaling = 1/sqrt(head_dim); Q proj already absorbed
            # it in the reference (q_proj * scaling), so apply it via scale here.
            attn_out = self._gqa_attention(
                q, kv_full, v_full, self.static_attn_mask, self._attn_scale,
                self.dtype, flags,
            )  # (1, n_q, 1, hd)

            # reshape + output projection
            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            attn_out = o_proj(attn_out)

            # fused residual add (alpha = 1.0 for Qwen3)
            hidden = k.fused_residual_scale(residual, attn_out, 1.0)

            # --- MLP block ---
            residual = hidden

            # fused post-attention RMSNorm
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)

            # gate/up projections: fused GEMM (byte-exact) or the model's own.
            if fused is not None:
                x3 = normed.view(1, -1)
                gu = torch.nn.functional.linear(x3, f["gu_w"], None).view(-1)
                gate, up = gu.split([inter, inter], dim=0)
                gate = gate.view(1, 1, inter)
                up = up.view(1, 1, inter)
                down_proj = f["down_proj"]
            else:
                gate = mlp.gate_proj(normed)
                up = mlp.up_proj(normed)
                down_proj = mlp.down_proj

            # fused SwiGLU: silu(gate) * up
            act = k.fused_silu_mul(gate, up)

            # down projection (cuBLAS bf16 GEMM)
            mlp_out = down_proj(act)

            # fused residual add
            hidden = k.fused_residual_scale(residual, mlp_out, 1.0)

        # (4) final fused RMSNorm + text lm_head (NO logit scaling)
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self._lm_head(hidden)
        self.static_logits.copy_(logits)
