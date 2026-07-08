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

from typing import Any, Optional

import torch

from ..attention import gqa_attention as _gqa_attention
from ..flags import get_default_flags
from .llm_mega import MossLLMMega


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: repeat KV heads to match Q heads.  (B, n_kv, S, D) -> (B, n_q, S, D).

    Kept for the manual (non-SDPA) attention fallback in
    :func:`starling.attention.gqa_attention`.
    """
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(
        B, n_kv * n_rep, S, D
    )


class FusedMossLLMMega(MossLLMMega):
    """CUDA-graph greedy decoder with fused Triton elementwise kernels."""

    def __init__(self, *args, fused_rope: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from . import llm_kernels as _k

        self._k = _k
        self._gqa_attention = _gqa_attention
        self._fused_rope = bool(fused_rope)
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
        self._intermediate = int(cfg.intermediate_size)
        self._flags = get_default_flags()
        self._fused_weights: Optional[list[dict]] = None
        if self._flags.fused_qkv:
            self._fused_weights = self._fuse_layer_weights()

        # fp8 weight-only quantization of the per-layer GEMMs (lm_head stays
        # bf16).  Pre-quantize once; the decode step reads these instead of the
        # bf16 fused weights.  See starling.moss.fp8.
        self._fp8_weights: Optional[list[dict]] = None
        if self._flags.fp8_weights:
            self._fp8_weights = self._quantize_layer_weights_fp8()

    def _quantize_layer_weights_fp8(self) -> list[dict]:
        """Pre-quantize the fused qkv/gate-up + o/down weights to fp8e4m3."""
        from .fp8 import quantize_weight_e4m3

        assert self._fused_weights is not None, "fp8_weights requires fused_qkv"
        out = []
        for f in self._fused_weights:
            out.append({
                "qkv": quantize_weight_e4m3(f["qkv_w"]),
                "gu": quantize_weight_e4m3(f["gu_w"]),
                "o": quantize_weight_e4m3(f["o_proj"].weight),
                "down": quantize_weight_e4m3(f["down_proj"].weight),
            })
        return out

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
        """Custom single-token decode forward with fused Triton kernels."""
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        half = hd // 2
        qkv_split = [n_q * hd, n_kv * hd, n_kv * hd]
        inter = self._intermediate
        flags = self._flags
        fused = self._fused_weights
        fp8w = self._fp8_weights
        if fp8w is not None:
            from .fp8 import fp8_linear

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

            # Q/K/V projections: fp8 GEMM, fused bf16 GEMM (byte-exact), or the
            # model's own per-proj linears.
            if fp8w is not None:
                f = fused[idx]
                qf = fp8w[idx]
                x2 = normed.view(1, -1)
                qkv = fp8_linear(x2, qf["qkv"][0], qf["qkv"][1]).view(-1)
                q, kv, v = qkv.split(qkv_split, dim=0)
                q = q.view(1, 1, n_q, hd).transpose(1, 2)
                kv = kv.view(1, 1, n_kv, hd).transpose(1, 2)
                v = v.view(1, 1, n_kv, hd).transpose(1, 2)
                o_proj = f["o_proj"]
            elif fused is not None:
                f = fused[idx]
                x2 = normed.view(1, -1)
                qkv = torch.nn.functional.linear(x2, f["qkv_w"], None).view(-1)
                q, kv, v = qkv.split(qkv_split, dim=0)
                q = q.view(1, 1, n_q, hd).transpose(1, 2)
                kv = kv.view(1, 1, n_kv, hd).transpose(1, 2)
                v = v.view(1, 1, n_kv, hd).transpose(1, 2)
                o_proj = f["o_proj"]
            else:
                q = sa.q_proj(normed).view(1, 1, n_q, hd).transpose(1, 2)
                kv = sa.k_proj(normed).view(1, 1, n_kv, hd).transpose(1, 2)
                v = sa.v_proj(normed).view(1, 1, n_kv, hd).transpose(1, 2)
                o_proj = sa.o_proj

            # per-head Q/K RMSNorm (Qwen3 q_norm / k_norm) -- fused, over head_dim
            q = k.fused_rmsnorm(q, sa.q_norm.weight, self._rms_eps)
            kv = k.fused_rmsnorm(kv, sa.k_norm.weight, self._rms_eps)

            # RoPE.  The fused Triton kernel (default) applies rotary embedding to
            # Q and K in one launch, replacing ~8 PyTorch ops/layer (cat + 2 mul +
            # add, twice for Q and K).  On synthetic clamp-extreme inputs the
            # Triton bf16 product rounding diverges slightly from ATen for the
            # large post-k_norm K values (±400, max-abs diff ~1.0), but on the real
            # model this does NOT compound to an argmax flip over the full decode
            # (byte-exact verified on short 31-tok + medium 89-tok).  Pass
            # ``fused_rope=False`` for the bit-exact PyTorch RoPE path.
            if self._fused_rope:
                q, kv = k.fused_rope(q, kv, cos4, sin4)
            else:
                q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
                kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
                q = q * cos4 + q_rot * sin4
                kv = kv * cos4 + kv_rot * sin4

            # cache update (in-place on static-address K/V tensors)
            kv, v = self.cache.update(kv, v, idx)

            # attention (SDPA math + enable_gqa, or manual reference path).
            attn_out = self._gqa_attention(
                q, kv, v, self.static_attn_mask, self._attn_scale, self.dtype, flags
            )  # (1, n_q, 1, hd)

            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            if fp8w is not None:
                qf = fp8w[idx]
                attn_out = fp8_linear(
                    attn_out.view(1, -1), qf["o"][0], qf["o"][1]
                ).view(1, 1, -1)
            else:
                attn_out = o_proj(attn_out)

            hidden = k.fused_residual(residual, attn_out)

            # --- MLP block ---
            residual = hidden
            normed = k.fused_rmsnorm(
                hidden, layer.post_attention_layernorm.weight, self._rms_eps
            )
            if fp8w is not None:
                qf = fp8w[idx]
                x3 = normed.view(1, -1)
                gu = fp8_linear(x3, qf["gu"][0], qf["gu"][1]).view(-1)
                gate, up = gu.split([inter, inter], dim=0)
                gate = gate.view(1, 1, inter)
                up = up.view(1, 1, inter)
            elif fused is not None:
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
            act = k.fused_silu_mul(gate, up)
            if fp8w is not None:
                mlp_out = fp8_linear(
                    act.view(1, -1), qf["down"][0], qf["down"][1]
                ).view(1, 1, -1)
            else:
                mlp_out = down_proj(act)
            hidden = k.fused_residual(residual, mlp_out)

        # (4) final fused RMSNorm
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)

        # (5) lm_head
        logits = self.lm_head(hidden)
        self.static_logits.copy_(logits)
