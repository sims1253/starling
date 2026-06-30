"""Hand-iterated bidirectional granite-4.0-1b LLM forward for NAR.

**STATUS: NEGATIVE RESULT — not used by the default pipeline.** Kept as a
documented dead end so the next reader doesn't repeat it. The default
:class:`starling.nar.mega.NarMega` uses the ``torch.compile``d **stock**
``model.language_model`` forward (see ``mega.py``), which is byte-exact.

Why this hand-rolled forward was written (and abandoned):

A graph-safe reimplementation of the 40-layer bidirectional LLM seemed
attractive because the stock ``GraniteSpeechNarRotaryEmbedding.forward`` is
wrapped in ``@dynamic_rope_update``. This module computes rotary cos/sin on the
host, reuses ``starling.granite.llm_kernels`` (fused RMSNorm/SwiGLU/residual),
and uses SDPA ``is_causal=False`` with native GQA.

It is byte-exact vs the stock forward on short (T=16) sequences (0.0 max-abs).
But on the **real packed edit sequences** it diverges:

    tier    logit max-abs diff (fused vs stock eager)
    short   0.375
    medium  0.625
    long    1.06

The decoded tokens match on short/medium but the long tier flips one borderline
argmax (the leading token). Per-layer the cosine stays >0.9999 — this is **bf16
cuBLAS reduction-order noise compounding over 40 layers on tiled long-sequence
GEMMs**, the same pathology granite/moss documented. The hand-rolled path and
the stock path pick slightly different cuBLAS algorithms; over 40 layers + a
1280-token packed sequence the <1-ULP-per-op noise crosses an argmax threshold.

Lesson: for a full bidirectional forward (unlike the AR tracks' single-token
decode), matching the stock cuBLAS reduction order is infeasible by hand. The
``torch.compile``d stock forward sidesteps this — Inductor emits its own
deterministic Triton kernels whose compiled-then-graph-captured output is
byte-exact at the decoded-token level on every tier.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: repeat KV heads. x: (B, n_kv, S, D) -> (B, n_q, S, D)."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(B, n_kv * n_rep, S, D)


class FusedNarLLM:
    """Graph-safe hand-iterated forward for the NAR bidirectional granite LLM.

    Wraps the loaded ``GraniteSpeechNarLM`` and reuses its weights (q/k/v/o_proj,
    gate/up/down_proj, RMSNorm weights, embed_tokens, lm_head) — no copies.
    """

    def __init__(self, language_model: Any) -> None:
        self.lm = language_model
        self.model = language_model.model  # GraniteSpeechNarModel
        self.config = language_model.config
        self.embed_tokens = self.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.layers = list(self.model.layers)
        self.norm = self.model.norm
        self.rotary = self.model.rotary_emb

        cfg = self.config
        self.n_q = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self.n_q))
        self.n_kv_groups = self.n_q // self.n_kv
        self.attn_scale = float(cfg.attention_multiplier)
        self.res_mult = float(cfg.residual_multiplier)
        self.rms_eps = float(cfg.rms_norm_eps)
        self.emb_mult = float(cfg.embedding_multiplier)
        self.logits_scaling = float(cfg.logits_scaling)

        # Reuse the granite fused Triton kernels (generic over granite-4.0-1b).
        from ..granite import llm_kernels as k

        self._k = k

    def compute_rope(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """Compute rotary cos/sin on the host (outside any captured region).

        Returns cos, sin of shape (1, T, head_dim) in ``dtype``.
        """
        # Direct inv_freq matmul — bypass the @dynamic_rope_update decorator.
        inv_freq = self.rotary.inv_freq  # (head_dim/2,)
        # position_ids: (1, T)
        freqs = (inv_freq.float()[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (1, T, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Bidirectional LLM forward -> logits (1, T, vocab).

        Graph-safe: no host control flow, no mask allocation, rotary precomputed.
        """
        k = self._k
        B, T, _ = inputs_embeds.shape
        n_q, n_kv, hd = self.n_q, self.n_kv, self.head_dim

        hidden = inputs_embeds * self.emb_mult

        # Rotary cos/sin (host-precomputed; static across the captured region).
        # The stock RoPE applies the *same* cos/sin to all heads at a position;
        # shape (1, T, hd) -> broadcast to (B, n_q/n_kv, T, hd).
        cos, sin = self.compute_rope(position_ids, hidden.dtype)
        cos4 = cos.unsqueeze(1)  # (1, 1, T, hd)
        sin4 = sin.unsqueeze(1)

        for layer in self.layers:
            sa = layer.self_attn
            mlp = layer.mlp

            # --- attention block ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self.rms_eps)

            q = sa.q_proj(normed).view(B, T, n_q, hd).transpose(1, 2)   # (B, n_q, T, hd)
            kt = sa.k_proj(normed).view(B, T, n_kv, hd).transpose(1, 2) # (B, n_kv, T, hd)
            v = sa.v_proj(normed).view(B, T, n_kv, hd).transpose(1, 2)  # (B, n_kv, T, hd)

            # RoPE (PyTorch, byte-exact bf16 arithmetic matching the reference)
            half = hd // 2
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kt_rot = torch.cat((-kt[..., half:], kt[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kt = kt * cos4 + kt_rot * sin4

            # Bidirectional attention via SDPA with native GQA (enable_gqa=True).
            # The stock GraniteSpeechNarAttention (sdpa interface) prefers
            # ``enable_gqa=True`` over ``repeat_kv`` when the mask permits; the
            # two route to DIFFERENT cuDNN/flash kernels with different bf16
            # reduction orders, so matching enable_gqa is required for
            # byte-exactness over 40 layers (a repeat_kv path diverges 3.8+
            # max-abs and flips borderline argmaxes).
            attn_out = F.scaled_dot_product_attention(
                q, kt, v, is_causal=False, scale=self.attn_scale, enable_gqa=True
            )  # (B, n_q, T, hd)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, n_q * hd)
            attn_out = sa.o_proj(attn_out)

            hidden = k.fused_residual_scale(residual, attn_out, self.res_mult)

            # --- MLP block ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self.rms_eps)
            gate = mlp.gate_proj(normed)
            up = mlp.up_proj(normed)
            act = k.fused_silu_mul(gate, up)
            mlp_out = mlp.down_proj(act)
            hidden = k.fused_residual_scale(residual, mlp_out, self.res_mult)

        hidden = k.fused_rmsnorm(hidden, self.norm.weight, self.rms_eps)
        logits = self.lm_head(hidden) / self.logits_scaling
        return logits
