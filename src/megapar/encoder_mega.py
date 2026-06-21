"""Fused Granite-Speech-4.1-2b CTC conformer encoder (the "encoder megakernel").

This module reimplements the forward path of
:class:`transformers.models.granite_speech.modeling_granite_speech.GraniteSpeechCTCEncoder`
as a tight, fusion-friendly Python fast path that reuses the stock weights
(inference only) and exposes three acceleration strategies:

* ``mode="eager"``   -- a clean reimplementation that drops the transformers
  decorator overhead (``@capture_outputs`` / ``@merge_with_config_defaults``)
  and precomputes the per-layer Shaw relative-position bias table.
* ``mode="compile"`` -- wraps the eager fast path in ``torch.compile`` so the
  many LayerNorm / SiLU / residual elementwise ops fuse into a handful of
  kernels (and, with ``reduce-overhead``, the whole encoder runs as a single
  CUDA graph).
* ``mode="triton"``  -- additionally swaps the conformer FFN half-residual
  glue (LayerNorm variance in fp32 + SiLU + 0.5-scale + residual-add) and the
  post-conv residual-add for hand-written :mod:`triton` kernels. GEMMs and
  cuDNN convs stay as torch ops (they are already optimal).

The output is numerically faithful to the eager golden reference
(``golden/encoder_last_hidden.pt``) within :data:`megapar.config.ENCODER_ATOL`
(max abs 2e-2, mean abs 5e-3).

Public API
----------
``FusedEncoder(encoder, mode=..., compile_mode=..., compile_fullgraph=...)``
``FusedEncoder.forward(input_features) -> last_hidden_state``  (1, T, 1024) bf16
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import triton_kernels as tk  # noqa: F401  (lazy import inside methods)

# =========================================================================== #
# Fast-path block forward (a pure-Python reimplementation of the stock
# GraniteSpeechConformerBlock / GraniteSpeechConformerAttention forwards,
# but with the rel-pos bias precomputed and the nn.Softmax / decorator
# overhead removed).
# =========================================================================== #


class FusedEncoder(nn.Module):
    """Fused Granite-Speech CTC conformer encoder.

    Parameters
    ----------
    encoder : GraniteSpeechCTCEncoder
        The stock transformers encoder module. All weights/buffers are
        referenced (not copied) from its submodules, so this module shares
        parameters with the original.
    mode : {"eager", "compile", "triton"}
        Acceleration strategy (see module docstring).
    compile_mode : str
        ``torch.compile`` mode (only used when ``mode="compile"``). Use
        ``"reduce-overhead"`` for CUDA-graph capture or ``"max-autotune"``
        for aggressive kernel autotuning.
    compile_fullgraph : bool
        Whether to require a single compiled graph (no graph breaks).
    """

    def __init__(
        self,
        encoder,
        mode: str = "compile",
        compile_mode: str = "max-autotune",
        compile_fullgraph: bool = True,
    ) -> None:
        super().__init__()

        if mode not in ("eager", "compile", "triton"):
            raise ValueError(f"unknown mode {mode!r}; expected eager/compile/triton")

        # --- pull submodules (shared weights, no copy) ---------------------- #
        self.input_linear = encoder.input_linear
        self.layers = encoder.layers
        self.out = encoder.out
        self.out_mid = encoder.out_mid
        self.num_layers = int(encoder.num_layers)

        cfg = encoder.config
        self.context_size = int(cfg.context_size)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = int(cfg.dim_head)
        self.max_pos_emb = int(cfg.max_pos_emb)
        inner_dim = self.num_heads * self.head_dim
        self.inner_dim = int(inner_dim)
        self.scale = float(self.head_dim ** -0.5)

        # attention_dists buffer (200, 200) int64 -> keep on cuda as long.
        ad = encoder.attention_dists
        if ad.device.type != "cuda":
            ad = ad.cuda()
        self.register_buffer("attention_dists", ad.to(torch.long), persistent=False)

        # --- precompute per-layer Shaw rel-pos bias: ------------------------- #
        # rel_pos_bias[i] = layers[i].attn.rel_pos_emb(attention_dists)
        # shape (num_layers, context, context, head_dim) bf16, contiguous.
        with torch.no_grad():
            reps = [
                layer.attn.rel_pos_emb(self.attention_dists).contiguous()
                for layer in self.layers
            ]
            self.register_buffer(
                "rel_pos_bias", torch.stack(reps, dim=0).contiguous(), persistent=False
            )

        # --- precompute the last-block padding mask (for the remainder case). #
        # The stock encoder builds a (context, context) bool mask each forward
        # when the sequence length is not a multiple of context_size. For our
        # static seq=1247 the remainder is fixed at construction; we cache the
        # mask + the mask value so the forward is branch-free on the hot path.
        self._mask_value = float(-torch.finfo(torch.bfloat16).max)
        # remainder / num_blocks are filled lazily on first forward (they
        # depend on the input seq length). We cache the common-case mask.
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_num_features: int = 0

        # mid-layer self-conditioned CTC fires at 1-indexed idx == num_layers//2.
        self.mid_idx = self.num_layers // 2  # 8

        # --- dispatch config ------------------------------------------------- #
        self.mode = mode
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self._compiled_forward = None
        if mode == "compile":
            self._compiled_forward = torch.compile(
                self._forward_impl,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=False,
            )
        elif mode == "triton":
            self._compiled_forward = torch.compile(
                self._forward_triton,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=False,
            )

        # Triton kernels are loaded lazily (only needed for mode="triton").
        self._tk = None

    # ----------------------------------------------------------------------- #
    # mask cache (depends on the actual input length)
    # ----------------------------------------------------------------------- #
    def _last_block_mask(self, num_features: int, device) -> torch.Tensor:
        """Return the (context, context) bool mask for the last attention block.

        Mask is True (= will be masked) for any (q, k) pair where q or k is in
        the padded region. Cached per (num_features, device).
        """
        if (
            self._cached_mask is not None
            and self._cached_mask.device == device
            and self._cached_num_features == num_features
        ):
            return self._cached_mask
        remainder = num_features % self.context_size
        if remainder == 0:
            mask = torch.zeros(
                self.context_size, self.context_size, dtype=torch.bool, device=device
            )
        else:
            mask = torch.ones(
                self.context_size, self.context_size, dtype=torch.bool, device=device
            )
            mask[:remainder, :remainder] = False
        self._cached_mask = mask
        self._cached_num_features = int(num_features)
        return mask

    # ----------------------------------------------------------------------- #
    # public forward
    # ----------------------------------------------------------------------- #
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run the fused encoder.

        Parameters
        ----------
        input_features : Tensor
            Mel features of shape ``(1, T, 160)``, bf16, on cuda.

        Returns
        -------
        Tensor
            ``last_hidden_state`` of shape ``(1, T, 1024)`` bf16.
        """
        if input_features.dtype != torch.bfloat16:
            input_features = input_features.to(torch.bfloat16)
        if self._compiled_forward is not None:
            return self._compiled_forward(input_features)
        if self.mode == "triton":
            return self._forward_triton(input_features)
        return self._forward_impl(input_features)

    # ----------------------------------------------------------------------- #
    # eager / compile fast path (pure pytorch, fusion-friendly)
    # ----------------------------------------------------------------------- #
    def _forward_impl(self, input_features: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(input_features)
        for idx in range(self.num_layers):
            x = self._block_eager(idx, x)
            # mid-layer self-conditioned CTC (1-indexed == num_layers // 2)
            if (idx + 1) == self.mid_idx:
                mid = self.out(x)
                x = x + self.out_mid(_softmax_last_dim(mid))
        return x

    def _block_eager(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        layer = self.layers[idx]
        # ff1 half-step
        x = x + 0.5 * _ff_forward(layer.ff1, x)
        # attention + residual
        x = x + self._attn_forward(idx, layer.attn, x)
        # conv module + residual
        x = x + _conv_forward(layer.conv, x)
        # ff2 half-step
        x = x + 0.5 * _ff_forward(layer.ff2, x)
        # post norm
        x = layer.post_norm(x)
        return x

    # ----------------------------------------------------------------------- #
    # triton fast path (swaps FFN glue + residual-adds for triton kernels)
    # ----------------------------------------------------------------------- #
    def _ensure_triton(self):
        if self._tk is None:
            from . import triton_kernels as _tk
            self._tk = _tk

    def _forward_triton(self, input_features: torch.Tensor) -> torch.Tensor:
        self._ensure_triton()
        tk = self._tk
        x = self.input_linear(input_features)
        for idx in range(self.num_layers):
            x = self._block_triton(idx, x, tk)
            if (idx + 1) == self.mid_idx:
                mid = self.out(x)
                x = x + self.out_mid(_softmax_last_dim(mid))
        return x

    def _block_triton(self, idx: int, x: torch.Tensor, tk) -> torch.Tensor:
        layer = self.layers[idx]
        ff = layer.ff1
        # --- ff1: fused LN -> (cuBLAS up_proj) -> SiLU -> (cuBLAS down_proj)
        #         -> 0.5x scale -> residual add. We fuse the LN into a triton
        #         kernel that writes the normed tensor AND keeps the residual
        #         stream untouched; the final scale+add is also a triton kernel.
        normed = tk.fused_layernorm(x, ff.pre_norm.weight, ff.pre_norm.bias, ff.pre_norm.eps)
        up = ff.up_proj(normed)
        act = tk.fused_silu(up)
        down = ff.down_proj(act)
        x = tk.fused_residual_scale_add(x, down, 0.5)

        # attention (same as eager -- SDPA is already one launch)
        attn_out = self._attn_forward(idx, layer.attn, x)
        x = tk.fused_add(x, attn_out)

        # conv module (cuDNN convs stay; LN + residual-add fused)
        conv = layer.conv
        h = tk.fused_layernorm(x, conv.norm.weight, conv.norm.bias, conv.norm.eps)
        h = conv.up_conv(h.permute(0, 2, 1))
        h = conv.glu(h)
        h = conv.depth_conv(h)
        # fused batchnorm + silu
        h = tk.fused_batchnorm_silu(
            h, conv.batch_norm.weight, conv.batch_norm.bias,
            conv.batch_norm.running_mean, conv.batch_norm.running_var,
            conv.batch_norm.eps,
        )
        h = conv.down_conv(h).permute(0, 2, 1)
        x = tk.fused_add(x, h)

        # ff2 (same triton glue as ff1)
        ff = layer.ff2
        normed = tk.fused_layernorm(x, ff.pre_norm.weight, ff.pre_norm.bias, ff.pre_norm.eps)
        up = ff.up_proj(normed)
        act = tk.fused_silu(up)
        down = ff.down_proj(act)
        x = tk.fused_residual_scale_add(x, down, 0.5)

        # post norm
        x = layer.post_norm(x)
        return x

    # ----------------------------------------------------------------------- #
    # attention (shared by eager / triton paths)
    # ----------------------------------------------------------------------- #
    def _attn_forward(self, idx: int, attn, x: torch.Tensor) -> torch.Tensor:
        """Block-local Shaw relative-position attention.

        Mirrors GraniteSpeechConformerAttention.forward exactly, but reads the
        precomputed rel-pos bias from ``self.rel_pos_bias[idx]`` instead of
        doing an embedding lookup each call.
        """
        h = attn.pre_norm(x)
        bsz, num_features, _ = h.shape
        cs = self.context_size
        num_blocks = math.ceil(num_features / cs)
        remainder = num_features % cs
        if remainder > 0:
            h = torch.nn.functional.pad(h, (0, 0, 0, cs - remainder))

        q = attn.to_q(h)
        kv = attn.to_kv(h)
        k, v = kv.chunk(2, dim=-1)

        # (bsz, num_blocks, num_heads, context, head_dim)
        q = q.reshape(bsz, num_blocks, cs, self.num_heads, self.head_dim).transpose(2, 3)
        k = k.reshape(bsz, num_blocks, cs, self.num_heads, self.head_dim).transpose(2, 3)
        v = v.reshape(bsz, num_blocks, cs, self.num_heads, self.head_dim).transpose(2, 3)

        # Shaw rel-pos bias (precomputed): (context, context, head_dim)
        rep = self.rel_pos_bias[idx]
        pos_attn = torch.einsum("bmhcd,crd->bmhcr", q, rep) * self.scale

        if remainder > 0:
            mask = self._last_block_mask(num_features, pos_attn.device)
            pos_attn = pos_attn.clone()
            pos_attn[:, -1, :].masked_fill_(mask, self._mask_value)

        # Force the MATH backend (matches the stock encoder exactly).
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=pos_attn, scale=self.scale
            )
        out = out.transpose(2, 3).reshape(bsz, h.shape[1], self.inner_dim)
        out = attn.to_out(out[:, :num_features, :])
        return out


# =========================================================================== #
# small elementwise helpers (kept module-level so torch.compile can inline them)
# =========================================================================== #
def _ff_forward(ff, x: torch.Tensor) -> torch.Tensor:
    """GraniteSpeechConformerFeedForward.forward (eval-mode, dropout=identity)."""
    h = ff.pre_norm(x)
    h = ff.up_proj(h)
    h = F.silu(h)
    h = ff.down_proj(h)
    return h


def _conv_forward(conv, x: torch.Tensor) -> torch.Tensor:
    """GraniteSpeechConformerConvModule.forward (eval-mode, dropout=identity)."""
    h = conv.norm(x)
    h = conv.up_conv(h.permute(0, 2, 1))
    h = conv.glu(h)
    h = conv.depth_conv(h)
    h = F.silu(conv.batch_norm(h))
    h = conv.down_conv(h).permute(0, 2, 1)
    return h


def _softmax_last_dim(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)
