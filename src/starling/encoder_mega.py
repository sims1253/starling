"""Fused Granite-Speech-4.1-2b CTC conformer encoder (the "encoder megakernel").

This module reimplements the forward path of
:class:`transformers.models.granite_speech.modeling_granite_speech.GraniteSpeechCTCEncoder`
as a tight, fusion-friendly Python fast path that reuses the stock weights
(inference only) and exposes four acceleration strategies:

* ``mode="eager"``     -- a clean reimplementation that drops the transformers
  decorator overhead (``@capture_outputs`` / ``@merge_with_config_defaults``),
  replaces ``nn.Softmax`` with ``F.softmax``, and precomputes the per-layer Shaw
  relative-position bias table. **Byte-exact** vs the stock encoder.
* ``mode="cudagraph"`` -- captures the byte-exact eager forward into a manual
  ``torch.cuda.CUDAGraph`` (after a side-stream warmup). Every per-layer kernel
  launch (LayerNorm, GEMM, conv, SiLU, residual add, attention) collapses into
  a single ``graph.replay()``. **Byte-exact** vs stock, zero Python overhead.
  This is the recommended default and the fastest byte-faithful path.
* ``mode="compile"``   -- wraps the eager forward in ``torch.compile``. Inductor
  fuses the elementwise glue aggressively but upcasts some attention intermediates
  to fp32, so the output is numerically close but NOT bitwise identical to the
  bf16 golden reference (see benchmark / test output for the actual diff).
* ``mode="triton"``    -- additionally swaps the conformer FFN half-residual
  glue (LayerNorm + SiLU + 0.5-scale + residual-add) and the conv-module
  BatchNorm+SiLU for hand-written :mod:`triton` kernels. GEMMs and cuDNN convs
  stay as torch ops (they are already optimal). Stride-aware so it handles the
  conv module's non-contiguous ``permute`` output without an extra copy.

The output is numerically faithful to the eager golden reference
(``golden/encoder_last_hidden.pt``). The ``eager`` and ``cudagraph`` modes are
byte-exact (0.0 diff); ``compile``/``triton`` stay within
:data:`starling.config.ENCODER_ATOL`.

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


class FusedEncoder(nn.Module):
    """Fused Granite-Speech CTC conformer encoder.

    Parameters
    ----------
    encoder : GraniteSpeechCTCEncoder
        The stock transformers encoder module. All weights/buffers are
        referenced (not copied) from its submodules, so this module shares
        parameters with the original.
    mode : {"eager", "cudagraph", "compile", "triton"}
        Acceleration strategy (see module docstring). ``"cudagraph"`` is the
        recommended default (byte-exact + zero launch overhead).
    compile_mode : str
        ``torch.compile`` mode (only used when ``mode`` is ``"compile"`` or
        ``"triton"``). Use ``"reduce-overhead"`` for CUDA-graph capture or
        ``"max-autotune"`` for aggressive kernel autotuning.
    compile_fullgraph : bool
        Whether to require a single compiled graph (no graph breaks).
    """

    def __init__(
        self,
        encoder,
        mode: str = "cudagraph",
        compile_mode: str = "max-autotune",
        compile_fullgraph: bool = True,
    ) -> None:
        super().__init__()

        if mode not in ("eager", "cudagraph", "compile", "triton"):
            raise ValueError(
                f"unknown mode {mode!r}; expected eager/cudagraph/compile/triton"
            )

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

        # --- mask value for padding (-max representable, NOT -inf, so softmax #
        # of a fully-masked row is well defined).                               #
        self.mask_value = float(-torch.finfo(torch.bfloat16).max)

        # --- last-block padding mask buffer.                                  #
        # The stock encoder builds a (context, context) bool mask each forward
        # when the sequence length is not a multiple of context_size. For our
        # static seq length this is a constant, so we precompute it ONCE into a
        # registered buffer (visible to torch.compile / CUDA graphs as a static
        # input, never mutated inside a compiled/captured region).
        self.register_buffer(
            "block_mask",
            torch.zeros(self.context_size, self.context_size, dtype=torch.bool),
            persistent=False,
        )
        self._block_mask_for: int = -1  # plain python int; never read inside compile

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

        # --- CUDA graph capture state (mode="cudagraph") --------------------- #
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None

        # Triton kernels are loaded lazily (only needed for mode="triton").
        self._tk = None

    # ----------------------------------------------------------------------- #
    # block-mask preparation (called OUTSIDE compiled/captured regions)
    # ----------------------------------------------------------------------- #
    def _prepare_block_mask(self, num_features: int, device) -> None:
        """Populate ``self.block_mask`` for the given seq length (no-op if cached).

        Pure-Python; runs before torch.compile enters its graph or before CUDA
        graph capture. The compiled/captured forward then reads the buffer as a
        static input.
        """
        if self._block_mask_for == num_features and self.block_mask.device == device:
            return
        remainder = num_features % self.context_size
        mask = torch.ones(
            self.context_size, self.context_size, dtype=torch.bool, device=device
        )
        if remainder > 0:
            mask[:remainder, :remainder] = False
        else:
            mask.fill_(False)
        self.block_mask = mask
        self._block_mask_for = int(num_features)

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
        # Prepare the block-attention padding mask outside any compiled /
        # captured region (it depends only on the seq length).
        self._prepare_block_mask(int(input_features.shape[1]), input_features.device)

        if self.mode == "cudagraph":
            return self._forward_cudagraph(input_features)
        if self._compiled_forward is not None:
            return self._compiled_forward(input_features)
        if self.mode == "triton":
            return self._forward_triton(input_features)
        return self._forward_impl(input_features)

    # ----------------------------------------------------------------------- #
    # CUDA-graph capture path (byte-exact + zero launch overhead)
    # ----------------------------------------------------------------------- #
    def _forward_cudagraph(self, input_features: torch.Tensor) -> torch.Tensor:
        if self._graph is None:
            self._capture_graph(input_features)
        # Validate the captured shape (CUDA graphs need static shapes).
        if input_features.shape != self._static_input.shape:
            raise RuntimeError(
                f"cudagraph captured for shape {tuple(self._static_input.shape)} "
                f"but got {tuple(input_features.shape)}; re-construct for new shape"
            )
        self._static_input.copy_(input_features)
        self._graph.replay()
        # Clone so the caller gets an owned tensor (the static output buffer is
        # reused across replays and would otherwise be overwritten next call).
        return self._static_output.clone()

    @torch.inference_mode()
    def _capture_graph(self, input_features: torch.Tensor) -> None:
        """Warmup on a side stream then capture the eager forward into a graph."""
        device = input_features.device
        # Static input/output buffers (own memory, stable addresses).
        self._static_input = torch.empty_like(input_features)
        self._static_input.copy_(input_features)

        # Warmup on a side stream (3 iters) so lazy initialisations (cuBLAS
        # handles, cuDNN algo selection, SDPA backend choice) settle BEFORE
        # capture. All warmup ops happen on the side stream's memory pool.
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                _ = self._forward_impl(self._static_input)
        torch.cuda.current_stream(device).wait_stream(side)

        # Capture.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self._forward_impl(self._static_input)

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
        # --- ff1: LayerNorm -> (cuBLAS up_proj) -> SiLU -> (cuBLAS down_proj)
        #         -> 0.5x scale -> residual add.
        #
        # NOTE on numerics: the conv module's BatchNorm has running_var as small
        # as 4e-10 (rstd up to 316x), which amplifies ANY difference in the
        # residual stream (from ANY upstream op, not just the conv module's own
        # LayerNorm) by 316x per block. Over 16 blocks this makes the encoder
        # numerically fragile: every op must be byte-exact vs the stock path.
        # We therefore use torch's native layer_norm + batch_norm (which ARE
        # byte-exact vs stock) and reserve the triton kernels for the
        # elementwise glue (SiLU, residual scale-add, add) that is provably
        # byte-exact (verified 0.0 diff vs F.silu / torch.add). The triton
        # fused_layernorm / fused_batchnorm_silu kernels remain available in
        # triton_kernels.py for experimentation / non-amplified architectures.
        normed = ff.pre_norm(x)
        up = ff.up_proj(normed)
        act = tk.fused_silu(up)
        down = ff.down_proj(act)
        x = tk.fused_residual_scale_add(x, down, 0.5)

        # attention (same as eager -- SDPA is already one launch)
        attn_out = self._attn_forward(idx, layer.attn, x)
        x = tk.fused_add(x, attn_out)

        # conv module (cuDNN convs stay; torch layernorm + batchnorm for exactness)
        conv = layer.conv
        h = conv.norm(x)
        h = conv.up_conv(h.permute(0, 2, 1))
        h = conv.glu(h)
        h = conv.depth_conv(h)
        h = F.silu(conv.batch_norm(h))
        h = conv.down_conv(h).permute(0, 2, 1)
        x = tk.fused_add(x, h)

        # ff2 (same byte-exact glue as ff1)
        ff = layer.ff2
        normed = ff.pre_norm(x)
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
            # Mask the padded region of the LAST block. Out-of-place masked_fill
            # (no in-place mutation) so this is compile/capture-friendly.
            mask = self.block_mask  # (context, context) bool, precomputed
            last = pos_attn[:, -1, :, :].masked_fill(mask, self.mask_value)
            pos_attn = torch.cat([pos_attn[:, :-1, :, :], last.unsqueeze(1)], dim=1)

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
