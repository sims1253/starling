"""CUDA-graph-captured Qwen3-ASR audio encoder (the "encoder megakernel").

The stock ``Qwen3ASREncoder.forward`` is byte-exact but NOT directly
CUDA-graph-capturable: it computes ``cu_seqlens`` (window boundaries) and a
packed valid-only sequence via ``index_select`` using indices derived from the
mask with data-dependent host ops (``.item()``, ``.tolist()``,
``.max().item()``). Those host syncs invalidate stream capture.

This module offers two modes:

* ``mode="eager"`` -- runs the stock encoder forward directly. Byte-exact, no
  capture. Used as the reference and for the first end-to-end correctness gate.
* ``mode="cudagraph"`` -- uses :class:`StaticEncoder`, a byte-exact
  reimplementation that **hoists the host-dependent packing out of the captured
  region**. For a fixed ``(B, padded_feat_len)`` shape with an all-valid mask
  (the common ASR case -- the processor pads to a multiple of ``chunk_len``),
  the packed length, ``cu_seqlens`` and ``valid_indices`` are deterministic and
  precomputed once on the host. The captured graph then runs only the
  pure-tensor compute (conv frontend, positional embed, 24 windowed-attention
  layers, final LayerNorm) at a fixed packed length, which is fully
  graph-capturable.

The graph-captured output is bit-identical to the eager ``Qwen3ASREncoder``
because it replays the model's own ops unchanged -- only host launch overhead
(and the packing bookkeeping, which is moved out) is removed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch._dynamo import mark_static_address as _mark_static
except Exception:  # pragma: no cover - older torch

    def _mark_static(t):  # type: ignore[misc]
        return t


def _mark_many(tensors) -> None:
    for t in tensors:
        try:
            _mark_static(t)
        except Exception:
            pass


def _round_up(feat_len: int, chunk_len: int) -> int:
    if feat_len % chunk_len == 0:
        return feat_len
    return ((feat_len // chunk_len) + 1) * chunk_len


# =========================================================================== #
# Custom graph-capturable windowed-attention encoder layer
# =========================================================================== #
# The stock Qwen3ASRAudioAttention processes the packed sequence one window at
# a time via torch.split(cu_seqlens) -- a host-driven variable-length loop with
# a .tolist() sync that invalidates CUDA-graph capture. We replace it with a
# reshape into (num_windows, heads, window_size, hd) + batched attention, which
# is graph-capturable (fixed shapes) and byte-exact (each window's attention is
# independent, and the padded tail of the last window is masked within-window).


class _WindowedAttn(nn.Module):
    """Static-shape windowed attention replacing the stock cu_seqlens split.

    Operates on a padded packed sequence reshaped to
    ``(num_windows, num_heads, window_size, head_dim)``. Within each window the
    attention is full (non-causal); the last window's padded tail is masked via
    a precomputed additive mask so padded positions do not leak.
    """

    def __init__(self, stock_attn: nn.Module, num_windows: int, window_size: int) -> None:
        super().__init__()
        # Share weights with the stock attention (no copy).
        self.q_proj = stock_attn.q_proj
        self.k_proj = stock_attn.k_proj
        self.v_proj = stock_attn.v_proj
        self.out_proj = stock_attn.out_proj
        self.num_heads = int(stock_attn.num_heads)
        self.head_dim = int(stock_attn.head_dim)
        self.scaling = float(stock_attn.scaling)
        self.num_windows = int(num_windows)
        self.window_size = int(window_size)

    def forward(self, hidden_states: torch.Tensor, window_attn_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (P, d_model) packed; reshape to (W, ws, d)
        W, ws = self.num_windows, self.window_size
        nh, hd = self.num_heads, self.head_dim
        h = hidden_states.view(W, ws, nh * hd)
        q = self.q_proj(h).view(W, ws, nh, hd).permute(0, 2, 1, 3)  # (W, nh, ws, hd)
        k = self.k_proj(h).view(W, ws, nh, hd).permute(0, 2, 1, 3)
        v = self.v_proj(h).view(W, ws, nh, hd).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # (W, nh, ws, ws)
        scores = scores + window_attn_mask  # (1, 1, ws, ws) or (W, nh, ws, ws) broadcast
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        out = torch.matmul(attn, v)  # (W, nh, ws, hd)
        out = out.permute(0, 2, 1, 3).reshape(W * ws, nh * hd)
        return self.out_proj(out)


# =========================================================================== #
# Static (graph-capturable) encoder forward
# =========================================================================== #
class StaticEncoder:
    """Byte-exact reimplementation of ``Qwen3ASREncoder.forward`` minus host ops.

    For a fixed ``(B, padded_feat_len)`` shape and an all-valid mask, the
    packing (``valid_indices``, ``cu_seqlens``, the per-chunk post-CNN lengths)
    is a deterministic function of the shape and is precomputed on the host.
    The captured graph then runs only pure-tensor ops at a fixed packed length.
    """

    def __init__(self, encoder) -> None:
        enc = encoder
        self.enc = enc
        # Reuse the encoder's own weights (shared, no copy).
        self.conv2d1 = enc.conv2d1
        self.conv2d2 = enc.conv2d2
        self.conv2d3 = enc.conv2d3
        self.conv_out = enc.conv_out
        self.positional_embedding = enc.positional_embedding
        self.layers = enc.layers
        self.ln_post = enc.ln_post
        self.n_window = int(enc.n_window)
        self.chunk_len = self.n_window * 2

    def _compute_packing(self, B: int, padded_feat_len: int, device, input_features_mask=None):
        """Host-side: precompute windowing from the REAL mask (byte-exact).

        Mirrors ``Qwen3ASREncoder.forward`` packing exactly:
          * feature_lens = mask.sum(-1)
          * chunk_lengths = mask reshaped per chunk, summed
          * cu_seqlens via get_audio_cu_seqlens
          * valid_indices from the per-chunk post-CNN lengths
        All host-computed (deterministic from the mask), then fed as static
        inputs to the graph.

        If ``input_features_mask`` is None, assumes all-valid (padded_len
        frames present) -- only correct for clips that fill the padded length.
        """
        from transformers.models.qwen3_asr.modeling_qwen3_asr import (
            _get_feat_extract_output_lengths,
            get_audio_cu_seqlens,
        )

        chunk_len = self.chunk_len
        num_chunks = padded_feat_len // chunk_len
        ts = chunk_len
        for _ in range(3):
            ts = (ts - 1) // 2 + 1
        time_steps = ts

        if input_features_mask is None:
            # all-valid fallback
            feature_lens = torch.full((B,), padded_feat_len, dtype=torch.long, device=device)
            chunk_lengths = torch.full((B * num_chunks,), chunk_len, dtype=torch.long, device=device)
        else:
            m = input_features_mask.to(device)
            feature_lens = m.sum(-1).to(torch.long)
            chunk_lengths = (
                m.view(B, num_chunks, chunk_len).sum(dim=-1).reshape(-1).to(torch.long)
            )

        # cu_seqlens (inference windows) -- exactly as the stock encoder.
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, feature_lens, self.enc.n_window_infer, self.n_window)
        # per-chunk post-CNN lengths -> valid_indices (the packed valid rows).
        chunk_post_cnn_lens = self.enc._post_cnn_length(chunk_lengths)
        valid_mask = torch.arange(time_steps, device=device) < chunk_post_cnn_lens.unsqueeze(1)
        valid_indices = valid_mask.flatten().nonzero().squeeze(-1).to(torch.int64)
        packed_len = int(valid_indices.numel())
        # pack into full windows of size window_size for graph-capturable
        # batched attention; the partial tail is masked within-window.
        window_size = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        # total packed across all windows = cu_seqlens[-1]; pad to full windows.
        total = int(cu_seqlens[-1].item())
        num_windows = int(cu_seqlens.numel() - 1)
        pad_len = num_windows * window_size - total

        # within-window additive mask: each window w has real length
        # cu_seqlens[w+1]-cu_seqlens[w]; pad its tail. Build (num_windows, ws, ws).
        neg = torch.finfo(self.enc.conv2d1.weight.dtype).min
        wmask = torch.zeros(num_windows, 1, window_size, window_size, dtype=torch.float32, device=device)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        for w, ln in enumerate(lengths):
            if ln < window_size:
                valid_row = torch.ones(window_size, dtype=torch.bool, device=device)
                valid_row[ln:] = False
                valid2d = valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
                lm = torch.zeros(window_size, window_size, dtype=torch.float32, device=device)
                lm.masked_fill_(~valid2d, neg)
                wmask[w, 0] = lm
        wmask = wmask.to(self.enc.conv2d1.weight.dtype)
        return (valid_indices, packed_len, num_chunks, time_steps, num_windows,
                window_size, pad_len, wmask)

    def _forward_tensor(self, input_features, valid_indices, num_chunks, time_steps,
                        num_windows, window_size, pad_len, window_attn_mask, windowed_layers):
        """Pure-tensor encoder compute (graph-capturable). Mirrors stock forward."""
        B, num_mel, padded_feat_len = input_features.shape
        chunked = (
            input_features.view(B, num_mel, num_chunks, self.chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(B * num_chunks, 1, num_mel, self.chunk_len)
        )
        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, _ = conv_out.size()
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, conv_channels * freq_bins)
        )
        conv_out = conv_out + self.positional_embedding.positional_embedding[:time_steps].to(conv_out.dtype)
        hidden_states = torch.index_select(
            conv_out.reshape(-1, conv_out.shape[-1]), 0, valid_indices
        )
        # pad the packed sequence to num_windows*window_size so it reshapes to
        # full windows for the graph-capturable windowed attention. The padded
        # rows are masked within-window (window_attn_mask) and trimmed after.
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
        for layer_norm, post_norm, attn, fc1, fc2, act, win_attn in windowed_layers:
            residual = hidden_states
            h = layer_norm(hidden_states)
            h = win_attn(h, window_attn_mask)
            hidden_states = residual + h
            residual = hidden_states
            h = post_norm(hidden_states)
            h = fc1(h)
            h = act(h)
            h = fc2(h)
            hidden_states = residual + h
        hidden_states = self.ln_post(hidden_states)
        if pad_len > 0:
            hidden_states = hidden_states[: hidden_states.shape[0] - pad_len]
        return hidden_states


# =========================================================================== #
# Graphed encoder wrapper (shape-keyed capture, parakeet-style)
# =========================================================================== #
class GraphedEncoder:
    """Capture the Qwen3-ASR audio encoder; encode many inputs.

    ``mode="eager"`` runs the stock encoder forward (byte-exact, no capture).
    ``mode="cudagraph"`` captures :class:`StaticEncoder` per
    ``(B, padded_feat_len)`` shape (byte-exact + zero launch overhead).

    Args:
        encoder: a loaded ``Qwen3ASREncoder`` on cuda (eval mode, bf16).
        warmup_iters: side-stream warmup iterations before graph capture.
        mode: ``"eager"`` or ``"cudagraph"``.
    """

    def __init__(self, encoder, *, warmup_iters: int = 3, mode: str = "eager",
                 max_cached_shapes: int = 8) -> None:
        self.encoder = encoder
        self.n_window = int(encoder.n_window)
        self.chunk_len = self.n_window * 2
        self.warmup_iters = int(warmup_iters)
        self.mode = mode
        self._static = StaticEncoder(encoder) if mode == "cudagraph" else None
        # Bound the per-shape graph cache. Each captured graph pins a private
        # CUDA memory pool, so unbounded growth across many clip lengths (a
        # 7-dataset x 50-clip leaderboard sweep has dozens of distinct shapes)
        # saturates VRAM and spills to shared/unified memory -> OOM. Eviction is
        # byte-exact-safe: capture is shape-only and static buffers are rewritten
        # each call, so re-capturing an evicted shape reproduces the identical graph.
        self.max_cached_shapes = int(max_cached_shapes)
        self._graphs: Dict[Tuple[int, int], dict] = {}

    # ------------------------------------------------------------------ #
    # shape-keyed capture (cudagraph mode)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _capture(self, input_features: torch.Tensor, input_features_mask: torch.Tensor) -> dict:
        B = int(input_features.shape[0])
        padded_feat_len = int(input_features.shape[2])
        device = input_features.device
        (valid_indices, packed_len, num_chunks, time_steps, num_windows,
         window_size, pad_len, wmask) = self._static._compute_packing(
            B, padded_feat_len, device, input_features_mask
        )

        windowed_layers = []
        for layer in self._static.layers:
            win_attn = _WindowedAttn(layer.self_attn, num_windows, window_size).to(device)
            windowed_layers.append((
                layer.self_attn_layer_norm, layer.final_layer_norm, layer.self_attn,
                layer.fc1, layer.fc2, layer.activation_fn, win_attn,
            ))

        static_inp = torch.empty_like(input_features)
        static_valid = valid_indices.clone()
        static_wmask = wmask.clone()
        _mark_many([static_inp, static_valid, static_wmask])
        static_inp.copy_(input_features)

        def _run():
            return self._static._forward_tensor(
                static_inp, static_valid, num_chunks, time_steps, num_windows,
                window_size, pad_len, static_wmask, windowed_layers,
            )

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                _run()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = _run()

        key = (B, padded_feat_len, num_windows, window_size, packed_len)
        bundle = {
            "B": B, "padded_feat_len": padded_feat_len, "packed_len": packed_len,
            "num_windows": num_windows, "window_size": window_size,
            "static_inp": static_inp, "static_valid": static_valid,
            "static_wmask": static_wmask, "static_out": static_out, "graph": graph,
        }
        # Bound the cache before inserting (LRU by insertion order; re-capture
        # of an evicted shape is byte-exact — see __init__ docstring).
        if len(self._graphs) >= self.max_cached_shapes:
            self._graphs.pop(next(iter(self._graphs)))
        self._graphs[key] = bundle
        return bundle

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder; return ``last_hidden_state`` ``(packed_len, d_model)``.

        ``mode="eager"``: delegates to the stock encoder (byte-exact).
        ``mode="cudagraph"``: right-pads to a captured shape and replays; the
        padded tail's extra output rows are trimmed to the original valid count.
        """
        if self.mode == "eager":
            out = self.encoder(
                input_features=input_features,
                input_features_mask=input_features_mask,
                return_dict=True,
            )
            return out.last_hidden_state

        # cudagraph mode
        B = int(input_features.shape[0])
        feat_len = int(input_features.shape[2])
        target = _round_up(feat_len, self.chunk_len)
        feats = _pad_feats(input_features, target)
        mask = input_features_mask
        if mask is not None and mask.shape[1] != target:
            mask = _pad_mask(input_features_mask, target)

        # Compute the (mask-dependent) windowing shape to look up / capture a graph.
        # packed_len MUST be in the key: static_valid has shape (packed_len,) and
        # static_out's valid prefix is packed_len, so a graph captured at one
        # packed_len cannot serve another (the .copy_ into static_valid would
        # shape-mismatch, and the valid-region output length differs).
        (_, packed_len, _, _, num_windows, window_size, _, _) = self._static._compute_packing(
            B, target, feats.device, mask
        )
        key = (B, target, num_windows, window_size, packed_len)
        bundle = self._graphs.get(key)
        if bundle is None:
            bundle = self._capture(feats, mask)

        # Recompute the per-call valid_indices + window mask from the REAL mask
        # and copy into the static buffers (they are the only mask-dependent
        # inputs; the conv/layers are shape-fixed by the captured windowing).
        (valid_indices, packed_len, _, _, _, _, _, wmask) = self._static._compute_packing(
            B, target, feats.device, mask
        )
        bundle["static_inp"].copy_(feats)
        bundle["static_valid"].copy_(valid_indices)
        bundle["static_wmask"].copy_(wmask)
        bundle["graph"].replay()
        out = bundle["static_out"]
        return out[:packed_len].clone()


def _pad_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = int(mask.shape[1])
    if cur == target_len:
        return mask
    return F.pad(mask, (0, target_len - cur), value=0)


def _pad_feats(input_features: torch.Tensor, target_len: int) -> torch.Tensor:
    feat_len = int(input_features.shape[2])
    if feat_len == target_len:
        return input_features
    return F.pad(input_features, (0, target_len - feat_len))


__all__ = ["GraphedEncoder", "StaticEncoder"]
