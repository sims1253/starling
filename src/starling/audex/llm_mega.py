"""CUDA-graph-captured greedy decoder for the Audex-2B Nemotron-Dense LLM.

The Nemotron-Dense decoder (28 layers, hidden 2048, GQA 16Q/8KV, relu2 MLP,
RoPE theta 1e8, untied embeddings) is the bulk of the Audex ASR runtime. The
stock eager ``model.generate`` path launches dozens of small kernels per token
and rebuilds Python state on every step, capping throughput far below the
memory-bandwidth ceiling.

This module mirrors the qwen3/granite ``llm_mega.py`` design (the decoder is a
standard Llama-family transformer) with key Nemotron-Dense differences:

* **MLP is relu2** (squared ReLU), NOT SwiGLU. The MLP is
  ``up_proj → relu2 → down_proj`` with no gate projection. The fused SwiGLU
  kernels don't apply; relu2 is cheap elementwise.
* **No QK-norm** (unlike qwen3 which applies RMSNorm to q/k before RoPE).
* **norm_eps = 1e-5** (granite-style, not qwen3's 1e-6).
* **Untied embeddings**: ``lm_head.weight`` is a separate parameter.
* **RoPE theta = 1e8**.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .config import (
    LLM_RMS_NORM_EPS,
)


@dataclass
class GenerateResult:
    """Output of :meth:`LLMMega.generate`."""

    ids: torch.Tensor  # (1, n_new) int64 on CPU
    text: str
    n_tokens: int
    total_ms: float
    tok_per_s: float


@dataclass
class BenchReport:
    prefill_ms: float = 0.0
    decode_ms_per_token: float = 0.0
    decode_tok_per_s: float = 0.0
    total_ms: float = 0.0
    total_tok_per_s: float = 0.0
    notes: str = ""


# =========================================================================== #
# Phase A: CUDA-graph-captured greedy decoder (model's own layers)
# =========================================================================== #
class LLMMega:
    """CUDA-graph-captured greedy decoder for the Nemotron-Dense LLM.

    Wraps a loaded ``NemotronDenseModel`` (the ``model.model`` from
    :func:`starling.audex.loader.get_components`) plus the parent model's
    ``lm_head``. The LLM's own layers are used unchanged so decode output is
    bit-exact with the eager golden reference.
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 4096,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        eos_token_id: int = 11,
        prefill_use_graph: bool = True,
    ) -> None:
        self.lm = language_model
        self.lm_head = lm_head
        self.config = language_model.config
        self.max_cache_len = int(max_cache_len)
        self.warmup_iters = int(warmup_iters)
        self.device = device
        self.dtype = dtype
        self.eos_token_id = int(eos_token_id)
        self.prefill_use_graph = bool(prefill_use_graph)

        self.vocab_size = int(self.config.vocab_size)
        self.num_layers = int(self.config.num_hidden_layers)

        # ---- static input / output buffers ----
        self.static_input_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
        self.static_position_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
        self.static_logits = torch.zeros(
            (1, 1, self.vocab_size), dtype=dtype, device=device
        )
        neg = torch.finfo(dtype).min
        self._neg_val = neg
        self.static_attn_mask = torch.full(
            (1, 1, 1, self.max_cache_len), neg, dtype=dtype, device=device
        )

        from transformers.cache_utils import StaticCache

        self.cache = StaticCache(config=self.config, max_cache_len=self.max_cache_len)
        self._cache_cls = StaticCache

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False
        self._prefill_graphs: OrderedDict[int, tuple[torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]] = OrderedDict()
        self._prefill_masks: dict[int, torch.Tensor] = {}
        self._max_prefill_graphs = 8

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _reset_cache_pos(self, n: int) -> None:
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(n)

    def _set_mask(self, valid_len: int) -> None:
        self.static_attn_mask.fill_(self._neg_val)
        self.static_attn_mask[:, :, :, :valid_len] = 0.0

    def _decode_step_eager(self) -> None:
        """One eager decode forward writing into ``static_logits``."""
        out = self.lm(
            input_ids=self.static_input_ids,
            position_ids=self.static_position_ids,
            attention_mask=self.static_attn_mask,
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        self.static_logits.copy_(self.lm_head(hidden))

    # ------------------------------------------------------------------ #
    # prefill (iterate layers directly — NemotronDenseModel has no
    # inputs_embeds path)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def prefill(self, inputs_embeds: torch.Tensor, *, use_graph: bool = True) -> torch.Tensor:
        """Prefill: fill the StaticCache, return the first token id."""
        T = inputs_embeds.shape[1]
        assert T < self.max_cache_len, f"prompt {T} >= max_cache_len {self.max_cache_len}"
        if use_graph:
            entry = self._prefill_graphs.get(T)
            if entry is None:
                entry = self._capture_prefill(inputs_embeds)
                self._prefill_graphs[T] = entry
                while len(self._prefill_graphs) > self._max_prefill_graphs:
                    _, old = self._prefill_graphs.popitem(last=False)
                    try:
                        old[1].reset()
                    except Exception:
                        pass
            else:
                self._prefill_graphs.move_to_end(T)
            static_emb, graph, out_tok = entry
            static_emb.copy_(inputs_embeds)
            self._reset_cache_pos(0)
            graph.replay()
            return out_tok.clone()

        return self._prefill_eager(inputs_embeds)

    def _prefill_eager(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Reference prefill forward (iterate layers from inputs_embeds)."""
        T = inputs_embeds.shape[1]
        self._reset_cache_pos(0)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0)
        attn_mask = self._prefill_mask(T)
        hidden = inputs_embeds
        for layer in self.lm.layers:
            hidden = layer(
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=self.cache,
                use_cache=True,
            )
        hidden = self.lm.norm(hidden)
        logits = self.lm_head(hidden[:, -1:, :])
        return logits.argmax(dim=-1)

    def _prefill_mask(self, T: int) -> torch.Tensor:
        """Graph-safe 4D causal mask for prefill over a StaticCache."""
        m = self._prefill_masks.get(T)
        if m is None:
            neg = self._neg_val
            ar = torch.arange(self.max_cache_len, device=self.device)
            q = torch.arange(T, device=self.device).unsqueeze(1)
            m = torch.where(
                ar[None, None, None, :] <= q[None, None, :, :],
                0.0,
                neg,
            ).to(self.dtype)
            self._prefill_masks[T] = m
        return m

    @torch.inference_mode()
    def _capture_prefill(self, inputs_embeds: torch.Tensor):
        device = inputs_embeds.device
        static_emb = torch.empty_like(inputs_embeds)
        static_emb.copy_(inputs_embeds)

        def _run():
            self._reset_cache_pos(0)
            return self._prefill_eager(static_emb)

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):
                _ = _run()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        self._reset_cache_pos(0)
        with torch.cuda.graph(graph):
            out_tok = _run()
        self._reset_cache_pos(0)
        return static_emb, graph, out_tok

    # ------------------------------------------------------------------ #
    # CUDA-graph capture of the decode step
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_position_ids.copy_(torch.tensor([[prefill_len]], device=self.device))
        self._set_mask(prefill_len + 1)
        for _ in range(self.warmup_iters):
            self._decode_step_eager()
        torch.cuda.synchronize()
        self._reset_cache_pos(prefill_len)
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_position_ids.copy_(torch.tensor([[prefill_len]], device=self.device))
        self._set_mask(prefill_len + 1)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._decode_step_eager()
        self._reset_cache_pos(prefill_len)
        self._captured = True

    # ------------------------------------------------------------------ #
    # generate
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int] = None,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        eos = int(eos_token_id) if eos_token_id is not None else self.eos_token_id
        T = inputs_embeds.shape[1]
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} overflows cache "
                f"(T={T}, max_cache_len={self.max_cache_len}; max {max_safe})."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError("LLMMega only supports batch=1.")
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)

        next_token = self.prefill(inputs_embeds, use_graph=self.prefill_use_graph)
        gen_ids = [int(next_token.item())]
        if max_new_tokens <= 1:
            return self._finalize(gen_ids, 0.0, tokenizer)

        if capture and not self._captured:
            self.capture(next_token, T)

        t0 = time.perf_counter()
        for i in range(max_new_tokens - 1):
            cur_pos = T + i
            self.static_input_ids.copy_(next_token.reshape(1, 1))
            self.static_position_ids.copy_(torch.tensor([[cur_pos]], device=self.device))
            self._set_mask(cur_pos + 1)
            if self._captured:
                self._graph.replay()
            else:
                self._decode_step_eager()
            next_token = self.static_logits.argmax(dim=-1)
            gen_ids.append(int(next_token.item()))
            if int(next_token.item()) == eos:
                break
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000.0
        return self._finalize(gen_ids, wall_ms, tokenizer)

    def _finalize(self, gen_ids: list[int], decode_wall_ms: float, tokenizer: Any) -> GenerateResult:
        ids = torch.tensor(gen_ids, dtype=torch.int64).unsqueeze(0)
        n = len(gen_ids)
        text = ""
        if tokenizer is not None:
            try:
                text = tokenizer.decode(ids, skip_special_tokens=True)[0]
            except TypeError:
                text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        decode_tps = n / max(decode_wall_ms / 1000.0, 1e-9)
        return GenerateResult(ids=ids, text=text, n_tokens=n, total_ms=decode_wall_ms, tok_per_s=decode_tps)


# =========================================================================== #
# Phase C: Fused decode path with shared Triton elementwise kernels
# =========================================================================== #
class FusedLLMMega(LLMMega):
    """CUDA-graph-captured greedy decoder with fused Triton elementwise kernels.

    Inherits all graph-capture / generate machinery from :class:`LLMMega` and
    overrides :meth:`_decode_step_eager` with a custom forward that manually
    iterates the 28 Nemotron-Dense decoder layers, replacing the small
    elementwise ops (RMSNorm, residual add) with single-launch Triton kernels
    from :mod:`starling.granite.llm_kernels`.

    Nemotron-Dense vs Qwen3 decode differences:
    * No QK-norm (skip the q_norm/k_norm RMSNorm steps).
    * MLP is relu2: ``up_proj(x) → relu(x)^2 → down_proj`` (no gate_proj).
    * No embedding/attention/residual/logits multipliers (alpha = 1.0).
    * RMSNorm eps = 1e-5.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from ..granite import llm_kernels as _k
        from ..attention import gqa_attention as _gqa_attention
        from ..flags import get_default_flags

        self._k = _k
        self._gqa_attention = _gqa_attention
        self._layers = list(self.lm.layers)
        self._embed = self.lm.embed_tokens
        self._final_norm = self.lm.norm
        self._rotary = self._layers[0].self_attn.rotary_emb
        cfg = self.config
        self._n_q_heads = int(cfg.num_attention_heads)
        self._n_kv_heads = int(cfg.num_key_value_heads)
        self._head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self._n_q_heads))
        self._n_kv_groups = self._n_q_heads // self._n_kv_heads
        self._attn_scale = float(self._head_dim ** -0.5)
        self._rms_eps = float(getattr(cfg, "norm_eps", LLM_RMS_NORM_EPS))
        self._intermediate = int(cfg.intermediate_size)
        self._flags = get_default_flags()
        # Precomputed RoPE ``rotate_half`` index + sign buffers (byte-exact).
        # Replaces ``torch.cat(-x[half:], x[:half])`` (2 allocs/layer) with
        # a single gather + mul. Gated by ``OptFlags.rope_alloc_free``.
        _hd2 = self._head_dim // 2
        _idx = torch.arange(self._head_dim, device=self.device)
        self._rope_idx = torch.cat([_idx[_hd2:], _idx[:_hd2]]).long()
        self._rope_sign = torch.cat([
            -torch.ones(_hd2, dtype=self.dtype, device=self.device),
            torch.ones(_hd2, dtype=self.dtype, device=self.device),
        ])
        self._fused: Optional[list[dict]] = None
        if self._flags.fused_qkv:
            self._fused = self._fuse_layer_weights()

    def _fuse_layer_weights(self) -> list[dict]:
        """Pre-concatenate QKV weights per layer (byte-exact).

        Nemotron-Dense MLP has only up_proj + down_proj (no gate_proj), so
        there is no gate/up GEMM to fuse — only the QKV fusion applies.
        """
        fused = []
        for layer in self._layers:
            sa = layer.self_attn
            qkv_w = torch.cat(
                [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], dim=0
            )
            fused.append({
                "qkv_w": qkv_w.contiguous(),
                "o_proj": sa.o_proj,
                "up_proj": layer.mlp.up_proj,
                "down_proj": layer.mlp.down_proj,
            })
        return fused

    def _decode_step_eager(self) -> None:
        """Custom single-token decode forward with fused Triton kernels.

        Replicates NemotronDenseDecoderLayer.forward exactly but replaces
        elementwise glue with fused kernels. Writes logits into static_logits.
        """
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        half = hd // 2
        qkv_split = [n_q * hd, n_kv * hd, n_kv * hd]
        flags = self._flags
        fused = self._fused

        # (1) embedding lookup (no multiplier)
        hidden = self._embed(self.static_input_ids)  # (1, 1, 2048)

        # (2) rotary cos/sin for this position.
        # NemotronDenseRotaryEmbedding returns (B, 1, seq, hd) — already 4D,
        # so no extra unsqueeze is needed (unlike qwen3 whose rotary returns 3D).
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        cos4 = cos  # (1, 1, 1, hd)
        sin4 = sin

        # (3) iterate layers
        for idx, layer in enumerate(self._layers):
            sa = layer.self_attn

            # --- attention block ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)

            # Q/K/V projections: fused GEMM (byte-exact) or the model's own.
            # Nemotron-Dense has NO QK-norm (unlike qwen3).
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

            q = q.transpose(1, 2)    # (1, n_q, 1, hd)
            kv = kv.transpose(1, 2)  # (1, n_kv, 1, hd)
            v = v.transpose(1, 2)    # (1, n_kv, 1, hd)

            # RoPE: rotate_half via precomputed index+sign buffers (byte-exact
            # with the reference cat-based recipe; avoids 2 allocs/layer).
            if flags.rope_alloc_free:
                q_rot = q[..., self._rope_idx] * self._rope_sign
                k_rot = kv[..., self._rope_idx] * self._rope_sign
            else:
                q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
                k_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + k_rot * sin4

            kv, v = self.cache.update(kv, v, sa.layer_idx)

            attn_out = self._gqa_attention(
                q, kv, v, self.static_attn_mask, self._attn_scale, self.dtype, flags
            )

            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            attn_out = o_proj(attn_out)

            # fused residual add (alpha = 1.0)
            hidden = k.fused_residual_scale(residual, attn_out, 1.0)

            # --- MLP block (relu2, NOT SwiGLU) ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)
            if fused is not None:
                up = f["up_proj"](normed)
                act = torch.relu(up).pow(2)
                mlp_out = f["down_proj"](act)
            else:
                mlp_out = layer.mlp(normed)
            hidden = k.fused_residual_scale(residual, mlp_out, 1.0)

        # (4) final fused RMSNorm + lm_head (no logits scaling)
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self.lm_head(hidden)
        self.static_logits.copy_(logits)


__all__ = ["LLMMega", "FusedLLMMega", "GenerateResult", "BenchReport"]
