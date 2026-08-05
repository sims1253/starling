"""CUDA-graph-captured greedy decoder for the Qwen3 ASR text decoder (1.7B).

The Qwen3 LLM decoder (28 layers, hidden 2048, GQA 16Q/8KV, SwiGLU, RoPE,
tied embeddings) is the bulk of the Qwen3-ASR runtime, just like the Granite
decoder in Granite-Speech. The stock eager ``model.generate`` path launches
dozens of small kernels per token and rebuilds Python state on every step,
capping throughput far below the memory-bandwidth ceiling of the RTX 5090.

This module mirrors the Granite ``llm_mega.py`` design (the decoder is
structurally a standard Llama-family transformer) but drops Granite's
embedding/residual/logits/attention multipliers — Qwen3 uses plain residuals
and the standard ``head_dim**-0.5`` attention scale.

* **Phase A** — a correct CUDA-graph-captured greedy decode built on the
  model's own layers and ``transformers.StaticCache``. Graph replay of the
  model's own ops is bit-exact with eager.
* **Phase C** — a fused decode path that swaps in the shared Triton kernels
  from :mod:`starling.granite.llm_kernels` (fused RMSNorm, SwiGLU, residual
  add) to cut memory traffic and launch count. These kernels are
  model-agnostic; Qwen3 just calls ``fused_residual_scale`` with ``alpha=1.0``
  and omits the embedding/logits scaling.

Qwen3 vs Granite decode differences
-----------------------------------
* No embedding multiplier (``embed_tokens(x)`` directly).
* No attention multiplier (``scores * (head_dim**-0.5)``).
* No residual multiplier (``hidden + delta``, i.e. alpha = 1.0).
* No logits scaling (``lm_head(hidden)`` directly).
* RMSNorm eps = 1e-6 (Granite 1e-5).
* head_dim = 128 explicit in config (== hidden // num_heads here).
* Tied embeddings: ``lm_head.weight is embed_tokens.weight``.
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
    """Aggregated benchmark numbers for printing / JSON."""

    prefill_ms: float = 0.0
    decode_ms_per_token: float = 0.0
    decode_tok_per_s: float = 0.0
    total_ms: float = 0.0
    total_tok_per_s: float = 0.0
    notes: str = ""


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: repeat KV heads to match Q heads. (B, n_kv, S, D) -> (B, n_q, S, D)."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(B, n_kv * n_rep, S, D)


# =========================================================================== #
# Phase A: CUDA-graph-captured greedy decoder (model's own layers)
# =========================================================================== #
class LLMMega:
    """CUDA-graph-captured greedy decoder for the Qwen3 ASR text decoder.

    Wraps a loaded ``Qwen3ForCausalLM``-style language model (the
    ``language_model`` from :func:`starling.qwen3.loader.get_components`) plus
    the parent model's ``lm_head``. The LLM's own layers are used unchanged so
    decode output is bit-exact with the eager golden reference.
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 1024,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        eos_token_id: int = 151645,
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
        # Prefill graphs are captured per prompt length T (cap 8, evict+reset).
        # On a diverse-length sweep (~50 distinct T/dataset) that churn corrupts
        # the CUDA-graph allocator and surfaces as an illegal memory access a few
        # datasets in. Prefill is a one-shot compute-bound forward, so graphing
        # it saves only launch overhead; run it eager to keep the allocator quiet.
        # The decode loop stays graphed. Byte-exact either way (both run the
        # model's own layers). See MegaPipeline.prefill_use_graph.
        self.prefill_use_graph = bool(prefill_use_graph)

        self.vocab_size = int(self.config.vocab_size)
        self.num_layers = int(self.config.num_hidden_layers)

        # ---- static input / output buffers (fixed addresses for the graph) --
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
    # prefill
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def prefill(self, inputs_embeds: torch.Tensor, *, use_graph: bool = True) -> torch.Tensor:
        """Eager prefill: fill the StaticCache, return the first token id."""
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
        """Reference prefill forward."""
        T = inputs_embeds.shape[1]
        self._reset_cache_pos(0)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=self._prefill_mask(T),
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.lm_head(hidden)
        return logits.argmax(dim=-1)  # (1, 1)

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
        """Capture a prompt-length-specific prefill CUDA graph."""
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
                f"max_new_tokens={max_new_tokens} would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len}; at most "
                f"{max_safe} new tokens fit)."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError(f"LLMMega only supports batch=1, got batch={inputs_embeds.shape[0]}.")
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

    # ------------------------------------------------------------------ #
    # benchmark
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def bench(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 200,
        decode_iters: int = 20,
    ) -> BenchReport:
        T = inputs_embeds.shape[1]
        pos_ids = torch.arange(T, device=self.device).unsqueeze(0)

        def _prefill():
            self._reset_cache_pos(0)
            self.lm(inputs_embeds=inputs_embeds, position_ids=pos_ids, past_key_values=self.cache, use_cache=True)

        prefill_ms = self._cuda_timer(_prefill, warmup=3, iters=10)

        self._reset_cache_pos(0)
        first_tok = self.prefill(inputs_embeds, use_graph=self.prefill_use_graph)
        self.capture(first_tok, T)
        self.static_input_ids.copy_(first_tok.reshape(1, 1))
        self.static_position_ids.copy_(torch.tensor([[T]], device=self.device))
        self._set_mask(T + 1)

        def _one_decode():
            self._graph.replay()
            self._reset_cache_pos(T)

        decode_ms = self._cuda_timer(_one_decode, warmup=3, iters=decode_iters)
        decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

        self._reset_cache_pos(0)
        self._captured = False
        res = self.generate(inputs_embeds, max_new_tokens=max_new_tokens)
        return BenchReport(
            prefill_ms=prefill_ms,
            decode_ms_per_token=decode_ms,
            decode_tok_per_s=decode_tps,
            total_ms=res.total_ms,
            total_tok_per_s=res.tok_per_s,
            notes=f"decoded {res.n_tokens} tokens; cache_len={self.max_cache_len}",
        )

    @staticmethod
    def _cuda_timer(fn, warmup: int = 3, iters: int = 20) -> float:
        import statistics

        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        return statistics.median(times)


# =========================================================================== #
# Phase C: Fused decode path with shared Triton elementwise kernels
# =========================================================================== #
class FusedLLMMega(LLMMega):
    """CUDA-graph-captured greedy decoder with fused Triton elementwise kernels.

    Inherits all graph-capture / generate / bench machinery from
    :class:`LLMMega` and overrides :meth:`_decode_step_eager` with a custom
    forward that manually iterates the 28 Qwen3 decoder layers, replacing the
    small elementwise ops (RMSNorm, SwiGLU, residual add) with single-launch
    Triton kernels from :mod:`starling.granite.llm_kernels`.

    GEMMs (q/k/v/o_proj, gate/up/down_proj, lm_head) and the attention
    softmax/matmul stay as stock PyTorch ops (cuBLAS). RoPE stays in PyTorch
    (matching the reference's bf16 arithmetic exactly; the granite sibling
    found a Triton RoPE diverges for large Q magnitudes).

    Qwen3 has NO embedding/attention/residual/logits multipliers, so the fused
    residual uses ``alpha=1.0`` and embeddings/logits are used directly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Reuse the shared, model-agnostic Triton kernels from the granite track.
        from ..granite import llm_kernels as _k
        from ..attention import gqa_attention as _gqa_attention
        from ..flags import get_default_flags

        self._k = _k
        self._gqa_attention = _gqa_attention
        self._layers = list(self.lm.layers)
        self._embed = self.lm.embed_tokens
        self._final_norm = self.lm.norm
        self._rotary = self.lm.rotary_emb
        cfg = self.config
        self._n_q_heads = int(cfg.num_attention_heads)
        self._n_kv_heads = int(cfg.num_key_value_heads)
        self._head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self._n_q_heads))
        self._n_kv_groups = self._n_q_heads // self._n_kv_heads
        self._attn_scale = float(self._head_dim ** -0.5)
        self._rms_eps = float(getattr(cfg, "rms_norm_eps", LLM_RMS_NORM_EPS))
        self._intermediate = int(cfg.intermediate_size)
        self._flags = get_default_flags()
        self._fused: Optional[list[dict]] = None
        if self._flags.fused_qkv:
            self._fused = self._fuse_layer_weights()

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
        """Custom single-token decode forward with fused Triton kernels.

        Replicates Qwen3DecoderLayer.forward exactly but replaces elementwise
        glue with fused kernels. Writes the final logits into ``static_logits``.
        """
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        half = hd // 2
        qkv_split = [n_q * hd, n_kv * hd, n_kv * hd]
        inter = self._intermediate
        flags = self._flags
        fused = self._fused

        # (1) embedding lookup (NO multiplier for Qwen3)
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

            # Q/K/V projections: fused GEMM (byte-exact) or the model's own.
            # Qwen3 applies QK-norm (RMSNorm over head_dim) to q/k BEFORE RoPE,
            # so split then norm regardless of which projection path ran.
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
            q = k.fused_rmsnorm(q, sa.q_norm.weight, self._rms_eps)
            kv = k.fused_rmsnorm(kv, sa.k_norm.weight, self._rms_eps)
            q = q.transpose(1, 2)    # (1, n_q, 1, hd)
            kv = kv.transpose(1, 2)  # (1, n_kv, 1, hd)
            v = v.transpose(1, 2)    # (1, n_kv, 1, hd)

            # RoPE (PyTorch, matching the reference's bf16 arithmetic exactly)
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            k_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + k_rot * sin4

            kv, v = self.cache.update(kv, v, layer.self_attn.layer_idx)

            attn_out = self._gqa_attention(
                q, kv, v, self.static_attn_mask, self._attn_scale, self.dtype, flags
            )

            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            attn_out = o_proj(attn_out)

            # fused residual add (alpha = 1.0 for Qwen3)
            hidden = k.fused_residual_scale(residual, attn_out, 1.0)

            # --- MLP block ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)
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
            act = k.fused_silu_mul(gate, up)
            mlp_out = down_proj(act)
            hidden = k.fused_residual_scale(residual, mlp_out, 1.0)

        # (4) final fused RMSNorm + lm_head (NO logits scaling for Qwen3)
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self.lm_head(hidden)
        self.static_logits.copy_(logits)


__all__ = ["LLMMega", "FusedLLMMega", "GenerateResult", "BenchReport"]
