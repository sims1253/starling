"""Batched (B > 1) inference pipeline for Qwen3-ASR-1.7B.

The batch=1 pipeline keeps the RTX 5090 ~10% busy during LLM decode: each of
the ~280 GEMVs per token is launch-latency bound, so the tensor cores sit idle.
Batching ``B`` independent audio streams turns those tiny GEMVs into real GEMMs
that saturate the tensor cores, and reads the ~3.4 GB of weights *once for B
tokens* instead of once per token. Aggregate throughput
(RTFx = sum(audio_seconds) / wall_time) scales with ``B`` until the GPU
saturates.

Design (mirrors ``starling.granite.batched``, adapted for Qwen3 — no
embedding/attention/logits multipliers):
* **Encoder + projector: per-stream (batch=1), byte-exact.** Run each stream's
  audio through the eager encoder separately.
* **LLM decode: batched, CUDA-graph-captured, byte-exact per stream.** The
  Qwen3 decoder trunk and ``transformers.StaticCache`` support batch>1
  natively. A precomputed 4D attention mask makes ``create_causal_mask``
  early-exit so capture isn't aborted by CPU-scalar allocation. Each stream's
  Q/K/V only ever touches its own KV cache rows -> cross-stream independence.
* **Per-stream EOS handling.** A ``finished`` bool mask tracks streams;
  finished streams feed the pad token while active streams continue. All
  streams stay lock-step (shared ``cumulative_length``), so one CUDA graph is
  valid for the whole batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .config import EOS_TOKEN_ID, PAD_TOKEN_ID
from .encoder_mega import GraphedEncoder
from .loader import get_components


@dataclass
class BatchedGenerateResult:
    """Output of :meth:`BatchedLLMMega.generate`."""

    ids_list: list[torch.Tensor]
    n_tokens_per_stream: list[int]
    total_tokens: int
    n_streams: int
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    max_new_tokens: int = 0

    @property
    def decode_tok_per_s(self) -> float:
        return self.total_tokens / max(self.decode_ms / 1000.0, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_streams": self.n_streams,
            "total_tokens": self.total_tokens,
            "prefill_ms": round(self.prefill_ms, 3),
            "decode_ms": round(self.decode_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "decode_tok_per_s": round(self.decode_tok_per_s, 1),
            "max_new_tokens": self.max_new_tokens,
        }


class BatchedLLMMega:
    """Batched CUDA-graph-captured greedy decoder for the Qwen3 LLM.

    Processes ``B = max_batch_size`` independent streams in lock-step. The
    decode step is the model's own forward captured into a single CUDA graph.
    Output is byte-exact per stream vs the batch=1 decoder.
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 4096,
        max_batch_size: int = 8,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        eos_token_id: int = EOS_TOKEN_ID,
        pad_token_id: int = PAD_TOKEN_ID,
    ) -> None:
        self.lm = language_model
        self.lm_head = lm_head
        self.config = language_model.config
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = int(max_batch_size)
        self.warmup_iters = int(warmup_iters)
        self.device = device
        self.dtype = dtype
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)

        self.vocab_size = int(self.config.vocab_size)
        B = self.max_batch_size

        self.static_input_ids = torch.zeros((B, 1), dtype=torch.int64, device=device)
        self.static_position_ids = torch.zeros((B, 1), dtype=torch.int64, device=device)
        self.static_logits = torch.zeros((B, 1, self.vocab_size), dtype=dtype, device=device)
        neg = torch.finfo(dtype).min
        self._neg_val = neg
        self.static_attn_mask = torch.full(
            (B, 1, 1, self.max_cache_len), neg, dtype=dtype, device=device
        )

        from transformers.cache_utils import StaticCache

        self.cache = StaticCache(config=self.config, max_cache_len=self.max_cache_len)
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False

    def _reset_cache_pos(self, n: int) -> None:
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(n)

    def _fill_shared_mask(self, valid_len: int) -> None:
        self.static_attn_mask.fill_(self._neg_val)
        self.static_attn_mask[:, :, :, :valid_len] = 0.0

    def _fill_batched_mask(
        self, prompt_lengths: torch.Tensor, cur_pos: int, pad_len: int
    ) -> None:
        M = self.max_cache_len
        pos = torch.arange(M, device=self.device)
        prompt_valid = pos.unsqueeze(0) < prompt_lengths.unsqueeze(1)
        decode_valid = (pos >= pad_len) & (pos <= cur_pos)
        valid = prompt_valid | decode_valid.unsqueeze(0)
        m = torch.zeros((self.max_batch_size, M), dtype=self.dtype, device=self.device)
        m.masked_fill_(~valid, self._neg_val)
        self.static_attn_mask.copy_(m.view(self.max_batch_size, 1, 1, M))

    def _decode_step_eager(self) -> None:
        out = self.lm(
            input_ids=self.static_input_ids,
            position_ids=self.static_position_ids,
            attention_mask=self.static_attn_mask,
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        # Qwen3: no logits scaling.
        self.static_logits.copy_(self.lm_head(hidden))

    @torch.inference_mode()
    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        prompt_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = inputs_embeds.shape[:2]
        assert T < self.max_cache_len, f"prompt {T} >= max_cache_len {self.max_cache_len}"
        self._reset_cache_pos(0)
        if prompt_lengths is not None:
            prompt_lengths = prompt_lengths.to(self.device)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)
        if prompt_lengths is None:
            attn_mask = None
        else:
            attn_mask = (
                torch.arange(T, device=self.device).unsqueeze(0)
                < prompt_lengths.to(self.device).long().unsqueeze(1)
            )
        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state
        if prompt_lengths is None:
            last_hidden = hidden[:, -1:, :]
        else:
            last_idx = (prompt_lengths.long() - 1).view(B, 1, 1).expand(B, 1, hidden.shape[-1])
            last_hidden = hidden.gather(1, last_idx)
        logits = self.lm_head(last_hidden)
        return logits.argmax(dim=-1)  # (B, 1)

    @torch.inference_mode()
    def capture(
        self,
        first_tokens: torch.Tensor,
        prefill_len: int,
        prompt_lengths: Optional[torch.Tensor] = None,
    ) -> None:
        finished0 = first_tokens.view(-1) == self.eos_token_id
        primed = torch.where(
            finished0.view(-1, 1),
            torch.full_like(first_tokens, self.pad_token_id),
            first_tokens,
        )

        def _prime(pos: int) -> None:
            self.static_input_ids.copy_(primed)
            self.static_position_ids.fill_(pos)
            if prompt_lengths is None:
                self._fill_shared_mask(pos + 1)
            else:
                self._fill_batched_mask(prompt_lengths, pos, prefill_len)

        _prime(prefill_len)
        for _ in range(self.warmup_iters):
            self._decode_step_eager()
        torch.cuda.synchronize()
        self._reset_cache_pos(prefill_len)
        _prime(prefill_len)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._decode_step_eager()
        self._reset_cache_pos(prefill_len)
        self._captured = True

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        prompt_lengths: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int] = None,
    ) -> BatchedGenerateResult:
        eos = int(eos_token_id) if eos_token_id is not None else self.eos_token_id
        B, T = inputs_embeds.shape[:2]
        if B != self.max_batch_size:
            raise ValueError(
                f"inputs_embeds batch {B} != max_batch_size {self.max_batch_size}"
            )
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} overflows cache (T={T}, "
                f"max_cache_len={self.max_cache_len}; at most {max_safe})."
            )

        t_prefill0 = time.perf_counter()
        next_tokens = self.prefill(inputs_embeds, prompt_lengths)  # (B, 1)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_prefill0) * 1000.0

        pad_len = T
        gen_per_stream: list[list[int]] = [[int(t)] for t in next_tokens.view(-1).tolist()]
        finished = next_tokens.view(-1) == eos

        if not self._captured:
            self.capture(next_tokens, T, prompt_lengths)

        t0 = time.perf_counter()
        for i in range(max_new_tokens - 1):
            cur_pos = T + i
            # feed finished streams the pad token; active streams their last token
            feed = torch.where(
                finished.view(-1, 1),
                torch.full_like(next_tokens, self.pad_token_id),
                next_tokens,
            )
            self.static_input_ids.copy_(feed)
            self.static_position_ids.fill_(cur_pos)
            if prompt_lengths is None:
                self._fill_shared_mask(cur_pos + 1)
            else:
                self._fill_batched_mask(prompt_lengths, cur_pos, pad_len)
            self._graph.replay()
            next_tokens = self.static_logits.argmax(dim=-1)  # (B, 1)
            toks = next_tokens.view(-1).tolist()
            for b in range(B):
                if not finished[b]:
                    gen_per_stream[b].append(toks[b])
                    if toks[b] == eos:
                        finished[b] = True
            if bool(finished.all()):
                break
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000.0

        ids_list = [torch.tensor(g, dtype=torch.int64).unsqueeze(0) for g in gen_per_stream]
        n_per = [len(g) for g in gen_per_stream]
        return BatchedGenerateResult(
            ids_list=ids_list,
            n_tokens_per_stream=n_per,
            total_tokens=sum(n_per),
            n_streams=B,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=prefill_ms + decode_ms,
            max_new_tokens=max_new_tokens,
        )


# =========================================================================== #
# Batched pipeline
# =========================================================================== #
class BatchedPipeline:
    """Batched end-to-end ASR: encode each stream, then batch-decode all at once."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_batch_size: int = 8,
        max_cache_len: int = 4096,
        encoder_mode: str = "cudagraph",
    ) -> None:
        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        self.audio_token_id = int(getattr(model.config, "audio_token_id", 151676))
        comps = get_components(model)
        self.fused_encoder = GraphedEncoder(comps["encoder"], mode=encoder_mode)
        self.projector = comps["projector"]
        self.embed_tokens = comps["language_model"].get_input_embeddings()
        self.llm = BatchedLLMMega(
            comps["language_model"],
            model.lm_head,
            max_cache_len=max_cache_len,
            max_batch_size=max_batch_size,
        )

    def build_inputs_embeds(self, input_ids, audio_embeds):
        inputs_embeds = self.embed_tokens(input_ids)
        mask = (input_ids == self.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(mask, audio_embeds.to(inputs_embeds.dtype))

    @torch.inference_mode()
    def transcribe_batch(
        self,
        list_input_features: list[torch.Tensor],
        list_input_ids: list[torch.Tensor],
        list_input_features_mask=None,
        max_new_tokens: int = 200,
    ) -> list[str]:
        """Encode each stream (batch=1, byte-exact), then batch-decode."""
        B = len(list_input_features)
        if B != self.llm.max_batch_size:
            raise ValueError(f"got {B} streams, pipeline built for {self.llm.max_batch_size}")

        # (1) per-stream encode + project
        audio_embeds_list = []
        for b in range(B):
            feats = list_input_features[b]
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)
            if list_input_features_mask is not None:
                mask = list_input_features_mask[b]
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
            else:
                mask = torch.ones(feats.shape[0], feats.shape[2], dtype=torch.long, device=feats.device)
            enc_lhs = self.fused_encoder(feats, mask)
            ae = self.projector(enc_lhs.clone())
            audio_embeds_list.append(ae)

        # (2) build per-stream inputs_embeds and right-pad to common T
        per_stream_embeds = [
            self.build_inputs_embeds(list_input_ids[b].to("cuda"), audio_embeds_list[b])
            for b in range(B)
        ]
        prompt_lengths = torch.tensor([e.shape[1] for e in per_stream_embeds], dtype=torch.long, device="cuda")
        T = int(prompt_lengths.max())
        H = per_stream_embeds[0].shape[-1]
        padded = torch.zeros((B, T, H), dtype=self.dtype, device="cuda")
        for b in range(B):
            t = per_stream_embeds[b].shape[1]
            padded[b, :t] = per_stream_embeds[b]

        # (3) batched generate
        res = self.llm.generate(padded, prompt_lengths=prompt_lengths, max_new_tokens=max_new_tokens)

        # (4) decode
        texts = []
        for ids in res.ids_list:
            try:
                texts.append(self.processor.decode(ids, return_format="transcription_only")[0])
            except Exception:
                texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
        return texts


__all__ = ["BatchedLLMMega", "BatchedPipeline", "BatchedGenerateResult"]
