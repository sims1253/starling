"""Beam-4 decoder driver for the Hojo-ASR-V1 Qwen3-4B LLM.

Hojo-ASR-V1 decodes with **beam search** (``num_beams=4``, greedy over beams,
``repetition_penalty=2.0``, ``length_penalty=1``) -- unlike higgs/granite/ark
which are greedy. A custom CUDA-graph-captured beam-4 megakernel is
significantly harder than greedy (beam expansion + pruning + reordering of the
KV cache across 4 hypotheses is dynamic-shape work), and the repo has no prior
beam-search megakernel to reuse.

This first landing therefore drives the stock ``decoder_model.generate`` for
**byte-exactness** with the golden oracle (the simplest path that is guaranteed
to reproduce the reference). The API mirrors ``starling.higgs.llm_mega.LLMMega``
(``generate`` returns a :class:`GenerateResult`) so a future custom beam-graph
can subclass and override the decode loop without touching the pipeline.

Forward path (matches ``HOJO_ASR.infer`` exactly):
    speech_embeddings (1, N, 2560) + bos_embed (1, 1, 2560)
        -> inputs_embeds (1, N+1, 2560)
        -> decoder_model.generate(num_beams=4, inputs_embeds=..., eos=151645)
        -> output_ids (1, n_new)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import StoppingCriteria

from .config import (
    AUTOMIX_DTYPE,
    DO_SAMPLE,
    LENGTH_PENALTY,
    MAX_NEW_TOKENS,
    MIN_LENGTH,
    NUM_BEAMS,
    PAD_TOKEN_ID,
    REPETITION_PENALTY,
    STOP_TOKEN_SEQS,
    TEMPERATURE,
    TOP_P,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class GenerateResult:
    """Output of :meth:`LLMMega.generate`."""

    ids: torch.Tensor          # (1, n_new) int64 on CPU, the newly generated tokens
    text: str
    n_tokens: int
    total_ms: float
    tok_per_s: float


@dataclass
class BeamDecodeConfig:
    """Beam-search hyperparameters (defaults match the golden decode block)."""

    num_beams: int = NUM_BEAMS
    do_sample: bool = DO_SAMPLE
    repetition_penalty: float = REPETITION_PENALTY
    length_penalty: float = LENGTH_PENALTY
    max_new_tokens: int = MAX_NEW_TOKENS
    min_length: int = MIN_LENGTH
    temperature: float = TEMPERATURE
    top_p: float = TOP_P


class StopOnTokenSequences(StoppingCriteria):
    """Replicate ``hojo_asr.hojo_asr_model.StopOnTokenSequences``.

    Stops generation when the generated suffix matches any of the given token
    sequences. Hojo's reference passes ``[-100]`` (a token id that can never
    appear in generated ids), making this a no-op suffix stop -- the real stop
    is the ``eos_token_id``. We reproduce it verbatim so the stopping-criteria
    call count matches the reference exactly.
    """

    def __init__(self, stop_token_seqs: list[torch.Tensor] | None = None) -> None:
        super().__init__()
        self.stop_token_seqs = stop_token_seqs or []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        for seq in self.stop_token_seqs:
            tail = input_ids[0, -seq.numel():]
            if seq.numel() > 0 and torch.all(tail == seq).item():
                return True
        return False


class LLMMega:
    """Beam-4 decoder for the Hojo-ASR-V1 Qwen3-4B LLM.

    Wraps the loaded ``HOJO_ASR`` and drives the stock
    ``decoder_model.generate(num_beams=4, ...)`` under fp16 autocast (the
    reference runs the whole ``infer`` under ``autocast_context()``). Output is
    byte-exact with the golden oracle.

    Args:
        model: The loaded ``HOJO_ASR`` (provides ``decoder_model``,
            ``autocast_context``, and ``bos_token_id``).
        cfg: Beam-search hyperparameters (defaults match the golden).
    """

    def __init__(
        self,
        model: Any,
        cfg: BeamDecodeConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or BeamDecodeConfig()
        self.bos_token_id = int(getattr(model, "bos_token_id", 151644))

    # ------------------------------------------------------------------ #
    # prompt assembly (matches HOJO_ASR.infer)
    # ------------------------------------------------------------------ #
    def build_inputs_embeds(
        self,
        speech_embeddings: torch.Tensor,
        speech_attn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble decoder ``inputs_embeds`` + ``attention_mask``.

        Mirrors ``HOJO_ASR.infer``: ``inputs_embeds = cat([bos_embed,
        speech_embeddings], dim=1)`` (NO audio placeholder, NO text prompt),
        ``attention_mask = cat([bos_column, speech_attn], dim=1)``.

        Args:
            speech_embeddings: ``(B, N, 2560)`` encoder output (float32 under
                autocast -> cast by the decoder).
            speech_attn: ``(B, N)`` encoder attention mask.

        Returns:
            ``(inputs_embeds, attention_mask)`` of shapes ``(B, N+1, 2560)`` and
            ``(B, N+1)``.
        """
        device = speech_embeddings.device
        B = speech_embeddings.shape[0]
        bos_column = speech_attn[:, :1]
        bos_ids = torch.ones(B, 1, dtype=torch.int32, device=device) * self.bos_token_id
        bos_embeds = self.model.decoder_model.model.embed_tokens(bos_ids)
        inputs_embeds = torch.cat([bos_embeds, speech_embeddings], dim=1)
        attention_mask = torch.cat([bos_column, speech_attn], dim=1)
        return inputs_embeds, attention_mask

    # ------------------------------------------------------------------ #
    # generate (beam-4 via stock decoder_model.generate)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        speech_embeddings: torch.Tensor,
        speech_attn: torch.Tensor,
        max_new_tokens: int | None = None,
        tokenizer: Any = None,
    ) -> GenerateResult:
        """Beam-4 generate from encoder outputs (byte-exact with the golden).

        Args:
            speech_embeddings: ``(B, N, 2560)`` encoder output.
            speech_attn: ``(B, N)`` encoder attention mask.
            max_new_tokens: Cap on new tokens. If ``None``, uses
                ``min(cfg.max_new_tokens, N*2 + 10)`` (the reference formula),
                floored at 10.
            tokenizer: Optional tokenizer for the result ``text``.

        Returns:
            :class:`GenerateResult` (``ids`` is the *new* tokens, ``(1, n)``).
        """
        cfg = self.cfg
        inputs_embeds, attention_mask = self.build_inputs_embeds(
            speech_embeddings, speech_attn
        )

        # Reference max_new_tokens formula: min(max_tokens, feat_len*2 + 10),
        # floored at 10 (feat_len = speech_embeddings.size(1)).
        feat_len = speech_embeddings.shape[1]
        if max_new_tokens is None:
            max_new_tokens = min(cfg.max_new_tokens, int(feat_len * 2) + 10)
            max_new_tokens = max(max_new_tokens, 10)

        # Replicate the reference StopOnTokenSequences([-100]) (a no-op suffix
        # stop; the real stop is eos_token_id).
        stop_tensor = torch.tensor(
            list(STOP_TOKEN_SEQS[0]), device=inputs_embeds.device
        )
        from transformers import StoppingCriteriaList

        criteria = StoppingCriteriaList(
            [StopOnTokenSequences(stop_token_seqs=[stop_tensor])]
        )

        # The decoder is bf16; the reference runs generate under fp16 autocast.
        autocast_dtype = getattr(torch, AUTOMIX_DTYPE, torch.float16)
        eos_token_id = self.model.tokenizer.eos_token_id
        t0 = time.perf_counter()
        with self.model.autocast_context(autocast_dtype):
            gen_kwargs: dict[str, Any] = dict(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                num_beams=cfg.num_beams,
                do_sample=cfg.do_sample,
                min_length=cfg.min_length,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                length_penalty=cfg.length_penalty,
                attention_mask=attention_mask,
                pad_token_id=PAD_TOKEN_ID,
                stopping_criteria=criteria,
            )
            output_ids = self.model.decoder_model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        return self._finalize(output_ids, wall_ms, tokenizer)

    def _finalize(
        self,
        output_ids: torch.Tensor,
        wall_ms: float,
        tokenizer: Any,
    ) -> GenerateResult:
        """Build a :class:`GenerateResult` from the raw ``generate`` output.

        ``output_ids`` from ``inputs_embeds``-driven generate contains only the
        *new* tokens (no prompt prefix), matching the golden ``gen_ids``.
        """
        ids = output_ids[0].detach().cpu().to(torch.int64)
        n = int(ids.numel())
        text = ""
        if tokenizer is not None:
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        tok_per_s = n / max(wall_ms / 1000.0, 1e-9)
        return GenerateResult(
            ids=ids.unsqueeze(0),
            text=text,
            n_tokens=n,
            total_ms=wall_ms,
            tok_per_s=tok_per_s,
        )
