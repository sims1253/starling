"""End-to-end CUDA-graph normalization pipeline for S1-mini.

S1-mini is a pure text-to-text Qwen3-0.6B: the audio encoder of the Qwen3-ASR
track disappears and the input embedding is a plain ``embed_tokens`` lookup.

    chat template (system + control line + transcript,
    enable_thinking=False assistant prefix)
        -> embed_tokens(input_ids)            (1, T, 1024) bf16
        -> S1MultiStepLLMMega.generate(...)   K-step CUDA-graph greedy decode
        -> tokenizer.decode(skip_special_tokens=True)

The decoder machinery is the Qwen3-ASR track's :class:`MultiStepLLMMega`
(S1-mini is structurally a Qwen3ForCausalLM: same module layout, same GQA +
qk_norm + tied embeddings, just 1024-wide instead of 2048). The only
behavioural difference is the stop rule: stock ``model.generate`` for S1-mini
stops on **both** ``<|im_end|>`` (151645) and ``<|endoftext|>`` (151643), so
the subclass below carries an EOS *set* instead of the single id.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch

from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_STYLING,
    DEFAULT_STRUCTURE,
    EOS_TOKEN_IDS,
    MAX_INPUT_TOKENS,
    control_line,
    max_new_tokens_for,
)
from .loader import get_components, load_model_and_tokenizer

from ..qwen3.llm_mega import GenerateResult, LLMMega
from ..qwen3.multistep import MultiStepLLMMega


class S1MultiStepLLMMega(MultiStepLLMMega):
    """MultiStepLLMMega with the dual-EOS stop S1-mini's generation config uses.

    The emitted token sequence is byte-exact with stock ``model.generate``
    (greedy = greedy; only the stop predicate widens to a set).
    """

    def __init__(self, *args, eos_token_ids: tuple[int, ...] = EOS_TOKEN_IDS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eos_token_ids = tuple(int(t) for t in eos_token_ids)
        self._eos_set = frozenset(self.eos_token_ids)

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int] = None,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        # Parent loop with `tok in self._eos_set` instead of `tok == eos`.
        T = inputs_embeds.shape[1]
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} overflows cache (T={T}, "
                f"max_cache_len={self.max_cache_len})."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError("S1MultiStepLLMMega only supports batch=1.")
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)

        K = self.K
        n_decode = max_new_tokens - 1
        next_token = self.prefill(inputs_embeds, use_graph=self.prefill_use_graph)
        gen_ids = [int(next_token.item())]
        if max_new_tokens <= 1 or n_decode <= 0:
            return self._finalize(gen_ids, 0.0, tokenizer)

        n_chunks = (n_decode + K - 1) // K
        total_steps = n_chunks * K
        if T - 1 + total_steps >= self.max_cache_len:
            raise ValueError(
                f"multi-step rounded-up decode ({total_steps} steps across "
                f"{n_chunks} chunks of K={self.K}) would overflow the cache."
            )

        if capture and not self._ms_captured:
            self.capture(next_token, T)
        self._reset_to_chunk_start(T, next_token)

        import time

        t0 = time.perf_counter()
        done = False
        for _chunk in range(n_chunks):
            self._ms_graph.replay()
            out = self.output_ids.tolist()
            for tok in out:
                if len(gen_ids) >= max_new_tokens:
                    done = True
                    break
                gen_ids.append(tok)
                if tok in self._eos_set:
                    done = True
                    break
            if done:
                break
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return self._finalize(gen_ids, wall_ms, tokenizer)


class NormalizePipeline:
    """Text normalization pipeline owning the tokenizer + fused LLM decoder."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_cache_len: int = 4096,
        use_fused_llm: bool = True,
        steps_per_replay: int | None = None,
        prefill_use_graph: bool = False,
        compile_decode: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        # Eager prefill by default (same allocator-churn rationale as the
        # Qwen3-ASR track): diverse prompt lengths evict per-length graphs.
        self.prefill_use_graph = bool(prefill_use_graph)

        comps = get_components(model)
        self.embed_tokens = comps["embed_tokens"]
        self._language_model = comps["language_model"]
        self._lm_head = comps["lm_head"]
        self._max_cache_len = int(max_cache_len)
        self.steps_per_replay = (
            None if steps_per_replay is None else max(1, int(steps_per_replay))
        )
        self._llms_by_k: dict[int, S1MultiStepLLMMega] = {}
        self.use_fused_llm = bool(use_fused_llm)
        if use_fused_llm:
            self.llm = self._get_multistep_llm(self._steps_for_prompt(0))
        else:
            self.llm = LLMMega(
                self._language_model,
                self._lm_head,
                max_cache_len=self._max_cache_len,
                eos_token_id=EOS_TOKEN_IDS[0],
                prefill_use_graph=self.prefill_use_graph,
            )

    @classmethod
    def from_pretrained(
        cls,
        *,
        attn_impl: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_cache_len: int = 4096,
        use_fused_llm: bool = True,
        steps_per_replay: int | None = None,
        prefill_use_graph: bool = False,
    ) -> "NormalizePipeline":
        model, tokenizer = load_model_and_tokenizer(
            attn_impl=attn_impl, dtype=dtype, device=device
        )
        return cls(
            model,
            tokenizer,
            max_cache_len=max_cache_len,
            use_fused_llm=use_fused_llm,
            steps_per_replay=steps_per_replay,
            prefill_use_graph=prefill_use_graph,
        )

    def _steps_for_prompt(self, prompt_len: int) -> int:
        """Replay chunk size. RTX 5090 sweep on 2026-08-19 (see
        benchmarks/s1/bench_normalize.py --sweep-k): K=1 wins on short
        prompts (11-token output loses to K-chunk overcompute), K=4 is best
        for medium (79 tok) and long (281 tok) outputs. Explicit constructor
        values still override this policy.
        """
        if self.steps_per_replay is not None:
            return self.steps_per_replay
        return 1 if int(prompt_len) <= 128 else 4

    def _get_multistep_llm(self, steps_per_replay: int) -> S1MultiStepLLMMega:
        k = max(1, int(steps_per_replay))
        llm = self._llms_by_k.get(k)
        if llm is None:
            llm = S1MultiStepLLMMega(
                self._language_model,
                self._lm_head,
                max_cache_len=self._max_cache_len,
                steps_per_replay=k,
                eos_token_ids=EOS_TOKEN_IDS,
                prefill_use_graph=self.prefill_use_graph,
            )
            self._llms_by_k[k] = llm
        return llm

    # ------------------------------------------------------------------ #
    # prompt construction (verbatim model-card quickstart shape)
    # ------------------------------------------------------------------ #
    def build_prompt(
        self,
        transcript: str,
        *,
        styling: str = DEFAULT_STYLING,
        structure: str = DEFAULT_STRUCTURE,
        context: str = DEFAULT_CONTEXT,
    ) -> str:
        from .config import SYSTEM_PROMPT

        user = f"{control_line(styling, structure, context)}\n{transcript}"
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return text

    def build_prompt_ids(
        self,
        transcript: str,
        *,
        styling: str = DEFAULT_STYLING,
        structure: str = DEFAULT_STRUCTURE,
        context: str = DEFAULT_CONTEXT,
    ) -> torch.Tensor:
        """Token ids of the full prompt (template special tokens included)."""
        text = self.build_prompt(transcript, styling=styling, structure=structure, context=context)
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    # ------------------------------------------------------------------ #
    # normalize
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def normalize(
        self,
        transcript: str,
        *,
        styling: str = DEFAULT_STYLING,
        structure: str = DEFAULT_STRUCTURE,
        context: str = DEFAULT_CONTEXT,
        max_new_tokens: int | None = None,
    ) -> tuple[str, torch.Tensor]:
        """Normalize one raw ASR transcript. Returns ``(text, ids)``.

        ``ids`` are the generated tokens only, ``(1, n_new)`` int64 on CPU
        (stop token included), matching what stock ``model.generate`` emits
        before ``skip_special_tokens`` decoding.
        """
        input_ids = self.build_prompt_ids(
            transcript, styling=styling, structure=structure, context=context
        )
        T = int(input_ids.shape[1])
        if T > MAX_INPUT_TOKENS:
            raise ValueError(
                f"prompt is {T} tokens > trained max {MAX_INPUT_TOKENS}; "
                "chunk the transcript at sentence boundaries first "
                "(see chunk_transcript)"
            )
        if max_new_tokens is None:
            max_new_tokens = max_new_tokens_for(T)
        max_new_tokens = min(max_new_tokens, self._max_cache_len - T + 1)

        inputs_embeds = self.embed_tokens(input_ids)
        if self.use_fused_llm:
            self.llm = self._get_multistep_llm(self._steps_for_prompt(T))
        res = self.llm.generate(inputs_embeds, max_new_tokens=max_new_tokens)
        text = self.tokenizer.decode(res.ids[0], skip_special_tokens=True)
        return text, res.ids


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_transcript(transcript: str, max_words: int = 600) -> list[str]:
    """Best-effort sentence-boundary chunking for over-length transcripts.

    The card: chunk inputs longer than ~1,000 tokens at sentence boundaries.
    Raw ASR text often has no punctuation, so after sentence splitting any
    remaining run longer than ``max_words`` words is cut at word boundaries.
    (~600 words ~= 800 tokens for English.)
    """
    words = transcript.split()
    if len(words) <= max_words:
        return [transcript] if transcript.strip() else []
    pieces: list[str] = []
    for sent in _SENTENCE_SPLIT.split(transcript.strip()):
        if not sent.strip():
            continue
        if len(sent.split()) <= max_words:
            pieces.append(sent.strip())
            continue
        for i in range(0, len(sent.split()), max_words):
            pieces.append(" ".join(sent.split()[i : i + max_words]))
    return pieces


def main() -> int:
    import time

    raw = "so um i need to like send the the report by uh friday no wait make that thursday"

    print("[s1] loading model + building NormalizePipeline ...")
    t0 = time.perf_counter()
    pipe = NormalizePipeline.from_pretrained()
    print(f"[s1] built in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    text, ids = pipe.normalize(raw)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"[s1] {ids.shape[1]} tokens in {ms:.1f} ms")
    print(f"[s1] in : {raw}")
    print(f"[s1] out: {text}")
    print("[s1] expected: I need to send the report by Thursday.")
    return 0 if text.strip() == "I need to send the report by Thursday." else 1


if __name__ == "__main__":
    raise SystemExit(main())
