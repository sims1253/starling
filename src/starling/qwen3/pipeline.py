"""End-to-end fused ASR megakernel pipeline for Qwen3-ASR-1.7B.

Wires the megakernel components into one transcription path:

    mel (1,128,T) -> GraphedEncoder (cudagraph) -> enc_last_hidden (P,1024)
                  -> stock projector (eager) -> audio_embeds (P,2048)
                  -> merge into LLM inputs_embeds (replicating
                     Qwen3ASRModel.forward EXACTLY)
                  -> FusedLLMMega.generate(...) -> generated token ids
                  -> processor.decode -> transcript text

Numerics
--------
The merge step mirrors ``transformers`` ``Qwen3ASRModel.forward`` byte for
byte (verified 0.0 diff vs ``golden/inputs_embeds.pt``):

  1. embed ``input_ids`` (Qwen3 has no embedding multiplier);
  2. run the encoder + projector -> ``audio_embeds`` (pooler_output);
  3. build the placeholder mask (``input_ids == audio_token_id``) and
     ``masked_scatter`` the audio embeds into the audio-token slots.

Both the graphed encoder and the fused LLM decoder are byte-exact vs the eager
reference, so the end-to-end transcript reproduces the golden reference.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import AUDIO_TOKEN_ID, EOS_TOKEN_ID
from .encoder_mega import GraphedEncoder
from .loader import get_components, load_model_and_processor
from .llm_mega import FusedLLMMega, LLMMega


class MegaPipeline:
    """End-to-end fused ASR pipeline owning encoder + projector + fused LLM."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_cache_len: int = 4096,
        use_fused_llm: bool = True,
        steps_per_replay: int | None = None,
        encoder_warmup_iters: int = 3,
        encoder_mode: str = "cudagraph",
        prefill_use_graph: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        self.audio_token_id = int(getattr(model.config, "audio_token_id", AUDIO_TOKEN_ID))
        # Prefill eager by default: the per-prompt-length prefill graphs (cap 8,
        # evict+reset) churn the CUDA-graph allocator on a diverse-length sweep
        # and corrupt it into an illegal memory access. Eager prefill keeps the
        # allocator quiet at ~no speed cost; the decode loop stays graphed. Set
        # True to restore graphed prefill (repeated-length / latency-critical).
        self.prefill_use_graph = bool(prefill_use_graph)

        comps = get_components(model)
        # (1) encoder: eager (byte-exact) by default; cudagraph mode captures
        # the pure-tensor compute for zero launch overhead.
        self.fused_encoder = GraphedEncoder(
            comps["encoder"], warmup_iters=encoder_warmup_iters, mode=encoder_mode
        )
        # (2) projector stays the stock eager 2-layer MLP (small, cheap).
        self.projector = comps["projector"]
        self.embed_tokens = comps["language_model"].get_input_embeddings()

        # (3) LLM decoder trunk + lm_head from the TOP-LEVEL model.
        #     Use the K-step multi-step graph (byte-exact, fewer host syncs)
        #     for single-stream; fall back to the single-step fused decoder.
        self._language_model = comps["language_model"]
        self._lm_head = model.lm_head
        self._max_cache_len = int(max_cache_len)
        self.steps_per_replay = (
            None if steps_per_replay is None else max(1, int(steps_per_replay))
        )
        self._llms_by_k: dict[int, MultiStepLLMMega] = {}
        if use_fused_llm:
            self.llm = self._get_multistep_llm(self._steps_for_prompt(0))
        else:
            self.llm = LLMMega(
                self._language_model,
                self._lm_head,
                max_cache_len=self._max_cache_len,
                eos_token_id=EOS_TOKEN_ID,
                prefill_use_graph=self.prefill_use_graph,
            )
        self.use_fused_llm = use_fused_llm

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
        encoder_mode: str = "cudagraph",
        prefill_use_graph: bool = False,
    ) -> "MegaPipeline":
        model, processor = load_model_and_processor(attn_impl=attn_impl, dtype=dtype, device=device)
        return cls(
            model,
            processor,
            max_cache_len=max_cache_len,
            use_fused_llm=use_fused_llm,
            steps_per_replay=steps_per_replay,
            encoder_mode=encoder_mode,
            prefill_use_graph=prefill_use_graph,
        )

    def _steps_for_prompt(self, prompt_len: int) -> int:
        """Replay chunk size for this prompt length.

        RTX 5090 sweep on 2026-07-06: the short fixture (prompt 112, 35 output
        tokens) loses time to K-chunk overcompute, medium (prompt 305, 97
        output tokens) benefits from K=8, and long (prompt 982, 314 output
        tokens) is fastest at K=1. Explicit constructor values still override
        this auto policy.
        """
        if self.steps_per_replay is not None:
            return self.steps_per_replay
        prompt_len = int(prompt_len)
        if prompt_len <= 128:
            return 1
        if prompt_len <= 512:
            return 8
        return 1

    def _get_multistep_llm(self, steps_per_replay: int):
        from .multistep import MultiStepLLMMega

        k = max(1, int(steps_per_replay))
        llm = self._llms_by_k.get(k)
        if llm is None:
            llm = MultiStepLLMMega(
                self._language_model,
                self._lm_head,
                max_cache_len=self._max_cache_len,
                steps_per_replay=k,
                eos_token_id=EOS_TOKEN_ID,
                prefill_use_graph=self.prefill_use_graph,
            )
            self._llms_by_k[k] = llm
        return llm

    def set_prefill_use_graph(self, on: bool) -> None:
        """Toggle graphed vs eager prefill at runtime (byte-exact either way).

        Graphed prefill amortises across repeated prompt lengths (fixed-chunk
        streaming) but re-captures (and churns the allocator) on diverse audio;
        eager avoids it. The server flips this per request mode. Updates the flag
        for future decoders and every already-cached one.
        """
        on = bool(on)
        self.prefill_use_graph = on
        for llm in self._llms_by_k.values():
            llm.prefill_use_graph = on

    # ------------------------------------------------------------------ #
    # merge step (byte-exact replica of Qwen3ASRModel.forward scatter)
    # ------------------------------------------------------------------ #
    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Merge projected audio embeds into the LLM token embeddings.

        Replicates ``Qwen3ASRModel.forward`` exactly: embed input_ids, then
        ``masked_scatter`` the projected audio embeds into the ``audio_token``
        positions.
        """
        inputs_embeds = self.embed_tokens(input_ids)
        special_audio_mask = (input_ids == self.audio_token_id)
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
        af = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # audio_embeds is (P, hidden); masked_scatter reads flat in row-major.
        return inputs_embeds.masked_scatter(special_audio_mask, af)

    def encode_audio(self, input_features: torch.Tensor, input_features_mask: torch.Tensor):
        """Run encoder + eager projector. Returns ``audio_embeds``."""
        enc_lhs = self.fused_encoder(input_features, input_features_mask)
        # The encoder runs under inference_mode; clone so the (non-inference)
        # projector forward doesn't choke on inference tensors.
        return self.projector(enc_lhs.clone())

    # ------------------------------------------------------------------ #
    # full transcribe
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
    ) -> tuple[str, torch.Tensor]:
        """End-to-end ASR: mel -> transcript text.

        Args:
            input_features: mel features ``(1, 128, T)`` bf16 on cuda.
            input_ids: ``(1, L)`` token ids containing audio_token placeholders.
            input_features_mask: ``(1, T)`` validity mask for the mel frames.
            max_new_tokens: greedy decode budget.

        Returns:
            ``(transcript_text, generated_token_ids)`` where ids are
            ``(1, n_new)`` int64 on CPU (generated tokens only).
        """
        if input_features_mask is None:
            input_features_mask = torch.ones(
                input_features.shape[0], input_features.shape[2], dtype=torch.long, device=input_features.device
            )

        # (1)+(2) graphed encoder + eager projector
        audio_embeds = self.encode_audio(input_features, input_features_mask)

        # (3) merge into multimodal inputs_embeds (byte-exact vs stock)
        inputs_embeds = self.build_inputs_embeds(input_ids, audio_embeds)

        # (4) greedy generate with the fused CUDA-graph decoder
        if self.use_fused_llm:
            self.llm = self._get_multistep_llm(
                self._steps_for_prompt(inputs_embeds.shape[1])
            )
        res = self.llm.generate(inputs_embeds, max_new_tokens=max_new_tokens)

        # (5) decode generated ids to text
        try:
            text = self.processor.decode(res.ids, return_format="transcription_only")[0]
        except Exception:
            text = self.processor.batch_decode(res.ids, skip_special_tokens=True)[0]
        return text, res.ids


def main() -> int:
    import time

    from .audio import build_inputs, load_wav
    from .golden import _fixture_wav, load_golden, load_golden_text, INPUTS_EMBEDS, GREEDY_TEXT

    print("[pipeline] loading model + building MegaPipeline ...")
    t0 = time.perf_counter()
    pipe = MegaPipeline.from_pretrained()
    print(f"[pipeline] built in {time.perf_counter() - t0:.1f}s")

    wav, sr = load_wav(_fixture_wav())
    inputs = build_inputs(pipe.processor, wav, sr=sr)
    audio_seconds = wav.shape[1] / sr

    t0 = time.perf_counter()
    text, ids = pipe.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=200,
    )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[pipeline] generated {ids.shape[1]} tokens in {ms:.1f} ms")
    print(f"[pipeline] RTFx = {audio_seconds / (ms / 1000.0):.2f}x")
    print(f"[pipeline] transcript:\n{text}")

    # correctness vs golden
    gie = load_golden(INPUTS_EMBEDS).to("cuda", torch.bfloat16)
    audio_embeds = pipe.encode_audio(inputs["input_features"], inputs["input_features_mask"])
    mine = pipe.build_inputs_embeds(inputs["input_ids"], audio_embeds)
    diff = (mine.float() - gie.float()).abs().max().item()
    print(f"[pipeline] inputs_embeds max abs diff vs golden = {diff:.3e}")

    golden_text = load_golden_text().strip()
    match = text.strip() == golden_text
    print(f"[pipeline] transcript exact-match vs golden = {match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
