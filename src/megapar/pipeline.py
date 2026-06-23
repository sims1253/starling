"""End-to-end fused ASR megakernel pipeline for Granite-Speech-4.1-2b.

This module wires the three existing megakernel components into one
end-to-end transcription path:

    mel (1,T,160) -> FusedEncoder (cudagraph) -> encoder_last_hidden (1,T,1024)
                  -> stock projector (eager BLIP2 q-former) -> audio_embeds (1,N,2048)
                  -> merge into LLM inputs_embeds (replicating
                     GraniteSpeechModel.get_merged_audio_embeddings EXACTLY)
                  -> FusedLLMMega.generate(...) -> generated token ids
                  -> tokenizer.batch_decode -> transcript text

Numerics
--------
The merge step mirrors ``transformers`` ``get_merged_audio_embeddings`` byte
for byte (verified 0.0 diff vs ``golden/inputs_embeds.pt``):

  1. zero out the audio-token positions in ``input_ids``;
  2. look up ``embed_tokens`` (NO Granite embedding multiplier here -- that is
     applied INSIDE ``GraniteModel.forward`` during prefill, exactly as the
     stock path does);
  3. optionally select audio rows via ``input_features_mask``;
  4. ``masked_scatter`` the projected audio embeds into the audio-token slots.

Because both the fused encoder and the fused LLM decoder are byte-exact vs the
eager reference, the end-to-end transcript reproduces the golden reference
exactly.

Public API
----------
``MegaPipeline(model, processor, *, encoder_mode="cudagraph", use_fused_llm=True)``
``MegaPipeline.from_pretrained(...)``
``MegaPipeline.transcribe(input_features, input_ids, input_features_mask=None,
                          max_new_tokens=200) -> (text, token_ids)``
``MegaPipeline.build_inputs_embeds(input_ids, audio_embeds, input_features_mask=None)``
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import AUDIO_TOKEN_ID, LLM_EOS_TOKEN_ID
from .encoder_mega import FusedEncoder
from .loader import get_components, load_model_and_processor
from .llm_mega import FusedLLMMega, LLMMega


class MegaPipeline:
    """End-to-end fused ASR pipeline owning encoder + projector + fused LLM.

    Parameters
    ----------
    model : GraniteSpeechForConditionalGeneration
        The fully loaded top-level speech model (lm_head lives on it).
    processor : GraniteSpeech processor
    encoder_mode : {"eager","cudagraph","compile","triton"}
        Forwarded to :class:`FusedEncoder`. ``"cudagraph"`` is the byte-exact,
        zero-launch-overhead default.
    use_fused_llm : bool
        If True (default) use :class:`FusedLLMMega` (fused Triton elementwise
        kernels); else fall back to :class:`LLMMega` (model's own layers).
        Ignored when ``flags.multistep_graph`` is True (the K-step decoder is
        always fused).
    flags : OptFlags or dict or None
        Runtime feature flags (see :mod:`megapar.flags`).  ``None`` uses the
        process-global default.  ``multistep_graph=True`` (default) selects
        :class:`MultiStepLLMMega` for lower per-token sync overhead.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        encoder_mode: str = "cudagraph",
        use_fused_llm: bool = True,
        flags: Any = None,
    ) -> None:
        from .flags import OptFlags, get_default_flags

        if flags is None:
            flags = get_default_flags()
        elif isinstance(flags, dict):
            flags = OptFlags(**flags)
        self.flags = flags

        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)

        comps = get_components(model)
        # (1) fused encoder (cudagraph = byte-exact + zero launch overhead)
        self.fused_encoder = FusedEncoder(comps["encoder"], mode=encoder_mode)
        # (2) projector stays the stock eager BLIP2 q-former (no sdpa).
        self.projector = comps["projector"]
        # embed_tokens used by the merge step (== language_model.embed_tokens).
        self.embed_tokens = comps["language_model"].get_input_embeddings()

        # (3) LLM decoder trunk + lm_head from the TOP-LEVEL model.
        #     ``multistep_graph`` (byte-exact, default on) selects the K-step
        #     CUDA-graph decoder for lower per-token sync overhead; otherwise
        #     fall back to the single-step fused/model-forward decoder.
        #     ``quantized_weights`` (tolerance mode only) selects the weight-only
        #     INT8 decoder (:class:`megapar.quant.QuantLLMMega`).
        if flags.quantized_weights:
            from .quant import QuantLLMMega

            self.llm = QuantLLMMega(
                comps["language_model"],
                model.lm_head,
                max_cache_len=640,
            )
        elif flags.multistep_graph:
            from .multistep import MultiStepLLMMega

            self.llm = MultiStepLLMMega(
                comps["language_model"],
                model.lm_head,
                max_cache_len=640,
            )
        else:
            llm_cls = FusedLLMMega if use_fused_llm else LLMMega
            self.llm = llm_cls(
                comps["language_model"],
                model.lm_head,
                max_cache_len=640,
            )
        self.use_fused_llm = use_fused_llm

    # ------------------------------------------------------------------ #
    # convenience constructor
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder_mode: str = "cudagraph",
        use_fused_llm: bool = True,
        attn_impl: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        flags: Any = None,
    ) -> "MegaPipeline":
        """Load the model + processor and wrap them in a MegaPipeline."""
        model, processor = load_model_and_processor(
            attn_impl=attn_impl, dtype=dtype, device=device
        )
        return cls(
            model,
            processor,
            encoder_mode=encoder_mode,
            use_fused_llm=use_fused_llm,
            flags=flags,
        )

    # ------------------------------------------------------------------ #
    # merge step (byte-exact replica of get_merged_audio_embeddings)
    # ------------------------------------------------------------------ #
    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Merge projected audio embeds into the LLM token embeddings.

        Replicates ``GraniteSpeechModel.get_merged_audio_embeddings`` exactly.
        The Granite ``embedding_multiplier`` is NOT applied here -- it is
        applied inside ``GraniteModel.forward`` during prefill (matching the
        stock path and the golden ``inputs_embeds.pt``).
        """
        is_audio_index = input_ids == AUDIO_TOKEN_ID
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.embed_tokens(llm_input_ids)

        af = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        if input_features_mask is not None:
            af = af[input_features_mask]

        special_audio_mask = is_audio_index.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(special_audio_mask, af)

    # ------------------------------------------------------------------ #
    # audio features (encoder + projector)
    # ------------------------------------------------------------------ #
    def encode_audio(
        self, input_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run fused encoder + eager projector.

        Returns ``(encoder_last_hidden, audio_embeds)``.
        """
        encoder_last_hidden = self.fused_encoder(input_features)
        audio_embeds = self.projector(encoder_last_hidden)
        return encoder_last_hidden, audio_embeds

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
        speculative: bool = False,
    ) -> tuple[str, torch.Tensor]:
        """End-to-end ASR: mel -> transcript text.

        Args:
            input_features: mel features ``(1, T, 160)`` (bf16 recommended;
                cast to bf16 internally by the fused encoder).
            input_ids: ``(1, L)`` token ids containing ``AUDIO_TOKEN_ID``
                placeholders.
            input_features_mask: optional bool mask selecting valid audio rows
                before scatter (matches the stock ``get_merged_audio_embeddings``
                behaviour).
            max_new_tokens: greedy decode budget.
            speculative: if True, use self-speculative decoding with the
                encoder's BPE CTC head.  The output is **byte-identical** to
                the non-speculative greedy path (greedy-verify guarantee).

        Returns:
            ``(transcript_text, generated_token_ids)`` where the ids are
            ``(1, n_new)`` int64 on CPU (the generated tokens only, excluding
            the prompt).
        """
        if speculative:
            return self._transcribe_speculative(
                input_features, input_ids, input_features_mask, max_new_tokens
            )

        # (1)+(2) fused encoder + eager projector
        _enc, audio_embeds = self.encode_audio(input_features)

        # (3) merge into multimodal inputs_embeds (byte-exact vs stock)
        inputs_embeds = self.build_inputs_embeds(
            input_ids, audio_embeds, input_features_mask
        )

        # (4) greedy generate with the fused CUDA-graph decoder
        res = self.llm.generate(
            inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=LLM_EOS_TOKEN_ID,
        )

        # (5) decode generated ids to text
        text = self.processor.tokenizer.batch_decode(
            res.ids, skip_special_tokens=True
        )[0]
        return text, res.ids

    # ------------------------------------------------------------------ #
    # speculative decoding path
    # ------------------------------------------------------------------ #
    def _get_spec_components(self):
        """Lazily initialize the CTC draft extractor + speculative decoder."""
        if not hasattr(self, "_ctc_draft"):
            from .speculative import CTCBPEDraft, SpeculativeDecoder, load_out_llm

            out_llm = load_out_llm(device="cuda", dtype=self.dtype)
            self._ctc_draft = CTCBPEDraft(
                self.fused_encoder, out_llm,
                device="cuda", dtype=self.dtype,
            )
            self._spec_decoder = SpeculativeDecoder(self.llm, self.embed_tokens)
        return self._ctc_draft, self._spec_decoder

    @torch.inference_mode()
    def _transcribe_speculative(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        input_features_mask: Optional[torch.Tensor],
        max_new_tokens: int,
    ) -> tuple[str, torch.Tensor]:
        """Self-speculative greedy transcribe (byte-identical to greedy)."""
        ctc_draft, spec_decoder = self._get_spec_components()

        # (1) Run encoder eagerly ONCE, capturing mid-layer for the draft.
        #     The resulting enc_hidden is reused for the projector -> audio_embeds.
        mid_h, enc_hidden = ctc_draft.encode_with_mid(input_features)

        # (2) Extract the BPE CTC draft (cheap, deterministic).
        draft = ctc_draft.draft(enc_hidden, mid_h)

        # (3) Project enc_hidden -> audio_embeds (same as non-spec path).
        audio_embeds = self.projector(enc_hidden)

        # (4) Merge into multimodal inputs_embeds (byte-exact vs stock).
        inputs_embeds = self.build_inputs_embeds(
            input_ids, audio_embeds, input_features_mask
        )

        # (5) Self-speculative greedy generate.
        res = spec_decoder.generate(
            inputs_embeds, draft,
            max_new_tokens=max_new_tokens,
            eos_token_id=LLM_EOS_TOKEN_ID,
        )

        # (6) Decode generated ids to text.
        text = self.processor.tokenizer.batch_decode(
            res.ids, skip_special_tokens=True
        )[0]
        return text, res.ids


def main() -> int:
    """Quick CLI smoke: transcribe the sample audio and print the result."""
    import time

    from .audio import build_inputs, load_sample_audio
    from .golden import load_golden, load_golden_text

    print("[pipeline] loading model + building MegaPipeline ...")
    t0 = time.perf_counter()
    pipe = MegaPipeline.from_pretrained()
    print(f"[pipeline] built in {time.perf_counter() - t0:.1f}s")

    wav, sr = load_sample_audio()
    inputs = build_inputs(pipe.processor, wav)
    audio_seconds = wav.shape[1] / sr

    t0 = time.perf_counter()
    text, ids = pipe.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=100,
    )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[pipeline] generated {ids.shape[1]} tokens in {ms:.1f} ms")
    print(f"[pipeline] RTFx = {audio_seconds / (ms / 1000.0):.2f}x")
    print(f"[pipeline] transcript:\n{text}")

    # correctness vs golden
    gie = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
    _enc, audio_embeds = pipe.encode_audio(inputs["input_features"])
    mine = pipe.build_inputs_embeds(
        inputs["input_ids"],
        audio_embeds,
        inputs.get("input_features_mask"),
    )
    diff = (mine.float() - gie.float()).abs().max().item()
    print(f"[pipeline] inputs_embeds max abs diff vs golden = {diff:.3e}")

    golden_text = load_golden_text().strip()
    if "ASSISTANT:" in golden_text:
        golden_resp = golden_text.split("ASSISTANT:", 1)[1].strip()
    else:
        golden_resp = golden_text
    match = text.strip() == golden_resp
    print(f"[pipeline] transcript exact-match vs golden = {match}")

    golden_gen = load_golden("greedy_ids.pt")[0, 271:]
    tok_match = bool((ids[0] == golden_gen).all().item())
    print(f"[pipeline] token exact-match vs golden = {tok_match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
