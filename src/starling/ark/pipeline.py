"""End-to-end fused ASR megakernel pipeline for AutoArk-AI/ARK-ASR-3B.

This module wires the three existing megakernel components into one end-to-end
transcription path:

    wav -> mel (1,128,mel_T)
        -> FusedEncoder (cudagraph) -> audio_features (1,N,2048)
        -> build_inputs_embeds (byte-exact audio-embedding injection)
        -> MultiStepLLMMega.generate(...) -> generated token ids
        -> tokenizer.decode -> transcript text

Numerics
--------
The audio-embedding injection mirrors
``ArkasrForConditionalGeneration._inject_audio_embeddings`` byte for byte:

  1. zero out the audio-token positions in ``input_ids``;
  2. look up ``embed_tokens`` (Qwen2.5 has NO embedding multiplier);
  3. scatter the adapter's audio features into the audio-token slots.

Because both the fused encoder and the fused LLM decoder are byte-exact vs the
eager reference, the end-to-end transcript reproduces the golden reference
exactly.

Public API
----------
``MegaPipeline(model, processor, *, steps_per_replay=None, max_cache_len=4096)``
``MegaPipeline.from_pretrained(...)``
``MegaPipeline.transcribe(audio_path_or_array, instruction=..., max_new_tokens=200) -> (text, token_ids)``
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

from .audio import build_inputs_embeds, build_prompt_ids, extract_mel, read_wav
from .config import DEFAULT_INSTRUCTION, ENCODER_MAX_SOURCE_POSITIONS, EOS_TOKEN_ID
from .encoder_mega import FusedEncoder
from .loader import get_components
from .multistep import MultiStepLLMMega


class MegaPipeline:
    """End-to-end fused ARK-ASR-3B pipeline owning encoder + fused LLM.

    Parameters
    ----------
    model : ArkasrForConditionalGeneration
        The fully loaded top-level model (lm_head + embed_tokens live on it /
        its decoder trunk).
    processor : ARK-ASR processor
        Provides ``feature_extractor`` (mel) and ``tokenizer`` (text).
    encoder_mode : {"cudagraph"}
        Selects the CUDA-graphed encoder (``"cudagraph"``, the byte-exact,
        zero-launch-overhead default). Kept as an API hook for a future eager
        A/B path; currently only ``"cudagraph"`` is implemented.
    steps_per_replay : int | None
        K -- number of decode steps captured per CUDA-graph replay. ``None``
        selects K from the prompt length (K=2 short, K=4 medium, K=16 long).
    max_cache_len : int
        Fixed K/V cache length (must fit prompt T + max_new_tokens).
    shape_bucketing : bool
        If True (default), right-pad the mel up to a canonical bucket ``mel_T``
        (next multiple of ``mel_bucket_frames``, capped at the Whisper 3000-frame
        limit) before the encoder. Diverse clip lengths then hit the SAME
        captured encoder CUDA graph, so the one-off per-shape capture cost is
        amortised instead of re-paid (and, crucially, the encoder graph pool
        stops accumulating one ~GB-scale graph per distinct length -- the cause
        of the n=50 leaderboard OOM). The padding is trailing zeros (silence,
        which the Whisper encoder is trained on); the adapter's extra trailing
        features are truncated back to the natural audio-token count by
        ``build_inputs_embeds``, so the prompt is unchanged. **Text-byte-exact**
        on the fixtures (the decoded transcript is identical; the mirror of the
        parakeet ``shape_bucketing`` fix). Set False for strict per-shape
        capture.
    mel_bucket_frames : int
        Bucket granularity in mel frames (default 512 = ~5.1 s). Clips pad up to
        the next multiple, capped at 3000 (the encoder's max source positions),
        collapsing the ~30-distinct-length leaderboard split to ~6 buckets.
        ``1`` disables bucketing (equivalent to ``shape_bucketing=False``).
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        encoder_mode: str = "cudagraph",
        steps_per_replay: Optional[int] = None,
        max_cache_len: int = 4096,
        shape_bucketing: bool = True,
        mel_bucket_frames: int = 512,
        prefill_use_graph: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        self.encoder_mode = encoder_mode
        self.steps_per_replay = steps_per_replay
        self.max_cache_len = int(max_cache_len)
        # Prefill eager by default: the per-prompt-length prefill graphs are the
        # dominant memory accumulator on a diverse-length sweep (they overflow
        # VRAM into WSL shared memory -> RTFx collapse). Eager prefill keeps
        # memory flat at ~no speed cost; the decode loop stays graphed. Set True
        # to restore graphed prefill (repeated-length / latency-critical use).
        self.prefill_use_graph = bool(prefill_use_graph)
        # Shape bucketing: pad mel up to the next multiple of mel_bucket_frames
        # (capped at the 3000-frame Whisper limit) so clips of similar length hit
        # the same captured encoder graph -- bounds the encoder graph pool and
        # amortises capture. mel_bucket_frames=1 disables it. See class docstring.
        self.shape_bucketing = bool(shape_bucketing) and int(mel_bucket_frames) > 1
        self.mel_bucket_frames = max(1, int(mel_bucket_frames))

        comps = get_components(model)
        # ONE shared CUDA graph pool for encoder + LLM captures, so LRU eviction
        # of a graph (``del graph``) frees only that graph's blocks without
        # corrupting the context (the bug that caused cudaErrorIllegalAddress).
        self._graph_pool = torch.cuda.graph_pool_handle()
        # (1) fused audio encoder (cudagraph = byte-exact + zero launch overhead).
        self.fused_encoder = FusedEncoder(comps["audio_encoder"], graph_pool=self._graph_pool)
        # embed_tokens used by the audio-embedding injection step.
        self.embed_tokens = comps["embed_tokens"]
        self._language_model = comps["language_model"]
        self._lm_head = model.lm_head
        self._llms: dict[int, MultiStepLLMMega] = {}
        self.llm = self._get_llm(0)

    def _steps_for_shape(self, prompt_len: int) -> int:
        """Choose a decode graph replay length from the prompt length."""
        if self.steps_per_replay is not None:
            return max(1, int(self.steps_per_replay))
        if prompt_len <= 160:
            return 2
        if prompt_len <= 512:
            return 4
        return 16

    def _get_llm(self, prompt_len: int) -> MultiStepLLMMega:
        k = self._steps_for_shape(prompt_len)
        llm = self._llms.get(k)
        if llm is None:
            llm = MultiStepLLMMega(
                self._language_model,
                self._lm_head,
                max_cache_len=self.max_cache_len,
                steps_per_replay=k,
                prefill_use_graph=self.prefill_use_graph,
                graph_pool=self._graph_pool,
            )
            self._llms[k] = llm
        self.llm = llm
        return llm

    def set_prefill_use_graph(self, on: bool) -> None:
        """Toggle graphed vs eager prefill at runtime (byte-exact either way).

        Graphed prefill amortises across repeated prompt lengths (fixed-chunk
        streaming) but re-captures per length on diverse audio; eager avoids the
        capture. The server flips this per request mode. Updates the flag for
        future decoders and every already-cached one, so the change takes effect
        on the next transcribe.
        """
        on = bool(on)
        self.prefill_use_graph = on
        for llm in self._llms.values():
            llm.prefill_use_graph = on

    # ------------------------------------------------------------------ #
    # shape bucketing (pad mel up so diverse lengths share one encoder graph)
    # ------------------------------------------------------------------ #
    def _bucket_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Right-pad ``mel`` (B, 128, mel_T) up to a canonical bucket ``mel_T``.

        Pads the time axis up to the next multiple of ``mel_bucket_frames``,
        capped at ``2 * ENCODER_MAX_SOURCE_POSITIONS`` (the 3000-frame Whisper
        limit -- exceeding it overruns the encoder's positional embeddings). The
        padding is trailing zeros, so the adapter emits a few extra trailing
        features that ``build_inputs_embeds`` truncates back to the natural
        audio-token count -- the prompt and injected slots are unchanged. Returns
        ``mel`` untouched (no copy) when bucketing is disabled or the length is
        already on the grid (e.g. long audio already capped at 3000).
        """
        if not self.shape_bucketing:
            return mel
        mel_T = int(mel.shape[2])
        g = self.mel_bucket_frames
        max_T = 2 * ENCODER_MAX_SOURCE_POSITIONS  # 3000: encoder positional cap
        bucket_T = min(((mel_T + g - 1) // g) * g, max_T)
        if bucket_T <= mel_T:
            return mel  # already on the grid / at the cap
        B, C, _ = mel.shape
        out = mel.new_zeros((B, C, bucket_T))
        out[:, :, :mel_T].copy_(mel)
        return out

    # ------------------------------------------------------------------ #
    # convenience constructor
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder_mode: str = "cudagraph",
        steps_per_replay: Optional[int] = None,
        max_cache_len: int = 4096,
        attn_impl: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        shape_bucketing: bool = True,
        mel_bucket_frames: int = 512,
        prefill_use_graph: bool = False,
    ) -> "MegaPipeline":
        """Load the model + processor and wrap them in a MegaPipeline."""
        from .loader import load_model_and_processor

        model, processor = load_model_and_processor(
            attn_impl=attn_impl, dtype=dtype, device=device
        )
        return cls(
            model,
            processor,
            encoder_mode=encoder_mode,
            steps_per_replay=steps_per_replay,
            max_cache_len=max_cache_len,
            shape_bucketing=shape_bucketing,
            mel_bucket_frames=mel_bucket_frames,
            prefill_use_graph=prefill_use_graph,
        )

    # ------------------------------------------------------------------ #
    # full transcribe
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        audio_path_or_array: Union[str, np.ndarray],
        instruction: str = DEFAULT_INSTRUCTION,
        max_new_tokens: int = 200,
    ) -> tuple[str, torch.Tensor]:
        """End-to-end ASR: wav -> transcript text.

        Args:
            audio_path_or_array: Path to a 16 kHz wav file, or a 1-D float32
                numpy waveform at 16 kHz.
            instruction: The user instruction text wrapped after the audio block.
            max_new_tokens: Greedy decode budget.

        Returns:
            ``(transcript_text, generated_token_ids)`` where the ids are
            ``(1, n_new)`` int64 on CPU (the generated tokens only, excluding
            the prompt).
        """
        # (1) read wav (if path) else assume a numpy waveform.
        if isinstance(audio_path_or_array, str):
            wav, _sr = read_wav(audio_path_or_array)
        else:
            wav = np.ascontiguousarray(audio_path_or_array, dtype=np.float32)

        # (2) mel -> bf16 on cuda.
        mel = extract_mel(self.processor, [wav])
        mel = mel.to(dtype=torch.bfloat16, device="cuda")
        # The audio-token count derives from the *uncapped* mel-frame count
        # (raw audio length / hop_length), matching the eager reference's
        # ``processor.calculate_audio_token_count``; the Whisper feature
        # extractor may cap the mel at 3000 frames for long audio, in which case
        # the adapter emits fewer features than audio slots and the injection
        # zero-pads the remainder.
        hop_length = int(getattr(self.processor.feature_extractor, "hop_length", 160))
        n_mel_frames = int(wav.shape[0]) // hop_length

        # (3) prompt token ids (with N <|audio|> placeholders) -> cuda.
        input_ids = build_prompt_ids(
            self.processor.tokenizer, instruction, n_mel_frames=n_mel_frames
        ).to("cuda")

        # (3b) shape-bucket the mel so diverse clip lengths share ONE captured
        # encoder graph (bounds the encoder graph pool; amortises capture). The
        # audio-token count N above derives from the uncapped raw length, so it
        # is unaffected; the extra trailing adapter features from the padding are
        # truncated by build_inputs_embeds. Text-byte-exact (see class docstring).
        mel = self._bucket_mel(mel)

        # (4) fused encoder -> audio features (1, N, 2048).
        audio_features = self.fused_encoder(mel)

        # (5) byte-exact audio-embedding injection -> inputs_embeds (1, T, 2048).
        inputs_embeds = build_inputs_embeds(self.model, input_ids, audio_features)

        # (6) K-step CUDA-graph greedy generate.
        llm = self._get_llm(int(inputs_embeds.shape[1]))
        res = llm.generate(
            inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS_TOKEN_ID,
        )

        # (7) decode generated ids to text.
        text = self.processor.tokenizer.decode(res.ids[0], skip_special_tokens=True)
        return text, res.ids

    def prewarm(self) -> None:
        """Pre-capture CUDA graphs with a short dummy utterance.

        The encoder graph and LLM decode graphs are lazily initialized on the
        first real call. This method runs a short dummy transcribe (a 5s zero
        array) to pay all capture costs upfront, eliminating first-utterance
        latency for live/streaming use.
        """
        dummy = np.zeros(int(5.0 * 16000), dtype=np.float32)
        self.transcribe(
            dummy,
            instruction=DEFAULT_INSTRUCTION,
            max_new_tokens=8,
        )
        torch.cuda.synchronize()
