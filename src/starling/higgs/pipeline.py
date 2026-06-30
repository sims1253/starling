"""End-to-end ASR megakernel pipeline for bosonai/higgs-audio-v3-stt.

Wires the audio->mel collator + the model's eager encoder/projector prefill +
the CUDA-graph-captured Qwen3 decode (single- or multi-step) into one
transcription path:

    audio (np) -> HiggsAudioSampleCollator -> merged batch
                -> model.forward (eager prefill, fills StaticCache) -> first token
                -> LLMMega / MultiStepLLMMega.generate (graphed decode) -> token ids
                -> tokenizer.decode -> transcript text

The encoder/projector prefill runs eager (the Whisper tower + projector are run
once per clip; their cost is fixed and small relative to the decode loop, and
graphing them is not byte-exact -- the same finding as granite's conformer).
The Qwen3 decode -- the launch-bound bottleneck -- is CUDA-graph-captured, which
is byte-exact with the eager reference.

Public API
----------
``HiggsMega.from_pretrained() -> HiggsMega``
``HiggsMega.transcribe(audio_np, sample_rate=16000, max_new_tokens=512) -> str``
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch

from .config import EOS_TOKEN_IDS, MODEL_ID
from .llm_mega import GenerateResult, LLMMega

DEFAULT_PROMPT = (
    "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
)
ENABLE_THINKING = True


def _build_input_tokens(tokenizer, user_prompt: str, enable_thinking: bool = True) -> list[int]:
    """Build the ChatML input token sequence (matches upstream ``transcribe``)."""
    def enc(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    input_tokens: list[int] = []
    input_tokens += enc("<|im_start|>user\n")
    input_tokens += enc(user_prompt)
    input_tokens += enc("<|audio_bos|><|AUDIO|><|audio_eos|>")
    input_tokens += enc("<|im_end|>\n")
    input_tokens += enc("<|im_start|>assistant\n")
    if not enable_thinking:
        input_tokens += enc("<think>\n\n</think>\n\n")
    return input_tokens


def _parse_output(full_text: str) -> str:
    """Extract the transcription from a decoded output string (upstream parity)."""
    parts = full_text.split("assistant\n")
    hyp = parts[-1] if len(parts) > 1 else full_text
    hyp = re.sub(r"<think>.*?</think>", "", hyp, flags=re.DOTALL)
    if "<think>" in hyp:
        hyp = hyp[hyp.index("<think>") + len("<think>"):]
    hyp = re.sub(r"<\|.*?\|>", "", hyp)
    return hyp.strip()


class HiggsMega:
    """End-to-end Higgs-Audio-v3 ASR megakernel pipeline.

    Parameters
    ----------
    model : HiggsAudio3Model
        The fully loaded model (runs under ``transformers 4.51`` in ``.venv-higgs``).
    tokenizer : Qwen3 tokenizer
    decoder : {"single", "multi"}
        ``"single"`` uses :class:`LLMMega` (one graph replay per token);
        ``"multi"`` uses :class:`MultiStepLLMMega` (K-step replays). ``"multi"``
        wins on longer audio where decode-step count is high.
    max_cache_len : int
        Static KV cache length. Must exceed the merged prompt length + desired
        new tokens (long audio expands to many whisper frames; use >= 2048 for
        ~75s clips).
    steps_per_replay : int
        K for the multi-step decoder.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        decoder: str = "multi",
        max_cache_len: int = 2048,
        steps_per_replay: int = 8,
        compile_decode: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.collator = _make_collator(model)

        if decoder == "multi":
            from .multistep import MultiStepLLMMega
            self.llm: LLMMega = MultiStepLLMMega(
                model, max_cache_len=max_cache_len, steps_per_replay=steps_per_replay,
                compile_decode=compile_decode,
            )
        elif decoder == "single":
            from .fused_decode import FusedLLMMega
            self.llm = FusedLLMMega(model, max_cache_len=max_cache_len, compile_decode=compile_decode)
        else:
            raise ValueError(f"unknown decoder {decoder!r}; use 'single' or 'multi'")

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_impl: str = "eager",
        **kwargs: Any,
    ) -> "HiggsMega":
        """Load the model + tokenizer and build the pipeline."""
        from .loader import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(dtype=dtype, device=device, attn_impl=attn_impl)
        return cls(model, tokenizer, **kwargs)

    # ------------------------------------------------------------------ #
    # audio -> batch
    # ------------------------------------------------------------------ #
    def _audio_to_batch(self, audio_np: np.ndarray, sample_rate: int) -> dict[str, torch.Tensor]:
        audio_np = _resample(audio_np, sample_rate, 16000)
        input_ids = _build_input_tokens(self.tokenizer, DEFAULT_PROMPT, ENABLE_THINKING)
        sample = _build_sample(audio_np, input_ids)
        batch = asdict(self.collator([sample]))
        return {
            k: (v.to(self.device).contiguous() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    # ------------------------------------------------------------------ #
    # transcribe
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        max_new_tokens: int = 512,
    ) -> str:
        """Transcribe one audio clip -> text (byte-exact with the golden oracle)."""
        batch = self._audio_to_batch(audio_np, sample_rate)
        res: GenerateResult = self.llm.generate(
            batch,
            max_new_tokens=max_new_tokens,
            eos_token_ids=EOS_TOKEN_IDS,
            tokenizer=self.tokenizer,
        )
        # Decode the FULL sequence (prompt + generated) then parse, matching the
        # upstream ``transcribe()`` output extraction.
        full_ids = torch.cat(
            [batch["input_ids"][0].cpu(), res.ids[0].cpu()], dim=0
        ).unsqueeze(0)
        full_text = self.tokenizer.decode(full_ids[0], skip_special_tokens=False)
        return _parse_output(full_text)


# ---------------------------------------------------------------------------
# preprocessing helpers (mirror upstream transcribe.py, transformers-agnostic)
# ---------------------------------------------------------------------------
def _resample(audio_np: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio_np, dtype=np.float32)
    import torchaudio
    t = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    t = torchaudio.functional.resample(t, src_sr, dst_sr)
    return t.squeeze(0).numpy().astype(np.float32)


def _build_sample(audio_np: np.ndarray, input_ids: list[int], sample_rate: int = 16000):
    from .vendor import ChatMLDatasetSample
    return ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_ids),
        label_ids=torch.LongTensor([-100] * len(input_ids)),
        audio_ids_concat=torch.zeros((1, 0), dtype=torch.long),
        audio_ids_start=torch.tensor([0], dtype=torch.long),
        audio_waveforms_concat=torch.tensor(audio_np, dtype=torch.float32),
        audio_waveforms_start=torch.tensor([0]),
        audio_sample_rate=torch.tensor([sample_rate]),
        audio_speaker_indices=torch.tensor([0]),
    )


def _make_collator(model: Any):
    from .loader import make_collator
    return make_collator(model)
