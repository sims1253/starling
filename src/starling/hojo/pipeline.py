"""End-to-end ASR pipeline for HojoAI/Hojo-ASR-V1.

Wires the eager audio encoder (tower + Conformer) + the beam-4 Qwen3-4B decode
into one transcription path:

    audio (np) -> Whisper mel -> FusedEncoder (eager) -> speech_embeds
                -> LLMMega.generate (beam-4, stock Qwen3) -> token ids
                -> tokenizer.decode -> strip special tokens -> transcript text

The encoder runs eager (see :mod:`starling.hojo.encoder_mega`); the decoder is
beam-4 via ``decoder_model.generate`` (byte-exact with the golden oracle). A
custom CUDA-graph-captured beam loop is future work.

Public API
----------
``HojoMega.from_pretrained() -> HojoMega``
``HojoMega.transcribe(audio_np, sample_rate=16000, max_new_tokens=None) -> str``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .audio import extract_mel, spectrogram_lengths
from .config import AUTOMIX_DTYPE
from .encoder_mega import FusedEncoder
from .llm_mega import BeamDecodeConfig, GenerateResult, LLMMega


class HojoMega:
    """End-to-end Hojo-ASR-V1 ASR pipeline.

    Parameters
    ----------
    model : HOJO_ASR
        The fully loaded model (runs under ``transformers 4.57`` in
        ``.venv-hojo``).
    cfg : BeamDecodeConfig, optional
        Beam-search hyperparameters (defaults match the golden decode block).
    """

    def __init__(
        self,
        model: Any,
        cfg: BeamDecodeConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or BeamDecodeConfig()
        self.device = next(model.parameters()).device
        self.encoder = FusedEncoder(model, device=str(self.device))
        self.llm = LLMMega(model, cfg=self.cfg)
        self.tokenizer = getattr(model, "tokenizer", None)
        self.feat_extractor = getattr(model, "feat_extractor", None)
        self._autocast_dtype = getattr(torch, AUTOMIX_DTYPE, torch.float16)

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        *,
        folder_path: str | None = None,
        device: str = "cuda:0",
        cfg: BeamDecodeConfig | None = None,
    ) -> "HojoMega":
        """Load the model + tokenizer and build the pipeline."""
        from .loader import load_model
        model = load_model(folder_path=folder_path, device=device)
        return cls(model, cfg=cfg)

    # ------------------------------------------------------------------ #
    # audio -> mel (on device, batch_size=1)
    # ------------------------------------------------------------------ #
    def _audio_to_mel(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """audio -> ``(spectrogram (1,T,128), spectrogram_lengths (1,))``.

        Mirrors ``hojo_asr.dataset._load_wav_path_to_sample`` +
        ``padding_for_batch`` for a single utterance (no padding needed at B=1).
        """
        audio_np = _resample(audio_np, sample_rate, 16000)
        mel = extract_mel(self.feat_extractor, audio_np, sample_rate=16000)  # (T,128)
        mel = mel.to(self.device)
        spectrogram = mel.unsqueeze(0)                                       # (1,T,128)
        lengths = spectrogram_lengths(int(mel.shape[0])).to(self.device)     # (1,)
        return spectrogram, lengths

    # ------------------------------------------------------------------ #
    # transcribe
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        max_new_tokens: int | None = None,
    ) -> str:
        """Transcribe one audio clip -> text (byte-exact with the golden).

        Args:
            audio_np: 1-D float32 waveform.
            sample_rate: Sample rate of ``audio_np`` (resampled to 16 kHz).
            max_new_tokens: Cap on generated tokens. ``None`` uses the reference
                formula ``min(200, N*2 + 10)``.

        Returns:
            The transcript string, with ``<|im_end|>`` / ``<|endoftext|>``
            stripped and whitespace-trimmed (matching the reference
            ``run_infer`` post-processing).
        """
        spectrogram, lengths = self._audio_to_mel(audio_np, sample_rate)
        speech_embeddings, speech_attn = self.encoder(spectrogram, lengths)
        res: GenerateResult = self.llm.generate(
            speech_embeddings,
            speech_attn,
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer,
        )
        return _clean_transcript(res.text)

    @torch.inference_mode()
    def transcribe_gen_ids(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        max_new_tokens: int | None = None,
    ) -> tuple[str, list[int]]:
        """Transcribe and also return the raw generated token ids.

        Useful for byte-exact parity checks against ``golden gen_ids`` (the ids
        are the authoritative gate).
        """
        spectrogram, lengths = self._audio_to_mel(audio_np, sample_rate)
        speech_embeddings, speech_attn = self.encoder(spectrogram, lengths)
        res: GenerateResult = self.llm.generate(
            speech_embeddings,
            speech_attn,
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer,
        )
        text = _clean_transcript(res.text)
        return text, res.ids[0].tolist()


# ---------------------------------------------------------------------------
# post-processing (matches hojo_asr.HOJO_ASR.run_infer)
# ---------------------------------------------------------------------------
def _clean_transcript(text: str) -> str:
    """Strip ``<|im_end|>`` / ``<|endoftext|>`` and trim whitespace.

    Matches the reference ``run_infer`` post-processing:
    ``text.replace('<|im_end|>','').replace('<|endoftext|>','').strip()``.
    """
    return (
        text.replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .strip()
    )


# ---------------------------------------------------------------------------
# preprocessing helpers (mirror upstream dataset.py, transformers-agnostic)
# ---------------------------------------------------------------------------
def _resample(audio_np: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio_np, dtype=np.float32)
    import torchaudio
    t = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    t = torchaudio.functional.resample(t, src_sr, dst_sr)
    return t.squeeze(0).numpy().astype(np.float32)
