"""Audio loading and mel extraction for HojoAI/Hojo-ASR-V1.

The reference ``hojo-asr`` pipeline reads each wav with ``torchaudio.load``,
extracts Whisper-large-v3 log-mel features via ``WhisperFeatureExtractor``
(``chunk_length=40``, no padding), and transposes to ``(T, 128)`` -- the layout
the Qwen3-Omni audio tower consumes. These helpers mirror that path so the mel
fed to the megakernel encoder matches the reference byte-for-byte.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a wav file as mono float32 at its native sample rate.

    Args:
        path: Path to a wav file.

    Returns:
        ``(samples, sample_rate)`` where ``samples`` is a 1-D ``float32`` numpy
        array (the first channel is taken if the file is multi-channel, matching
        the reference's ``waveform[0:1, :]`` then ``squeeze(0)``).
    """
    import soundfile as sf

    samples, sr = sf.read(path)
    if samples.ndim > 1:
        # Reference takes channel 0 (waveform[0:1, :]); sf returns (n, channels).
        samples = samples[:, 0]
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


def extract_mel(
    feat_extractor: Any,
    wav: np.ndarray,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Extract Whisper-large-v3 log-mel features for one waveform.

    Mirrors ``hojo_asr.dataset._load_wav_path_to_sample`` exactly:
    ``WhisperFeatureExtractor(wav, sampling_rate=sr, return_tensors='pt',
    padding=False)`` then ``squeeze(0).transpose(0, 1)`` -> ``(T, 128)``.

    Args:
        feat_extractor: A ``WhisperFeatureExtractor`` (from the model bundle).
        wav: 1-D float32 waveform.
        sample_rate: Sample rate of ``wav`` (must be 16 kHz; resample upstream
            if different -- :func:`starling.hojo.pipeline.HojoMega.transcribe`
            handles resampling).

    Returns:
        ``(T, 128)`` float32 mel tensor (T = ``len(wav) // hop``).
    """
    feats = feat_extractor(
        np.asarray(wav, dtype=np.float32),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
    ).input_features
    return feats.squeeze(0).transpose(0, 1).contiguous()


def spectrogram_lengths(mel_T: int) -> torch.Tensor:
    """Per-utterance mel length as a (1,) int64 tensor (batch_size=1).

    The reference batches with ``padding_for_batch`` (sort by length, pad). For
    single-utterance ASR the spectrogram is already unpadded, so the length is
    just ``mel_T``.
    """
    return torch.tensor([mel_T], dtype=torch.int64)
