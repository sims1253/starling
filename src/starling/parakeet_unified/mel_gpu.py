"""GPU mel feature extractor for parakeet-unified-en-0.6b.

Pure-GPU port of NeMo's ``AudioToMelSpectrogramPreprocessor`` for this model.
The math is identical to the sibling ``starling.parakeet.mel_gpu`` (same NeMo
frontend: 128 mels, n_fft 512, hop 160, win 400, hann(periodic=False),
preemphasis 0.97, mag_power 2, per_feature CMVN, log guard 2**-24). The only
differences vs the TDT mel:

* the mel filterbank + Hann window come from the **state_dict**
  (``preprocessor.featurizer.{fb,window}``), not a transformers processor --
  there is no processor for this NeMo checkpoint.
* the output layout is ``(B, F, T)`` to match the encoder IO
  (sherpa encoder takes ``audio_signal`` of shape ``(B, 128, T)``), whereas the
  TDT extractor emits ``(B, T, F)``.

Verified byte-exact against the sherpa-onnx unified encoder on random mel
(max-abs ~1e-6, fp32 noise).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from . import config as C


class GpuMelExtractor:
    """GPU mel feature extractor matching NeMo's preprocessor for this model.

    Construct from a loaded state_dict (it pulls ``preprocessor.featurizer.fb``
    and ``.window``); all per-call work happens on-device.

    Args:
        state_dict: the parakeet-unified state_dict (from ``loader.load_state_dict``).
        device: where to place the precomputed filterbank / window.
    """

    def __init__(self, state_dict: dict, *, device: str | torch.device | None = None) -> None:
        fb = state_dict["preprocessor.featurizer.fb"].to(torch.float32)   # (1, 128, 257)
        self.mel_filters = fb.squeeze(0).contiguous()                      # (128, 257)
        win = state_dict["preprocessor.featurizer.window"].to(torch.float32)
        self.window = win.contiguous()                                     # (400,)

        self.n_fft = C.N_FFT
        self.hop_length = C.HOP_LENGTH
        self.win_length = C.WIN_LENGTH
        self.feature_size = C.N_MELS
        self.preemphasis = C.PREEMPHASIS
        self.padding_value = C.PADDING_VALUE
        self.log_zero_guard = C.LOG_ZERO_GUARD_VALUE
        self.epsilon = C.EPSILON

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    def to(self, device: str | torch.device) -> "GpuMelExtractor":
        self.device = torch.device(device)
        self.mel_filters = self.mel_filters.to(self.device)
        self.window = self.window.to(self.device)
        return self

    @torch.inference_mode()
    def __call__(self, audio_list: Iterable[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full mel pipeline on GPU.

        Args:
            audio_list: iterable of 1D float32 arrays (varying lengths), mono,
                16 kHz.

        Returns:
            ``(features, lengths)`` where ``features`` is ``(B, F=128, T)``
            float32 on the extractor's device (per_feature CMVN-normalized) and
            ``lengths`` is ``(B,)`` long giving the valid mel-frame count per
            utterance (== ``samples // hop``).
        """
        audio_list = [np.asarray(a, dtype=np.float32) for a in audio_list]
        if len(audio_list) == 0:
            raise ValueError("audio_list must contain at least one array")
        lengths = [int(a.shape[0]) for a in audio_list]
        B = len(audio_list)
        L_max = max(lengths)
        if L_max == 0:
            raise ValueError("all audio arrays are empty")

        device = self.device
        waveform = torch.full((B, L_max), float(self.padding_value),
                              dtype=torch.float32, device=device)
        for i, a in enumerate(audio_list):
            if a.shape[0]:
                waveform[i, : a.shape[0]] = torch.from_numpy(np.ascontiguousarray(a))
        audio_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        return self._run(waveform, audio_lengths)

    def extract_from_tensor(
        self, waveform: torch.Tensor, audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run on an already-batched ``(B, L_max)`` waveform tensor."""
        if waveform.dim() != 2:
            raise ValueError(f"waveform must be (B, L_max); got {tuple(waveform.shape)}")
        if waveform.device != self.device:
            waveform = waveform.to(self.device)
        if audio_lengths.device != self.device:
            audio_lengths = audio_lengths.to(self.device)
        return self._run(waveform.to(torch.float32), audio_lengths.to(torch.long))

    def _run(self, waveform: torch.Tensor, audio_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L_max = waveform.shape
        device = waveform.device
        hop = self.hop_length

        # (1) pre-emphasis IIR + padding mask
        timemask = torch.arange(L_max, device=device)[None, :] < audio_lengths[:, None]
        y = torch.empty_like(waveform)
        y[:, 0] = waveform[:, 0]
        y[:, 1:] = waveform[:, 1:] - self.preemphasis * waveform[:, :-1]
        y = y.masked_fill(~timemask, 0.0)

        # (2) STFT
        stft = torch.stft(
            y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, return_complex=True, pad_mode="constant", center=True,
        )                                                   # (B, n_fft//2+1, T)

        # (3) magnitude squared
        magnitudes = stft.abs() ** 2                        # (B, K, T)

        # (4) mel filterbank -> (B, F, T)
        mel_spec = torch.matmul(self.mel_filters, magnitudes)

        # (5) log
        mel_spec = torch.log(mel_spec + self.log_zero_guard)

        # (6) per_feature CMVN (mean/var over time, per mel bin, ignore padding)
        features_lengths = audio_lengths // hop             # (B,)
        T = mel_spec.shape[-1]
        fmask = torch.arange(T, device=device)[None, :] < features_lengths[:, None]   # (B, T)
        m = fmask.unsqueeze(1).to(mel_spec.dtype)           # (B, 1, T)
        masked = mel_spec * m
        fl = features_lengths.unsqueeze(-1).to(torch.float32).clamp(min=1.0)
        mean = masked.sum(dim=-1, keepdim=True) / fl        # (B, F, 1)
        var = ((masked - mean) ** 2 * m).sum(dim=-1, keepdim=True) / (fl - 1).clamp(min=1.0)
        std = torch.sqrt(var.clamp(min=1e-20))
        mel_spec = (mel_spec - mean) / (std + self.epsilon)
        mel_spec = mel_spec * m                             # re-zero padding frames

        return mel_spec.contiguous(), features_lengths


__all__ = ["GpuMelExtractor"]
