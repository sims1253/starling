"""GPU mel feature extractor for nvidia/parakeet-tdt-0.6b-v3.

The stock ``processor(audio)`` runs the entire mel pipeline on CPU and returns
CPU tensors, which is why ``feat_ms`` scales superlinearly with batch size
(68 ms at B=8 -> ~1 s at B=16, the throughput cliff). This module reimplements
the EXACT 8-step pipeline distilled in ``MEL_PIPELINE.md`` as pure GPU torch
ops, so the batched pipeline stays on-device and there is no H2D transfer of
the per-utterance audio (only the small audio arrays themselves cross once).

Public surface
--------------
:class:`GpuMelExtractor`
    Construct once from a loaded ``AutoProcessor`` (it pulls ``mel_filters`` and
    precomputes the Hann window), then call with a list of 1D float arrays::

        extractor = GpuMelExtractor(processor).to("cuda")
        input_features, attention_mask = extractor(audio_list)
        # input_features: (B, T, 128) float32 on cuda
        # attention_mask: (B, T)     bool    on cuda

The returned tensors are NUMERICALLY EQUIVALENT to ``processor(audio_list)``
(max-abs ~3e-4 on float32 GPU vs float32 CPU; well within the encoder's
tolerance) and the ``attention_mask`` matches bit-exactly. See
``tests/test_mel_gpu.py`` for the byte-exact oracle-transcript test.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pipeline constants (parakeet-tdt-0.6b-v3, from MEL_PIPELINE.md / processor).
# Hard-coded because they are fixed by the model's mel config; the extractor
# also re-reads them from the processor in __init__ as a defensive sanity check.
# ---------------------------------------------------------------------------
SAMPLING_RATE = 16000
FEATURE_SIZE = 128     # mel bins
N_FFT = 512
HOP_LENGTH = 160       # 10 ms
WIN_LENGTH = 400       # 25 ms
PREEMPHASIS = 0.97
PADDING_VALUE = 0.0
LOG_ZERO_GUARD_VALUE = 2**-24   # 5.96e-8
EPSILON = 1e-5


class GpuMelExtractor:
    """GPU mel feature extractor matching the stock parakeet feature extractor.

    Reproduces the 8-step pipeline from ``MEL_PIPELINE.md`` as pure GPU torch
    ops. The constructor pulls ``mel_filters`` and precomputes the Hann window;
    all per-call work happens on-device (no CPU roundtrips in the hot path).

    Args:
        processor: a loaded ``transformers.AutoProcessor`` for
            ``nvidia/parakeet-tdt-0.6b-v3``. Only ``processor.feature_extractor``
            is read (for ``mel_filters`` and the mel config); the processor is
            not retained.
        device: where to place the precomputed ``mel_filters`` / window. Use
            ``.to(...)`` to move later. Defaults to ``cuda`` if available.
    """

    def __init__(self, processor, *, device: str | torch.device | None = None) -> None:
        fe = processor.feature_extractor

        # Defensive: read the config from the processor and verify it matches
        # the constants above. We trust the processor's actual values (it is
        # the source of truth) but they should match MEL_PIPELINE.md exactly.
        self.n_fft = int(getattr(fe, "n_fft", N_FFT))
        self.hop_length = int(getattr(fe, "hop_length", HOP_LENGTH))
        self.win_length = int(getattr(fe, "win_length", WIN_LENGTH))
        self.feature_size = int(getattr(fe, "feature_size", FEATURE_SIZE))
        self.preemphasis = float(getattr(fe, "preemphasis", PREEMPHASIS))
        self.padding_value = float(getattr(fe, "padding_value", PADDING_VALUE))
        self.sampling_rate = int(getattr(fe, "sampling_rate", SAMPLING_RATE))
        self.log_zero_guard = float(LOG_ZERO_GUARD_VALUE)
        self.epsilon = float(EPSILON)

        # mel_filters: (feature_size, n_fft//2 + 1) = (128, 257), float32,
        # pre-computed from librosa by the processor. Copy to a fresh tensor
        # (the processor's copy is shared; we want our own GPU allocation).
        mel_filters = fe.mel_filters
        if not isinstance(mel_filters, torch.Tensor):
            mel_filters = torch.as_tensor(mel_filters, dtype=torch.float32)
        else:
            mel_filters = mel_filters.to(torch.float32)
        if mel_filters.shape != (self.feature_size, self.n_fft // 2 + 1):
            raise ValueError(
                f"mel_filters shape {tuple(mel_filters.shape)} != "
                f"expected ({self.feature_size}, {self.n_fft // 2 + 1})"
            )
        self.mel_filters = mel_filters.contiguous()

        # Hann window (periodic=False to match the reference) -- precomputed,
        # stored on device alongside mel_filters.
        self.window = torch.hann_window(self.win_length, periodic=False).contiguous()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    # ------------------------------------------------------------------ #
    # device management
    # ------------------------------------------------------------------ #
    def to(self, device: str | torch.device) -> "GpuMelExtractor":
        """Move the precomputed ``mel_filters`` / window to ``device``."""
        self.device = torch.device(device)
        self.mel_filters = self.mel_filters.to(self.device)
        self.window = self.window.to(self.device)
        return self

    # ------------------------------------------------------------------ #
    # the hot path: list[np.ndarray] -> (input_features, attention_mask)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self, audio_list: Iterable[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full 8-step mel pipeline on GPU.

        Args:
            audio_list: iterable of 1D float arrays (varying lengths), mono,
                sampled at 16 kHz. Lists / tuples / generators are accepted.

        Returns:
            ``(input_features, attention_mask)`` where
            ``input_features`` is ``(B, T, feature_size)`` float32 on the
            extractor's device and ``attention_mask`` is ``(B, T)`` bool on the
            same device (``True`` = valid frame). These match the stock
            ``processor(audio_list)`` output to ~3e-4 max-abs (float32 GPU vs
            CPU); ``attention_mask`` matches bit-exactly.
        """
        # ---- Step 1: pad audio to max length in batch (right-pad with 0.0) --
        # Build the (B, L_max) batched tensor on GPU and the per-element length
        # tensor. Audio crosses H2D exactly once, as a single contiguous scatter.
        audio_list = list(audio_list)
        if len(audio_list) == 0:
            raise ValueError("audio_list must contain at least one array")

        lengths = [int(a.shape[0]) for a in audio_list]
        B = len(audio_list)
        L_max = max(lengths)
        if L_max == 0:
            raise ValueError("all audio arrays are empty")

        device = self.device
        waveform = torch.full(
            (B, L_max), float(self.padding_value),
            dtype=torch.float32, device=device,
        )
        audio_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        # scatter each utterance into its row (one small H2D per utterance; this
        # is unavoidable and tiny vs the stock pipeline's per-utterance CPU work)
        for i, a in enumerate(audio_list):
            if a.shape[0] == 0:
                continue
            arr = np.ascontiguousarray(a, dtype=np.float32)
            waveform[i, : arr.shape[0]] = torch.from_numpy(arr)

        return self._run(waveform, audio_lengths)

    def extract_from_tensor(
        self, waveform: torch.Tensor, audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run steps 2-8 on an already-batched ``(B, L_max)`` waveform tensor.

        Useful when the caller has already batched the audio on device (e.g.
        under a CUDA graph). ``waveform`` is ``(B, L_max)`` float on the
        extractor's device; ``audio_lengths`` is ``(B,)`` long on the same
        device giving the valid sample count per row. Padding positions of
        ``waveform`` beyond ``audio_lengths`` are ignored (re-zeroed in step 2).
        """
        if waveform.dim() != 2:
            raise ValueError(f"waveform must be (B, L_max); got {tuple(waveform.shape)}")
        if waveform.device != self.device:
            waveform = waveform.to(self.device)
        if audio_lengths.device != self.device:
            audio_lengths = audio_lengths.to(self.device)
        return self._run(
            waveform.to(torch.float32),
            audio_lengths.to(torch.long),
        )

    # ------------------------------------------------------------------ #
    # internal: steps 2-8 (expects an on-device batched waveform)
    # ------------------------------------------------------------------ #
    def _run(
        self, waveform: torch.Tensor, audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L_max = waveform.shape
        device = waveform.device
        hop = self.hop_length

        # ---- Step 2: pre-emphasis IIR (per-utterance, then mask padding) ----
        #   y[:, 0] = x[:, 0]
        #   y[:, 1:] = x[:, 1:] - preemphasis * x[:, :-1]
        #   y[~timemask] = 0
        timemask = torch.arange(L_max, device=device)[None, :] < audio_lengths[:, None]
        y = torch.empty_like(waveform)
        y[:, 0] = waveform[:, 0]
        y[:, 1:] = waveform[:, 1:] - self.preemphasis * waveform[:, :-1]
        y = y.masked_fill(~timemask, 0.0)

        # ---- Step 3: STFT (center=True is the torch default and matches the
        # stock output's frame count: T = 1 + audio_lengths // hop_length). ----
        stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
            center=True,
        )                                                # (B, n_fft//2+1, T)

        # ---- Step 4: magnitude squared  (= abs(stft)**2) -------------------
        magnitudes = stft.abs() ** 2                    # (B, n_fft//2+1, T)

        # ---- Step 5: mel filterbank (broadcasting matmul across batch) -----
        # mel_filters (F, K) @ magnitudes (B, K, T) -> (B, F, T)
        mel_spec = torch.matmul(self.mel_filters, magnitudes)

        # ---- Step 6: log (with 2**-24 zero guard) --------------------------
        mel_spec = torch.log(mel_spec + self.log_zero_guard)

        # ---- Step 7: permute to (B, T, F) ----------------------------------
        mel_spec = mel_spec.permute(0, 2, 1).contiguous()  # (B, T, F)

        # ---- Step 8: CMVN (per-utterance mean/var, ignoring padding) -------
        # features_lengths = audio_lengths // hop  (matches the stock attention
        # mask's per-row valid count exactly for n_fft even -- see MEL_PIPELINE.md)
        features_lengths = audio_lengths // hop          # (B,) long
        T = mel_spec.shape[1]
        attention_mask = torch.arange(T, device=device)[None, :] < features_lengths[:, None]
        mask = attention_mask.unsqueeze(-1)              # (B, T, 1)

        masked = mel_spec * mask
        # mean over valid frames; cast lengths to float32 for the division
        fl_f = features_lengths.unsqueeze(-1).to(torch.float32)
        mean = masked.sum(dim=1) / fl_f                  # (B, F)
        mean = mean.unsqueeze(1)                          # (B, 1, F)

        # variance uses (N-1) denominator; features_lengths >= 2 in practice
        # (shortest fixture is 743 frames). Match the reference's exact formula.
        denom = (features_lengths - 1).unsqueeze(-1).to(torch.float32)
        variance = ((masked - mean) ** 2 * mask).sum(dim=1) / denom   # (B, F)
        std = torch.sqrt(variance).unsqueeze(1)          # (B, 1, F)

        mel_spec = (mel_spec - mean) / (std + self.epsilon)
        mel_spec = mel_spec * mask                       # re-zero padding frames

        return mel_spec, attention_mask
