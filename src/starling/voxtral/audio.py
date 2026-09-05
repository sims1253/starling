"""Audio loading, mel extraction, and frame arithmetic for Voxtral Realtime.

Covers the path from a wav on disk to the inputs the pipeline consumes:
the log-mel spectrogram fed to the embedder, the conv output lengths, and
the audio-token count implied by the mel length.

Conv length convention: both embedder convs left-pad their input by
``left_pad`` (offline: plain zeros; stock ``VoxtralRealtimeCausalConv1d``
pads ``(left_pad, 0)`` then runs a valid cross-correlation), so a conv
with kernel k and stride s maps ``L_in -> floor((L_in + left_pad - k) / s)
+ 1`` frames. With k=3, conv1 (s=1, pad 2) preserves length
(``L -> L``) and conv2 (s=2, pad 1) maps ``L -> floor(L / 2)``.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch

from .config import (
    AUDIO_LENGTH_PER_TOK,
    DOWNSAMPLE_FACTOR,
    EMBEDDER_CONV1_LEFT_PAD,
    EMBEDDER_CONV1_STRIDE,
    EMBEDDER_CONV2_LEFT_PAD,
    EMBEDDER_CONV2_STRIDE,
    EMBEDDER_KERNEL,
    HOP_LENGTH,
    RAW_SAMPLES_PER_AUDIO_TOK,
    SAMPLE_RATE,
    STREAMING_LEFT_PAD_TOKENS,
    STREAMING_RIGHT_PAD_TOKENS,
)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a wav file as mono float32 at its native sample rate.

    Args:
        path: Path to a wav file.

    Returns:
        ``(samples, sample_rate)`` where ``samples`` is a 1-D ``float32``
        numpy array (channels averaged to mono if needed).
    """
    import soundfile as sf

    samples, sr = sf.read(path)
    if samples.ndim == 2:
        # soundfile returns (n, channels) for multi-channel; collapse to mono.
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


def _as_mono_float32(wav: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Collapse stereo to mono and return a contiguous float32 numpy array."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def offline_padded_samples(n_samples: int) -> int:
    """Waveform length after the offline streaming-pad.

    mistral-common pads offline audio to a whole number of audio tokens
    (``raw_audio_length_per_tok`` = 1280 samples = 8 mel frames), then adds
    the left silence pad (32 tokens) and the right pad (delay 6 + BOS 1 +
    buffer 10 = 17 tokens)::

        padded = ceil(n / 1280) * 1280 + (32 + 17) * 1280

    Args:
        n_samples: Raw waveform sample count.

    Returns:
        Padded sample count whose mel length is a multiple of 8.
    """
    unit = RAW_SAMPLES_PER_AUDIO_TOK
    body = ((int(n_samples) + unit - 1) // unit) * unit
    pad = (STREAMING_LEFT_PAD_TOKENS + STREAMING_RIGHT_PAD_TOKENS) * unit
    return body + pad


def mel_frames(n_samples: int) -> int:
    """Mel-frame count for the offline (padded) waveform.

    The offline mel length follows the STFT convention used by the stock
    feature extractor call (``center=True``): the padded waveform of
    ``offline_padded_samples(n)`` samples with hop 160 gives
    ``mel = 1 + padded // 160`` frames. The padded length is always a
    multiple of 1280 = 8 hops, so the offline mel length is always
    ``1 mod 8``.

    Args:
        n_samples: Raw waveform sample count.

    Returns:
        Offline mel-frame count (includes the streaming pads).
    """
    return 1 + offline_padded_samples(int(n_samples)) // HOP_LENGTH


def conv_out_len(
    l_in: int, *, left_pad: int, kernel: int = EMBEDDER_KERNEL, stride: int = 1
) -> int:
    """Output length of one embedder causal conv.

    The conv left-pads by ``left_pad`` then runs a valid correlation:
    ``floor((L_in + left_pad - kernel) / stride) + 1``, floored at 0 for
    degenerate (empty) inputs.

    Args:
        l_in: Input frame count.
        left_pad: Causal left pad (2 for conv1, 1 for conv2).
        kernel: Kernel size (3 for both embedder convs).
        stride: Stride (1 for conv1, 2 for conv2).

    Returns:
        Output frame count.
    """
    return max(0, (int(l_in) + int(left_pad) - int(kernel)) // int(stride) + 1)


def conv1_out_len(mel_T: int) -> int:
    """Embedder conv1 (k3 s1, left-pad 2) output length: preserves length."""
    return conv_out_len(
        mel_T,
        left_pad=EMBEDDER_CONV1_LEFT_PAD,
        stride=EMBEDDER_CONV1_STRIDE,
    )


def conv2_out_len(l_in: int) -> int:
    """Embedder conv2 (k3 s2, left-pad 1) output length: ``floor(L / 2)``."""
    return conv_out_len(
        l_in, left_pad=EMBEDDER_CONV2_LEFT_PAD, stride=EMBEDDER_CONV2_STRIDE
    )


def num_audio_tokens_from_conv2(conv2_len: int) -> int:
    """Audio tokens from the conv2 output length (projector groups by 4).

    The projector reshapes ``(B, T', 1280) -> (B, T' // 4, 5120)``; a
    trailing partial group of ``T' % 4`` encoder frames is dropped by the
    reshape, so only ``conv2_len // 4`` tokens survive. Offline mel lengths
    are ``1 mod 8`` (see :func:`mel_frames`), which gives conv2 lengths of
    ``0 mod 4`` -- no partial group, no dropped frames.

    Args:
        conv2_len: Conv2 (embedder output) frame count.

    Returns:
        Projected audio-token count.
    """
    return int(conv2_len) // DOWNSAMPLE_FACTOR


def num_audio_tokens_from_mel(mel_T: int) -> int:
    """Audio-token count from a mel length via the conv chain (exact).

    Equivalent to ``conv2_out_len(conv1_out_len(mel_T)) // 4``; for offline
    mel lengths (``1 mod 8``) this equals ``mel_T // 8`` with no remainder.

    Args:
        mel_T: Mel-frame count (post feature extractor).

    Returns:
        Audio-token count.
    """
    return num_audio_tokens_from_conv2(conv2_out_len(conv1_out_len(mel_T)))


def stock_max_length(mel_T: int) -> int:
    """Stock ``generate`` total-length bound: ``ceil(mel_T / 8)``.

    From ``_prepare_generation_config``: ``num_audio_tokens = ceil(mel_len /
    audio_length_per_tok)``, used as the default ``max_length`` (total
    prompt + generated length) and as a hard clamp. Prompt token ids come
    from the processor, so the loop must stop at the earlier of EOS and
    this bound.

    Args:
        mel_T: Mel-frame count (post feature extractor).

    Returns:
        The stock total-sequence-length cap.
    """
    return (int(mel_T) + AUDIO_LENGTH_PER_TOK - 1) // AUDIO_LENGTH_PER_TOK


def extract_mel(processor: Any, wav_list: list) -> torch.Tensor:
    """Extract log-mel features for a batch of waveforms.

    Wraps the stock ``VoxtralRealtimeFeatureExtractor`` call the processor
    itself makes offline: 16 kHz, no truncation, longest-padding, float32
    output of shape ``(B, 128, mel_T)``.

    Args:
        processor: A ``VoxtralRealtimeProcessor``.
        wav_list: List of 1-D float32 waveforms at 16 kHz (numpy arrays or
            tensors).

    Returns:
        ``(B, 128, mel_T)`` float32 mel tensor.
    """
    out = processor.feature_extractor(
        wav_list,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        return_attention_mask=False,
        center=True,
    )
    mel = out["input_features"]
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    return mel.to(torch.float32)


def prepare_processor_inputs(
    processor: Any, wav: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run the stock processor on one waveform.

    The processor (mistral-common ``encode_transcription`` + feature
    extractor) owns the prompt token ids -- including the streaming silence
    pads and the delay-token prefix -- and the matching mel. Treat both as
    opaque processor output; the frame-arithmetic helpers above only
    *account* for the mel length the processor produces.

    Args:
        processor: A ``VoxtralRealtimeProcessor``.
        wav: 1-D float32 waveform at 16 kHz.

    Returns:
        ``(input_ids, input_features, num_delay_tokens)``: ``(1, P)`` int64
        prompt ids, ``(1, 128, mel_T)`` float32 mel, and the fixed
        per-request delay-token count.
    """
    wav = _as_mono_float32(wav)
    out = processor(wav, return_tensors="pt")
    input_ids = out["input_ids"]
    mel = out["input_features"]
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    num_delay = out.get("num_delay_tokens", 6)
    if isinstance(num_delay, torch.Tensor):
        num_delay = int(num_delay.reshape(-1)[0].item())
    else:
        num_delay = int(num_delay)
    return (
        input_ids.to(torch.int64),
        mel.to(torch.float32),
        num_delay,
    )


def check_mel_accounting(mel_T: int) -> dict[str, int]:
    """Cross-check the conv-chain token count against the stock bound.

    For offline mel lengths (``1 mod 8``) the two agree up to the known
    off-by-one: the stock bound ``ceil(mel/8) = mel//8 + 1`` counts one
    extra length unit over the exact ``mel//8`` conv-chain tokens. The
    returned ``"delta"`` is that difference (1 offline; larger only for
    synthetic non-``1-mod-8`` lengths where padding/partial groups differ).

    Args:
        mel_T: Mel-frame count.

    Returns:
        Dict with ``mel_T``, ``conv1``, ``conv2``, ``tokens`` (exact),
        ``stock_bound`` (``ceil(mel/8)``), and ``delta``.
    """
    c1 = conv1_out_len(mel_T)
    c2 = conv2_out_len(c1)
    toks = num_audio_tokens_from_conv2(c2)
    bound = stock_max_length(mel_T)
    return {
        "mel_T": int(mel_T),
        "conv1": c1,
        "conv2": c2,
        "tokens": toks,
        "stock_bound": bound,
        "delta": bound - toks,
    }
