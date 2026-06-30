"""Audio loading and processor-input construction for Qwen3-ASR-1.7B.

These helpers go from a wav tensor on disk to a fully-formed processor output
dict (``input_ids``, ``input_features``, ``input_features_mask``) ready to feed
into ``Qwen3ASRForConditionalGeneration``.
"""

from __future__ import annotations

from typing import Any

import torch


def load_wav(path: str, *, sr_target: int = 16000) -> tuple[torch.Tensor, int]:
    """Load a wav file as a mono float32 tensor of shape ``(1, n_samples)``.

    Resamples to ``sr_target`` if needed (the processor expects 16 kHz mono).
    """
    import soundfile as sf

    samples, sr = sf.read(path)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    wav = torch.from_numpy(samples).float().unsqueeze(0)
    if sr != sr_target:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sr_target)
        sr = sr_target
    return wav, int(sr)


def build_inputs(
    processor: Any,
    wav: torch.Tensor,
    *,
    language: str | None = None,
    sr: int = 16000,
) -> dict[str, torch.Tensor]:
    """Build a CUDA-resident processor output dict for one wav clip.

    Uses ``processor.apply_transcription_request`` which handles the Qwen3-ASR
    chat template (``<|audio|>`` placeholder + optional language hint) and the
    Whisper-style log-mel feature extraction in one call.

    Args:
        processor: A ``Qwen3ASRProcessor``.
        wav: ``float32`` tensor of shape ``(1, n_samples)`` or ``(n_samples,)``
            at 16 kHz.
        language: Optional language hint (e.g. ``"English"``/``"zh"``). ``None``
            triggers auto-detection (the default ASR behaviour).
        sr: Sample rate of ``wav`` (must be 16000 for the processor).

    Returns:
        Dict with ``input_ids``, ``input_features`` and ``input_features_mask``
        moved to CUDA. Audio features are cast to bf16 to match the model dtype.
    """
    # apply_transcription_request accepts a raw 1-D numpy/tensor waveform.
    w = wav.squeeze(0).numpy() if wav.dim() == 2 else wav.numpy()
    inputs = processor.apply_transcription_request(audio=w, language=language, sampling_rate=sr)
    inputs = inputs.to("cuda")
    # Mel features -> bf16 (model dtype). Token ids stay int64.
    if "input_features" in inputs and inputs["input_features"].is_floating_point():
        inputs["input_features"] = inputs["input_features"].bfloat16()
    return inputs
