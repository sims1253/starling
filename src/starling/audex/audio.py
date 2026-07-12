"""Audio loading, mel extraction, and prompt construction for Audex-2B ASR.

These helpers replicate the reference ``inference_scripts_hf/`` pipeline:
waveform → 30 s clips → Whisper mel features → ChatML prompt with expanded
``<so_embedding>`` placeholders → tokenised ``input_ids``.

The prompt for non-thinking ASR (greedy) follows the model card recipe:

    <|im_start|>user
    <so_start><so_embedding>×N<so_end>
    Transcribe the speech in the input audio.<|im_end|>
    <|im_start|>assistant
    <think></think>

The trailing ``<think></think>`` activates instruct (non-thinking) mode so the
model emits the transcript directly instead of burning tokens on reasoning.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .config import (
    DEFAULT_TASK_PROMPT,
    SOUND_CLIP_DURATION,
    SOUND_EMBEDDING_SIZE,
    SOUND_END_TOKEN_STR,
    SOUND_PLACEHOLDER_STR,
    SOUND_START_TOKEN_STR,
    SOUND_TARGET_RATE,
    SOUND_TOKEN_STR,
)


def load_wav(path: str, *, sr_target: int = SOUND_TARGET_RATE) -> tuple[np.ndarray, int]:
    """Load a wav file as a mono float32 numpy array at ``sr_target`` Hz."""
    import soundfile as sf

    samples, sr = sf.read(path)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    wav = np.ascontiguousarray(samples, dtype=np.float32)
    if sr != sr_target:
        import torchaudio

        t = torch.from_numpy(wav).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, sr_target)
        wav = t.squeeze(0).numpy()
        sr = sr_target
    return wav, int(sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Mono float32 in [-1, 1], matching the Megatron eval path."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[1] <= 2:
            audio = audio.mean(axis=1)
        else:
            audio = audio.mean(axis=0)
    max_abs = float(np.abs(audio).max()) if audio.size else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio.astype(np.float32, copy=False)


def split_audio_into_clips(
    audio: np.ndarray,
    sample_rate: int = SOUND_TARGET_RATE,
    clip_duration: float = SOUND_CLIP_DURATION,
) -> list[np.ndarray]:
    """Split audio into fixed 30 s clips; pad the final clip with zeros."""
    audio = normalize_audio(audio)
    clip_samples = int(round(sample_rate * clip_duration))
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    num_clips = max(1, math.ceil(audio.shape[0] / clip_samples))
    clips: list[np.ndarray] = []
    for idx in range(num_clips):
        start = idx * clip_samples
        clip = audio[start : start + clip_samples]
        if clip.shape[0] < clip_samples:
            clip = np.pad(clip, (0, clip_samples - clip.shape[0]))
        clips.append(clip.astype(np.float32, copy=False))
    return clips


def extract_mel(
    feature_extractor: Any,
    audio: np.ndarray,
    *,
    sample_rate: int = SOUND_TARGET_RATE,
    clip_duration: float = SOUND_CLIP_DURATION,
) -> torch.Tensor:
    """Whisper mel features ``(num_clips, 128, 3000)`` for a 1-D waveform.

    Splits into 30 s clips (padding the last), runs the Whisper feature
    extractor with ``padding="max_length"`` so every clip is exactly 3000 frames.
    """
    clips = split_audio_into_clips(audio, sample_rate, clip_duration)
    features = feature_extractor(
        clips,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=False,
    )
    return features.input_features


def build_prompt_text(
    task_prompt: str,
    num_embeddings: int,
) -> str:
    """Build the ChatML prompt text for non-thinking ASR.

    Layout (matching ``inference_scripts_hf/audio_utils.py``):

        <|im_start|>user
        <sound>
        {task_prompt}<|im_end|>
        <|im_start|>assistant
        <think></think>

    The ``<sound>`` placeholder is then expanded into
    ``<so_start> + <so_embedding>×N + <so_end>``.
    """
    prompt = (
        "<|im_start|>user\n"
        f"{SOUND_PLACEHOLDER_STR}\n"
        f"{task_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think></think>"
    )
    replacement = (
        SOUND_START_TOKEN_STR
        + (SOUND_TOKEN_STR * num_embeddings)
        + SOUND_END_TOKEN_STR
    )
    return prompt.replace(SOUND_PLACEHOLDER_STR, replacement)


def build_inputs(
    tokenizer: Any,
    feature_extractor: Any,
    wav: np.ndarray,
    *,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    sample_rate: int = SOUND_TARGET_RATE,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Build CUDA-resident inputs for one utterance.

    Args:
        tokenizer: The Audex ``AutoTokenizer``.
        feature_extractor: The Whisper ``AutoFeatureExtractor``.
        wav: 1-D float32 waveform at 16 kHz (numpy or torch).
        task_prompt: ASR instruction text.
        sample_rate: Sample rate of ``wav``.
        dtype: Mel feature dtype (bf16 to match the model).

    Returns:
        Dict with:
        * ``input_features`` -- ``(num_clips, 128, 3000)`` bf16 mel on cuda.
        * ``input_ids`` -- ``(1, T)`` int64 prompt token ids on cuda.
    """
    if isinstance(wav, torch.Tensor):
        wav = wav.squeeze().numpy()
    wav = normalize_audio(wav)

    # (1) Whisper mel features.
    input_features = extract_mel(feature_extractor, wav, sample_rate=sample_rate)

    # (2) Number of audio embeddings = clips × 750.
    num_clips = int(input_features.shape[0])
    num_embeddings = num_clips * SOUND_EMBEDDING_SIZE

    # (3) Build + expand + tokenize the ChatML prompt.
    prompt_text = build_prompt_text(task_prompt, num_embeddings)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]

    return {
        "input_features": input_features.to(device="cuda", dtype=dtype),
        "input_ids": input_ids.to(device="cuda"),
    }
