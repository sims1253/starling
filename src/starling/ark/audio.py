"""Audio loading, mel extraction, and prompt construction for ARK-ASR-3B.

These helpers cover the path from a wav array on disk to the inputs the
megakernel pipeline consumes: the mel spectrogram fed to the fused encoder, the
audio token count N implied by the mel length, the chat-templated prompt token
ids (with N ``<|audio|>`` placeholders), and the byte-exact audio-embedding
injection that scatters the adapter output into the decoder's input embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import (
    ADAPTER_MERGE_FACTOR,
    AUDIO_TOKEN,
    AUDIO_TOKEN_ID,
    ASSISTANT_TOKEN,
    BEGIN_AUDIO_TOKEN,
    END_AUDIO_TOKEN,
    USER_TOKEN,
)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a wav file as mono float32 at its native sample rate.

    Args:
        path: Path to a wav file.

    Returns:
        ``(samples, sample_rate)`` where ``samples`` is a 1-D ``float32`` numpy
        array (channels averaged to mono if needed).
    """
    import soundfile as sf

    samples, sr = sf.read(path)
    if samples.ndim == 2:
        # soundfile returns (n, channels) for multi-channel; collapse to mono.
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


def extract_mel(processor: Any, wav_list: list) -> torch.Tensor:
    """Extract log-mel features for a batch of waveforms.

    Wraps ``processor.feature_extractor`` with the exact arguments the eager
    reference uses (16 kHz, no attention mask, longest-padding) and returns the
    resulting ``input_features`` tensor of shape ``(B, 128, mel_T)`` as float32.

    The Whisper feature extractor caps ``mel_T`` at 3000 (its ``max_length``),
    so for audio longer than ~30 s ``mel_T`` is smaller than the raw
    ``len(wav) // hop_length`` mel-frame count. :func:`num_audio_tokens` takes
    the **uncapped** mel-frame count (from the raw audio length) so the prompt's
    audio-token count matches the eager reference's
    ``processor.calculate_audio_token_count`` exactly; the adapter then produces
    fewer features than there are audio slots and the injection zero-pads the
    remainder (see :func:`build_inputs_embeds`).

    Args:
        processor: An ARK-ASR processor (its ``feature_extractor`` is the
            Whisper feature extractor).
        wav_list: List of 1-D float32 waveforms at 16 kHz (numpy arrays or
            tensors).

    Returns:
        ``(B, 128, mel_T)`` float32 mel tensor.
    """
    out = processor.feature_extractor(
        wav_list,
        sampling_rate=16000,
        return_tensors="pt",
        padding="longest",
        return_attention_mask=False,
    )
    return out["input_features"]


def num_audio_tokens(
    n_mel_frames: int, merge_factor: int = ADAPTER_MERGE_FACTOR
) -> int:
    """Number of LLM audio tokens implied by a (uncapped) mel-frame count.

    Replicates ``processor.calculate_audio_token_count``: the Whisper encoder
    downsamples by 2 along time, giving ``(n_mel_frames + 1) // 2`` frames; the
    adapter then groups ``merge_factor`` frames per token.

    **Important:** ``n_mel_frames`` is the *uncapped* mel-frame count
    ``len(wav) // hop_length`` (the hop is 160), NOT the post-extraction
    ``mel_T`` which the Whisper feature extractor caps at 3000. The prompt's
    audio-token count must be derived from the raw audio length so it matches
    the eager reference; the adapter's (possibly fewer) features are then
    zero-padded into the longer prompt.

    Args:
        n_mel_frames: Uncapped mel-frame count (``len(wav) // hop_length``).
        merge_factor: Adapter merge factor (4 for ARK-ASR-3B).

    Returns:
        The audio token count N.
    """
    return ((n_mel_frames + 1) // 2) // merge_factor


def build_prompt_ids(
    tokenizer: Any,
    instruction: str,
    n_mel_frames: int,
    merge_factor: int = ADAPTER_MERGE_FACTOR,
) -> torch.Tensor:
    """Build the chat-templated prompt token ids for one utterance.

    Layout:
        ``<|user|><|begin_of_audio|>`` + ``<|audio|>`` * N +
        ``<|end_of_audio|>`` + instruction + ``<|assistant|>``
    tokenized with ``add_special_tokens=False``. The N ``<|audio|>`` positions
    (token id ``AUDIO_TOKEN_ID``) are later clobbered by the adapter's audio
    embeddings.

    Args:
        tokenizer: The ARK-ASR tokenizer.
        instruction: The user instruction text following the audio block.
        n_mel_frames: Uncapped mel-frame count (``len(wav) // hop_length``);
            determines N via :func:`num_audio_tokens`.
        merge_factor: Adapter merge factor (4 for ARK-ASR-3B).

    Returns:
        ``(1, T)`` int64 tensor of prompt token ids.
    """
    n = num_audio_tokens(n_mel_frames, merge_factor=merge_factor)
    prompt = (
        USER_TOKEN
        + BEGIN_AUDIO_TOKEN
        + AUDIO_TOKEN * n
        + END_AUDIO_TOKEN
        + instruction
        + ASSISTANT_TOKEN
    )
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    return enc["input_ids"]


def build_inputs_embeds(
    model: Any,
    input_ids: torch.Tensor,
    audio_features: torch.Tensor,
) -> torch.Tensor:
    """Merge audio embeddings into the decoder token embeddings.

    Byte-exact replica of ``ArkasrForConditionalGeneration._inject_audio_embeddings``:

    1. build a mask of the audio-token positions (``input_ids == AUDIO_TOKEN_ID``);
    2. zero those positions in the id tensor and look up ``embed_tokens``;
    3. scatter the adapter's audio features into the audio-token slots.

    For a single batch row with exactly N audio positions and N audio feature
    rows, the scatter is a direct positional copy (no truncation/padding).

    Args:
        model: The loaded ARK-ASR model (provides ``model.model.embed_tokens``).
        input_ids: ``(1, T)`` prompt token ids containing N audio placeholders.
        audio_features: ``(1, N, 2048)`` bf16 adapter output.

    Returns:
        ``(1, T, 2048)`` bf16 multimodal input embeddings.
    """
    embed_tokens = model.model.embed_tokens
    audio_mask = input_ids == AUDIO_TOKEN_ID
    llm_ids = torch.where(audio_mask, 0, input_ids)
    inputs_embeds = embed_tokens(llm_ids)
    audio_positions = audio_mask[0].nonzero().squeeze(-1)
    n_slots = int(audio_positions.numel())
    feat = audio_features[0]
    sa = int(feat.shape[0])
    # Match ``_inject_audio_embeddings`` alignment: when the adapter emits fewer
    # features than there are audio slots (long audio, where the Whisper feature
    # extractor caps mel at 3000 but the prompt's audio-token count derives from
    # the uncapped raw length), zero-pad the features to the slot count; when it
    # emits more, truncate. The common case (short/medium audio) has sa == n_slots
    # and is a direct positional copy.
    if sa < n_slots:
        pad = feat.new_zeros((n_slots - sa, feat.shape[-1]))
        feat = torch.cat([feat, pad], dim=0)
    elif sa > n_slots:
        feat = feat[:n_slots]
    inputs_embeds[0, audio_positions, :] = feat
    return inputs_embeds
