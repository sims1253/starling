"""Audio loading and processor-input construction for Granite-Speech-4.1-2b.

These helpers wrap the slightly fiddly bits of going from a wav file on disk to
a fully-formed processor output dict that can be fed straight into the model.
"""

from __future__ import annotations

from typing import Any

import torch

from ..config import DEFAULT_TASK_PROMPT, MODEL_ID, SAMPLE_AUDIO_FILENAME


def load_sample_audio() -> tuple[torch.Tensor, int]:
    """Download (cached) and load the Granite-Speech sample wav.

    Returns:
        ``(wav_tensor, sample_rate)`` where ``wav_tensor`` has shape
        ``(1, n_samples)`` and dtype ``float32``. Ready to feed to
        :func:`build_inputs`.
    """
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODEL_ID, filename=SAMPLE_AUDIO_FILENAME)
    samples, sr = sf.read(path)
    if samples.ndim == 2:
        # soundfile returns (n, channels) for multi-channel; collapse to mono.
        samples = samples.mean(axis=1)
    wav = torch.from_numpy(samples).float().unsqueeze(0)
    return wav, int(sr)


def build_inputs(
    processor: Any,
    wav: torch.Tensor,
    task_prompt: str = DEFAULT_TASK_PROMPT,
) -> dict[str, torch.Tensor]:
    """Build a CUDA-resident processor output dict for one wav clip.

    The processor leaves tokenizer outputs on CPU and only moves the audio
    features to the requested device, so this helper finishes the job by
    moving every tensor to CUDA for a clean downstream feed.

    Args:
        processor: A GraniteSpeech processor (audio + tokenizer).
        wav: ``float32`` tensor of shape ``(1, n_samples)`` at 16 kHz.
        task_prompt: The instruction following the ``<|audio|>`` placeholder.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``input_features`` and
        ``input_features_mask`` (all on cuda).
    """
    chat = [{"role": "user", "content": f"<|audio|>{task_prompt}"}]
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, wav, device="cuda", return_tensors="pt")
    # The processor only moves the audio tensor; move the rest ourselves so the
    # returned dict is uniformly CUDA-resident.
    out = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            out[k] = v.to("cuda")
        else:
            out[k] = v
    return out


def build_prompt(processor: Any, task_prompt: str = DEFAULT_TASK_PROMPT) -> str:
    """Return the raw chat-templated prompt string (useful for debugging)."""
    chat = [{"role": "user", "content": f"<|audio|>{task_prompt}"}]
    return processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
