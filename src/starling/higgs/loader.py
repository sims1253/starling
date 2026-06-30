"""Model + processor loading helpers for bosonai/higgs-audio-v3-stt.

Unlike granite/parakeet (which run on the repo's shared ``transformers 5.13``
venv), Higgs-Audio-v3 targets ``transformers 4.51``: its ``trust_remote_code``
modeling (Whisper audio-tower mask plumbing, ``Qwen3DecoderLayer`` return shape,
``GenerationConfig.generation_kwargs``) breaks across the 4.51->5.x boundary, and
``boson-multimodal`` further pins ``<4.47``. So Higgs runs in its OWN isolated
venv ``.venv-higgs`` (``transformers==4.51.3`` + our vendored collator +
``torch 2.12.1+cu130``). The shared ``.venv`` is untouched -- the megakernel
imports + the golden oracle both run under ``.venv-higgs``.

The model loads via ``trust_remote_code=True`` (its remote modeling is
transformers-4.51-clean). Audio->mel preprocessing uses our vendored
``HiggsAudioSampleCollator`` (transformers-version-independent).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from .config import MODEL_ID


def load_model_and_tokenizer(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_impl: str = "eager",
) -> tuple[Any, Any]:
    """Load the Higgs-Audio-v3 model + tokenizer.

    Args:
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.
        attn_impl: attention implementation. ``"eager"`` is the byte-exact
            golden path (matches how the golden oracle was captured).

    Returns:
        ``(model, tokenizer)`` with the model in eval mode.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, config=cfg, trust_remote_code=True,
        torch_dtype=dtype, device_map=device,
        attn_implementation=attn_impl,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def get_components(model: Any) -> dict[str, Any]:
    """Resolve the starling-relevant submodules of ``HiggsAudio3Model``.

    The Qwen3 decoder pieces live directly on the top-level model:
    ``embed_tokens``, ``layers`` (28 x Qwen3DecoderLayer), ``norm``,
    ``rotary_emb``. The text lm_head is on ``audio_decoder_proj.text_lm_head``
    (NOT tied to embed_tokens -- a separate nn.Linear).
    """
    adp = getattr(model, "audio_decoder_proj", None)
    if adp is None or not hasattr(adp, "text_lm_head"):
        raise AttributeError(
            f"could not find text_lm_head on {type(model).__name__}." "audio_decoder_proj"
        )
    return {
        "embed_tokens": model.embed_tokens,
        "layers": list(model.layers),
        "norm": model.norm,
        "rotary_emb": model.rotary_emb,
        "text_lm_head": adp.text_lm_head,
        "model": model,  # full model (for encoder/projector prefill + _forward_core)
    }


def make_collator(model_or_config: Any):
    """Build the HiggsAudioSampleCollator from a model or config.

    Uses ``openai/whisper-large-v3``'s WhisperProcessor for the mel frontend,
    matching the upstream ``transcribe()`` collator exactly.
    """
    from transformers import WhisperProcessor
    from .vendor import HiggsAudioSampleCollator

    config = model_or_config.config if hasattr(model_or_config, "config") else model_or_config
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    return HiggsAudioSampleCollator(
        whisper_processor=whisper_proc,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=1,
        audio_num_codebooks=config.audio_num_codebooks,
        chunk_size_seconds=getattr(config, "chunk_size_seconds", 30),
        pad_left=False,
    )


@contextmanager
def inference_mode() -> Iterator[None]:
    """Thin wrapper around ``torch.inference_mode``."""
    with torch.inference_mode():
        yield
