"""Model + processor loading helpers for mistralai/Voxtral-Mini-4B-Realtime-2602.

``load_model_and_processor`` returns the eager-mode reference model +
processor, and ``get_components`` resolves the submodules the pipeline
wires together: the audio tower, the pre-encoder embedder, the multimodal
projector, the text decoder trunk, the tied lm_head, and the time
embedding used to precompute the per-layer AdaRMSNorm modulation.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import MODEL_ID


def load_model_and_processor(
    attn_impl: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load the Voxtral Realtime model and processor.

    Native transformers model (no trust_remote_code). Eager attention is the
    byte-exact parity reference the CUDA-graph decoder is checked against.

    Args:
        attn_impl: Attention implementation for both the audio encoder and
            the text decoder (``"eager"`` is the parity reference).
        dtype: Model dtype (bf16 is the checkpoint dtype).
        device: Target device.

    Returns:
        ``(model, processor)`` with the model in eval mode. The processor is
        a ``VoxtralRealtimeProcessor`` (requires the ``mistral-common``
        package; its tokenizer backend reads ``tekken.json``).
    """
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def get_components(model: Any) -> dict[str, Any]:
    """Return the starling-relevant submodules of the Voxtral Realtime model.

    ``VoxtralRealtimeForConditionalGeneration`` exposes:
      * ``model.audio_tower`` -- the ``VoxtralRealtimeEncoder`` (embedder +
        32 transformer layers + final norm).
      * ``model.audio_tower.embedder`` -- the ``VoxtralRealtimeEmbedder``
        (causal conv1d k3 s1 -> GELU -> causal conv1d k3 s2 -> GELU).
      * ``model.multi_modal_projector`` -- the downsample-4 projector
        (Linear 5120->3072, no bias -> GELU -> Linear 3072->3072, no bias).
      * ``model.language_model`` -- the ``VoxtralRealtimeTextModel`` decoder
        trunk (26 layers + final norm).
      * ``model.lm_head`` -- ``nn.Linear`` (tied to the token embeddings).
      * ``model.language_model.embed_tokens`` -- the decoder token embeddings.
      * ``model.time_embedding`` -- the sinusoidal ``TimeEmbedding`` mapping
        ``num_delay_tokens`` to ``t_cond`` for the per-layer AdaRMSNorm.

    Returns:
        Dict with keys ``"audio_tower"``, ``"embedder"``,
        ``"multi_modal_projector"``, ``"language_model"``, ``"lm_head"``,
        ``"embed_tokens"``, ``"time_embedding"``.
    """
    inner = getattr(model, "model", None)
    audio_tower = getattr(inner, "audio_tower", None)
    projector = getattr(inner, "multi_modal_projector", None)
    language_model = getattr(inner, "language_model", None)
    lm_head = getattr(model, "lm_head", None)
    time_embedding = getattr(inner, "time_embedding", None)
    embedder = getattr(audio_tower, "embedder", None)
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if any(
        v is None
        for v in (
            audio_tower,
            embedder,
            projector,
            language_model,
            lm_head,
            embed_tokens,
            time_embedding,
        )
    ):
        raise AttributeError(
            "Could not resolve audio_tower/embedder/multi_modal_projector/"
            f"language_model/lm_head/embed_tokens/time_embedding on {type(model).__name__}"
        )
    return {
        "audio_tower": audio_tower,
        "embedder": embedder,
        "multi_modal_projector": projector,
        "language_model": language_model,
        "lm_head": lm_head,
        "embed_tokens": embed_tokens,
        "time_embedding": time_embedding,
    }
