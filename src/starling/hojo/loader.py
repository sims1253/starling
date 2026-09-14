"""Model + tokenizer loading helpers for HojoAI/Hojo-ASR-V1.

The model is loaded via the reference ``hojo-asr`` package's
``HOJO_ASR.load_model`` (which resolves the bundle, builds the three components
-- ``speech_encoder`` (Qwen3-Omni tower), ``bottleneck`` (WeNet Conformer),
``ln_speech``, and ``decoder_model`` (Qwen3-4B) -- from ``config.yaml`` and
applies ``merged_full_model.safetensors`` with strict ``assign=True``).

Unlike higgs (which patches graph-unsafe remote helpers), Hojo's modeling is
plain PyTorch + transformers, so no patching is required. The encoder is run
eager (see :mod:`starling.hojo.encoder_mega`), and the decoder is a stock
``Qwen3ForCausalLM`` driven via ``.generate(num_beams=4, ...)``.
"""

from __future__ import annotations

from typing import Any

from .config import DEFAULT_MODEL_DIR


def load_model(
    folder_path: str | None = None,
    device: str = "cuda:0",
) -> Any:
    """Load the HOJO_ASR model (all components) into eval mode.

    Args:
        folder_path: Local bundle dir containing ``config.yaml`` +
            ``merged_full_model.safetensors`` (+ the three submodel dirs).
            Defaults to :data:`DEFAULT_MODEL_DIR` (``.hf-cache/hojo-asr-v1``).
        device: Target device.

    Returns:
        The ``HOJO_ASR`` module in eval mode on ``device``.
    """
    from hojo_asr import HOJO_ASR

    folder = str(folder_path) if folder_path is not None else str(DEFAULT_MODEL_DIR)
    model = HOJO_ASR.load_model(folder, device=device)
    model.eval()
    return model


def get_components(model: Any) -> dict[str, Any]:
    """Resolve the starling-relevant submodules of a loaded ``HOJO_ASR`` model.

    Returns a dict with:
      * ``speech_encoder``  -- the ``ModifyQwen3OmniMoeAudioEncoder`` (tower).
      * ``bottleneck``      -- the WeNet ``ConformerEncoder`` (LinearNoSubsampling
        + 2 ConformerEncoderLayer + after_norm).
      * ``ln_speech``       -- the ``nn.LayerNorm(2560)`` after the bottleneck.
      * ``decoder_model``   -- the stock ``Qwen3ForCausalLM`` (36 layers, GQA,
        separate lm_head).
      * ``tokenizer``       -- the Qwen3 tokenizer ([PAD] added, right-padded).
      * ``feat_extractor``  -- the Whisper-large-v3 feature extractor.
      * ``bos_token_id``    -- 151644 (<|im_start|>) for Qwen.
      * ``model``           -- the full ``HOJO_ASR`` (for encoder prefill +
        autocast_context()).
    """
    decoder = getattr(model, "decoder_model", None)
    if decoder is None:
        raise AttributeError(f"could not find decoder_model on {type(model).__name__}")
    speech_encoder = getattr(model, "speech_encoder", None)
    bottleneck = getattr(model, "bottleneck", None)
    ln_speech = getattr(model, "ln_speech", None)
    tokenizer = getattr(model, "tokenizer", None)
    feat_extractor = getattr(model, "feat_extractor", None)
    if speech_encoder is None or bottleneck is None or ln_speech is None:
        raise AttributeError(
            f"could not resolve speech_encoder/bottleneck/ln_speech on "
            f"{type(model).__name__}"
        )
    return {
        "speech_encoder": speech_encoder,
        "bottleneck": bottleneck,
        "ln_speech": ln_speech,
        "decoder_model": decoder,
        "tokenizer": tokenizer,
        "feat_extractor": feat_extractor,
        "bos_token_id": int(getattr(model, "bos_token_id", 151644)),
        "model": model,
    }
