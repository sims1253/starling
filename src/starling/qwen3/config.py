"""Qwen3-ASR-1.7B architecture constants (single source of truth).

Mirrors the layout of :mod:`starling.config` (granite) so the qwen3 megakernel
shares the same conventions. All values come from
``Qwen/Qwen3-ASR-1.7B-hf/config.json`` (audio_config + text_config).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "Qwen/Qwen3-ASR-1.7B-hf"
"""HF hub repo id for the transformers-native (``-hf``) Qwen3-ASR variant.

We use ``-hf`` (not the original ``Qwen/Qwen3-ASR-1.7B``) because it ships a
single ``model.safetensors`` + native ``Qwen3ASRProcessor``/chat template that
loads straight into ``transformers.Qwen3ASRForConditionalGeneration``.
"""

# ---------------------------------------------------------------------------
# Audio encoder dims (Qwen3ASREncoderConfig / audio_config)
# ---------------------------------------------------------------------------
AUDIO_D_MODEL: int = 1024            # encoder hidden
AUDIO_NUM_LAYERS: int = 24           # encoder layers
AUDIO_NUM_HEADS: int = 16            # encoder_attention_heads
AUDIO_HEAD_DIM: int = 64             # d_model // num_heads (1024 // 16)
AUDIO_FFN_DIM: int = 4096            # encoder_ffn_dim
AUDIO_NUM_KV_HEADS: int = 16         # MHA in the audio tower (num_key_value_heads)
AUDIO_NUM_MEL_BINS: int = 128        # mel bins
AUDIO_DOWNSAMPLE_HIDDEN: int = 480   # conv2d channel dim
AUDIO_OUTPUT_DIM: int = 2048         # projector output dim (== LLM hidden)
AUDIO_N_WINDOW: int = 50             # chunk size is n_window*2 = 100 frames
AUDIO_N_WINDOW_INFER: int = 800      # inference attention window (raw frames)
AUDIO_CONV_CHUNKSIZE: int = 500      # conv_chunksize (unused at infer, kept for ref)
AUDIO_MAX_POS_EMB: int = 13          # max_position_embeddings (post-CNN frames)

# ---------------------------------------------------------------------------
# Text decoder dims (Qwen3 text_config)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2048
LLM_NUM_LAYERS: int = 28
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 8            # GQA (16 Q / 8 KV)
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 6144
LLM_VOCAB_SIZE: int = 151936
LLM_MAX_POS_EMB: int = 65536
LLM_ROPE_THETA: float = 1_000_000.0
LLM_RMS_NORM_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Tokenisation / multimodal
# ---------------------------------------------------------------------------
AUDIO_TOKEN_ID: int = 151676
"""The ``<|audio|>`` placeholder token id; positions carrying it are clobbered
by the projected audio embeddings inside ``Qwen3ASRModel.forward``."""
EOS_TOKEN_ID: int = 151645           # im_end (primary EOS for greedy stop)
PAD_TOKEN_ID: int = 151645
TIMESTAMP_TOKEN_ID: int = 151705

DEFAULT_TASK_PROMPT: str = ""
"""Qwen3-ASR takes no instruction prompt: the chat template just wraps the
``<|audio|>`` placeholder. ``build_inputs`` leaves the user content empty."""

# ---------------------------------------------------------------------------
# Correctness tolerances for kernel comparisons (bf16 eager reference)
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
LLM_LOGIT_ATOL: float = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""Repository root (the dir that contains src/, golden/, ...)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden" / "qwen3"
"""Directory where Qwen3-ASR reference tensors are persisted (gitignored)."""
