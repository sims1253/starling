"""Nemotron-Labs-Audex-2B architecture constants (single source of truth).

All values are verified against the downloaded ``checkpoint_folder_full/config.json``
and the remote modeling code (``modeling_nemotron_h_audio.py``,
``modeling_nemotron_dense.py``). ASR path only.

Model structure:
  - Audio encoder: ``Qwen2AudioEncoder`` (Whisper-large-v3 shaped, 32 layers,
    d_model 1280, avg-pooler halves 1500→750 output frames per 30 s clip).
  - Projector: RMSNorm → fc1 → relu2 → fc2 (1280→4096→2048).
  - LLM decoder: Nemotron-Dense 2B (28 layers, GQA 16Q/8KV, relu2 MLP,
    untied embeddings, RoPE theta 1e8).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "nvidia/Nemotron-Labs-Audex-2B"
"""HF hub repo id."""

MODEL_SUBFOLDER: str = "checkpoint_folder_full"
"""Sub-folder containing the full (audio + text) checkpoint."""

# ---------------------------------------------------------------------------
# Audio encoder dims (Qwen2AudioEncoder / audio_config)
# ---------------------------------------------------------------------------
AUDIO_D_MODEL: int = 1280            # encoder hidden
AUDIO_NUM_LAYERS: int = 32           # encoder_layers
AUDIO_NUM_HEADS: int = 20            # encoder_attention_heads
AUDIO_FFN_DIM: int = 5120            # encoder_ffn_dim
AUDIO_NUM_MEL_BINS: int = 128        # num_mel_bins
AUDIO_MAX_SOURCE_POSITIONS: int = 1500  # whisper positional embedding length
AUDIO_INPUT_FRAMES: int = 3000       # mel frames per 30 s clip (nb_max_frames)
AUDIO_OUTPUT_FRAMES: int = 750       # post avg-pooler frames per clip

# ---------------------------------------------------------------------------
# Projector dims (NemotronDenseAudexProjector)
# ---------------------------------------------------------------------------
PROJECTOR_INTERMEDIATE_SIZE: int = 4096
PROJECTOR_ACTIVATION: str = "relu2"
PROJECTOR_NORM_EPS: float = 1e-5
PROJECTOR_OUTPUT_DIM: int = 2048     # == LLM hidden size

# ---------------------------------------------------------------------------
# LLM decoder dims (NemotronDenseConfig)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2048
LLM_NUM_LAYERS: int = 28
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 8            # GQA (16 Q / 8 KV)
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 9216
LLM_VOCAB_SIZE: int = 205312
LLM_MAX_POS_EMB: int = 131072
LLM_ROPE_THETA: float = 100_000_000.0  # 1e8
LLM_RMS_NORM_EPS: float = 1e-5
LLM_HIDDEN_ACT: str = "relu2"        # squared ReLU, NOT SwiGLU
LLM_TIE_EMBEDDINGS: bool = False     # separate lm_head weight

# NemotronDense uses plain (unscaled) numerics.
LLM_ATTENTION_SCALE: float = 1.0 / (LLM_HEAD_DIM ** 0.5)
LLM_RESIDUAL_MULTIPLIER: float = 1.0
LLM_EMBEDDING_MULTIPLIER: float = 1.0
LLM_LOGITS_SCALING: float = 1.0

# ---------------------------------------------------------------------------
# Tokenisation / multimodal
# ---------------------------------------------------------------------------
SOUND_TOKEN_ID: int = 29             # <so_embedding> — audio placeholder
SOUND_START_TOKEN_ID: int = 30       # <so_start>
SOUND_END_TOKEN_ID: int = 31         # <so_end>
EOS_TOKEN_ID: int = 11               # <|im_end|>
PAD_TOKEN_ID: int = 0
BOS_TOKEN_ID: int = 1

SOUND_EMBEDDING_SIZE: int = 750      # per 30 s clip
SOUND_CLIP_DURATION: float = 30.0    # seconds
SOUND_TARGET_RATE: int = 16000       # Hz

SOUND_TOKEN_STR: str = "<so_embedding>"
SOUND_START_TOKEN_STR: str = "<so_start>"
SOUND_END_TOKEN_STR: str = "<so_end>"
SOUND_PLACEHOLDER_STR: str = "<sound>"
IM_END_TOKEN_STR: str = "<|im_end|>"

DEFAULT_TASK_PROMPT: str = "Transcribe the speech in the input audio."
"""ASR task instruction (matches the model card / LibriSpeech eval recipe)."""

# ---------------------------------------------------------------------------
# KV cache sizing
# ---------------------------------------------------------------------------
DEFAULT_MAX_CACHE_LEN: int = 4096

# ---------------------------------------------------------------------------
# Correctness tolerances for kernel / graph comparisons (bf16 eager reference)
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
LLM_LOGIT_ATOL: float = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
GOLDEN_DIR: Path = REPO_ROOT / "golden" / "audex"
