"""Architecture constants for the ARK-ASR-3B megakernel pipeline.

Everything the fused kernels, CUDA-graph capture, and correctness checks need to
size buffers, decode shapes, or compare against the eager reference lives here so
there is a single source of truth for the AutoArk-AI/ARK-ASR-3B architecture:
a Whisper encoder (32 layers, d_model 1280) feeding an MLP adapter that merges
audio frames by 4 into the Qwen2.5 decoder hidden size (2048).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "AutoArk-AI/ARK-ASR-3B"
"""HF hub repo id for the ARK-ASR-3B model (loaded with trust_remote_code=True)."""

DEFAULT_INSTRUCTION: str = "Transcribe the audio to text."
"""Default user instruction wrapped around the audio placeholder tokens."""

# ---------------------------------------------------------------------------
# Encoder dims (Whisper encoder inside AudioMLPAdapter)
# ---------------------------------------------------------------------------
ENCODER_D_MODEL: int = 1280          # whisper encoder hidden size
ENCODER_NUM_LAYERS: int = 32         # whisper encoder layers
ENCODER_NUM_HEADS: int = 20          # whisper attention heads
ENCODER_NUM_MEL_BINS: int = 128      # mel feature bins (input mel channels)
ENCODER_MAX_SOURCE_POSITIONS: int = 1500  # whisper positional embedding length

# ---------------------------------------------------------------------------
# Adapter dims (AudioMLPAdapter.adapting: 1280*4 -> 2048*2 -> 2048)
# ---------------------------------------------------------------------------
ADAPTER_MERGE_FACTOR: int = 4        # audio frames merged per LLM token
ADAPTER_HIDDEN: int = 2048           # adapter output dim == LLM hidden size

# ---------------------------------------------------------------------------
# LLM dims (Qwen2.5 decoder trunk, m.model)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2048
LLM_NUM_LAYERS: int = 36
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 2            # GQA
LLM_HEAD_DIM: int = 128              # == hidden_size // num_attn_heads
LLM_INTERMEDIATE_SIZE: int = 11008
LLM_VOCAB_SIZE: int = 151936
LLM_RMS_NORM_EPS: float = 1e-6
LLM_ROPE_THETA: float = 1000000.0
LLM_MAX_POS_EMB: int = 32768

# Qwen2 uses plain (unscaled) numerics; these are 1.0 / standard so the fused
# decode path matches the model's own layers bit-for-bit.
LLM_ATTENTION_SCALE: float = 1.0 / (LLM_HEAD_DIM ** 0.5)
"""Standard attention scale 1/sqrt(head_dim) = 1/sqrt(128)."""
LLM_RESIDUAL_MULTIPLIER: float = 1.0
"""Qwen2 residuals are unscaled (x + delta), unlike granite's 0.22."""
LLM_EMBEDDING_MULTIPLIER: float = 1.0
"""Qwen2 has no embedding multiplier (granite had 12.0)."""
LLM_LOGITS_SCALING: float = 1.0
"""Qwen2 logits are unscaled (granite divided by 8.0)."""

# ---------------------------------------------------------------------------
# Tokenisation / multimodal
# ---------------------------------------------------------------------------
AUDIO_TOKEN_ID: int = 151663
"""The `<|audio|>` token id; positions with this id are clobbered by the
adapter's audio embeddings."""
BEGIN_AUDIO_ID: int = 151666          # <|begin_of_audio|>
END_AUDIO_ID: int = 151667            # <|end_of_audio|>
USER_ID: int = 151665                 # <|user|>
ASSISTANT_ID: int = 151668            # <|assistant|>
EOS_TOKEN_ID: int = 151645            # <|im_end|>
PAD_TOKEN_ID: int = 151643
BOS_TOKEN_ID: int = 151643

# Prompt template token strings.
USER_TOKEN: str = "<|user|>"
ASSISTANT_TOKEN: str = "<|assistant|>"
BEGIN_AUDIO_TOKEN: str = "<|begin_of_audio|>"
END_AUDIO_TOKEN: str = "<|end_of_audio|>"
AUDIO_TOKEN: str = "<|audio|>"

# ---------------------------------------------------------------------------
# KV cache sizing
# ---------------------------------------------------------------------------
DEFAULT_MAX_CACHE_LEN: int = 4096
"""Static KV cache length. Must fit prompt T + max_new_tokens; 4096 is safe for
ASR-length utterances and responses."""

# ---------------------------------------------------------------------------
# Correctness tolerances for kernel / graph comparisons
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
"""Absolute tolerance when comparing encoder outputs (bf16 eager reference)."""
LLM_LOGIT_ATOL: float = 0.05
"""Absolute tolerance when comparing LLM logits (bf16 eager reference)."""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root (the dir that contains src/, golden/, ...)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden"
"""Directory where reference tensors are persisted (gitignored)."""
