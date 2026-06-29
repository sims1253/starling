"""Architecture constants for OpenMOSS-Team/MOSS-Transcribe-preview-2B.

Single source of truth for the MOSS-Transcribe dims, mirroring the layout of
``starling.config`` (granite) and the other model packages.  Values are taken
verbatim from the model card ``config.json``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "OpenMOSS-Team/MOSS-Transcribe-preview-2B"
"""HF hub repo id for MOSS-Transcribe preview 2B."""

# ---------------------------------------------------------------------------
# Audio encoder (Qwen3OmniMoeAudioEncoder)
# ---------------------------------------------------------------------------
AUDIO_NUM_MEL_BINS: int = 128          # mel feature dim (conv frontend input)
AUDIO_DOWNSAMPLE_HIDDEN: int = 480     # conv2d channel width
AUDIO_D_MODEL: int = 1280              # encoder hidden dim
AUDIO_NUM_LAYERS: int = 32             # encoder transformer blocks
AUDIO_NUM_HEADS: int = 20              # encoder attention heads
AUDIO_HEAD_DIM: int = 64               # d_model // heads = 1280 // 20
AUDIO_FFN_DIM: int = 5120              # encoder FFN intermediate
AUDIO_OUTPUT_DIM: int = 2048           # proj2 output (= adapter input)
AUDIO_MAX_POS: int = 1500              # sinusoid positional embedding table size
AUDIO_N_WINDOW: int = 50               # half chunk size (raw frames); chunk = 100
AUDIO_N_WINDOW_INFER: int = 800        # inference attention window (raw frames)
AUDIO_CONV_CHUNKSIZE: int = 500        # conv processing chunk (num chunks)

# ---------------------------------------------------------------------------
# LLM decoder (Qwen3)  -- defined before adapter (adapter output == LLM hidden)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2048
LLM_NUM_LAYERS: int = 28
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 8              # GQA
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 6144
LLM_VOCAB_SIZE: int = 151936
LLM_MAX_POS_EMB: int = 40960
LLM_ROPE_THETA: float = 1_000_000.0
LLM_RMS_NORM_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Adapter (MossGatedMLP: SiLU gate)
# ---------------------------------------------------------------------------
ADAPTER_INPUT_DIM: int = AUDIO_OUTPUT_DIM   # 2048
ADAPTER_HIDDEN_DIM: int = 8192              # gated-MLP intermediate
ADAPTER_OUTPUT_DIM: int = LLM_HIDDEN_SIZE   # 2048
LLM_NUM_LAYERS: int = 28
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 8              # GQA
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 6144
LLM_VOCAB_SIZE: int = 151936
LLM_MAX_POS_EMB: int = 40960
LLM_ROPE_THETA: float = 1_000_000.0
LLM_RMS_NORM_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Tokens (Qwen3 tokenizer)
# ---------------------------------------------------------------------------
LLM_PAD_TOKEN_ID: int = 151643         # <|endoftext|>
LLM_EOS_TOKEN_ID: int = 151645         # <|im_end|>
START_TOKEN_ID: int = 151644           # <|im_start|>
AUDIO_START_TOKEN_ID: int = 151669
AUDIO_END_TOKEN_ID: int = 151670
AUDIO_PLACEHOLDER_ID: int = 0          # token id used as the audio slot

# mel frontend (matches processor MelConfig the model was trained with)
MEL_SR: int = 16000
MEL_DIM: int = AUDIO_NUM_MEL_BINS
MEL_N_FFT: int = 640
MEL_HOP_LENGTH: int = 160

# audio tokens per second of audio (after the encoder downsample), per model card
AUDIO_TOKENS_PER_SECOND: float = 12.5

# ---------------------------------------------------------------------------
# Correctness tolerances
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
LLM_LOGIT_ATOL: float = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""Repository root (the dir that contains src/, golden/, …)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden"
"""Directory where reference tensors are persisted (gitignored)."""
