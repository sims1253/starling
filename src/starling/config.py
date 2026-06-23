"""Module-level constants for the starling foundation layer.

Everything later phases (Triton kernels, fused pipelines) need to size tiles,
check shapes, or compare against reference outputs lives here so there is a
single source of truth for the Granite-Speech-4.1-2b architecture.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "ibm-granite/granite-speech-4.1-2b"
"""HF hub repo id for the Granite Speech 4.1 2B model."""

SAMPLE_AUDIO_FILENAME: str = "multilingual_sample.wav"
"""Sample wav shipped in the model repo (24.9s, mono 16kHz, multilingual)."""

# ---------------------------------------------------------------------------
# Encoder dims (GraniteSpeechCTCEncoder)
# ---------------------------------------------------------------------------
ENCODER_INPUT_DIM: int = 160           # mel feature dim
ENCODER_HIDDEN_DIM: int = 1024         # conformer hidden dim
ENCODER_NUM_LAYERS: int = 16           # conformer blocks
ENCODER_NUM_HEADS: int = 8             # attention heads
ENCODER_HEAD_DIM: int = 128            # dim per head
ENCODER_CONV_KERNEL: int = 15          # conv kernel size
ENCODER_CONTEXT_SIZE: int = 200        # block attention span
ENCODER_MAX_POS_EMB: int = 512         # max positional embedding
ENCODER_OUTPUT_DIM: int = 348          # output projection dim

# ---------------------------------------------------------------------------
# Projector dims (Blip2QFormer based GraniteSpeechEncoderProjector)
# ---------------------------------------------------------------------------
PROJECTOR_WINDOW_SIZE: int = 15        # blocks the encoder output is grouped into
PROJECTOR_DOWNSAMPLE_RATE: int = 5     # frames emitted per block
PROJECTOR_NUM_QUERIES: int = 3         # q-former queries per block
PROJECTOR_HIDDEN: int = 1024           # q-former hidden size
PROJECTOR_OUTPUT_DIM: int = 2048       # output dim fed to LLM (= LLM hidden)

# ---------------------------------------------------------------------------
# LLM dims (Granite-4.0-1b base decoder)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2048
LLM_NUM_LAYERS: int = 40
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 4              # GQA
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 4096
LLM_VOCAB_SIZE: int = 100353
LLM_MAX_POS_EMB: int = 4096
LLM_ROPE_THETA: float = 10000.0
LLM_RMS_NORM_EPS: float = 1e-5
# Granite-specific multipliers
LLM_ATTENTION_MULTIPLIER: float = 0.0078125
LLM_EMBEDDING_MULTIPLIER: float = 12.0
LLM_RESIDUAL_MULTIPLIER: float = 0.22
LLM_LOGITS_SCALING: float = 8.0

# ---------------------------------------------------------------------------
# Tokenisation / multimodal
# ---------------------------------------------------------------------------
AUDIO_TOKEN_ID: int = 100352
"""The `<|audio|>` token id; positions in input_ids with this id are clobbered
by the projected audio embeddings."""
LLM_PAD_TOKEN_ID: int = 100256
LLM_BOS_TOKEN_ID: int = 100257
LLM_EOS_TOKEN_ID: int = 100257

DEFAULT_TASK_PROMPT: str = (
    "transcribe the speech with proper punctuation and capitalization."
)

# ---------------------------------------------------------------------------
# Correctness tolerances for later-phase kernel comparisons
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
"""Absolute tolerance when comparing encoder outputs (bf16 eager reference)."""
LLM_LOGIT_ATOL: float = 0.05
"""Absolute tolerance when comparing LLM logits (bf16 eager reference)."""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root (the dir that contains src/, golden/, …)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden"
"""Directory where reference tensors are persisted (gitignored)."""

TRACES_DIR: Path = REPO_ROOT / "traces"
"""Directory where chrome traces are written (gitignored)."""
