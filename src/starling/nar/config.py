"""Module-level constants for the Granite-Speech-4.1-2b-NAR track.

Single source of truth for the non-autoregressive granite-speech architecture.
These mirror the model's own ``config.json`` (captured once, frozen here so the
megakernel code can size static buffers without touching the live config).
"""

from __future__ import annotations

from pathlib import Path

MODEL_ID: str = "ibm-granite/granite-speech-4.1-2b-nar"
"""HF hub repo id for the Granite Speech 4.1 2B non-autoregressive model."""

# ---------------------------------------------------------------------------
# Encoder (GraniteSpeechNarCTCEncoder) — 16-block conformer
# ---------------------------------------------------------------------------
ENCODER_INPUT_DIM: int = 160          # stacked 80-bin mel pairs
ENCODER_HIDDEN_DIM: int = 1024
ENCODER_NUM_LAYERS: int = 16
ENCODER_NUM_HEADS: int = 8
ENCODER_HEAD_DIM: int = 128
ENCODER_CONTEXT_SIZE: int = 200       # block attention span
ENCODER_MAX_POS_EMB: int = 512
ENCODER_OUTPUT_DIM: int = 348         # mid-layer CTC head dim
ENCODER_SELF_CONDITIONING_LAYER: int = 8
ENCODER_BPE_POOLING_WINDOW: int = 4
ENCODER_CONV_KERNEL: int = 15
ENCODER_LAYER_INDICES: tuple[int, ...] = (4, 8, 12, -1)
"""Encoder layers concatenated as projector input."""

# ---------------------------------------------------------------------------
# Projector (GraniteSpeechNarProjector) — windowed Q-Former
# ---------------------------------------------------------------------------
PROJECTOR_ENCODER_DIM: int = 1024
PROJECTOR_NUM_ENCODER_LAYERS: int = 4
PROJECTOR_HIDDEN_SIZE: int = 2048
PROJECTOR_LLM_DIM: int = 2048
PROJECTOR_BLOCK_SIZE: int = 15
PROJECTOR_DOWNSAMPLE_RATE: int = 5

# ---------------------------------------------------------------------------
# LLM editor (Granite-4.0-1b, bidirectional / is_causal=False)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2048
LLM_NUM_LAYERS: int = 40
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 4           # GQA
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 4096
LLM_VOCAB_SIZE: int = 100352
LLM_RMS_NORM_EPS: float = 1e-5
LLM_ATTENTION_MULTIPLIER: float = 0.0078125
LLM_EMBEDDING_MULTIPLIER: float = 12.0
LLM_RESIDUAL_MULTIPLIER: float = 0.22
LLM_LOGITS_SCALING: float = 8.0

BLANK_TOKEN_ID: int = 100257
"""CTC blank symbol (reuses eos). Used both in the encoder BPE head and the LLM
output collapse."""
MIN_EDIT_SEQUENCE_LENGTH: int = 8

# ---------------------------------------------------------------------------
# Feature extractor (torchaudio log-mel)
# ---------------------------------------------------------------------------
FEAT_SAMPLING_RATE: int = 16000
FEAT_N_FFT: int = 512
FEAT_WIN_LENGTH: int = 400
FEAT_HOP_LENGTH: int = 160
FEAT_N_MELS: int = 80               # 80-band, stacked to 160

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
GOLDEN_DIR: Path = REPO_ROOT / "golden" / "nar"
"""Directory where NAR golden reference artefacts are persisted (gitignored)."""

# ---------------------------------------------------------------------------
# Correctness tolerances
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
LLM_LOGIT_ATOL: float = 0.05
