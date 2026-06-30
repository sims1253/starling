"""Architecture constants for CohereLabs/cohere-transcribe-03-2026.

Single source of truth for the cohere-transcribe dims, mirroring the layout of
``starling.config`` (granite) and the other model packages. Values are taken
verbatim from the model card ``config.json``.

cohere-transcribe is the repo's first **seq2seq encoder-decoder** (Whisper-
style) model: a Parakeet FastConformer encoder + an 8-layer Transformer decoder
with self-attention AND cross-attention.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "CohereLabs/cohere-transcribe-03-2026"
"""HF hub repo id for cohere-transcribe-03-2026 (~2B params, conformer enc-dec)."""

SAMPLE_RATE: int = 16000
MAX_AUDIO_CLIP_S: int = 35
"""Audio is auto-chunked into <= this many seconds by the processor (overlap 5s)."""

# ---------------------------------------------------------------------------
# Encoder (Parakeet FastConformer) — values from config.encoder
# ---------------------------------------------------------------------------
ENC_D_MODEL: int = 1280
ENC_NUM_LAYERS: int = 48
ENC_NUM_HEADS: int = 8
ENC_HEAD_DIM: int = 160                 # 1280 // 8
ENC_FEAT_IN: int = 128                  # mel bins
ENC_SUBSAMPLING_FACTOR: int = 8
ENC_CONV_KERNEL_SIZE: int = 9
ENC_SUBSAMPLING_CONV_CHANNELS: int = 256
ENC_FF_EXPANSION: int = 4
ENC_POS_EMB_MAX_LEN: int = 5000

# ---------------------------------------------------------------------------
# Decoder (Transformer: self-attn + cross-attn, ReLU MLP, LayerNorm, learned
# positional embeddings — no RoPE)
# ---------------------------------------------------------------------------
DEC_HIDDEN_SIZE: int = 1024
DEC_NUM_LAYERS: int = 8
DEC_NUM_ATTN_HEADS: int = 8
DEC_NUM_KV_HEADS: int = 8               # no GQA (kv == heads)
DEC_HEAD_DIM: int = 128
DEC_INTERMEDIATE_SIZE: int = 4096
DEC_VOCAB_SIZE: int = 16384
DEC_MAX_POS_EMB: int = 1024             # learned positional embedding table size
DEC_HIDDEN_ACT: str = "relu"

# encoder hidden is projected to decoder hidden by decoder.proj before cross-attn
ENC_PROJ_DIM: int = DEC_HIDDEN_SIZE     # 1024

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
PAD_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3
BOS_TOKEN_ID: int = 4
DECODER_PROMPT_LEN: int = 10
"""Length of the chat-format decoder_input_ids the processor emits
(language + control slots)."""

# ---------------------------------------------------------------------------
# Correctness tolerances
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
DECODER_LOGIT_ATOL: float = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""Repository root (the dir that contains src/, golden/, …)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden"
"""Directory where reference tensors are persisted (gitignored)."""
