"""Architecture constants for bosonai/higgs-audio-v3-stt.

Single source of truth for the Higgs-Audio-v3 dims, mirroring the layout of
``starling.config`` (granite) and the other model packages. Values are taken
verbatim from the model card ``config.json``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "bosonai/higgs-audio-v3-stt"
"""HF hub repo id for Higgs-Audio v3 STT."""

# ---------------------------------------------------------------------------
# Audio encoder (Whisper-large-v3 derived: 32 Whisper layers, d=1280, avgpool 2x)
# ---------------------------------------------------------------------------
AUDIO_NUM_MEL_BINS: int = 128
AUDIO_D_MODEL: int = 1280
AUDIO_NUM_LAYERS: int = 32
AUDIO_NUM_HEADS: int = 20
AUDIO_FFN_DIM: int = 5120
AUDIO_MAX_POS: int = 1500  # whisper max_source_positions (post-conv frames)
AUDIO_FRAME_RATE: float = 25.0  # whisper hidden states per second

# ---------------------------------------------------------------------------
# LLM decoder (Qwen3-1.7B-Base)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2048
LLM_NUM_LAYERS: int = 28
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 8            # GQA
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 6144
LLM_VOCAB_SIZE: int = 151936
LLM_MAX_POS_EMB: int = 32768
LLM_ROPE_THETA: float = 1_000_000.0
LLM_RMS_NORM_EPS: float = 1e-6
# No embedding multiplier, no logit scaling for higgs (unlike granite).

# ---------------------------------------------------------------------------
# Tokens (Qwen3 tokenizer)
# ---------------------------------------------------------------------------
LLM_PAD_TOKEN_ID: int = 151643       # <|endoftext|>
LLM_EOS_TOKEN_ID: int = 151643       # <|endoftext|> (generation_config eos)
IM_END_TOKEN_ID: int = 151645        # <|im_end|>  (ChatML turn end -> also stops)
AUDIO_IN_TOKEN_IDX: int = 151672     # <|AUDIO|> placeholder
AUDIO_OUT_TOKEN_IDX: int = 151673    # <|AUDIO_OUT|> (unused for ASR)
AUDIO_BOS_TOKEN_ID: int = 151669     # <|audio_bos|>
AUDIO_EOS_TOKEN_ID: int = 151670     # <|audio_eos|>

# ASR stops on either <|endoftext|> or <|im_end|> (matches transcribe() stop_strings).
EOS_TOKEN_IDS: tuple[int, ...] = (LLM_EOS_TOKEN_ID, IM_END_TOKEN_ID)

# tps: model emits ~12.5 LLM tokens per second of audio (config.tps).
LLM_TOKENS_PER_SECOND: float = 12.5

# mel frontend (Whisper-large-v3 processor)
MEL_SR: int = 16000
MEL_DIM: int = AUDIO_NUM_MEL_BINS

# ---------------------------------------------------------------------------
# Correctness tolerances (the megakernel is byte-exact, these are safety nets)
# ---------------------------------------------------------------------------
LLM_LOGIT_ATOL: float = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""Repository root (the dir that contains src/, golden/, …)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden"
"""Directory where reference tensors are persisted (gitignored)."""
