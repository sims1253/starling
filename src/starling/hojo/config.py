"""Architecture constants for HojoAI/Hojo-ASR-V1.

Single source of truth for the Hojo-ASR-V1 dimensions, mirroring the layout of
``starling.higgs.config`` and the other model packages. Values are taken verbatim
from the model's ``config.yaml`` and the live ``hojo-asr==0.1.3`` reference
package (cross-checked against ``merged_full_model.safetensors``).

Forward path
------------
``Whisper-large-v3 mel -> Qwen3-Omni audio tower (32 layers, d=1280) ->
WeNet Conformer bottleneck (2 layers, d=2560) -> ln_speech ->
Qwen3-4B decoder (beam-4)``. ~5.19 B params.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "HojoAI/Hojo-ASR-V1"
"""HF hub repo id for Hojo-ASR-V1 (also resolvable as a local bundle dir)."""

# ---------------------------------------------------------------------------
# Mel frontend (Whisper-large-v3 feature extractor)
# ---------------------------------------------------------------------------
MEL_SR: int = 16000
"""Expected audio sample rate (the Whisper extractor downsamples to 16 kHz)."""
MEL_NUM_BINS: int = 128
"""Whisper-large-v3 mel channels."""
MEL_HOP_LENGTH: int = 160
"""Whisper hop length (samples) -> 100 frames/sec at 16 kHz."""
MEL_N_FFT: int = 400
"""Whisper n_fft (samples)."""

# ---------------------------------------------------------------------------
# Qwen3-Omni audio tower (speech_encoder.*, F32 weights)
# ---------------------------------------------------------------------------
TOWER_D_MODEL: int = 1280
TOWER_NUM_LAYERS: int = 32
TOWER_NUM_HEADS: int = 20
TOWER_HEAD_DIM: int = 64
TOWER_FFN_DIM: int = 5120
TOWER_DOWNSAMPLE_HIDDEN: int = 480
TOWER_MAX_SOURCE_POSITIONS: int = 1500
"""SinusoidsPositionEmbedding length (max post-conv frames per chunk)."""
TOWER_OUTPUT_DIM: int = 2048
"""proj2 output dim (input to the Conformer bottleneck)."""
TOWER_SCALING: float = TOWER_HEAD_DIM ** -0.5
"""Attention scaling 1/sqrt(head_dim) = 1/8 (MHA uses 64**-0.5)."""
TOWER_N_WINDOW: int = 1500
"""Qwen3-Omni audio window (set on config in HOJO_ASR.__init__)."""
TOWER_N_WINDOW_INFER: int = 3000
"""Inference window for output-length computation."""
TOWER_CONV_CHUNKSIZE: int = 500
"""Conv2d chunking window (avoid OOM during the 3 conv2d strides)."""

# ---------------------------------------------------------------------------
# WeNet Conformer bottleneck (bottleneck.*, F32) + ln_speech
# ---------------------------------------------------------------------------
BOTTLENECK_HIDDEN: int = 2560
"""Conformer output dim == LLM hidden size."""
BOTTLENECK_INPUT: int = TOWER_OUTPUT_DIM
"""LinearNoSubsampling input (== tower proj2 output)."""
BOTTLENECK_LINEAR_UNITS: int = 640
"""Conformer FFN intermediate size (w_1: 2560->640, w_2: 640->2560)."""
BOTTLENECK_NUM_BLOCKS: int = 2
BOTTLENECK_NUM_HEADS: int = 4
BOTTLENECK_HEAD_DIM: int = BOTTLENECK_HIDDEN // BOTTLENECK_NUM_HEADS  # 640
BOTTLENECK_CNN_KERNEL: int = 15
"""Depthwise conv kernel size in the conv module."""
BOTTLENECK_POS_ENC_MAX_LEN: int = 5000
"""RelPositionalEncoding buffer length ``pos_enc.pe [1,5000,2560]``."""

# ---------------------------------------------------------------------------
# LLM decoder (Qwen3-4B, decoder_model.*, BF16)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 2560
LLM_NUM_LAYERS: int = 36
LLM_NUM_ATTN_HEADS: int = 32
LLM_NUM_KV_HEADS: int = 8             # GQA
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 9728
LLM_VOCAB_SIZE: int = 151670          # tokenizer length after pad-token resize
LLM_MAX_POS_EMB: int = 40960
LLM_ROPE_THETA: float = 5_000_000.0
LLM_RMS_NORM_EPS: float = 1e-6
# Qwen3: plain numerics (no embedding multiplier, no logit scaling) like higgs.
LLM_ATTENTION_SCALE: float = 1.0 / (LLM_HEAD_DIM ** 0.5)
LLM_RESIDUAL_MULTIPLIER: float = 1.0
LLM_EMBEDDING_MULTIPLIER: float = 1.0
LLM_LOGITS_SCALING: float = 1.0
# lm_head is SEPARATE (not tied to embed_tokens) -- decoder_model.lm_head.weight.

# ---------------------------------------------------------------------------
# Tokens (Qwen3 tokenizer; Hojo adds a [PAD] special token then resizes)
# ---------------------------------------------------------------------------
BOS_TOKEN_ID: int = 151644            # <|im_start|> (set by HOJO_ASR for Qwen)
EOS_TOKEN_ID: int = 151645            # <|im_end|> (generation eos)
PAD_TOKEN_ID: int = 151645            # pad auto-set to <|im_end|> (eos)
ENDOFTEXT_TOKEN_ID: int = 151643      # <|endoftext|> (also stripped at decode)

# ---------------------------------------------------------------------------
# Beam search (matches config.yaml ``generate:`` + the golden decode block)
# ---------------------------------------------------------------------------
NUM_BEAMS: int = 4
DO_SAMPLE: bool = False
REPETITION_PENALTY: float = 2.0
LENGTH_PENALTY: float = 1.0
MAX_NEW_TOKENS: int = 200
MIN_LENGTH: int = 1
TEMPERATURE: float = 1.0
TOP_P: float = 0.9
# StopOnTokenSequences([-100]) is a no-op suffix stop (the real stop is eos).
STOP_TOKEN_SEQS: tuple[tuple[int, ...], ...] = ((-100,),)
# Hojo's generate feeds inputs_embeds (no input_ids), so the LLM prefix is
#   bos_embed (1) + speech_embeds (N)  -> prompt_len = N + 1.

# ---------------------------------------------------------------------------
# Autocast
# ---------------------------------------------------------------------------
AUTOMIX_DTYPE: str = "float16"
"""The reference runs the whole ``infer`` (encoder + generate) under fp16
``torch.amp.autocast('cuda')``; matching it keeps the bf16 decoder / f32 encoder
casts byte-exact with the golden."""

# ---------------------------------------------------------------------------
# Correctness tolerances (the decode path is byte-exact vs the golden oracle;
# these are safety nets for any future fused/optimized paths)
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
LLM_LOGIT_ATOL: float = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""Repository root (the dir that contains src/, golden/, ...)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden"
"""Directory where reference tensors are persisted (gitignored)."""

DEFAULT_MODEL_DIR: Path = REPO_ROOT / ".hf-cache" / "hojo-asr-v1"
"""Local model bundle dir (config.yaml + merged_full_model.safetensors + submodels)."""
