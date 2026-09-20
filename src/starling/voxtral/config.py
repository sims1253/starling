"""Architecture constants for the Voxtral Realtime megakernel pipeline.

Single source of truth for ``mistralai/Voxtral-Mini-4B-Realtime-2602``: a
Whisper-style causal audio encoder (32 layers, d_model 1280) feeding a
downsample-4 projector into a Ministral-3-class text decoder (26 layers,
hidden 3072, tied lm_head).

Architecture summary (verified against transformers 5.15
``models/voxtral_realtime`` source and the hub snapshot config):

* Features: Whisper-style log-mel, 128 bins, hop 160, n_fft = win 400, hann,
  slaney filters, STFT magnitude squared dropping the last freq bin, log10
  with clamp min 1e-10, a GLOBAL fixed log-mel max of 1.5 (streaming-safe,
  not per-utterance), floor at max-8, then (x+4)/4. ``center=True`` offline.
* Embedder: causal conv1d k3 s1 (left-pad 2) -> GELU -> causal conv1d k3 s2
  (left-pad 1) -> GELU -> transpose, giving (B, T//2, 1280).
* Audio encoder: 32 pre-norm layers, attention projects up to 32 heads x 64
  (q/v/o bias, k no bias), sliding window 750, RoPE theta 1e6, causal;
  SwiGLU MLP with down_proj bias; RMSNorm eps 1e-5; final norm. Takes
  ``past_key_values`` + conv ``padding_cache`` for streaming.
* Projector: reshape (B, T', 1280*4) -> Linear(5120 -> 3072, no bias) ->
  GELU -> Linear(3072 -> 3072, no bias). downsample_factor 4, i.e. 8 mel
  frames per audio token (12.5 audio tok/s, 80 ms each).
* Text decoder: 26 layers, hidden 3072, GQA 32q/8kv x head_dim 128 (no
  bias anywhere), sliding window 8192, RoPE theta 1e6, RMSNorm eps 1e-5,
  SwiGLU no bias, tied lm_head (vocab 131072), plus an AdaRMSNorm on the
  MLP branch only: ``h = h * (1 + ada_rms_norm(t_cond))`` after
  post_attention_layernorm, where ``t_cond`` is the sinusoidal
  TimeEmbedding(dim 3072)(num_delay_tokens). ``num_delay_tokens`` is FIXED
  per request (default 6), so each layer's modulation vector is a constant
  that is precomputable once per request instead of per decode step.

Audio injection is ADDITIVE: ``inputs_embeds += audio_embeds`` at every
position of the current forward slice (not scatter-into-slots). Each
generation step consumes a fixed slice of 4 pre-encoder embeds (= 1 audio
token); the encoder runs on just that slice with its own KV cache. The
fixed per-step shapes (encoder slice 4x1280, text step 1x3072) are what
make the future CUDA-graph decode path static-shaped; the offline v1 loop
here mirrors stock ``generate`` eagerly.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "mistralai/Voxtral-Mini-4B-Realtime-2602"
"""HF hub repo id (native transformers model, no trust_remote_code)."""

# ---------------------------------------------------------------------------
# Feature extraction (VoxtralRealtimeFeatureExtractor)
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000
NUM_MEL_BINS: int = 128
HOP_LENGTH: int = 160
N_FFT: int = 400
WIN_LENGTH: int = 400
GLOBAL_LOG_MEL_MAX: float = 1.5
"""Fixed global log-mel max (streaming-safe); per-utterance max is NOT used."""
LOG_MEL_FLOOR_RANGE: float = 8.0
"""Floor at (max - 8); then (x + 4) / 4."""
LOG_MEL_CLAMP_MIN: float = 1e-10
FEATURE_CENTER: bool = True
"""center=True for offline STFT."""

# ---------------------------------------------------------------------------
# Embedder dims (VoxtralRealtimeEmbedder: causal convs over mel bins)
# ---------------------------------------------------------------------------
EMBEDDER_IN_CHANNELS: int = 128        # == NUM_MEL_BINS
EMBEDDER_HIDDEN: int = 1280
EMBEDDER_KERNEL: int = 3
EMBEDDER_CONV1_STRIDE: int = 1
EMBEDDER_CONV1_LEFT_PAD: int = 2       # effective_k(3) - stride(1)
EMBEDDER_CONV2_STRIDE: int = 2
EMBEDDER_CONV2_LEFT_PAD: int = 1       # effective_k(3) - stride(2)

# ---------------------------------------------------------------------------
# Audio encoder dims (VoxtralRealtimeEncoderConfig)
# ---------------------------------------------------------------------------
ENCODER_HIDDEN: int = 1280
ENCODER_NUM_LAYERS: int = 32
ENCODER_NUM_HEADS: int = 32
ENCODER_NUM_KV_HEADS: int = 32         # no GQA; q/k/v heads all 32
ENCODER_HEAD_DIM: int = 64
ENCODER_INTERMEDIATE: int = 5120
ENCODER_SLIDING_WINDOW: int = 750
ENCODER_ROPE_THETA: float = 1_000_000.0
ENCODER_RMS_NORM_EPS: float = 1e-5
ENCODER_MLP_ACT: str = "silu"
"""Effective MLP activation is silu (SwiGLU): the MLP reads
``config.hidden_act``; ``activation_function="gelu"`` is unused."""

# ---------------------------------------------------------------------------
# Projector dims (VoxtralRealtimeMultiModalProjector)
# ---------------------------------------------------------------------------
DOWNSAMPLE_FACTOR: int = 4
"""Encoder frames grouped per audio token by the projector reshape."""
PROJECTOR_IN_DIM: int = 5120           # ENCODER_HIDDEN * DOWNSAMPLE_FACTOR
PROJECTOR_HIDDEN: int = 3072           # == LLM hidden size
PROJECTOR_ACT: str = "gelu"
AUDIO_LENGTH_PER_TOK: int = 8
"""Mel frames per audio token (12.5 audio tok/s, 80 ms each)."""

# ---------------------------------------------------------------------------
# Text decoder dims (VoxtralRealtimeTextConfig)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 3072
LLM_NUM_LAYERS: int = 26
LLM_NUM_ATTN_HEADS: int = 32
LLM_NUM_KV_HEADS: int = 8              # GQA
LLM_HEAD_DIM: int = 128
LLM_INTERMEDIATE_SIZE: int = 9216
LLM_VOCAB_SIZE: int = 131072
LLM_SLIDING_WINDOW: int = 8192
LLM_ROPE_THETA: float = 1_000_000.0
LLM_RMS_NORM_EPS: float = 1e-5
LLM_HIDDEN_ACT: str = "silu"           # SwiGLU, no biases anywhere
LLM_TIE_EMBEDDINGS: bool = True

# ---------------------------------------------------------------------------
# Delay / streaming-pad accounting (mistral-common audio config in tekken.json)
# ---------------------------------------------------------------------------
DEFAULT_NUM_DELAY_TOKENS: int = 6
"""Fixed per request; each layer's ada modulation is constant for it."""
TIME_EMBEDDING_DIM: int = 3072         # == LLM_HIDDEN_SIZE
TIME_EMBEDDING_THETA: float = 10_000.0
ADA_BOTTLENECK: int = 32               # Linear(3072->32) -> GELU -> Linear(32->3072)
STREAMING_LEFT_PAD_TOKENS: int = 32
STREAMING_BUFFER_TOKENS: int = 10      # OFFLINE_STREAMING_BUFFER_TOKENS
STREAMING_RIGHT_PAD_TOKENS: int = DEFAULT_NUM_DELAY_TOKENS + 1 + STREAMING_BUFFER_TOKENS
RAW_SAMPLES_PER_AUDIO_TOK: int = HOP_LENGTH * AUDIO_LENGTH_PER_TOK  # 160 * 8
"""Waveform samples per audio token (== sampling_rate // frame_rate 12.5)."""

# ---------------------------------------------------------------------------
# Tokenisation / generation
# ---------------------------------------------------------------------------
BOS_TOKEN_ID: int = 1
EOS_TOKEN_ID: int = 2
PAD_TOKEN_ID: int = 11

# ---------------------------------------------------------------------------
# KV cache sizing
# ---------------------------------------------------------------------------
DEFAULT_MAX_CACHE_LEN: int = 4096
"""Upper bound checked against prompt + decode budget (v1 uses DynamicCache;
a static cache with this length slots in later)."""

# ---------------------------------------------------------------------------
# Correctness tolerances for kernel / graph comparisons (bf16 eager reference)
# ---------------------------------------------------------------------------
ENCODER_ATOL: float = 2e-2
LLM_LOGIT_ATOL: float = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""Repository root (the dir that contains src/, tests/, ...)."""

GOLDEN_DIR: Path = REPO_ROOT / "golden"
"""Directory holding reference JSON captures (gitignored)."""

GOLDEN_FILENAME: str = "voxtral_reference.json"
GOLDEN_PATH: Path = GOLDEN_DIR / GOLDEN_FILENAME
