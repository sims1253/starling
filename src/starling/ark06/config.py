"""Architecture constants for the ARK-ASR-0.6B megakernel pipeline.

Everything the fused kernels, CUDA-graph capture, and correctness checks need to
size buffers, decode shapes, or compare against the eager reference lives here so
there is a single source of truth for the Audio8/ARK-ASR-0.6B architecture:
a Whisper encoder (32 layers, d_model 1280) feeding an MLP adapter that merges
audio frames by 4 into the Qwen2.5-0.5B-class decoder hidden size (896).

Reuse contract with the 3B track: the 0.6B ships byte-identical remote modeling
code (``modeling_arkasr.py`` / ``modeling_audio.py`` / ``processing_arkasr.py`` /
``configuration_arkasr.py`` / ``chat_template.jinja`` / ``preprocessor_config.json``),
identical special-token ids, the same adapter (MLP, merge factor 4, gelu), the
same default instruction, and the same Whisper-large-v3 encoder dims. Only the
LLM decoder dims below differ (24 layers, 14 Q heads, head_dim 64, 4864-wide
MLP, 163958-token vocab whose trailing ~12k bicodec rows ASR never emits). The
megakernels derive all other dims from the loaded tensors, so the ark track's
encoder/LLM/pipeline modules are reused directly.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "Audio8/ARK-ASR-0.6B"
"""HF hub repo id for the ARK-ASR-0.6B model (loaded with trust_remote_code=True)."""

DEFAULT_INSTRUCTION: str = "Transcribe the audio to text."
"""Default user instruction wrapped around the audio placeholder tokens."""

# ---------------------------------------------------------------------------
# Encoder dims (Whisper encoder inside AudioMLPAdapter; identical to the 3B)
# ---------------------------------------------------------------------------
ENCODER_D_MODEL: int = 1280          # whisper encoder hidden size
ENCODER_NUM_LAYERS: int = 32         # whisper encoder layers
ENCODER_NUM_HEADS: int = 20          # whisper attention heads
ENCODER_NUM_MEL_BINS: int = 128      # mel feature bins (input mel channels)
ENCODER_MAX_SOURCE_POSITIONS: int = 1500  # whisper positional embedding length

# ---------------------------------------------------------------------------
# Adapter dims (AudioMLPAdapter.adapting: 1280*4 -> 896*2 -> 896)
# ---------------------------------------------------------------------------
ADAPTER_MERGE_FACTOR: int = 4        # audio frames merged per LLM token
ADAPTER_HIDDEN: int = 896            # adapter output dim == LLM hidden size

# ---------------------------------------------------------------------------
# LLM dims (Qwen2.5-0.5B-class decoder trunk, m.model)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 896
LLM_NUM_LAYERS: int = 24
LLM_NUM_ATTN_HEADS: int = 14
LLM_NUM_KV_HEADS: int = 2            # GQA (same as the 3B)
LLM_HEAD_DIM: int = 64               # == hidden_size // num_attn_heads
LLM_INTERMEDIATE_SIZE: int = 4864
LLM_VOCAB_SIZE: int = 163958
LLM_RMS_NORM_EPS: float = 1e-6
LLM_ROPE_THETA: float = 1000000.0
LLM_MAX_POS_EMB: int = 32768

# Qwen2 uses plain (unscaled) numerics; these are 1.0 / standard so the fused
# decode path matches the model's own layers bit-for-bit.
LLM_ATTENTION_SCALE: float = 1.0 / (LLM_HEAD_DIM ** 0.5)
"""Standard attention scale 1/sqrt(head_dim) = 1/sqrt(64)."""
LLM_RESIDUAL_MULTIPLIER: float = 1.0
"""Qwen2 residuals are unscaled (x + delta), unlike granite's 0.22."""
LLM_EMBEDDING_MULTIPLIER: float = 1.0
"""Qwen2 has no embedding multiplier (granite had 12.0)."""
LLM_LOGITS_SCALING: float = 1.0
"""Qwen2 logits are unscaled (granite divided by 8.0)."""

# ---------------------------------------------------------------------------
# Tokenisation / multimodal (identical to the 3B track)
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
# Greedy-decode token suppression (the model card's bad_words_ids recipe)
# ---------------------------------------------------------------------------
# ARK-ASR-0.6B degenerates under plain greedy decode: after ~11 correct words it
# spirals into repeated special tokens. The card README section
# "build_bad_words_ids" prescribes suppressing
#   bad = (all_special_ids ∪ added-vocab tokens "<...>") − {eos_token_id},
# which restores the exact reference transcript. EOS must remain emittable or
# the decode can never terminate. Computed lazily from the loaded tokenizer
# (robust to vocab drift) — no hard-coded id list.


def build_bad_token_ids(tokenizer) -> set[int]:
    """Build the card-recipe ban set from a loaded tokenizer.

    ``bad = (all_special_ids ∪ added-vocab "<...>" tokens) − {eos}``.
    """
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = EOS_TOKEN_ID
    bad: set[int] = set(getattr(tokenizer, "all_special_ids", None) or [])
    get_added = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added):
        try:
            added = get_added() or {}
        except Exception:
            added = {}
        for tok_str, tid in added.items():
            if (
                isinstance(tok_str, str)
                and len(tok_str) >= 2
                and tok_str.startswith("<")
                and tok_str.endswith(">")
            ):
                bad.add(int(tid))
    else:
        # Fallback for tokenizer shims exposing only added_tokens_decoder.
        dec = getattr(tokenizer, "added_tokens_decoder", None) or {}
        try:
            items = dec.items()
        except AttributeError:
            items = []
        for tid, tok in items:
            s = getattr(tok, "content", tok)
            if isinstance(s, str) and len(s) >= 2 and s.startswith("<") and s.endswith(">"):
                try:
                    bad.add(int(tid))
                except (TypeError, ValueError):
                    continue
    # Guard against vocab drift (ids outside the head are un-emittable anyway).
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        bad = {i for i in bad if 0 <= i < vocab_size}
    bad.discard(int(eos))
    return bad

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
