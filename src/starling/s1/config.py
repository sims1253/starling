"""S1-mini architecture + prompt constants (single source of truth).

S1-mini (``superwhisper/s1-mini``) is Superwhisper's 0.6B text normalizer for
speech-to-text output: a decoder-only Qwen3 causal LM fine-tuned from
``Qwen/Qwen3-0.6B``. It takes a raw ASR transcript (usually lowercase,
unpunctuated, full of fillers and self-corrections) and rewrites it as clean
written text. No audio front-end — text in, text out.

All values come from ``superwhisper/s1-mini/config.json`` (Qwen3Config,
``transformers >= 4.51``).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "superwhisper/s1-mini"
"""HF hub repo id. Revision ``v1`` tag exists; we track main."""

# ---------------------------------------------------------------------------
# Decoder dims (Qwen3Config)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 1024
LLM_NUM_LAYERS: int = 28
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 8             # GQA (16 Q / 8 KV)
LLM_HEAD_DIM: int = 128               # explicit in config (2x hidden/n_heads)
LLM_INTERMEDIATE_SIZE: int = 3072
LLM_VOCAB_SIZE: int = 151936
LLM_MAX_POS_EMB: int = 40960
LLM_ROPE_THETA: float = 1_000_000.0
LLM_RMS_NORM_EPS: float = 1e-6
LLM_TIED_EMBEDDINGS: bool = True

# ---------------------------------------------------------------------------
# Tokenisation / prompt contract (from the model card — trained-on shapes only)
# ---------------------------------------------------------------------------
EOS_TOKEN_IDS: tuple[int, ...] = (151645, 151643)   # <|im_end|>, <|endoftext|>
PAD_TOKEN_ID: int = 151643

SYSTEM_PROMPT: str = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
"""Required, verbatim. The card warns: reworded system prompts hallucinate."""

STYLING_VALUES = ("casual", "semi-casual", "semi-formal", "formal")
STRUCTURE_VALUES = ("prose", "lists")
CONTEXT_VALUES = ("general", "email")

DEFAULT_STYLING: str = "semi-formal"
DEFAULT_STRUCTURE: str = "prose"
DEFAULT_CONTEXT: str = "general"

MAX_INPUT_TOKENS: int = 1000
"""Card: inputs up to ~1,000 tokens; chunk longer transcripts first."""

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS_INPUT_FACTOR: float = 1.3
MAX_NEW_TOKENS_FIXED: int = 32
"""Card: ``1.3 x input_tokens + 32`` is a safe greedy ceiling."""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
GOLDEN_DIR: Path = REPO_ROOT / "golden" / "s1"
"""Directory where S1-mini reference artefacts are persisted (gitignored)."""


def control_line(
    styling: str = DEFAULT_STYLING,
    structure: str = DEFAULT_STRUCTURE,
    context: str = DEFAULT_CONTEXT,
) -> str:
    """Build the trained control line. Values are validated, not invented."""
    if styling not in STYLING_VALUES:
        raise ValueError(f"styling must be one of {STYLING_VALUES}, got {styling!r}")
    if structure not in STRUCTURE_VALUES:
        raise ValueError(f"structure must be one of {STRUCTURE_VALUES}, got {structure!r}")
    if context not in CONTEXT_VALUES:
        raise ValueError(f"context must be one of {CONTEXT_VALUES}, got {context!r}")
    return f"[Styling: {styling}] [Structure: {structure}] [Context: {context}]"


def max_new_tokens_for(n_input_tokens: int) -> int:
    """Greedy budget per the card: ``1.3 x input + 32``."""
    return int(n_input_tokens * MAX_NEW_TOKENS_INPUT_FACTOR) + MAX_NEW_TOKENS_FIXED
