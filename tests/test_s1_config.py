"""S1-mini CPU-only contract tests: control line, chunking, budget formula.

No CUDA, no model, no goldens — this module runs in the CPU CI matrix
(tests/test_s1_pipeline.py guards the GPU + golden gates separately).

Run with:  uv run pytest tests/test_s1_config.py -q
"""

from __future__ import annotations

import pytest


def test_control_line_validates_values() -> None:
    from starling.s1.config import control_line

    assert control_line() == (
        "[Styling: semi-formal] [Structure: prose] [Context: general]")
    assert control_line("casual", "lists", "email") == (
        "[Styling: casual] [Structure: lists] [Context: email]")
    with pytest.raises(ValueError):
        control_line(styling="pirate")
    with pytest.raises(ValueError):
        control_line(structure="table")
    with pytest.raises(ValueError):
        control_line(context="meeting")


def test_chunk_transcript_short_input_passthrough() -> None:
    from starling.s1.pipeline import chunk_transcript

    t = "um so like one two three"
    assert chunk_transcript(t, max_words=600) == [t]
    # no-punctuation run longer than the budget splits at word boundaries
    long_t = " ".join(["word"] * 10)
    parts = chunk_transcript(long_t, max_words=4)
    assert parts == ["word word word word", "word word word word", "word word"]
    # punctuated text splits at sentence boundaries when they fit the budget
    assert chunk_transcript("a b. c d. e f.", max_words=2) == ["a b.", "c d.", "e f."]
    # ... and falls back to word windows when a sentence exceeds the budget
    assert chunk_transcript("a b. c d.", max_words=1) == ["a", "b.", "c", "d."]


def test_max_new_tokens_formula() -> None:
    from starling.s1.config import max_new_tokens_for

    assert max_new_tokens_for(100) == 162  # 1.3*100 + 32
