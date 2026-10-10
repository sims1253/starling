"""Insertion-boundary oracle — the executable contract behind
``packages/contracts/insertion-boundary`` (the README there is
authoritative).

Stdlib only, like every oracle here. ``tests/test_insertion_boundary.py``
replays the fixture table through ``adjust``; the Rust port
(``starling-processing``'s ``boundary`` module) replays the same file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CASES_PATH = (
    Path(__file__).parent.parent
    / "packages"
    / "contracts"
    / "insertion-boundary"
    / "fixtures"
    / "boundary-cases.json"
)

# After these, dictated content is starting (no space, case kept). Straight
# `"` and `'` are ambiguous at an insertion point; the contract pins them as
# opening.
OPENING = set("([{\"'“‘„«「『【（")

# Punctuation that continues a sentence (besides alphanumerics), incl. the
# Arabic comma and semicolon. Sentence enders are deliberately absent: the
# case is kept after them.
CONTINUING = set(",;:)]}”’»」』】）》،؛")

# Han, kana and CJK punctuation: no space between two of them. Hangul is
# absent on purpose (Korean separates words with spaces).
NO_SPACE_SCRIPTS = (
    ("\u3000", "\u30ff"),  # CJK symbols and punctuation, Hiragana, Katakana
    ("\u31f0", "\u31ff"),  # Katakana phonetic extensions
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\uf900", "\ufaff"),  # CJK Compatibility Ideographs
    ("\uff01", "\uff0f"),  # fullwidth punctuation
    ("\uff1a", "\uff20"),  # fullwidth punctuation
    ("\uff5b", "\uff9f"),  # fullwidth punctuation, halfwidth Katakana
    ("\U00020000", "\U0003ffff"),  # CJK Extensions B and later
)


def _no_space_boundary(ch: str) -> bool:
    return any(low <= ch <= high for low, high in NO_SPACE_SCRIPTS)


def _is_cased(ch: str) -> bool:
    return ch != ch.lower() or ch != ch.upper()


def _continues_sentence(before: str) -> bool:
    trimmed = before.rstrip()
    if not trimmed:
        return False
    trailing = before[len(trimmed) :]
    last = trimmed[-1]
    return "\n" not in trailing and "\r" not in trailing and (last.isalnum() or last in CONTINUING)


def _protected_token(raw: str) -> bool:
    parts = raw.split()
    if not parts:
        return False
    token = parts[0]
    if any(c in token for c in "_/\\@#") or token.startswith("www."):
        return True
    if token[0] == "I" and not token[1:2].isalnum():
        return True  # the pronoun, also before punctuation or an apostrophe
    if any(a.islower() and b.isupper() for a, b in zip(token, token[1:])):
        return True  # camel hump
    letters = [c for c in token if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)  # ALL-CAPS


def adjust(
    before: str, raw: str, verbatim: bool, showing_hint: bool = False
) -> tuple[str, list[str]]:
    """The adjusted text and the kinds of the rules that fired, in order.
    ``showing_hint``: the field shows only its placeholder, so ``before`` is
    hint text and the field is treated as empty."""
    if verbatim or not raw:
        return raw, []
    if showing_hint:
        before = ""
    text, changes = raw, []

    if (
        before
        and not raw[0].isspace()
        and not before[-1].isspace()
        and before[-1] not in OPENING
        and not (_no_space_boundary(before[-1]) and _no_space_boundary(raw[0]))
    ):
        text = " " + text
        changes.append("leading_space")

    if _continues_sentence(before) and not _protected_token(raw):
        # Only the first cased character is considered: if it is already
        # lowercase, later capitals are left alone.
        i = next((i for i, ch in enumerate(text) if _is_cased(ch)), None)
        if i is not None and text[i].lower() != text[i]:
            text = text[:i] + text[i].lower() + text[i + 1 :]
            changes.append("first_letter_case")

    return text, changes


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))
