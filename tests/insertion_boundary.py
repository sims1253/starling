"""Insertion-boundary oracle (#341) — the executable contract behind
``packages/contracts/insertion-boundary``.

Stdlib only, like every oracle here: this module is the rules, the fixtures
are the pinned truth, and ``tests/test_insertion_boundary.py`` replays the
fixture table through it. The Rust port (``starling-processing``'s ``boundary``
module) replays the same files, so neither implementation can drift from the
contract.

The rules in full are frozen in the contract README; this docstring is a
summary, the README is authoritative:

1. ``verbatim`` disables every rule.
2. Leading space: prepend one U+0020 iff ``before`` is non-empty, its last
   character is neither whitespace nor in the opening set, and ``raw`` does
   not already start with whitespace.
3. First-letter case: lowercase the first cased character of ``raw`` iff the
   first token of ``raw`` is not protected, the trailing whitespace run of
   ``before`` contains no newline, and ``before``'s last non-whitespace
   character is mid-sentence (alphanumeric or one of the listed continuing
   punctuation characters; sentence enders and the opening set keep the
   case).
4. Trailing: never — ``after`` is carried for validation and future rules,
   but no rule touches the end of ``raw``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONTRACT = Path(__file__).parent.parent / "packages" / "contracts" / "insertion-boundary"

# Characters after which dictated content is *starting* (no space, case
# kept). Straight `"` and `'` are ambiguous at an insertion point; the
# contract pins them as opening.
OPENING = set('([{"\'“‘„「『【（')

# Mid-sentence punctuation (continues the sentence) beyond alphanumerics.
CONTINUING = set(",;:)]}”’»」』】）》")

# Sentence enders: the case is kept as recognized.
SENTENCE_END = set(".!?…。！？")

LEADING_SPACE = "leading_space"
FIRST_LETTER_CASE = "first_letter_case"


@dataclass
class BoundaryCase:
    """One fixture case, field-for-field with boundary.schema.json."""

    case_id: str
    before: str
    after: str
    raw: str
    verbatim: bool
    description: str = ""

    @classmethod
    def from_json(cls, doc: dict) -> "BoundaryCase":
        return cls(
            case_id=doc["case_id"],
            before=doc["before"],
            after=doc["after"],
            raw=doc["raw"],
            verbatim=doc["verbatim"],
            description=doc.get("description", ""),
        )


@dataclass
class BoundaryAdjustment:
    """The adjusted text plus the changes that fired, in order."""

    text: str
    changes: list[dict] = field(default_factory=list)


def _is_cased(ch: str) -> bool:
    return ch != ch.lower() or ch != ch.upper()


def _has_newline_in_trailing_whitespace(before: str) -> bool:
    i = len(before)
    while i > 0 and before[i - 1].isspace():
        i -= 1
    return any(c in "\r\n" for c in before[i:])


def _last_non_whitespace(before: str) -> str | None:
    for ch in reversed(before):
        if not ch.isspace():
            return ch
    return None


def _first_token(raw: str) -> str:
    parts = raw.split()
    return parts[0] if parts else ""


def _protected_token(token: str) -> bool:
    if not token:
        return False
    if any(c in token for c in "_/\\@#"):
        return True
    if "://" in token or token.startswith("www."):
        return True
    if any(a.islower() and b.isupper() for a, b in zip(token, token[1:])):
        return True  # camel hump: a lowercase letter immediately followed by an uppercase one
    letters = [c for c in token if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return True  # ALL-CAPS
    if token == "I":
        return True  # the English pronoun
    return False


def adjust(case: BoundaryCase) -> BoundaryAdjustment:
    """Apply the frozen rules; never more than the boundary."""
    changes: list[dict] = []
    if case.verbatim or not case.raw:
        return BoundaryAdjustment(case.raw, changes)

    text = case.raw

    # Rule 2: leading space.
    if (
        case.before
        and not text[0].isspace()
        and not case.before[-1].isspace()
        and case.before[-1] not in OPENING
    ):
        text = " " + text
        changes.append(
            {
                "kind": LEADING_SPACE,
                "detail": (
                    f"previous character {case.before[-1]!r} is not whitespace "
                    "or an opening bracket/quote"
                ),
            }
        )

    # Rule 3: first-letter case.
    last = _last_non_whitespace(case.before)
    continues = (
        last is not None
        and not _has_newline_in_trailing_whitespace(case.before)
        and (last.isalnum() or last in CONTINUING)
    )
    if continues and not _protected_token(_first_token(text)):
        chars = list(text)
        for i, ch in enumerate(chars):
            if _is_cased(ch) and ch.lower() != ch:
                chars[i] = ch.lower()
                text = "".join(chars)
                changes.append(
                    {
                        "kind": FIRST_LETTER_CASE,
                        "detail": (
                            f"previous non-whitespace {last!r} continues the sentence"
                        ),
                    }
                )
                break

    return BoundaryAdjustment(text, changes)


def load_cases() -> list[tuple[dict, BoundaryCase, dict]]:
    """(fixture doc, parsed case, expected block) for every pinned case."""
    path = CONTRACT / "fixtures" / "boundary-cases.json"
    docs = json.loads(path.read_text(encoding="utf-8"))
    return [
        (doc, BoundaryCase.from_json(doc), doc)
        for doc in docs
    ]
