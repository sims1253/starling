"""Executable stdlib oracle for the Starling multilingual text contract (E24).

Semantic half of the frozen v1 contract in
``packages/contracts/multilingual/`` (``language.schema.json`` is the
structural half). Deliberately stdlib-only so any implementation - the
native runtime, the desktop client, the s1 normalizer or a third-party
tool - can be differentially tested against it.

Provides:

* fixture loading + schema validation (prefers the third-party
  ``jsonschema`` package when installed, falls back to ``minischema.py``,
  exactly like ``tests/test_fidelity_corpus.py``);
* ``segment_problems`` - code-switching span well-formedness: exactly two
  integer offsets, ``0 <= start <= end <= code-point length``, non-empty
  segments, spans sorted, non-overlapping, partitioning the utterance with
  no gaps, per-segment ``text`` equal to the code-point slice, and every
  segment language declared in the fixture's language descriptors;
* ``reconstruct`` - concatenating segment texts must be the identity on
  the utterance;
* frozen locale rules with their Unicode/CLDR basis (below).

Frozen locale rules and their basis
-----------------------------------
Numbers (separator and grouping conventions frozen per locale, v1 covers
non-negative plain decimals in canonical form only):

* ``en-US``  - digits ``0-9``, thousands ``,`` decimal ``.``, groups of 3.
* ``de-DE``  - digits ``0-9``, thousands ``.`` decimal ``,``, groups of 3
  (``1.234,56``).
* ``en-IN``  - digits ``0-9``, thousands ``,`` decimal ``.``, rightmost
  group of 3 then groups of 2 (``1,23,45,678.9``).
* ``fa-IR``  - extended Arabic-Indic digits U+06F0..U+06F9, thousands
  U+066C ARABIC THOUSANDS SEPARATOR, decimal U+066B ARABIC DECIMAL
  SEPARATOR (``۱٬۲۳۴٫۵۶``).

Quotation marks (exact code points frozen per locale):

* ``en-US`` ``“…”``  U+201C / U+201D
* ``de-DE`` ``„…“``  U+201E / U+201C
* ``fr-FR`` ``«\u00a0…\u00a0»``  U+00AB / U+00BB with U+00A0 NO-BREAK
  SPACE on both inner sides
* ``ru-RU`` ``«…»``  U+00AB / U+00BB, no inner space
* ``ja-JP`` ``「…」``  U+300C / U+300D

Casing (``tr-TR``, per Unicode ``SpecialCasing.txt`` "Turkic mappings",
which apply to Turkish and Azerbaijani):

* upper: ``i`` -> ``İ`` (U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE),
  ``ı`` -> ``I``;
* lower: ``I`` -> ``ı`` (U+0131 LATIN SMALL LETTER DOTLESS I), ``İ`` ->
  plain ``i`` (U+0069), overriding the unconditional default
  ``İ`` -> ``i`` + U+0307 COMBINING DOT ABOVE.

Python's ``str.upper`` / ``str.lower`` / ``str.casefold`` implement the
locale-independent default Unicode case conversions (UnicodeData.txt plus
unconditional SpecialCasing entries) and therefore cannot produce the
Turkish results - the oracle implements them as an explicit per-code-point
mapping. This is the documented rule; ``str.casefold`` is NOT a Turkish
caser.

ASR quality on these fixtures is a future model-gated gap; the frozen
declaration ``ASR_QUALITY_GAP`` must appear verbatim in the package
README.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "multilingual"
FIXTURE_DIR = CONTRACT / "fixtures"
SCHEMA_PATH = CONTRACT / "language.schema.json"
README_PATH = CONTRACT / "README.md"

#: Frozen gap declaration; must appear verbatim in the package README.
ASR_QUALITY_GAP = (
    "ASR recognition quality on these fixtures is a future model-gated gap "
    "and is not implied by these contracts."
)

try:
    import jsonschema  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

import minischema


# --------------------------------------------------------------------------- #
# Loading + schema validation
# --------------------------------------------------------------------------- #
def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_manifest(filename: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))


def assert_valid(instance: Any, schema: dict[str, Any]) -> None:
    """Validate against jsonschema when available, minischema otherwise."""
    if jsonschema is not None:
        jsonschema.Draft202012Validator(schema).validate(instance)
    else:
        problems = minischema.errors(instance, schema)
        assert not problems, problems


def text_fixtures(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    kinds = {"code-switching", "script-mixing", "rtl-in-ltr"}
    return [f for f in manifest["fixtures"] if f["kind"] in kinds]


def language_index(manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for fixture in manifest["fixtures"]:
            for language in fixture["languages"]:
                index.setdefault(language["tag"], language)
    return index


# --------------------------------------------------------------------------- #
# Language tags
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


def canonical_tag(tag: str) -> str:
    """Canonical BCP-47 casing: lowercase primary subtag, Titlecase a
    4-letter script subtag, uppercase a 2-letter region subtag, lowercase
    anything else."""
    parts = tag.split("-")
    out = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            out.append(part.capitalize())
        elif len(part) == 2 and part.isalpha():
            out.append(part.upper())
        else:
            out.append(part.lower())
    return "-".join(out)


# --------------------------------------------------------------------------- #
# Code-switching segment well-formedness and round-trip
# --------------------------------------------------------------------------- #
def segment_problems(fixture: dict[str, Any]) -> list[str]:
    """All violations of the partition contract for one text fixture.

    A well-formed fixture produces ``[]``. Spans are half-open
    ``[start, end)`` code-point offsets (``span_encoding`` is fixed to
    ``unicode_codepoints`` by the schema, matching mode-routing).
    """
    problems: list[str] = []
    utterance = fixture["utterance"]
    declared = {lang["tag"] for lang in fixture["languages"]}
    segments = fixture["segments"]
    # Only well-formed integer pairs enter the ordering/cover checks.
    spans: list[list[int]] = []

    for i, seg in enumerate(segments):
        span = seg["span"]
        label = f"segment {i} ({seg['language']})"
        if len(span) != 2:
            problems.append(f"{label}: span must have exactly 2 elements, got {len(span)}")
            continue
        start, end = span
        if not (isinstance(start, int) and isinstance(end, int)
                and not isinstance(start, bool) and not isinstance(end, bool)):
            problems.append(f"{label}: span offsets must be integers")
            continue
        if not (0 <= start <= end <= len(utterance)):
            problems.append(
                f"{label}: span [{start}, {end}] out of range for "
                f"{len(utterance)} code points"
            )
        if end == start:
            problems.append(
                f"{label}: empty span - a code-switching segment covers at "
                "least one code point"
            )
        if seg["text"] != utterance[start:end]:
            problems.append(
                f"{label}: text {seg['text']!r} != code-point slice "
                f"utterance[{start}:{end}] = {utterance[start:end]!r}"
            )
        if seg["language"] not in declared:
            problems.append(
                f"{label}: language {seg['language']!r} not declared in fixture languages"
            )
        spans.append(span)

    for i in range(len(spans) - 1):
        a_start, a_end = spans[i]
        b_start, b_end = spans[i + 1]
        if b_start < a_start:
            problems.append(f"spans not sorted: span {i + 1} starts before span {i}")
        if b_start < a_end:
            problems.append(
                f"spans overlap: span {i} [{a_start}, {a_end}] and span {i + 1} "
                f"[{b_start}, {b_end}]"
            )
        elif b_start > a_end:
            problems.append(
                f"spans leave a gap: span {i} ends at {a_end} but span {i + 1} "
                f"starts at {b_start}"
            )

    if spans:
        if spans[0][0] != 0:
            problems.append(f"cover: first span must start at 0, got {spans[0][0]}")
        if spans[-1][1] != len(utterance):
            problems.append(
                f"cover: last span must end at {len(utterance)} code points, "
                f"got {spans[-1][1]}"
            )
    return problems


def reconstruct(segments: list[dict[str, Any]]) -> str:
    """Concatenate per-segment texts; must equal the original utterance."""
    return "".join(seg["text"] for seg in segments)


#: Approximate script check via Unicode character names (``unicodedata``);
#: scripts not listed here are not checked. Used only as an oracle for the
#: scripts that appear in the frozen fixtures.
SCRIPT_NAME_HINTS = {
    "Latn": "LATIN",
    "Hebr": "HEBREW",
    "Hani": "CJK",
    "Arab": "ARABIC",
    "Cyrl": "CYRILLIC",
}


# --------------------------------------------------------------------------- #
# Frozen number rules (v1: canonical non-negative plain decimals only)
# --------------------------------------------------------------------------- #
FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0..U+06F9
ASCII_DIGITS = "0123456789"
FA_TO_ASCII = str.maketrans(FA_DIGITS, ASCII_DIGITS)
ASCII_TO_FA = str.maketrans(ASCII_DIGITS, FA_DIGITS)

NUMBER_RULES: dict[str, dict[str, Any]] = {
    "en-US": {
        "rule_id": "latn-group3-comma-decimal-dot",
        "group_sep": ",",
        "decimal_sep": ".",
        "grouping": (3,),
        "digit_map": None,
    },
    "de-DE": {
        "rule_id": "latn-group3-dot-decimal-comma",
        "group_sep": ".",
        "decimal_sep": ",",
        "grouping": (3,),
        "digit_map": None,
    },
    "en-IN": {
        "rule_id": "latn-group322-comma-decimal-dot",
        "group_sep": ",",
        "decimal_sep": ".",
        "grouping": (3, 2),  # rightmost group of 3, then groups of 2
        "digit_map": None,
    },
    "fa-IR": {
        "rule_id": "arabext-group3-thousands-decimal",
        "group_sep": "\u066c",  # ARABIC THOUSANDS SEPARATOR
        "decimal_sep": "\u066b",  # ARABIC DECIMAL SEPARATOR
        "grouping": (3,),
        "digit_map": ASCII_TO_FA,
    },
}


def _to_ascii_digits(text: str, locale: str) -> str:
    """Map locale digits to ASCII digits (identity for Latin-digit locales).

    v1 has exactly one non-ASCII digit system (extended Arabic-Indic, used
    by fa-IR), so the reverse map is the global ``FA_TO_ASCII`` table."""
    if NUMBER_RULES[locale]["digit_map"] is None:
        return text
    return text.translate(FA_TO_ASCII)


def _expected_group_sizes(grouping: tuple[int, ...], group_count: int) -> list[int]:
    """Group sizes right-to-left: first the primary size, then the secondary
    size (== primary when the scheme has one size). The leftmost group may
    be shorter."""
    primary, *rest = grouping
    secondary = rest[0] if rest else primary
    return [primary] + [secondary] * (group_count - 1)


def parse_number(text: str, locale: str) -> Decimal:
    """Strictly parse a canonically formatted number in ``locale``.

    Raises ``ValueError`` on any deviation from the frozen rule: wrong
    separators, missing canonical grouping (non-leftmost groups must be
    exactly their scheme size), empty integer part, stray characters,
    multiple decimal separators.
    """
    rule = NUMBER_RULES[locale]
    group_sep, decimal_sep = rule["group_sep"], rule["decimal_sep"]
    if group_sep == decimal_sep:
        raise ValueError("rule separators must differ")  # pragma: no cover
    if text.count(decimal_sep) > 1:
        raise ValueError(f"{text!r}: more than one decimal separator {decimal_sep!r}")
    int_part, _, frac_part = text.partition(decimal_sep)
    if not int_part:
        raise ValueError(f"{text!r}: missing integer part before {decimal_sep!r}")
    if decimal_sep in text and not _to_ascii_digits(frac_part, locale).isdigit():
        raise ValueError(f"{text!r}: fractional part must be digits only")
    if group_sep in frac_part:
        raise ValueError(f"{text!r}: group separator in fractional part")

    raw_groups = int_part.split(group_sep)
    digits = _to_ascii_digits("".join(raw_groups), locale)
    if not digits.isdigit():
        raise ValueError(f"{text!r}: non-digit characters in integer part")

    # Canonical grouping, checked right-to-left: every non-leftmost group
    # is exactly its scheme size; the leftmost may be shorter (1..size).
    expected = _expected_group_sizes(rule["grouping"], len(raw_groups))
    for i, (group, size) in enumerate(zip(reversed(raw_groups), expected)):
        if i == len(raw_groups) - 1:  # leftmost
            ok = 1 <= len(group) <= size
        else:
            ok = len(group) == size
        if not ok:
            raise ValueError(
                f"{text!r}: group {group!r} violates canonical {locale} "
                f"grouping (expected {size} digits)"
            )
    if len(raw_groups[0]) > 1 and raw_groups[0].startswith("0"):
        raise ValueError(f"{text!r}: leading zero in integer part")

    ascii_int = _to_ascii_digits(int_part.replace(group_sep, ""), locale)
    ascii_frac = _to_ascii_digits(frac_part, locale)
    return Decimal(f"{ascii_int}.{ascii_frac}") if ascii_frac else Decimal(ascii_int)


def format_number(value: Decimal, locale: str) -> str:
    """Format a Decimal per the frozen ``locale`` rule, canonically."""
    rule = NUMBER_RULES[locale]
    literal = format(value, "f")
    int_part, _, frac_part = literal.partition(".")
    if int_part.startswith("-") or frac_part.startswith("-"):  # pragma: no cover
        raise ValueError("v1 covers non-negative decimals only")

    grouping = rule["grouping"]
    primary, *rest = grouping
    secondary = rest[0] if rest else primary
    chunks: list[str] = []
    i = len(int_part)
    size = primary
    while i > 0:
        if i <= size:
            chunks.append(int_part[:i])
            i = 0
        else:
            chunks.append(int_part[i - size:i])
            i -= size
            size = secondary
    grouped = rule["group_sep"].join(reversed(chunks))
    result = grouped + (rule["decimal_sep"] + frac_part if frac_part else "")
    digit_map = rule["digit_map"]
    return result.translate(digit_map) if digit_map is not None else result


def convert_number(text: str, source_locale: str, target_locale: str) -> str:
    return format_number(parse_number(text, source_locale), target_locale)


# --------------------------------------------------------------------------- #
# Frozen quotation-mark rules
# --------------------------------------------------------------------------- #
QUOTE_RULES: dict[str, dict[str, str]] = {
    "en-US": {"rule_id": "double-curved", "open": "\u201c", "close": "\u201d", "pad": ""},
    "de-DE": {"rule_id": "low-open-high-close", "open": "\u201e", "close": "\u201c", "pad": ""},
    "fr-FR": {"rule_id": "guillemets-nbsp-inside", "open": "\u00ab", "close": "\u00bb",
              "pad": "\u00a0"},
    "ru-RU": {"rule_id": "guillemets-no-inner-space", "open": "\u00ab", "close": "\u00bb",
              "pad": ""},
    "ja-JP": {"rule_id": "corner-brackets", "open": "\u300c", "close": "\u300d", "pad": ""},
}


def quote(text: str, locale: str) -> str:
    rule = QUOTE_RULES[locale]
    return f"{rule['open']}{rule['pad']}{text}{rule['pad']}{rule['close']}"


# --------------------------------------------------------------------------- #
# Frozen casing rules (Turkish; Unicode SpecialCasing.txt, tr/az)
# --------------------------------------------------------------------------- #
CASING_RULES: dict[str, str] = {
    "tr-TR": "unicode-specialcasing-tr-az",
}

#: SpecialCasing.txt "Turkic mappings" (tr, az), upper direction:
#: U+0069 i -> U+0130 İ; U+0131 ı -> U+0049 I.
TURKISH_UPPER_MAP = {"i": "\u0130", "\u0131": "I"}
#: lower direction: U+0049 I -> U+0131 ı; U+0130 İ -> U+0069 i (plain,
#: overriding the unconditional default İ -> i + U+0307).
TURKISH_LOWER_MAP = {"I": "\u0131", "\u0130": "i"}


def turkish_upper(text: str) -> str:
    return "".join(TURKISH_UPPER_MAP.get(ch, ch.upper()) for ch in text)


def turkish_lower(text: str) -> str:
    return "".join(TURKISH_LOWER_MAP.get(ch, ch.lower()) for ch in text)


def apply_casing(text: str, locale: str, operation: str) -> str:
    if locale not in CASING_RULES:
        raise ValueError(f"no frozen casing rule for locale {locale!r}")
    if operation == "upper":
        return turkish_upper(text) if locale == "tr-TR" else text.upper()
    if operation == "lower":
        return turkish_lower(text) if locale == "tr-TR" else text.lower()
    raise ValueError(f"unknown casing operation {operation!r}")
