"""Contract tests for the Starling multilingual text contracts (E24).

Proves five things against packages/contracts/multilingual/:

(a) both fixture manifests validate against language.schema.json and the
    provenance is frozen to synthetic;
(b) code-switching segment spans are well-formed: exactly two integer
    code-point offsets in range, non-empty, sorted, non-overlapping, and
    partitioning the utterance with no gaps (cover from 0 to the full
    code-point length, astral characters counted as one code point, not
    two UTF-16 units);
(c) round-trip: concatenating the per-segment texts reconstructs the
    utterance exactly, and each text equals its code-point slice;
(d) the digit/punctuation/casing locale expectations are exact transforms
    under explicitly frozen rules (number separators/grouping incl. the
    Indian 3-2-2 scheme and Persian digits, quotation-mark code points,
    Turkish casing per Unicode SpecialCasing tr/az - demonstrably NOT
    what locale-independent str.upper/lower/casefold do);
(e) the contract declares that ASR quality on these fixtures is a future
    model-gated gap, and its closed vocabulary cannot express a quality
    claim.

The oracle lives in tests/multilingual_contracts.py (stdlib only).
"""

from __future__ import annotations

import copy
import json
import unicodedata
from pathlib import Path

import pytest

import multilingual_contracts as ml

REPO = Path(__file__).resolve().parents[1]
SCHEMA = ml.load_schema()
CODE_SWITCHING = ml.load_manifest("code_switching.json")
LOCALE_TEXT = ml.load_manifest("locale_text.json")
MANIFESTS = [CODE_SWITCHING, LOCALE_TEXT]
TEXT_FIXTURES = {f["id"]: f for f in ml.text_fixtures(CODE_SWITCHING)}
NUMBER_FIXTURES = {f["id"]: f for f in LOCALE_TEXT["fixtures"]
                   if f["kind"] == "number-locale"}
QUOTE_FIXTURES = {f["id"]: f for f in LOCALE_TEXT["fixtures"]
                  if f["kind"] == "quotation-locale"}
CASING_FIXTURES = {f["id"]: f for f in LOCALE_TEXT["fixtures"]
                   if f["kind"] == "casing-locale"}
ALL_FIXTURES = [f for m in MANIFESTS for f in m["fixtures"]]
LANGUAGES = ml.language_index(MANIFESTS)


# --------------------------------------------------------------------------- #
# (a) schema validation
# --------------------------------------------------------------------------- #
def test_code_switching_manifest_validates_against_schema() -> None:
    ml.assert_valid(CODE_SWITCHING, SCHEMA)


def test_locale_text_manifest_validates_against_schema() -> None:
    ml.assert_valid(LOCALE_TEXT, SCHEMA)


def test_all_six_fixture_kinds_present() -> None:
    assert {f["kind"] for f in ALL_FIXTURES} == {
        "code-switching", "script-mixing", "rtl-in-ltr",
        "number-locale", "quotation-locale", "casing-locale",
    }


def test_provenance_is_synthetic_only() -> None:
    assert all(f["provenance"] == "synthetic" for f in ALL_FIXTURES)


def test_span_encoding_is_frozen_to_codepoints() -> None:
    # Same convention as mode-routing decision.schema.json.
    for manifest in MANIFESTS:
        assert manifest["span_encoding"] == "unicode_codepoints"


def test_language_tags_are_canonical_bcp47() -> None:
    for tag, language in LANGUAGES.items():
        assert tag == ml.canonical_tag(tag), f"non-canonical tag {tag!r}"
        assert language["script"] == language["script"].capitalize()
        assert language["script"].isalpha() and len(language["script"]) == 4


def test_language_descriptor_fields_are_coherent() -> None:
    # Direction and script must match the frozen reality of each language.
    for tag, language in LANGUAGES.items():
        if language["direction"] == "rtl":
            assert language["script"] in {"Arab", "Hebr"}, tag
        if language["script"] == "Latn":
            assert language["direction"] == "ltr", tag


# --------------------------------------------------------------------------- #
# (b) segment-span well-formedness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_id", sorted(TEXT_FIXTURES))
def test_segment_spans_partition_the_utterance(fixture_id: str) -> None:
    problems = ml.segment_problems(TEXT_FIXTURES[fixture_id])
    assert problems == [], problems


@pytest.mark.parametrize("fixture_id", sorted(TEXT_FIXTURES))
def test_utterances_are_nfc_normalized(fixture_id: str) -> None:
    # Frozen text is stored NFC so spans, diffs and selection ranges stay
    # stable under editing. (NFC closure under concatenation is NOT assumed
    # in general; each stored string is checked individually.)
    fixture = TEXT_FIXTURES[fixture_id]
    assert unicodedata.normalize("NFC", fixture["utterance"]) == fixture["utterance"]
    for seg in fixture["segments"]:
        assert unicodedata.normalize("NFC", seg["text"]) == seg["text"]


def test_oracle_catches_overlapping_spans() -> None:
    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    end = broken["segments"][1]["span"][1]
    broken["segments"][0]["span"] = [0, end - 2]
    assert any("overlap" in p for p in ml.segment_problems(broken))


def test_oracle_catches_unsorted_spans() -> None:
    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    broken["segments"].reverse()
    problems = ml.segment_problems(broken)
    assert any("not sorted" in p for p in problems)
    assert any("overlap" in p for p in problems)


def test_oracle_catches_uncovered_utterance() -> None:
    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    broken["segments"][0]["span"] = [2, 33]  # head not covered
    problems = ml.segment_problems(broken)
    assert any("first span must start at 0" in p for p in problems)

    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    broken["segments"][0]["span"] = [0, 31]  # interior gap 31..33
    problems = ml.segment_problems(broken)
    assert any("gap" in p for p in problems)
    assert any("code-point slice" in p for p in problems)  # stale text

    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    broken["segments"][1]["span"] = [33, 70]  # tail overshoots the text
    problems = ml.segment_problems(broken)
    assert any("last span must end at" in p for p in problems)
    assert any("out of range" in p for p in problems)


def test_oracle_catches_mismatched_segment_text() -> None:
    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    broken["segments"][0]["text"] = "der Server laeuft"
    assert any("text" in p and "code-point slice" in p
               for p in ml.segment_problems(broken))


def test_oracle_catches_empty_and_undeclared_segments() -> None:
    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    broken["segments"].insert(
        1, {"language": "fr", "span": [33, 33], "text": ""})
    problems = ml.segment_problems(broken)
    assert any("empty span" in p for p in problems)
    assert any("not declared" in p for p in problems)


def test_oracle_catches_utf16_indexed_spans() -> None:
    # Re-index the astral fixture with UTF-16 code-unit offsets (the exact
    # mistake the span_encoding const forbids): every span after the rocket
    # shifts by one and the partition check must fail.
    fixture = TEXT_FIXTURES["ml-astral-01"]
    utterance = fixture["utterance"]
    utf16_len = len(utterance.encode("utf-16-le")) // 2
    assert utf16_len == len(utterance) + 1  # the rocket costs one extra unit
    broken = copy.deepcopy(fixture)
    for seg in broken["segments"]:
        seg["span"] = [
            len(utterance[:seg["span"][0]].encode("utf-16-le")) // 2,
            len(utterance[:seg["span"][1]].encode("utf-16-le")) // 2,
        ]
    problems = ml.segment_problems(broken)
    assert problems and any(
        "code-point slice" in p or "last span must end at" in p for p in problems
    )


def test_schema_span_shape_requires_two_elements() -> None:
    # minischema cannot express maxItems; the oracle enforces exactly 2.
    broken = copy.deepcopy(TEXT_FIXTURES["ml-en-de-01"])
    broken["segments"][0]["span"] = [0, 33, 99]
    assert any("exactly 2 elements" in p for p in ml.segment_problems(broken))


# --------------------------------------------------------------------------- #
# (c) round-trip reconstruction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_id", sorted(TEXT_FIXTURES))
def test_reconstruction_is_the_identity(fixture_id: str) -> None:
    fixture = TEXT_FIXTURES[fixture_id]
    assert fixture["expectations"]["reconstruction"] == "identity"
    assert ml.reconstruct(fixture["segments"]) == fixture["utterance"]


def test_astral_codepoints_are_declared_and_single_codepoints() -> None:
    fixture = TEXT_FIXTURES["ml-astral-01"]
    for char in fixture["expectations"]["astral_codepoints"]:
        assert len(char) == 1, "an astral character is one code point"
        assert ord(char) > 0xFFFF, f"{char!r} is not astral"
        assert char in fixture["utterance"]
    # The partition above already proves spans count it once, not twice.


def test_rtl_segment_keeps_logical_codepoint_order() -> None:
    # Bidi is a display concern: the Hebrew run is stored as ordinary
    # logical-order code points and its span indexes them directly.
    fixture = TEXT_FIXTURES["ml-rtl-in-ltr-01"]
    hebrew = [s for s in fixture["segments"] if s["language"] == "he"]
    assert len(hebrew) == 1
    seg = hebrew[0]
    start, end = seg["span"]
    for ch in seg["text"]:
        assert unicodedata.name(ch).startswith("HEBREW"), repr(ch)
    assert fixture["utterance"][start:end] == seg["text"]


@pytest.mark.parametrize("fixture_id", sorted(TEXT_FIXTURES))
def test_segment_scripts_match_declared_language(fixture_id: str) -> None:
    fixture = TEXT_FIXTURES[fixture_id]
    for seg in fixture["segments"]:
        script = LANGUAGES[seg["language"]]["script"]
        hint = ml.SCRIPT_NAME_HINTS.get(script)
        if hint is None:  # e.g. Zyyy (undetermined) has no letters to check
            continue
        assert any(hint in unicodedata.name(ch, "") for ch in seg["text"]), (
            f"{fixture_id}: segment {seg['language']!r} has no {script} letters"
        )


def test_rtl_languages_only_in_rtl_fixtures() -> None:
    for fixture in ml.text_fixtures(CODE_SWITCHING):
        rtl_used = any(
            LANGUAGES[seg["language"]]["direction"] == "rtl"
            for seg in fixture["segments"]
        )
        if rtl_used:
            assert fixture["kind"] == "rtl-in-ltr", fixture["id"]


def test_fidelity_corpus_languages_have_descriptors_here() -> None:
    # Cross-contract tie-in: every language code used by the (settled)
    # fidelity corpus must have a descriptor in this package.
    corpus = json.loads(
        (REPO / "packages" / "contracts" / "fidelity-corpus" / "corpus.json")
        .read_text(encoding="utf-8")
    )
    corpus_codes = {code for f in corpus["fixtures"] for code in f["languages"]}
    declared_primaries = {tag.split("-")[0] for tag in LANGUAGES}
    assert corpus_codes <= declared_primaries | set(LANGUAGES)


# --------------------------------------------------------------------------- #
# (d) frozen locale rules as exact transforms
# --------------------------------------------------------------------------- #
def _number_case_params():
    return [
        pytest.param(fixture, case, id=f"{fixture['id']}->{case['locale']}")
        for fixture in NUMBER_FIXTURES.values()
        for case in fixture["cases"]
    ]


@pytest.mark.parametrize("fixture,case", _number_case_params())
def test_number_transforms_match_frozen_rules(fixture: dict, case: dict) -> None:
    locale = case["locale"]
    assert case["rule"] == ml.NUMBER_RULES[locale]["rule_id"]
    assert ml.convert_number(fixture["input"], fixture["source_locale"], locale) \
        == case["expected"]
    # Rules are invertible: converting back restores the canonical input.
    assert ml.convert_number(case["expected"], locale, fixture["source_locale"]) \
        == fixture["input"]


def test_number_grouping_schemes_differ_exactly() -> None:
    assert ml.convert_number("12,345,678.9", "en-US", "en-IN") == "1,23,45,678.9"
    assert ml.convert_number("12,345,678.9", "en-US", "de-DE") == "12.345.678,9"
    assert ml.convert_number("1,234.56", "en-US", "fa-IR") == "\u06f1\u066c\u06f2\u06f3\u06f4\u066b\u06f5\u06f6"


def test_parse_number_rejects_mixed_locale_separators() -> None:
    # en-US rule must reject the de-DE rendering and vice versa: there is
    # no silent English formatting assumption.
    with pytest.raises(ValueError):
        ml.parse_number("1.234,56", "en-US")
    with pytest.raises(ValueError):
        ml.parse_number("1,234.56", "de-DE")
    with pytest.raises(ValueError):
        ml.parse_number("1,23,45,678.9", "en-US")  # en-IN grouping elsewhere
    with pytest.raises(ValueError):
        ml.parse_number("12.345.678,9", "en-IN")


def test_quotation_marks_match_frozen_codepoints() -> None:
    for fixture in QUOTE_FIXTURES.values():
        for case in fixture["cases"]:
            locale = case["locale"]
            assert case["rule"] == ml.QUOTE_RULES[locale]["rule_id"]
            assert ml.quote(fixture["text"], locale) == case["quoted"]


def test_quotation_codepoints_are_frozen_by_escape() -> None:
    # Pin the exact code points independent of file display: en U+201C/U+201D,
    # de U+201E/U+201C, fr U+00AB/U+00BB + U+00A0 inner, ru U+00AB/U+00BB,
    # ja U+300C/U+300D.
    assert ml.quote("x", "en-US") == "\u201cx\u201d"
    assert ml.quote("x", "de-DE") == "\u201ex\u201c"
    assert ml.quote("x", "fr-FR") == "\u00ab\u00a0x\u00a0\u00bb"
    assert ml.quote("x", "ru-RU") == "\u00abx\u00bb"
    assert ml.quote("x", "ja-JP") == "\u300cx\u300d"


@pytest.mark.parametrize("fixture_id", sorted(CASING_FIXTURES))
def test_turkish_casing_matches_special_casing(fixture_id: str) -> None:
    fixture = CASING_FIXTURES[fixture_id]
    assert fixture["locale"] == "tr-TR"
    assert fixture["rule"] == ml.CASING_RULES["tr-TR"]
    for case in fixture["cases"]:
        assert ml.apply_casing(
            case["input"], fixture["locale"], case["operation"]
        ) == case["expected"]


def test_default_unicode_casing_is_not_turkish() -> None:
    # Documented rule: Python's str.upper/lower/casefold are the
    # locale-independent DEFAULT Unicode case conversions. For Turkish you
    # need SpecialCasing.txt's tr/az mappings, implemented as an explicit
    # map in the oracle. These assertions freeze the difference:
    assert "i".upper() == "I" != "İ"            # default loses the dot
    assert "ı".upper() == "I"
    assert "I".lower() == "i" != "ı"            # default gains a dot
    assert "İ".casefold() == "i\u0307"          # i + COMBINING DOT ABOVE,
    assert len("İ".casefold()) == 2             # NOT the tr plain "i"
    assert "I".casefold() == "i" != "ı"
    # The oracle implements the tr/az mappings instead:
    assert ml.turkish_upper("i") == "\u0130"
    assert ml.turkish_upper("ı") == "I"
    assert ml.turkish_lower("I") == "\u0131"
    assert ml.turkish_lower("İ") == "i"         # plain, no combining dot


# --------------------------------------------------------------------------- #
# (e) ASR quality is a declared future gap, not implied here
# --------------------------------------------------------------------------- #
def test_asr_quality_gap_is_declared_verbatim() -> None:
    # Whitespace-normalized compare so README line wrapping is free to
    # change; the sentence itself is frozen in ml.ASR_QUALITY_GAP.
    readme = " ".join(ml.README_PATH.read_text(encoding="utf-8").split())
    assert ml.ASR_QUALITY_GAP in readme


def test_contract_vocabulary_cannot_express_quality_claims() -> None:
    # additionalProperties is false everywhere, so the frozen vocabulary is
    # exactly the whitelisted field set; double-check no key names a
    # quality metric in any fixture.
    forbidden = {"wer", "accuracy", "precision", "recall", "score", "quality"}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key.lower() not in forbidden, key
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for manifest in MANIFESTS:
        walk(manifest)
