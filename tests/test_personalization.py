"""Replays the personalization contract fixtures against the oracle in
``personalization.py`` and validates every payload against the schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import minischema
import personalization
from parity_contract import levenshtein

CONTRACT = Path(__file__).resolve().parents[1] / "packages" / "contracts" / "personalization"
SCHEMA = json.loads((CONTRACT / "personalization.schema.json").read_text(encoding="utf-8"))


def load(name: str) -> dict:
    return json.loads((CONTRACT / "fixtures" / name).read_text(encoding="utf-8"))


def assert_valid(instance: dict, definition: str) -> None:
    errs = list(minischema.errors(instance, SCHEMA["$defs"][definition], root=SCHEMA))
    assert not errs, errs


def test_schema_is_valid_draft_2020_12():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("case", load("suggestions.json")["cases"], ids=lambda c: c["id"])
def test_suggestion_fixture(case):
    for record in case["records"]:
        assert_valid(record, "correctionRecord")
    actual = personalization.suggest_vocabulary_and_replacements(
        records=case["records"],
        existing_vocabulary=case["existing_vocabulary"],
        existing_snippets=case["existing_snippets"],
        min_frequency=case["min_frequency"],
        min_consistency=case["min_consistency"],
    )
    for suggestion in actual:
        assert_valid(suggestion, "suggestion")
    assert actual == case["expected_suggestions"]


@pytest.mark.parametrize("case", load("retrieval.json")["cases"], ids=lambda c: c["id"])
def test_retrieval_fixture(case):
    for record in case["history"]:
        assert_valid(record, "correctionRecord")
    assert_valid(case["request"], "retrievalRequest")
    actual = personalization.retrieve_style_examples(
        history=case["history"],
        request=case["request"],
        deleted_captures=case["deleted_captures"],
    )
    assert_valid(actual, "retrievalResult")
    assert actual == case["expected_result"]


@pytest.mark.parametrize(
    "session", load("evaluation.json")["sessions"], ids=lambda s: s["session_id"]
)
def test_suggestions_reduce_correction_burden(session):
    before = after = 0
    for take in session["takes"]:
        personalized = personalization.apply_suggestions_to_text(
            take["raw"], session["active_suggestions"]
        )
        before += levenshtein(take["unpersonalized_output"], take["ground_truth_target"])
        after += levenshtein(personalized, take["ground_truth_target"])
        for span in take["protected_spans"]:
            assert span in personalized
    assert after < before


def test_apply_matches_case_folded_spellings():
    suggestions = [{"source_phrase": "izmir", "target_phrase": "Izmir"}]
    assert personalization.apply_suggestions_to_text("İzmir", suggestions) == "Izmir"
