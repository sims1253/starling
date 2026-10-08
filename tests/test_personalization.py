"""Test suite for personalization contract and oracle (#305).

Replays shared contract fixtures, tests schema conformance, and asserts
acceptance invariants for training-free personalization.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

import minischema
import personalization

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "personalization"
SCHEMA_PATH = CONTRACT / "personalization.schema.json"


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def assert_valid(instance, subschema, root=None):
    errs = list(minischema.errors(instance, subschema, root=root))
    assert not errs, f"Validation errors: {errs}"


def test_schema_itself_is_valid(schema):
    assert schema["$id"] == "urn:starling:contracts:personalization:v1"
    assert "suggestion" in schema["$defs"]
    assert "retrievalRequest" in schema["$defs"]
    assert "retrievalResult" in schema["$defs"]


def test_suggestion_fixtures(schema):
    fixture_path = CONTRACT / "fixtures" / "suggestions.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in data["cases"]:
        case_id = case["id"]
        records = case["records"]
        existing_vocab = case.get("existing_vocabulary", [])
        existing_snippets = case.get("existing_snippets", [])
        min_freq = case.get("min_frequency", 2)
        min_cons = case.get("min_consistency", 0.8)

        actual = personalization.suggest_vocabulary_and_replacements(
            records=records,
            existing_vocabulary=existing_vocab,
            existing_snippets=existing_snippets,
            min_frequency=min_freq,
            min_consistency=min_cons,
        )

        expected = case["expected_suggestions"]
        assert len(actual) == len(expected), f"Case {case_id}: count mismatch"

        for act, exp in zip(actual, expected):
            # Validate against suggestion schema
            assert_valid(act, schema["$defs"]["suggestion"], root=schema)

            assert act["target_type"] == exp["target_type"], f"Case {case_id}: type mismatch"
            assert act["source_phrase"] == exp["source_phrase"], f"Case {case_id}: source mismatch"
            assert act["target_phrase"] == exp["target_phrase"], f"Case {case_id}: target mismatch"
            assert act["frequency"] == exp["frequency"], f"Case {case_id}: frequency mismatch"
            assert act["consistency"] == exp["consistency"], f"Case {case_id}: consistency mismatch"
            assert act["status"] == exp["status"], f"Case {case_id}: status mismatch"
            assert act["conflict"] == exp["conflict"], f"Case {case_id}: conflict mismatch"


def test_retrieval_fixtures(schema):
    fixture_path = CONTRACT / "fixtures" / "retrieval.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in data["cases"]:
        case_id = case["id"]
        history = case["history"]
        deleted = case.get("deleted_captures", [])
        request = case["request"]

        # Validate request schema
        assert_valid(request, schema["$defs"]["retrievalRequest"], root=schema)

        actual = personalization.retrieve_style_examples(
            history=history,
            request=request,
            deleted_captures=deleted,
        )

        # Validate result schema
        assert_valid(actual, schema["$defs"]["retrievalResult"], root=schema)

        expected = case["expected_result"]
        assert actual["examples_used"] == expected["examples_used"], f"Case {case_id}: examples_used mismatch"
        assert actual["character_count"] == expected["character_count"], f"Case {case_id}: character_count mismatch"
        assert actual["formatted_context"] == expected["formatted_context"], f"Case {case_id}: formatted_context mismatch"


def test_evaluation_baseline_reduction():
    fixture_path = CONTRACT / "fixtures" / "evaluation.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))

    for session in data["sessions"]:
        takes = session["takes"]
        suggestions = session.get("active_suggestions", [])

        total_unpersonalized_burden = 0
        total_personalized_burden = 0

        for take in takes:
            raw = take["raw"]
            ground_truth = take["ground_truth_target"]

            # Unpersonalized burden (edit distance from raw/unpersonalized to target)
            unpersonalized = take["unpersonalized_output"]
            burden_unpersonalized = personalization.levenshtein(unpersonalized, ground_truth)
            total_unpersonalized_burden += burden_unpersonalized

            # Personalized output with suggestions applied
            personalized_text = personalization.apply_suggestions_to_text(raw, suggestions)
            burden_personalized = personalization.levenshtein(personalized_text, ground_truth)
            total_personalized_burden += burden_personalized

            # Check protected spans are never modified
            for protected in take.get("protected_spans", []):
                assert protected in personalized_text, f"Protected span {protected} was modified"

        # Assert correction burden strictly drops when suggestions are applicable
        if suggestions:
            assert total_personalized_burden < total_unpersonalized_burden


def test_isolated_project_and_language():
    history = [
        {
            "id": "rec_proj_1",
            "capture_id": "c1",
            "request_id": "r1",
            "raw_text": "project 1 text",
            "final_text": "Project 1 text.",
            "decision": "accepted",
            "decision_utc": "2026-10-01T10:00:00Z",
            "mode_id": "clean",
            "language": "en",
            "project_id": "alpha",
            "secure_field": False,
        },
        {
            "id": "rec_proj_2",
            "capture_id": "c2",
            "request_id": "r2",
            "raw_text": "project 2 text",
            "final_text": "Project 2 text.",
            "decision": "accepted",
            "decision_utc": "2026-10-01T11:00:00Z",
            "mode_id": "clean",
            "language": "en",
            "project_id": "beta",
            "secure_field": False,
        },
    ]

    # Query for alpha
    res_alpha = personalization.retrieve_style_examples(
        history=history,
        request={"mode_id": "clean", "language": "en", "project_id": "alpha"},
    )
    assert res_alpha["examples_used"] == ["rec_proj_1"]
    assert "project 2" not in res_alpha["formatted_context"]

    # Query for beta
    res_beta = personalization.retrieve_style_examples(
        history=history,
        request={"mode_id": "clean", "language": "en", "project_id": "beta"},
    )
    assert res_beta["examples_used"] == ["rec_proj_2"]
    assert "project 1" not in res_beta["formatted_context"]


def test_immediate_tombstone_deletion():
    history = [
        {
            "id": "rec_del",
            "capture_id": "cap_to_delete",
            "request_id": "r_del",
            "raw_text": "deleted phrase",
            "final_text": "Deleted phrase.",
            "decision": "accepted",
            "decision_utc": "2026-10-01T10:00:00Z",
            "mode_id": "clean",
            "language": "en",
            "project_id": None,
            "secure_field": False,
        }
    ]

    # Without tombstone: retrieved
    before = personalization.retrieve_style_examples(
        history=history,
        request={"mode_id": "clean", "language": "en"},
        deleted_captures=[],
    )
    assert before["examples_used"] == ["rec_del"]

    # With tombstone: immediately omitted
    after = personalization.retrieve_style_examples(
        history=history,
        request={"mode_id": "clean", "language": "en"},
        deleted_captures=["cap_to_delete"],
    )
    assert after["examples_used"] == []
    assert after["formatted_context"] == ""
