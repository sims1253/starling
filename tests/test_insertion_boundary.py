"""Replays ``packages/contracts/insertion-boundary/fixtures/boundary-cases.json``
through the oracle in ``tests/insertion_boundary.py``. The Rust port
replays the same file (``starling-processing``'s ``boundary_conformance``).
"""

from __future__ import annotations

import json

import pytest

import insertion_boundary as oracle
import minischema

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

CASES = oracle.load_cases()
SCHEMA = json.loads(
    (oracle.CASES_PATH.parent.parent / "boundary.schema.json").read_text(encoding="utf-8")
)


def test_fixture_table_is_complete_and_unique() -> None:
    assert len(CASES) >= 30, "the acceptance fixture classes must stay pinned"
    ids = [case["case_id"] for case in CASES]
    assert len(ids) == len(set(ids)), "case ids must be unique"


@pytest.mark.parametrize("doc", CASES, ids=lambda doc: doc["case_id"])
def test_case(doc: dict) -> None:
    errors = minischema.errors(doc, SCHEMA)
    if jsonschema is not None:
        validator = jsonschema.Draft202012Validator(SCHEMA)
        errors += [f"jsonschema: {e.message}" for e in validator.iter_errors(doc)]
    assert not errors, errors

    text, changes = oracle.adjust(doc["before"], doc["raw"], doc["verbatim"])
    assert text == doc["expected_text"]
    assert changes == [change["kind"] for change in doc["expected_changes"]]


@pytest.mark.parametrize("doc", CASES, ids=lambda doc: doc["case_id"])
def test_expected_text_changes_only_the_boundary(doc: dict) -> None:
    """Pins the fixture table itself: an expected text may differ from the
    raw text only by one prepended space and the full lowercase mapping of
    the first cased character."""
    raw, expected = doc["raw"], doc["expected_text"]
    kinds = [change["kind"] for change in doc["expected_changes"]]
    if "leading_space" in kinds:
        assert expected.startswith(" ")
        expected = expected[1:]
    if "first_letter_case" in kinds:
        i = next(i for i, ch in enumerate(raw) if oracle._is_cased(ch))
        raw = raw[:i] + raw[i].lower() + raw[i + 1 :]
    assert expected == raw
