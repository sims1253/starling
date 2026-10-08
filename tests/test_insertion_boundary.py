"""Replays ``packages/contracts/insertion-boundary/fixtures/boundary-cases.json``
through the oracle in ``tests/insertion_boundary.py`` (#341).

Run with:

    uv run python -m pytest tests/test_insertion_boundary.py -q

The Rust port replays the same file (``starling-processing`` conformance
test), so a case changed on one side fails on the other.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import insertion_boundary as oracle
import minischema

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

CONTRACT = pathlib.Path(__file__).parent.parent / "packages" / "contracts" / "insertion-boundary"
CASES_PATH = CONTRACT / "fixtures" / "boundary-cases.json"
SCHEMA_PATH = CONTRACT / "boundary.schema.json"


def _cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def test_fixture_table_is_valid_and_unique() -> None:
    cases = _cases()
    assert len(cases) >= 30, "the acceptance fixture classes must stay pinned"
    ids = [case["case_id"] for case in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"


@pytest.mark.parametrize("doc", _cases(), ids=lambda doc: doc["case_id"])
def test_case_conforms_to_schema(doc: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = minischema.errors(doc, schema)
    assert not errors, errors
    if jsonschema is not None:  # pragma: no branch - both paths run when present
        validator = jsonschema.Draft202012Validator(schema)
        extra = [f"jsonschema: {e.message}" for e in validator.iter_errors(doc)]
        assert not extra, extra


@pytest.mark.parametrize("doc", _cases(), ids=lambda doc: doc["case_id"])
def test_case(doc: dict) -> None:
    case = oracle.BoundaryCase.from_json(doc)
    got = oracle.adjust(case)

    assert got.text == doc["expected_text"], (
        f"{doc['case_id']}: expected {doc['expected_text']!r}, got {got.text!r}"
    )
    assert got.changes == doc["expected_changes"], (
        f"{doc['case_id']}: expected changes {doc['expected_changes']}, "
        f"got {got.changes}"
    )


def test_raw_is_never_modified_beyond_the_boundary() -> None:
    """The adjustment may only prepend one space and lowercase the first
    cased character: modulo an optional leading space, the result differs
    from the raw text in exactly that one character, by its lowercase."""
    for doc in _cases():
        got = oracle.adjust(oracle.BoundaryCase.from_json(doc)).text
        raw = doc["raw"]
        stripped = got[1:] if got.startswith(" ") and not raw.startswith(" ") else got
        # `stripped` is index-aligned with `raw` in both branches.
        diffs = [i for i, (a, b) in enumerate(zip(stripped, raw)) if a != b]
        if not diffs:
            continue
        first_cased = next(
            (i for i, ch in enumerate(raw) if ch.isalpha() and ch != ch.lower()),
            None,
        )
        assert first_cased is not None, f"{doc['case_id']}: unexpected diff at {diffs}"
        assert diffs == [first_cased], (
            f"{doc['case_id']}: changes beyond the boundary at {diffs}"
        )
        assert stripped[first_cased] == raw[first_cased].lower(), (
            f"{doc['case_id']}: not a case change"
        )
