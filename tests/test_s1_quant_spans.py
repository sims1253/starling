"""The S1 quant pilot reports only candidate-induced protected span losses."""

import json
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/s1"))
from quant_spans import audit_output, check_cases, compare


def test_cases_parse_and_have_unique_labels():
    cases = json.loads((Path(__file__).resolve().parent / "fixtures/s1_quant_spans.json").read_text())
    check_cases(cases)
    assert len(cases) == 8
    assert audit_output(cases[0], "Do not send this to Maya.") == {
        "negation": True, "recipient": True}


def test_comparison_counts_new_and_preexisting_violations():
    shared = {"schema": "s1-quant-spans-v1", "source_sha256": "s",
              "engine_sha256": "e", "cases_sha256": "c", "device": "CPU"}
    baseline = {**shared, "model_bytes": 100, "results": [
        {"id": "a", "output": "do not send to Maya", "spans_present": {
            "negation": True, "name": True}},
        {"id": "b", "output": "empty", "spans_present": {"number": False}},
    ]}
    candidate = {**shared, "model_bytes": 80, "results": [
        {"id": "a", "output": "send to Maya", "spans_present": {
            "negation": False, "name": True}},
        {"id": "b", "output": "empty", "spans_present": {"number": False}},
    ]}
    result = compare(baseline, candidate)
    assert result["new_violations"] == ["a:negation"]
    assert result["baseline_violations"] == ["b:number"]
    assert result["changed_outputs"] == ["a"]
    assert result["bytes_saved"] == 20
    candidate["cases_sha256"] = "changed"
    with pytest.raises(ValueError, match="cases_sha256"):
        compare(baseline, candidate)


def test_comparison_rechecks_stored_spans_against_source_cases(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "a", "transcript": "keep it",
                                  "spans": {"negation": r"\bnot\b"}}]))
    shared = {"schema": "s1-quant-spans-v1", "source_sha256": "s",
              "engine_sha256": "e", "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
              "device": "CPU", "model_bytes": 100,
              "results": [{"id": "a", "output": "do not send", "spans_present": {
                  "negation": True}}]}
    candidate = json.loads(json.dumps(shared))
    candidate["model_bytes"] = 80
    candidate["results"][0]["output"] = "send"
    with pytest.raises(ValueError, match="stored span status"):
        compare(shared, candidate, cases)
