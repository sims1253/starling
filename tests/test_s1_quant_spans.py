"""The S1 quant pilot reports only candidate-induced protected span losses."""

import json
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/s1"))
from quant_spans import audit_output, check_cases, compare


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_cases_parse_and_have_unique_labels():
    cases = json.loads((Path(__file__).resolve().parent / "fixtures/s1_quant_spans.json").read_text())
    check_cases(cases)
    assert len(cases) == 8
    assert audit_output(cases[0], "Do not send this to Maya.") == {
        "negation": True, "recipient": True}


@pytest.mark.parametrize("case_id,label,valid,mutants", [
    ("amount_correction", "corrected_amount", "$45", ["145", "$145", "forty fives"]),
    ("email_address", "address", "alex.chen@example.org",
     ["alex.chen@example.org.evil", "malex.chen@example.org"]),
    ("url", "link", "example.org/pricing",
     ["example.org/pricing-old", "example.org/pricing/old"]),
])
def test_protected_spans_reject_seeded_boundary_mutants(case_id, label, valid, mutants):
    cases = json.loads((Path(__file__).resolve().parent / "fixtures/s1_quant_spans.json").read_text())
    case = next(case for case in cases if case["id"] == case_id)
    assert audit_output(case, valid)[label]
    for mutant in mutants:
        assert not audit_output(case, mutant)[label], mutant


def test_comparison_counts_new_and_preexisting_violations():
    shared = {"schema": "s1-quant-spans-v1", "source_sha256": _digest("source"),
              "engine_sha256": "e", "cases_sha256": "c", "device": "CPU"}
    baseline = {**shared, "model_sha256": _digest("source"), "model_bytes": 100, "results": [
        {"id": "a", "output": "do not send to Maya", "spans_present": {
            "negation": True, "name": True}},
        {"id": "b", "output": "empty", "spans_present": {"number": False}},
    ]}
    candidate = {**shared, "model_sha256": _digest("candidate"), "model_bytes": 80, "results": [
        {"id": "a", "output": "send to Maya", "spans_present": {
            "negation": False, "name": True}},
        {"id": "b", "output": "empty", "spans_present": {"number": False}},
    ]}
    result = compare(baseline, candidate)
    assert result["new_violations"] == ["a:negation"]
    assert result["baseline_violations"] == ["b:number"]
    assert result["changed_outputs"] == ["a"]
    assert result["bytes_saved"] == 20
    assert result["baseline_model_sha256"] == baseline["model_sha256"]
    assert result["candidate_model_sha256"] == candidate["model_sha256"]
    candidate["cases_sha256"] = "changed"
    with pytest.raises(ValueError, match="cases_sha256"):
        compare(baseline, candidate)


def test_comparison_rechecks_stored_spans_against_source_cases(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "a", "transcript": "keep it",
                                  "spans": {"negation": r"\bnot\b"}}]))
    shared = {"schema": "s1-quant-spans-v1", "source_sha256": _digest("source"),
              "engine_sha256": "e", "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
              "device": "CPU", "model_sha256": _digest("source"), "model_bytes": 100,
              "results": [{"id": "a", "output": "do not send", "spans_present": {
                  "negation": True}}]}
    candidate = json.loads(json.dumps(shared))
    candidate["model_sha256"] = _digest("candidate")
    candidate["model_bytes"] = 80
    candidate["results"][0]["output"] = "send"
    with pytest.raises(ValueError, match="stored span status"):
        compare(shared, candidate, cases)


@pytest.mark.parametrize("field,value,message", [
    ("model_sha256", None, "invalid model_sha256"),
    ("model_sha256", "bad", "invalid model_sha256"),
    ("model_bytes", None, "invalid model_bytes"),
    ("model_bytes", True, "invalid model_bytes"),
    ("model_bytes", 0, "invalid model_bytes"),
])
def test_comparison_rejects_missing_or_invalid_candidate_identity(field, value, message):
    source = _digest("source")
    baseline = {"schema": "s1-quant-spans-v1", "source_sha256": source,
                "model_sha256": source, "model_bytes": 100,
                "engine_sha256": "e", "cases_sha256": "c", "device": "CPU",
                "results": [{"id": "a", "output": "ok", "spans_present": {"span": True}}]}
    candidate = {**baseline, "model_sha256": _digest("candidate"), "model_bytes": 80}
    candidate[field] = value
    with pytest.raises(ValueError, match=message):
        compare(baseline, candidate)


def test_comparison_rejects_wrong_baseline_and_identical_candidate_model():
    source = _digest("source")
    baseline = {"schema": "s1-quant-spans-v1", "source_sha256": source,
                "model_sha256": _digest("other"), "model_bytes": 100,
                "engine_sha256": "e", "cases_sha256": "c", "device": "CPU",
                "results": [{"id": "a", "output": "ok", "spans_present": {"span": True}}]}
    candidate = {**baseline, "model_sha256": _digest("candidate"), "model_bytes": 80}
    with pytest.raises(ValueError, match="baseline model_sha256"):
        compare(baseline, candidate)
    baseline["model_sha256"] = source
    candidate["model_sha256"] = source
    with pytest.raises(ValueError, match="candidate model_sha256"):
        compare(baseline, candidate)
