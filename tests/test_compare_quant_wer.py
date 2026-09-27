"""Paired quality gates must reject incompatible or underpowered studies."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from compare_quant_wer import InvalidComparison, compare
from wer import wer_pct


def _inputs(tmp_path, *, delta_clip=False):
    spec = tmp_path / "protocol.json"
    spec.write_text(json.dumps({
        "schema": "quant-wer-noninferiority-v1", "cohorts": {"en_us_validation": 10},
        "max_wer_delta_pp": 0.2, "min_bytes_saved": 15_000_000,
        "bootstrap_resamples": 1000, "bootstrap_seed": 7,
    }))
    seal = hashlib.sha256(spec.read_bytes()).hexdigest()

    def row(arm):
        clips = []
        for i in range(10):
            ref = "one two three"
            hyp = "one two three" if arm == "baseline" or not delta_clip or i else "one two"
            clips.append({"id": f"en_us_validation_{i}.wav",
                          "audio_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                          "reference": ref, "hypothesis": hyp,
                          "wer": wer_pct(ref, hyp)})
        return {"model": arm, "clips": {"en_us_validation": clips},
                "provenance": {"protocol_sha256": seal,
                               "model_sha256": "a" * 64 if arm == "baseline" else "b" * 64,
                               "model_bytes": 325_000_000 if arm == "baseline" else 309_000_000,
                               "source_sha256": "c" * 64, "imatrix_sha256": "d" * 64,
                               "engine_sha256": "e" * 64,
                               "scorer_sha256": hashlib.sha256(
                                   (Path(__file__).resolve().parents[1] / "benchmarks/wer.py")
                                   .read_bytes()).hexdigest(),
                               "device": "cpu"}}

    baseline, candidate = row("baseline"), row("candidate")
    bp, cp = tmp_path / "baseline.json", tmp_path / "candidate.json"

    def save():
        bp.write_text(json.dumps([baseline]))
        cp.write_text(json.dumps([candidate]))

    save()
    return spec, bp, cp, baseline, candidate, save


def test_paired_gate_passes_identical_hypotheses_with_real_byte_saving(tmp_path):
    spec, bp, cp, *_ = _inputs(tmp_path)
    result = compare(spec, bp, cp)
    assert result["verdict"] == "pass"
    assert result["bytes_saved"] == 16_000_000
    assert result["cohorts"]["en_us_validation"]["paired_95_ci_pp"] == [0.0, 0.0]


def test_paired_gate_keeps_wide_interval_inconclusive(tmp_path):
    spec, bp, cp, *_ = _inputs(tmp_path, delta_clip=True)
    result = compare(spec, bp, cp)
    assert result["verdict"] == "inconclusive"
    assert result["cohorts"]["en_us_validation"]["paired_95_ci_pp"][1] > 0.2


def test_established_regression_fails(tmp_path):
    spec, bp, cp, baseline, candidate, save = _inputs(tmp_path)
    for clip in candidate["clips"]["en_us_validation"]:
        clip["hypothesis"] = "one two"
        clip["wer"] = wer_pct(clip["reference"], clip["hypothesis"])
    save()
    result = compare(spec, bp, cp)
    assert result["verdict"] == "fail"
    assert result["cohorts"]["en_us_validation"]["regression_established"]


@pytest.mark.parametrize("mutate, message", [
    (lambda b, c: c["clips"]["en_us_validation"][0].update(audio_sha256="f" * 64),
     "audio or reference differs"),
    (lambda b, c: c["clips"]["en_us_validation"][1].update(id="en_us_validation_0.wav"),
     "duplicate clip ID"),
    (lambda b, c: c["provenance"].update(device="CUDA0"), "device differs"),
    (lambda b, c: c["provenance"].update(imatrix_sha256="x" * 64), "imatrix_sha256 differs"),
    (lambda b, c: c["provenance"].update(protocol_sha256="wrong"), "protocol seal differs"),
    (lambda b, c: c["clips"]["en_us_validation"][0].update(wer=12),
     "stored WER does not match scorer"),
])
def test_incompatible_studies_are_rejected(tmp_path, mutate, message):
    spec, bp, cp, baseline, candidate, save = _inputs(tmp_path)
    mutate(baseline, candidate)
    save()
    with pytest.raises(InvalidComparison, match=message):
        compare(spec, bp, cp)


def test_protocol_change_after_runs_is_rejected(tmp_path):
    spec, bp, cp, *_ = _inputs(tmp_path)
    data = json.loads(spec.read_text())
    data["max_wer_delta_pp"] = 1.0
    spec.write_text(json.dumps(data))
    with pytest.raises(InvalidComparison, match="protocol seal differs"):
        compare(spec, bp, cp)
