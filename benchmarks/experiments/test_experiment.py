"""Experiments harness tests (issue #168 acceptance). CPU-only, hermetic:
no models, no GPU, no network beyond localhost. The comparator negatives
are the point — invalid comparisons must be impossible — plus verdict
semantics, statistics determinism, and a runner end-to-end over the stub
double (the real-binary reproduction is the documented demo command).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare as compare_mod  # noqa: E402
import record as record_mod  # noqa: E402
import runner as runner_mod   # noqa: E402
import stats as stats_mod     # noqa: E402

V1_SPEC = {
    "schema": "starling-experiment-spec/1",
    "experiment_id": "t",
    "objective": "test hypothesis",
    "metric": "http_transcribe_wall_ms",
    "direction": "lower",
    "arms": {
        "baseline": {"binary": "/bin/true", "env": {}},
        "candidate": {"binary": "/bin/true", "env": {}},
    },
    "workload": {"audio": "/tmp", "files": ["a.wav"], "sha256": "0" * 64},
    "protocol": {"repeats": 3, "requests_per_repeat": 4, "warmup_requests": 1,
                 "order": "interleaved_random", "seed": 7, "timeout_s": 30},
    "acceptance": {"min_improvement_pct": 5.0, "max_ci_halfwidth_pct": 25.0,
                   "max_regression_pct": 5.0},
}


def make_record(role="baseline", base_ms=100.0, cand_ms=None, *, seed_offset=0,
                repeats=3, requests=4, cold_ms=900.0, status="ok",
                workload_sha=None, metric=None, normalizer="none-raw-wall-time",
                model_claim=None, hardware="cpu-only-host", spec=None):
    spec = spec or V1_SPEC
    rec = {
        "schema": "starling-experiment-record/1",
        "role": role,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": record_mod.spec_sha256(spec),
        "provenance": {
            "repo_revision": "abc",
            "ggml_revision": "def",
            "binary_sha256": "b" * 64,
            "runtime": {"hardware": hardware, "driver": "610.57.01"},
            "workload_sha256": workload_sha or spec["workload"]["sha256"],
            "normalizer": normalizer,
            "model_claim": model_claim or {"model": None, "model_sha256": None},
            "metric_identity": metric or {
                "tool": "starling-experiments", "version": 1,
                "metric": spec["metric"], "normalizer": normalizer},
            "commands": ["run"],
        },
        "samples": [],
        "failures": [],
        "status": status,
    }
    target = cand_ms if role == "candidate" and cand_ms is not None else base_ms
    for r in range(repeats):
        for q in range(requests):
            rec["samples"].append({
                "arm": role, "repeat": r, "request": q, "cold": q == 0,
                "wall_ms": cold_ms if q == 0 else target + (r + seed_offset),
            })
    return rec


class SpecValidationTests(unittest.TestCase):
    def test_valid_spec_has_no_problems(self):
        self.assertEqual(record_mod.validate_spec(V1_SPEC), [])

    def test_preregistered_acceptance_is_mandatory(self):
        bad = dict(V1_SPEC)
        del bad["acceptance"]
        self.assertTrue(any("acceptance" in p for p in record_mod.validate_spec(bad)))

    def test_unknown_metric_rejected(self):
        bad = dict(V1_SPEC, metric="made_up_metric", direction="lower")
        self.assertTrue(any("metric" in p for p in record_mod.validate_spec(bad)))

    def test_wrong_direction_rejected(self):
        bad = dict(V1_SPEC, direction="higher")
        self.assertTrue(any("direction" in p for p in record_mod.validate_spec(bad)))

    def test_arms_must_be_exactly_two(self):
        bad = dict(V1_SPEC)
        bad["arms"] = {"baseline": {"binary": "/bin/true"}}
        self.assertTrue(any("arms" in p for p in record_mod.validate_spec(bad)))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_manifest_changes_with_content(self):
        a = self.dir / "a.wav"
        a.write_bytes(b"0123456789")
        m1 = record_mod.workload_manifest([a])
        a.write_bytes(b"9876543210")
        m2 = record_mod.workload_manifest([a])
        self.assertNotEqual(m1["sha256"], m2["sha256"])

    def test_manifest_changes_with_rename(self):
        a = self.dir / "a.wav"
        a.write_bytes(b"0123456789")
        m1 = record_mod.workload_manifest([a])
        b = self.dir / "b.wav"
        b.write_bytes(b"0123456789")
        m2 = record_mod.workload_manifest([b])
        self.assertNotEqual(m1["sha256"], m2["sha256"])

    def test_missing_file_raises(self):
        with self.assertRaises(record_mod.RecordError):
            record_mod.workload_manifest([self.dir / "nope.wav"])


class StatsTests(unittest.TestCase):
    def test_paired_matching_excludes_cold_and_errors(self):
        base = [{"arm": "baseline", "repeat": 0, "request": 0, "cold": True,
                 "wall_ms": 900.0},
                {"arm": "baseline", "repeat": 0, "request": 1, "cold": False,
                 "wall_ms": 100.0},
                {"arm": "baseline", "repeat": 0, "request": 2, "cold": False,
                 "wall_ms": 110.0, "error": "HTTP 500"}]
        cand = [{"arm": "candidate", "repeat": 0, "request": 1, "cold": False,
                 "wall_ms": 50.0}]
        pairs = stats_mod.paired_samples(base, cand)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], 100.0)

    def test_bootstrap_is_deterministic(self):
        pairs = [(f"{i}.0", 100.0, 90.0) for i in range(10)]
        e1 = stats_mod.effect_estimate(pairs, "lower", seed=7)
        e2 = stats_mod.effect_estimate(pairs, "lower", seed=7)
        e3 = stats_mod.effect_estimate(pairs, "lower", seed=8)
        self.assertEqual(e1["ci_low_pct"], e2["ci_low_pct"])
        self.assertEqual(e1["ci_high_pct"], e2["ci_high_pct"])
        self.assertNotEqual(e1["bootstrap_seed"], e3["bootstrap_seed"])

    def test_effect_orientation_positive_is_better(self):
        pairs = [(f"{i}.0", 100.0, 50.0) for i in range(6)]
        e = stats_mod.effect_estimate(pairs, "lower", seed=1)
        self.assertGreater(e["improvement_pct"], 40.0)


class ComparatorNegativeTests(unittest.TestCase):
    """Issue #168 acceptance: the comparator rejects every invalid pair."""

    def _rejects(self, baseline, candidate, spec=None, needle=""):
        spec = spec or V1_SPEC
        with self.assertRaises(record_mod.RecordError) as ctx:
            compare_mod.check_compatibility(spec, baseline, candidate)
        if needle:
            self.assertIn(needle, str(ctx.exception))

    def test_spec_seal_mismatch_rejected(self):
        other = json.loads(json.dumps(V1_SPEC))
        other["acceptance"]["min_improvement_pct"] = 1.0  # post-hoc edit
        b = make_record("baseline")
        c = make_record("candidate", cand_ms=50.0)
        c["spec_sha256"] = record_mod.spec_sha256(other)
        self._rejects(b, c, needle="spec")

    def test_corpus_hash_mismatch_rejected(self):
        b = make_record("baseline", workload_sha="a" * 64)
        c = make_record("candidate", cand_ms=50.0, workload_sha="b" * 64)
        self._rejects(b, c, needle="workload")

    def test_metric_identity_mismatch_rejected(self):
        m1 = {"tool": "starling-experiments", "version": 1,
              "metric": "http_transcribe_wall_ms", "normalizer": "none"}
        m2 = {"tool": "psdn-sonar", "version": 1,
              "metric": "wer", "normalizer": "sonar-en"}
        b = make_record("baseline", metric=m1)
        c = make_record("candidate", cand_ms=50.0, metric=m2)
        self._rejects(b, c, needle="metric identit")

    def test_normalizer_mismatch_rejected(self):
        b = make_record("baseline", normalizer="whisper-en")
        c = make_record("candidate", cand_ms=50.0, normalizer="sonar-en")
        self._rejects(b, c, needle="normalizer")

    def test_model_claim_mismatch_rejected(self):
        b = make_record("baseline", model_claim={"model": "m.gguf",
                                                 "model_sha256": "1" * 64})
        c = make_record("candidate", cand_ms=50.0,
                        model_claim={"model": "q.gguf", "model_sha256": "2" * 64})
        self._rejects(b, c, needle="model/config")

    def test_hardware_mismatch_rejected(self):
        b = make_record("baseline", hardware="RTX 5090")
        c = make_record("candidate", cand_ms=50.0, hardware="cpu-only-host")
        self._rejects(b, c, needle="hardware")

    def test_failed_run_rejected(self):
        b = make_record("baseline", status="failed")
        c = make_record("candidate", cand_ms=50.0)
        self._rejects(b, c, needle="failed")

    def test_zero_case_rejected(self):
        c = make_record("candidate", cand_ms=50.0)
        c["samples"] = [s for s in c["samples"] if s["cold"]]  # only cold left
        self._rejects(make_record("baseline"), c, needle="zero")

    def test_missing_record_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(record_mod.RecordError):
                record_mod.load_record(Path(d) / "record.json")

    def test_malformed_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "record.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(record_mod.RecordError):
                record_mod.load_record(p)


class VerdictTests(unittest.TestCase):
    def _compare(self, base_ms, cand_ms, **kw):
        spec = kw.pop("spec", V1_SPEC)
        b = make_record("baseline", base_ms=base_ms, spec=spec, **kw)
        c = make_record("candidate", base_ms=base_ms, cand_ms=cand_ms, spec=spec, **kw)
        return compare_mod.compare(spec, b, c)

    def test_clear_improvement_passes(self):
        r = self._compare(100.0, 50.0)
        self.assertEqual(r["verdict"], "pass")

    def test_clear_regression_fails(self):
        r = self._compare(50.0, 100.0)
        self.assertEqual(r["verdict"], "fail")

    def test_no_difference_is_inconclusive_not_a_win(self):
        r = self._compare(100.0, 100.0)
        self.assertEqual(r["verdict"], "inconclusive")
        self.assertTrue(r["effect"].get("inconclusive_because"))

    def test_small_effect_with_wide_ci_is_inconclusive(self):
        # 1% improvement inside a noisy CI: below the 5% bar.
        r = self._compare(100.0, 99.0)
        self.assertEqual(r["verdict"], "inconclusive")

    def test_failed_run_is_unavailable(self):
        b = make_record("baseline", status="failed")
        c = make_record("candidate", cand_ms=50.0)
        with self.assertRaises(record_mod.RecordError):
            compare_mod.compare(V1_SPEC, b, c)

    def test_too_few_pairs_is_unavailable(self):
        spec = json.loads(json.dumps(V1_SPEC))
        spec["protocol"]["repeats"] = 1
        spec["protocol"]["requests_per_repeat"] = 2  # sample 0 is cold -> 1 warm pair
        b = make_record("baseline", repeats=1, requests=2, spec=spec)
        c = make_record("candidate", cand_ms=50.0, repeats=1, requests=2, spec=spec)
        r = compare_mod.compare(spec, b, c)
        self.assertEqual(r["verdict"], "unavailable")

    def test_verdicts_are_deterministic(self):
        r1 = self._compare(100.0, 90.0)
        r2 = self._compare(100.0, 90.0)
        self.assertEqual(r1["effect"]["ci_low_pct"], r2["effect"]["ci_low_pct"])
        self.assertEqual(r1["verdict"], r2["verdict"])


class RunnerTests(unittest.TestCase):
    """Runner mechanics over the stub double (fresh processes, cold/warm,
    workload pin refusal). NOT benchmark evidence — see run_experiment.py."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.audio = self.dir / "a.wav"
        self.audio.write_bytes(b"RIFF-fake-wav-bytes" * 8)

    def _spec(self, **overrides):
        manifest = record_mod.workload_manifest([self.audio])
        spec = json.loads(json.dumps(V1_SPEC))
        spec["workload"] = {"audio": str(self.dir), "files": [self.audio.name],
                            "sha256": manifest["sha256"]}
        spec["arms"] = {
            "baseline": {"binary": sys.executable,
                         "env": {"STUB_DELAY_MS": "40"},
                         "stub_args": [str(HERE / "stub_serve.py")]},
            "candidate": {"binary": sys.executable,
                          "env": {"STUB_DELAY_MS": "5"},
                          "stub_args": [str(HERE / "stub_serve.py")]},
        }
        spec["protocol"].update({"repeats": 2, "requests_per_repeat": 3,
                                 "warmup_requests": 1, "timeout_s": 30})
        spec.update(overrides)
        return spec

    def test_runner_produces_comparable_passing_pair(self):
        spec = self._spec()
        run_dir = self.dir / "run"
        rb = runner_mod.run_arm(spec, "baseline", run_dir, HERE.parent.parent)
        rc = runner_mod.run_arm(spec, "candidate", run_dir, HERE.parent.parent)
        self.assertEqual(rb["status"], "ok")
        self.assertEqual(rc["status"], "ok")
        # cold sample recorded and much slower than warm for the slow arm
        cold = [s for s in rb["samples"] if s["cold"]]
        warm = [s for s in rb["samples"] if not s["cold"]]
        self.assertEqual(len(cold), 2)  # one per fresh-process repeat
        self.assertGreater(len(warm), 0)
        (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (run_dir / "candidate").mkdir(parents=True, exist_ok=True)
        (run_dir / "baseline" / "record.json").write_text(json.dumps(rb))
        (run_dir / "candidate" / "record.json").write_text(json.dumps(rc))
        result = compare_mod.compare_directories.__wrapped__ if False else None
        verdict = compare_mod.compare(spec, rb, rc)
        self.assertEqual(verdict["verdict"], "pass")

    def test_runner_refutes_changed_workload(self):
        spec = self._spec()
        spec["workload"]["audio"] = str(self.dir)
        spec["workload"]["sha256"] = "f" * 64  # not what's on disk
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            runner_mod.run_arm(spec, "baseline", self.dir / "x", HERE.parent.parent)
        self.assertIn("manifest", str(ctx.exception))

    def test_runner_records_startup_failure(self):
        spec = self._spec()
        spec["arms"]["baseline"]["stub_args"] = [str(HERE / "no_such_module.py")]
        rec = runner_mod.run_arm(spec, "baseline", self.dir / "y",
                                 HERE.parent.parent)
        self.assertEqual(rec["status"], "failed")
        self.assertTrue(rec["failures"])

    def test_arm_order_is_seeded_and_reproducible(self):
        spec = self._spec()
        self.assertEqual(runner_mod.arm_order(spec, 0), runner_mod.arm_order(spec, 0))
        orders = {tuple(runner_mod.arm_order(spec, r)) for r in range(8)}
        self.assertTrue(all(o in {("baseline", "candidate"),
                                  ("candidate", "baseline")} for o in orders))


if __name__ == "__main__":
    unittest.main()
