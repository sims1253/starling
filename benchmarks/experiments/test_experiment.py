"""Experiments harness tests (issue #168 acceptance). CPU-only, hermetic:
no models, no GPU, no network beyond localhost. The comparator negatives
are the point — invalid comparisons must be impossible — plus verdict
semantics, statistics determinism, and a runner end-to-end over the stub
double (the real-binary reproduction is the documented demo command).
"""

from __future__ import annotations

import json
import os
import random
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
import run_experiment as cli_mod  # noqa: E402

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

    def test_workload_pin_is_mandatory(self):
        bad = json.loads(json.dumps(V1_SPEC))
        del bad["workload"]["sha256"]
        self.assertTrue(any("sha256" in p for p in record_mod.validate_spec(bad)))


class RecordValidationTests(unittest.TestCase):
    def test_malformed_samples_are_rejected(self):
        rec = make_record("baseline")
        rec["samples"][1]["wall_ms"] = float("nan")
        self.assertTrue(any("wall_ms" in p for p in record_mod.validate_record(rec)))
        rec = make_record("baseline")
        rec["samples"][2]["arm"] = "candidate"  # disagrees with role
        self.assertTrue(any("role" in p for p in record_mod.validate_record(rec)))
        rec = make_record("baseline")
        rec["samples"].append(dict(rec["samples"][2]))  # duplicate (repeat, request)
        self.assertTrue(any("duplicates" in p for p in record_mod.validate_record(rec)))
        rec = make_record("baseline")
        rec["samples"][0]["repeat"] = -1
        self.assertTrue(any("repeat" in p for p in record_mod.validate_record(rec)))


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
        self.assertEqual(pairs[0][2], 100.0)

    def test_bootstrap_is_deterministic(self):
        pairs = [(i, 0, 100.0, 90.0) for i in range(10)]
        e1 = stats_mod.effect_estimate(pairs, "lower", seed=7)
        e2 = stats_mod.effect_estimate(pairs, "lower", seed=7)
        e3 = stats_mod.effect_estimate(pairs, "lower", seed=8)
        self.assertEqual(e1["ci_low_pct"], e2["ci_low_pct"])
        self.assertEqual(e1["ci_high_pct"], e2["ci_high_pct"])
        self.assertNotEqual(e1["bootstrap_seed"], e3["bootstrap_seed"])

    def test_bootstrap_resamples_whole_repeats_not_requests(self):
        # Two repeat clusters with very different levels: a request-level
        # bootstrap would narrow the CI by treating the 2x4 correlated
        # observations as independent evidence; the cluster bootstrap keeps
        # the repeat-level spread in the interval.
        pairs = [(r, q, 100.0, 100.0 if r == 0 else 50.0)
                 for r in (0, 1) for q in range(4)]
        e = stats_mod.effect_estimate(pairs, "lower", seed=3)
        self.assertEqual(e["n_pairs"], 8)
        self.assertEqual(e["n_repeat_clusters"], 2)
        self.assertLess(e["ci_low_pct"], 20.0)   # repeat 0: 0% improvement
        self.assertGreater(e["ci_high_pct"], 20.0)  # repeat 1: 50% improvement

    def test_effect_orientation_positive_is_better(self):
        pairs = [(i, 0, 100.0, 50.0) for i in range(6)]
        e = stats_mod.effect_estimate(pairs, "lower", seed=1)
        self.assertGreater(e["improvement_pct"], 40.0)

    def test_higher_is_better_direction_is_not_inverted(self):
        pairs = [(i, 0, 100.0, 150.0) for i in range(6)]
        e = stats_mod.effect_estimate(pairs, "higher", seed=1)
        self.assertGreater(e["improvement_pct"], 40.0)

    def test_unknown_direction_is_refused(self):
        pairs = [(i, 0, 100.0, 50.0) for i in range(6)]
        with self.assertRaises(ValueError):
            stats_mod.effect_estimate(pairs, "sideways", seed=1)


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

    def test_failed_run_is_unavailable_not_a_crash(self):
        b = make_record("baseline", status="failed")
        b["failures"] = [{"repeat": 0, "error": "server did not become healthy"}]
        c = make_record("candidate", cand_ms=50.0)
        r = compare_mod.compare(V1_SPEC, b, c)
        self.assertEqual(r["verdict"], "unavailable")
        self.assertIn("failed", r["reason"])
        self.assertTrue(r["failures"]["baseline"])  # diagnostics accompany it

    def test_zero_case_is_unavailable(self):
        c = make_record("candidate", cand_ms=50.0)
        c["samples"] = [s for s in c["samples"] if s["cold"]]  # only cold left
        r = compare_mod.compare(V1_SPEC, make_record("baseline"), c)
        self.assertEqual(r["verdict"], "unavailable")
        self.assertIn("zero usable", r["reason"])

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
        r = compare_mod.compare(V1_SPEC, b, c)
        self.assertEqual(r["verdict"], "unavailable")
        self.assertIn("failed", r["reason"])

    def test_too_few_pairs_is_unavailable(self):
        spec = json.loads(json.dumps(V1_SPEC))
        spec["protocol"]["repeats"] = 1
        spec["protocol"]["requests_per_repeat"] = 2  # sample 0 is cold -> 1 warm pair
        b = make_record("baseline", repeats=1, requests=2, spec=spec)
        c = make_record("candidate", cand_ms=50.0, repeats=1, requests=2, spec=spec)
        r = compare_mod.compare(spec, b, c)
        self.assertEqual(r["verdict"], "unavailable")

    def test_single_repeat_is_unavailable_not_a_zero_width_pass(self):
        # repeats=1 is validation-legal, but one fresh-process cluster means
        # every bootstrap resample redraws the same data: ci == point
        # estimate, a manufactured 0%-width "certainty" that must not pass.
        spec = json.loads(json.dumps(V1_SPEC))
        spec["protocol"]["repeats"] = 1
        spec["protocol"]["requests_per_repeat"] = 4  # 3 warm pairs, 1 cluster
        b = make_record("baseline", repeats=1, requests=4, spec=spec)
        c = make_record("candidate", cand_ms=50.0, repeats=1, requests=4, spec=spec)
        r = compare_mod.compare(spec, b, c)
        self.assertEqual(r["verdict"], "unavailable")
        self.assertIn("single fresh-process repeat", r["reason"])

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

    def _run_interleaved(self, spec, run_dir):
        """The CLI's calling convention: repeat-by-repeat, both arms per
        repeat, in the seeded order (run_experiment._run_interleaved)."""
        records = {}
        for repeat in range(spec["protocol"]["repeats"]):
            for arm in runner_mod.arm_order(spec, repeat):
                records[arm] = runner_mod.run_arm(spec, arm, run_dir,
                                                  HERE.parent.parent, repeat)
        return records["baseline"], records["candidate"]

    def test_runner_produces_comparable_passing_pair(self):
        spec = self._spec()
        run_dir = self.dir / "run"
        rb, rc = self._run_interleaved(spec, run_dir)
        self.assertEqual(rb["status"], "ok")
        self.assertEqual(rc["status"], "ok")
        # one cold sample per fresh-process repeat; 1 cold + 1 warmup + 3
        # timed requests per repeat, accumulated across the repeat calls
        cold = [s for s in rb["samples"] if s["cold"]]
        self.assertEqual(len(cold), 2)
        self.assertEqual(len(rb["samples"]), 2 * (1 + 1 + 3))
        # the on-disk record is the merged one the comparator reads back
        on_disk = json.loads((run_dir / "baseline" / "record.json").read_text())
        self.assertEqual(on_disk["samples"], rb["samples"])
        verdict = compare_mod.compare(spec, rb, rc)
        self.assertEqual(verdict["verdict"], "pass")

    def test_cold_then_warmup_then_timed_split(self):
        spec = self._spec()
        spec["protocol"]["repeats"] = 1
        rb = runner_mod.run_arm(spec, "baseline", self.dir / "split",
                                HERE.parent.parent, 0)
        r0 = sorted(rb["samples"], key=lambda s: s["request"])
        self.assertEqual([s["request"] for s in r0], list(range(5)))
        self.assertTrue(r0[0]["cold"])
        self.assertFalse(r0[0]["warmup"])           # cold is its own category
        self.assertFalse(r0[1]["cold"])
        self.assertTrue(r0[1]["warmup"])            # warmup_requests=1 MORE
        self.assertFalse(any(s["cold"] or s["warmup"] for s in r0[2:]))  # then timed

    def test_paired_requests_measure_the_same_clip(self):
        second = self.dir / "b.wav"
        second.write_bytes(b"RIFF-other-fake-wav" * 6)
        spec = self._spec()
        manifest = record_mod.workload_manifest([self.audio, second])
        spec["workload"] = {"audio": str(self.dir),
                            "files": [self.audio.name, second.name],
                            "sha256": manifest["sha256"]}
        rb, rc = self._run_interleaved(spec, self.dir / "paired")
        base_clip = {(s["repeat"], s["request"]): s["audio"] for s in rb["samples"]}
        self.assertEqual(len(base_clip), len(rb["samples"]))  # keys unique
        for s in rc["samples"]:
            self.assertEqual(base_clip[(s["repeat"], s["request"])], s["audio"],
                             "paired (repeat, request) must send the same clip")

    def test_runner_refutes_changed_workload(self):
        spec = self._spec()
        spec["workload"]["audio"] = str(self.dir)
        spec["workload"]["sha256"] = "f" * 64  # not what's on disk
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            runner_mod.run_arm(spec, "baseline", self.dir / "x",
                               HERE.parent.parent, 0)
        self.assertIn("manifest", str(ctx.exception))

    def test_runner_records_startup_failure(self):
        spec = self._spec()
        spec["arms"]["baseline"]["stub_args"] = [str(HERE / "no_such_module.py")]
        rec = runner_mod.run_arm(spec, "baseline", self.dir / "y",
                                 HERE.parent.parent, 0)
        self.assertEqual(rec["status"], "failed")
        self.assertTrue(rec["failures"])

    def test_missing_binary_is_a_recorded_failure_not_a_traceback(self):
        spec = self._spec()
        spec["arms"]["baseline"].pop("stub_args")
        spec["arms"]["baseline"]["binary"] = str(self.dir / "no-such-binary")
        rec = runner_mod.run_arm(spec, "baseline", self.dir / "z",
                                 HERE.parent.parent, 0)
        self.assertEqual(rec["status"], "failed")
        self.assertIn("cannot start server", rec["failures"][0]["error"])

    def test_stale_record_in_run_dir_is_refused(self):
        spec = self._spec()
        spec["protocol"]["repeats"] = 1
        run_dir = self.dir / "stale"
        runner_mod.run_arm(spec, "baseline", run_dir, HERE.parent.parent, 0)
        other = json.loads(json.dumps(spec))
        other["objective"] = "a different preregistered hypothesis"
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            runner_mod.run_arm(other, "baseline", run_dir, HERE.parent.parent, 0)
        self.assertIn("different spec", str(ctx.exception))

    def test_rerunning_a_completed_repeat_is_refused(self):
        # The natural move after an early stop is a re-run into the same
        # directory; appending would duplicate (repeat, request) samples.
        spec = self._spec()
        spec["protocol"]["repeats"] = 2
        run_dir = self.dir / "rerun"
        runner_mod.run_arm(spec, "baseline", run_dir, HERE.parent.parent, 0)
        with self.assertRaises(runner_mod.RunnerError) as ctx:
            runner_mod.run_arm(spec, "baseline", run_dir, HERE.parent.parent, 0)
        self.assertIn("already contains repeat 0", str(ctx.exception))

    def test_arm_order_is_seeded_and_reproducible(self):
        spec = self._spec()
        self.assertEqual(runner_mod.arm_order(spec, 0), runner_mod.arm_order(spec, 0))
        orders = {tuple(runner_mod.arm_order(spec, r)) for r in range(8)}
        self.assertTrue(all(o in {("baseline", "candidate"),
                                  ("candidate", "baseline")} for o in orders))


class DemoNegativeControlTests(unittest.TestCase):
    """Issue #256: the CI demo runs IDENTICAL arms as a statistical negative
    control — a 'pass' verdict there is a false win manufactured from runner
    noise. These tests pin the demo's preregistered protocol and acceptance
    bar (run_experiment.DEMO_PROTOCOL / DEMO_ACCEPTANCE) to the property
    "identical arms + shared-runner noise => never pass", with a positive
    control proving the same bar still passes a genuinely faster candidate.
    Fully deterministic: synthetic timings, seeded worlds, seeded bootstrap."""

    # Noise calibrated against measurements from the fixture binary
    # (per-request paired improvements ~±10%, per-process repeat clusters
    # ~±7% on a QUIET box), then inflated: shared CI runners are noisier.
    CLUSTER_SD_PCT = 10.0
    REQUEST_SD_PCT = 15.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        # The real demo spec (protocol, acceptance, seed) — only the arms'
        # timings are synthetic. Binary path is inert: no process runs here.
        self.spec = cli_mod._demo_spec(self.dir / "demo", Path("/bin/true"))

    def _records(self, world, cand_gain_pct=0.0):
        """Two valid same-seal records for the demo spec with synthetic
        timings: independent per-(arm, repeat) process effects (the dominant
        shared-runner noise term), per-request jitter, and an optional true
        candidate gain (positive control)."""
        proto = self.spec["protocol"]
        repeats = proto["repeats"]
        timed = proto["requests_per_repeat"]
        per_process = 1 + proto["warmup_requests"] + timed

        def samples(role, gain_pct):
            out = []
            for r in range(repeats):
                cluster = world.gauss(0.0, self.CLUSTER_SD_PCT)
                for q in range(per_process):
                    noise = cluster + world.gauss(0.0, self.REQUEST_SD_PCT)
                    factor = (1.0 + noise / 100.0) * (1.0 - gain_pct / 100.0)
                    out.append({
                        "arm": role, "repeat": r, "request": q,
                        "cold": q == 0, "warmup": 0 < q <= proto["warmup_requests"],
                        "wall_ms": round(0.94 * factor, 4),
                    })
            return out

        seal = record_mod.spec_sha256(self.spec)
        recs = []
        for role, gain in (("baseline", 0.0), ("candidate", cand_gain_pct)):
            recs.append({
                "schema": "starling-experiment-record/1", "role": role,
                "experiment_id": self.spec["experiment_id"], "spec_sha256": seal,
                "provenance": {
                    "repo_revision": "x", "ggml_revision": "x",
                    "binary_sha256": "b" * 64,
                    "runtime": {"hardware": "cpu-only-host", "driver": "d"},
                    "workload_sha256": self.spec["workload"]["sha256"],
                    "normalizer": "none-raw-wall-time",
                    "model_claim": {"model": None, "model_sha256": None},
                    "metric_identity": {"tool": "starling-experiments",
                                        "metric": self.spec["metric"],
                                        "normalizer": "none-raw-wall-time"},
                    "commands": ["run"],
                },
                "samples": samples(role, gain), "failures": [], "status": "ok",
            })
        return recs

    def test_identical_arms_never_pass_under_runner_noise(self):
        # 40 seeded noise worlds; every world must refuse the win (inconclusive
        # or an honest noise-tipped fail — the demo accepts both, never pass).
        world = random.Random(20260922)
        verdicts = []
        for _ in range(40):
            b, c = self._records(world)
            verdicts.append(compare_mod.compare(self.spec, b, c)["verdict"])
        self.assertNotIn("pass", verdicts)
        self.assertIn("inconclusive", verdicts)  # the honest verdict dominates

    def test_one_sided_noise_window_never_passes(self):
        # Worst-case coordinated jitter: a noise window that slows ONE arm
        # uniformly across every repeat (the whole interleaved run). The
        # paired CI may sit well inside improvement territory, but the demo's
        # 12% bar must keep it below a win.
        world = random.Random(7)
        for slowdown_pct in (4.0, 8.0):
            b, c = self._records(world)
            for s in b["samples"]:
                s["wall_ms"] = round(s["wall_ms"] * (1.0 + slowdown_pct / 100.0), 4)
            verdict = compare_mod.compare(self.spec, b, c)["verdict"]
            self.assertNotEqual(verdict, "pass",
                                f"a uniform {slowdown_pct}% one-arm slowdown must not win")

    def test_real_improvement_still_passes_the_demo_bar(self):
        # Positive control: with a genuinely 30% faster candidate the same
        # protocol and bar MUST produce 'pass' — the negative control above
        # is a two-sided gate, not a comparator that refuses everything.
        world = random.Random(11)
        b, c = self._records(world, cand_gain_pct=30.0)
        result = compare_mod.compare(self.spec, b, c)
        self.assertEqual(result["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
