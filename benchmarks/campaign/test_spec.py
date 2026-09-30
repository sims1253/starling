"""Spec/profile validation, sealing and the **-glob matcher (issue #176)."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import spec  # noqa: E402
import toy as toy_mod  # noqa: E402


def valid_task(profile_id="toy--notebook", baseline=None):
    baseline = baseline or "a" * 40
    return {
        "schema": spec.TASK_SCHEMA,
        "campaign_id": "test-campaign",
        "profile": profile_id,
        "hypothesis_backlog": [],
        "baseline_revision": baseline,
        "allowed_paths": ["engine/**"],
        "protected_paths": [],
        "objective": {"gate": "perf", "metric": "improvement_pct",
                      "direction": "higher", "min_improvement": 10.0},
        "constraints": [{"gate": "correct", "metric": "transcripts_match",
                         "op": "==", "value": 1}],
        "budgets": {"max_attempts": 3, "campaign_wall_clock_s": 3600.0,
                    "attempt_wall_clock_s": 600.0, "gate_timeout_s": 60.0,
                    "agent_timeout_s": 60.0, "token_budget": None,
                    "cooldown_max_s": 30.0},
        "measurement_protocol": {"fresh_process": True, "declared_deviations": []},
        "agent": {"command": ["{python}", "agent.py"], "timeout_s": 60.0},
    }


class GlobMatcherTests(unittest.TestCase):
    def test_double_star_spans_segments_and_matches_zero(self):
        self.assertTrue(spec.glob_match("cpp/fast/**", "cpp/fast"))
        self.assertTrue(spec.glob_match("cpp/fast/**", "cpp/fast/a/b.c"))
        self.assertFalse(spec.glob_match("cpp/fast/**", "cpp/fastx/a.c"))
        self.assertFalse(spec.glob_match("cpp/fast/**", "cpp/other/a.c"))

    def test_star_stays_within_a_segment(self):
        self.assertTrue(spec.glob_match("benchmarks/fast_engine/phone_*.sh",
                                        "benchmarks/fast_engine/phone_gates.sh"))
        self.assertFalse(spec.glob_match("benchmarks/fast_engine/phone_*.sh",
                                         "benchmarks/fast_engine/sub/phone_gates.sh"))
        self.assertFalse(spec.glob_match("benchmarks/fast_engine/phone_*.sh",
                                         "benchmarks/fast_engine/other.sh"))

    def test_exact_and_prefix(self):
        self.assertTrue(spec.glob_match("engine/config.json", "engine/config.json"))
        self.assertFalse(spec.glob_match("engine/config.json", "engine/config.jsonx"))
        self.assertTrue(spec.glob_match("**/evaluator/**", "a/b/evaluator/c/d.py"))

    def test_question_and_class(self):
        self.assertTrue(spec.glob_match("a/?.py", "a/x.py"))
        self.assertFalse(spec.glob_match("a/?.py", "a/xy.py"))
        self.assertTrue(spec.glob_match("a/[abc].py", "a/b.py"))
        self.assertFalse(spec.glob_match("a/[abc].py", "a/d.py"))


class ProtectionTests(unittest.TestCase):
    def test_always_protected_included(self):
        profile = {"trusted_paths": ["evaluator"]}
        task = {"protected_paths": ["mydir/**"]}
        protected = spec.effective_protected(profile, task)
        for p in spec.ALWAYS_PROTECTED + ("evaluator", "evaluator/**", "mydir/**"):
            self.assertIn(p, protected)
        violations = spec.path_violations(
            ["benchmarks/campaign/gates/evil.sh", "benchmarks/experiments/record.py",
             "benchmarks/fast_engine/phone_gates.sh", "evaluator/gate_correct.py",
             ".github/workflows/evil.yml", "tests/fixtures/x.wav"],
            allowed=["**"], protected=protected)
        self.assertEqual(len(violations), 6)

    def test_allowed_region(self):
        self.assertEqual(spec.path_violations(["engine/config.json", "engine/x.py"],
                                              ["engine/**"], []), [])
        self.assertEqual(spec.path_violations(["cpp/serve/main.cpp"], ["engine/**"], []),
                         ["cpp/serve/main.cpp"])


class TaskValidationTests(unittest.TestCase):
    def test_valid_task(self):
        self.assertEqual(spec.validate_task(valid_task()), [])

    def test_short_baseline_rejected(self):
        task = valid_task(baseline="abc123")
        problems = spec.validate_task(task)
        self.assertTrue(any("baseline_revision" in p for p in problems))

    def test_budget_bounds(self):
        task = valid_task()
        task["budgets"]["max_attempts"] = 0
        task["budgets"]["campaign_wall_clock_s"] = 0
        problems = spec.validate_task(task)
        self.assertTrue(any("max_attempts" in p for p in problems))
        self.assertTrue(any("campaign_wall_clock_s" in p for p in problems))

    def test_bad_constraint_op(self):
        task = valid_task()
        task["constraints"][0]["op"] = "~="
        self.assertTrue(any("op" in p for p in spec.validate_task(task)))

    def test_cross_validation_against_profile(self):
        profile = toy_mod.toy_profile(Path("."))
        task = valid_task()
        task["objective"]["gate"] = "nonexistent"
        problems = spec.validate_task(task, profile)
        self.assertTrue(any("not a gate" in p for p in problems))
        task["objective"]["gate"] = "perf"
        self.assertEqual(spec.validate_task(task, profile), [])


class ProfileValidationTests(unittest.TestCase):
    def test_toy_profile_valid(self):
        self.assertEqual(spec.validate_profile(toy_mod.toy_profile(Path("."))), [])

    def test_all_committed_profiles_validate(self):
        profiles = spec.load_profiles(HERE / "profiles")
        self.assertGreaterEqual(len(profiles), 17)
        for pid, profile in profiles.items():
            self.assertEqual(spec.validate_profile(profile), [], pid)

    def test_blocked_profile_needs_reason(self):
        profile = toy_mod.toy_profile(Path("."))
        profile["status"] = "blocked"
        profile.pop("blocked_on", None)
        self.assertTrue(any("blocked" in p for p in spec.validate_profile(profile)))

    def test_unavailable_workload_needs_owner(self):
        profile = toy_mod.toy_profile(Path("."))
        profile["workloads"].append({"id": "ghost", "description": "x",
                                     "status": "unavailable"})
        self.assertTrue(any("ghost" in p for p in spec.validate_profile(profile)))


class GatePhaseTests(unittest.TestCase):
    def gate(self, **kw):
        g = {"name": "g", "stage": "resource", "argv": ["true"], "rules": []}
        g.update(kw)
        return g

    def test_finalize_phase_must_be_a_non_deciding_measurement(self):
        self.assertEqual(spec.validate_gate(self.gate(phase="finalize", required=False)), [])
        self.assertTrue(spec.validate_gate(self.gate(phase="finalize")))  # required defaults True
        self.assertTrue(spec.validate_gate(self.gate(phase="finalize", required=False,
                                                     objective=True)))
        self.assertTrue(spec.validate_gate(self.gate(phase="nightly")))


class SealTests(unittest.TestCase):
    def test_seal_changes_with_content_not_key_order(self):
        a = valid_task()
        b = copy.deepcopy(a)
        self.assertEqual(spec.seal_of(a), spec.seal_of(b))
        b["campaign_id"] = "other"
        self.assertNotEqual(spec.seal_of(a), spec.seal_of(b))

    def test_hash_tree_detects_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "a" / "f.txt").write_text("one", encoding="utf-8")
            h1 = spec.hash_tree(root)
            (root / "a" / "f.txt").write_text("two", encoding="utf-8")
            self.assertNotEqual(h1, spec.hash_tree(root))
            (root / "a" / "g.txt").write_text("x", encoding="utf-8")
            self.assertNotEqual(h1, spec.hash_tree(root))


if __name__ == "__main__":
    unittest.main()
