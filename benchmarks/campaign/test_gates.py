"""Gate execution: METRIC parsing, rules, verdicts, credential scrubbing,
timeouts and process-group cleanup (issue #176 §3)."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gates as gates_mod  # noqa: E402


def _child_gone(pid: int) -> bool:
    """True when the pid is dead OR a zombie. On hosts whose PID 1 does not
    reap adopted orphans (some containers), a killed grandchild stays in
    state Z forever — os.kill(pid, 0) alone would never confirm its death."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
        return state == "Z"
    except (OSError, IndexError):
        return True


class ParseOutputTests(unittest.TestCase):
    def test_numbers_and_strings(self):
        metrics, div = gates_mod.parse_output(
            "METRIC n=3\nMETRIC f=1.5\nMETRIC v=pass\nMETRIC u=unavailable\n"
            "DIVERGENCE: first differing char at 3: expected 'a' got 'b'\nnoise\n")
        self.assertEqual(metrics, {"n": 3, "f": 1.5, "v": "pass", "u": "unavailable"})
        self.assertEqual(div, ["first differing char at 3: expected 'a' got 'b'"])

    def test_later_metric_wins(self):
        metrics, _ = gates_mod.parse_output("METRIC x=0\nMETRIC x=1\n")
        self.assertEqual(metrics["x"], 1)

    def test_no_metrics(self):
        self.assertEqual(gates_mod.parse_output("nothing here\n"), ({}, []))


class RuleTests(unittest.TestCase):
    RULES = [{"metric": "m", "op": ">=", "value": 5}]

    def test_pass_fail(self):
        v, _ = gates_mod.evaluate_rules({"m": 7}, self.RULES)
        self.assertEqual(v, "pass")
        v, _ = gates_mod.evaluate_rules({"m": 4}, self.RULES)
        self.assertEqual(v, "fail")

    def test_missing_or_unavailable_is_inconclusive_never_pass(self):
        v, _ = gates_mod.evaluate_rules({}, self.RULES)
        self.assertEqual(v, "inconclusive")
        v, _ = gates_mod.evaluate_rules({"m": "unavailable"}, self.RULES)
        self.assertEqual(v, "inconclusive")

    def test_string_vs_number_incomparable(self):
        v, _ = gates_mod.evaluate_rules({"m": "pass"}, self.RULES)
        self.assertEqual(v, "inconclusive")

    def test_string_equality(self):
        rules = [{"metric": "verdict", "op": "==", "value": "pass"}]
        self.assertEqual(gates_mod.evaluate_rules({"verdict": "pass"}, rules)[0], "pass")
        self.assertEqual(gates_mod.evaluate_rules({"verdict": "fail"}, rules)[0], "fail")

    def test_ordering_only_for_numbers(self):
        rules = [{"metric": "v", "op": "<", "value": "z"}]
        self.assertEqual(gates_mod.evaluate_rules({"v": "a"}, rules)[0], "inconclusive")


class SubstituteTests(unittest.TestCase):
    def test_only_known_placeholders(self):
        out = gates_mod.substitute("awk '{print $1}' {worktree}", {"worktree": "/w"})
        self.assertEqual(out, "awk '{print $1}' /w")

    def test_argv_expands_env_when_asked(self):
        os.environ["CAMP_TEST_VAR"] = "/models"
        self.addCleanup(os.environ.pop, "CAMP_TEST_VAR", None)
        argv = gates_mod.substitute_command(
            ["{python}", "g.py", "${CAMP_TEST_VAR}/m.gguf"],
            {"python": "py"}, expand_vars=True)
        self.assertEqual(argv, ["py", "g.py", "/models/m.gguf"])
        # shell strings keep their $-syntax for the shell
        shell = gates_mod.substitute_command("echo ${CAMP_TEST_VAR} {python}",
                                             {"python": "py"}, expand_vars=False)
        self.assertEqual(shell, "echo ${CAMP_TEST_VAR} py")


class EnvScrubbingTests(unittest.TestCase):
    BASE = {
        "PATH": "/usr/bin", "HOME": "/home/x",
        "OPENAI_API_KEY": "sk", "ANTHROPIC_API_KEY": "sk", "GH_TOKEN": "t",
        "GITHUB_TOKEN": "t", "AWS_SECRET_ACCESS_KEY": "k", "MY_PASSWORD": "p",
        "RELEASE_KEYSTORE": "k", "APK_SIGNING_KEY": "k", "ACTIONS_ID_TOKEN_REQUEST_URL": "u",
        "STARLING_HELDOUT_DIR": "/heldout", "GEMINI_API_KEY": "g",
    }

    def test_gate_env_never_writes_bytecode_into_the_trusted_tree(self):
        env, _removed = gates_mod.gate_env({"PATH": "/usr/bin"})
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_gate_env_loses_all_credentials_and_heldout(self):
        env, removed = gates_mod.gate_env(self.BASE, ["STARLING_HELDOUT_DIR"])
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("STARLING_HELDOUT_DIR", env)
        self.assertIn("PATH", env)
        self.assertIn("HOME", env)
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GH_TOKEN", "GITHUB_TOKEN",
                     "AWS_SECRET_ACCESS_KEY", "MY_PASSWORD", "RELEASE_KEYSTORE",
                     "APK_SIGNING_KEY", "ACTIONS_ID_TOKEN_REQUEST_URL",
                     "STARLING_HELDOUT_DIR", "GEMINI_API_KEY"):
            self.assertIn(name, removed)
        self.assertNotIn("PATH", removed)

    def test_agent_env_keeps_provider_keys_loses_release_and_heldout(self):
        env, removed = gates_mod.agent_env(self.BASE, ["STARLING_HELDOUT_DIR"])
        self.assertIn("ANTHROPIC_API_KEY", env)   # orchestration boundary
        self.assertIn("OPENAI_API_KEY", env)
        self.assertNotIn("GH_TOKEN", env)
        self.assertNotIn("RELEASE_KEYSTORE", env)
        self.assertNotIn("STARLING_HELDOUT_DIR", env)
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "RELEASE_KEYSTORE", "APK_SIGNING_KEY",
                     "ACTIONS_ID_TOKEN_REQUEST_URL", "STARLING_HELDOUT_DIR"):
            self.assertIn(name, removed)
        self.assertNotIn("ANTHROPIC_API_KEY", removed)


class RunGateTests(unittest.TestCase):
    VALUES = {"worktree": str(HERE)}

    def _gate(self, shell, rules=(), **kw):
        gate = {"name": "g", "stage": "diagnostic", "shell": shell,
                "rules": list(rules), "required": True}
        gate.update(kw)
        return gate

    def test_exit_zero_with_rules_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "g.log"
            rec = gates_mod.run_gate(
                self._gate("echo METRIC m=7; exit 0",
                           rules=[{"metric": "m", "op": ">=", "value": 5}]),
                self.VALUES, dict(os.environ), log)
            self.assertEqual(rec["verdict"], "pass")
            self.assertEqual(rec["metrics"], {"m": 7})
            self.assertIsNotNone(rec["resource"]["maxrss_kb"])

    def test_exit_nonzero_fails_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "g.log"
            rec = gates_mod.run_gate(self._gate("echo boom >&2; exit 1"),
                                     self.VALUES, dict(os.environ), log)
            self.assertEqual(rec["verdict"], "fail")
            self.assertTrue(any("boom" in d for d in rec["rule_details"]))

    def test_exit_three_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "g.log"
            rec = gates_mod.run_gate(
                self._gate("echo METRIC energy=unavailable; exit 3",
                           rules=[{"metric": "energy", "op": "<", "value": 5}]),
                self.VALUES, dict(os.environ), log)
            self.assertEqual(rec["verdict"], "inconclusive")

    def test_missing_metric_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "g.log"
            rec = gates_mod.run_gate(
                self._gate("exit 0", rules=[{"metric": "absent", "op": "==", "value": 1}]),
                self.VALUES, dict(os.environ), log)
            self.assertEqual(rec["verdict"], "inconclusive")

    def test_remote_cleanup_runs_after_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "cleaned"
            log = Path(tmp) / "g.log"
            rec = gates_mod.run_gate(
                self._gate("sleep 60", timeout_s=0.5,
                           remote_cleanup={"shell": f"touch {marker}"}),
                self.VALUES, dict(os.environ), log)
            self.assertEqual(rec["verdict"], "fail")
            self.assertTrue(rec["timed_out"])
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.exists())

    def test_timeout_kills_the_whole_process_group(self):
        # A gate spawning a grandchild `sleep 1000` with a 1 s timeout must
        # not leave the grandchild behind (issue #176 acceptance).
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "grandchild.pid"
            gate_py = Path(tmp) / "gate.py"
            gate_py.write_text(
                "import subprocess, sys, time\n"
                "g = subprocess.Popen(['sleep', '1000'])\n"
                f"open({str(pidfile)!r}, 'w').write(str(g.pid))\n"
                "time.sleep(100)\n", encoding="utf-8")
            log = Path(tmp) / "g.log"
            rec = gates_mod.run_gate(
                self._gate(f"{sys.executable} {gate_py}", timeout_s=1.0),
                self.VALUES, dict(os.environ), log)
            self.assertEqual(rec["verdict"], "fail")
            self.assertTrue(rec["timed_out"])
            pid = int(pidfile.read_text().strip())
            deadline = time.monotonic() + 10
            gone = False
            while time.monotonic() < deadline:
                if _child_gone(pid):
                    gone = True
                    break
                time.sleep(0.05)
            self.assertTrue(gone, "grandchild survived the gate timeout")
            # no separate PID re-check: the loop above already confirmed it is
            # gone, and a re-check can flake if the OS recycled the PID.


class VerdictTests(unittest.TestCase):
    def test_ordered_gates_fixed_stage_order(self):
        gates = [
            {"name": "perf", "stage": "perf"},
            {"name": "correct", "stage": "correctness"},
            {"name": "diag", "stage": "diagnostic"},
        ]
        self.assertEqual([g["name"] for g in gates_mod.ordered_gates(gates)],
                         ["correct", "diag", "perf"])

    def test_attempt_verdict_requires_required_gates(self):
        ok = {"verdict": "pass", "required": True, "objective": True}
        skipped = {"verdict": "skipped", "required": True, "objective": False}
        self.assertEqual(gates_mod.attempt_verdict([ok]), "pass")
        self.assertEqual(gates_mod.attempt_verdict([ok, skipped]), "pass")
        self.assertEqual(gates_mod.attempt_verdict([{"verdict": "fail"}]), "fail")
        self.assertEqual(gates_mod.attempt_verdict(
            [{"verdict": "inconclusive", "required": True}]), "inconclusive")
        # a non-required inconclusive gate does not block a pass
        self.assertEqual(gates_mod.attempt_verdict(
            [ok, {"verdict": "inconclusive", "required": False}]), "pass")


if __name__ == "__main__":
    unittest.main()
