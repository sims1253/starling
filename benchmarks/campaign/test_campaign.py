"""Campaign runner acceptance tests (issue #176 §7): the hermetic toy pilot
end-to-end, negative controls, budgets, seals/tamper, interrupts/resume,
safety stops, preview/list behavior. CPU-only, no network, no models."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import campaign as campaign_mod  # noqa: E402
import monitors  # noqa: E402
import spec  # noqa: E402
import toy as toy_mod  # noqa: E402

SLOW_AGENT = r"""
import os, subprocess, sys, time
n = int(os.environ.get("CAMPAIGN_ATTEMPT", "1"))
d = os.environ["CAMPAIGN_ATTEMPT_DIR"]
if n == 1:
    g = subprocess.Popen(["sleep", "1000"])
    open(os.path.join(d, "agent.pid"), "w").write(str(os.getpid()))
    open(os.path.join(d, "grandchild.pid"), "w").write(str(g.pid))
    time.sleep(100)
else:
    t = open("engine/transcribe.py").read()
    open("engine/transcribe.py", "w").write(t.replace("time.sleep", "time . sleep", 1))
    open(os.path.join(d, "hypothesis.json"), "w").write(
        '{"hypothesis": "whitespace only, no effect"}')
"""

NOOP_AGENT = r"""
import json, os
d = os.environ["CAMPAIGN_ATTEMPT_DIR"]
open(os.path.join(d, "hypothesis.json"), "w").write(
    json.dumps({"hypothesis": "whitespace only, no effect"}))
t = open("engine/transcribe.py").read()
open("engine/transcribe.py", "w").write(t.replace("time.sleep", "time . sleep", 1))
"""

FAIL_AGENT = "import sys; sys.exit(1)\n"


class ToyCampaignTest(unittest.TestCase):
    """Starts real toy campaigns through the real CLI code paths."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="campaign-test-"))
        self.lock_path = self.tmp / "campaign.lock"
        self._old_env = os.environ.get("STARLING_CAMPAIGN_LOCK")
        os.environ["STARLING_CAMPAIGN_LOCK"] = str(self.lock_path)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self._old_env is None:
            os.environ.pop("STARLING_CAMPAIGN_LOCK", None)
        else:
            os.environ["STARLING_CAMPAIGN_LOCK"] = self._old_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def start_campaign(self, *, task_overrides=None, agent_command="__default__",
                       max_attempts=None, run=True, agent_null=False):
        toy_repo = self.tmp / "toyrepo"
        baseline = toy_mod.build_toy_repo(toy_repo)
        profile = toy_mod.toy_profile(toy_repo)
        profile_dir = self.tmp / "profile"
        profile_dir.mkdir(exist_ok=True)
        (profile_dir / f"{profile['id']}.json").write_text(
            json.dumps(profile, indent=2), encoding="utf-8")
        if agent_command == "__default__":
            agent_command = ["{python}", str(HERE / "toy_agent.py")]
        task = toy_mod.toy_task(profile["id"], baseline, agent_command)
        if agent_null:
            task["agent"] = None
        if max_attempts is not None:
            task["budgets"]["max_attempts"] = max_attempts
        task.update(task_overrides or {})
        task_path = self.tmp / "task.json"
        task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
        out = self.tmp / "campaign"
        rc = campaign_mod.main([
            "start", "--profile-file", str(profile_dir / f"{profile['id']}.json"),
            "--repo", str(toy_repo), "--baseline", baseline,
            "--out", str(out), "--task", str(task_path)])
        self.assertEqual(rc, 0, "start must succeed")
        if run:
            rc = campaign_mod.main(["run", "--campaign", str(out)])
        return out, rc

    def ledger(self, out: Path) -> list[dict]:
        return [json.loads(l) for l in
                (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l]

    def state(self, out: Path) -> dict:
        return json.loads((out / "state.json").read_text(encoding="utf-8"))


class PilotTests(ToyCampaignTest):
    def test_pilot_end_to_end(self):
        out = self.tmp / "pilot"
        with mock.patch.dict(os.environ, {"STARLING_CAMPAIGN_LOCK":
                                          str(self.tmp / "pilot.lock")}):
            rc = campaign_mod.main(["pilot", "--out", str(out)])
        self.assertEqual(rc, 0)
        camp = spec.load_json(out / "campaign.json")
        self.assertNotEqual(camp["best_sha"], camp["baseline_revision"])
        ledger = self.ledger(out)
        self.assertEqual([e["kind"] for e in ledger],
                         ["control"] + ["attempt"] * 5)
        self.assertEqual([e["verdict"] for e in ledger[1:]],
                         ["pass", "fail", "fail", "fail", "fail"])
        kept = [e for e in ledger[1:] if e["kept"]]
        self.assertEqual([e["n"] for e in kept], [1])
        report = (out / "report.md").read_text(encoding="utf-8")
        for needle in ("Exact rerun commands", "gh pr create --draft",
                       "transcripts_match", "first differing char"):
            self.assertIn(needle, report)

    def test_pilot_is_deterministic(self):
        verdicts = []
        for i in range(2):
            out = self.tmp / f"pilot-{i}"
            with mock.patch.dict(os.environ, {"STARLING_CAMPAIGN_LOCK":
                                              str(self.tmp / f"pilot-{i}.lock")}):
                rc = campaign_mod.main(["pilot", "--out", str(out)])
            self.assertEqual(rc, 0)
            verdicts.append([(e["verdict"],
                              next((g["stage"] for g in e["gates"]
                                    if g.get("verdict") == "fail"), None))
                             for e in self.ledger(out)])
        self.assertEqual(verdicts[0], verdicts[1])


class CliTests(ToyCampaignTest):
    def test_list_shows_all_profiles_with_status(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = campaign_mod.main(["list"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("17 profiles", text)
        for pid in ("parakeet--pixel", "moss-diarize--notebook",
                    "parakeet-unified--pixel", "fixture--notebook-cpu"):
            self.assertIn(pid, text)
        self.assertIn("blocked (native-support)", text)

    def test_preview_ready_toy_ok_and_mutates_nothing(self):
        toy_repo = self.tmp / "toyrepo"
        baseline = toy_mod.build_toy_repo(toy_repo)
        profile = toy_mod.toy_profile(toy_repo)
        profile_dir = self.tmp / "profile"
        profile_dir.mkdir()
        (profile_dir / f"{profile['id']}.json").write_text(
            json.dumps(profile, indent=2), encoding="utf-8")
        out = self.tmp / "would-be-campaign"
        before = sorted(p.relative_to(self.tmp).as_posix()
                        for p in self.tmp.rglob("*"))
        import contextlib
        import io
        buf = io.StringIO()
        lock_dir = Path(tempfile.mkdtemp(prefix="campaign-locktest-"))
        self.addCleanup(lambda: shutil.rmtree(lock_dir, ignore_errors=True))
        lock_elsewhere = str(lock_dir / "campaign.lock")
        with contextlib.redirect_stdout(buf), \
                mock.patch.dict(os.environ, {"STARLING_CAMPAIGN_LOCK": lock_elsewhere}):
            rc = campaign_mod.main([
                "preview", "--profile-file", str(profile_dir / f"{profile['id']}.json"),
                "--repo", str(toy_repo), "--baseline", baseline, "--out", str(out)])
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn("ready", buf.getvalue())
        self.assertIn("gates (fixed stage order, fail-fast)", buf.getvalue())
        after = sorted(p.relative_to(self.tmp).as_posix()
                       for p in self.tmp.rglob("*"))
        self.assertEqual(before, after, "preview must mutate nothing")

    def test_preview_missing_prereqs_exits_2_without_mutating(self):
        # adb presence varies by machine, so force a prerequisite that cannot
        # exist anywhere: a GGUF artifact under a nonexistent directory.
        profile = copy.deepcopy(spec.load_profiles(HERE / "profiles")["parakeet--pixel"])
        profile["artifacts"] = [
            dict(a, path="/nonexistent-starling-test/model.gguf")
            for a in profile.get("artifacts", [])
        ]
        profile_dir = self.tmp / "missing-prereq-profiles"
        profile_dir.mkdir()
        profile_path = profile_dir / f"{profile['id']}.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        out = self.tmp / "never"
        before = sorted(p.name for p in self.tmp.iterdir())
        import contextlib
        import io
        buf = io.StringIO()
        lock_dir = Path(tempfile.mkdtemp(prefix="campaign-locktest-"))
        self.addCleanup(lambda: shutil.rmtree(lock_dir, ignore_errors=True))
        lock_elsewhere = str(lock_dir / "campaign.lock")
        with contextlib.redirect_stdout(buf), \
                mock.patch.dict(os.environ, {"STARLING_CAMPAIGN_LOCK": lock_elsewhere}):
            rc = campaign_mod.main([
                "preview", "--profile-file", str(profile_path),
                "--baseline", "HEAD", "--out", str(out)])
        self.assertEqual(rc, 2)
        self.assertIn("prerequisites missing", buf.getvalue())
        self.assertIn("artifact gguf missing", buf.getvalue())
        after = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(before, after)

    def test_preflight_lists_missing_adb_and_artifact(self):
        profiles = spec.load_profiles(HERE / "profiles")
        profile = profiles["parakeet--pixel"]
        task = campaign_mod.generate_task(
            profile, "a" * 40, mock.Mock(max_attempts=None, wall_clock=None,
                                         allowed=None, agent_cmd=None))
        task["baseline_revision"] = "a" * 40
        fake = _FakeProbeNoAdb()
        # adb being installed on THIS machine must not silence the check:
        # force the pixel tool requirement to fail deterministically.
        real_which = shutil.which

        def no_adb(cmd, path=None):
            return None if cmd == "adb" else real_which(cmd, path=path)

        with mock.patch.object(campaign_mod.shutil, "which", no_adb):
            problems, _ctx = campaign_mod.preflight(
                profile, task, Path("."), self.tmp / "x", probe=fake)
        joined = "\n".join(problems)
        self.assertIn("required tool 'adb' not on PATH", joined)
        self.assertIn("artifact gguf missing", joined)

    def test_blocked_profile_refused_by_preview_and_start(self):
        for cmd in (["preview", "--profile", "moss-diarize--notebook",
                     "--baseline", "HEAD", "--out", str(self.tmp / "o")],
                    ["start", "--profile", "moss-diarize--notebook",
                     "--baseline", "HEAD", "--out", str(self.tmp / "o2")]):
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = campaign_mod.main(cmd)
            self.assertEqual(rc, 2, cmd)
            self.assertIn("blocked on native-support", buf.getvalue())
        self.assertFalse((self.tmp / "o2").exists(), "start must not create anything")

    def test_start_refuses_when_control_cannot_confirm_the_baseline(self):
        real_profile = toy_mod.toy_profile

        def undecidable(repo):
            prof = real_profile(repo)
            prof["gates"][0]["argv"] = [
                "{python}", "-c",
                "import sys; print('METRIC transcripts_match=unavailable'); sys.exit(3)"]
            return prof

        with mock.patch.object(toy_mod, "toy_profile", undecidable):
            toy_repo = self.tmp / "toyrepo"
            baseline = toy_mod.build_toy_repo(toy_repo)
            profile = toy_mod.toy_profile(toy_repo)
        profile_path = self.tmp / f"{profile['id']}.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        task = toy_mod.toy_task(profile["id"], baseline,
                                ["{python}", str(HERE / "toy_agent.py")])
        task_path = self.tmp / "t.json"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        out = self.tmp / "campaign"
        rc = campaign_mod.main(["start", "--profile-file", str(profile_path),
                                "--repo", str(toy_repo), "--baseline", baseline,
                                "--out", str(out), "--task", str(task_path)])
        self.assertEqual(rc, 2)
        self.assertEqual(self.state(out)["status"], "control_failed")
        for cmd in (["run"], ["resume"], ["attempt", "--hypothesis", "x"]):
            self.assertEqual(campaign_mod.main([cmd[0], "--campaign", str(out), *cmd[1:]]), 2)
        self.assertFalse((out / "attempts" / "001").exists(), "no attempt may run")

    def test_start_refuses_existing_nonempty_out(self):
        out, _rc = self.start_campaign(run=False)
        # a fresh start into the existing non-empty campaign dir must refuse
        toy_repo = self.tmp / "toyrepo"
        baseline = subprocess.run(
            ["git", "-C", str(toy_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        profile_dir = self.tmp / "profile"
        rc = campaign_mod.main([
            "start", "--profile-file",
            str(profile_dir / "toy--notebook.json"),
            "--repo", str(toy_repo), "--baseline", baseline, "--out", str(out)])
        self.assertEqual(rc, 2)

    def test_run_without_agent_command_is_a_usage_error(self):
        out, rc = self.start_campaign(agent_null=True, run=True)
        self.assertEqual(rc, 2)


class SealAndTamperTests(ToyCampaignTest):
    def test_changed_sealed_spec_refused_by_resume_and_report(self):
        out, _rc = self.start_campaign(run=False, max_attempts=1)
        sealed = out / "task.sealed.json"
        task = json.loads(sealed.read_text(encoding="utf-8"))
        task["budgets"]["max_attempts"] = 99  # post-hoc rule edit
        sealed.write_text(json.dumps(task, indent=2), encoding="utf-8")
        for cmd in (["resume", "--campaign", str(out)], ["report", "--campaign", str(out)]):
            rc = campaign_mod.main(cmd)
            self.assertEqual(rc, 4, cmd)
        # the campaign state is untouched by the refusal (a human must inspect)
        self.assertEqual(self.state(out)["attempts_used"], 0)

    def test_evaluator_tamper_stops_run_with_exit_4(self):
        out, _rc = self.start_campaign(run=False, max_attempts=3)
        victim = out / "trusted" / "evaluator" / "input.txt"
        if not (hasattr(os, "geteuid") and os.geteuid() == 0):
            # root ignores permission bits, so the write succeeds there
            with self.assertRaises(PermissionError, msg="trusted files are read-only"):
                victim.write_text("tampered corpus\n", encoding="utf-8")
        victim.chmod(0o644)  # a determined tamperer: still detected by the hash
        victim.write_text("tampered corpus\n", encoding="utf-8")
        rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 4)
        state = self.state(out)
        self.assertEqual(state["stop_reason"], "evaluator_tampered")
        self.assertEqual(state["attempts_used"], 0, "no attempt may run on a tampered evaluator")
        # resume must also refuse until a human inspects
        rc = campaign_mod.main(["resume", "--campaign", str(out)])
        self.assertEqual(rc, 4)

    def test_evaluator_tamper_mid_run_stops_before_next_attempt(self):
        out, _rc = self.start_campaign(run=False, max_attempts=3)
        # let attempt 1 run, then tamper before attempt 2
        real_run_attempt = campaign_mod.run_attempt
        box = {"tampered": False}

        def run_attempt_then_tamper(camp, **kw):
            entry = real_run_attempt(camp, **kw)
            if entry["n"] == 1 and not box["tampered"]:
                victim = camp.dir / "trusted" / "evaluator" / "input.txt"
                victim.chmod(0o644)
                victim.write_text("tampered\n", encoding="utf-8")
                box["tampered"] = True
            return entry

        with mock.patch.object(campaign_mod, "run_attempt", run_attempt_then_tamper):
            rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 4)
        self.assertEqual(self.state(out)["attempts_used"], 1)


class BudgetTests(ToyCampaignTest):
    def test_max_attempts_bounds_the_loop(self):
        out, rc = self.start_campaign(max_attempts=2)
        self.assertEqual(rc, 0)
        state = self.state(out)
        self.assertEqual(state["stop_reason"], "max_attempts")
        self.assertEqual(state["attempts_used"], 2)
        self.assertEqual(len(self.ledger(out)), 3)  # control + 2 attempts

    def test_wall_clock_budget_stops_before_any_attempt(self):
        out, rc = self.start_campaign(run=False, task_overrides={
            "budgets": {"max_attempts": 5, "campaign_wall_clock_s": 0.001,
                        "attempt_wall_clock_s": 600.0, "gate_timeout_s": 120.0,
                        "agent_timeout_s": 120.0, "token_budget": None,
                        "cooldown_max_s": 60.0}})
        rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        state = self.state(out)
        self.assertEqual(state["stop_reason"], "wall_clock")
        self.assertEqual(state["attempts_used"], 0)

    def test_token_budget_stops_the_loop(self):
        out, rc = self.start_campaign(run=False, task_overrides={
            "budgets": {"max_attempts": 5, "campaign_wall_clock_s": 3600.0,
                        "attempt_wall_clock_s": 600.0, "gate_timeout_s": 120.0,
                        "agent_timeout_s": 120.0, "token_budget": 60000,
                        "cooldown_max_s": 60.0}})
        rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        state = self.state(out)
        self.assertEqual(state["stop_reason"], "token_budget")
        self.assertEqual(state["attempts_used"], 2)  # 2 x 55000 >= 60000

    def test_agent_failure_streak_stops(self):
        agent = self.tmp / "fail_agent.py"
        agent.write_text(FAIL_AGENT, encoding="utf-8")
        out, rc = self.start_campaign(
            agent_command=["{python}", str(agent)],
            task_overrides={"budgets": {
                "max_attempts": 10, "campaign_wall_clock_s": 3600.0,
                "attempt_wall_clock_s": 60.0, "gate_timeout_s": 60.0,
                "agent_timeout_s": 30.0, "token_budget": None,
                "cooldown_max_s": 60.0}})
        self.assertEqual(rc, 0)
        state = self.state(out)
        self.assertEqual(state["stop_reason"], "agent_error")
        self.assertEqual(state["attempts_used"], 3)
        self.assertEqual(state["agent_consecutive_failures"], 3)


class AttemptModeTests(ToyCampaignTest):
    def test_evaluator_commit_needs_no_identity_and_skips_hooks(self):
        out, _rc = self.start_campaign(agent_null=True, run=False)
        toy_repo = self.tmp / "toyrepo"
        for key in ("user.name", "user.email"):
            subprocess.run(["git", "-C", str(toy_repo), "config", "--unset", key], check=True)
        hooks = toy_repo / ".git" / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        hook = hooks / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)
        cfg = out / "worktree" / "engine" / "config.json"
        cfg.write_text(json.dumps({"WORK_MS": 20}, indent=2), encoding="utf-8")
        # CI runners have no global identity: hide the developer's.
        with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": os.devnull,
                                          "GIT_CONFIG_NOSYSTEM": "1"}):
            rc = campaign_mod.main(["attempt", "--campaign", str(out),
                                    "--hypothesis", "WORK_MS 60->20"])
        self.assertEqual(rc, 0)
        self.assertTrue(self.ledger(out)[-1]["kept"],
                        "a real change must not degrade to no_change without an identity")
        local = subprocess.run(["git", "-C", str(toy_repo), "config", "--local",
                                "--get", "user.name"], capture_output=True, text=True)
        self.assertEqual(local.stdout.strip(), "",
                         "the campaign must never write the shared repository config")

    def test_attempt_evaluates_current_worktree_change(self):
        out, _rc = self.start_campaign(agent_null=True, run=False)
        worktree = out / "worktree"
        cfg = worktree / "engine" / "config.json"
        cfg.write_text(json.dumps({"WORK_MS": 20}, indent=2), encoding="utf-8")
        rc = campaign_mod.main(["attempt", "--campaign", str(out),
                                "--hypothesis", "manual WORK_MS 60->20"])
        self.assertEqual(rc, 0)
        ledger = self.ledger(out)
        self.assertEqual(ledger[-1]["verdict"], "pass")
        self.assertTrue(ledger[-1]["kept"])
        # the change is committed and stays
        status = subprocess.run(["git", "-C", str(worktree), "status", "--porcelain"],
                                capture_output=True, text=True).stdout
        self.assertEqual(status.strip(), "")

        # a failing manual attempt reverts to the kept best
        cfg.write_text(json.dumps({"WORK_MS": 20, "SUFFIX": "!"}, indent=2),
                       encoding="utf-8")
        rc = campaign_mod.main(["attempt", "--campaign", str(out),
                                "--hypothesis", "incorrect suffix"])
        self.assertEqual(rc, 0)
        ledger = self.ledger(out)
        self.assertEqual(ledger[-1]["verdict"], "fail")
        content = json.loads(cfg.read_text(encoding="utf-8"))
        self.assertNotIn("SUFFIX", content, "failed attempt must be reverted")
        # fail-fast: the perf gate never ran on the incorrect candidate
        by_name = {g["name"]: g for g in ledger[-1]["gates"]}
        self.assertEqual(by_name["correct"]["verdict"], "fail")
        self.assertEqual(by_name["perf"]["verdict"], "skipped")
        self.assertIn("skipped", " ".join(by_name["perf"]["rule_details"]))

    def test_attempt_records_no_change_as_inconclusive(self):
        out, _rc = self.start_campaign(agent_null=True, run=False)
        rc = campaign_mod.main(["attempt", "--campaign", str(out),
                                "--hypothesis", "nothing changed"])
        self.assertEqual(rc, 0)
        ledger = self.ledger(out)
        self.assertEqual(ledger[-1]["verdict"], "inconclusive")
        self.assertEqual(ledger[-1]["reason"], "no_change")
        # no gates ran and nothing was kept
        self.assertEqual(ledger[-1]["gates"][0]["name"], "authority")
        self.assertFalse(ledger[-1]["kept"])

    def test_attempt_rejects_protected_path_edits_at_authority(self):
        out, _rc = self.start_campaign(agent_null=True, run=False)
        victim = out / "worktree" / "evaluator" / "gate_correct.py"
        victim.write_text(
            victim.read_text(encoding="utf-8")
            + '\nprint("METRIC transcripts_match=1")  # agent-injected\n',
            encoding="utf-8")
        rc = campaign_mod.main(["attempt", "--campaign", str(out),
                                "--hypothesis", "skip validation"])
        self.assertEqual(rc, 0)
        ledger = self.ledger(out)
        entry = ledger[-1]
        self.assertEqual(entry["verdict"], "fail")
        authority = next(g for g in entry["gates"] if g["name"] == "authority")
        self.assertEqual(authority["verdict"], "fail")
        self.assertIn("evaluator/gate_correct.py", authority["rule_details"][0])
        # the protected file is reverted, not committed
        text = victim.read_text(encoding="utf-8")
        self.assertNotIn("agent-injected", text)


class InterruptTests(ToyCampaignTest):
    def test_sigterm_interrupts_run_and_resume_completes(self):
        agent = self.tmp / "slow_agent.py"
        agent.write_text(SLOW_AGENT, encoding="utf-8")
        out, _rc = self.start_campaign(
            run=False, agent_command=["{python}", str(agent)],
            task_overrides={"budgets": {
                "max_attempts": 2, "campaign_wall_clock_s": 3600.0,
                "attempt_wall_clock_s": 600.0, "gate_timeout_s": 120.0,
                "agent_timeout_s": 120.0, "token_budget": None,
                "cooldown_max_s": 60.0}})
        env = dict(os.environ)
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "campaign.py"), "run", "--campaign", str(out)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pid_file = out / "attempts" / "001" / "grandchild.pid"
        deadline = time.monotonic() + 60
        while not pid_file.exists() and time.monotonic() < deadline:
            self.assertIsNone(proc.poll(), "campaign exited before the agent ran")
            time.sleep(0.05)
        self.assertTrue(pid_file.exists())
        grandchild = int(pid_file.read_text().strip())

        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130, stderr)
        # no surviving grandchild (process-group kill)
        gone = False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except (ProcessLookupError, PermissionError):
                gone = True
                break
            time.sleep(0.05)
        self.assertTrue(gone, "grandchild survived the campaign interrupt")
        state = self.state(out)
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["stop_reason"], "interrupted")
        # the lock is released and re-acquirable
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

        # resume: the interrupted attempt counts, the rest of the budget runs
        rc = campaign_mod.main(["resume", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        state = self.state(out)
        self.assertEqual(state["stop_reason"], "max_attempts")
        ledger = self.ledger(out)
        kinds = [(e["kind"], e["verdict"]) for e in ledger]
        self.assertIn(("attempt", "interrupted"), kinds)
        self.assertEqual(state["attempts_used"], 2)


class SafetyStopTests(ToyCampaignTest):
    def test_wedge_stop_is_terminal_no_second_attempt(self):
        out, _rc = self.start_campaign(run=False, max_attempts=5)
        real_probe = monitors.probe
        calls = {"n": 0}

        def probe_then_wedge(profile, last_gate_log=None, probe_obj=None):
            calls["n"] += 1
            if calls["n"] <= 3:  # before attempt 1 + between its two gates
                return monitors.ProbeResult("ok")
            return monitors.ProbeResult("stop", "wedge", {"message": "test wedge"})

        with mock.patch.object(monitors, "probe", probe_then_wedge), \
             mock.patch.object(monitors, "wait_cooldown",
                               lambda *a, **k: monitors.ProbeResult("stop", "wedge")):
            rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 3)
        state = self.state(out)
        self.assertEqual(state["stop_reason"], "wedge")
        self.assertEqual(state["attempts_used"], 1,
                         "a wedge stop must not start another attempt")
        self.assertFalse((out / "attempts" / "002").exists(),
                         "no gate or agent command may run again after a wedge")
        # resume refuses while the wedge persists
        with mock.patch.object(monitors, "probe",
                               lambda *a, **k: monitors.ProbeResult("stop", "wedge",
                                                                    {"message": "x"})):
            rc = campaign_mod.main(["resume", "--campaign", str(out)])
        self.assertEqual(rc, 3)

    def _head(self, out: Path) -> str:
        return subprocess.run(["git", "-C", str(out / "worktree"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()

    def test_stop_between_gates_records_and_reverts_the_partial_attempt(self):
        out, _rc = self.start_campaign(run=False, max_attempts=5)
        baseline = json.loads((out / "campaign.json").read_text(encoding="utf-8"))["baseline_revision"]
        calls = {"n": 0}

        def probe(profile, last_gate_log=None, probe_obj=None):
            calls["n"] += 1
            if calls["n"] <= 2:  # before attempt 1 + before its first gate
                return monitors.ProbeResult("ok")
            return monitors.ProbeResult("stop", "adb_disconnected", {"message": "gone"})

        with mock.patch.object(monitors, "probe", probe):
            rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 3)
        entry = self.ledger(out)[-1]
        self.assertEqual(entry["n"], 1, "the stopped attempt stays in the audit trail")
        self.assertEqual(entry["verdict"], "inconclusive")
        self.assertEqual(entry["stop_reason"], "adb_disconnected")
        self.assertFalse(entry["kept"])
        ran = [g["name"] for g in entry["gates"] if g["verdict"] not in ("skipped",)]
        self.assertIn("correct", ran, "gates that ran before the stop are recorded")
        self.assertEqual(self._head(out), baseline, "the partial candidate is reverted")
        self.assertTrue((out / "attempts" / "001" / "change.patch").exists())
        state = self.state(out)
        self.assertEqual(state["attempts_used"], 1)
        self.assertIsNone(state["current_attempt"])

    def test_last_gate_log_is_probed_for_driver_failures(self):
        out, _rc = self.start_campaign(run=False, max_attempts=1)
        seen = []

        def probe(profile, last_gate_log=None, probe_obj=None):
            seen.append(last_gate_log)
            return monitors.ProbeResult("ok")

        with mock.patch.object(monitors, "probe", probe):
            rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        perf_log = (out / "attempts" / "001" / "gate-perf.log").read_text(encoding="utf-8")
        self.assertEqual(seen[-1], perf_log,
                         "the final gate's log must reach the wedge/driver-failure probe")


class ReportTests(ToyCampaignTest):
    def test_no_improvement_is_an_honest_valid_outcome(self):
        agent = self.tmp / "noop_agent.py"
        agent.write_text(NOOP_AGENT, encoding="utf-8")
        out, rc = self.start_campaign(
            max_attempts=1, agent_command=["{python}", str(agent)])
        self.assertEqual(rc, 0)
        camp = spec.load_json(out / "campaign.json")
        self.assertEqual(camp["best_sha"], camp["baseline_revision"])
        rc = campaign_mod.main(["report", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("no improvement: baseline retained", report)
        self.assertIn("blocked", report.lower())
        self.assertIn("unavailable", report)

    def test_report_contains_all_attempts_and_rerun_commands(self):
        out, rc = self.start_campaign(max_attempts=2)
        self.assertEqual(rc, 0)
        rc = campaign_mod.main(["report", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("resume --campaign", report)
        self.assertIn("finalize --campaign", report)
        self.assertIn("git -C", report)
        self.assertIn("gate_correct.py", report)  # substituted gate command
        data = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["attempts"]), 3)  # control + 2 attempts
        self.assertIn("never merges", data["footer"])


class FinalizeTests(ToyCampaignTest):
    def start_kept_campaign(self, **kw):
        out, _rc = self.start_campaign(run=False, max_attempts=1, **kw)
        rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        return out

    def test_finalize_revalidates_best_vs_original_baseline(self):
        out = self.start_kept_campaign()
        heldout = self.tmp / "heldout"
        heldout.mkdir()
        (heldout / "corpus.txt").write_text("held-out clips manifest\n", encoding="utf-8")
        rc = campaign_mod.main(["finalize", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        state = self.state(out)
        self.assertEqual(state["status"], "finalized")
        # the kept win must PASS against the ORIGINAL baseline, not best-vs-best
        self.assertEqual(state["finalize"]["verdict"], "pass")
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("blocked: held-out corpus not provided", report)
        (heldout / "input.txt").write_text("held-out words the agent never saw\n",
                                           encoding="utf-8")
        # Unpinned held-out: the gates run, but promotion stays blocked.
        rc = campaign_mod.main(["finalize", "--campaign", str(out),
                                "--heldout", str(heldout)])
        self.assertEqual(rc, 0)
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("blocked: held-out corpus not pinned", report)
        self.assertEqual(self.state(out)["finalize"]["heldout"]["verdict"], "pass")
        # Pinned (sealed at start in real use): promotion is validated.
        camp = json.loads((out / "campaign.json").read_text(encoding="utf-8"))
        camp["seals"]["heldout_pin"] = spec.hash_tree(heldout)
        (out / "campaign.json").write_text(json.dumps(camp, indent=2), encoding="utf-8")
        rc = campaign_mod.main(["finalize", "--campaign", str(out),
                                "--heldout", str(heldout)])
        self.assertEqual(rc, 0)
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("validated on notebook (pinned held-out gates passed)", report)
        self.assertIsNotNone(self.state(out)["finalize"]["heldout"]["sha256"])

    def test_finalize_phase_gate_runs_only_at_finalize(self):
        toy_repo = self.tmp / "toyrepo"
        real_profile = toy_mod.toy_profile

        def profile_with_measurement(repo):
            prof = real_profile(repo)
            prof["gates"].append({
                "name": "energy", "stage": "resource", "phase": "finalize",
                "required": False, "rules": [],
                "argv": ["{python}", "-c", "print('METRIC energy_mwh=1.5')"],
            })
            return prof

        with mock.patch.object(toy_mod, "toy_profile", profile_with_measurement):
            out = self.start_kept_campaign()
        for entry in self.ledger(out):
            self.assertNotIn("energy", [g["name"] for g in entry["gates"]],
                             "a finalize-phase gate must not run per attempt/control")
        rc = campaign_mod.main(["finalize", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        gates = {g["name"]: g for g in self.state(out)["finalize"]["gates"]}
        self.assertEqual(gates["energy"]["metrics"].get("energy_mwh"), 1.5)
        self.assertEqual(self.state(out)["finalize"]["verdict"], "pass")
        del toy_repo

    def test_finalize_refuses_wrong_heldout_pin(self):
        out, _rc = self.start_campaign(run=False, max_attempts=1)
        camp = json.loads((out / "campaign.json").read_text(encoding="utf-8"))
        camp["seals"]["heldout_pin"] = "0" * 64  # simulate a pinned corpus
        (out / "campaign.json").write_text(json.dumps(camp, indent=2), encoding="utf-8")
        rc = campaign_mod.main(["run", "--campaign", str(out)])
        self.assertEqual(rc, 0)
        heldout = self.tmp / "heldout"
        heldout.mkdir()
        (heldout / "corpus.txt").write_text("different corpus\n", encoding="utf-8")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = campaign_mod.main(["finalize", "--campaign", str(out),
                                    "--heldout", str(heldout)])
        self.assertEqual(rc, 2)
        self.assertIn("hash mismatch", buf.getvalue())


def _FakeProbeNoAdb():
    import test_monitors
    return test_monitors.ScriptedProbe(adb_ok=False)


if __name__ == "__main__":
    unittest.main()
