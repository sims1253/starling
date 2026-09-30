#!/usr/bin/env python3
"""campaign.py — the bounded hypothesis -> experiment -> review campaign
runner (issue #176).

One campaign = one PREREGISTERED task spec + one profile, sealed at `start`
into a campaign directory together with the trusted evaluator extracted
from the BASELINE revision. `run` then loops: probe device safety, hand the
campaign worktree to an optimizing agent (or accept an agent-driven change
via `attempt`), and evaluate the change under the sealed gates — authority
(allowed/protected paths) first, then build, then gates in fixed stage
order with fail-fast. Only measured `pass` attempts are kept; everything
else is reverted to the best-so-far commit and stored as a diff plus a
`refs/campaigns/<id>/attempt-NNN` ref for the audit trail.

Subcommands (see README.md for the full walkthrough):

    list | preview | start | run | attempt | resume | status | report |
    finalize | pilot

Exit codes: 0 ok, 2 preflight/usage error, 3 stopped by a safety monitor,
4 evaluator tampered / seal mismatch, 130 interrupted.

Stdlib only, Python 3.10+. The campaign code runs git only against the
campaign's own worktree/temp repos (never the invoking checkout's working
tree or branches, beyond `git worktree add` and refs/campaigns/* refs).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gates as gates_mod  # noqa: E402
import monitors  # noqa: E402
import report as report_mod  # noqa: E402
import spec  # noqa: E402
import toy as toy_mod  # noqa: E402
from spec import SpecError  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_SAFETY = 3
EXIT_TAMPER = 4
# Control/finalize run with no attempt deadline (the campaign wall clock
# bounds agent attempts, not the pre-registration control or the closing
# re-validation).
UNCAPPED_OFFSET_S = 1e12
EXIT_INTERRUPT = 130

SAFETY_STOPS = {
    "thermal", "battery", "adb_disconnected", "wedge", "driver_failure",
    "memory_pressure",
}
AGENT_FAILURE_STREAK_LIMIT = 3
DISK_MIN_BYTES = 256 * 1024 * 1024

PROFILES_DIR = HERE / "profiles"

_INTERRUPT = {"signal": None}


def _interrupt_check():
    return _INTERRUPT["signal"]


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        _INTERRUPT["signal"] = signum

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def _now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


class CampaignError(RuntimeError):
    def __init__(self, message: str, exit_code: int = EXIT_USAGE):
        super().__init__(message)
        self.exit_code = exit_code


class SafetyStop(Exception):
    def __init__(self, reason: str, readings: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.readings = readings or {}


class Interrupted(Exception):
    pass


# ---------------------------------------------------------------------------
# Locks (host-wide + per-serial for phones)
# ---------------------------------------------------------------------------

try:
    import fcntl
except ImportError:  # non-POSIX
    fcntl = None  # type: ignore


class LockHeld(RuntimeError):
    pass


def _flock(path: Path, wait: bool):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    flags = fcntl.LOCK_EX if wait else (fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        fcntl.flock(fd, flags)
    except OSError:
        holder = os.pread(fd, 128, 0).decode("utf-8", "replace").strip()
        os.close(fd)
        raise LockHeld(
            f"another campaign holds the lock {path} (pid {holder or 'unknown'}); "
            "use --wait-lock to wait for it"
        )
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def host_lock_path() -> Path:
    return Path(os.environ.get("STARLING_CAMPAIGN_LOCK")
                or Path(tempfile.gettempdir()) / "starling-campaign.lock")


def _release(fd) -> None:
    try:
        os.close(fd)  # closing releases the flock
    except OSError:
        pass


def campaign_locks(profile: dict | None = None, wait: bool = False):
    """Host-wide flock (+ per-adb-serial flock for pixel profiles).

    Deliberately NOT the experiments lock ($STARLING_EXPERIMENT_LOCK): the
    perf gate invokes run_experiment.py as a child, which takes that lock
    itself — holding it here would deadlock.
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        if fcntl is None:
            yield
            return
        fds = [_flock(host_lock_path(), wait)]
        try:
            if profile and profile.get("device") == "pixel":
                serial = profile.get("adb_serial") or "default"
                path = Path(tempfile.gettempdir()) / f"starling-campaign-adb-{serial}.lock"
                try:
                    fds.append(_flock(path, wait))
                except BaseException:
                    for fd in fds:  # never leak the host-wide lock on contention
                        _release(fd)
                    raise
            yield
        finally:
            for fd in fds:
                _release(fd)

    return _cm()


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

EVALUATOR_COMMIT_ARGS = (
    "-c", "user.name=starling-campaign",
    "-c", "user.email=campaign@starling.local",
    "-c", "commit.gpgsign=false",
    "-c", "core.hooksPath=/dev/null",
)


def git(repo: Path, *args: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise CampaignError(f"git {' '.join(args)} timed out after {timeout}s in {repo}") from e
    if check and out.returncode != 0:
        raise CampaignError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out


def resolve_revision(repo: Path, rev: str) -> str:
    out = git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}", check=False)
    sha = out.stdout.strip()
    if out.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        raise CampaignError(f"cannot resolve baseline revision {rev!r} in {repo}")
    return sha


def default_repo() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip())
    return HERE.parents[1]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _required_tools(profile: dict) -> list[str]:
    tools = ["git"]
    build = profile.get("build", {})
    text = " ".join(build.get("argv") or []) + " " + (build.get("shell") or "")
    if "cmake" in text:
        tools.append("cmake")
        if "-G Ninja" in text:
            tools.append("ninja")
    if profile.get("device") == "pixel":
        tools.append("adb")
    tools.extend(profile.get("requires_tools") or [])
    seen: list[str] = []
    for t in tools:
        if t not in seen:
            seen.append(t)
    return seen


def preflight(profile: dict, task: dict, repo: Path, out_dir: Path,
              adb_serial: str | None = None, probe: monitors.Probe | None = None,
              assume_lock_held: bool = False) -> tuple[list[str], dict]:
    """All prerequisites, listed together. Returns (problems, context)."""
    problems: list[str] = []
    probe = probe or monitors.Probe()
    check_profile = dict(profile)
    if adb_serial:
        check_profile = dict(profile, adb_serial=adb_serial)

    if profile.get("status") == "blocked":
        problems.append(
            f"profile {profile.get('id')} is blocked on {profile.get('blocked_on')}: "
            f"{profile.get('blocked_reason')}"
        )
    for p in spec.validate_task(task, profile):
        problems.append(f"task: {p}")

    for artifact in profile.get("artifacts", []):
        path = spec.expand_artifact_path(artifact["path"])
        if not path.is_file():
            problems.append(f"artifact {artifact['name']} missing: {path}")
            continue
        if artifact.get("sha256"):
            actual = spec.sha256_file(path)
            if actual != artifact["sha256"]:
                problems.append(
                    f"artifact {artifact['name']} hash mismatch: pinned {artifact['sha256'][:12]}… "
                    f"!= actual {actual[:12]}…"
                )

    try:
        resolve_revision(repo, task.get("baseline_revision", "HEAD"))
    except CampaignError as e:
        problems.append(str(e))

    for tool in _required_tools(check_profile):
        if not shutil.which(tool):
            problems.append(f"required tool {tool!r} not on PATH")

    identity = monitors.discover(check_profile, probe)
    match = monitors.device_matches(check_profile, identity)
    if match is False:
        problems.append(
            f"device mismatch: discovered {identity.get('model') or identity.get('cpu')!r} "
            f"does not match device_expect {profile.get('device_expect')!r} — "
            "do not substitute another device's result"
        )
    elif match is None and profile.get("device_expect"):
        problems.append(
            "cannot verify device_expect "
            f"{profile.get('device_expect')!r}: device identity unavailable "
            f"({json.dumps(identity)[:200]})"
        )
    marker = probe.wedge_marker_present(check_profile, adb_serial)
    if marker is True:
        problems.append("GPU wedge marker present; inspect and clear it manually before starting")

    if fcntl is not None and not assume_lock_held:
        try:
            with campaign_locks(check_profile):
                pass
        except LockHeld as e:
            problems.append(str(e))

    try:
        free = shutil.disk_usage(out_dir.parent if out_dir.parent.exists() else out_dir).free
        if free < DISK_MIN_BYTES:
            problems.append(f"only {free // (1 << 20)} MB free below {out_dir} (need {DISK_MIN_BYTES >> 20} MB)")
    except OSError:
        pass

    context = {"device_identity": identity, "repo": str(repo)}
    return problems, context


def _print_plan(profile: dict, task: dict, context: dict) -> None:
    print("== resolved plan ==")
    print(f"profile      {profile['id']} ({profile.get('model', {}).get('name')})")
    print(f"device       {profile.get('device')} engine={profile.get('engine')} backend={profile.get('backend')}")
    print(f"baseline     {task['baseline_revision']}")
    print(f"objective    {task['objective']['gate']}.{task['objective']['metric']} "
          f"{task['objective']['direction']} min_improvement={task['objective']['min_improvement']}")
    print("budgets:")
    for key, value in sorted(task["budgets"].items()):
        print(f"  {key} = {value}")
    print("allowed paths:")
    for p in task["allowed_paths"]:
        print(f"  {p}")
    print("protected paths (always-protected + task + profile evaluator paths):")
    for p in spec.effective_protected(profile, task):
        print(f"  {p}")
    print(f"measurement_protocol fresh_process={task['measurement_protocol']['fresh_process']} "
          f"declared_deviations={task['measurement_protocol']['declared_deviations'] or 'none'}")
    print("gates (fixed stage order, fail-fast):")
    for gate in gates_mod.ordered_gates(profile["gates"]):
        command = gate.get("argv") or gate.get("shell")
        command = gates_mod.substitute_command(command, {k: f"<{k}>" for k in gates_mod.PLACEHOLDERS},
                                               expand_vars=False)
        print(f"  [{gate['stage']:>10}] {gate['name']} (required={gate.get('required', True)}, "
              f"objective={gate.get('objective', False)}): {command}")
        for rule in gate.get("rules", []):
            print(f"      rule {rule['metric']} {rule['op']} {rule['value']}")
    print("workloads:")
    for w in profile.get("workloads", []):
        owner = f" (owner {w['owner_issue']})" if w.get("owner_issue") else ""
        print(f"  {w['id']}: {w['status']}{owner}")
    print(f"device identity: {json.dumps(context.get('device_identity', {}))[:400]}")


# ---------------------------------------------------------------------------
# Task generation from a profile
# ---------------------------------------------------------------------------

def _parse_duration(text: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", str(text).strip())
    if not m:
        raise CampaignError(f"cannot parse duration {text!r} (use e.g. 3600, 90m, 8h)")
    factor = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[m.group(2)]
    return float(m.group(1)) * factor


def generate_task(profile: dict, baseline_sha: str, args) -> dict:
    defaults = profile.get("defaults", {})
    budgets = dict(defaults.get("budgets", {}))
    if getattr(args, "max_attempts", None):
        budgets["max_attempts"] = args.max_attempts
    if getattr(args, "wall_clock", None):
        budgets["campaign_wall_clock_s"] = _parse_duration(args.wall_clock)
    objectives = profile.get("objectives") or [{}]
    obj = objectives[0]
    fallback_gate = profile["gates"][0]["name"] if profile.get("gates") else "none"
    task = {
        "schema": spec.TASK_SCHEMA,
        "campaign_id": f"{profile['id']}-{_dt.date.today().isoformat()}",
        "profile": profile["id"],
        "hypothesis_backlog": [],
        "baseline_revision": baseline_sha,
        "allowed_paths": list(defaults.get("allowed_paths") or ["cpp/**"]),
        "protected_paths": [],
        "objective": {
            "gate": obj.get("gate", fallback_gate),
            "metric": obj.get("metric", "improvement_pct"),
            "direction": obj.get("direction", "higher"),
            "min_improvement": obj.get("min_improvement", 0.0),
        },
        "constraints": [],
        "budgets": budgets,
        "measurement_protocol": {"fresh_process": True, "declared_deviations": []},
        "agent": {"command": getattr(args, "agent_cmd", None), "timeout_s": budgets.get("agent_timeout_s")},
    }
    if getattr(args, "allowed", None):
        task["allowed_paths"] = list(args.allowed)
    if task["agent"]["command"] is None:
        task["agent"] = None
    return task


# ---------------------------------------------------------------------------
# The Campaign object
# ---------------------------------------------------------------------------

class Campaign:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.campaign = spec.load_json(self.dir / "campaign.json")
        self.campaign["dir"] = str(self.dir)
        self.profile = self.campaign["profile"]
        self.task = self.campaign["task"]
        self.state = spec.load_json(self.dir / "state.json")
        self.repo = Path(self.campaign["repo_root"])
        self.worktree = self.dir / "worktree"
        self.trusted = self.dir / "trusted"
        self.baseline_dir = self.dir / "baseline"
        self.best_dir = self.dir / "best"

    # -- git in the campaign worktree -----------------------------------
    def wt(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return git(self.worktree, *args, check=check)

    def wt_sha(self) -> str:
        return self.wt("rev-parse", "HEAD").stdout.strip()

    # -- checkpointing ---------------------------------------------------
    def checkpoint(self, **updates) -> None:
        self.state.update(updates)
        self.state["updated_at"] = _now_iso()
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.dir / "state.json")

    def ledger(self) -> list[dict]:
        path = self.dir / "ledger.jsonl"
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    def append_ledger(self, entry: dict) -> None:
        entry["finished_at"] = _now_iso()
        with open(self.dir / "ledger.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    # -- seals -------------------------------------------------------------
    def verify_seals(self) -> list[str]:
        problems: list[str] = []
        sealed = spec.load_json(self.dir / "task.sealed.json")
        if spec.seal_of(sealed) != self.campaign["seals"].get("task"):
            problems.append(
                "task.sealed.json no longer matches its seal: evaluation rules changed "
                "after preregistration (hard error)"
            )
        elif spec.seal_of(self.task) != self.campaign["seals"].get("task"):
            problems.append("campaign.json task no longer matches the sealed task")
        if spec.seal_of(self.profile) != self.campaign["seals"].get("profile"):
            problems.append("campaign.json profile no longer matches its seal")
        for name, sha in (self.campaign["seals"].get("artifacts") or {}).items():
            artifact = next((a for a in self.profile.get("artifacts", []) if a["name"] == name), None)
            if artifact is None:
                problems.append(f"sealed artifact {name} missing from the profile")
                continue
            path = spec.expand_artifact_path(artifact["path"])
            if not path.is_file():
                problems.append(f"sealed artifact {name} disappeared: {path}")
            elif spec.sha256_file(path) != sha:
                problems.append(f"sealed artifact {name} changed on disk since start")
        return problems

    def evaluator_hash(self) -> str:
        return spec.hash_tree(self.trusted)

    def evaluator_ok(self) -> bool:
        return self.evaluator_hash() == self.campaign["seals"].get("evaluator")

    # -- placeholders + env ------------------------------------------------
    def heldout_vars(self) -> list[str]:
        env = (self.profile.get("heldout") or {}).get("env")
        return [env] if env else []

    def base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        serial = self.campaign.get("adb_serial")
        if serial:
            env["ANDROID_SERIAL"] = serial
        return env

    def placeholder_values(self, attempt_dir: Path, candidate_dir: Path,
                           heldout_dir: Path | None = None) -> dict[str, str]:
        return {
            "trusted": str(self.trusted),
            "worktree": str(self.worktree),
            "candidate": str(candidate_dir),
            "baseline": str(self.baseline_dir),
            "best": str(self.best_dir),
            "attempt_dir": str(attempt_dir),
            "campaign_dir": str(self.dir),
            "python": sys.executable,
            "adb_serial": self.campaign.get("adb_serial") or "",
            "heldout": str(heldout_dir) if heldout_dir else "",
        }

    # -- budgets -----------------------------------------------------------
    def campaign_deadline(self) -> float:
        started = self.campaign.get("started_at")
        budget = self.task["budgets"].get("campaign_wall_clock_s", 1e12)
        # Wall clock on every path: remaining_campaign_s() subtracts
        # time.time(); a monotonic fallback would look instantly expired.
        if not started:
            return time.time() + budget
        try:
            t0 = _dt.datetime.fromisoformat(str(started)).timestamp()
        except ValueError:
            return time.time() + budget
        return t0 + float(budget)

    def remaining_campaign_s(self) -> float:
        return self.campaign_deadline() - time.time()

    def remaining_attempt_s(self, attempt_deadline: float) -> float:
        # attempt_deadline is monotonic-based while the campaign deadline is
        # wall-clock (it must survive process restarts). Compare both as
        # REMAINING seconds — mixing the two clocks directly would make the
        # campaign term dead code (epoch vs monotonic differ by ~1e9).
        # The UNCAPPED sentinel marks the control/finalize passes: the wall
        # clock bounds agent attempts, not the pre-registration measurement.
        remaining = attempt_deadline - time.monotonic()
        if remaining >= UNCAPPED_OFFSET_S / 2:
            return remaining
        return min(remaining, self.remaining_campaign_s())

    # -- probes ------------------------------------------------------------
    def safety_probe(self, last_gate_log: str | None = None) -> None:
        """Raise SafetyStop/Interrupted; cool down when hot."""
        result = monitors.probe(self.profile_with_serial(), last_gate_log)
        if result.status == "stop":
            raise SafetyStop(result.reason, result.readings)
        if result.status == "cooldown":
            max_s = self.task["budgets"].get("cooldown_max_s") or 0.0
            result = monitors.wait_cooldown(
                self.profile_with_serial(), max_s, last_gate_log,
                sleep=lambda s: time.sleep(min(s, 0.5)),
            )
            if _interrupt_check():
                raise Interrupted()
            if result.status == "stop":
                raise SafetyStop(result.reason, result.readings)
        if _interrupt_check():
            raise Interrupted()

    def profile_with_serial(self) -> dict:
        serial = self.campaign.get("adb_serial")
        return dict(self.profile, adb_serial=serial) if serial else self.profile


# ---------------------------------------------------------------------------
# Trusted evaluator extraction
# ---------------------------------------------------------------------------

def extract_trusted(repo: Path, sha: str, configured_paths: list[str], dest: Path) -> dict:
    """git-archive the evaluator from the BASELINE revision into `dest`.

    Paths missing from the baseline tree are skipped and reported (a young
    campaign checkout may not contain benchmarks/campaign at the baseline).
    """
    import tarfile

    listing = git(repo, "ls-tree", "-r", "--name-only", sha).stdout.splitlines()
    used: list[str] = []
    skipped: list[str] = []
    for configured in configured_paths:
        prefix = configured.rstrip("/")
        matched = any(
            f == configured or f.startswith(prefix + "/") or spec.glob_match(configured, f)
            for f in listing
        )
        (used if matched else skipped).append(configured)
    dest.mkdir(parents=True, exist_ok=True)
    if not used:
        return {"paths": used, "skipped": skipped, "files": 0}
    proc = subprocess.Popen(
        ["git", "-C", str(repo), "archive", sha, *used],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12: no filter argument
            for member in tf:  # keep it hardened: regular files and dirs only
                if member.isreg() or member.isdir():
                    tf.extract(member, dest)
    rc = proc.wait()
    if rc != 0:
        raise CampaignError(f"git archive failed for baseline {sha} paths {used}")
    # Read-only files: a gate that writes into the sealed evaluator fails
    # loudly at the write instead of silently changing the evaluator hash.
    for path in dest.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    files = sum(1 for p in dest.rglob("*") if p.is_file())
    return {"paths": used, "skipped": skipped, "files": files}


def default_trusted_paths(profile: dict) -> list[str]:
    return list(profile.get("trusted_paths") or [
        "benchmarks/campaign",
        "benchmarks/experiments",
        "benchmarks/fast_engine",
        "tests/fixtures",
    ])


# ---------------------------------------------------------------------------
# Attempt evaluation
# ---------------------------------------------------------------------------

def _copy_artifacts(src_root: Path, dst_root: Path, relpaths: list[str]) -> list[str]:
    copied = []
    for rel in relpaths:
        src = src_root / rel
        if not src.exists():
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def applicable_gates(gates: list[dict], kind: str) -> list[dict]:
    """Ordered gates for this evaluation kind. Finalize-phase gates (costly
    measurements such as phone energy) run only at finalize — never per
    attempt or for the control."""
    return [g for g in gates_mod.ordered_gates(gates)
            if kind == "finalize" or g.get("phase", "attempt") != "finalize"]


class Evaluator:
    """Runs authority -> build -> gates for one attempt (or the control)."""

    def __init__(self, camp: Campaign):
        self.camp = camp
        self.gate_timeout_default = camp.task["budgets"].get("gate_timeout_s") or 600.0

    def values(self, attempt_dir: Path, candidate_dir: Path) -> dict[str, str]:
        return self.camp.placeholder_values(attempt_dir, candidate_dir)

    def authority(self, last_kept_sha: str) -> dict:
        """Commit pending changes, then classify the diff. Never evaluates a
        change outside allowed_paths or inside protected_paths."""
        camp = self.camp
        # Submodule CONTENT edits are invisible to the parent-repo diff below
        # (--ignore-submodules=dirty) and can never be committed there, so an
        # agent edit inside e.g. third_party/ggml would silently persist into
        # every later build and finalize. Builds dirty the submodule too
        # (configure applies the ggml patch series), but _restore_submodules
        # cleans that up after every build — anything left here was written
        # by the candidate. Reject it as a protected-path violation.
        dirty = camp.wt("submodule", "foreach", "--quiet", "--recursive",
                        "test -z \"$(git status --porcelain)\" || echo DIRTY",
                        check=False)
        if "DIRTY" in dirty.stdout:
            return {
                "verdict": "fail",
                "reason": "submodule content modified (always protected): "
                          "uncommitted changes inside a submodule working tree",
                "changed_paths": [],
                "candidate_sha": camp.wt_sha(),
            }
        # Dirty submodule CONTENT is ignored: the build's configure step applies
        # the ggml patch series inside third_party/ggml. A changed submodule
        # commit still shows (and is rejected as a protected path).
        status = ("status", "--porcelain", "--ignore-submodules=dirty")
        if camp.wt(*status).stdout.strip():
            camp.wt("add", "-A")
            # Identity and hook policy ride on -c (never `git config`: linked
            # worktrees share the operator's repository config), and the
            # repository's hooks must not run with the evaluator's authority.
            done = camp.wt(*EVALUATOR_COMMIT_ARGS, "commit", "-q", "--no-verify", "-m",
                           "campaign attempt (auto-committed by evaluator)", check=False)
            if camp.wt(*status).stdout.strip():
                raise CampaignError(
                    "cannot commit the candidate's changes in the campaign worktree: "
                    + (done.stderr.strip() or done.stdout.strip()))
        head = camp.wt_sha()
        changed = camp.wt("diff", "--name-only", last_kept_sha + ".." + head).stdout.splitlines()
        changed = sorted(set(c for c in changed if c.strip()))
        if not changed:
            return {"verdict": "inconclusive", "reason": "no_change",
                    "changed_paths": [], "candidate_sha": head}
        allowed = camp.task["allowed_paths"]
        protected = spec.effective_protected(camp.profile, camp.task)
        violations = spec.path_violations(changed, allowed, protected)
        if violations:
            return {
                "verdict": "fail",
                "reason": "paths outside allowed_paths or inside protected_paths: "
                          + ", ".join(violations),
                "changed_paths": changed,
                "candidate_sha": head,
            }
        return {"verdict": "pass", "reason": "", "changed_paths": changed, "candidate_sha": head}

    def build(self, attempt_dir: Path, attempt_deadline: float) -> dict:
        camp = self.camp
        build = camp.profile.get("build", {})
        command = build.get("argv") or build.get("shell")
        if not command:
            return {"verdict": "pass", "artifacts": []}
        values = self.values(attempt_dir, attempt_dir)
        command = gates_mod.substitute_command(command, values,
                                               expand_vars=build.get("argv") is not None)
        timeout = min(float(self.gate_timeout_default),
                      max(self.camp.remaining_attempt_s(attempt_deadline), 1.0))
        env, removed = gates_mod.gate_env(camp.base_env(), camp.heldout_vars())
        log_path = attempt_dir / "build.log"
        with open(log_path, "wb") as log:
            result = gates_mod.run_child(
                command, env=env, cwd=camp.worktree, timeout_s=timeout,
                log_fh=log, interrupt_check=_interrupt_check,
            )
        record = {
            "name": "build",
            "stage": "build",
            "required": True,
            "objective": False,
            "verdict": None,
            "exit_code": result["exit_code"],
            "timed_out": result["timed_out"],
            "interrupted": result["interrupted"],
            "metrics": {},
            "divergence": [],
            "rules": [],
            "rule_details": [],
            "wall_s": round(result["wall_s"], 3),
            "resource": result["resource"],
            "log": log_path.name,
            "scrubbed_env": removed,
        }
        if result["interrupted"]:
            record["verdict"] = "interrupted"
        elif result["timed_out"]:
            record["verdict"] = "fail"
            record["rule_details"] = [f"build timed out after {timeout:.0f}s"]
        elif result["exit_code"] != 0:
            record["verdict"] = "fail"
            record["rule_details"] = [f"build exit code {result['exit_code']}; "
                                      "log tail:\n" + gates_mod.log_tail(log_path)]
        else:
            record["verdict"] = "pass"
        if record["verdict"] == "pass":
            copied = _copy_artifacts(camp.worktree, attempt_dir, build.get("artifacts_out", []))
            record["artifacts"] = copied
            record["artifact_hashes"] = {
                rel: spec.sha256_file(attempt_dir / rel)
                for rel in copied if (attempt_dir / rel).is_file()
            }
        return record

    def gates(self, attempt_dir: Path, candidate_dir: Path, attempt_deadline: float,
              kind: str, values: dict[str, str] | None = None) -> list[dict]:
        camp = self.camp
        results: list[dict] = []
        values = values or self.values(attempt_dir, candidate_dir)
        env = camp.base_env()
        try:
            self._run_gates(results, values, env, attempt_dir, attempt_deadline, kind)
        except SafetyStop as stop:
            stop.partial = results  # the caller records what ran before the stop
            raise
        return results

    def _run_gates(self, results: list[dict], values: dict[str, str], env: dict,
                   attempt_dir: Path, attempt_deadline: float, kind: str) -> None:
        camp = self.camp
        last_log_text = None
        applicable = applicable_gates(camp.profile["gates"], kind)
        for index, gate in enumerate(applicable):
            if kind != "control":  # safety probes between gates (not before the control)
                camp.safety_probe(last_log_text)
            if _interrupt_check():
                raise Interrupted()
            remaining = camp.remaining_attempt_s(attempt_deadline)
            if remaining <= 0:
                results.append({
                    "name": gate["name"], "stage": gate["stage"],
                    "required": gate.get("required", True),
                    "objective": gate.get("objective", False),
                    "verdict": "fail", "exit_code": None, "timed_out": False,
                    "interrupted": None, "metrics": {}, "divergence": [], "rules": [],
                    "rule_details": ["attempt wall clock exhausted before this gate"],
                    "wall_s": 0.0,
                    "resource": {"maxrss_kb": spec.UNAVAILABLE, "utime_s": spec.UNAVAILABLE,
                                 "stime_s": spec.UNAVAILABLE},
                    "log": None, "scrubbed_env": [],
                })
                continue
            default = min(float(self.gate_timeout_default), max(remaining, 1.0))
            gate_def = dict(gate)
            if gate.get("timeout_s") is None or gate["timeout_s"] > default:
                gate_def["timeout_s"] = default
            log_path = attempt_dir / f"gate-{gate['name']}.log"
            camp.checkpoint(current_attempt_stage=f"gates:{gate['name']}")
            result = gates_mod.run_gate(
                gate_def, values, env, log_path,
                default_timeout_s=default,
                heldout_vars=camp.heldout_vars(),
                cwd=camp.worktree,
                interrupt_check=_interrupt_check,
            )
            results.append(result)
            if result["verdict"] == "interrupted":
                raise Interrupted()
            if result["log"]:
                log_path_full = attempt_dir / result["log"]
                last_log_text = log_path_full.read_text(encoding="utf-8", errors="replace") \
                    if log_path_full.exists() else None
            if result["verdict"] == "fail":
                # Fail-fast: later gates are recorded skipped, never run.
                for later in applicable[index + 1:]:
                    results.append({
                        "name": later["name"], "stage": later["stage"],
                        "required": later.get("required", True),
                        "objective": later.get("objective", False),
                        "verdict": "skipped", "exit_code": None, "timed_out": False,
                        "interrupted": None, "metrics": {}, "divergence": [],
                        "rules": later.get("rules", []),
                        "rule_details": ["skipped: an earlier gate failed"],
                        "wall_s": 0.0,
                        "resource": {"maxrss_kb": spec.UNAVAILABLE, "utime_s": spec.UNAVAILABLE,
                                     "stime_s": spec.UNAVAILABLE},
                        "log": None, "scrubbed_env": [],
                    })
                break
        # The last gate's log is probed too (control included): a wedge or
        # driver failure in the final gate must stop the campaign before the
        # next attempt loads a model into a wedged driver.
        if last_log_text is not None:
            camp.safety_probe(last_log_text)


# ---------------------------------------------------------------------------
# Attempt orchestration
# ---------------------------------------------------------------------------

def _write_task_md(camp: Campaign, attempt_dir: Path, n: int) -> None:
    task = camp.task
    ledger = camp.ledger()
    history = [e for e in ledger if e.get("kind") == "attempt"]
    lines = [
        f"# Campaign {task['campaign_id']} — attempt {n}",
        "",
        "You are the optimizing agent. ONE hypothesis per attempt: make the",
        "smallest change that tests it, in the code region you are allowed to",
        "edit. The evaluator (gates, thresholds, held-out data) is sealed and",
        "outside your writable authority; never edit or run it. Outcomes:",
        "pass = kept; fail/inconclusive = reverted (both are recorded).",
        "",
        "## Where to write your files",
        "",
        f"- hypothesis (required): {attempt_dir / 'hypothesis.json'} — "
        '`{"hypothesis": "...", "expected_gain": "..."}\'',
        f"- usage (optional):     {attempt_dir / 'usage.json'} — "
        '`{"input_tokens": N, "output_tokens": N, "cost_usd": F}\'',
        "",
        "## Allowed paths (everything else is protected)",
        "",
        *[f"- {p}" for p in task["allowed_paths"]],
        "",
        "## Budgets",
        "",
        *[f"- {k}: {v}" for k, v in sorted(task["budgets"].items())],
        "",
        "## Objective",
        "",
        f"- gate `{task['objective']['gate']}` metric `{task['objective']['metric']}` "
        f"direction {task['objective']['direction']} "
        f"min improvement {task['objective']['min_improvement']}",
        "",
        "## Measurement protocol",
        "",
        f"- fresh process: {task['measurement_protocol']['fresh_process']}",
        f"- declared deviations: "
        f"{task['measurement_protocol']['declared_deviations'] or 'none'}",
        "",
    ]
    backlog = task.get("hypothesis_backlog") or []
    if backlog:
        lines += ["## Hypothesis backlog (pick one)", ""]
        lines += [f"- {h}" for h in backlog]
        lines.append("")
    if history:
        lines += ["## History (every attempt, failures included)", "",
                  "| # | hypothesis | verdict | failed stage | first divergence | key metrics |",
                  "|---|---|---|---|---|---|"]
        for e in history:
            failed = next((g["stage"] for g in e.get("gates", []) if g.get("verdict") == "fail"), "")
            div = next((d for g in e.get("gates", []) for d in (g.get("divergence") or [])), "")
            metrics = " ".join(
                f"{k}={v}" for g in e.get("gates", [])
                for k, v in (g.get("metrics") or {}).items()
            )
            lines.append(
                f"| {e.get('n')} | {str(e.get('hypothesis'))[:60]} | {e.get('verdict')} "
                f"| {failed} | {str(div)[:40]} | {metrics[:60]} |"
            )
        lines.append("")
    # The ledger is append-ordered and several attempts can be kept (each
    # pass replaces the best): the CURRENT best is the LAST kept entry —
    # scanning forward would brief the agent with a stale attempt.
    best_entry = next((e for e in reversed(ledger) if e.get("kept")), None)
    if best_entry:
        metrics = " ".join(
            f"{k}={v}" for g in best_entry.get("gates", [])
            for k, v in (g.get("metrics") or {}).items()
        )
        lines += ["## Current best (kept)", "",
                  f"- attempt {best_entry.get('n')} commit `{best_entry.get('candidate_sha')}`",
                  f"- metrics: {metrics or spec.UNAVAILABLE}", ""]
    (attempt_dir / "task.md").write_text("\n".join(lines), encoding="utf-8")


def _read_json_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_attempt(camp: Campaign, *, agent: bool = True, hypothesis: str | None = None) -> dict:
    """One full attempt. Returns the ledger entry; never raises for evaluation
    outcomes (safety stops and interrupts are raised for the run loop)."""
    n = camp.state.get("attempts_used", 0) + 1
    attempt_dir = camp.dir / "attempts" / f"{n:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    budgets = camp.task["budgets"]
    attempt_deadline = time.monotonic() + float(budgets.get("attempt_wall_clock_s", 1e12))
    camp.checkpoint(status="running", current_attempt={"n": n, "stage": "start"})
    _write_task_md(camp, attempt_dir, n)

    entry = {
        "n": n, "kind": "attempt", "started_at": _now_iso(),
        "hypothesis": hypothesis or spec.UNAVAILABLE, "verdict": None,
        "gates": [], "changed_paths": [], "candidate_sha": None,
        "usage": None, "kept": False, "wall_s": 0.0, "stop_reason": None,
        "task_seal": camp.campaign["seals"]["task"],
        "evaluator_hash": camp.evaluator_hash(),
    }

    # ---- agent ----------------------------------------------------------
    agent_failed = None
    if agent:
        agent_cfg = camp.task.get("agent") or {}
        command = agent_cfg.get("command")
        if not command:
            raise CampaignError(
                "the sealed task has no agent command; use agent-driven mode: "
                "campaign.py attempt --campaign DIR --hypothesis ..."
            )
        camp.checkpoint(current_attempt={"n": n, "stage": "agent"})
        values = camp.placeholder_values(attempt_dir, attempt_dir)
        command = gates_mod.substitute_command(command, values, expand_vars=isinstance(command, list))
        env, removed = gates_mod.agent_env(camp.base_env(), camp.heldout_vars())
        env["CAMPAIGN_ATTEMPT"] = str(n)
        env["CAMPAIGN_ATTEMPT_DIR"] = str(attempt_dir)
        timeout = float(agent_cfg.get("timeout_s") or budgets.get("agent_timeout_s") or 600.0)
        timeout = min(timeout, max(camp.remaining_attempt_s(attempt_deadline), 1.0))
        log_path = attempt_dir / "agent.log"
        with open(log_path, "wb") as log:
            result = gates_mod.run_child(
                command, env=env, cwd=camp.worktree, timeout_s=timeout,
                log_fh=log, interrupt_check=_interrupt_check,
            )
        entry["agent_log"] = log_path.name
        entry["agent_scrubbed_env"] = removed
        if result["interrupted"]:
            raise Interrupted()
        if result["exit_code"] != 0:
            agent_failed = f"agent exited with {result['exit_code']}"
    if agent_failed is None:
        hyp = _read_json_file(attempt_dir / "hypothesis.json")
        if hyp and isinstance(hyp.get("hypothesis"), str):
            entry["hypothesis"] = hyp["hypothesis"]
        elif not agent:
            pass  # agent-driven: --hypothesis already recorded
        entry["usage"] = _read_json_file(attempt_dir / "usage.json")

    evaluator = Evaluator(camp)
    pending_stop: SafetyStop | None = None

    # ---- authority -------------------------------------------------------
    camp.checkpoint(current_attempt={"n": n, "stage": "authority"})
    if agent_failed is not None:
        entry["verdict"] = "fail"
        entry["reason"] = agent_failed
        entry["gates"] = [{
            "name": "agent", "stage": "agent", "required": True, "objective": False,
            "verdict": "fail", "exit_code": None, "timed_out": False, "interrupted": None,
            "metrics": {}, "divergence": [], "rules": [], "rule_details": [agent_failed],
            "wall_s": 0.0, "resource": {"maxrss_kb": spec.UNAVAILABLE,
                                        "utime_s": spec.UNAVAILABLE, "stime_s": spec.UNAVAILABLE},
            "log": "agent.log", "scrubbed_env": [],
        }]
    else:
        auth = evaluator.authority(camp.state.get("best_sha") or camp.campaign["baseline_revision"])
        entry["changed_paths"] = auth["changed_paths"]
        entry["candidate_sha"] = auth["candidate_sha"]
        auth_record = {
            "name": "authority", "stage": "authority", "required": True, "objective": False,
            "verdict": auth["verdict"], "exit_code": 0, "timed_out": False, "interrupted": None,
            "metrics": {}, "divergence": [], "rules": [],
            "rule_details": [auth["reason"]] if auth["reason"] else [],
            "changed_paths": auth["changed_paths"], "wall_s": 0.0,
            "resource": {"maxrss_kb": spec.UNAVAILABLE, "utime_s": spec.UNAVAILABLE,
                         "stime_s": spec.UNAVAILABLE},
            "log": None, "scrubbed_env": [],
        }
        gates_records: list[dict] = []
        if auth["verdict"] == "fail":
            entry["verdict"] = "fail"
            entry["reason"] = auth["reason"]
            gates_records = [auth_record] + _skip_all(camp, "authority")
        elif auth["verdict"] == "inconclusive":
            entry["verdict"] = "inconclusive"
            entry["reason"] = auth["reason"]
            gates_records = [auth_record] + _skip_all(camp, "authority")
        else:
            gates_records = [auth_record]
            camp.checkpoint(current_attempt={"n": n, "stage": "build"})
            build_record = evaluator.build(attempt_dir, attempt_deadline)
            gates_records.append(build_record)
            if build_record["verdict"] == "pass":
                camp.checkpoint(current_attempt={"n": n, "stage": "gates"})
                try:
                    gates_records.extend(
                        evaluator.gates(attempt_dir, attempt_dir, attempt_deadline,
                                        kind="attempt")
                    )
                except SafetyStop as stop:
                    # Record the partial attempt and revert it BEFORE the stop
                    # propagates: the audit trail keeps it and the worktree
                    # returns to the best commit (never a half-evaluated HEAD).
                    gates_records.extend(getattr(stop, "partial", []))
                    entry["verdict"] = "inconclusive"
                    entry["reason"] = f"campaign stopped mid-attempt: {stop.reason}"
                    entry["stop_reason"] = stop.reason
                    pending_stop = stop
            else:
                gates_records.extend(_skip_all(camp, "build"))
        entry["gates"] = gates_records
        if entry["verdict"] is None:
            entry["verdict"] = gates_mod.attempt_verdict(
                [g for g in gates_records if g["verdict"] != "skipped"]
            )

    # ---- decide: keep or revert ------------------------------------------
    camp.checkpoint(current_attempt={"n": n, "stage": "decision"})
    best_sha = camp.state.get("best_sha") or camp.campaign["baseline_revision"]
    if entry["verdict"] == "pass" and entry["candidate_sha"]:
        entry["kept"] = True
        key_metrics = " ".join(
            f"{k}={v}" for g in entry["gates"] for k, v in (g.get("metrics") or {}).items()
        )
        entry["note"] = f"KEEP: measured win retained ({key_metrics})"
        camp.campaign["best_sha"] = entry["candidate_sha"]
        _rewrite_campaign_json(camp)
        if camp.best_dir.exists():
            shutil.rmtree(camp.best_dir)
        camp.best_dir.mkdir(parents=True)
        _copy_artifacts(attempt_dir, camp.best_dir,
                        camp.profile.get("build", {}).get("artifacts_out", []))
        camp.checkpoint(best_sha=entry["candidate_sha"])
    else:
        cand = entry["candidate_sha"]
        if cand and cand != best_sha:
            patch = camp.wt("diff", best_sha + ".." + cand, check=False).stdout
            (attempt_dir / "change.patch").write_text(patch, encoding="utf-8")
            camp.wt("update-ref", f"refs/campaigns/{camp.task['campaign_id']}/attempt-{n:03d}",
                    cand, check=False)
        camp.wt("reset", "--hard", best_sha)
        camp.wt("clean", "-fd")
    # Both paths: the build dirtied third_party/ggml (configure applies the
    # patch series); leave the worktree pristine for the next authority check.
    _restore_submodules(camp)

    entry["wall_s"] = round(time.monotonic() - started, 1)
    camp.append_ledger(entry)
    camp.checkpoint(
        attempts_used=n,
        current_attempt=None,
        best_sha=camp.campaign["best_sha"],
        tokens_used=camp.state.get("tokens_used", 0)
        + _entry_tokens(entry),
        agent_consecutive_failures=(
            camp.state.get("agent_consecutive_failures", 0) + 1
            if agent_failed else 0
        ),
    )
    if pending_stop is not None:
        raise pending_stop
    return entry


def _entry_tokens(entry: dict) -> int:
    usage = entry.get("usage") or {}
    total = 0
    for key in ("input_tokens", "output_tokens"):
        v = usage.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total += int(v)
    return total


def _restore_submodules(camp: Campaign) -> None:
    """Reset submodule working trees to their pinned commits. Builds apply the
    ggml patch series inside third_party/ggml, and parent-level `reset --hard`
    + `clean -fd` never touch submodule content — without this, submodule dirt
    persists across attempts (and into finalize's rebuild) and would blind the
    authority check's dirty-submodule rejection."""
    camp.wt("submodule", "foreach", "--quiet", "--recursive",
            "git reset -q --hard && git clean -q -fd", check=False)


def _skip_all(camp: Campaign, after_stage: str) -> list[dict]:
    skipped = []
    for gate in applicable_gates(camp.profile["gates"], "attempt"):
        skipped.append({
            "name": gate["name"], "stage": gate["stage"],
            "required": gate.get("required", True), "objective": gate.get("objective", False),
            "verdict": "skipped", "exit_code": None, "timed_out": False, "interrupted": None,
            "metrics": {}, "divergence": [], "rules": gate.get("rules", []),
            "rule_details": [f"skipped: failed before this gate ({after_stage})"],
            "wall_s": 0.0,
            "resource": {"maxrss_kb": spec.UNAVAILABLE, "utime_s": spec.UNAVAILABLE,
                         "stime_s": spec.UNAVAILABLE},
            "log": None, "scrubbed_env": [],
        })
    return skipped


def _rewrite_campaign_json(camp: Campaign) -> None:
    data = dict(camp.campaign)
    data.pop("dir", None)
    tmp = camp.dir / "campaign.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, camp.dir / "campaign.json")


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def _run_loop(camp: Campaign, wait_lock: bool) -> int:
    budgets = camp.task["budgets"]
    max_attempts = budgets.get("max_attempts", 1)
    if not camp.task.get("agent") or not camp.task["agent"].get("command"):
        print("sealed task has no agent command; nothing for `run` to do — use "
              "`attempt` (agent-driven mode) or start with --agent-cmd", file=sys.stderr)
        return EXIT_USAGE

    while True:
        if _interrupt_check():
            return _interrupt_finish(camp)
        if camp.state.get("attempts_used", 0) >= max_attempts:
            return _stop(camp, "max_attempts", EXIT_OK)
        if camp.remaining_campaign_s() <= 0:
            return _stop(camp, "wall_clock", EXIT_OK)
        token_budget = budgets.get("token_budget")
        if token_budget is not None and camp.state.get("tokens_used", 0) >= token_budget:
            return _stop(camp, "token_budget", EXIT_OK)
        if not camp.evaluator_ok():
            return _stop(camp, "evaluator_tampered", EXIT_TAMPER)
        problems = camp.verify_seals()
        if problems:
            print("; ".join(problems), file=sys.stderr)
            return _stop(camp, "evaluator_tampered", EXIT_TAMPER)

        try:
            camp.safety_probe()
            entry = run_attempt(camp, agent=True)
        except Interrupted:
            return _interrupt_finish(camp)
        except SafetyStop as e:
            return _stop(camp, e.reason, EXIT_SAFETY if e.reason in SAFETY_STOPS else EXIT_OK,
                         str(e))
        if entry["verdict"] != "pass":
            print(f"attempt {entry['n']}: {entry['verdict']} "
                  f"({entry.get('reason') or _failed_stage_text(entry)})")
        else:
            print(f"attempt {entry['n']}: PASS, kept ({entry['candidate_sha'][:12]}…)")
        if camp.state.get("agent_consecutive_failures", 0) >= AGENT_FAILURE_STREAK_LIMIT:
            return _stop(camp, "agent_error", EXIT_OK)


def _failed_stage_text(entry: dict) -> str:
    for g in entry.get("gates", []):
        if g.get("verdict") == "fail":
            return f"failed at {g.get('stage')}/{g.get('name')}"
    return "no gate failed"


def _stop(camp: Campaign, reason: str, exit_code: int, message: str = "") -> int:
    camp.checkpoint(status="stopped", stop_reason=reason, current_attempt=None,
                    finished_at=_now_iso())
    text = f"campaign stopped: {reason}"
    if message:
        text += f" — {message}"
    print(text)
    return exit_code


def _interrupt_finish(camp: Campaign) -> int:
    best_sha = camp.state.get("best_sha") or camp.campaign["baseline_revision"]
    camp.wt("reset", "--hard", best_sha, check=False)
    camp.wt("clean", "-fd", check=False)
    _restore_submodules(camp)
    # current_attempt is KEPT so `resume` records the interrupted attempt
    # (it counts toward the budget) instead of silently re-running it.
    camp.checkpoint(status="interrupted", stop_reason="interrupted",
                    finished_at=_now_iso())
    print("interrupted: checkpoint saved (status=interrupted); run `resume` to continue",
          file=sys.stderr)
    return EXIT_INTERRUPT


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _load_profile_arg(args) -> dict:
    if getattr(args, "profile_file", None):
        return spec.load_profile(Path(args.profile_file))
    profiles = spec.load_profiles(PROFILES_DIR)
    pid = args.profile
    if pid not in profiles:
        raise CampaignError(
            f"unknown profile {pid!r}; available: {', '.join(sorted(profiles))} "
            "(or pass --profile-file PATH)"
        )
    return profiles[pid]


def _load_task_arg(args, profile: dict, baseline_sha: str) -> dict:
    task_dict = getattr(args, "task_dict", None)
    if task_dict is not None:
        task = dict(task_dict)
    elif getattr(args, "task", None):
        task = spec.load_json(Path(args.task))
    else:
        task = generate_task(profile, baseline_sha, args)
    task["baseline_revision"] = baseline_sha  # the CLI resolves and seals the SHA
    # A blocked profile has no gates to reference; let preflight report the
    # blocked status instead of a confusing task cross-validation error.
    problems = spec.validate_task(task, None if profile.get("status") == "blocked" else profile)
    if problems:
        raise CampaignError("invalid task: " + "; ".join(problems))
    return task


def cmd_list(args) -> int:
    profiles = spec.load_profiles(PROFILES_DIR)
    print(f"{'profile':<28} {'model':<28} {'dev':<9} {'eng':<6} {'status':<8} tracking")
    for pid, p in sorted(profiles.items()):
        status = p["status"]
        if status == "blocked":
            status += f" ({p.get('blocked_on')})"
        print(f"{pid:<28} {p['model']['name'][:28]:<28} {p['device']:<9} "
              f"{p['engine']:<6} {status:<8} {p['tracking_issue']}")
    print(f"{len(profiles)} profiles")
    return EXIT_OK


def cmd_preview(args) -> int:
    repo = Path(args.repo) if args.repo else default_repo()
    profile = _load_profile_arg(args)
    baseline_sha = resolve_revision(repo, args.baseline or "HEAD")
    task = _load_task_arg(args, profile, baseline_sha)
    out = Path(args.out) if getattr(args, "out", None) else Path(tempfile.mkdtemp(prefix="preview-"))
    problems, context = preflight(profile, task, repo, out,
                                  adb_serial=getattr(args, "adb_serial", None))
    print(f"== preview: campaign {task['campaign_id']} (mutates nothing) ==")
    if problems:
        print("prerequisites missing:")
        for p in problems:
            print(f"  - {p}")
        return EXIT_USAGE
    _print_plan(profile, task, context)
    print("ready: all prerequisites satisfied")
    return EXIT_OK


def cmd_start(args) -> int:
    repo = Path(args.repo) if args.repo else default_repo()
    profile = _load_profile_arg(args)
    baseline_sha = resolve_revision(repo, args.baseline)
    task = _load_task_arg(args, profile, baseline_sha)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise CampaignError(f"campaign dir {out} exists and is not empty")

    lock_profile = dict(profile)
    if args.adb_serial:
        lock_profile["adb_serial"] = args.adb_serial
    with campaign_locks(lock_profile, wait=args.wait_lock):
        problems, context = preflight(profile, task, repo, out, adb_serial=args.adb_serial,
                                      assume_lock_held=True)
        if problems:
            print("prerequisites missing:")
            for p in problems:
                print(f"  - {p}")
            return EXIT_USAGE
        if out.exists() and any(out.iterdir()):
            raise CampaignError(f"campaign dir {out} exists and is not empty")
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("attempts", "baseline", "best", "trusted"):
            (out / sub).mkdir(exist_ok=True)
        worktree = out / "worktree"
        git(repo, "worktree", "add", "--detach", str(worktree), baseline_sha)
        branch = f"campaign/{task['campaign_id']}"
        switched = git(worktree, "switch", "-c", branch, check=False)
        suffix = 1
        while switched.returncode != 0:
            branch = f"campaign/{task['campaign_id']}-{suffix}"
            switched = git(worktree, "switch", "-c", branch, check=False)
            suffix += 1
            if suffix > 99:
                raise CampaignError(f"cannot create a fresh campaign branch in {worktree}")
        if (worktree / ".gitmodules").is_file():
            # A linked worktree starts with empty submodules; every native
            # build needs third_party/ggml at the baseline's pinned commit.
            git(worktree, "submodule", "update", "--init", "--recursive", timeout=1800)

        extraction = extract_trusted(repo, baseline_sha, default_trusted_paths(profile),
                                     out / "trusted")
        evaluator_sha = spec.hash_tree(out / "trusted")

        artifact_seals = {}
        for artifact in profile.get("artifacts", []):
            path = spec.expand_artifact_path(artifact["path"])
            artifact_seals[artifact["name"]] = spec.sha256_file(path)

        seals = {
            "task": spec.seal_of(task),
            "profile": spec.seal_of(profile),
            "evaluator": evaluator_sha,
            "artifacts": artifact_seals,
            "heldout_pin": getattr(args, "heldout_pin", None) or (profile.get("heldout") or {}).get("sha256"),
        }
        (out / "task.sealed.json").write_text(
            json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "task.sealed.sha256").write_text(seals["task"] + "\n", encoding="utf-8")
        campaign_doc = {
            "schema": spec.STATE_SCHEMA,
            "campaign_id": task["campaign_id"],
            "repo_root": str(repo),
            "profile": profile,
            "task": task,
            "seals": seals,
            "baseline_revision": baseline_sha,
            "best_sha": baseline_sha,
            "adb_serial": args.adb_serial,
            "evaluator_paths": extraction["paths"],
            "evaluator_skipped_paths": extraction["skipped"],
            "device_identity": context["device_identity"],
            "created_at": _now_iso(),
            "started_at": _now_iso(),
        }
        (out / "campaign.json").write_text(
            json.dumps(campaign_doc, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")
        (out / "state.json").write_text(json.dumps({
            "schema": spec.STATE_SCHEMA, "status": "starting", "stop_reason": None,
            "attempts_used": 0, "current_attempt": None, "best_sha": baseline_sha,
            "tokens_used": 0, "agent_consecutive_failures": 0,
            "updated_at": _now_iso(), "started_at": _now_iso(), "finished_at": None,
            "finalize": None,
        }, indent=2) + "\n", encoding="utf-8")

        camp = Campaign(out)

        # Build the baseline once, copy artifacts to baseline/ and best/.
        camp.checkpoint(status="building-baseline")
        evaluator = Evaluator(camp)
        build_record = evaluator.build(camp.baseline_dir, time.monotonic() + UNCAPPED_OFFSET_S)
        if build_record["verdict"] != "pass":
            print(f"baseline build failed:\n{build_record['rule_details']}", file=sys.stderr)
            return EXIT_USAGE
        # The baseline build dirtied third_party/ggml (patch series); restore
        # it so attempt 1's authority check starts from a pristine submodule.
        _restore_submodules(camp)
        _copy_artifacts(camp.baseline_dir, camp.best_dir,
                        profile.get("build", {}).get("artifacts_out", []))

        # Control attempt 0: the full gate set baseline-vs-baseline. This is
        # the negative control AND the baseline measurement record.
        camp.checkpoint(status="control")
        control_dir = camp.dir / "attempts" / "000"
        control_dir.mkdir(exist_ok=True)
        try:
            camp.safety_probe()
            control_gates = evaluator.gates(control_dir, camp.baseline_dir,
                                            time.monotonic() + UNCAPPED_OFFSET_S, kind="control")
        except (SafetyStop, Interrupted) as e:
            print(f"control attempt could not run: {e}", file=sys.stderr)
            return EXIT_USAGE
        control_entry = {
            "n": 0, "kind": "control", "hypothesis": "baseline control (baseline vs baseline)",
            "verdict": gates_mod.attempt_verdict(control_gates), "gates": control_gates,
            "changed_paths": [], "candidate_sha": baseline_sha, "usage": None,
            "kept": False, "wall_s": 0.0, "started_at": _now_iso(), "stop_reason": None,
            "note": "control: baseline cannot beat itself; records baseline metrics",
        }
        camp.append_ledger(control_entry)
        # Baseline vs baseline must PASS every required non-objective gate
        # (correctness, quality, resource). If the evaluator cannot even judge
        # the baseline against itself (binary path wrong, fixture missing,
        # nondeterministic output), every attempt would come out a silent
        # "inconclusive" — refuse to start instead of burning the budget.
        undecided = [
            f"{g['name']} ({g['verdict']}: {'; '.join(g.get('rule_details') or []) or 'see log'})"
            for g in control_gates
            if g.get("required", True) and not g.get("objective") and g["verdict"] != "pass"
        ]
        if undecided:
            camp.checkpoint(status="control_failed", current_attempt=None,
                            stop_reason="control_failed")
            print("control attempt 0 failed: the evaluator cannot confirm the baseline "
                  "against itself — fix the profile/environment before any attempt:",
                  file=sys.stderr)
            for line in undecided:
                print(f"  - {line}", file=sys.stderr)
            print(f"  logs: {control_dir}", file=sys.stderr)
            return EXIT_USAGE
        camp.checkpoint(status="ready", current_attempt=None)
        print(f"campaign ready at {out}")
        print(f"  baseline {baseline_sha[:12]}…  evaluator {seals['evaluator'][:12]}… "
              f"({extraction['files']} files)")
        if extraction["skipped"]:
            print(f"  note: trusted paths missing at baseline (skipped): "
                  f"{', '.join(extraction['skipped'])}")
        print(f"next: python {Path(__file__).name} run --campaign {out}")
    return EXIT_OK


def _open_campaign(args) -> Campaign:
    camp = Campaign(Path(args.campaign))
    problems = camp.verify_seals()
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        raise CampaignError("seal verification failed (evaluation rules changed after "
                            "preregistration)", exit_code=EXIT_TAMPER)
    return camp


def _refuse_unusable(camp: Campaign) -> int | None:
    """A campaign whose control failed (or never finished start) has no
    trustworthy evaluator; no attempt may run on it."""
    status = camp.state.get("status")
    if status in ("control_failed", "starting", "building-baseline", "control"):
        print(f"refusing: campaign status is {status!r} — start did not complete a "
              "passing control attempt; fix the cause and start a fresh campaign",
              file=sys.stderr)
        return EXIT_USAGE
    return None


def _needs_recovery(camp: Campaign) -> bool:
    return camp.state.get("current_attempt") is not None and camp.state.get("status") in (
        "running", "interrupted"
    )


def cmd_run(args) -> int:
    camp = _open_campaign(args)
    refused = _refuse_unusable(camp)
    if refused is not None:
        return refused
    with campaign_locks(camp.profile_with_serial(), wait=args.wait_lock):
        # Recovery mutates the worktree: only under the campaign lock.
        if _needs_recovery(camp):
            _recover_interrupted(camp)
        camp.checkpoint(status="running", stop_reason=None, finished_at=None)
        return _run_loop(camp, wait_lock=args.wait_lock)


def _recover_interrupted(camp: Campaign) -> None:
    """An attempt was running when the campaign died or was signalled: record
    it as interrupted (it counts toward the budget) and reset the worktree."""
    current = camp.state.get("current_attempt") or {}
    n = current.get("n")
    if n:
        entry = {
            "n": n, "kind": "attempt", "hypothesis": spec.UNAVAILABLE,
            "verdict": "interrupted", "reason": "recovered interrupted attempt",
            "gates": [], "changed_paths": [], "candidate_sha": None, "usage": None,
            "kept": False, "wall_s": 0.0, "started_at": _now_iso(), "stop_reason": "interrupted",
        }
        camp.append_ledger(entry)
        best_sha = camp.state.get("best_sha") or camp.campaign["baseline_revision"]
        camp.wt("reset", "--hard", best_sha, check=False)
        camp.wt("clean", "-fd", check=False)
        # A hard kill can strike mid-build: the build's ggml patch series is
        # still applied inside third_party/ggml, and the next attempt's
        # authority check would blame the candidate for that dirt. Restore.
        _restore_submodules(camp)
        camp.checkpoint(attempts_used=n, current_attempt=None, status="ready")
    else:
        camp.checkpoint(status="ready", current_attempt=None)


def cmd_resume(args) -> int:
    camp = _open_campaign(args)
    refused = _refuse_unusable(camp)
    if refused is not None:
        return refused
    with campaign_locks(camp.profile_with_serial(), wait=args.wait_lock):
        if _needs_recovery(camp):
            _recover_interrupted(camp)
        if not camp.evaluator_ok():
            return _stop(camp, "evaluator_tampered", EXIT_TAMPER)
        try:
            camp.safety_probe()
        except SafetyStop as e:
            # The stop condition persists (wedge marker, adb gone, hot, ...):
            # refuse without touching the device. Never retry into a wedge.
            print(f"refusing to resume: {e.reason} — {e}", file=sys.stderr)
            return _stop(camp, e.reason,
                         EXIT_SAFETY if e.reason in SAFETY_STOPS else EXIT_OK)
        except Interrupted:
            return _interrupt_finish(camp)
        camp.checkpoint(status="running", stop_reason=None, finished_at=None)
        return _run_loop(camp, wait_lock=args.wait_lock)


def cmd_attempt(args) -> int:
    camp = _open_campaign(args)
    refused = _refuse_unusable(camp)
    if refused is not None:
        return refused
    budgets = camp.task["budgets"]
    with campaign_locks(camp.profile_with_serial(), wait=args.wait_lock):
        if _needs_recovery(camp):
            _recover_interrupted(camp)
        # AFTER recovery: the interrupted attempt now counts toward the
        # budget, and this command must not start one past max_attempts.
        if camp.state.get("attempts_used", 0) >= budgets.get("max_attempts", 1):
            print("attempt budget already exhausted; nothing to do")
            return EXIT_OK
        if not camp.evaluator_ok():
            return _stop(camp, "evaluator_tampered", EXIT_TAMPER)
        problems = camp.verify_seals()
        if problems:
            print("; ".join(problems), file=sys.stderr)
            return _stop(camp, "evaluator_tampered", EXIT_TAMPER)
        if camp.remaining_campaign_s() <= 0:
            return _stop(camp, "wall_clock", EXIT_OK)
        token_budget = budgets.get("token_budget")
        if token_budget is not None and camp.state.get("tokens_used", 0) >= token_budget:
            return _stop(camp, "token_budget", EXIT_OK)
        try:
            camp.safety_probe()
            entry = run_attempt(camp, agent=False, hypothesis=args.hypothesis)
        except Interrupted:
            return _interrupt_finish(camp)
        except SafetyStop as e:
            return _stop(camp, e.reason, EXIT_SAFETY if e.reason in SAFETY_STOPS else EXIT_OK,
                         str(e))
        print(f"attempt {entry['n']}: {entry['verdict']}"
              + (" (kept)" if entry["kept"] else ""))
        return EXIT_OK


def cmd_status(args) -> int:
    camp = Campaign(Path(args.campaign))
    c = camp.campaign
    s = camp.state
    print(f"campaign     {c['campaign_id']} ({c['profile']['id']})")
    print(f"status       {s.get('status')} stop_reason={s.get('stop_reason')}")
    print(f"baseline     {c['baseline_revision']}")
    print(f"best         {c.get('best_sha')} (kept: {c.get('best_sha') != c['baseline_revision']})")
    print(f"attempts     {s.get('attempts_used', 0)}/{c['task']['budgets'].get('max_attempts')} "
          f"tokens {s.get('tokens_used', 0)}/{c['task']['budgets'].get('token_budget')}")
    print(f"evaluator    {'ok' if camp.evaluator_ok() else 'TAMPERED'} "
          f"({c['seals']['evaluator'][:12]}…)")
    print("attempts:")
    for e in camp.ledger():
        failed = next((g["stage"] + "/" + g["name"] for g in e.get("gates", [])
                       if g.get("verdict") == "fail"), "")
        print(f"  {e.get('n'):>3} {e.get('kind'):<8} {str(e.get('verdict')):<14} "
              f"{failed:<24} {str(e.get('hypothesis'))[:60]}")
    return EXIT_OK


def cmd_report(args) -> int:
    camp = _open_campaign(args)
    ledger = camp.ledger()
    rep = report_mod.build_report(camp.campaign, camp.state, ledger)
    out_md = Path(args.out) if getattr(args, "out", None) else camp.dir / "report.md"
    if getattr(args, "out", None):
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report_mod.render_markdown(rep), encoding="utf-8")
        (out_md.with_suffix(".json")).write_text(
            json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    else:
        report_mod.write_report(rep, camp.dir)
    print(f"report written: {out_md}")
    return EXIT_OK


def cmd_finalize(args) -> int:
    camp = _open_campaign(args)
    # The guard the other commands use: a campaign whose control failed has a
    # proven-untrustworthy evaluator, and finalize is the most consequential
    # step (promotion) — it must refuse too, not emit a finalize verdict.
    refused = _refuse_unusable(camp)
    if refused is not None:
        return refused
    with campaign_locks(camp.profile_with_serial(), wait=args.wait_lock):
        if not camp.evaluator_ok():
            return _stop(camp, "evaluator_tampered", EXIT_TAMPER)
        problems = camp.verify_seals()
        if problems:
            print("; ".join(problems), file=sys.stderr)
            return _stop(camp, "evaluator_tampered", EXIT_TAMPER)
        identity = monitors.discover(camp.profile_with_serial())
        match = monitors.device_matches(camp.profile, identity)
        if match is False:
            print("refusing to finalize: device mismatch — never substitute another "
                  "device's result", file=sys.stderr)
            return EXIT_USAGE
        best_sha = camp.campaign["best_sha"]
        baseline_sha = camp.campaign["baseline_revision"]
        camp.wt("reset", "--hard", best_sha)
        camp.wt("clean", "-fd")
        # Parent-level reset/clean never touch submodule content: restore it,
        # or leftover submodule edits would leak into the best-commit rebuild.
        _restore_submodules(camp)
        finalize_dir = camp.dir / "finalize"
        if finalize_dir.exists():
            shutil.rmtree(finalize_dir)
        finalize_dir.mkdir(parents=True)
        evaluator = Evaluator(camp)
        try:
            camp.safety_probe()
            build_record = evaluator.build(finalize_dir, time.monotonic() + UNCAPPED_OFFSET_S)
            if build_record["verdict"] != "pass":
                print("cannot rebuild the best commit under the sealed build",
                      file=sys.stderr)
                return EXIT_USAGE
            # Re-validate the exact best commit/artifact against the ORIGINAL
            # baseline (issue #176 §5): at finalize the comparison side is the
            # baseline, so `{best}` resolves to the baseline artifact dir.
            finalize_values = camp.placeholder_values(finalize_dir, finalize_dir)
            finalize_values["best"] = str(camp.baseline_dir)
            gates_records = evaluator.gates(finalize_dir, finalize_dir,
                                            time.monotonic() + UNCAPPED_OFFSET_S, kind="finalize",
                                            values=finalize_values)
        except (SafetyStop, Interrupted) as e:
            _restore_submodules(camp)
            print(f"finalize stopped: {e}", file=sys.stderr)
            return EXIT_SAFETY if isinstance(e, SafetyStop) else EXIT_INTERRUPT
        _restore_submodules(camp)
        verdict = gates_mod.attempt_verdict(gates_records)

        heldout_info = {"provided": False, "pin": None, "sha256": None, "verdict": None}
        if getattr(args, "heldout", None):
            heldout_dir = Path(args.heldout).resolve()
            if not heldout_dir.is_dir():
                raise CampaignError(f"held-out dir {heldout_dir} does not exist")
            pin = camp.campaign["seals"].get("heldout_pin")
            actual = spec.hash_tree(heldout_dir)
            if pin and actual != pin:
                raise CampaignError(
                    f"held-out corpus hash mismatch: pinned {pin[:12]}… != actual {actual[:12]}…"
                )
            heldout_info.update({"provided": True, "pin": pin, "sha256": actual, "pinned": bool(pin)})
            heldout_gates = camp.profile.get("heldout_gates") or []
            hg_records = []
            if heldout_gates:
                heldout_values = camp.placeholder_values(finalize_dir, finalize_dir, heldout_dir)
                heldout_values["best"] = str(camp.baseline_dir)
                for gate in gates_mod.ordered_gates(heldout_gates):
                    log_path = finalize_dir / f"gate-{gate['name']}.log"
                    hg_records.append(gates_mod.run_gate(
                        gate, heldout_values, camp.base_env(), log_path,
                        default_timeout_s=evaluator.gate_timeout_default,
                        heldout_vars=camp.heldout_vars(), cwd=camp.worktree,
                        interrupt_check=_interrupt_check,
                    ))
                heldout_info["verdict"] = gates_mod.attempt_verdict(hg_records)
                gates_records.extend(heldout_records(hg_records))
            else:
                heldout_info["note"] = "profile defines no held-out gates"
        if verdict == "pass" and heldout_info.get("verdict") not in (None, "pass"):
            verdict = heldout_info["verdict"]

        finalize_record = {
            "finished_at": _now_iso(),
            "best_sha": best_sha, "baseline_sha": baseline_sha,
            "verdict": verdict, "gates": gates_records,
            "device_identity": identity, "heldout": heldout_info,
        }
        camp.checkpoint(status="finalized", finalize=finalize_record, finished_at=_now_iso())
        rep = report_mod.build_report(camp.campaign, camp.state, camp.ledger())
        report_mod.write_report(rep, camp.dir)
        print(f"finalize verdict: {verdict} (held-out provided: {heldout_info['provided']})")
        print(f"promotion: {rep['promotion']['status']}")
    return EXIT_OK


def heldout_records(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        r = dict(r)
        r["stage"] = "heldout"
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------

def cmd_pilot(args) -> int:
    """Hermetic toy pilot: the full start -> run -> report pipeline on a toy
    repo, with every negative control from issue #176 §7."""
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise CampaignError(f"pilot out dir {out} exists and is not empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    toy_repo_dir = out.parent / f"{out.name}-toyrepo"
    baseline = _toy_baseline(toy_repo_dir)
    profile = toy_mod.toy_profile(toy_repo_dir)
    agent_cmd = ["{python}", str(HERE / "toy_agent.py")]
    task = toy_mod.toy_task(profile["id"], baseline, agent_cmd)

    ns = argparse.Namespace(
        profile=None, profile_file=None, repo=str(toy_repo_dir),
        baseline=baseline, out=str(out), task=None,
        max_attempts=None, wall_clock=None, allowed=None, agent_cmd=None,
        adb_serial=None, wait_lock=False, heldout_pin=None,
    )
    # start (writes the toy profile to a file named <id>.json so the
    # profile loader's id<->filename invariant holds)
    profile_dir = out.parent / f"{out.name}-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_file = profile_dir / f"{profile['id']}.json"
    profile_file.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    ns.profile_file = str(profile_file)
    ns.task_dict = task
    rc = cmd_start(ns)
    if rc != EXIT_OK:
        return rc
    ns_campaign = argparse.Namespace(campaign=str(out), wait_lock=False)
    rc = cmd_run(ns_campaign)
    if rc != EXIT_OK:
        print(f"pilot run exited {rc}", file=sys.stderr)
        return rc
    rc = cmd_report(argparse.Namespace(campaign=str(out), out=None))
    if rc != EXIT_OK:
        return rc

    camp = Campaign(out)
    ledger = camp.ledger()
    attempts = [e for e in ledger if e.get("kind") == "attempt"]
    expected = [("pass", None), ("fail", "correctness"), ("fail", "authority"),
                ("fail", "correctness"), ("fail", "perf")]
    failures = []
    got = [(e["verdict"], _first_failed_stage(e)) for e in attempts]
    if got != expected:
        failures.append(f"verdict sequence {got} != {expected}")
    if len(ledger) != 6:
        failures.append(f"ledger has {len(ledger)} entries (control + 5 attempts expected)")
    kept = [e for e in attempts if e.get("kept")]
    if len(kept) != 1 or kept[0]["n"] != 1:
        failures.append("best must be attempt 1 (the only measured win)")
    improved = camp.campaign["best_sha"] != camp.campaign["baseline_revision"]
    if not improved:
        failures.append("best_sha must advance past the baseline")
    report_text = (out / "report.md").read_text(encoding="utf-8")
    for needle in ("Exact rerun commands", "gh pr create --draft",
                   "baseline control", "transcripts_match"):
        if needle not in report_text:
            failures.append(f"report.md missing {needle!r}")
    if camp.state.get("stop_reason") != "max_attempts":
        failures.append(f"stop_reason {camp.state.get('stop_reason')!r} != 'max_attempts'")

    print("== pilot summary ==")
    for e in ledger:
        stage = _first_failed_stage(e) or ""
        print(f"  {e['n']:>3} {e['kind']:<8} {e['verdict']:<14} {stage:<12} "
              f"{str(e.get('hypothesis'))[:64]}")
    if failures:
        print("PILOT FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("pilot OK: attempt 1 kept; incorrect, skip-validation, fake-speedup and "
          "no-op patches all rejected at the right stages; ledger complete.")
    return EXIT_OK


def _toy_baseline(toy_repo_dir: Path) -> str:
    """Rebuild the pilot toy repo when the existing dir is exactly our toy
    baseline; never touch anything else."""
    if toy_repo_dir.exists() and any(toy_repo_dir.iterdir()):
        head = subprocess.run(
            ["git", "-C", str(toy_repo_dir), "log", "-1", "--format=%s"],
            capture_output=True, text=True,
        )
        if head.returncode == 0 and head.stdout.strip() == "toy baseline: engine + evaluator":
            shutil.rmtree(toy_repo_dir)
        else:
            raise CampaignError(f"{toy_repo_dir} exists and is not a pilot toy repo")
    return toy_mod.build_toy_repo(toy_repo_dir)


def _first_failed_stage(entry: dict) -> str | None:
    for g in entry.get("gates", []):
        if g.get("verdict") == "fail":
            return g.get("stage")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="campaign.py",
        description="Bounded hypothesis -> experiment -> review campaign runner (#176)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list committed campaign profiles")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("preview", help="resolve + preflight, mutate nothing (dry-run)")
    p.add_argument("--repo", help="target repository (default: git toplevel of cwd)")
    p.add_argument("--profile", help="profile id from benchmarks/campaign/profiles/")
    p.add_argument("--profile-file", type=Path, help="... or a profile JSON path")
    p.add_argument("--baseline", help="baseline revision (default HEAD)")
    p.add_argument("--task", type=Path, help="task spec JSON (default: generated from profile)")
    p.add_argument("--out", type=Path, help="planned campaign dir (for disk/preflight checks)")
    p.add_argument("--adb-serial")
    p.add_argument("--max-attempts", type=int)
    p.add_argument("--wall-clock", help="e.g. 8h / 90m / 3600")
    p.add_argument("--allowed", nargs="+", help="allowed path globs")
    p.add_argument("--agent-cmd", help="agent command template")
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser("start", help="preflight, seal, worktree, control attempt")
    p.add_argument("--repo", help="target repository (default: git toplevel of cwd)")
    p.add_argument("--profile")
    p.add_argument("--profile-file", type=Path)
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--task", type=Path)
    p.add_argument("--max-attempts", type=int)
    p.add_argument("--wall-clock")
    p.add_argument("--allowed", nargs="+")
    p.add_argument("--agent-cmd")
    p.add_argument("--adb-serial")
    p.add_argument("--heldout-pin", help="sha256 pin for the held-out corpus (sealed at start)")
    p.add_argument("--wait-lock", action="store_true")
    p.set_defaults(fn=cmd_start)

    p = sub.add_parser("run", help="agent loop until a budget or a safety stop")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--wait-lock", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("attempt", help="agent-driven: evaluate the worktree's current change once")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--wait-lock", action="store_true")
    p.set_defaults(fn=cmd_attempt)

    p = sub.add_parser("resume", help="recover an interrupted/stopped campaign and continue")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--wait-lock", action="store_true")
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("status", help="print campaign state + attempt table")
    p.add_argument("--campaign", required=True, type=Path)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("report", help="write report.md + report.json")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--out", type=Path, help="report path (default <campaign>/report.md)")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("finalize", help="revalidate best vs baseline (+ held-out gate)")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--heldout", type=Path, help="held-out corpus dir (uses it ONLY here)")
    p.add_argument("--wait-lock", action="store_true")
    p.set_defaults(fn=cmd_finalize)

    p = sub.add_parser("pilot", help="hermetic toy pilot (negative controls included)")
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(fn=cmd_pilot)

    args = parser.parse_args(argv)
    _install_signal_handlers()
    try:
        return args.fn(args)
    except (CampaignError, SpecError, LockHeld) as e:
        code = getattr(e, "exit_code", EXIT_USAGE)
        print(f"error: {e}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
