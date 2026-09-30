"""Gate execution: process-group isolation, timeouts, credential scrubbing,
METRIC parsing, acceptance rules and verdicts (issue #176 §3).

A gate is a trusted subprocess (always resolved from the extracted evaluator
tree, never from the candidate worktree) that prints `METRIC name=value`
lines — the phone_gates.sh convention — and optionally `DIVERGENCE: <text>`
lines describing the first observable difference from the baseline. The
runner gives it a scrubbed environment, its own process group, a timeout
enforced with SIGTERM -> 10 s -> SIGKILL on the whole group, and records
wall time and child resource usage.

Binding rules:

- Exit code != 0 -> gate `fail` (log tail as diagnostics), EXCEPT exit 3 ->
  `inconclusive` ("ran but cannot decide", e.g. charging invalidated an
  energy measurement). A missing metric or a value of `unavailable` makes a
  rule `inconclusive`, never `pass`.
- Gate/build subprocesses get NO provider, cloud, release or held-out
  credentials. The agent subprocess keeps provider credentials (the
  orchestration boundary) but loses held-out, release/signing/GitHub ones.
  Only scrubbed variable NAMES are recorded, never values.
- Every child runs with `start_new_session=True` so a timeout (or an
  interrupt of the campaign) cannot leave grandchildren behind.

Stdlib only, Python 3.10+.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

UNAVAILABLE = "unavailable"

# Placeholders substituted in argv/shell/build/agent command templates. Only
# these exact tokens are replaced; everything else (shell `$VAR`, awk
# braces) passes through untouched.
PLACEHOLDERS = (
    "trusted", "worktree", "candidate", "baseline", "best", "attempt_dir",
    "campaign_dir", "python", "adb_serial", "heldout",
)

# Environment variable scrubbing. Gate/build children get no credentials at
# all; the agent keeps provider keys (orchestration boundary) but not
# held-out, release/signing or GitHub credentials.
GATE_ENV_DENY = (
    "*_API_KEY", "*_TOKEN", "*_SECRET*", "*PASSWORD*",
    "AWS_*", "AZURE_*", "VERTEX_*",
    "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_GENERATIVE_AI_API_KEY",
    "GEMINI_API_KEY", "ANTHROPIC_*", "OPENAI_*", "OPENROUTER_*",
    "GH_*", "GITHUB_TOKEN", "ACTIONS_ID_TOKEN_REQUEST*",
    "*KEYSTORE*", "*SIGNING*",
)
AGENT_ENV_DENY = (
    "GH_*", "GITHUB_TOKEN", "ACTIONS_ID_TOKEN_REQUEST*",
    "*KEYSTORE*", "*SIGNING*",
)

_METRIC_RE = re.compile(r"^METRIC\s+([A-Za-z0-9_.\-]+)=(\S+)[ \t]*$", re.MULTILINE)
_DIVERGENCE_RE = re.compile(r"^DIVERGENCE:\s?(.*)$", re.MULTILINE)
_KILL_GRACE_S = 10.0

STAGE_ORDER = {name: i for i, name in enumerate(
    ("authority", "build", "correctness", "diagnostic", "perf", "quality", "resource")
)}


class GateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------

def substitute(text: str, values: dict[str, str]) -> str:
    """Replace only the known `{token}` placeholders; leave the rest alone."""
    out = text
    for name in PLACEHOLDERS:
        if name in values:
            out = out.replace("{" + name + "}", str(values[name]))
    return out


def substitute_command(command, values: dict[str, str], *, expand_vars: bool = False):
    """Substitute placeholders in an argv list or shell string.

    `expand_vars` additionally expands $ENV / ${ENV} in argv entries (used
    for model-artifact paths such as ${STARLING_MODELS_DIR}/...). It is
    never applied to shell strings: those must keep their own $-syntax for
    the shell that will run them.
    """
    if isinstance(command, str):
        return substitute(command, values)
    argv = [substitute(str(a), values) for a in command]
    if expand_vars:
        argv = [os.path.expandvars(a) for a in argv]
    return argv


# ---------------------------------------------------------------------------
# Credential scrubbing
# ---------------------------------------------------------------------------

def _name_matches(name: str, patterns) -> bool:
    import fnmatch

    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def scrub_env(
    env: dict[str, str],
    deny_patterns=(),
    extra_names: list[str] = (),
) -> tuple[dict[str, str], list[str]]:
    """Return (scrubbed env, sorted removed variable NAMES)."""
    deny = tuple(deny_patterns)
    removed = []
    out = {}
    for key, value in env.items():
        if key in extra_names or _name_matches(key, deny):
            removed.append(key)
        else:
            out[key] = value
    return out, sorted(removed)


def gate_env(base_env: dict[str, str], heldout_vars: list[str] = ()) -> tuple[dict[str, str], list[str]]:
    return scrub_env(base_env, GATE_ENV_DENY, heldout_vars)


def agent_env(base_env: dict[str, str], heldout_vars: list[str] = ()) -> tuple[dict[str, str], list[str]]:
    return scrub_env(base_env, AGENT_ENV_DENY, heldout_vars)


# ---------------------------------------------------------------------------
# METRIC parsing + rules
# ---------------------------------------------------------------------------

def parse_metric_value(raw: str):
    if raw == UNAVAILABLE:
        return UNAVAILABLE
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_output(text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse METRIC and DIVERGENCE lines (later METRIC lines win)."""
    metrics: dict[str, Any] = {}
    for name, value in _METRIC_RE.findall(text):
        metrics[name] = parse_metric_value(value)
    divergences = [d.strip() for d in _DIVERGENCE_RE.findall(text)]
    return metrics, divergences


def evaluate_rules(metrics: dict[str, Any], rules: list[dict]) -> tuple[str, list[str]]:
    """Apply `rules` to parsed metrics -> ("pass"|"fail"|"inconclusive", details)."""
    details = []
    verdict = "pass"
    for rule in rules:
        name = rule["metric"]
        op = rule["op"]
        expected = rule["value"]
        if name not in metrics:
            verdict = _worse(verdict, "inconclusive")
            details.append(f"metric {name!r} missing -> inconclusive")
            continue
        got = metrics[name]
        if got == UNAVAILABLE:
            verdict = _worse(verdict, "inconclusive")
            details.append(f"metric {name}={UNAVAILABLE} -> inconclusive")
            continue
        got_num = isinstance(got, (int, float)) and not isinstance(got, bool)
        exp_num = isinstance(expected, (int, float)) and not isinstance(expected, bool)
        if got_num != exp_num or (op in ("<", "<=", ">", ">=") and not got_num):
            verdict = _worse(verdict, "inconclusive")
            details.append(
                f"metric {name}={got!r} and rule value {expected!r} are not comparable "
                f"under {op!r} -> inconclusive"
            )
            continue
        ok = _compare(got, op, expected)
        details.append(f"{name}={got!r} {op} {expected!r} -> {'ok' if ok else 'violated'}")
        if not ok:
            verdict = "fail"
    return verdict, details


def _worse(current: str, new: str) -> str:
    if current == "fail":
        return "fail"
    if current == "inconclusive":
        return "inconclusive"
    return new


def _compare(got, op, expected) -> bool:
    if op == "==":
        return got == expected
    if op == "!=":
        return got != expected
    if not isinstance(got, (int, float)) or not isinstance(expected, (int, float)):
        return False  # ordering is only defined for numbers
    if op == "<":
        return got < expected
    if op == "<=":
        return got <= expected
    if op == ">":
        return got > expected
    if op == ">=":
        return got >= expected
    raise GateError(f"unknown op {op!r}")


def ordered_gates(gates: list[dict]) -> list[dict]:
    """Gates in the fixed stage order, preserving profile order within a stage."""
    indexed = list(enumerate(gates))
    indexed.sort(key=lambda pair: (STAGE_ORDER.get(pair[1].get("stage"), 99), pair[0]))
    return [g for _i, g in indexed]


def attempt_verdict(results: list[dict]) -> str:
    """pass iff no fail and every required (incl. objective) gate passed.
    Skipped gates never ran (fail-fast) and do not count against pass."""
    if any(r["verdict"] == "fail" for r in results):
        return "fail"
    if all(
        r["verdict"] == "pass"
        for r in results
        if r["verdict"] != "skipped" and (r.get("required", True) or r.get("objective"))
    ):
        return "pass"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Child execution
# ---------------------------------------------------------------------------

def _signal_pg(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def run_child(
    command,
    *,
    env: dict[str, str],
    cwd,
    timeout_s: float,
    log_fh,
    interrupt_check: Callable[[], Any] | None = None,
) -> dict:
    """Run one child in its own process group with a hard timeout.

    Returns {exit_code, wall_s, resource, timed_out, interrupted}. The child
    is reaped with os.wait4 so the rusage is THIS child's, not the sum of
    all children so far.
    """
    if isinstance(command, str):
        argv = ["/bin/sh", "-c", command]
    else:
        argv = [str(a) for a in command]
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            env=env,
            cwd=str(cwd),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        return {
            "exit_code": None,
            "wall_s": 0.0,
            "resource": _resource(None),
            "timed_out": False,
            "interrupted": None,
            "spawn_error": str(e),
        }
    deadline = start + max(timeout_s, 0.1)
    grace_deadline = None
    timed_out = False
    interrupted = None
    status = 0
    ru = None
    while True:
        try:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
        except ChildProcessError:
            pid, status, ru = proc.pid, 0, None
        if pid == proc.pid:
            break
        now = time.monotonic()
        sig = interrupt_check() if interrupt_check else None
        if sig and grace_deadline is None:
            interrupted = sig
            grace_deadline = now + _KILL_GRACE_S
            _signal_pg(proc.pid, signal.SIGTERM)
        elif now >= deadline and grace_deadline is None:
            timed_out = True
            grace_deadline = now + _KILL_GRACE_S
            _signal_pg(proc.pid, signal.SIGTERM)
        elif grace_deadline is not None and now >= grace_deadline:
            _signal_pg(proc.pid, signal.SIGKILL)
        time.sleep(0.02)
    proc.returncode = os.waitstatus_to_exitcode(status)  # reaped by wait4 above
    return {
        "exit_code": proc.returncode,
        "wall_s": time.monotonic() - start,
        "resource": _resource(ru),
        "timed_out": timed_out,
        "interrupted": interrupted,
    }


def _resource(ru) -> dict:
    if ru is None:
        return {"maxrss_kb": UNAVAILABLE, "utime_s": UNAVAILABLE, "stime_s": UNAVAILABLE}
    return {
        "maxrss_kb": int(ru.ru_maxrss),  # KiB on Linux
        "utime_s": round(ru.ru_utime, 3),
        "stime_s": round(ru.ru_stime, 3),
    }


def log_tail(path: Path, lines: int = 25) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return UNAVAILABLE
    return "\n".join(text.splitlines()[-lines:])


# ---------------------------------------------------------------------------
# The gate runner
# ---------------------------------------------------------------------------

def run_gate(
    gate: dict,
    values: dict[str, str],
    base_env: dict[str, str],
    log_path: Path,
    *,
    default_timeout_s: float = 600.0,
    heldout_vars: list[str] = (),
    cwd=None,
    interrupt_check: Callable[[], Any] | None = None,
) -> dict:
    """Execute one gate and return its record (verdict, metrics, ...)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env, removed = gate_env(base_env, heldout_vars)
    command = gate.get("argv") or gate.get("shell")
    command = substitute_command(command, values, expand_vars=isinstance(command, list))
    timeout = float(gate.get("timeout_s") or default_timeout_s)
    with open(log_path, "wb") as log:
        result = run_child(
            command, env=env, cwd=cwd or values.get("worktree") or ".",
            timeout_s=timeout, log_fh=log, interrupt_check=interrupt_check,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics, divergences = parse_output(text)

    record = {
        "name": gate.get("name"),
        "stage": gate.get("stage"),
        "required": gate.get("required", True),
        "objective": gate.get("objective", False),
        "verdict": None,
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
        "interrupted": result["interrupted"],
        "metrics": metrics,
        "divergence": divergences,
        "rules": gate.get("rules", []),
        "rule_details": [],
        "wall_s": round(result["wall_s"], 3),
        "resource": result["resource"],
        "log": log_path.name,
        "scrubbed_env": removed,
    }
    if result.get("spawn_error"):
        record["verdict"] = "fail"
        record["rule_details"] = [f"cannot start gate: {result['spawn_error']}"]
        return record
    if result["interrupted"]:
        record["verdict"] = "interrupted"
        return record
    if result["timed_out"]:
        record["verdict"] = "fail"
        record["rule_details"] = [f"gate timed out after {timeout:.0f}s (process group killed)"]
    elif result["exit_code"] == 3:
        record["verdict"] = "inconclusive"
        record["rule_details"] = ["exit code 3: ran but cannot decide"]
    elif result["exit_code"] != 0:
        record["verdict"] = "fail"
        record["rule_details"] = [f"exit code {result['exit_code']}; log tail:\n" + log_tail(log_path)]
    else:
        verdict, details = evaluate_rules(metrics, gate.get("rules", []))
        record["verdict"] = verdict
        record["rule_details"] = details

    cleanup = gate.get("remote_cleanup")
    if cleanup:
        _run_cleanup(cleanup, values, env, log_path, interrupt_check=interrupt_check)
    return record


def _run_cleanup(cleanup: dict, values: dict[str, str], env: dict, log_path: Path,
                 interrupt_check=None) -> None:
    """remote_cleanup (e.g. kill remote benches) with its own short timeout."""
    command = cleanup.get("argv") or cleanup.get("shell")
    command = substitute_command(command, values, expand_vars=cleanup.get("argv") is not None)
    timeout = float(cleanup.get("timeout_s") or 60.0)
    try:
        with open(log_path, "ab") as log:
            run_child(command, env=env, cwd=".", timeout_s=timeout, log_fh=log,
                      interrupt_check=interrupt_check)
    except OSError:
        pass
