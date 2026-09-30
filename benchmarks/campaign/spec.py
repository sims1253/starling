"""Task-spec and campaign-profile schemas, validation and sealing (issue #176).

The campaign runner wraps a coding agent in a bounded
hypothesis -> experiment -> review loop. Two PREREGISTERED documents drive
it, and both are sealed before any attempt runs:

- the **task spec** (`starling-campaign-task/1`): provider-neutral JSON the
  operator writes (or the CLI generates from a profile): baseline revision,
  allowed code region, objective, constraints, budgets, measurement
  protocol, agent command.
- the **campaign profile** (`starling-campaign-profile/1`): the per-model x
  per-device readiness record (artifacts, build, gates, workloads,
  thresholds) committed under `profiles/`.

Design rules (binding, mirroring benchmarks/experiments/record.py):

- Unavailable data is the string "unavailable", never zero, None-shaped
  silence, or a fabricated value.
- Everything that decides acceptance (task, profile, evaluator tree,
  artifacts) is hashed at `start`; every attempt embeds the seals, and
  `resume`/`report`/`finalize` refuse a changed seal ("evaluation rules
  changed after preregistration") instead of silently re-reading them.
- Validation returns a list of problems (empty = valid), never raises on
  data errors; structural errors that prevent validation raise
  SpecError.

Stdlib only (same contract as benchmarks/experiments/): production server
installations gain no Python dependency from this code.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Reuse (do not duplicate) the seal helpers from the experiments record
# (issue #168); the campaign must interoperate with its spec seals anyway.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from record import UNAVAILABLE, canonical_json, sha256_file, spec_sha256  # noqa: E402

TASK_SCHEMA = "starling-campaign-task/1"
PROFILE_SCHEMA = "starling-campaign-profile/1"
STATE_SCHEMA = "starling-campaign-state/1"

# Gate stages run in this fixed order; the built-in `authority` check always
# runs first of all and never appears in a profile's gate list.
GATE_STAGES = ("authority", "build", "correctness", "diagnostic", "perf", "quality", "resource")
PROFILE_GATE_STAGES = GATE_STAGES[1:]  # authority is generated, not configured

RULE_OPS = ("==", "!=", "<", "<=", ">", ">=")
DEVICES = ("notebook", "pixel")
BACKENDS = ("vulkan", "cpu")
ENGINES = ("fast", "ggml", "auto")
STATUSES = ("ready", "blocked")
WORKLOAD_STATUSES = ("available", "unavailable")
QUALITY_POLICIES = ("exact_numerical_contract", "quant_wer_gate")

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Paths the candidate may NEVER write, no matter what the task spec allows
# (issue #176 §1): the evaluator itself, the shared experiment record, the
# phone measurement scripts, the quality harness, the fixtures and CI.
ALWAYS_PROTECTED = (
    "benchmarks/campaign/**",
    "benchmarks/experiments/**",
    "benchmarks/fast_engine/phone_*.sh",
    "benchmarks/sonar/**",
    "tests/fixtures/**",
    ".github/**",
    # Vendored code and its patch series (incl. the ggml gitlink itself):
    # changing ggml is separate reviewable work, never a campaign candidate.
    "third_party/**",
    ".gitmodules",
)

DEFAULT_GATE_TIMEOUT_S = 600.0
DEFAULT_COOLDOWN_MAX_S = 600.0


class SpecError(ValueError):
    """A task spec or profile is structurally unreadable."""


# ---------------------------------------------------------------------------
# **-aware glob matching (fnmatch per path segment, `**` spans segments)
# ---------------------------------------------------------------------------

def _segments(path: str) -> list[str]:
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    return parts


def glob_match(pattern: str, path: str) -> bool:
    """Match `path` against `pattern` with `**` support.

    `*` and `?` and `[...]` match within one segment (never across `/`);
    a `**` segment matches zero or more whole segments, so `a/**` covers
    `a` itself and everything under it.
    """
    import fnmatch

    pat = _segments(pattern)
    seg = _segments(path)

    def match(pi: int, si: int) -> bool:
        if pi == len(pat):
            return si == len(seg)
        if pat[pi] == "**":
            return any(match(pi + 1, k) for k in range(si, len(seg) + 1))
        if si == len(seg):
            return False
        return fnmatch.fnmatchcase(seg[si], pat[pi]) and match(pi + 1, si + 1)

    return match(0, 0)


def match_any(patterns: list[str], path: str) -> bool:
    return any(glob_match(p, path) for p in patterns)


def effective_protected(profile: dict, task: dict) -> list[str]:
    """ALWAYS_PROTECTED + the task's list + the profile's evaluator paths."""
    protected = list(ALWAYS_PROTECTED)
    protected.extend(task.get("protected_paths") or [])
    for p in profile.get("trusted_paths") or []:
        protected.append(p)
        protected.append(p.rstrip("/") + "/**")
    # de-duplicate, stable
    seen: set[str] = set()
    out = []
    for p in protected:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def path_violations(paths: list[str], allowed: list[str], protected: list[str]) -> list[str]:
    """Paths outside `allowed` or inside `protected` (the authority check)."""
    bad = []
    for p in paths:
        if match_any(protected, p) or not match_any(allowed, p):
            bad.append(p)
    return bad


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pos(v: Any) -> bool:
    return _num(v) and v > 0


def _nonneg(v: Any) -> bool:
    return _num(v) and v >= 0


# Gate phases: "attempt" gates run for every attempt (and the control and
# finalize); "finalize" gates are expensive measurements (e.g. phone energy:
# extra model loads cost the PowerVR driver's per-boot health budget, #325)
# that run only when the final commit is re-validated.
GATE_PHASES = ("attempt", "finalize")


def validate_gate(gate: Any, where: str = "gate") -> list[str]:
    problems: list[str] = []
    if not isinstance(gate, dict):
        return [f"{where} must be an object"]
    if not isinstance(gate.get("name"), str) or not gate["name"]:
        problems.append(f"{where}.name must be a non-empty string")
    if gate.get("stage") not in PROFILE_GATE_STAGES:
        problems.append(
            f"{where}.stage must be one of {list(PROFILE_GATE_STAGES)} "
            "(the authority stage is built in, not configured)"
        )
    has_argv = isinstance(gate.get("argv"), list) and all(
        isinstance(a, str) for a in gate["argv"]
    )
    has_shell = isinstance(gate.get("shell"), str) and gate["shell"]
    if not (has_argv or has_shell) or (has_argv and has_shell):
        problems.append(f"{where} must have exactly one of argv (list of strings) or shell (string)")
    if "timeout_s" in gate and gate["timeout_s"] is not None and not _pos(gate["timeout_s"]):
        problems.append(f"{where}.timeout_s must be a positive number or null")
    cleanup = gate.get("remote_cleanup")
    if cleanup is not None:
        if not isinstance(cleanup, dict):
            problems.append(f"{where}.remote_cleanup must be an object with argv or shell")
        elif not (
            (isinstance(cleanup.get("argv"), list) and all(isinstance(a, str) for a in cleanup["argv"]))
            ^ (isinstance(cleanup.get("shell"), str) and bool(cleanup["shell"]))
        ):
            problems.append(f"{where}.remote_cleanup must have exactly one of argv or shell")
    rules = gate.get("rules", [])
    if not isinstance(rules, list):
        problems.append(f"{where}.rules must be a list")
    else:
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                problems.append(f"{where}.rules[{i}] must be an object")
                continue
            if not isinstance(rule.get("metric"), str) or not rule["metric"]:
                problems.append(f"{where}.rules[{i}].metric must be a non-empty string")
            if rule.get("op") not in RULE_OPS:
                problems.append(
                    f"{where}.rules[{i}].op must be one of {list(RULE_OPS)}"
                )
            if not isinstance(rule.get("value"), (str, int, float)) or isinstance(rule.get("value"), bool):
                problems.append(f"{where}.rules[{i}].value must be a number or string")
    if not isinstance(gate.get("required", True), bool):
        problems.append(f"{where}.required must be a boolean")
    if not isinstance(gate.get("objective", False), bool):
        problems.append(f"{where}.objective must be a boolean")
    if gate.get("phase", "attempt") not in GATE_PHASES:
        problems.append(f"{where}.phase must be one of {list(GATE_PHASES)}")
    if gate.get("phase") == "finalize" and (gate.get("required", True) or gate.get("objective")):
        # A finalize-only gate never ran on the kept attempts, so it cannot
        # have decided any keep: it is a measurement, not an acceptance gate.
        problems.append(f"{where}: a finalize-phase gate must be required=false, objective=false")
    return problems


def validate_profile(profile: Any) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems: list[str] = []
    if not isinstance(profile, dict):
        return ["profile must be a JSON object"]
    if profile.get("schema") != PROFILE_SCHEMA:
        problems.append(f"schema must be {PROFILE_SCHEMA!r}")
    pid = profile.get("id")
    if not isinstance(pid, str) or not SLUG_RE.match(pid):
        problems.append("id must be a slug like 'parakeet--pixel'")
    model = profile.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("name"), str) or not model["name"]:
        problems.append("model.name must be a non-empty string")
    elif model.get("hf_id") is not None and not isinstance(model.get("hf_id"), str):
        problems.append("model.hf_id must be a string or null")
    if not isinstance(profile.get("tracking_issue"), str) or not profile["tracking_issue"]:
        problems.append("tracking_issue must be a non-empty string (e.g. '#349')")
    status = profile.get("status")
    if status not in STATUSES:
        problems.append(f"status must be one of {list(STATUSES)}")
    if status == "blocked":
        if not profile.get("blocked_on") or not profile.get("blocked_reason"):
            problems.append("a blocked profile needs blocked_on and blocked_reason")
    if profile.get("slug") is not None and not isinstance(profile.get("slug"), str):
        problems.append("slug must be a native model slug string or null")
    if profile.get("device") not in DEVICES:
        problems.append(f"device must be one of {list(DEVICES)}")
    if profile.get("backend") not in BACKENDS:
        problems.append(f"backend must be one of {list(BACKENDS)}")
    if profile.get("engine") not in ENGINES:
        problems.append(f"engine must be one of {list(ENGINES)}")
    for key in ("readiness_notes", "device_expect"):
        if key in profile and profile[key] is not None and not isinstance(profile[key], str):
            problems.append(f"{key} must be a string or null")
    artifacts = profile.get("artifacts")
    if not isinstance(artifacts, list):
        problems.append("artifacts must be a list of {name, path, sha256}")
    else:
        for i, a in enumerate(artifacts):
            if not isinstance(a, dict) or not isinstance(a.get("name"), str) or not a["name"]:
                problems.append(f"artifacts[{i}].name must be a non-empty string")
                continue
            if not isinstance(a.get("path"), str) or not a["path"]:
                problems.append(f"artifacts[{i}].path must be a string (may use ${{ENV}} expansion)")
            sha = a.get("sha256")
            if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
                problems.append(f"artifacts[{i}].sha256 must be null or a 64-hex digest")
    build = profile.get("build")
    if not isinstance(build, dict):
        problems.append("build must be an object with argv|shell and artifacts_out")
    else:
        has_argv = isinstance(build.get("argv"), list) and all(isinstance(x, str) for x in build["argv"])
        has_shell = isinstance(build.get("shell"), str) and bool(build["shell"])
        if not (has_argv or has_shell) or (has_argv and has_shell):
            problems.append("build must have exactly one of argv (list) or shell (string)")
        outs = build.get("artifacts_out")
        if not isinstance(outs, list) or not all(isinstance(x, str) for x in outs):
            problems.append("build.artifacts_out must be a list of worktree-relative paths")
    gates = profile.get("gates")
    if not isinstance(gates, list) or (not gates and status != "blocked"):
        problems.append("gates must be a non-empty ordered list"
                        + ("" if status == "blocked" else
                           " (a ready profile must have at least one gate)"))
    else:
        names = set()
        for i, gate in enumerate(gates):
            problems.extend(validate_gate(gate, f"gates[{i}]"))
            if isinstance(gate, dict) and isinstance(gate.get("name"), str):
                if gate["name"] in names:
                    problems.append(f"duplicate gate name {gate['name']!r}")
                names.add(gate["name"])
    workloads = profile.get("workloads")
    if not isinstance(workloads, list):
        problems.append("workloads must be a list of {id, description, status, owner_issue?}")
    else:
        for i, w in enumerate(workloads):
            if not isinstance(w, dict) or not isinstance(w.get("id"), str) or not w["id"]:
                problems.append(f"workloads[{i}].id must be a non-empty string")
                continue
            if w.get("status") not in WORKLOAD_STATUSES:
                problems.append(
                    f"workloads[{i}].status must be one of {list(WORKLOAD_STATUSES)} "
                    "(do not fake unavailable workloads)"
                )
            if w.get("status") == "unavailable" and not w.get("owner_issue"):
                problems.append(
                    f"workloads[{i}] ({w.get('id')!r}) is unavailable and must name its owner issue"
                )
            if "owner_issue" in w and w["owner_issue"] is not None and not isinstance(w["owner_issue"], str):
                problems.append(f"workloads[{i}].owner_issue must be a string or null")
    objectives = profile.get("objectives")
    gate_names = {g.get("name") for g in gates if isinstance(g, dict)} if isinstance(gates, list) else set()
    if not isinstance(objectives, list) or (not objectives and status != "blocked"):
        problems.append("objectives must be a non-empty list"
                        + (" (blocked profiles are exempt)" if status == "blocked" else ""))
    else:
        for i, o in enumerate(objectives):
            if not isinstance(o, dict):
                problems.append(f"objectives[{i}] must be an object")
                continue
            if not isinstance(o.get("name"), str) or not o["name"]:
                problems.append(f"objectives[{i}].name must be a non-empty string")
            if o.get("gate") not in gate_names:
                problems.append(f"objectives[{i}].gate {o.get('gate')!r} is not a profile gate")
            if not isinstance(o.get("metric"), str) or not o["metric"]:
                problems.append(f"objectives[{i}].metric must be a string")
            if o.get("direction") not in ("lower", "higher"):
                problems.append(f"objectives[{i}].direction must be 'lower' or 'higher'")
            if not _nonneg(o.get("min_improvement")):
                problems.append(f"objectives[{i}].min_improvement must be a number >= 0")
    qp = profile.get("quality_policy")
    if not isinstance(qp, dict) or qp.get("policy") not in QUALITY_POLICIES:
        problems.append(
            f"quality_policy.policy must be one of {list(QUALITY_POLICIES)}"
        )
    heldout = profile.get("heldout")
    if not isinstance(heldout, dict):
        problems.append("heldout must be an object {env, sha256, description}")
    else:
        if not isinstance(heldout.get("env"), str) or not heldout["env"]:
            problems.append("heldout.env must be the environment variable name")
        sha = heldout.get("sha256")
        if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
            problems.append("heldout.sha256 must be null or a 64-hex digest")
        if not isinstance(heldout.get("description"), str):
            problems.append("heldout.description must be a string")
    defaults = profile.get("defaults")
    if not isinstance(defaults, dict):
        problems.append("defaults must be an object {allowed_paths, budgets}")
    else:
        ap = defaults.get("allowed_paths")
        if not isinstance(ap, list) or not ap or not all(isinstance(x, str) and x for x in ap):
            problems.append("defaults.allowed_paths must be a non-empty list of globs")
        budgets = defaults.get("budgets", {})
        if not isinstance(budgets, dict):
            problems.append("defaults.budgets must be an object")
    thresholds = profile.get("thresholds")
    if not isinstance(thresholds, dict) or not all(_num(v) for v in thresholds.values()):
        problems.append("thresholds must be an object of numbers (max_temp_c, ...)")
    tp = profile.get("trusted_paths")
    if tp is not None and (not isinstance(tp, list) or not all(isinstance(x, str) and x for x in tp)):
        problems.append("trusted_paths must be a list of path strings or null")
    hg = profile.get("heldout_gates")
    if hg is not None:
        if not isinstance(hg, list):
            problems.append("heldout_gates must be a list of gates or null")
        else:
            for i, gate in enumerate(hg):
                problems.extend(validate_gate(gate, f"heldout_gates[{i}]"))
    problems.extend(artifact_reference_problems(profile))
    return problems


_ARTIFACT_REF_RE = re.compile(r"\{(candidate|best|baseline)\}/([^\s\"';]+)")


def artifact_reference_problems(profile: dict) -> list[str]:
    """Every `{candidate|best|baseline}/<path>` a gate references must be (or
    lie inside) a declared build.artifacts_out entry: those three directories
    hold ONLY the copied build artifacts, so any other path is a gate that can
    never find its binary and would turn every attempt into a silent
    "inconclusive"."""
    build = profile.get("build")
    outs = build.get("artifacts_out") if isinstance(build, dict) else None
    if not isinstance(outs, list):
        return []
    outs = [o.rstrip("/") for o in outs if isinstance(o, str)]
    problems = []
    for key in ("gates", "heldout_gates"):
        for gate in profile.get(key) or []:
            if not isinstance(gate, dict):
                continue
            command = gate.get("argv") or gate.get("shell") or []
            texts = [command] if isinstance(command, str) else [str(a) for a in command]
            for text in texts:
                for _dir, rel in _ARTIFACT_REF_RE.findall(text):
                    rel = rel.rstrip("/")
                    if not any(rel == o or rel.startswith(o + "/") for o in outs):
                        problems.append(
                            f"{key} {gate.get('name')!r} references {{{_dir}}}/{rel}, "
                            f"which is not a build.artifacts_out entry {outs}")
    return problems


def validate_task(task: Any, profile: dict | None = None) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems: list[str] = []
    if not isinstance(task, dict):
        return ["task must be a JSON object"]
    if task.get("schema") != TASK_SCHEMA:
        problems.append(f"schema must be {TASK_SCHEMA!r}")
    cid = task.get("campaign_id")
    if not isinstance(cid, str) or not SLUG_RE.match(cid):
        problems.append("campaign_id must be a slug")
    if not isinstance(task.get("profile"), str) or not task["profile"]:
        problems.append("profile must be the profile id string")
    backlog = task.get("hypothesis_backlog")
    if not isinstance(backlog, list) or not all(isinstance(h, str) for h in backlog):
        problems.append("hypothesis_backlog must be a list of strings (may be empty)")
    if not isinstance(task.get("baseline_revision"), str) or not SHA_RE.match(task["baseline_revision"]):
        problems.append("baseline_revision must be a full 40-char commit SHA (resolve --baseline first)")
    allowed = task.get("allowed_paths")
    if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) and x for x in allowed):
        problems.append("allowed_paths must be a non-empty glob list (the code region the candidate may write)")
    protected = task.get("protected_paths")
    if not isinstance(protected, list) or not all(isinstance(x, str) for x in protected):
        problems.append("protected_paths must be a glob list (ALWAYS_PROTECTED is added implicitly)")
    objective = task.get("objective")
    if not isinstance(objective, dict):
        problems.append("objective must be {gate, metric, direction, min_improvement}")
    else:
        if not isinstance(objective.get("gate"), str) or not objective["gate"]:
            problems.append("objective.gate must be the objective gate name")
        if not isinstance(objective.get("metric"), str) or not objective["metric"]:
            problems.append("objective.metric must be a metric name")
        if objective.get("direction") not in ("lower", "higher"):
            problems.append("objective.direction must be 'lower' or 'higher'")
        if not _nonneg(objective.get("min_improvement")):
            problems.append("objective.min_improvement must be a number >= 0")
    constraints = task.get("constraints")
    if not isinstance(constraints, list):
        problems.append("constraints must be a list of {gate, metric, op, value} rules")
    else:
        for i, c in enumerate(constraints):
            if not isinstance(c, dict) or not isinstance(c.get("gate"), str):
                problems.append(f"constraints[{i}].gate must be a gate name")
                continue
            if not isinstance(c.get("metric"), str):
                problems.append(f"constraints[{i}].metric must be a metric name")
            if c.get("op") not in RULE_OPS:
                problems.append(f"constraints[{i}].op must be one of {list(RULE_OPS)}")
            if isinstance(c.get("value"), bool) or not isinstance(c.get("value"), (str, int, float)):
                problems.append(f"constraints[{i}].value must be a number or string")
    budgets = task.get("budgets")
    if not isinstance(budgets, dict):
        problems.append("budgets must be an object")
    else:
        if not isinstance(budgets.get("max_attempts"), int) or isinstance(budgets.get("max_attempts"), bool) or budgets.get("max_attempts", 0) < 1:
            problems.append("budgets.max_attempts must be an integer >= 1")
        for key in ("campaign_wall_clock_s", "attempt_wall_clock_s"):
            if not _pos(budgets.get(key)):
                problems.append(f"budgets.{key} must be a positive number")
        if "gate_timeout_s" in budgets and budgets["gate_timeout_s"] is not None and not _pos(budgets["gate_timeout_s"]):
            problems.append("budgets.gate_timeout_s must be a positive number or null")
        if "agent_timeout_s" in budgets and budgets["agent_timeout_s"] is not None and not _pos(budgets["agent_timeout_s"]):
            problems.append("budgets.agent_timeout_s must be a positive number or null")
        tb = budgets.get("token_budget")
        if tb is not None and (not isinstance(tb, int) or isinstance(tb, bool) or tb < 0):
            problems.append("budgets.token_budget must be an integer >= 0 or null")
        if "cooldown_max_s" in budgets and budgets["cooldown_max_s"] is not None and not _nonneg(budgets["cooldown_max_s"]):
            problems.append("budgets.cooldown_max_s must be a number >= 0 or null")
    protocol = task.get("measurement_protocol")
    if not isinstance(protocol, dict):
        problems.append("measurement_protocol must be {fresh_process: bool, declared_deviations: [str]}")
    else:
        if not isinstance(protocol.get("fresh_process"), bool):
            problems.append("measurement_protocol.fresh_process must be a boolean")
        devs = protocol.get("declared_deviations")
        if not isinstance(devs, list) or not all(isinstance(d, str) for d in devs):
            problems.append("measurement_protocol.declared_deviations must be a list of strings")
    agent = task.get("agent")
    if agent is not None:
        if not isinstance(agent, dict):
            problems.append("agent must be null (agent-driven mode) or {command, timeout_s}")
        else:
            cmd = agent.get("command")
            ok = isinstance(cmd, str) and bool(cmd) or (
                isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd)
            )
            if not ok:
                problems.append("agent.command must be a command template (string or argv list) or null")
            if "timeout_s" in agent and agent["timeout_s"] is not None and not _pos(agent["timeout_s"]):
                problems.append("agent.timeout_s must be a positive number or null")
    if profile is not None:
        if task.get("profile") != profile.get("id"):
            problems.append(f"task.profile {task.get('profile')!r} != profile id {profile.get('id')!r}")
        gate_names = {g.get("name") for g in profile.get("gates", []) if isinstance(g, dict)}
        if isinstance(objective, dict) and objective.get("gate") not in gate_names:
            problems.append(f"objective.gate {objective.get('gate')!r} is not a gate of profile {profile.get('id')!r}")
        for i, c in enumerate(constraints if isinstance(constraints, list) else []):
            if isinstance(c, dict) and c.get("gate") not in gate_names:
                problems.append(f"constraints[{i}].gate {c.get('gate')!r} is not a gate of this profile")
    return problems


# ---------------------------------------------------------------------------
# Loading + sealing
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise SpecError(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise SpecError(f"malformed JSON in {path}: {e}") from e
    if not isinstance(obj, dict):
        raise SpecError(f"{path} must contain a JSON object")
    return obj


def load_profile(path: Path) -> dict:
    profile = load_json(path)
    problems = validate_profile(profile)
    if problems:
        raise SpecError(f"invalid profile {path}: " + "; ".join(problems))
    expected = f"{profile.get('id')}.json"
    if Path(path).name != expected:
        raise SpecError(f"profile id {profile.get('id')!r} must match file name {expected!r}")
    return profile


def load_profiles(directory: Path) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for path in sorted(Path(directory).glob("*.json")):
        profile = load_profile(path)
        profiles[profile["id"]] = profile
    return profiles


def seal_of(obj: dict) -> str:
    """The seal: sha256 over the canonical JSON (same rule as experiments)."""
    return spec_sha256(obj)


def hash_tree(root: Path) -> str:
    """Hash a whole directory tree: sorted relpath + per-file sha256 -> aggregate.

    Used for the trusted evaluator: any edit to any file under `trusted/`
    changes the aggregate, so a tampered evaluator stops the campaign.
    """
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append([path.relative_to(root).as_posix(), sha256_file(path)])
    return sha256_of_entries(entries)


def sha256_of_entries(entries: list) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def expand_artifact_path(path: str) -> Path:
    """Expand ${ENV} (and ~) in a profile artifact path."""
    import os

    return Path(os.path.expandvars(path)).expanduser()
