"""Report rendering (markdown + json) for a finished (or stopped) campaign.

Everything the brief for issue #176 §6 requires: identity and seals, declared
protocol deviations, budgets used, the best retained commit or an honest
"no improvement: baseline retained", a before/after metric table (baseline
control, kept attempt, finalize re-validation; within-measurement spread is
whatever *_min/*_max metrics the gates themselves report), the full
attempt table including failures, unresolved failures, stop reason, workloads
not measured, device promotion status, and exact rerun commands — plus a
SUGGESTED (never executed) draft-PR command.

Unknown quantities are the string "unavailable", never zero or None-shaped
silence.

Stdlib only, Python 3.10+.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

UNAVAILABLE = "unavailable"


def _tokens(entry: dict):
    usage = entry.get("usage") or {}
    if not usage:
        return UNAVAILABLE
    return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def _best_attempt(ledger: list[dict], best_sha: str | None):
    for entry in ledger:
        if entry.get("candidate_sha") and entry.get("candidate_sha") == best_sha \
                and entry.get("verdict") == "pass":
            return entry
    return None


def build_report(campaign: dict, state: dict, ledger: list[dict]) -> dict:
    profile = campaign["profile"]
    task = campaign["task"]
    control = next((e for e in ledger if e.get("kind") == "control"), None)
    best_sha = campaign.get("best_sha")
    baseline_sha = campaign.get("baseline_revision")
    improved = best_sha not in (None, baseline_sha)
    best = _best_attempt(ledger, best_sha) if improved else None

    # Before/after per metric: baseline control vs the kept attempt vs the
    # finalize re-validation. Values are never pooled across attempts (failed
    # candidates must not leak into a "spread"): measurement spread comes from
    # the gates' own *_min/*_max metrics.
    def _gate_metrics(entry):
        metrics: dict = {}
        for g in (entry or {}).get("gates", []):
            for name, value in (g.get("metrics") or {}).items():
                metrics[name] = value
        return metrics

    control_metrics = _gate_metrics(control)
    best_metrics = _gate_metrics(best)
    finalize_metrics = _gate_metrics(state.get("finalize"))
    metric_names = sorted(set(control_metrics) | set(best_metrics) | set(finalize_metrics))
    metric_table = {}
    for name in metric_names:
        metric_table[name] = {
            "baseline_control": control_metrics.get(name, UNAVAILABLE),
            "best": best_metrics.get(name, UNAVAILABLE),
            "finalize": finalize_metrics.get(name, UNAVAILABLE),
        }

    tokens = [ _tokens(e) for e in ledger ]
    token_values = [t for t in tokens if isinstance(t, (int, float))]
    started = campaign.get("started_at")
    finalized = (state.get("finalize") or {})
    report = {
        "schema": "starling-campaign-report/1",
        "campaign_id": campaign.get("campaign_id"),
        "profile": {
            "id": profile.get("id"),
            "model": (profile.get("model") or {}).get("name"),
            "tracking_issue": profile.get("tracking_issue"),
            "device": profile.get("device"),
            "engine": profile.get("engine"),
            "backend": profile.get("backend"),
        },
        "identity": {
            "baseline_revision": baseline_sha,
            "task_seal": campaign.get("seals", {}).get("task", UNAVAILABLE),
            "profile_seal": campaign.get("seals", {}).get("profile", UNAVAILABLE),
            "evaluator_hash": campaign.get("seals", {}).get("evaluator", UNAVAILABLE),
            "artifact_hashes": campaign.get("seals", {}).get("artifacts", {}),
            "device_at_launch": campaign.get("device_identity", UNAVAILABLE),
            "device_at_finalize": finalized.get("device_identity", UNAVAILABLE),
        },
        "declared_protocol_deviations": (task.get("measurement_protocol") or {}).get(
            "declared_deviations", []
        ),
        "measurement_protocol": task.get("measurement_protocol", {}),
        "budgets": {
            "declared": task.get("budgets", {}),
            "used": {
                "attempts": state.get("attempts_used", 0),
                "wall_clock_s": _wall_clock_used(state, started),
                "tokens": sum(token_values) if token_values else UNAVAILABLE,
            },
        },
        "best": {
            "improved": improved,
            "best_sha": best_sha,
            "baseline_sha": baseline_sha,
            "kept_attempt": (best or {}).get("n"),
            "note": None if improved else "no improvement: baseline retained (a valid outcome)",
        },
        "metrics": metric_table,
        "attempts": [
            {
                "n": e.get("n"),
                "kind": e.get("kind", "attempt"),
                "hypothesis": e.get("hypothesis", UNAVAILABLE),
                "verdict": e.get("verdict", UNAVAILABLE),
                "failed_stage": _failed_stage(e),
                "first_divergence": _first_divergence(e),
                "key_metrics": _key_metrics(e),
                "duration_s": e.get("wall_s", UNAVAILABLE),
                "tokens": _tokens(e),
                "stop_reason": e.get("stop_reason"),
            }
            for e in ledger
        ],
        "unresolved_failures": _unresolved(ledger),
        "stop_reason": state.get("stop_reason"),
        "status": state.get("status"),
        "workloads_not_measured": [
            {"id": w.get("id"), "owner_issue": w.get("owner_issue", UNAVAILABLE)}
            for w in profile.get("workloads", [])
            if w.get("status") == "unavailable"
        ],
        "promotion": _promotion(profile, state, finalized),
        "rerun_commands": _rerun_commands(campaign, task),
        "footer": (
            "This report is review material only: the runner never merges, never "
            "pushes, and never changes default models. A human decides."
        ),
    }
    return report


def _wall_clock_used(state: dict, started) -> float | str:
    end = state.get("finished_at") or state.get("updated_at")
    if not started or not end:
        return UNAVAILABLE
    try:
        t0 = _dt.datetime.fromisoformat(str(started)).timestamp()
        t1 = _dt.datetime.fromisoformat(str(end)).timestamp()
        return round(t1 - t0, 1)
    except ValueError:
        return UNAVAILABLE


def _failed_stage(entry: dict):
    for g in entry.get("gates", []):
        if g.get("verdict") == "fail":
            return g.get("stage")
    if entry.get("verdict") == "inconclusive" and entry.get("reason") == "no_change":
        return "authority"
    return None


def _first_divergence(entry: dict):
    for g in entry.get("gates", []):
        for d in g.get("divergence") or []:
            return d
    return None


def _key_metrics(entry: dict):
    parts = []
    for g in entry.get("gates", []):
        for name, value in (g.get("metrics") or {}).items():
            parts.append(f"{name}={_fmt(value)}")
    return " ".join(parts) if parts else UNAVAILABLE


def _unresolved(ledger: list[dict]) -> list[dict]:
    out = []
    for e in ledger:
        if e.get("verdict") == "fail" and e.get("kind") != "control":
            out.append({
                "n": e.get("n"),
                "hypothesis": e.get("hypothesis", UNAVAILABLE),
                "failed_stage": _failed_stage(e),
                "first_divergence": _first_divergence(e),
            })
    return out


def _promotion(profile: dict, state: dict, finalized: dict) -> dict:
    device = profile.get("device")
    if not finalized:
        status = f"blocked: not validated on {device}"
    elif finalized.get("verdict") != "pass":
        status = f"blocked: finalize verdict {finalized.get('verdict')}"
    elif not finalized.get("heldout", {}).get("provided"):
        status = "blocked: held-out corpus not provided"
    elif not finalized["heldout"].get("pinned"):
        # An unpinned corpus could be swapped for an easier one after the fact.
        status = "blocked: held-out corpus not pinned (start with --heldout-pin)"
    elif finalized["heldout"].get("verdict") is None:
        status = "blocked: no held-out gates defined for this profile"
    elif finalized["heldout"].get("verdict") != "pass":
        status = f"blocked: held-out gate verdict {finalized['heldout'].get('verdict')}"
    else:
        status = f"validated on {device} (pinned held-out gates passed)"
    return {
        "status": status,
        "device": device,
        "finalized": bool(finalized),
        "heldout": finalized.get("heldout", {"provided": False}),
    }


def _rerun_commands(campaign: dict, task: dict) -> list[str]:
    import gates as gates_mod
    import spec as spec_mod

    d = campaign.get("dir", "<campaign-dir>")
    profile = campaign["profile"]
    cmds = [
        f"python benchmarks/campaign/campaign.py resume --campaign {d}",
        f"python benchmarks/campaign/campaign.py finalize --campaign {d} --heldout $STARLING_HELDOUT_DIR",
        f"git -C {d}/worktree log {campaign.get('baseline_revision', '<baseline>')}..{campaign.get('best_sha', '<best>')}",
    ]
    values = {
        "trusted": str(Path(d) / "trusted"),
        "worktree": str(Path(d) / "worktree"),
        "candidate": "<attempt-artifact-dir>",
        "baseline": str(Path(d) / "baseline"),
        "best": str(Path(d) / "best"),
        "attempt_dir": "<attempt-dir>",
        "campaign_dir": str(d),
        "python": sys.executable,
        "adb_serial": campaign.get("adb_serial") or "",
        "heldout": "$STARLING_HELDOUT_DIR",
    }
    for gate in profile.get("gates", []):
        command = gate.get("argv") or gate.get("shell")
        command = gates_mod.substitute_command(command, values)
        if isinstance(command, list):
            command = " ".join(command)
        cmds.append(f"# gate {gate.get('name')} ({gate.get('stage')}): {command}")
    return cmds


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    p = report["profile"]
    add(f"# Campaign report: {report['campaign_id']}")
    add("")
    add(f"- profile: **{p['id']}** ({p['model']}, tracking {p['tracking_issue']})")
    add(f"- device: {p['device']} / engine {p['engine']} / backend {p['backend']}")
    ident = report["identity"]
    add(f"- baseline revision: `{ident['baseline_revision']}`")
    add(f"- task seal: `{ident['task_seal']}`")
    add(f"- evaluator hash: `{ident['evaluator_hash']}`")
    if ident["artifact_hashes"]:
        for name, sha in ident["artifact_hashes"].items():
            add(f"- artifact {name}: `{sha}`")
    add(f"- device at launch: `{json.dumps(ident['device_at_launch'])[:400]}`")
    add(f"- device at finalize: `{json.dumps(ident['device_at_finalize'])[:400]}`")
    add("")
    devs = report["declared_protocol_deviations"]
    add("## Measurement protocol")
    add("")
    mp = report["measurement_protocol"]
    add(f"- fresh process: {mp.get('fresh_process', UNAVAILABLE)}")
    add(f"- declared deviations: {devs if devs else 'none'}")
    add("")
    add("## Budgets used")
    b = report["budgets"]
    add("")
    add(f"- attempts: {b['used']['attempts']} (max {b['declared'].get('max_attempts')})")
    add(f"- wall clock: {b['used']['wall_clock_s']} s "
        f"(budget {b['declared'].get('campaign_wall_clock_s')} s)")
    add(f"- tokens: {b['used']['tokens']} (budget {b['declared'].get('token_budget')})")
    add("")
    add("## Result")
    add("")
    best = report["best"]
    if best["improved"]:
        add(f"- **kept**: attempt {best['kept_attempt']} commit `{best['best_sha']}`")
    else:
        add(f"- **no improvement: baseline retained** (`{best['baseline_sha']}`) — an honest null result is a valid outcome")
    if report["stop_reason"]:
        add(f"- stop reason: `{report['stop_reason']}` (status {report['status']})")
    add(f"- promotion: {report['promotion']['status']}")
    add("")
    add("## Before / after")
    add("")
    add("Baseline control = baseline vs itself (attempt 0); best = the kept attempt "
        "vs the previous best; finalize = best vs the ORIGINAL baseline on this "
        "device. Spread within a measurement is the gates' own *_min/*_max metrics.")
    add("")
    add("| metric | baseline control | best | finalize |")
    add("|---|---|---|---|")
    for name, row in report["metrics"].items():
        add(f"| {name} | {_fmt(row['baseline_control'])} | {_fmt(row['best'])} | "
            f"{_fmt(row['finalize'])} |")
    if not report["metrics"]:
        add(f"| {UNAVAILABLE} | | | |")
    add("")
    add("## Attempts (failures included)")
    add("")
    add("| # | kind | hypothesis | verdict | failed stage | first divergence | key metrics | duration s | tokens |")
    add("|---|---|---|---|---|---|---|---|---|")
    for a in report["attempts"]:
        hyp = str(a["hypothesis"]).replace("|", "\\|")
        if len(hyp) > 60:
            hyp = hyp[:57] + "..."
        div = str(a["first_divergence"] or "").replace("|", "\\|")
        if len(div) > 50:
            div = div[:47] + "..."
        km = str(a["key_metrics"]).replace("|", "\\|")
        if len(km) > 80:
            km = km[:77] + "..."
        add(f"| {a['n']} | {a['kind']} | {hyp} | {a['verdict']} | {a['failed_stage'] or ''} "
            f"| {div} | {km} | {a['duration_s']} | {a['tokens']} |")
    add("")
    if report["unresolved_failures"]:
        add("## Unresolved failures")
        add("")
        for f in report["unresolved_failures"]:
            add(f"- attempt {f['n']} ({f['failed_stage']}): {f['first_divergence'] or f['hypothesis']}")
        add("")
    if report["workloads_not_measured"]:
        add("## Workloads not measured")
        add("")
        for w in report["workloads_not_measured"]:
            add(f"- {w['id']} — blocked, owner {w['owner_issue']} (do not fake them)")
        add("")
    add("## Exact rerun commands")
    add("")
    add("```bash")
    for cmd in report["rerun_commands"]:
        add(cmd)
        add("")
    best_sha = report["best"]["best_sha"]
    add(f'gh pr create --draft --title "campaign {report["campaign_id"]}: measured optimization" '
        f'--body "See {report["campaign_id"]}/report.md; baseline {report["identity"]["baseline_revision"]} -> {best_sha}" '
        f"# SUGGESTED ONLY — never executed by the runner")
    add("```")
    add("")
    add(f"> {report['footer']}")
    add("")
    return "\n".join(lines)


def write_report(report: dict, campaign_dir: Path) -> tuple[Path, Path]:
    md = campaign_dir / "report.md"
    js = campaign_dir / "report.json"
    md.write_text(render_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return md, js
