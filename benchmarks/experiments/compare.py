"""The experiment comparison: compatibility gates + preregistered verdict.

This is the command that decides whether a candidate may be claimed as an
improvement. Two run records enter; one verdict leaves:

- pass: the CI for the paired improvement clears the preregistered
  min_improvement_pct entirely, does not regress beyond
  max_regression_pct, and is narrow enough (max_ci_halfwidth_pct) to be
  called a result.
- fail: the CI sits entirely on the regression side.
- inconclusive: the CI spans zero or is too wide. A point estimate inside
  this band is NEVER a win.
- unavailable: a required run failed or is missing. Refused, not guessed.

Every rejection below is a hard error with the precise fields that
disagree — the point is to make invalid comparisons impossible, not
annoying (issue #168 acceptance).
"""

from __future__ import annotations

import json
from pathlib import Path

from record import RECORD_SCHEMA, RecordError, load_record, load_spec, spec_sha256
from stats import effect_estimate, paired_samples

VERDICTS = ("pass", "fail", "inconclusive", "unavailable")


class IncompatibleRecords(RecordError):
    """The two records cannot be compared at all."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompatibleRecords(message)


def _usable_samples(rec: dict) -> list[dict]:
    return [s for s in rec.get("samples", [])
            if not s.get("cold") and not s.get("error") and not s.get("warmup")]


def check_compatibility(spec: dict, baseline: dict, candidate: dict) -> None:
    """Identity gates: a mismatch means the comparison is INVALID, not unusable.

    Run status and sample usability are deliberately NOT checked here — a
    failed or empty run yields the structured `unavailable` verdict from
    compare() with its diagnostics; an identity mismatch is a hard refusal.
    """
    # Same sealed spec: preregistered rules cannot be swapped post hoc.
    digest = spec_sha256(spec)
    _require(
        baseline.get("spec_sha256") == digest,
        f"baseline spec seal {baseline.get('spec_sha256')!r} != provided spec {digest!r}",
    )
    _require(
        candidate.get("spec_sha256") == digest,
        f"candidate spec seal {candidate.get('spec_sha256')!r} != provided spec {digest!r}",
    )

    for name, rec in (("baseline", baseline), ("candidate", candidate)):
        _require(
            rec.get("schema") == RECORD_SCHEMA,
            f"{name} record schema is {rec.get('schema')!r}",
        )

    pb, pc = baseline["provenance"], candidate["provenance"]
    _require(
        pb["workload_sha256"] == pc["workload_sha256"],
        "workload manifests differ "
        f"({pb['workload_sha256']!r} vs {pc['workload_sha256']!r}): the corpus changed "
        "between arms; the comparison is invalid",
    )
    _require(
        pb["metric_identity"] == pc["metric_identity"],
        "metric identities differ "
        f"({pb['metric_identity']!r} vs {pc['metric_identity']!r}): incompatible metric "
        "definitions (e.g. SONAR WER vs quantization-driver WER) are never compared",
    )
    _require(
        pb["normalizer"] == pc["normalizer"],
        f"normalizers differ ({pb['normalizer']!r} vs {pc['normalizer']!r})",
    )
    _require(
        pb["model_claim"] == pc["model_claim"],
        "model/config claims differ "
        f"({pb['model_claim']!r} vs {pc['model_claim']!r}): baseline and candidate "
        "must run the same model and configuration",
    )
    hw = spec.get("hardware_claim")
    if hw is not None:
        # A declared claim deliberately authorizes cross-machine records;
        # each arm must match the claim (hardware model at least).
        _require(
            pb["runtime"].get("hardware") == hw and pc["runtime"].get("hardware") == hw,
            "runtime does not match the declared hardware claim "
            f"({pb['runtime'].get('hardware')!r} / {pc['runtime'].get('hardware')!r} vs {hw!r})",
        )
    else:
        # No claim: both arms must come from the SAME runtime identity —
        # hardware AND driver, so the same GPU model under different driver
        # versions is still refused as a confound.
        _require(
            pb["runtime"] == pc["runtime"],
            "runtime identities differ "
            f"({pb['runtime']!r} vs {pc['runtime']!r}); "
            "declare a hardware_claim in the spec to compare cross-machine runs "
            "deliberately",
        )


def _unavailable(spec: dict, reason: str, **extra) -> dict:
    result = {
        "verdict": "unavailable",
        "reason": reason,
        "experiment_id": spec.get("experiment_id"),
    }
    result.update(extra)
    return result


def compare(spec: dict, baseline: dict, candidate: dict) -> dict:
    """Apply the preregistered acceptance rules to two validated records."""
    check_compatibility(spec, baseline, candidate)

    # Failed or empty runs are unavailable (diagnostics, no verdict guessed),
    # not hard refusals: they say the experiment could not run, not that the
    # comparison logic is invalid.
    for name, rec in (("baseline", baseline), ("candidate", candidate)):
        if rec.get("status") != "ok":
            return _unavailable(
                spec,
                f"{name} run status is {rec.get('status')!r} "
                f"({len(rec.get('failures', []))} failure(s) recorded)",
                failures={
                    "baseline": baseline.get("failures", []),
                    "candidate": candidate.get("failures", []),
                },
            )
        if not _usable_samples(rec):
            return _unavailable(
                spec, f"{name} record has zero usable (warm, non-failed) samples"
            )

    pairs = paired_samples(baseline["samples"], candidate["samples"])
    if len(pairs) < 2:
        return _unavailable(
            spec,
            f"only {len(pairs)} paired sample(s); two or more are required",
        )

    direction = spec["direction"]
    acc = spec["acceptance"]
    est = effect_estimate(pairs, direction, seed=spec["protocol"]["seed"], confidence=0.95)

    halfwidth = (est["ci_high_pct"] - est["ci_low_pct"]) / 2.0
    ci_too_wide = halfwidth > acc["max_ci_halfwidth_pct"]
    clears_bar = est["ci_low_pct"] >= acc["min_improvement_pct"]
    regresses = est["ci_high_pct"] <= -acc["max_regression_pct"]

    if regresses:
        verdict = "fail"
    elif clears_bar and not ci_too_wide:
        verdict = "pass"
    else:
        verdict = "inconclusive"
        reasons = []
        if not clears_bar:
            reasons.append(
                f"CI low {est['ci_low_pct']:.2f}% does not clear the preregistered "
                f"min improvement {acc['min_improvement_pct']}%"
            )
        if ci_too_wide:
            reasons.append(
                f"CI halfwidth {halfwidth:.2f}% exceeds the preregistered "
                f"{acc['max_ci_halfwidth_pct']}% — collect more paired samples "
                "before claiming anything"
            )
        reasons.append("an inconclusive result cannot be promoted as a win")
        est["inconclusive_because"] = reasons

    cold_b = [s for s in baseline["samples"] if s.get("cold")]
    cold_c = [s for s in candidate["samples"] if s.get("cold")]
    return {
        "verdict": verdict,
        "experiment_id": spec.get("experiment_id"),
        "objective": spec.get("objective"),
        "metric_identity": baseline["provenance"]["metric_identity"],
        "acceptance": acc,
        "effect": est,
        "cold_start_report": {
            "note": "cold/capture samples are reported separately and excluded "
                    "from the gated estimate",
            "baseline_n": len(cold_b),
            "candidate_n": len(cold_c),
            "baseline_ms": [s["wall_ms"] for s in cold_b],
            "candidate_ms": [s["wall_ms"] for s in cold_c],
        },
        "failures": {
            "baseline": baseline.get("failures", []),
            "candidate": candidate.get("failures", []),
        },
    }


def render_summary(comparison: dict) -> str:
    lines = [
        f"experiment: {comparison.get('experiment_id')}",
        f"objective:  {comparison.get('objective')}",
        f"metric:     {comparison.get('metric_identity', {}).get('metric')}",
        f"verdict:    {comparison.get('verdict')}",
    ]
    est = comparison.get("effect")
    if est:
        clusters = est.get("n_repeat_clusters")
        cluster_note = f" in {clusters} repeat clusters" if clusters else ""
        lines.append(
            f"effect:     median improvement {est['improvement_pct']:+.2f}% "
            f"(95% CI {est['ci_low_pct']:+.2f}% .. {est['ci_high_pct']:+.2f}%, "
            f"n={est['n_pairs']} pairs{cluster_note}, "
            f"bootstrap seed {est['bootstrap_seed']})"
        )
        lines.append(
            f"            median wall {est['median_baseline_ms']:.1f} ms -> "
            f"{est['median_candidate_ms']:.1f} ms"
        )
        for reason in est.get("inconclusive_because", []):
            lines.append(f"note:        {reason}")
    if comparison.get("verdict") == "unavailable":
        lines.append(f"note:        {comparison.get('reason', 'run unavailable')}")
    return "\n".join(lines)


def compare_directories(spec_path: Path, baseline_dir: Path, candidate_dir: Path,
                        out_path: Path | None = None) -> dict:
    """The comparison command: load, gate, compare, persist."""
    spec = load_spec(spec_path)
    baseline = load_record(baseline_dir / "record.json")
    candidate = load_record(candidate_dir / "record.json")
    result = compare(spec, baseline, candidate)
    if out_path is not None:
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
