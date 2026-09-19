"""Shared experiment records for baseline-versus-candidate comparisons.

Issue #168: make each claimed improvement reproducible and prevent invalid
benchmark comparisons. This module defines the two JSON artifacts every
experiment produces — the PREREGISTERED spec (written and hashed before any
arm runs) and the per-arm RECORD (provenance + raw samples) — plus the
validation both must pass.

Design rules (binding):

- Nothing is averaged across incompatible identities. A record carries its
  metric identity (tool, metric, normalizer) and the workload manifest hash;
  the comparator refuses records whose identities differ (see compare.py).
- Unavailable data is recorded as the string "unavailable", never as zero,
  None-shaped silence, or a fabricated value.
- Every random choice (arm ordering, bootstrap resampling) is seeded, and the
  seed is recorded in the artifacts that used it.
- The spec is sealed before evaluation: each record embeds the spec's
  sha256, and the comparator rejects a pair whose embedded spec hashes
  disagree (post-hoc acceptance-rule edits cannot pass unnoticed).

Stdlib only: this module is importable without the repository's Python
project installed (production server installs gain no dependency).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_SCHEMA = "starling-experiment-spec/1"
RECORD_SCHEMA = "starling-experiment-record/1"

UNAVAILABLE = "unavailable"

# Metric registry: the comparator needs to know which direction is "better".
# Adding a metric requires a comparator test (see test_compare.py).
METRIC_DIRECTIONS = {
    "http_transcribe_wall_ms": "lower",
}


class RecordError(ValueError):
    """A spec or record failed validation. Comparators treat this as fatal."""


def canonical_json(obj: Any) -> str:
    """Stable serialization used for the spec seal (sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def workload_manifest(paths: list[Path]) -> dict:
    """Ordered audio manifest: names, sizes, hashes, and the aggregate hash.

    The aggregate hashes the canonical JSON of the per-file entries so any
    reordering, rename, or content change invalidates it — the comparator
    rejects baseline/candidate pairs whose aggregate hashes differ.
    """
    entries = []
    for p in sorted(paths, key=lambda q: q.name):
        if not p.is_file():
            raise RecordError(f"workload file missing: {p}")
        entries.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    aggregate = hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()
    return {"files": entries, "sha256": aggregate}


def validate_spec(spec: dict) -> list[str]:
    """Return a list of problems (empty = valid). Preregistration fields —
    objective, acceptance, protocol — must all be present BEFORE any arm
    runs; the runner refuses otherwise."""
    problems = []
    if spec.get("schema") != SPEC_SCHEMA:
        problems.append(f"schema must be {SPEC_SCHEMA!r}")
    if not isinstance(spec.get("experiment_id"), str) or not spec["experiment_id"]:
        problems.append("experiment_id must be a non-empty string")
    metric = spec.get("metric")
    if metric not in METRIC_DIRECTIONS:
        problems.append(
            f"metric must be one of {sorted(METRIC_DIRECTIONS)} (got {metric!r}); "
            "new metrics need a comparator test before use"
        )
    if METRIC_DIRECTIONS.get(metric) != spec.get("direction"):
        problems.append(
            f"direction must be {METRIC_DIRECTIONS.get(metric)!r} for metric {metric!r}"
        )
    arms = spec.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"baseline", "candidate"}:
        problems.append("arms must have exactly 'baseline' and 'candidate'")
    else:
        for name, arm in arms.items():
            if not isinstance(arm, dict) or not isinstance(arm.get("binary"), str):
                problems.append(f"arms.{name}.binary must be a path string")
            model = arm.get("model")
            if model is not None and not isinstance(model, str):
                problems.append(f"arms.{name}.model must be a path string or null")
            if not isinstance(arm.get("env"), dict):
                problems.append(f"arms.{name}.env must be a string->string mapping")
    protocol = spec.get("protocol")
    if not isinstance(protocol, dict):
        problems.append("protocol must be an object")
    else:
        for key in ("repeats", "requests_per_repeat", "warmup_requests"):
            v = protocol.get(key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                problems.append(f"protocol.{key} must be an integer >= 1")
        if protocol.get("order") not in ("interleaved_random", "fixed"):
            problems.append("protocol.order must be 'interleaved_random' or 'fixed'")
        seed = protocol.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            problems.append("protocol.seed must be an integer")
        t = protocol.get("timeout_s")
        if not isinstance(t, (int, float)) or t <= 0:
            problems.append("protocol.timeout_s must be a positive number")
    acceptance = spec.get("acceptance")
    if not isinstance(acceptance, dict):
        problems.append("acceptance (preregistered rules) must be an object")
    else:
        for key in ("min_improvement_pct", "max_ci_halfwidth_pct", "max_regression_pct"):
            v = acceptance.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                problems.append(f"acceptance.{key} must be a number >= 0")
    if not isinstance(spec.get("objective"), str) or not spec["objective"]:
        problems.append("objective (the declared hypothesis) must be a non-empty string")
    workload = spec.get("workload")
    if not isinstance(workload, dict):
        problems.append("workload must be an object (pin it via pin-workload)")
    else:
        if not isinstance(workload.get("files"), list) or not workload["files"]:
            problems.append("workload.files must be a non-empty list")
        if not isinstance(workload.get("audio"), str):
            problems.append("workload.audio must be the directory path string")
        pin = workload.get("sha256")
        if pin is not None and (not isinstance(pin, str) or len(pin) != 64):
            problems.append("workload.sha256 pin must be a 64-char digest")
    return problems


def load_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise RecordError(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise RecordError(f"malformed JSON in {path}: {e}") from e
    if not isinstance(obj, dict):
        raise RecordError(f"{path} must contain a JSON object")
    return obj


def load_spec(path: Path) -> dict:
    spec = load_json(path)
    problems = validate_spec(spec)
    if problems:
        raise RecordError("invalid spec " + str(path) + ": " + "; ".join(problems))
    return spec


def validate_record(record: dict) -> list[str]:
    problems = []
    if record.get("schema") != RECORD_SCHEMA:
        problems.append(f"schema must be {RECORD_SCHEMA!r}")
    if record.get("role") not in ("baseline", "candidate"):
        problems.append("role must be 'baseline' or 'candidate'")
    if not isinstance(record.get("spec_sha256"), str) or len(record["spec_sha256"]) != 64:
        problems.append("spec_sha256 (the sealed spec's hash) must be recorded")
    prov = record.get("provenance")
    if not isinstance(prov, dict):
        problems.append("provenance must be an object")
    else:
        for key in ("repo_revision", "ggml_revision", "binary_sha256", "runtime",
                    "workload_sha256", "normalizer", "model_claim", "commands"):
            if key not in prov:
                problems.append(f"provenance.{key} missing")
        if not isinstance(prov.get("runtime"), dict) or "hardware" not in prov.get("runtime", {}):
            problems.append("provenance.runtime.hardware missing")
        ident = prov.get("metric_identity")
        if not isinstance(ident, dict) or not all(k in ident for k in ("tool", "metric", "normalizer")):
            problems.append("provenance.metric_identity must have tool/metric/normalizer")
    samples = record.get("samples")
    if not isinstance(samples, list):
        problems.append("samples must be a list")
    else:
        for i, s in enumerate(samples):
            if not isinstance(s, dict) or not {"arm", "repeat", "request", "cold", "wall_ms"} <= set(s):
                problems.append(f"samples[{i}] missing required fields")
    if record.get("status") not in ("ok", "failed"):
        problems.append("status must be 'ok' or 'failed'")
    return problems


def load_record(path: Path) -> dict:
    record = load_json(path)
    problems = validate_record(record)
    if problems:
        raise RecordError("invalid record " + str(path) + ": " + "; ".join(problems))
    return record
