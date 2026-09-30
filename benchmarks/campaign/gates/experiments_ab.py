#!/usr/bin/env python3
"""experiments_ab.py — trusted perf gate: a thin adapter that runs the sealed
benchmarks/experiments comparator (issue #168) on {best} vs {candidate}
binaries and reports its verdict as METRIC lines.

The adapter WRITES a fresh experiment spec (same schema the comparator
validates and seals), runs `run_experiment.py run` (fresh processes,
interleaved, seeded) and `compare`, and prints:

    METRIC verdict=pass|fail|inconclusive|unavailable
    METRIC improvement_pct=…   METRIC ci_low_pct=…   METRIC ci_high_pct=…

Acceptance is decided by the profile's rule (`verdict == pass`), never here.
The adapter runs run_experiment.py as a subprocess, so the experiments
advisory lock ($STARLING_EXPERIMENT_LOCK) is taken by the child only — the
campaign lock is a different lock by design (no deadlock).

Usage (placeholders substituted by the campaign runner):
    {python} {trusted}/benchmarks/campaign/gates/experiments_ab.py \
        --base-bin {best}/starling-serve-contract-fixture \
        --cand-bin {candidate}/starling-serve-contract-fixture \
        --run-dir {attempt_dir}/experiment [--model parakeet] [--gguf …]

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent.parent / "experiments"  # <trusted>/benchmarks/experiments
sys.path.insert(0, str(EXPERIMENTS))

from record import validate_spec, workload_manifest  # noqa: E402


def make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)


def build_spec(args, wav: Path) -> dict:
    manifest = workload_manifest([wav])
    return {
        "schema": "starling-experiment-spec/1",
        "experiment_id": f"campaign-ab-{args.label}",
        "objective": "campaign gate: candidate must clear the preregistered "
                     "improvement bar against the best-so-far binary",
        "metric": "http_transcribe_wall_ms",
        "direction": "lower",
        "normalizer_identity": "none-raw-wall-time",
        "arms": {
            "baseline": {"binary": args.base_bin, "model_slug": args.model,
                         "model": args.gguf, "env": {}},
            "candidate": {"binary": args.cand_bin, "model_slug": args.model,
                          "model": args.gguf, "env": {}},
        },
        "workload": {"audio": str(wav.parent), "files": [wav.name],
                     "sha256": manifest["sha256"]},
        "protocol": {
            "repeats": args.repeats,
            "requests_per_repeat": args.requests,
            "warmup_requests": 1,
            "order": "interleaved_random",
            "seed": args.seed,
            "timeout_s": args.timeout,
        },
        "acceptance": {
            "min_improvement_pct": args.min_improvement,
            "max_ci_halfwidth_pct": args.max_ci_halfwidth,
            "max_regression_pct": 0.0,
        },
        "tolerated_failures": 2,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-bin", required=True)
    ap.add_argument("--cand-bin", required=True)
    ap.add_argument("--model", default="parakeet")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", default="gate")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--seed", type=int, default=176)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--min-improvement", type=float, default=5.0)
    ap.add_argument("--max-ci-halfwidth", type=float, default=25.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wav = run_dir / "audio" / "silence.wav"
    make_wav(wav)

    spec = build_spec(args, wav)
    problems = validate_spec(spec)
    if problems:
        print("METRIC verdict=unavailable")
        print("adapter spec invalid: " + "; ".join(problems), file=sys.stderr)
        return 3
    spec_path = run_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    cli = EXPERIMENTS / "run_experiment.py"
    for sub in (["run", "--spec", str(spec_path), "--run-dir", str(run_dir)],
                ["compare", "--spec", str(spec_path), "--run-dir", str(run_dir),
                 "--out", str(run_dir / "comparison.json")]):
        proc = subprocess.run([sys.executable, str(cli), *sub],
                              capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode not in (0, 3):
            print("METRIC verdict=unavailable")
            print(f"run_experiment {sub[0]} exited {proc.returncode}", file=sys.stderr)
            return 3

    comparison_path = run_dir / "comparison.json"
    if not comparison_path.exists():
        print("METRIC verdict=unavailable")
        print("no comparison.json written", file=sys.stderr)
        return 3
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    verdict = comparison.get("verdict", "unavailable")
    effect = comparison.get("effect") or {}
    print(f"METRIC verdict={verdict}")
    for key in ("improvement_pct", "ci_low_pct", "ci_high_pct"):
        value = effect.get(key)
        if isinstance(value, (int, float)):
            print(f"METRIC {key}={value:.3f}")
        else:
            print(f"METRIC {key}=unavailable")
    return 0 if verdict in ("pass", "fail", "inconclusive") else 3


if __name__ == "__main__":
    sys.exit(main())
