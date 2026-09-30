"""Hermetic toy fixture for the campaign runner (issue #176 §7).

Builds a throwaway git repository containing a toy "engine" (a script that
sleeps WORK_MS and prints the input text) and a toy evaluator under
`evaluator/` (correctness + perf gates plus the fixed input corpus). The
evaluator is listed as a trusted path and PROTECTED, so the real
trusted-extraction, sealing, authority and tamper-detection code paths are
exercised exactly as they are on the real repository — with millisecond
"transcriptions" instead of models.

Used by `campaign.py pilot` and by the acceptance tests. CPU-only, no
network, no models, stdlib only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from spec import PROFILE_SCHEMA, TASK_SCHEMA

INPUT_TEXT = (
    "the quick brown fox jumps over the lazy dog while the campaign runner "
    "measures hypotheses that must survive independent gates\n"
)

ENGINE = '''#!/usr/bin/env python3
"""Toy engine: sleep WORK_MS, print the input text (optionally truncated).

config.json next to this file is the tuning surface an optimizing agent may
legitimately touch; the evaluator (evaluator/**) is protected.
"""
import json
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "config.json")) as fh:
    cfg = json.load(fh)
with open(sys.argv[1]) as fh:
    out = fh.read()
time.sleep(cfg.get("WORK_MS", 0) / 1000.0)
if cfg.get("TRUNCATE"):
    out = out[: len(out) // 2]
if cfg.get("DROP_FIRST_CHAR"):
    out = out[1:]
if cfg.get("SUFFIX"):
    out += cfg["SUFFIX"]
sys.stdout.write(out)
'''

BUILD = '''#!/usr/bin/env python3
"""Toy build step: bundle the engine + config into build-toy/bundle (the
"binary" the gates run), so build/artifact hashing is real."""
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out = here.parent / "build-toy" / "bundle"
out.mkdir(parents=True, exist_ok=True)
shutil.copy2(here / "transcribe.py", out / "transcribe.py")
shutil.copy2(here / "config.json", out / "config.json")
print("toy build ok ->", out)
'''

GATE_CORRECT = '''#!/usr/bin/env python3
"""Toy correctness gate: the candidate's transcript must match the baseline's
(exact numerical contract). Prints METRIC transcripts_match / output_chars
and a DIVERGENCE line naming the first differing character. Empty output on
either side is inconclusive (exit 3), never a pass.
"""
import subprocess
import sys

def run(binary, path):
    r = subprocess.run([sys.executable, binary, path], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr

def main():
    args = sys.argv[1:]
    def opt(name):
        return args[args.index(name) + 1]
    rc_b, base, err_b = run(opt("--baseline-bin"), opt("--input"))
    rc_c, cand, err_c = run(opt("--candidate-bin"), opt("--input"))
    if rc_b != 0 or rc_c != 0:
        sys.stderr.write("engine failed: base rc=%s (%s) cand rc=%s (%s)\\n" % (rc_b, err_b[-200:], rc_c, err_c[-200:]))
        sys.exit(1)
    if not base.strip() or not cand.strip():
        print("METRIC transcripts_match=unavailable")
        print("METRIC output_chars=%d" % len(cand))
        sys.stderr.write("empty transcript on at least one side; cannot decide\\n")
        sys.exit(3)
    match = 1 if base == cand else 0
    print("METRIC transcripts_match=%d" % match)
    print("METRIC output_chars=%d" % len(cand))
    if not match:
        for i, (a, b) in enumerate(zip(base, cand)):
            if a != b:
                print("DIVERGENCE: first differing char at %d: expected %r got %r" % (i, a, b))
                break
        else:
            print("DIVERGENCE: outputs differ in length: expected %d got %d" % (len(base), len(cand)))
    sys.exit(0)

main()
'''

GATE_PERF = '''#!/usr/bin/env python3
"""Toy perf gate: run baseline-best vs candidate alternately R rounds and
print median/min/max wall ms per side plus improvement_pct. A non-positive
baseline median is inconclusive (exit 3)."""
import statistics
import subprocess
import sys
import time

def run(binary, path):
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, binary, path], capture_output=True, text=True, timeout=120)
    ms = (time.monotonic() - t0) * 1000.0
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-300:])
        sys.exit(1)
    return ms

def main():
    args = sys.argv[1:]
    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default
    base_bin = opt("--base-bin")
    cand_bin = opt("--candidate-bin")
    path = opt("--input")
    rounds = int(opt("--rounds", "3") or 3)
    base, cand = [], []
    for r in range(rounds):
        pair = ((base_bin, base), (cand_bin, cand)) if r % 2 == 0 else ((cand_bin, cand), (base_bin, base))
        for binary, sink in pair:
            sink.append(run(binary, path))
    bm, cm = statistics.median(base), statistics.median(cand)
    if bm <= 0:
        print("METRIC improvement_pct=unavailable")
        sys.exit(3)
    print("METRIC wall_ms_base_median=%.2f" % bm)
    print("METRIC wall_ms_median=%.2f" % cm)
    print("METRIC wall_ms_base_min=%.2f" % min(base))
    print("METRIC wall_ms_base_max=%.2f" % max(base))
    print("METRIC wall_ms_min=%.2f" % min(cand))
    print("METRIC wall_ms_max=%.2f" % max(cand))
    print("METRIC improvement_pct=%.2f" % ((bm - cm) / bm * 100.0))
    sys.exit(0)

main()
'''


def build_toy_repo(dest: Path) -> str:
    """Create the toy repo at `dest` and return its baseline commit SHA."""
    dest = Path(dest)
    (dest / "engine").mkdir(parents=True, exist_ok=True)
    (dest / "evaluator").mkdir(parents=True, exist_ok=True)
    (dest / "engine" / "transcribe.py").write_text(ENGINE, encoding="utf-8")
    (dest / "engine" / "build.py").write_text(BUILD, encoding="utf-8")
    (dest / "engine" / "config.json").write_text(
        json.dumps({"WORK_MS": 60}, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "evaluator" / "gate_correct.py").write_text(GATE_CORRECT, encoding="utf-8")
    (dest / "evaluator" / "gate_perf.py").write_text(GATE_PERF, encoding="utf-8")
    (dest / "evaluator" / "input.txt").write_text(INPUT_TEXT, encoding="utf-8")
    (dest / ".gitignore").write_text("build-toy/\n", encoding="utf-8")

    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(dest), *args], capture_output=True, text=True
        )
        if out.returncode != 0:
            raise RuntimeError(f"git {args} failed: {out.stderr}")
        return out.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "toy")
    git("config", "user.email", "toy@starling.local")
    git("add", "-A")
    git("commit", "-q", "-m", "toy baseline: engine + evaluator")
    return git("rev-parse", "HEAD")


def toy_profile(repo_dir: Path) -> dict:
    return {
        "schema": PROFILE_SCHEMA,
        "id": "toy--notebook",
        "model": {"name": "Toy engine", "hf_id": None},
        "tracking_issue": "#176",
        "status": "ready",
        "readiness_notes": "Hermetic toy engine used by the pilot and tests; "
                           "exercises the real extraction/seal/authority paths.",
        "slug": None,
        "device": "notebook",
        "backend": "cpu",
        "engine": "auto",
        "artifacts": [],
        "build": {
            "argv": ["{python}", "engine/build.py"],
            "artifacts_out": ["build-toy/bundle/transcribe.py", "build-toy/bundle/config.json"],
        },
        "gates": [
            {
                "name": "correct",
                "stage": "correctness",
                "argv": [
                    "{python}", "{trusted}/evaluator/gate_correct.py",
                    "--baseline-bin", "{baseline}/build-toy/bundle/transcribe.py",
                    "--candidate-bin", "{candidate}/build-toy/bundle/transcribe.py",
                    "--input", "{trusted}/evaluator/input.txt",
                ],
                "rules": [{"metric": "transcripts_match", "op": "==", "value": 1}],
                "required": True,
            },
            {
                "name": "perf",
                "stage": "perf",
                "argv": [
                    "{python}", "{trusted}/evaluator/gate_perf.py",
                    "--base-bin", "{best}/build-toy/bundle/transcribe.py",
                    "--candidate-bin", "{candidate}/build-toy/bundle/transcribe.py",
                    "--input", "{trusted}/evaluator/input.txt",
                    "--rounds", "3",
                ],
                "rules": [{"metric": "improvement_pct", "op": ">=", "value": 20.0}],
                "required": True,
                "objective": True,
            },
        ],
        "trusted_paths": ["evaluator"],
        "workloads": [
            {"id": "toy-fixed-input", "description": "one fixed input corpus",
             "status": "available"},
            {"id": "toy-streaming-session", "description": "5 min paced session",
             "status": "unavailable", "owner_issue": "#226"},
        ],
        "objectives": [
            {"name": "latency", "gate": "perf", "metric": "improvement_pct",
             "direction": "higher", "min_improvement": 20.0},
            {"name": "quality", "gate": "correct", "metric": "transcripts_match",
             "direction": "higher", "min_improvement": 0.0},
        ],
        "quality_policy": {"policy": "exact_numerical_contract"},
        # Held-out promotion check: same correctness contract on inputs the
        # optimizing agent never sees ({heldout} is substituted ONLY by finalize).
        "heldout_gates": [
            {
                "name": "heldout-correct",
                "stage": "quality",
                "argv": [
                    "{python}", "{trusted}/evaluator/gate_correct.py",
                    "--baseline-bin", "{baseline}/build-toy/bundle/transcribe.py",
                    "--candidate-bin", "{candidate}/build-toy/bundle/transcribe.py",
                    "--input", "{heldout}/input.txt",
                ],
                "rules": [{"metric": "transcripts_match", "op": "==", "value": 1}],
                "required": True,
            },
        ],
        "heldout": {
            "env": "STARLING_TOY_HELDOUT_DIR",
            "sha256": None,
            "description": "toy held-out corpus; only finalize reads it",
        },
        "defaults": {
            "allowed_paths": ["engine/**"],
            "budgets": {
                "max_attempts": 5,
                "campaign_wall_clock_s": 3600.0,
                "attempt_wall_clock_s": 600.0,
                "gate_timeout_s": 120.0,
                "agent_timeout_s": 120.0,
                "token_budget": None,
                "cooldown_max_s": 60.0,
            },
        },
        "thresholds": {"max_temp_c": 105.0, "min_mem_available_mb": 0.0},
        "device_expect": None,
    }


def toy_task(profile_id: str, baseline: str, agent_command, **overrides) -> dict:
    task = {
        "schema": TASK_SCHEMA,
        "campaign_id": "toy-pilot",
        "profile": profile_id,
        "hypothesis_backlog": [],
        "baseline_revision": baseline,
        "allowed_paths": ["engine/**"],
        "protected_paths": [],
        "objective": {
            "gate": "perf",
            "metric": "improvement_pct",
            "direction": "higher",
            "min_improvement": 20.0,
        },
        "constraints": [
            {"gate": "correct", "metric": "transcripts_match", "op": "==", "value": 1},
        ],
        "budgets": {
            "max_attempts": 5,
            "campaign_wall_clock_s": 3600.0,
            "attempt_wall_clock_s": 600.0,
            "gate_timeout_s": 120.0,
            "agent_timeout_s": 120.0,
            "token_budget": None,
            "cooldown_max_s": 60.0,
        },
        "measurement_protocol": {"fresh_process": True, "declared_deviations": []},
        "agent": {"command": agent_command, "timeout_s": 120.0},
    }
    task.update(overrides)
    return task
