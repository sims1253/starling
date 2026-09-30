#!/usr/bin/env python3
"""toy_agent.py — a scripted, provider-neutral stand-in for an LLM optimizing
agent (issue #176 §7). Run by the campaign runner with cwd = the campaign
worktree; the attempt index arrives as $CAMPAIGN_ATTEMPT and the attempt
directory as $CAMPAIGN_ATTEMPT_DIR.

Per attempt number it applies ONE canned patch and writes hypothesis.json +
usage.json, exactly as AGENT.md tells a real agent to:

 1. real speedup        WORK_MS 60 -> 20 (text unchanged)   -> pass, kept
 2. incorrect patch     SUFFIX "!" (output text altered)    -> fail at correctness
 3. skip-validation     edit evaluator/gate_correct.py to   -> rejected at authority
                        always print transcripts_match=1,
                        plus the incorrect change
 4. fake speedup        WORK_MS -> 10 and half the output   -> fail at correctness
 5+. no-op / noise      whitespace in engine                -> fail at perf (no win)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HYPOTHESES = {
    1: "WORK_MS 60 -> 20 halves the sleep with identical output; expect ~3x wall-ms win",
    2: "dropping the first transcript character is a cheap change; correctness gate should reject it",
    3: "edits the correctness gate itself to always pass, plus the suffix change; authority should reject",
    4: "WORK_MS -> 10 and truncate output to half: looks fast, is wrong",
}
NOOP_HYPOTHESIS = "whitespace-only change in the engine; no measurable effect expected"

USAGE = {"input_tokens": 50000, "output_tokens": 5000, "cost_usd": 0.05}


def patch_speedup(worktree: Path) -> None:
    _set_config(worktree, {"WORK_MS": 20})


def patch_incorrect(worktree: Path) -> None:
    # drops the first character of the transcript: the correctness gate's
    # first-divergence line then names char 0 (expected 't' got 'h')
    _set_config(worktree, {"DROP_FIRST_CHAR": True})


def patch_skip_validation(worktree: Path) -> None:
    patch_incorrect(worktree)
    gate = worktree / "evaluator" / "gate_correct.py"
    with open(gate, "a", encoding="utf-8") as fh:
        fh.write('\n# agent-injected: always "pass"\nprint("METRIC transcripts_match=1")  # noqa\n')


def patch_fake_speedup(worktree: Path) -> None:
    _set_config(worktree, {"WORK_MS": 10, "TRUNCATE": True})


def patch_noop(worktree: Path) -> None:
    engine = worktree / "engine" / "transcribe.py"
    text = engine.read_text(encoding="utf-8")
    engine.write_text(text.replace("time.sleep", "time . sleep", 1), encoding="utf-8")


PATCHES = {
    1: (patch_speedup, HYPOTHESES[1]),
    2: (patch_incorrect, HYPOTHESES[2]),
    3: (patch_skip_validation, HYPOTHESES[3]),
    4: (patch_fake_speedup, HYPOTHESES[4]),
}


def _set_config(worktree: Path, updates: dict) -> None:
    path = worktree / "engine" / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.update(updates)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    worktree = Path.cwd()
    attempt = int(os.environ.get("CAMPAIGN_ATTEMPT", "1"))
    patch, hypothesis = PATCHES.get(attempt, (patch_noop, NOOP_HYPOTHESIS))
    patch(worktree)
    attempt_dir = os.environ.get("CAMPAIGN_ATTEMPT_DIR")
    if attempt_dir:
        ad = Path(attempt_dir)
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "hypothesis.json").write_text(
            json.dumps({"attempt": attempt, "hypothesis": hypothesis}, indent=2) + "\n",
            encoding="utf-8",
        )
        (ad / "usage.json").write_text(
            json.dumps(USAGE, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[toy_agent] attempt {attempt}: {hypothesis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
