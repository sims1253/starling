"""Deterministic paired statistics for experiment comparisons (issue #168).

Paired bootstrap over per-request wall times, warm samples only (cold-start
and capture costs are separated by the runner and reported, never averaged
into the steady-state estimate). The resampling uses random.Random(seed)
with the seed recorded in the comparison artifact — rerunning the comparison
reproduces the interval bit-for-bit.

Statistically inconclusive results cannot be promoted as wins: the verdict
logic in compare.py treats a CI that spans the decision boundary as
inconclusive, whatever the point estimate looks like.
"""

from __future__ import annotations

import random
import statistics
from typing import Sequence

BOOTSTRAP_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95


def paired_samples(baseline: Sequence[dict], candidate: Sequence[dict]) -> list[tuple[str, float, float]]:
    """Match warm samples across arms on (repeat, request).

    Cold samples, failures, and unmatched keys are excluded — the caller
    reports them separately. Returns [(key, base_ms, cand_ms)].
    """
    def usable(s: dict) -> bool:
        return not s.get("cold") and not s.get("error") and not s.get("warmup")

    base_idx = {}
    for s in baseline:
        if usable(s):
            base_idx[(s["repeat"], s["request"])] = s["wall_ms"]
    cand_idx = {}
    for s in candidate:
        if usable(s):
            cand_idx[(s["repeat"], s["request"])] = s["wall_ms"]
    keys = sorted(set(base_idx) & set(cand_idx))
    return [(f"{r}.{q}", base_idx[(r, q)], cand_idx[(r, q)]) for r, q in keys]


def effect_estimate(pairs: Sequence[tuple[str, float, float]], direction: str,
                    seed: int, confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """Median wall time per arm, median paired improvement, bootstrap CI.

    improvement_pct is oriented so that positive always means "candidate is
    better" (for lower-is-better metrics: (base - cand) / base * 100).
    The CI is the percentile interval over resampled paired improvements —
    every resample keeps each pair's two observations together, preserving
    within-pair correlation.
    """
    if not pairs:
        raise ValueError("no paired samples")
    base = [b for _, b, _ in pairs]
    cand = [c for _, _, c in pairs]
    med_base = statistics.median(base)
    med_cand = statistics.median(cand)
    improvements = [(b - c) / b * 100.0 for _, b, c in pairs]
    med_improvement = statistics.median(improvements)

    rng = random.Random(seed)
    n = len(improvements)
    stats = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = [improvements[rng.randrange(n)] for _ in range(n)]
        stats.append(statistics.median(resample))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = stats[max(0, int(alpha * len(stats)))]
    hi = stats[min(len(stats) - 1, int((1.0 - alpha) * len(stats)))]

    return {
        "n_pairs": n,
        "median_baseline_ms": med_base,
        "median_candidate_ms": med_cand,
        "improvement_pct": med_improvement,
        "ci_low_pct": lo,
        "ci_high_pct": hi,
        "confidence": confidence,
        "bootstrap_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "direction": direction,
    }
