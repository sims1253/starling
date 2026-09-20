"""Deterministic paired statistics for experiment comparisons (issue #168).

Paired bootstrap over per-request wall times, warm samples only (cold-start
and capture costs are separated by the runner and reported, never averaged
into the steady-state estimate). Resampling is CLUSTERED BY REPEAT: requests
from the same fresh server process share repeat-level startup, thermal, and
scheduling effects, so each resample redraws whole repeats (processes), not
individual requests — treating correlated observations as separate evidence
would spuriously narrow the interval. The resampling uses
random.Random(seed) with the seed recorded in the comparison artifact —
rerunning the comparison reproduces the interval bit-for-bit.

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


def paired_samples(baseline: Sequence[dict],
                   candidate: Sequence[dict]) -> list[tuple[int, int, float, float]]:
    """Match warm samples across arms on (repeat, request).

    Cold samples, failures, and unmatched keys are excluded — the caller
    reports them separately. Returns [(repeat, request, base_ms, cand_ms)];
    the repeat is carried so the bootstrap can keep process clusters whole.
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
    return [(r, q, base_idx[(r, q)], cand_idx[(r, q)]) for r, q in keys]


def effect_estimate(pairs: Sequence[tuple[int, int, float, float]], direction: str,
                    seed: int, confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """Median wall time per arm, median paired improvement, bootstrap CI.

    improvement_pct is oriented so that positive always means "candidate is
    better" (lower-is-better: (base - cand) / base * 100; higher-is-better:
    (cand - base) / base * 100). The CI is the percentile interval over
    resampled paired improvements — every resample keeps each pair's two
    observations together AND redraws whole repeats (clusters), preserving
    both within-pair correlation and within-process correlation.
    """
    if not pairs:
        raise ValueError("no paired samples")
    if direction not in ("lower", "higher"):
        raise ValueError(f"unknown metric direction {direction!r}")
    base = [b for _, _, b, _ in pairs]
    cand = [c for _, _, _, c in pairs]
    med_base = statistics.median(base)
    med_cand = statistics.median(cand)

    # Cluster the per-pair improvements by repeat (fresh server process).
    clusters: dict[int, list[float]] = {}
    for repeat, _request, b, c in pairs:
        pct = (b - c) / b * 100.0 if direction == "lower" else (c - b) / b * 100.0
        clusters.setdefault(repeat, []).append(pct)
    cluster_list = [clusters[r] for r in sorted(clusters)]
    all_improvements = [v for cluster in cluster_list for v in cluster]
    med_improvement = statistics.median(all_improvements)

    rng = random.Random(seed)
    stats = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        pooled: list[float] = []
        for _ in range(len(cluster_list)):
            pooled.extend(cluster_list[rng.randrange(len(cluster_list))])
        stats.append(statistics.median(pooled))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = stats[max(0, int(alpha * len(stats)))]
    hi = stats[min(len(stats) - 1, int((1.0 - alpha) * len(stats)))]

    return {
        "n_pairs": len(all_improvements),
        "n_repeat_clusters": len(cluster_list),
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
