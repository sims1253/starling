#!/usr/bin/env python3
"""Paired comparison of two wer_models.py results on the same clips.

Prints, for each candidate against the baseline label: mean per-clip WER
difference with a 95 % paired-bootstrap interval (10,000 resamples, seed 0,
the protocol docs/quantization.md uses), corpus WER of both, and how many
hypotheses differ.

Usage:
  compare_wer.py --base bf16 results.json [more.json ...]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

# benchmarks/wer.py, or benchmarks/fast_engine/wer.py once the Python stack
# removal (#451) has moved it there.
for _d in ("..", os.path.join("..", "fast_engine")):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), _d))
from wer import normalize  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from wer_models import edit_distance  # noqa: E402


def per_clip(refs: dict, names: list[str], hyps: list) -> dict[str, float]:
    out = {}
    for n, h in zip(names, hyps):
        r = normalize(refs[n]).split()
        out[n] = 100.0 * edit_distance(r, normalize(h or "").split()) / max(len(r), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline label")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    results, clips = {}, None
    for f in a.files:
        d = json.load(open(f))
        here = os.path.realpath(d["clips"])
        if clips is not None and here != clips:
            sys.exit(f"{f}: clips {here} differ from {clips}")
        clips = here
        for label, r in d["results"].items():
            if "hyps" in r:
                results[label] = r
    refs = json.load(open(os.path.join(clips, "refs.json")))
    if a.base not in results:
        sys.exit(f"baseline {a.base!r} not found")
    base = results[a.base]
    bclip = per_clip(refs, base["names"], base["hyps"])
    bhyp = dict(zip(base["names"], base["hyps"]))
    for label, r in results.items():
        if label == a.base:
            continue
        # Corpus WER is reported over each run's own clips, so a paired
        # comparison is only meaningful over the identical clip set.
        if sorted(r["names"]) != sorted(base["names"]):
            sys.exit(f"{label}: decoded clips differ from {a.base}'s; rerun on the same set")
        cclip = per_clip(refs, r["names"], r["hyps"])
        common = list(r["names"])
        diffs = [cclip[n] - bclip[n] for n in common]
        mean = sum(diffs) / len(diffs)
        rng = random.Random(0)  # per candidate: its CI must not depend on the other labels
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
        lo, hi = boots[249], boots[9749]
        changed = sum(dict(zip(r["names"], r["hyps"]))[n] != bhyp[n] for n in common)
        print(f"{label:24s} vs {a.base}: clipWER {mean:+.3f} pt [95% {lo:+.3f}, {hi:+.3f}]  "
              f"corpus {base['wer']:.2f}% -> {r['wer']:.2f}%  changed {changed}/{len(common)}  "
              f"file {r['size_mb']:.0f} MB  RTF {r['rtf']:.4f}  RSS {r['peak_rss_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
