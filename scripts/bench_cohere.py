"""Convenience wrapper to benchmark the cohere-transcribe megakernel.

Delegates to ``benchmarks.cohere.bench_pipeline``.

Run:  ``uv run python scripts/bench_cohere.py [--compare-stock] [--K N]``
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from benchmarks.cohere.bench_pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
