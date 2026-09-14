#!/usr/bin/env python3
"""Source-checkout entry point; run with the root uv environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from starling.server import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run())
