"""Golden reference access for the Granite-Speech-4.1-2b-NAR track.

Artefacts live under ``golden/nar/`` (gitignored) and are produced by
``scripts/nar_golden.py`` from the stock eager ``model.transcribe`` path.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from .config import GOLDEN_DIR

TIERS = ("short", "medium", "long")


def load_golden(name: str) -> torch.Tensor:
    """Load a tensor artefact by short name (e.g. ``"short_preds.pt"``)."""
    return torch.load(GOLDEN_DIR / name, map_location="cpu")


def load_golden_json() -> dict[str, Any]:
    """Load the ``golden.json`` summary."""
    return json.loads((GOLDEN_DIR / "golden.json").read_text(encoding="utf-8"))


def golden_text(tier: str) -> str:
    """Return the golden transcript string for a tier."""
    return load_golden_json()[tier]["text"]
