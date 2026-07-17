"""Byte-exact parity tests for the ggml/CUDA engine vs the golden references.

The ggml engine (``benchmarks.engines.GgmlParakeet`` wrapping mudler's
parakeet.cpp persistent server) must reproduce the golden transcripts
BYTE-FOR-BYTE on the short/medium/long fixtures. The goldens were captured by
``scripts/parakeet_tdt_golden.py`` (the byte-exact eager greedy-TDT path), so a
text match is a real correctness gate, not a proxy.

This test is GATED: it skips if the parakeet-server binary or model is absent
(build the server in the parakeet.cpp repo first) or if CUDA is unavailable.

Run with:  uv run pytest tests/test_ggml_parity.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

import make_fixtures as mkfx  # noqa: E402

GOLDEN = _REPO_ROOT / "golden"
FIXTURES = mkfx.load_fixtures()  # {short, medium, long} -> 1-D float32 @16kHz


def _ggml_available() -> bool:
    """True iff the parakeet-server binary + model exist (so the test can run)."""
    try:
        from engines import GgmlParakeet  # noqa: F401
    except Exception:
        return False
    try:
        return GgmlParakeet().available
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ggml_available(),
    reason="parakeet-server binary or model not built (build it in the "
           "parakeet.cpp repo, or set GGML_PARAKEET_SERVER / GGML_PARAKEET_MODEL)",
)


@pytest.fixture(scope="module")
def ggml_engine():
    """One persistent parakeet-server for the whole module (load paid once)."""
    from engines import GgmlParakeet

    eng = GgmlParakeet()
    eng.load()
    yield eng
    eng.close()


@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_ggml_parakeet_byte_exact(ggml_engine, name: str) -> None:
    """The ggml engine transcript must match the golden BYTE-FOR-BYTE."""
    golden_text = (GOLDEN / f"parakeet_tdt_{name}_text.txt").read_text()
    out = ggml_engine._run_one(FIXTURES[name])
    assert out == golden_text, (
        f"parakeet-tdt ggml transcript mismatch on {name}:\n"
        f"  golden: {golden_text!r}\n  ggml:   {out!r}"
    )
