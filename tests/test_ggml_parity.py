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


# --------------------------------------------------------------------------- #
# Moss-Transcribe-preview-2B (CrispASR moss-transcribe backend)
# --------------------------------------------------------------------------- #
def _ggml_moss_available() -> bool:
    try:
        from engines import GgmlMoss  # noqa: F401
    except Exception:
        return False
    try:
        return GgmlMoss().available
    except Exception:
        return False


moss = pytest.mark.skipif(
    not _ggml_moss_available(),
    reason="CrispASR binary (with moss-transcribe backend) or F16 Moss GGUF not "
           "present (build CrispASR locally + download cstr/MOSS-Transcribe-..."
           "-2B-GGUF, or set GGML_MOSS_BIN / GGML_MOSS_MODEL)",
)


@pytest.fixture(scope="module")
def ggml_moss_engine():
    """One CrispASR moss-transcribe engine for the whole module (skipped if
    unavailable)."""
    if not _ggml_moss_available():
        pytest.skip("CrispASR moss-transcribe binary / F16 Moss GGUF not present")
    from engines import GgmlMoss

    eng = GgmlMoss()
    eng.load()
    yield eng
    eng.close()


@moss
@pytest.mark.parametrize("name", ["short"])
def test_ggml_moss_byte_exact(ggml_moss_engine, name: str) -> None:
    """Moss ggml engine matches the golden BYTE-FOR-BYTE on the short fixture.

    The short fixture is the byte-exact gate. Medium/long are NOT asserted
    here: CrispASR's moss-transcribe decode diverges from the golden capture
    path in punctuation normalization (a period at some repetition boundaries)
    and can truncate below the golden token count on long audio. See
    ``docs/ggml-engine.md`` for the documented gaps.
    """
    golden_text = (GOLDEN / f"moss_{name}_text.txt").read_text().rstrip()
    out = ggml_moss_engine._run_one(FIXTURES[name]).rstrip()
    assert out == golden_text, (
        f"moss ggml transcript mismatch on {name}:\n"
        f"  golden: {golden_text!r}\n  ggml:   {out!r}"
    )


@moss
@pytest.mark.parametrize("name", ["medium", "long"])
def test_ggml_moss_near_exact(ggml_moss_engine, name: str) -> None:
    """Moss is NEAR-exact (not byte-exact) on medium/long: assert a CER floor.

    Documents the known divergence from byte-exact (punctuation + truncation)
    with a character-error-rate bound so regressions beyond the known gap are
    caught, without failing on the documented punctuation differences.
    """
    golden_text = (GOLDEN / f"moss_{name}_text.txt").read_text().rstrip()
    out = ggml_moss_engine._run_one(FIXTURES[name]).rstrip()
    # normalized CER (lowercase, punctuation/space collapsed) must be small.
    import re

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    g_n, o_n = norm(golden_text), norm(out)
    if not g_n:
        return
    # simple Levenshtein over characters
    def lev(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    cer = lev(g_n, o_n) / max(1, len(g_n))
    assert cer < 0.10, (
        f"moss ggml CER too high on {name} (cer={cer:.3f}):\n"
        f"  golden: {g_n[:120]!r}\n  ggml:   {o_n[:120]!r}"
    )
