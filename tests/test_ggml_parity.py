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

import os
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




@pytest.fixture(scope="module")
def ggml_engine():
    """One persistent parakeet engine for the whole module (load paid once).

    The default path is the in-process ctypes binding (fastest). ggml's global
    Backend static destructor aborts at process exit on some builds; that crash
    is after all tests pass and does not affect their outcome (pytest reports
    before the atexit crash on stdout-flushed runs). For environments where the
    atexit crash must be fully isolated, set GGML_PARAKEET_NATIVE=0 to use the
    HTTP-server path (ggml runs in a child process).
    """
    if not _ggml_available():
        pytest.skip("parakeet-server binary or model unavailable")
    from engines import GgmlParakeet

    eng = GgmlParakeet()
    eng.load()
    yield eng
    eng.close()


@pytest.mark.skipif(not _ggml_available(), reason="parakeet-server binary or model unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_ggml_parakeet_byte_exact(ggml_engine, name: str) -> None:
    """The ggml engine transcript must match the golden BYTE-FOR-BYTE.

    All three fixtures are byte-exact via the in-process C API
    (``parakeet_capi_transcribe_pcm``): short/medium use the K-step multistep
    decode fast path (T<=512); long (T=930) uses the byte-exact serial greedy
    loop (the multistep has a termination bug on long, so it's guarded out --
    see ``docs/ggml-parakeet-perf-analysis.md`` and parakeet.cpp 147ba98).
    """
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
    not (_ggml_available() and _ggml_moss_available()),
    reason="parakeet-server or CrispASR MOSS binary/model unavailable (preserves "
           "the historical external-engine module gate)",
)


@pytest.fixture(scope="module")
def ggml_moss_engine():
    """One CrispASR moss-transcribe engine for the whole module (skipped if
    unavailable)."""
    if not (_ggml_available() and _ggml_moss_available()):
        pytest.skip("historical external-engine gate: parakeet/CrispASR MOSS unavailable")
    from engines import GgmlMoss

    eng = GgmlMoss()
    eng.load()
    yield eng
    eng.close()


@moss
@pytest.mark.parametrize("name", ["short"])
def test_ggml_moss_byte_exact(ggml_moss_engine, name: str) -> None:
    """Moss ggml engine matches the golden BYTE-FOR-BYTE on the short fixture.

    The short fixture is the byte-exact gate. Both invocation paths (persistent
    server and one-shot CLI fallback) reproduce it exactly because the audio
    fits one 30 s chunk and the decode has not yet accumulated enough KV-cache
    context to diverge from the golden's HF eager greedy path.
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

    The residual divergence is NOT a flag/CrispASR-post-processing issue -- it
    is an inherent numeric-path difference between CrispASR's ggml f16 KV-cache
    decode and the golden's HF bf16 eager greedy decode. At the low-confidence
    ``eyes`` repetition boundary the two argmax results flip: CrispASR emits
    ``eyes. It`` (period + capital) where the golden emits ``eyes it``. This was
    confirmed by inspecting the raw token stream (``moss_transcribe: N tokens``
    verbose log): the period appears in the LLM output itself, before any
    post-processing. Flags tried that do NOT fix it: ``--no-punctuation`` (strips
    ALL punctuation, including the golden's own commas -- too aggressive),
    ``-nfa`` (no flash attention), ``--frequency-penalty 0``, ``-bs greedy``.

    Characterized residual (one-shot CLI / server single-chunk, byte-identical):
      * medium: 2 inserted periods -- golden has ``eyes. it`` only at boundary 1
        (no period at boundaries 2,3); CrispASR has ``eyes. it`` at all 3.
        normalized CER = 0.0000.
      * long:   6 inserted periods + 6 capitalizations (``eyes. It`` at every
        boundary; golden has ``eyes it`` throughout) + a different EOS/truncation
        point (CrispASR's extra period tokens change where the LLM emits
        ``<|im_end|>``, so the 200-token golden truncation point is not matched).
        normalized CER < 0.02.

    The CER bound catches any regression beyond this known, documented gap
    without failing on the residual punctuation/capitalization differences.
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
    # medium's normalized CER is 0.0; long's is <0.02. 0.10 leaves headroom for
    # the server path's 30 s-chunk long-audio boundary differences while still
    # catching a real regression (a totally broken decode is CER >> 0.10).
    assert cer < 0.10, (
        f"moss ggml CER too high on {name} (cer={cer:.3f}):\n"
        f"  golden: {g_n[:120]!r}\n  ggml:   {o_n[:120]!r}"
    )


# --------------------------------------------------------------------------- #
# In-tree MOSS C API
# --------------------------------------------------------------------------- #
def _starling_ggml_moss_available() -> bool:
    try:
        from engines import StarlingGgmlMoss
        return StarlingGgmlMoss().available
    except Exception:
        return False

@pytest.fixture(scope="module")
def starling_ggml_moss_engine():
    if not _starling_ggml_moss_available():
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_MOSS_MODEL unavailable")
    from engines import StarlingGgmlMoss
    engine = StarlingGgmlMoss()
    engine.load()
    yield engine
    engine.close()

@pytest.mark.skipif(not _starling_ggml_moss_available(),
                    reason="in-tree libstarling_ggml or MOSS GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_moss_text_parity(starling_ggml_moss_engine, name: str) -> None:
    """The in-tree C API returns the golden MOSS transcript exactly.

    This deliberately has no CrispASR tolerance: it gates Starling's own
    loader → mel → encoder → adapter → prompt → LLM → detokenizer pipeline.
    """
    golden_text = (GOLDEN / f"moss_{name}_text.txt").read_text().rstrip()
    out = starling_ggml_moss_engine._run_one(FIXTURES[name]).rstrip()
    assert out == golden_text, (
        f"in-tree MOSS transcript mismatch on {name}:\n"
        f"  golden: {golden_text!r}\n  ggml:   {out!r}"
    )
