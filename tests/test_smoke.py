"""Smoke / regression test for the starling baseline correctness oracle.

Loads the stock parakeet-tdt model, transcribes the SHORT fixture, and asserts the
transcript matches the GOLD entry in outputs/oracle.json byte-for-byte. Future
optimized kernels reuse this as a regression gate: any divergence here means the
new path is no longer equivalent to the reference.

Run with:  uv run pytest tests/test_smoke.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402
from starling.baseline import BaselineRunner  # noqa: E402

ORACLE_PATH = _REPO_ROOT / "outputs" / "oracle.json"

# Loading the model is expensive (~25s); cache it across tests in the module.
_RUNNER: BaselineRunner | None = None


def _get_runner() -> BaselineRunner:
    global _RUNNER
    if _RUNNER is None:
        import torch  # noqa: WPS433

        _RUNNER = BaselineRunner(dtype=torch.bfloat16)
    return _RUNNER


def _oracle_short() -> dict:
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle fixture missing: {ORACLE_PATH} (run benchmarks/bench_rtf.py first)")
    oracle = json.loads(ORACLE_PATH.read_text())
    for entry in oracle:
        if entry["name"] == "short":
            return entry
    pytest.fail("oracle.json has no 'short' entry")


def test_oracle_fixture_exists():
    """The oracle must exist and contain a non-empty short transcript."""
    entry = _oracle_short()
    assert entry["text"].strip(), "oracle short transcript must be non-empty"
    assert entry["num_tokens"] > 0


def test_short_transcript_matches_oracle():
    """The stock path must reproduce the oracle transcript byte-for-byte."""
    entry = _oracle_short()
    fixtures = mkfx.load_fixtures()
    runner = _get_runner()
    text, ntok = runner.oracle_transcribe(fixtures["short"])
    assert text == entry["text"], (
        f"transcript drift:\n  oracle: {entry['text']!r}\n  now:    {text!r}"
    )
    assert ntok == entry["num_tokens"], (
        f"token count drift: oracle={entry['num_tokens']} now={ntok}"
    )


def test_batched_decode_is_stable():
    """Batched decoding must reproduce each fixture's oracle transcript."""
    oracle = {e["name"]: e for e in json.loads(ORACLE_PATH.read_text())} if ORACLE_PATH.exists() else {}
    if not oracle:
        pytest.skip("oracle missing")
    fixtures = mkfx.load_fixtures()
    runner = _get_runner()
    texts, ntoks = runner.transcribe_batch(
        [fixtures["short"], fixtures["medium"], fixtures["long"]], return_tokens=True
    )
    # short is first in the batch and was padded; ensure it still decodes to gold.
    assert texts[0] == oracle["short"]["text"], "batched short transcript drifted"
