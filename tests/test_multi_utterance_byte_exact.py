"""Regression test: starling graphed paths must be byte-exact across MANY
utterances decoded through ONE captured graph, not just the single utterance
the graph was captured with.

The original per-model byte-exact tests (test_cohere/test_qwen3/test_ark/...)
capture a graph on utterance A and decode that SAME utterance A. They pass even
when the graphed path silently reuses capture-time state (frozen encoder hidden
states, stale cross-attention K/V, a static buffer never re-bound) — because
utterance A's state IS the capture-time state. The bug only appears when
utterance B (different audio) is decoded through A's captured graph: the stale
state makes B return A's transcript.

This is exactly what the Open ASR Leaderboard sweep surfaced: cohere returned
clip 0's transcript for every clip; ark OOM'd accumulating per-shape graphs;
qwen3 crashed on a shape mismatch; parakeet's allocator thrashed. Each was a
"works on one utterance, breaks on many" graph-capture bug that the single-
utterance tests could not catch.

This test drives each starling engine through several DIFFERENT real clips
(cached leaderboard corpus) and asserts the transcript is byte-identical to
that model's stock reference for EVERY clip. It is the multi-utterance
correctness gate the leaderboard needs.

Skips gracefully when the corpus cache or a model is unavailable (no network,
no GPU, golden models absent) so it never breaks a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "leaderboard_corpus"


def _clips(dataset: str = "spgispeech", n: int = 4):
    """Return n cached leaderboard clips, or skip if the cache is absent.

    Uses the spgispeech split (all clips exactly 15.0s — uniform length, so a
    single graph shape serves them all, which is the regime where stale-graph-
    state bugs hide: the captured graph is reused, not re-captured, so any
    per-utterance state that wasn't re-bound stays frozen at clip 0).
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))
    try:
        import leaderboard_corpus as lc
    except ImportError:
        pytest.skip("leaderboard_corpus module unavailable")
    d = CORPUS_DIR / f"{dataset}__n{n}"
    if not (d / "reference.json").exists():
        pytest.skip(f"leaderboard corpus cache {d} absent; run "
                    f"bench_leaderboard.py --num-samples {n} first")
    clips = lc.load_dataset_split(dataset, num_samples=n)
    if len(clips) < n:
        pytest.skip(f"only {len(clips)} clips cached (need {n})")
    return clips


def _starling_vs_stock_byte_exact(model: str, clips) -> None:
    """Assert starling transcript == stock transcript for every clip."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from engines import build_engines
    from starling.parakeet.gpu_lock import with_gpu_lock

    em = build_engines([model], ["starling", "stock"])
    if not em.get(model) or len(em[model]) < 2:
        pytest.skip(f"starling+stock engines for {model} unavailable")
    starling, stock = em[model]
    mismatches = []
    with with_gpu_lock(session="multiutt", model=model, eta_min=30,
                       note="multi-utterance byte-exact test"):
        starling.load()
        stock.load()
        for i, clip in enumerate(clips):
            hyp = starling.transcribe(clip.audio, B=1)[0].strip()
            ref = stock.transcribe(clip.audio, B=1)[0].strip()
            if hyp != ref:
                mismatches.append((i, hyp[:50], ref[:50]))
        starling.close()
        stock.close()
    assert not mismatches, (
        f"{model}: starling diverged from stock on {len(mismatches)}/{len(clips)} "
        f"clips (graph-capture stale-state bug). First mismatches: {mismatches[:3]}")


# One parametrized case per model that has both a starling and a stock engine
# and a real graphed-decode path. higgs is excluded (isolated venv).
@pytest.mark.parametrize("model", ["granite", "parakeet", "moss", "qwen3",
                                   "cohere", "ark"])
def test_starling_matches_stock_across_clips(model):
    """Decoding N different clips through the starling graphed path must match
    the stock reference for every clip — not just the capture utterance."""
    clips = _clips()
    _starling_vs_stock_byte_exact(model, clips)
