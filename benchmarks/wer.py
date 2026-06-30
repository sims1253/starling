"""Word/Corpus-Error-Rate utilities for the unified benchmark.

The fixtures (``tests/fixtures/make_fixtures.py``) are built deterministically
from a single LibriSpeech sample (``2086-149220-0033``) by plain concatenation:
``short`` = 1x, ``medium`` = 3x, ``long`` = 10x. The spoken transcript of that
sample is therefore the ground truth for every tier, and the reference text for
a tier is just the source transcript repeated to match.

We compute WER/CER against these references with the standard ``jiwer``
normalization (lowercase, strip punctuation, collapse whitespace) applied
identically to the reference and to every engine's hypothesis. So the WER here
is a real quality metric against ground truth -- not "does it match starling".

Public API
----------
``REFERENCE_TRANSCRIPTS : dict[str, str]``   -- ground-truth text per tier
``wer_pct(ref, hyp) -> float``               -- word error rate (percent)
``cer_pct(ref, hyp) -> float``               -- character error rate (percent)
``normalize(text) -> str``                   -- the shared normalization
"""

from __future__ import annotations

from jiwer import cer, wer
from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase

# The spoken transcript of LibriSpeech 2086-149220-0033 (the fixture source).
# Matches the byte-exact golden transcripts recorded in outputs/oracle.json
# (the only divergence there is a missing comma on a couple of "Well I" repeats
# at the long tier, which the normalization below erases).
_SOURCE_TRANSCRIPT = (
    "Well, I don't wish to see it any more, observed Phoebe, turning away her "
    "eyes. It is certainly very like the old portrait."
)

# Tier -> number of repetitions (mirrors FIXTURE_REPETITIONS in make_fixtures).
_TIER_REPEATS = {"short": 1, "medium": 3, "long": 10}

# Tier -> ground-truth transcript (the source repeated, space-separated).
REFERENCE_TRANSCRIPTS: dict[str, str] = {
    name: " ".join([_SOURCE_TRANSCRIPT] * reps)
    for name, reps in _TIER_REPEATS.items()
}


# Shared normalization pipeline. Applied to BOTH reference and hypothesis so the
# comparison is fair (case/punctuation differences -- e.g. the CrispASR granite
# GGUF lowercasing -- do not count as errors).
_NORMALIZE = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _NORMALIZE(text)


def wer_pct(ref: str, hyp: str) -> float:
    """Word error rate of ``hyp`` vs ``ref`` as a percent (0.0 = perfect)."""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 100.0
    return float(wer(ref_n, hyp_n)) * 100.0


def cer_pct(ref: str, hyp: str) -> float:
    """Character error rate of ``hyp`` vs ``ref`` as a percent (0.0 = perfect)."""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 100.0
    return float(cer(ref_n, hyp_n)) * 100.0
