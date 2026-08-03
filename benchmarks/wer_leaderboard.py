"""Open ASR Leaderboard scoring: normalization + WER + RTFx.

Reproduces the *accuracy* methodology of the Hugging Face Open ASR Leaderboard
(https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) English
short-form eval, so the headline number is a real quality metric on diverse
real audio -- not the synthetic-tile WER of ``wer.py`` (which only measures
"did starling drift from stock transformers?" for the byte-exact gate).

Methodology (verified against the leaderboard's ``normalizer/eval_utils.py``
and ``normalizer/data_utils.py``):

1. **Normalization.** The Whisper ``EnglishTextNormalizer`` is applied to BOTH
   the reference and the hypothesis. We use the ``whisper-normalizer`` PyPI
   package's implementation (the Whisper-paper normalizer). The leaderboard's
   in-repo normalizer adds extra acronym/name/compound layers on top; that
   extended normalizer is not packaged, so this is the closest available
   faithful reproduction. Differences only affect edge-case numeric/acronym
   tokens and wash out across thousands of clips.

2. **WER engine.** NOT jiwer: the leaderboard uses
   ``kaldialign.batch_error_rate(refs, hyps, merge_compounds=True)``.
   ``merge_compounds=True`` makes whitespace-only compound differences (e.g.
   "white paper" vs "whitepaper") count as 0 errors. WER is reported in
   percent, rounded to 2 dp.

3. **Composite.** The leaderboard's headline WER is the *unweighted mean* of
   the per-dataset WERs (one WER per dataset, then averaged across the 7
   datasets) -- NOT a micro-average over all clips.

4. **RTFx.** real-time factor = total audio duration / total inference time
   (higher = faster), summed across clips per dataset, matching the
   leaderboard.

This module is deliberately separate from ``wer.py``: that one is the
byte-exact *correctness* gate (tiled fixtures, starling-vs-stock drift);
this one is the *quality* metric (real diverse audio, absolute WER). Both
coexist.

Public API
----------
``normalize(text)``                  -- EnglishTextNormalizer __call__
``wer_pct(refs, hyps)``              -- kaldialign WER (%) over aligned lists
``score_dataset(refs, hyps, times_s, durations_s)``
                                     -- {wer_pct, rtfx, n, ins/del/sub}
``composite_wer(per_dataset)``       -- unweighted mean of per-dataset WERs
"""

from __future__ import annotations

from typing import Sequence

import kaldialign
from whisper_normalizer.english import EnglishTextNormalizer

_NORMALIZER = EnglishTextNormalizer()


def normalize(text: str) -> str:
    """Whisper English normalization (lowercase, de-punct, expand contractions/
    numbers/spellings). Applied to BOTH reference and hypothesis."""
    return _NORMALIZER(text)


def wer_pct(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Open-ASR-Leaderboard WER (%) for aligned reference/hypothesis lists.

    Uses kaldialign with ``merge_compounds=True`` so whitespace-only compound
    diffs are free. Returns ``nan`` if there are no (post-normalization) refs.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")
    r = [tuple(normalize(t).split()) for t in refs]
    h = [tuple(normalize(t).split()) for t in hyps]
    # kaldialign needs at least one non-empty ref; empty clips were already
    # filtered by the corpus loader, but guard anyway.
    if not any(r):
        return float("nan")
    res = kaldialign.batch_error_rate(r, h, merge_compounds=True)
    return round(100.0 * float(res["err_rate"]), 2)


def score_dataset(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    times_s: Sequence[float] | None = None,
    durations_s: Sequence[float] | None = None,
) -> dict:
    """Score one dataset: WER, optional RTFx, and edit counts.

    Args:
        refs: raw reference transcripts (normalized internally).
        hyps: raw hypothesis transcripts, aligned 1:1 with refs.
        times_s: per-clip inference time (s); if given with durations, RTFx.
        durations_s: per-clip audio duration (s).
    """
    wer = wer_pct(refs, hyps)
    out: dict = {"wer_pct": wer, "n": len(refs)}
    # edit counts from one kaldialign pass (over normalized word lists)
    r = [tuple(normalize(t).split()) for t in refs]
    h = [tuple(normalize(t).split()) for t in hyps]
    if any(r):
        res = kaldialign.batch_error_rate(r, h, merge_compounds=True)
        out.update(ins=int(res["ins"]), dele=int(res["del"]),
                   sub=int(res["sub"]), ref_len=int(res["ref_len"]))
    if times_s is not None and durations_s is not None and times_s and durations_s:
        total_dur = sum(durations_s)
        total_time = sum(times_s)
        out["rtfx"] = round(total_dur / total_time, 2) if total_time > 0 else float("inf")
        out["audio_s"] = round(total_dur, 1)
        out["infer_s"] = round(total_time, 1)
    return out


def composite_wer(per_dataset: dict[str, dict]) -> float:
    """Unweighted mean of per-dataset WERs (the leaderboard headline)."""
    vals = [d["wer_pct"] for d in per_dataset.values()
            if d and d.get("wer_pct") is not None
            and d["wer_pct"] == d["wer_pct"]]  # filter nan
    if not vals:
        return float("nan")
    return round(sum(vals) / len(vals), 2)
