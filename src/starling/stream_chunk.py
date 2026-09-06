"""Fixed-window overlapping-chunk streaming for long-form live dictation.

The naive ``/stream`` loop re-transcribes the *entire* growing buffer on every
partial.  Over a long dictation that is O(N^2) work, the LLM prompt grows without
bound (a multi-minute utterance overflows any static KV cache), and -- because
each buffer length is unique -- the CUDA-graph encoder can never reuse a capture.

This module finalizes the buffer in **fixed-length** windows that overlap by a
small margin:

* Each finalized window is a *constant* mel length, so the adaptive cudagraph
  encoder captures it once and replays it thereafter.
* Work is O(N): each second of audio is transcribed a bounded number of times.
* The prompt per transcribe is bounded by the window, so ``max_cache_len`` is
  never exceeded regardless of total session length.
* Consecutive windows overlap, so boundary words are never dropped -- their
  transcripts are stitched with :func:`stitch_words`.

Stitching has no word timestamps to lean on (MOSS emits one segment per call),
so it works on the decoded word sequence: consecutive windows share ``overlap``
seconds of audio and therefore transcribe the same words there.  We align the
longest common run in that region (``difflib``) and splice, deduping the overlap.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

# Retry/backoff for the final tail window on commit when the transcriber is busy.
_FLUSH_TAIL_MAX_RETRIES = 5
_FLUSH_TAIL_BACKOFF_SECONDS = 0.05

_WORD_NORM = re.compile(r"[^\w']+")


def _norm(word: str) -> str:
    """Lowercase, strip surrounding punctuation -- for overlap *matching* only."""
    return _WORD_NORM.sub("", word.lower())


def stitch_words(
    committed: list[str],
    new: list[str],
    *,
    max_overlap: int = 24,
    min_match: int = 2,
) -> list[str]:
    """Append ``new`` to ``committed``, deduping the overlapping boundary words.

    Looks at the last / first ``max_overlap`` words of ``committed`` / ``new``
    (the region the two windows share) and aligns the longest common run there.
    Words up to the end of that run are kept from ``committed``; everything after
    it is taken from ``new``.  If no run of at least ``min_match`` words is found
    the two are simply concatenated (rare; a duplicated word reads better than a
    dropped one for dictation).
    """
    if not committed:
        return list(new)
    if not new:
        return list(committed)

    tail = committed[-max_overlap:]
    head = new[:max_overlap]
    a = [_norm(w) for w in tail]
    b = [_norm(w) for w in head]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    if m.size >= min_match:
        keep = len(committed) - len(tail) + m.a + m.size  # committed up to run end
        start = m.b + m.size                              # new after the run
        return list(committed[:keep]) + list(new[start:])
    return list(committed) + list(new)


# Text transcription of a mono float32 window -> its text, or ``None`` if the
# transcribe could not run right now (e.g. server busy) and this step should be
# skipped without advancing state.
TranscribeFn = Callable[[np.ndarray], Optional[str]]


class ChunkStreamer:
    """Rolling fixed-window overlapping-chunk transcription state.

    Owns the committed transcript and the finalized-audio boundary.  Feed it the
    session's full sample buffer plus a text-transcribe callback; it finalizes
    any full windows and returns the current best text (committed + live tail).
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        chunk_seconds: float,
        overlap_seconds: float,
        min_seconds: float,
        partial_interval_seconds: float,
    ) -> None:
        self.sr = int(sample_rate)
        self.chunk = int(chunk_seconds * sample_rate)
        self.overlap = int(min(overlap_seconds, chunk_seconds * 0.5) * sample_rate)
        self.advance = max(1, self.chunk - self.overlap)
        self.min = int(min_seconds * sample_rate)
        self.partial_interval = float(partial_interval_seconds)
        # overlap words to search when stitching (~speech rate * overlap + margin)
        self.max_overlap_words = max(8, int(overlap_seconds * 6) + 6)

        self.committed: list[str] = []
        self.boundary = 0          # sample index; audio before this is finalized
        self.last_emit = 0.0

    # ------------------------------------------------------------------ #
    def _finalize_full_windows(self, samples: np.ndarray, tx: TranscribeFn) -> bool:
        """Finalize every complete window at the current boundary. Returns True
        if at least one window was committed."""
        did = False
        while (len(samples) - self.boundary) >= self.chunk:
            window = samples[self.boundary : self.boundary + self.chunk]
            text = tx(window)
            if text is None:  # busy/cancelled -> stop; boundary unchanged for retry
                break
            self.committed = stitch_words(
                self.committed, text.split(), max_overlap=self.max_overlap_words
            )
            self.boundary += self.advance
            did = True
        return did

    def step(self, samples: np.ndarray, now: float, tx: TranscribeFn) -> Optional[str]:
        """Advance streaming state for the current buffer.

        Finalizes any full windows, then (throttled) transcribes the live tail
        for a responsive partial.  Returns the full text to emit, or ``None`` if
        nothing should be emitted this tick.
        """
        finalized = self._finalize_full_windows(samples, tx)

        tail_len = len(samples) - self.boundary
        throttled = (now - self.last_emit) < self.partial_interval
        # emit if we just finalized, or the (throttled) live tail is long enough
        if not finalized and (throttled or tail_len < self.min):
            return None
        self.last_emit = now

        # emit committed + the live tail (transcribed only if long enough)
        if tail_len >= self.min:
            text = tx(samples[self.boundary :])
            if text is None:  # busy on the tail
                return " ".join(self.committed) if finalized else None
            return " ".join(stitch_words(
                self.committed, text.split(), max_overlap=self.max_overlap_words
            ))
        return " ".join(self.committed) if finalized else None

    def flush(self, samples: np.ndarray, tx: TranscribeFn) -> str:
        """Finalize all remaining audio (on ``commit``) and return the full text.

        If the transcribe callback is busy (returns ``None``), the tail window is
        retried a few times with a short backoff.  If it still cannot be
        transcribed, the tail is dropped and a warning is logged -- committed
        text is still returned so the caller receives the bulk of the dictation.
        """
        self._finalize_full_windows(samples, tx)
        tail = samples[self.boundary :]
        if len(tail) > 0:
            text = None
            for attempt in range(_FLUSH_TAIL_MAX_RETRIES):
                text = tx(tail)
                if text is not None:
                    break
                time.sleep(_FLUSH_TAIL_BACKOFF_SECONDS)
            if text is not None:
                # Any non-None result is a success, including an empty string
                # (silence transcribed to no words). stitch_words is a no-op for
                # empty ``new`` (returns committed unchanged), and we still
                # advance the boundary so this tail isn't retried.
                self.committed = stitch_words(
                    self.committed, text.split(), max_overlap=self.max_overlap_words
                )
                self.boundary = len(samples)
            else:
                log.warning(
                    "flush: dropped untranscribed tail (%d samples) after %d "
                    "retries; transcriber remained busy",
                    len(tail), _FLUSH_TAIL_MAX_RETRIES,
                )
        return " ".join(self.committed)

    def reset(self) -> None:
        self.committed = []
        self.boundary = 0
        self.last_emit = 0.0
