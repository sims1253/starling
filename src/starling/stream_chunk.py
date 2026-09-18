"""Fixed-window overlapping-chunk streaming for long-form live dictation.

Full windows overlap so boundary words can be matched with :func:`stitch_words`.
Matching uses decoded words, without timestamps; it can miss or duplicate words
when neighboring windows disagree. Each transcription call is bounded by the
configured window size, including retries when the server is busy.
"""

from __future__ import annotations

import difflib
import math
import re
import time
from typing import Callable, Optional

import numpy as np

# Bounded retries on commit when the transcriber is busy.
_FLUSH_MAX_RETRIES = 5
_FLUSH_BACKOFF_SECONDS = 0.05

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
    # Empty keys (words that normalize to "", e.g. pure punctuation) never
    # participate in a match: runs of empty keys would align unrelated
    # boundary words and drop them (issue #118 defense in depth; kept in
    # lockstep with the C++ port in cpp/serve/stream_session.cpp). The
    # sentinels are unique per side and position and cannot collide with a
    # real key -- ``_norm`` strips ``\x00`` along with other non-word chars.
    a = [key or f"\x00tail-{i}" for i, key in enumerate(a)]
    b = [key or f"\x00head-{i}" for i, key in enumerate(b)]
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

# Largest sample count the chunk counters hold; kept in lockstep with the int
# counters of the C++ port (cpp/serve/stream_session.cpp).
_MAX_SAMPLES = 2**31 - 1


def stream_window_config_error(
    *,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    min_seconds: float,
    partial_interval_seconds: float,
) -> Optional[str]:
    """Validate stream window configuration (issue #146).

    Returns ``None`` when valid, otherwise a human-readable error.  Checks
    finite values, nonnegative ranges, and the window/overlap relationships;
    kept in lockstep with ``stream_window_config_error`` in
    cpp/serve/stream_session.cpp.

    ``chunk_seconds == 0`` is valid ONLY at the CLI level (it selects the
    legacy whole-buffer mode and no chunker is constructed);
    :class:`ChunkStreamer` rejects it separately because a chunker needs a
    real window.
    """
    if int(sample_rate) <= 0:
        return f"stream sample rate must be positive (got {sample_rate})"
    for name, value in (
        ("stream chunk seconds", chunk_seconds),
        ("stream overlap seconds", overlap_seconds),
        ("stream min chunk seconds", min_seconds),
        ("stream partial interval seconds", partial_interval_seconds),
    ):
        if not math.isfinite(value):
            return f"{name} must be a finite number"
    if chunk_seconds < 0:
        return "stream chunk seconds must be nonnegative (0 selects whole-buffer mode)"
    if overlap_seconds < 0:
        return "stream overlap seconds must be nonnegative"
    if min_seconds < 0:
        return "stream min chunk seconds must be nonnegative"
    if partial_interval_seconds < 0:
        return "stream partial interval seconds must be nonnegative"
    max_seconds = _MAX_SAMPLES / int(sample_rate)
    if chunk_seconds > 0:
        if chunk_seconds * sample_rate < 1:
            return "stream chunk seconds too small: the window is shorter than one sample at this rate"
        if overlap_seconds >= chunk_seconds:
            return "stream overlap seconds must be smaller than stream chunk seconds"
        if chunk_seconds > max_seconds:
            return "stream chunk seconds too large: the window does not fit the sample counters"
    if min_seconds > max_seconds:
        return "stream min chunk seconds too large: the minimum does not fit the sample counters"
    return None


class ChunkStreamer:
    """Rolling fixed-window overlapping-chunk transcription state.

    Owns the committed transcript and the finalized-audio boundary.  Feed it the
    session's full sample buffer plus a text-transcribe callback; it finalizes
    any full windows and returns the current best text (committed + live tail).

    Raises ``ValueError`` when the window configuration is invalid (see
    :func:`stream_window_config_error`; ``chunk_seconds`` must be positive
    here — 0 selects the legacy whole-buffer mode and never builds a chunker).
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
        # Validate the whole configuration before deriving anything, then
        # build the state in dependency order (issue #146, kept in lockstep
        # with the C++ port): the old init computed self.advance from the
        # unclamped self.overlap, so a negative overlap baked itself into the
        # window advance and the late clamp below could not fix it.
        if chunk_seconds <= 0:
            raise ValueError(
                "stream chunk seconds must be positive to build a chunked stream"
            )
        error = stream_window_config_error(
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            min_seconds=min_seconds,
            partial_interval_seconds=partial_interval_seconds,
        )
        if error is not None:
            raise ValueError(error)

        self.sr = int(sample_rate)
        self.chunk = int(chunk_seconds * sample_rate)
        # Normalize overlap before deriving advance: clamp to [0, chunk/2]
        # (overlap >= chunk was already rejected above).
        overlap = max(0.0, min(overlap_seconds, chunk_seconds * 0.5))
        self.overlap = int(overlap * sample_rate)
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
        if tail_len >= self.chunk:  # a full window is still waiting for a retry
            return " ".join(self.committed) if finalized else None
        throttled = (now - self.last_emit) < self.partial_interval
        # emit if we just finalized, or the (throttled) live tail is long enough
        if not finalized and (throttled or tail_len < self.min):
            return None
        self.last_emit = now

        # emit committed + the live tail (transcribed only if long enough);
        # never hand the transcriber an empty window (issue #146)
        if tail_len > 0 and tail_len >= self.min:
            text = tx(samples[self.boundary :])
            if text is None:  # busy on the tail
                return " ".join(self.committed) if finalized else None
            return " ".join(stitch_words(
                self.committed, text.split(), max_overlap=self.max_overlap_words
            ))
        return " ".join(self.committed) if finalized else None

    def flush(self, samples: np.ndarray, tx: TranscribeFn) -> Optional[str]:
        """Commit all audio, or return ``None`` if bounded retries stay busy.

        Completed windows remain committed. The caller must retain the audio
        and retry commit after ``None``; only a string result is final.
        """
        for attempt in range(_FLUSH_MAX_RETRIES):
            self._finalize_full_windows(samples, tx)
            tail = samples[self.boundary :]
            if len(tail) == 0:
                return " ".join(self.committed)
            # Guard the tail's sign too (issue #146): the window geometry is
            # validated at construction, but the transcriber contract (a
            # nonempty window) is enforced here regardless.
            if 0 < len(tail) < self.chunk:
                text = tx(tail)
                if text is not None:
                    self.committed = stitch_words(
                        self.committed, text.split(), max_overlap=self.max_overlap_words
                    )
                    self.boundary = len(samples)
                    return " ".join(self.committed)
            if attempt + 1 < _FLUSH_MAX_RETRIES:
                time.sleep(_FLUSH_BACKOFF_SECONDS)
        return None

    def reset(self) -> None:
        self.committed = []
        self.boundary = 0
        self.last_emit = 0.0
