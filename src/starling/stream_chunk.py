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

# Asked right before a preview would start: True when newer audio (or a
# control message) is already queued, so the preview would be obsolete before
# it finished (issue #357).
PendingFn = Callable[[], bool]

# Largest sample count the chunk counters hold; kept in lockstep with the int
# counters of the C++ port (cpp/serve/stream_session.cpp).
_MAX_SAMPLES = 2**31 - 1

# Preview cadence bound (issue #357; kMaxPreviewDuty in the C++ port): the
# effective preview interval is at least the latest preview's engine time
# divided by this duty, so previews take at most this fraction of wall time
# on a slow model or device. Window and finalization work is never throttled.
MAX_PREVIEW_DUTY = 0.5


def preview_policy_error(
    *, sample_rate: int, min_seconds: float, interval_seconds: float
) -> Optional[str]:
    """Validate a preview cadence (issue #357); ``None`` when valid."""
    if not (math.isfinite(min_seconds) and math.isfinite(interval_seconds)):
        return "preview cadence must be finite"
    if min_seconds < 0 or interval_seconds < 0:
        return "preview cadence must be nonnegative"
    if int(sample_rate) <= 0 or min_seconds > _MAX_SAMPLES / int(sample_rate):
        return "first-partial minimum too large"
    return None


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

    The window geometry (chunk/overlap) decides what is committed; the
    preview cadence (``min_seconds``/``partial_interval_seconds``) only
    decides when the live tail is previewed (issue #357). ``min_seconds`` is
    the take's first-partial minimum: once the take holds that much audio,
    every nonempty tail is eligible, including the overlap right after a
    window commit.
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
        self.rebased = 0           # samples dropped before index 0 (rebase())
        self.last_emit = 0.0
        self.emit_due = False      # a commit changed the text, not yet emitted
        self.last_preview_cost = 0.0  # seconds, latest successful preview
        self.coalesced = 0         # previews skipped for newer queued audio
        # Why the transcribe call in flight was made (issue #226): "window",
        # "preview", "flush_window" or "flush_tail". Set before every call;
        # the session's call ledger reads it (lockstep with call_kind() in
        # cpp/serve/stream_session.hpp).
        self.call_kind = "window"

    def set_preview_policy(self, min_seconds: float, interval_seconds: float) -> None:
        """Replace the preview cadence (per connection); ``ValueError`` if invalid."""
        error = preview_policy_error(sample_rate=self.sr, min_seconds=min_seconds,
                                     interval_seconds=interval_seconds)
        if error is not None:
            raise ValueError(error)
        self.min = int(min_seconds * self.sr)
        self.partial_interval = float(interval_seconds)

    def samples_to_next_window(self, n_samples: int) -> int:
        """Audio still needed before the next window commit (samples)."""
        return max(0, self.chunk - (n_samples - self.boundary))

    @property
    def effective_interval(self) -> float:
        """The configured interval, stretched so the latest preview's engine
        time is at most ``MAX_PREVIEW_DUTY`` of it."""
        return max(self.partial_interval, self.last_preview_cost / MAX_PREVIEW_DUTY)

    def rebase(self, dropped: int) -> None:
        """Follow the session dropping ``dropped`` finalized samples."""
        self.boundary = max(0, self.boundary - dropped)
        self.rebased += dropped

    # ------------------------------------------------------------------ #
    def _finalize_full_windows(
        self, samples: np.ndarray, tx: TranscribeFn, *, flushing: bool = False
    ) -> bool:
        """Finalize every complete window at the current boundary. Returns True
        if at least one window was committed."""
        did = False
        while (len(samples) - self.boundary) >= self.chunk:
            window = samples[self.boundary : self.boundary + self.chunk]
            self.call_kind = "flush_window" if flushing else "window"
            text = tx(window)
            if text is None:  # busy/cancelled -> stop; boundary unchanged for retry
                break
            self.committed = stitch_words(
                self.committed, text.split(), max_overlap=self.max_overlap_words
            )
            self.boundary += self.advance
            did = True
        return did

    def _committed_update(self) -> Optional[str]:
        """Committed text the client has not seen yet, or ``None``."""
        if not self.emit_due:
            return None
        self.emit_due = False
        return " ".join(self.committed)

    def step(
        self,
        samples: np.ndarray,
        now: float,
        tx: TranscribeFn,
        newer_pending: Optional[PendingFn] = None,
    ) -> Optional[str]:
        """Advance streaming state for the current buffer.

        Finalizes any full windows (always), then (throttled) transcribes the
        live tail for a responsive partial.  When ``newer_pending`` reports
        queued audio right before the preview, the preview is skipped
        (coalesced): the next step previews the newer audio instead, carrying
        any committed-text update along.  Returns the full text to emit, or
        ``None`` if nothing should be emitted this tick.
        """
        # Window commits are required work: never throttled, never coalesced.
        # ``now`` is when the step began; the window decodes take real time,
        # so the preview decision below uses the clock after them (a slow
        # window must not eat the gap the interval promises).
        t_windows = time.monotonic()
        committed = self._finalize_full_windows(samples, tx)
        now += time.monotonic() - t_windows
        if committed:
            # The committed text covers the buffer up to the window's end, so
            # it is as fresh as a preview: it restarts the interval instead
            # of forcing a preview of the overlap right behind it (#357).
            self.emit_due = True
            self.last_emit = now

        tail_len = len(samples) - self.boundary
        if tail_len >= self.chunk:  # a full window is still waiting for a retry
            return self._committed_update()
        # Eligibility depends on the take, not on the tail: the tail shrinks
        # to the overlap after every window commit, and gating it on the
        # first-partial minimum stalled previews after each commit (#357).
        # Never hand the transcriber an empty window (issue #146).
        eligible = tail_len > 0 and self.rebased + len(samples) >= self.min
        # A window commit never forces a preview: the committed text is
        # emitted at once, and the tail waits for the (cost-stretched) interval.
        throttled = (now - self.last_emit) < self.effective_interval
        if not eligible or throttled:
            return self._committed_update()
        if newer_pending is not None and newer_pending():
            # Newer audio is already queued: this preview would be stale
            # before it finished. The next step previews the newer audio.
            self.coalesced += 1
            return None
        self.last_emit = now

        self.call_kind = "preview"
        t0 = time.monotonic()
        text = tx(samples[self.boundary :])
        if text is None:  # busy on the tail
            return self._committed_update()
        self.last_preview_cost = time.monotonic() - t0
        self.emit_due = False
        return " ".join(stitch_words(
            self.committed, text.split(), max_overlap=self.max_overlap_words
        ))

    def flush(self, samples: np.ndarray, tx: TranscribeFn) -> Optional[str]:
        """Commit all audio, or return ``None`` if bounded retries stay busy.

        Completed windows remain committed. The caller must retain the audio
        and retry commit after ``None``; only a string result is final.
        """
        for attempt in range(_FLUSH_MAX_RETRIES):
            self._finalize_full_windows(samples, tx, flushing=True)
            tail = samples[self.boundary :]
            if len(tail) == 0:
                return " ".join(self.committed)
            # Guard the tail's sign too (issue #146): the window geometry is
            # validated at construction, but the transcriber contract (a
            # nonempty window) is enforced here regardless.
            if 0 < len(tail) < self.chunk:
                self.call_kind = "flush_tail"
                text = tx(tail)
                if text is not None:
                    self.committed = stitch_words(
                        self.committed, text.split(), max_overlap=self.max_overlap_words
                    )
                    self.boundary = len(samples)
                    self.emit_due = False
                    return " ".join(self.committed)
            if attempt + 1 < _FLUSH_MAX_RETRIES:
                time.sleep(_FLUSH_BACKOFF_SECONDS)
        return None

    def reset(self) -> None:
        self.committed = []
        self.boundary = 0
        self.rebased = 0
        self.last_emit = 0.0
        self.emit_due = False
        self.last_preview_cost = 0.0
        self.coalesced = 0
        self.call_kind = "window"


# --------------------------------------------------------------------------- #
# Stream call ledger (issue #226)
# --------------------------------------------------------------------------- #
# Kept in lockstep with StreamCall / trace_*_json() in
# cpp/serve/stream_session.cpp: same kinds, results and JSON field names.

_LEDGER_KINDS = ("window", "preview", "flush_window", "flush_tail", "full_take")
# A 10-minute take at a 0.5 s preview cadence makes ~1300 calls; beyond this
# bound calls still count in the totals but are dropped from the list.
MAX_STREAM_CALLS = 20000


class _Totals:
    """Running totals over calls (engine calls only cost work; reused is free)."""

    __slots__ = ("calls", "engine_calls", "engine_samples", "engine_ms", "reused", "busy",
                 "preempted", "preempted_ms")

    def __init__(self) -> None:
        self.calls = self.engine_calls = self.engine_samples = 0
        self.engine_ms = 0.0
        self.reused = self.busy = 0
        self.preempted = 0      # previews cancelled while running
        self.preempted_ms = 0.0  # engine wall time those previews used

    def add(self, length: int, t0_ms: float, t1_ms: float, result: str) -> None:
        self.calls += 1
        if result == "ok":
            self.engine_calls += 1
            self.engine_samples += length
            self.engine_ms += t1_ms - t0_ms
        elif result == "reused":
            self.reused += 1
        elif result == "preempted":
            self.preempted += 1
            self.preempted_ms += t1_ms - t0_ms
        else:  # busy, cancelled, timed out
            self.busy += 1


class StreamTrace:
    """Per-take record of every transcribe call a streaming session made.

    Each call carries why it ran (``kind``), the original audio it covered
    (absolute take sample indices, stable across buffer trims), when it
    started and ended (ms since the take's first audio) and how it ended
    (``ok`` / ``reused`` / ``busy`` / ``timed_out`` / ``preempted``, a preview
    cancelled mid-call because required work was waiting behind it).  Overlapping windows and
    repeated previews each appear, so the ledger shows the inference work per
    recorded second rather than one batch pass.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sr = int(sample_rate)
        self.reset()

    def reset(self) -> None:
        self.t0: Optional[float] = None
        self.calls: list[tuple] = []
        self.calls_dropped = 0
        self.totals = _Totals()
        self.by_kind = {k: _Totals() for k in _LEDGER_KINDS}
        self.flush_totals = _Totals()
        self.flushing = False
        self.covered_end = 0
        self.flush_t0_ms = -1.0
        self.flush_t1_ms = -1.0
        self.flush_unfinalized = 0
        self.final_path = ""

    def mark_take_start(self) -> None:
        if self.t0 is None:
            self.t0 = time.monotonic()

    def now_ms(self) -> float:
        return 0.0 if self.t0 is None else (time.monotonic() - self.t0) * 1000.0

    def record(self, kind: str, abs_start: int, length: int, t0_ms: float,
               t1_ms: float, result: str) -> None:
        abs_start, length = int(abs_start), int(length)
        self.totals.add(length, t0_ms, t1_ms, result)
        self.by_kind[kind].add(length, t0_ms, t1_ms, result)
        if self.flushing:
            self.flush_totals.add(length, t0_ms, t1_ms, result)
        if result in ("ok", "reused"):
            self.covered_end = max(self.covered_end, abs_start + length)
        if len(self.calls) < MAX_STREAM_CALLS:
            self.calls.append((kind, abs_start, length, t0_ms, t1_ms, result))
        else:
            self.calls_dropped += 1

    def begin_flush(self, unfinalized: int) -> None:
        """Open the stop section; it covers this flush only (a busy flush is
        retried by a later commit, which opens a fresh section)."""
        self.flushing = True
        self.flush_totals = _Totals()
        self.flush_t0_ms = self.now_ms()
        self.flush_t1_ms = -1.0
        self.flush_unfinalized = int(unfinalized)
        self.final_path = ""

    def mark_empty_commit(self) -> None:
        """A commit of an empty take: no flush runs, but the stop section
        still reports the ``committed`` path and its time."""
        self.flush_totals = _Totals()
        self.flush_t0_ms = self.flush_t1_ms = self.now_ms()
        self.flush_unfinalized = 0
        self.final_path = "committed"

    def end_flush(self, ok: bool, *, full_take: bool = False) -> None:
        self.flushing = False
        self.flush_t1_ms = self.now_ms()
        if not ok:
            return
        if full_take:
            self.final_path = "full_take"
        elif self.flush_totals.engine_calls:
            self.final_path = "tail"
        elif self.flush_totals.reused:
            self.final_path = "reused"
        else:
            self.final_path = "committed"

    def _seconds(self, samples: int) -> float:
        return round(samples / self.sr, 3)

    def _totals_json(self, t: _Totals) -> dict:
        return {"calls": t.calls, "engine_calls": t.engine_calls,
                "engine_audio_s": self._seconds(t.engine_samples),
                "engine_ms": round(t.engine_ms, 3), "reused": t.reused,
                "busy": t.busy, "preempted": t.preempted,
                "preempted_ms": round(t.preempted_ms, 3)}

    def partial_json(self, audio_samples: int, chunker: Optional["ChunkStreamer"] = None) -> dict:
        out = {"v": 1, "t_ms": round(self.now_ms(), 3),
               "audio_s": self._seconds(audio_samples),
               "covered_s": self._seconds(self.covered_end),
               "totals": self._totals_json(self.totals)}
        if chunker is not None:
            out["preview"] = {
                "min_s": round(chunker.min / chunker.sr, 3),
                "interval_s": round(chunker.partial_interval, 3),
                "effective_interval_s": round(chunker.effective_interval, 3),
                "coalesced": chunker.coalesced,
            }
        return out

    def final_json(self, audio_samples: int, chunker: Optional["ChunkStreamer"] = None) -> dict:
        # full_take only exists on the Python whole-buffer path; listed only
        # when used so the chunked shape matches the native server.
        by_kind = {k: self._totals_json(t) for k, t in self.by_kind.items()
                   if k != "full_take" or t.calls}
        return {
            **self.partial_json(audio_samples, chunker),
            "by_kind": by_kind,
            "stop": {"path": self.final_path,
                     "t0_ms": round(self.flush_t0_ms, 3),
                     "t1_ms": round(self.flush_t1_ms, 3),
                     "unfinalized_s": self._seconds(self.flush_unfinalized),
                     "totals": self._totals_json(self.flush_totals)},
            "calls_dropped": self.calls_dropped,
            "calls": [{"kind": k, "start_s": self._seconds(a),
                       "end_s": self._seconds(a + n), "t0_ms": round(t0, 3),
                       "t1_ms": round(t1, 3), "result": r}
                      for k, a, n, t0, t1, r in self.calls],
        }
