"""Fixed-window overlapping-chunk streaming for long-form live dictation.

Full windows overlap so boundary words can be aligned with :func:`stitch_words`
(decoded words only: the engines expose no word timestamps). A committed window
whose text is implausible for its audio -- too sparse for its voiced audio, or
a decoding loop -- is decoded again from an earlier start (issue #357). Each
transcription call is bounded by the configured window size, including retries
when the server is busy.
"""

from __future__ import annotations

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


# Overlap alignment scores (issue #357; kStitch* in the C++ port).
_STITCH_MATCH = 2
_STITCH_MISMATCH = -1
_STITCH_GAP = -1
# The empty alignment scores 0; two matched words at the very edges score 4,
# three matches around four unmatched committed words score 2 (rejected).
_STITCH_MIN_SCORE = 3


def stitch_words(
    committed: list[str],
    new: list[str],
    *,
    max_overlap: int = 24,
    max_head: Optional[int] = None,
) -> list[str]:
    """Append ``new`` to ``committed``, deduping the overlapping boundary words.

    The two texts come from windows that share some audio, so the words the
    windows agree on sit at the END of ``committed`` and the START of
    ``new``. This aligns a suffix of ``committed``'s last ``max_overlap`` words
    with a prefix of ``new``'s first ``max_head`` words (default
    ``max_overlap``; a re-decoded window that starts earlier shares more
    audio, and its words before the committed tail align as gaps) (match
    +2, mismatch and gap -1; words left out on the committed side before the suffix and on
    the new side after the prefix are free). Every committed word after the
    aligned suffix's start and every new word before the prefix's end costs a
    gap, so a common phrase ("of the") far from the boundary cannot win the
    alignment and drop the words between (issue #357: one such match dropped
    30 words). The cut is the middle matched word of the alignment, where
    both windows hold the most context: ``committed`` up to and including
    it, then ``new`` after it, so each overlap word is taken from exactly one
    side. Without an alignment scoring at least ``_STITCH_MIN_SCORE`` (no
    shared words: a pause in the overlap, or a window that dropped them) the
    two are concatenated.

    Kept in lockstep with ``stitch_words`` in cpp/serve/stream_session.cpp,
    including the tie-breaking.
    """
    if not committed:
        return list(new)
    if not new:
        return list(committed)

    tail = committed[-max_overlap:]
    head = new[: max_overlap if max_head is None else max_head]
    # Empty keys (words that normalize to "", e.g. pure punctuation) never
    # match: runs of empty keys would align unrelated boundary words and drop
    # them (issue #118 defense in depth). The sentinels are unique per side
    # and position and cannot collide with a real key -- ``_norm`` strips
    # ``\x00`` along with other non-word chars.
    a = [_norm(w) or f"\x00tail-{i}" for i, w in enumerate(tail)]
    b = [_norm(w) or f"\x00head-{i}" for i, w in enumerate(head)]
    n, m = len(a), len(b)
    # d[i][j]: best score of aligning tail[i0:i] (any i0 <= i) with head[:j].
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        d[0][j] = d[0][j - 1] + _STITCH_GAP
    for i in range(1, n + 1):
        prev, row, ai = d[i - 1], d[i], a[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (_STITCH_MATCH if ai == b[j - 1] else _STITCH_MISMATCH)
            row[j] = max(diag, prev[j] + _STITCH_GAP, row[j - 1] + _STITCH_GAP)
    # The suffix runs to the end of the tail; the prefix may end anywhere
    # (the smallest end column wins a tie).
    best_j, best = 0, 0
    for j in range(1, m + 1):
        if d[n][j] > best:
            best, best_j = d[n][j], j
    if best < _STITCH_MIN_SCORE:
        return list(committed) + list(new)
    pairs = []
    i, j = n, best_j
    while i > 0 and j > 0:
        same = a[i - 1] == b[j - 1]
        if d[i][j] == d[i - 1][j - 1] + (_STITCH_MATCH if same else _STITCH_MISMATCH):
            if same:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + _STITCH_GAP:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    ci, cj = pairs[(len(pairs) - 1) // 2]
    keep = len(committed) - len(tail) + ci + 1
    return list(committed[:keep]) + list(new[cj + 1:])


# ---- window plausibility (issue #357) ---------------------------------------
# Parakeet (and the HF reference model) sometimes stops emitting partway
# through a window, or emits nothing for a window full of speech, and a
# window shifted by a second transcribes the same audio fine. A transducer can
# also loop on a phrase. Neither shows in the text alone, so a committed
# window is checked against the audio it covers: too few words for its voiced
# audio, or more words than its duration allows, and the window is decoded
# again from slightly earlier starts (REDECODE_SHIFTS) and the plausible
# result kept. Kept in lockstep with the C++ port.

# More words than this per second of audio (plus _MAX_WORDS_SLACK) is a
# decoding loop, never speech: fast dictation stays below ~5 words/s.
MAX_WORDS_PER_SECOND = 7.0
_MAX_WORDS_SLACK = 4
# Fewer words per voiced second than this, or than _RELATIVE_MIN_RATE of the
# take's median over its recent committed windows (_RATE_HISTORY), over at
# least _MIN_VOICED_SECONDS of voiced audio, marks a window that dropped
# speech. Healthy windows of the replay workload hold 1.5-4.3 words per voiced
# second; the dropped ones 0-1.5, against 3.4 for their speaker.
MIN_WORDS_PER_VOICED_SECOND = 1.25
_RELATIVE_MIN_RATE = 0.6
_RATE_HISTORY = 16
_MIN_VOICED_SECONDS = 2.0
# Earlier starts tried for an implausible window, in order, as fractions of
# the window overlap (1.5, 0.75 and 2.25 s at the default 3 s): below one
# overlap, a window moved back still overlaps the next one. The buffer keeps
# the largest of them before the boundary.
REDECODE_SHIFTS = (0.5, 0.25, 0.75)
# A sparse window is replaced only by a candidate this much denser.
_REDECODE_MIN_GAIN = 1.25
# Voiced-audio measure: 20 ms frames whose energy is 15 dB over the span's
# quiet floor (its 10th-percentile frame, at most -45 dBFS, so a span of
# continuous speech is not its own floor) and over -60 dBFS.
_VAD_FRAMES_PER_SECOND = 50
_VAD_FLOOR_PERCENTILE = 0.1
_VAD_FLOOR_MAX_DB = -45.0
_VAD_MARGIN_DB = 15.0
_VAD_MIN_DB = -60.0


def max_plausible_words(seconds: float) -> int:
    """Most words ``seconds`` of audio can hold; more is a decoding loop."""
    return int(seconds * MAX_WORDS_PER_SECOND) + _MAX_WORDS_SLACK


def voiced_seconds(samples: np.ndarray, sample_rate: int) -> float:
    """Seconds of ``samples`` loud enough to be speech (see _VAD_*)."""
    frame = sample_rate // _VAD_FRAMES_PER_SECOND
    count = len(samples) // frame if frame > 0 else 0
    if count == 0:
        return 0.0
    x = np.asarray(samples[: count * frame], dtype=np.float64).reshape(count, frame)
    db = 10.0 * np.log10(np.mean(x * x, axis=1) + 1e-10)
    floor = min(float(np.sort(db)[int(_VAD_FLOOR_PERCENTILE * (count - 1))]), _VAD_FLOOR_MAX_DB)
    threshold = max(floor + _VAD_MARGIN_DB, _VAD_MIN_DB)
    return float(np.count_nonzero(db > threshold)) * frame / sample_rate


def _repeat_run(keys: list[str], i: int, n: int) -> int:
    """How many times keys[i:i+n] repeats back to back from i."""
    reps = 1
    while keys[i + reps * n: i + (reps + 1) * n] == keys[i: i + n]:
        reps += 1
    return reps


def _longest_repeat(keys: list[str], max_repeats: int, max_n: int) -> tuple[int, int, int]:
    """(start, phrase length, repeats) of the back-to-back run of one phrase
    (1..``max_n`` words, more than ``max_repeats`` copies) covering the most
    words; the earliest, then the shortest phrase, wins a tie. (0, 0, 0)
    when there is none."""
    best = (0, 0, 0)
    for i in range(len(keys)):
        for n in range(1, max_n + 1):
            if i + n * (max_repeats + 1) > len(keys):
                break
            reps = _repeat_run(keys, i, n)
            if reps > max_repeats and reps * n > best[1] * best[2]:
                best = (i, n, reps)
    return best


def suppress_loops(words: list[str], seconds: float, *, max_repeats: int = 2,
                   max_n: int = 8) -> list[str]:
    """Words of a decode over ``seconds`` of audio with a decoding loop
    removed. Text within max_plausible_words() is returned unchanged, so
    real repeated speech is never touched. Longer text has its longest
    back-to-back run of one phrase cut to ``max_repeats`` copies, one run at
    a time, only until it fits; what still does not fit is cut at the bound
    (more words than the audio can hold are not speech)."""
    bound = max_plausible_words(seconds)
    out = list(words)
    keys = [_norm(w) for w in out]
    while len(out) > bound:
        i, n, reps = _longest_repeat(keys, max_repeats, max_n)
        if not n:
            break
        cut = slice(i + max_repeats * n, i + reps * n)
        del out[cut], keys[cut]
    return out[:max(bound, 0)]


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
        # Audio kept before the boundary for re-decoding an implausible
        # window from an earlier start (issue #357).
        self.lookback = int(max(REDECODE_SHIFTS) * self.overlap)
        # New words searched when stitching: a re-decoded window starts up to
        # the lookback earlier and shares that much more audio (issue #357).
        self.max_head_words = 2 * self.max_overlap_words

        self.committed: list[str] = []
        # Leading committed words stitching never touches (stable_words()).
        self.frozen = 0
        # Words per voiced second of the latest committed windows.
        self.rates: list[float] = []
        self.boundary = 0          # sample index; audio before this is finalized
        self.rebased = 0           # samples dropped before index 0 (rebase())
        self.last_emit = 0.0
        self.emit_due = False      # a commit changed the text, not yet emitted
        self.last_preview_cost = 0.0  # seconds, latest successful preview
        self.coalesced = 0         # previews skipped for newer queued audio
        # Why the transcribe call in flight was made (issue #226): "window",
        # "preview", "flush_window" or "flush_tail". Set before every call;
        # the session's call ledger reads it (lockstep with call_kind() in
        # cpp/serve/stream_session.hpp). "redecode" is a committed window
        # or flush tail decoded again from an earlier start (issue #357).
        self.call_kind = "window"
        # Buffer index where that call's audio starts.
        self.call_start = 0
        self.redecodes = 0         # windows replaced by a re-decode

    def set_preview_policy(self, min_seconds: float, interval_seconds: float) -> None:
        """Replace the preview cadence (per connection); ``ValueError`` if invalid."""
        error = preview_policy_error(sample_rate=self.sr, min_seconds=min_seconds,
                                     interval_seconds=interval_seconds)
        if error is not None:
            raise ValueError(error)
        self.min = int(min_seconds * self.sr)
        self.partial_interval = float(interval_seconds)

    @property
    def retain_from(self) -> int:
        """First buffer sample the streamer may still read; the session may
        trim everything before it."""
        return max(0, self.boundary - self.lookback)

    def stable_words(self) -> int:
        """Leading words of every later text that no window, tail or flush
        can change (lockstep with ChunkStreamer::stable_words in C++)."""
        return self.frozen

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
    def _stitched(self, words: list[str]) -> list[str]:
        """``committed`` with ``words`` stitched onto its unfrozen tail."""
        tail = stitch_words(self.committed[self.frozen:], words,
                            max_overlap=self.max_overlap_words,
                            max_head=self.max_head_words)
        return self.committed[: self.frozen] + tail

    def _commit(self, words: list[str]) -> None:
        self.committed = self._stitched(words)
        self.frozen = max(self.frozen, len(self.committed) - self.max_overlap_words)

    def _tx(self, samples: np.ndarray, start: int, end: int, kind: str,
            tx: TranscribeFn) -> Optional[str]:
        self.call_kind = kind
        self.call_start = start
        return tx(samples[start:end])

    def _min_rate(self) -> float:
        """Words per voiced second below which a window dropped speech."""
        if not self.rates:
            return MIN_WORDS_PER_VOICED_SECOND
        median = sorted(self.rates)[(len(self.rates) - 1) // 2]
        return max(MIN_WORDS_PER_VOICED_SECOND, _RELATIVE_MIN_RATE * median)

    def _verdict(self, words: list[str], seconds: float, voiced: float) -> int:
        """0 plausible, 1 sparse (dropped speech), 2 a decoding loop."""
        if len(words) > max_plausible_words(seconds):
            return 2
        if voiced >= _MIN_VOICED_SECONDS and len(words) < self._min_rate() * voiced:
            return 1
        return 0

    def _decode_committed(self, samples: np.ndarray, start: int, end: int, kind: str,
                          tx: TranscribeFn) -> Optional[tuple[list[str], int]]:
        """Words of samples[start:end] for the committed text and where the
        audio they come from ends, or ``None`` when the engine is busy. An implausible result (too sparse for its
        voiced audio, or a loop) is decoded again from the earlier starts in
        REDECODE_SHIFTS, until one is plausible: a full window moves
        back whole (the next window's overlap still covers its end), or
        ends earlier where the buffer holds no audio before it (the take's
        first window); the flush tail grows backwards up to one window
        (nothing else covers its end). A candidate replaces the current one
        when it is plausible and the current one loops, or when it is denser
        by _REDECODE_MIN_GAIN. A busy re-decode keeps the best so far if it is
        plausible, and otherwise leaves the window pending (``None``): the
        audio stays in the buffer for a retry instead of committing text
        known to be wrong."""
        text = self._tx(samples, start, end, kind, tx)
        if text is None:
            return None
        words = text.split()
        voiced = voiced_seconds(samples[start:end], self.sr)
        verdict = self._verdict(words, (end - start) / self.sr, voiced)
        best, best_verdict, best_span, best_voiced = words, verdict, (start, end), voiced
        tried = {(start, end)}
        full = end - start == self.chunk
        for shift in REDECODE_SHIFTS if verdict else ():
            back = int(shift * self.overlap)
            if full:
                a, z = (start - back, end - back) if start >= back else (start, end - back)
            else:
                a, z = max(start - back, end - self.chunk, 0), end
            if a >= z or (a, z) in tried or (not full and a >= start):
                continue
            tried.add((a, z))
            text = self._tx(samples, a, z, "redecode", tx)
            if text is None:
                if best_verdict:
                    return None
                break
            cand = text.split()
            v_voiced = voiced_seconds(samples[a:z], self.sr)
            v = self._verdict(cand, (z - a) / self.sr, v_voiced)
            if v != 2 and best_verdict == 2:
                better = True
            elif v != 2:
                better = (len(cand) * max(best_voiced, 1e-9)
                          >= _REDECODE_MIN_GAIN * len(best) * max(v_voiced, 1e-9))
            else:
                better = False
            if better:
                best, best_verdict, best_span, best_voiced = cand, v, (a, z), v_voiced
            if best_verdict == 0:
                break
        if best is not words:
            self.redecodes += 1
        if best_verdict == 2:
            best = suppress_loops(best, (best_span[1] - best_span[0]) / self.sr)
        elif best_voiced >= _MIN_VOICED_SECONDS:
            self.rates = (self.rates + [len(best) / best_voiced])[-_RATE_HISTORY:]
        return best, best_span[1]

    def _finalize_full_windows(
        self, samples: np.ndarray, tx: TranscribeFn, *, flushing: bool = False
    ) -> bool:
        """Finalize every complete window at the current boundary. Returns True
        if at least one window was committed."""
        did = False
        while (len(samples) - self.boundary) >= self.chunk:
            end = self.boundary + self.chunk
            got = self._decode_committed(
                samples, self.boundary, end, "flush_window" if flushing else "window", tx)
            if got is None:  # busy/cancelled -> stop; boundary unchanged for retry
                break
            words, used_end = got
            self._commit(words)
            # The next window overlaps the audio the committed text came
            # from by the full overlap, also when a re-decode ended earlier.
            self.boundary += self.advance - (end - used_end)
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

        t0 = time.monotonic()
        text = self._tx(samples, self.boundary, len(samples), "preview", tx)
        if text is None:  # busy on the tail
            return self._committed_update()
        self.last_preview_cost = time.monotonic() - t0
        self.emit_due = False
        # A preview is shown as decoded (no re-decode: the next one replaces
        # it), but never with a decoding loop in it (issue #357).
        return " ".join(self._stitched(suppress_loops(text.split(), tail_len / self.sr)))

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
                got = self._decode_committed(samples, self.boundary, len(samples),
                                             "flush_tail", tx)
                if got is not None:
                    self._commit(got[0])
                    self.boundary = len(samples)
                    self.emit_due = False
                    return " ".join(self.committed)
            if attempt + 1 < _FLUSH_MAX_RETRIES:
                time.sleep(_FLUSH_BACKOFF_SECONDS)
        return None

    def reset(self) -> None:
        self.committed = []
        self.frozen = 0
        self.rates = []
        self.boundary = 0
        self.rebased = 0
        self.last_emit = 0.0
        self.emit_due = False
        self.last_preview_cost = 0.0
        self.coalesced = 0
        self.call_kind = "window"
        self.call_start = 0
        self.redecodes = 0


# --------------------------------------------------------------------------- #
# Stream call ledger (issue #226)
# --------------------------------------------------------------------------- #
# Kept in lockstep with StreamCall / trace_*_json() in
# cpp/serve/stream_session.cpp: same kinds, results and JSON field names.

_LEDGER_KINDS = ("window", "preview", "flush_window", "flush_tail", "redecode", "full_take")
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
