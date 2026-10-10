"""Paced streaming replay against a live server (issues #226, #357).

Streams workload takes (stream_workload.py) to ``WS /stream?trace=1`` at
microphone pace — each frame is sent when its audio would have finished
capturing — commits at the end, and records what a dictating user sees:

- first nonempty partial (wall time from the first frame, and the audio it
  reflected),
- partial age: receive time minus the capture time of the last sample the
  partial reflects (the server's ``trace.covered_s``), p50/p95/max,
- backlog: audio sent but not yet reflected when a partial arrives,
- partial count and text revisions per audio minute, stable-prefix
  violations (a partial's ``stable_words`` that the final does not keep),
- inference work per recorded second from the server's call ledger: engine
  audio seconds (overlap and repeated previews included) and engine wall
  seconds (including previews cancelled mid-call), per call kind; a
  missing measurement stays missing and fails the rules that use it,
- stop-to-final wall time, the server's stop path (``tail``/``reused``/
  ``committed``/``full_take``) and the audio the stop actually transcribed,
- WER of the final against the reference and against the batch
  (``/v1/audio/transcriptions``) result of the same server process, and
  where the final departs from that batch text: omitted, inserted and
  duplicated words, each span placed in the take and labeled ``overlap``
  when it sits at a window boundary (``locate_errors``, issue #357),
- looping partials: previews holding one phrase repeated back to back at
  least four times (a decoding loop; issue #357).

Each repeat starts a fresh server process (runner.ArmServer). The first
take of a process is ``cold`` unless ``--warmup`` sends one batch request
first. Results are one JSON file per configuration;
``check`` compares a candidate against a baseline under a frozen
thresholds file (``stream_thresholds.json``); both must come from the same
workload manifest.

    python benchmarks/experiments/stream_replay.py run \\
        --binary build/starling-serve --model models/parakeet.gguf \\
        --workload build/stream-workload --takes short,medium,long \\
        --repeats 2 --label baseline --out runs/baseline.json
    python benchmarks/experiments/stream_replay.py check \\
        --thresholds benchmarks/experiments/stream_thresholds.json \\
        --baseline runs/baseline.json --candidate runs/candidate.json

Needs the ``websockets`` package (``uv sync --extra server``); everything else
is stdlib.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.request
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from record import sha256_file  # noqa: E402
from runner import ArmServer, RunnerError, _free_port, _git, _hardware_identity  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
RESULT_VERSION = 1

# ---- text metrics -----------------------------------------------------------

_NORM = re.compile(r"[^\w' ]+")


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation except apostrophes, split on whitespace."""
    return _NORM.sub(" ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate (fraction) by word-level Levenshtein distance."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Word-level Levenshtein alignment: ``(op, ref index, hyp index)`` with
    ``op`` one of ``eq``, ``sub``, ``del`` (ref word missing from hyp) and
    ``ins`` (hyp word not in ref), in order."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        row, prev, rw = d[i], d[i - 1], ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (rw != hyp[j - 1]))
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("eq" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def _word_times(words: list[str], reference: str, utterances: list[dict]) -> list[float]:
    """Estimated audio time (s) of each word: aligned to the reference, whose
    words are spread evenly over their utterance's span; unaligned words take
    their neighbor's time."""
    ref_t: list[float] = []
    for u in utterances:
        uw = normalize(u["text"])
        for k in range(len(uw)):
            ref_t.append(u["start_s"] + (k + 0.5) / len(uw) * (u["end_s"] - u["start_s"]))
    r = normalize(reference)
    if len(ref_t) != len(r):  # a manifest whose spans do not cover the reference
        return [0.0] * len(words)
    times: list[float | None] = [None] * len(words)
    for op, i, j in align(r, words):
        if j is not None and i is not None:
            times[j] = ref_t[i]
    last = 0.0
    for k, t in enumerate(times):
        if t is None:
            times[k] = last
        else:
            last = t
    return times  # type: ignore[return-value]


def locate_errors(batch_text: str, final_text: str, utterances: list[dict] | None,
                  reference: str, calls: list[dict]) -> dict:
    """Where the streaming final departs from the batch text of the same audio.

    Aligns the two word by word and groups consecutive differences into
    spans. Each span reports the batch words the final omits, the words the
    final inserts (``duplicated`` when the inserted run repeats the words
    right before or after it), its estimated audio time (from the workload's
    utterance spans) and whether it falls into a window overlap (between the
    next committing call's start and the current one's end, from the trace's
    ``window``/``flush_window``/``flush_tail``/``redecode`` spans, +-1 s) or
    elsewhere. The trace does not say which re-decode candidate was kept, so
    every candidate counts and the label errs toward ``overlap``.
    """
    b, f = normalize(batch_text), normalize(final_text)
    times = _word_times(b, reference, utterances or [])
    committing = sorted((c for c in calls
                         if c.get("kind") in ("window", "flush_window", "flush_tail",
                                              "redecode")
                         and c.get("result", "ok") in ("ok", "reused")),
                        key=lambda c: c["start_s"])
    overlaps = [(nxt["start_s"], cur["end_s"]) for cur, nxt in zip(committing, committing[1:])
                if nxt["start_s"] < cur["end_s"]]
    spans, run = [], []
    last_b = None  # the batch word before the current run
    for op in align(b, f) + [("eq", None, None)]:
        if op[0] != "eq":
            run.append(op)
            continue
        if not run:
            last_b = op[1]
            continue
        omitted = [b[i] for o, i, _ in run if o == "del"]
        inserted_idx = [j for o, _, j in run if o == "ins"]
        inserted = [f[j] for j in inserted_idx]
        subs = sum(1 for o, _, _ in run if o == "sub")
        dup = False
        if inserted:
            j0, j1 = inserted_idx[0], inserted_idx[-1] + 1
            n = j1 - j0
            dup = (j1 - j0 == len(inserted)
                   and (f[j0 - n:j0] == f[j0:j1] or f[j1:j1 + n] == f[j0:j1]))
        bi = [i for _, i, _ in run if i is not None]
        if bi:
            t = times[bi[0]]
        else:  # pure insertion: the batch word before it
            t = times[last_b] if last_b is not None else 0.0
        near = min(overlaps, key=lambda o: max(o[0] - t, t - o[1], 0.0), default=None)
        dist = None if near is None else max(near[0] - t, t - near[1], 0.0)
        spans.append({"t_s": round(t, 2),
                      "where": "overlap" if dist is not None and dist <= 1.0 else "window",
                      "overlap_s": None if near is None else [round(near[0], 2), round(near[1], 2)],
                      "omitted": " ".join(omitted), "inserted": " ".join(inserted),
                      "duplicated": dup, "substituted": subs})
        run = []
        last_b = op[1]
    return {
        "omitted_words": sum(len(s["omitted"].split()) for s in spans),
        "inserted_words": sum(len(s["inserted"].split()) for s in spans),
        "duplicated_words": sum(len(s["inserted"].split()) for s in spans if s["duplicated"]),
        "substituted_words": sum(s["substituted"] for s in spans),
        "overlap_spans": sum(1 for s in spans if s["where"] == "overlap"),
        "spans": spans,
    }


# A preview loop: a phrase of up to _LOOP_MAX_N words repeated back to back
# at least _LOOP_MIN_REPEATS times, covering at least _LOOP_MIN_WORDS words.
_LOOP_MAX_N = 8
_LOOP_MIN_REPEATS = 4
_LOOP_MIN_WORDS = 8


def longest_loop(words: list[str]) -> int:
    """Words covered by the longest back-to-back repeat run that counts as a
    loop (see _LOOP_*), 0 when there is none."""
    best = 0
    for n in range(1, _LOOP_MAX_N + 1):
        i = 0
        while i + n <= len(words):
            reps = 1
            while words[i + reps * n:i + (reps + 1) * n] == words[i:i + n]:
                reps += 1
            if reps >= _LOOP_MIN_REPEATS and reps * n >= _LOOP_MIN_WORDS:
                best = max(best, reps * n)
                i += reps * n
            else:
                i += 1
    return best


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ---- replay -------------------------------------------------------------------

def _read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise RunnerError(f"{path}: expected mono 16 kHz PCM16")
        return w.readframes(w.getnframes())


def replay(ws_url: str, pcm: bytes, frame_ms: float, timeout_s: float) -> dict:
    """Stream ``pcm`` at real-time pace, commit, and return the raw event log."""
    from websockets.sync.client import connect

    frame_bytes = int(SAMPLE_RATE * frame_ms / 1000) * 2
    frames = [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    events: list[tuple[float, dict]] = []
    done = threading.Event()
    sends: list[tuple[float, int]] = []  # (send time, cumulative samples sent)
    rx_errors: list[BaseException] = []  # why the receiver stopped, if it failed
    with connect(ws_url, max_size=None, ping_interval=None, open_timeout=30) as ws:
        def receive():
            try:
                for raw in ws:
                    msg = json.loads(raw)
                    events.append((time.monotonic(), msg))
                    if msg.get("type") in ("final", "error") and commit_sent.is_set():
                        done.set()
            except Exception as exc:  # noqa: BLE001 - surfaced by the sender below
                rx_errors.append(exc)
                events.append((time.monotonic(),
                               {"type": "error", "message": f"receiver: {exc!r}"}))
            done.set()

        commit_sent = threading.Event()
        rx = threading.Thread(target=receive, daemon=True)
        rx.start()
        t_start = time.monotonic()
        sent = 0
        for i, frame in enumerate(frames):
            sent += len(frame) // 2
            # Frame i is captured once its last sample exists.
            due = t_start + sent / SAMPLE_RATE
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            ws.send(frame)
            sends.append((time.monotonic(), sent))
        retries = 0
        commits: list[float] = []
        while True:
            done.clear()
            commit_sent.set()
            commits.append(time.monotonic())
            since = len(events)
            ws.send(json.dumps({"type": "commit"}))
            # A receiver that died before the commit already set (and lost)
            # `done`: fail with its cause instead of waiting for the timeout.
            # One that fails only after this commit's final arrived (a server
            # closing right after it) does not fail the run.
            if not rx_errors:
                done.wait(timeout_s)
            replies = [m for _, m in events[since:] if m.get("type") in ("final", "error")
                       and not str(m.get("message", "")).startswith("receiver: ")]
            if not replies:
                if rx_errors:
                    raise RunnerError(f"receiver failed: {rx_errors[0]!r}") from rx_errors[0]
                raise RunnerError("no final within timeout")
            last = replies[-1] if replies else {}
            if last.get("type") == "error" and last.get("message") == "server busy" and retries < 20:
                retries += 1
                time.sleep(0.2)
                continue
            break
    return {"t_start": t_start, "sends": sends, "events": events,
            "commits": commits, "busy_commit_retries": retries,
            "samples": len(pcm) // 2}


def take_metrics(log: dict, reference: str, batch_text: str | None,
                 utterances: list[dict] | None = None) -> dict:
    """Reduce one replay's event log to the reported metrics; ``utterances``
    (the workload's spans) place final-vs-batch differences in the take."""
    t0 = log["t_start"]
    send_t = [t for t, _ in log["sends"]]
    send_n = [n for _, n in log["sends"]]
    duration = log["samples"] / SAMPLE_RATE
    partials = [(t, m) for t, m in log["events"] if m.get("type") == "partial"]
    finals = [(t, m) for t, m in log["events"] if m.get("type") == "final"]
    errors = [m.get("message") for _, m in log["events"] if m.get("type") == "error"]
    out: dict = {"duration_s": round(duration, 3), "errors": errors,
                 "busy_commit_retries": log["busy_commit_retries"]}
    if not finals:
        out["failed"] = "no final"
        return out
    t_final, final = finals[-1]
    trace = final.get("trace") or {}

    ages, backlogs, first = [], [], None
    revisions, prev_text, stable_violations = 0, None, 0
    looping, longest = 0, 0
    final_words = final["text"].split()
    for t, m in partials:
        tr = m.get("trace") or {}
        covered = tr.get("covered_s")
        k = bisect.bisect_right(send_t, t)
        sent_s = (send_n[k - 1] if k else 0) / SAMPLE_RATE
        if covered is not None:
            ages.append((t - (t0 + covered)) * 1000.0)
            backlogs.append(max(0.0, sent_s - covered))
        if first is None and m["text"].strip():
            first = {"wall_s": round(t - t0, 3), "audio_sent_s": round(sent_s, 3),
                     "covered_s": covered}
        if m["text"] != prev_text:
            revisions += 1
            prev_text = m["text"]
        sw = int(m.get("stable_words", 0))
        if sw and m["text"].split()[:sw] != final_words[:sw]:
            stable_violations += 1
        loop = longest_loop(normalize(m["text"]))
        if loop:
            looping += 1
            longest = max(longest, loop)

    # One partial without `covered_s` makes the run's age and backlog
    # unmeasured: statistics over the measured rest would look complete.
    if len(ages) != len(partials):
        ages, backlogs = [], []

    # Missing trace measurements stay missing (None), so a server that drops
    # `trace` or its totals fails the work rules instead of looking free.
    totals = trace.get("totals") or {}
    stop = trace.get("stop") or {}
    stop_totals = stop.get("totals") or {}
    engine_audio = totals.get("engine_audio_s")
    engine_ms = totals.get("engine_ms")
    # Preempted previews used the engine too. A server older than #357's
    # preemption never cancels one and reports no field: 0 is exact there.
    engine_wall_ms = None if engine_ms is None else engine_ms + totals.get("preempted_ms", 0.0)
    out.update({
        "first_partial": first,
        "partials": len(partials),
        "revisions_per_min": round(revisions / (duration / 60.0), 2) if duration else None,
        "stable_violations": stable_violations,
        "looping_partials": looping,
        "longest_partial_loop_words": longest,
        "partial_age_ms": {"n": len(ages), "p50": _pct(ages, .5), "p95": _pct(ages, .95),
                           "max": max(ages) if ages else None},
        "backlog_s": {"n": len(backlogs), "p50": _pct(backlogs, .5),
                      "p95": _pct(backlogs, .95), "max": max(backlogs) if backlogs else None},
        # From the FIRST commit: busy retries are part of what the user waits.
        "stop_to_final_ms": round((t_final - log["commits"][0]) * 1000.0, 1),
        "final_duration_s": final.get("duration_s"),
        "audio_complete": abs(float(final.get("duration_s", -1)) - duration) < 1e-3,
        "work": {
            "engine_calls": totals.get("engine_calls"),
            "engine_audio_per_audio_s":
                None if engine_audio is None else round(engine_audio / duration, 3),
            "engine_wall_per_audio_s":
                None if engine_wall_ms is None else round(engine_wall_ms / 1000.0 / duration, 4),
            "busy_calls": totals.get("busy"),
            "reused_calls": totals.get("reused"),
            "preempted_calls": totals.get("preempted"),
            "by_kind": trace.get("by_kind"),
        },
        "stop": {"path": stop.get("path"), "unfinalized_s": stop.get("unfinalized_s"),
                 "engine_audio_s": stop_totals.get("engine_audio_s"),
                 "engine_ms": stop_totals.get("engine_ms")},
        "wer_final_vs_ref": round(wer(reference, final["text"]), 4),
        "wer_batch_vs_ref": None if batch_text is None else round(wer(reference, batch_text), 4),
        "wer_final_vs_batch": None if batch_text is None else round(wer(batch_text, final["text"]), 4),
        "final_text": final["text"],
        "batch_text": batch_text,
    })
    if batch_text is not None:
        out["vs_batch"] = locate_errors(batch_text, final["text"], utterances, reference,
                                        trace.get("calls") or [])
    return out


def _batch(base: str, wav: Path, model_slug: str, timeout_s: float) -> tuple[str, float]:
    return _batch_bytes(base, wav.read_bytes(), model_slug, timeout_s, wav.name)


def _batch_bytes(base: str, wav: bytes, model_slug: str, timeout_s: float,
                 filename: str = "take.wav") -> tuple[str, float]:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(), b'Content-Disposition: form-data; name="model"\r\n\r\n',
        model_slug.encode(), f"\r\n--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n", wav, f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(f"{base}/v1/audio/transcriptions", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        text = json.loads(r.read())["text"]
    return text, (time.monotonic() - t) * 1000.0


def _aggregate(runs: list[dict]) -> dict:
    """Per take: medians over repeats, pooled partial ages/backlogs."""
    by_take: dict[str, list[dict]] = {}
    for r in runs:
        by_take.setdefault(r["take"], []).append(r)
    agg = {}
    for take, rs in by_take.items():
        ok = [r for r in rs if "failed" not in r]
        med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
        # A measurement missing from any run leaves the metric missing.
        med_all = lambda xs: None if None in xs else med(xs)  # noqa: E731
        max_all = lambda xs: None if None in xs or not xs else max(xs)  # noqa: E731
        firsts = [r["first_partial"]["wall_s"] for r in ok if r.get("first_partial")]
        # Every workload take is speech: a run without any nonempty partial
        # is a preview failure, so its take has no latency (a threshold on it
        # then fails as missing) instead of a median over the lucky runs.
        missing = len(ok) - len(firsts)
        agg[take] = {
            "runs": len(rs), "failed": len(rs) - len(ok),
            "audio_complete_all": all(r.get("audio_complete") for r in ok) and bool(ok),
            "first_partial_missing": missing,
            "first_partial_wall_s_median": med(firsts) if not missing else None,
            "first_partial_wall_s_max": max(firsts) if firsts and not missing else None,
            # A run without age/backlog measurements (no partial carried
            # `covered_s`) leaves these missing instead of being skipped.
            "partial_age_ms_p50_median": med_all([r["partial_age_ms"]["p50"] for r in ok]),
            "partial_age_ms_p95_median": med_all([r["partial_age_ms"]["p95"] for r in ok]),
            "partial_age_ms_max": max_all([r["partial_age_ms"]["max"] for r in ok]),
            "backlog_s_max": max_all([r["backlog_s"]["max"] for r in ok]),
            "revisions_per_min_median": med([r["revisions_per_min"] for r in ok]),
            "stable_violations_total": sum(r["stable_violations"] for r in ok),
            "looping_partials_total": sum(r.get("looping_partials", 0) for r in ok),
            "omitted_vs_batch_median": med_all([(r.get("vs_batch") or {}).get("omitted_words")
                                                for r in ok]),
            "inserted_vs_batch_median": med_all([(r.get("vs_batch") or {}).get("inserted_words")
                                                 for r in ok]),
            "duplicated_vs_batch_median": med_all([(r.get("vs_batch") or {}).get(
                "duplicated_words") for r in ok]),
            "engine_audio_per_audio_s_median":
                med_all([r["work"]["engine_audio_per_audio_s"] for r in ok]),
            "engine_wall_per_audio_s_median":
                med_all([r["work"]["engine_wall_per_audio_s"] for r in ok]),
            "stop_to_final_ms_median": med([r["stop_to_final_ms"] for r in ok]),
            "stop_to_final_ms_max": max([r["stop_to_final_ms"] for r in ok], default=None),
            "stop_engine_audio_s_max": max_all([r["stop"]["engine_audio_s"] for r in ok]),
            "stop_paths": sorted({r["stop"]["path"] or "missing" for r in ok}),
            "wer_final_vs_ref_median": med([r["wer_final_vs_ref"] for r in ok]),
            "wer_batch_vs_ref_median": med([r["wer_batch_vs_ref"] for r in ok
                                            if r["wer_batch_vs_ref"] is not None]),
            "wer_final_vs_batch_median": med([r["wer_final_vs_batch"] for r in ok
                                              if r["wer_final_vs_batch"] is not None]),
        }
    return agg


def run(args: argparse.Namespace) -> dict:
    manifest_path = args.workload / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    takes = args.takes.split(",")
    for name in takes:
        t = manifest["takes"][name]
        if sha256_file(args.workload / t["wav"]) != t["sha256"]:
            raise RunnerError(f"{name}: WAV does not match the workload manifest")
    arm = {"binary": str(args.binary), "model": str(args.model),
           "model_slug": args.model_slug, "args": args.server_arg or []}
    query = "&".join(["trace=1", *(args.query or [])])
    runs = []
    for repeat in range(args.repeats):
        port = args.port or _free_port()
        log_path = args.out.with_suffix(f".r{repeat}.log")
        server = ArmServer(arm, port, log_path)
        try:
            load_s = server.wait_healthy(args.startup_timeout)
            served = 0
            if args.warmup:
                _batch(server.base, args.workload / manifest["takes"][takes[0]]["wav"],
                       args.model_slug, args.timeout)
                served += 1
            logs = {}
            for name in takes:
                t = manifest["takes"][name]
                state = "cold" if served == 0 else "warm"
                pcm = _read_pcm(args.workload / t["wav"])
                print(f"[{args.label}] repeat {repeat} {name} ({t['duration_s']:.0f} s, {state})",
                      flush=True)
                logs[name] = (state, replay(f"ws://127.0.0.1:{port}/stream?{query}", pcm,
                                            args.frame_ms, args.timeout))
                served += 1
            for name in takes:
                t = manifest["takes"][name]
                batch_text, batch_ms = _batch(server.base, args.workload / t["wav"],
                                              args.model_slug, args.timeout)
                state, log = logs[name]
                m = take_metrics(log, t["reference"], batch_text, t.get("utterances"))
                m.update({"take": name, "repeat": repeat, "state": state,
                          "batch_ms": round(batch_ms, 1), "server_load_s": round(load_s, 2)})
                runs.append(m)
                print(json.dumps({k: m.get(k) for k in (
                    "take", "first_partial", "partial_age_ms", "stop_to_final_ms",
                    "wer_final_vs_ref", "wer_final_vs_batch", "looping_partials")}), flush=True)
                vb = m.get("vs_batch") or {}
                print(f"  vs batch: omitted {vb.get('omitted_words')} inserted "
                      f"{vb.get('inserted_words')} duplicated {vb.get('duplicated_words')}",
                      flush=True)
                for span in vb.get("spans", []):
                    if span["omitted"] or span["inserted"]:
                        print("   ", json.dumps(span), flush=True)
        finally:
            server.stop()
    return {
        "version": RESULT_VERSION,
        "label": args.label,
        "provenance": {
            "repo_revision": _git(REPO, "rev-parse", "HEAD"),
            "binary": str(args.binary), "binary_sha256": sha256_file(args.binary),
            "model": str(args.model), "model_sha256": sha256_file(args.model),
            "model_slug": args.model_slug, "server_args": args.server_arg or [],
            "query": query, "frame_ms": args.frame_ms, "warmup": args.warmup,
            "runtime": _hardware_identity(),
            "workload_manifest_sha256": sha256_file(manifest_path),
            "env": {k: v for k, v in os.environ.items() if k.startswith("STARLING_")},
        },
        "runs": runs,
        "aggregate": _aggregate(runs),
    }


# ---- threshold check ----------------------------------------------------------

def check(thresholds: dict, baseline: dict, candidate: dict) -> list[dict]:
    """Evaluate a candidate against a baseline under frozen thresholds.

    Each rule names an aggregate metric and a take and gives either an
    absolute bound (``max``/``min``) or a bound relative to the baseline
    (``max_vs_baseline_ratio``, ``max_vs_baseline_delta``). Boolean rules
    (``equals``) must hold exactly.
    """
    out = []
    # Only runs over the same recordings are comparable: a regenerated
    # workload can change the audio and the references.
    wb = (baseline.get("provenance") or {}).get("workload_manifest_sha256")
    wc = (candidate.get("provenance") or {}).get("workload_manifest_sha256")
    if wb is None or wb != wc:
        out.append({"take": "*", "metric": "workload_manifest_sha256", "baseline": wb,
                    "candidate": wc, "pass": False,
                    "why": "missing" if wb is None else "different workloads",
                    "rule": "baseline and candidate replay the same recordings"})
    for rule in thresholds["rules"]:
        take, metric = rule["take"], rule["metric"]
        c = candidate["aggregate"].get(take, {}).get(metric)
        b = baseline["aggregate"].get(take, {}).get(metric)
        ok, why = True, []
        if c is None:
            ok, why = False, ["missing in candidate"]
        else:
            if "equals" in rule and c != rule["equals"]:
                ok = False
                why.append(f"!= {rule['equals']}")
            if "max" in rule and c > rule["max"]:
                ok = False
                why.append(f"> {rule['max']}")
            if "min" in rule and c < rule["min"]:
                ok = False
                why.append(f"< {rule['min']}")
            if "max_vs_baseline_ratio" in rule:
                if b is None:
                    ok, why = False, why + ["baseline missing"]
                elif c > b * rule["max_vs_baseline_ratio"]:
                    ok = False
                    why.append(f"> {rule['max_vs_baseline_ratio']} x baseline {b}")
            if "max_vs_baseline_delta" in rule:
                if b is None:
                    ok, why = False, why + ["baseline missing"]
                elif c > b + rule["max_vs_baseline_delta"]:
                    ok = False
                    why.append(f"> baseline {b} + {rule['max_vs_baseline_delta']}")
        out.append({"take": take, "metric": metric, "baseline": b, "candidate": c,
                    "pass": ok, "why": "; ".join(why), "rule": rule.get("why", "")})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--binary", type=Path, required=True)
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--model-slug", default="parakeet")
    r.add_argument("--workload", type=Path, required=True)
    r.add_argument("--takes", default="short,medium,long")
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--frame-ms", type=float, default=100.0)
    r.add_argument("--server-arg", action="append",
                   help="extra starling-serve flag (repeat; use --server-arg=--flag)")
    r.add_argument("--query", action="append", help="extra /stream query k=v (repeat)")
    r.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--port", type=int, default=0)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--startup-timeout", type=float, default=300.0)
    r.add_argument("--label", required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--thresholds", type=Path, required=True)
    c.add_argument("--baseline", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result = run(args)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["aggregate"], indent=1))
        return 0
    results = check(json.loads(args.thresholds.read_text()),
                    json.loads(args.baseline.read_text()),
                    json.loads(args.candidate.read_text()))
    for row in results:
        print(f"{'PASS' if row['pass'] else 'FAIL'} {row['take']:>6} {row['metric']}: "
              f"baseline={row['baseline']} candidate={row['candidate']} {row['why']}")
    return 0 if all(row["pass"] for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
