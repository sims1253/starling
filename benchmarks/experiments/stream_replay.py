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
  seconds, per call kind,
- stop-to-final wall time, the server's stop path (``tail``/``reused``/
  ``committed``/``full_take``) and the audio the stop actually transcribed,
- WER of the final against the reference and against the batch
  (``/v1/audio/transcriptions``) result of the same server process.

Each repeat starts a fresh server process (runner.ArmServer). The first
take of a process is ``cold`` unless ``--warmup`` sends one batch request
first. Results are one JSON file per configuration;
``check`` compares a candidate against a baseline under a frozen
thresholds file (``stream_thresholds.json``).

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
    with connect(ws_url, max_size=None, ping_interval=None, open_timeout=30) as ws:
        def receive():
            try:
                for raw in ws:
                    msg = json.loads(raw)
                    events.append((time.monotonic(), msg))
                    if msg.get("type") in ("final", "error") and commit_sent.is_set():
                        done.set()
            except Exception:  # noqa: BLE001 - connection closed
                pass
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
            ws.send(json.dumps({"type": "commit"}))
            if not done.wait(timeout_s):
                raise RunnerError("no final within timeout")
            last = events[-1][1] if events else {}
            if last.get("type") == "error" and last.get("message") == "server busy" and retries < 20:
                retries += 1
                time.sleep(0.2)
                continue
            break
    return {"t_start": t_start, "sends": sends, "events": events,
            "commits": commits, "busy_commit_retries": retries,
            "samples": len(pcm) // 2}


def take_metrics(log: dict, reference: str, batch_text: str | None) -> dict:
    """Reduce one replay's event log to the reported metrics."""
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

    totals = trace.get("totals", {})
    stop = trace.get("stop", {})
    out.update({
        "first_partial": first,
        "partials": len(partials),
        "revisions_per_min": round(revisions / (duration / 60.0), 2) if duration else None,
        "stable_violations": stable_violations,
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
            "engine_audio_per_audio_s": round(totals.get("engine_audio_s", 0) / duration, 3),
            "engine_wall_per_audio_s": round(totals.get("engine_ms", 0) / 1000.0 / duration, 4),
            "busy_calls": totals.get("busy"),
            "reused_calls": totals.get("reused"),
            "by_kind": trace.get("by_kind"),
        },
        "stop": {"path": stop.get("path"), "unfinalized_s": stop.get("unfinalized_s"),
                 "engine_audio_s": (stop.get("totals") or {}).get("engine_audio_s"),
                 "engine_ms": (stop.get("totals") or {}).get("engine_ms")},
        "wer_final_vs_ref": round(wer(reference, final["text"]), 4),
        "wer_batch_vs_ref": None if batch_text is None else round(wer(reference, batch_text), 4),
        "wer_final_vs_batch": None if batch_text is None else round(wer(batch_text, final["text"]), 4),
        "final_text": final["text"],
    })
    return out


def _batch(base: str, wav: Path, model_slug: str, timeout_s: float) -> tuple[str, float]:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(), b'Content-Disposition: form-data; name="model"\r\n\r\n',
        model_slug.encode(), f"\r\n--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n", wav.read_bytes(), f"\r\n--{boundary}--\r\n".encode(),
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
        firsts = [r["first_partial"]["wall_s"] for r in ok if r.get("first_partial")]
        agg[take] = {
            "runs": len(rs), "failed": len(rs) - len(ok),
            "audio_complete_all": all(r.get("audio_complete") for r in ok) and bool(ok),
            "first_partial_wall_s_median": med(firsts),
            "first_partial_wall_s_max": max(firsts) if firsts else None,
            "partial_age_ms_p50_median": med([r["partial_age_ms"]["p50"] for r in ok
                                              if r["partial_age_ms"]["p50"] is not None]),
            "partial_age_ms_p95_median": med([r["partial_age_ms"]["p95"] for r in ok
                                              if r["partial_age_ms"]["p95"] is not None]),
            "partial_age_ms_max": max([r["partial_age_ms"]["max"] for r in ok
                                       if r["partial_age_ms"]["max"] is not None], default=None),
            "backlog_s_max": max([r["backlog_s"]["max"] for r in ok
                                  if r["backlog_s"]["max"] is not None], default=None),
            "revisions_per_min_median": med([r["revisions_per_min"] for r in ok]),
            "stable_violations_total": sum(r["stable_violations"] for r in ok),
            "engine_audio_per_audio_s_median": med([r["work"]["engine_audio_per_audio_s"] for r in ok]),
            "engine_wall_per_audio_s_median": med([r["work"]["engine_wall_per_audio_s"] for r in ok]),
            "stop_to_final_ms_median": med([r["stop_to_final_ms"] for r in ok]),
            "stop_to_final_ms_max": max([r["stop_to_final_ms"] for r in ok], default=None),
            "stop_engine_audio_s_max": max([r["stop"]["engine_audio_s"] or 0 for r in ok], default=None),
            "stop_paths": sorted({r["stop"]["path"] for r in ok}),
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
                m = take_metrics(log, t["reference"], batch_text)
                m.update({"take": name, "repeat": repeat, "state": state,
                          "batch_ms": round(batch_ms, 1), "server_load_s": round(load_s, 2)})
                runs.append(m)
                print(json.dumps({k: m.get(k) for k in (
                    "take", "first_partial", "partial_age_ms", "stop_to_final_ms",
                    "wer_final_vs_ref", "wer_final_vs_batch")}), flush=True)
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
