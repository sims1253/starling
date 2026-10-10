"""Paced on-device streaming workload on an Android phone (issues #226, #357).

The phone-side counterpart of ``stream_replay.py``: the same workload takes
(``stream_workload.py``) are played at real-time pace through the debug
app's production capture and on-device live-streaming path by
``StreamBaselineDeviceTest`` (one ``am instrument`` per repeat, so every
repeat is a fresh app process), and each take's session trace is pulled
back and reduced with ``stream_replay.take_metrics`` — the server harness's
own metric code — so first text, partial age, backlog, engine work per
recorded second, stop path and WER mean the same on both.

Android-specific differences, all recorded in the result:

- ``stop_to_final_ms`` runs from the user's Stop (the moment the last
  sample was captured) to the session's final, so it includes stopping the
  capture and committing the WAV; ``finish_to_final_ms`` is the engine-side
  part (``finish()`` to the final), comparable to the server's
  commit-to-final; ``stop_to_delivered_ms`` ends when the stored transcript
  reaches the main thread.
- Partial times are when the session emitted them (worker thread), before
  the main-thread post that displays them.
- ``app_cpu_per_audio_s``: user+system CPU seconds of the app process
  (capture, engine and its worker threads) per recorded second, a resource
  proxy. Energy is not measured here: the battery gauge is meaningless
  while the phone charges (``phone_energy.sh`` needs it unplugged).
- Thermal state (``PowerManager`` status, headroom, battery temperature)
  before and after each take, and ``dumpsys thermalservice`` around each
  repeat.

    python benchmarks/experiments/android_stream_replay.py run \\
        --workload build/stream-workload --takes short,medium,long \\
        --repeats 3 --min 1.0 --interval 1.0 --label pixel-1s1s \\
        --out runs/pixel-1s1s.json

Needs the debug app and its test APK installed (``connectedDebugAndroidTest``
installs both; or ``adb install`` the two APKs) with an on-device model
imported, and ``adb`` on PATH (``ANDROID_SERIAL`` picks the phone).
Results compare with ``stream_replay.py check`` like server results.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from record import sha256_file  # noqa: E402
from runner import RunnerError, _git  # noqa: E402
from stream_replay import _aggregate, _pct, take_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULT_VERSION = 1
PACKAGE = "dev.starling.mobile.debug"
RUNNER = f"{PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "dev.starling.mobile.StreamBaselineDeviceTest"
STAGING = "/data/local/tmp/starling-stream"


def adb(*args: str, timeout: float = 120.0, binary: bool = False):
    out = subprocess.run(["adb", *args], capture_output=True, timeout=timeout)
    if out.returncode != 0:
        raise RunnerError(f"adb {' '.join(args)}: {out.stderr.decode(errors='replace').strip()}")
    return out.stdout if binary else out.stdout.decode(errors="replace")


def run_as(*cmd: str, timeout: float = 120.0) -> str:
    return adb("shell", "run-as", PACKAGE, *cmd, timeout=timeout)


def thermal_snapshot() -> dict:
    """Thermal status and the CPU/SoC/battery temperatures from thermalservice."""
    out = adb("shell", "dumpsys", "thermalservice")
    snap: dict = {}
    m = re.search(r"Thermal Status: (\d+)", out)
    snap["status"] = int(m.group(1)) if m else None
    # The HAL's current readings; "Cached temperatures" can be minutes old.
    current = out.split("Current temperatures from HAL:", 1)[-1].split("Current cooling devices", 1)[0]
    for value, name in re.findall(r"mValue=([-\d.E]+), mType=-?\d+, mName=([\w-]+)", current):
        if name in ("BIG", "MID", "LITTLE", "soc_therm", "battery", "VIRTUAL-SKIN"):
            snap[name] = round(float(value), 1)
    return snap


def battery_snapshot() -> dict:
    out = adb("shell", "dumpsys", "battery")
    fields = {}
    for key in ("status", "level", "temperature", "AC powered", "USB powered"):
        m = re.search(rf"^\s*{re.escape(key)}: (\S+)", out, re.M)
        fields[key.replace(" ", "_").lower()] = m.group(1) if m else None
    return fields


def cool_down(max_battery_c: float, limit_s: float) -> tuple[float, bool]:
    """Waits until the battery is at most ``max_battery_c`` and thermal status is 0.

    Returns (seconds waited, whether both conditions were met). A charging
    phone may never get there; the run then proceeds, recorded as not met
    (interleaved configurations share that drift, and every take carries
    its own thermal state).
    """
    t0 = time.monotonic()
    while True:
        temp = battery_snapshot().get("temperature")
        if temp is not None and int(temp) / 10.0 <= max_battery_c and thermal_snapshot()["status"] == 0:
            return round(time.monotonic() - t0, 1), True
        if time.monotonic() - t0 >= limit_s:
            print(f"WARNING: not cooled to {max_battery_c} C within {limit_s:.0f} s; running anyway",
                  flush=True)
            return round(time.monotonic() - t0, 1), False
        time.sleep(15)


def push_workload(workload: Path, manifest: dict, takes: list[str]) -> None:
    adb("shell", "mkdir", "-p", STAGING)
    run_as("mkdir", "-p", "files/debug/stream-workload")
    for name in sorted(set(takes)):
        wav = workload / manifest["takes"][name]["wav"]
        adb("push", str(wav), f"{STAGING}/{name}.wav", timeout=300)
        run_as("cp", f"{STAGING}/{name}.wav", f"files/debug/stream-workload/{name}.wav")


def instrument(label: str, repeat: int, takes: list[str], args: argparse.Namespace) -> str:
    extras = ["-e", "stream", "run", "-e", "takes", ",".join(takes), "-e", "label", label,
              "-e", "repeat", str(repeat), "-e", "min", str(args.min),
              "-e", "interval", str(args.interval),
              "-e", "warmup", "true" if args.warmup else "false",
              "-e", "batch", "true" if args.batch else "false", "-e", "model", args.model]
    out = adb("shell", "am", "instrument", "-w", "-r", *extras, "-e", "class", TEST_CLASS, RUNNER,
              timeout=args.timeout)
    if "OK (1 test)" not in out:
        raise RunnerError(f"instrumentation failed:\n{out[-3000:]}")
    return out


def run_metrics(log: dict, take: dict) -> dict:
    """take_metrics over one pulled trace, plus the Android-specific measurements."""
    marks = log.get("marks", {})
    # The user's Stop, not finish(): stopping the capture is part of the wait.
    finish_at = log["commits"][0] if log["commits"] else None
    # Capture start: the earliest origin the chunks imply (what StreamTrace
    # records since review round 1; older traces used the first chunk only).
    t_start = min((t - n / 16000.0 for t, n in log["sends"]), default=log["t_start"])
    log = dict(log, t_start=t_start, commits=[marks.get("stop_pressed", finish_at)])
    m = take_metrics(log, take["reference"], marks.get("batch_text"))
    finals = [t for t, e in log["events"] if e.get("type") in ("final", "error")]
    duration = log["samples"] / 16000.0
    m.update({
        "take": marks.get("take"), "repeat": marks.get("repeat"), "state": marks.get("state"),
        "model": marks.get("model"),
        "finish_to_final_ms": None if not finals or finish_at is None
        else round((finals[-1] - finish_at) * 1000.0, 1),
        "stop_to_delivered_ms": round((marks["delivered"] - marks["stop_pressed"]) * 1000.0, 1)
        if "delivered" in marks and "stop_pressed" in marks else None,
        "prepare_ms": log.get("prepare_ms"),
        "fallback": log.get("fallback"),
        "final_status": marks.get("final_status"),
        "final_provenance": marks.get("final_provenance"),
        "batch_ms": marks.get("batch_ms"),
        "app_cpu_per_audio_s": round(marks["app_cpu_s"] / duration, 4)
        if marks.get("app_cpu_s") is not None and duration else None,
        "thermal_before": marks.get("thermal_before"),
        "thermal_after": marks.get("thermal_after"),
    })
    if m.get("failed") is None and marks.get("final_provenance") != "LIVE_STREAM":
        # A batch fallback is not a streaming success, however fast.
        m["failed"] = f"final came from {marks.get('final_provenance')}: {log.get('fallback')}"
    return m


def aggregate(runs: list[dict]) -> dict:
    """stream_replay's aggregate, plus pooled distributions with sample counts."""
    agg = _aggregate(runs)
    for take, a in agg.items():
        ok = [r for r in runs if r["take"] == take and "failed" not in r]

        def dist(values: list) -> dict:
            vals = [v for v in values if v is not None]
            return {"n": len(vals), "p50": _pct(vals, .5), "p95": _pct(vals, .95),
                    "max": max(vals) if vals else None}

        def med(values: list):
            vals = [v for v in values if v is not None]
            return statistics.median(vals) if vals else None

        a["finish_to_final_ms_median"] = med([r["finish_to_final_ms"] for r in ok])
        a["stop_to_final_ms"] = dist([r["stop_to_final_ms"] for r in ok])
        a["finish_to_final_ms"] = dist([r["finish_to_final_ms"] for r in ok])
        a["stop_to_delivered_ms"] = dist([r["stop_to_delivered_ms"] for r in ok])
        a["first_partial_wall_s"] = dist([(r.get("first_partial") or {}).get("wall_s") for r in ok])
        a["app_cpu_per_audio_s_median"] = med([r["app_cpu_per_audio_s"] for r in ok])
        a["thermal_status_max"] = max(
            [max((r.get("thermal_before") or {}).get("status", -1),
                 (r.get("thermal_after") or {}).get("status", -1)) for r in ok], default=None)
        a["battery_temp_c_max"] = max(
            [(r.get("thermal_after") or {}).get("battery_temp_c", -1) for r in ok], default=None)
        a["states"] = sorted({r.get("state") or "?" for r in ok})
    return agg


def run(args: argparse.Namespace) -> dict:
    manifest_path = args.workload / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    takes = args.takes.split(",")
    for name in takes:
        t = manifest["takes"][name]
        if sha256_file(args.workload / t["wav"]) != t["sha256"]:
            raise RunnerError(f"{name}: WAV does not match the workload manifest")
    push_workload(args.workload, manifest, takes)
    run_as("rm", "-rf", f"files/debug/stream-runs/{args.label}")
    device = {k: adb("shell", "getprop", p).strip() for k, p in (
        ("model", "ro.product.model"), ("fingerprint", "ro.build.fingerprint"),
        ("soc", "ro.soc.model"))}
    runs, repeats_env = [], []
    for repeat in range(args.repeats):
        if repeat and args.cooldown:
            time.sleep(args.cooldown)
        cooled_s, cool_met = cool_down(args.cool_to, args.cool_limit) if args.cool_to else (0.0, None)
        env_before = {"thermal": thermal_snapshot(), "battery": battery_snapshot()}
        print(f"[{args.label}] repeat {repeat}: {','.join(takes)} "
              f"({args.min} s / {args.interval} s, warmup={args.warmup}) thermal {env_before['thermal']}",
              flush=True)
        t0 = time.monotonic()
        instrument(args.label, repeat, takes, args)
        env_after = {"thermal": thermal_snapshot(), "battery": battery_snapshot()}
        repeats_env.append({"repeat": repeat, "wall_s": round(time.monotonic() - t0, 1),
                            "cooled_s": cooled_s, "cool_met": cool_met,
                            "before": env_before, "after": env_after})
        for name in takes:
            raw = run_as("cat", f"files/debug/stream-runs/{args.label}/r{repeat}-{name}.json")
            log = json.loads(raw)
            if args.keep_traces:
                args.keep_traces.mkdir(parents=True, exist_ok=True)
                (args.keep_traces / f"{args.label}-r{repeat}-{name}.json").write_text(raw)
            m = run_metrics(log, manifest["takes"][name])
            runs.append(m)
            print(json.dumps({k: m.get(k) for k in (
                "take", "state", "first_partial", "partial_age_ms", "stop_to_final_ms",
                "finish_to_final_ms", "wer_final_vs_ref", "wer_final_vs_batch",
                "app_cpu_per_audio_s")}), flush=True)
    return {
        "version": RESULT_VERSION,
        "label": args.label,
        "provenance": {
            "repo_revision": _git(REPO, "rev-parse", "HEAD"),
            "client": "android-on-device",
            "device": device,
            "model": args.model,
            "cadence": {"min_partial_seconds": args.min, "partial_interval_seconds": args.interval},
            "warmup": args.warmup, "batch": args.batch, "cool_to_c": args.cool_to,
            "workload_manifest_sha256": sha256_file(manifest_path),
            "repeats": repeats_env,
        },
        "runs": runs,
        "aggregate": aggregate(runs),
    }


def merge(label: str, parts: list[dict]) -> dict:
    """One result from per-repeat results of the same configuration (interleaved runs)."""
    keys = ("client", "device", "model", "cadence", "warmup", "batch", "cool_to_c",
            "workload_manifest_sha256")
    first = parts[0]["provenance"]
    for p in parts:
        # Each part is one repeat (its runs share one repeat number).
        if len(p["provenance"]["repeats"]) != 1:
            raise RunnerError(f"{p['label']}: merge takes single-repeat results (run --repeats 1)")
    for p in parts[1:]:
        if any(p["provenance"].get(k) != first.get(k) for k in keys):
            raise RunnerError(f"{p['label']}: different configuration than {parts[0]['label']}")
    runs, repeats = [], []
    for i, p in enumerate(parts):
        for r in p["runs"]:
            runs.append(dict(r, repeat=i))
        repeats += [dict(e, repeat=i, part=p["label"]) for e in p["provenance"]["repeats"]]
    provenance = dict(first, repeats=repeats,
                      repo_revision=sorted({p["provenance"]["repo_revision"] for p in parts}))
    return {"version": RESULT_VERSION, "label": label, "provenance": provenance,
            "runs": runs, "aggregate": aggregate(runs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workload", type=Path, required=True)
    r.add_argument("--takes", default="short,medium,long")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--min", type=float, default=1.0, help="first-partial minimum (s)")
    r.add_argument("--interval", type=float, default=1.0, help="partial interval (s)")
    r.add_argument("--model", default="parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf",
                   help="installed on-device model file (default: the app's recommended model)")
    r.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--batch", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--cooldown", type=float, default=0.0, help="seconds between repeats")
    r.add_argument("--cool-to", type=float, default=None,
                   help="before each repeat, wait until the battery is at most this many degrees C "
                        "and thermal status is 0")
    r.add_argument("--cool-limit", type=float, default=1200.0, help="longest cool-down wait (s)")
    r.add_argument("--timeout", type=float, default=3600.0)
    r.add_argument("--keep-traces", type=Path, default=None)
    r.add_argument("--label", required=True)
    r.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge", help="combine per-repeat results of one configuration")
    m.add_argument("--label", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("parts", type=Path, nargs="+")
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "merge":
        result = merge(args.label, [json.loads(p.read_text()) for p in args.parts])
    else:
        result = run(args)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["aggregate"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
