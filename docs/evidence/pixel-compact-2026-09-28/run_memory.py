#!/usr/bin/env python3
"""Capture one arm's warmed memory; invoke once per arm for the B–E pair.

The snapshot follows measured short run 0. Each fixture gets one separate
warmup pass and eight measured runs. Raw timings are retained but are not
used for a latency verdict; see matched-memory-protocol.json.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADB = os.environ.get("STARLING_ADB", "adb")
SERIAL = os.environ.get("STARLING_ADB_SERIAL", "")
DEVICE = os.environ.get("STARLING_DEVICE_DIR", "/data/local/tmp/starling-issue-batch")


def adb_command(*args: str) -> list[str]:
    if not SERIAL:
        raise RuntimeError("set STARLING_ADB_SERIAL to the intended device")
    return [ADB, "-s", SERIAL, *args]


def shell(command: str) -> str:
    return subprocess.run(adb_command("shell", command), text=True,
                          capture_output=True, timeout=20, check=True).stdout


def bench_pids() -> set[str]:
    result = subprocess.run(adb_command("shell", "pidof starling-bench"), text=True,
                            capture_output=True, timeout=20)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pidof starling-bench failed: {result.stderr.strip()}")
    pids = set(result.stdout.split())
    if any(not pid.isdecimal() for pid in pids):
        raise RuntimeError(f"invalid benchmark PID list: {pids}")
    return pids


def is_owned_pid(pid: str, run_id: str) -> bool:
    try:
        environment = shell(f"cat /proc/{pid}/environ").split("\x00")
    except (OSError, subprocess.SubprocessError):
        return False
    return f"STARLING_BENCH_RUN_ID={run_id}" in environment


def number(pattern: str, text: str):
    m = re.search(pattern, text, re.MULTILINE)
    return int(m.group(1)) if m else None


def state() -> dict:
    battery = shell("dumpsys battery")
    thermal = shell("dumpsys thermalservice")
    power = shell("dumpsys power")
    display = shell("dumpsys display")
    def value(pattern: str, text: str):
        m = re.search(pattern, text, re.MULTILINE)
        return m.group(1) if m else None
    return {"ac_powered": value(r"^\s*AC powered:\s*(true|false)", battery),
            "battery_status": number(r"^\s*status:\s*(\d+)", battery),
            "battery_temp_deci_c": number(r"^\s*temperature:\s*(\d+)", battery),
            "thermal_status": number(r"^Thermal Status:\s*(\d+)", thermal),
            "wakefulness": value(r"^\s*mWakefulness=(\w+)", power),
            "screen_state": value(r"^\s*mScreenState=(\w+)", display)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "embedding"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if not SERIAL:
        ap.error("set STARLING_ADB_SERIAL to the intended device")
    output = ROOT / f"{args.tag}-matched.json"
    stdout_log = ROOT / f"{args.tag}-matched.stdout"
    stderr_log = ROOT / f"{args.tag}-matched.stderr"
    if any(path.exists() for path in (output, stdout_log, stderr_log)):
        ap.error("refusing to overwrite matched-memory artifacts")
    prior_pids = bench_pids()
    prior_state = state()
    if prior_state["screen_state"] == "ON":
        shell("input keyevent 223")
        time.sleep(2)
    start_state = state()
    run_id = uuid.uuid4().hex
    remote = (f"cd {DEVICE} && timeout 180 env LD_LIBRARY_PATH=. "
              f"STARLING_ENGINE=fast STARLING_FAST_VERBOSE=1 "
              f"STARLING_BENCH_RUN_ID={run_id} "
              f"STARLING_FAST_CACHE_DIR={DEVICE} STARLING_GGML_THREADS=6 "
              f"./starling-bench --model parakeet --gguf {DEVICE}/{args.arm}.gguf "
              f"--warmup --runs 8 {DEVICE}/short.wav {DEVICE}/medium.wav")
    pid = None
    lines = []
    row = None
    with stderr_log.open("wb") as err:
        process = subprocess.Popen(adb_command("shell", remote),
                                   stdout=subprocess.PIPE, stderr=err, text=True)
        try:
            for line in process.stdout:
                lines.append(line)
                if "short.wav run=0" in line and row is None:
                    new_pids = bench_pids() - prior_pids
                    owned = {candidate for candidate in new_pids
                             if is_owned_pid(candidate, run_id)}
                    if len(owned) != 1:
                        raise RuntimeError(f"owned benchmark PID ambiguous: {owned}")
                    pid = owned.pop()
                    meminfo = shell(f"dumpsys meminfo {pid}")
                    smaps = shell(f"cat /proc/{pid}/smaps_rollup")
                    battery = shell("dumpsys battery")
                    thermal = shell("dumpsys thermalservice")
                    row = {"arm": args.arm, "tag": args.tag, "pid": int(pid),
                           "run_id": run_id,
                           "prior_state": prior_state, "start_state": start_state,
                           "smaps_rss_kb": number(r"^Rss:\s*(\d+) kB", smaps),
                           "smaps_pss_kb": number(r"^Pss:\s*(\d+) kB", smaps),
                           "graphics_kb": number(r"^\s*GL mtrack\s+(\d+)", meminfo),
                           "total_pss_kb": number(r"^\s*TOTAL PSS:\s*(\d+)", meminfo),
                           "battery_temp_deci_c": number(r"^\s*temperature:\s*(\d+)", battery),
                           "thermal_status": number(r"^Thermal Status:\s*(\d+)", thermal)}
            rc = process.wait(timeout=15)
            if rc != 0 or row is None:
                raise RuntimeError(f"benchmark exited {rc} without a memory snapshot")
            row["end_state"] = state()
        finally:
            stdout_log.write_text("".join(lines))
            cleanup_errors = []
            if pid and process.poll() is None:
                try:
                    shell(f"kill -TERM {pid}")
                except (OSError, subprocess.SubprocessError) as error:
                    cleanup_errors.append(repr(error))
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except OSError as error:
                cleanup_errors.append(repr(error))
            if cleanup_errors:
                if sys.exc_info()[0] is None:
                    raise RuntimeError(f"benchmark cleanup failed: {cleanup_errors}")
                print(f"benchmark cleanup also failed: {cleanup_errors}", file=sys.stderr)
    output.write_text(json.dumps(row, indent=2) + "\n")
    print(output.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
