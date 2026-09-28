#!/usr/bin/env python3
"""Capture matched memory and warm timing in a three-process B–E–B bracket.

The snapshot follows short run 0. The predeclared timing analysis excludes
short runs 0–2 because that snapshot may delay them, then uses short runs 3–7
and all eight medium runs. Every file also gets one unreported warmup pass.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADB = "/home/m0hawk/android-sdk/platform-tools/adb"
SERIAL = "192.168.178.59:34113"
DEVICE = "/data/local/tmp/starling-issue-batch"


def shell(command: str) -> str:
    return subprocess.run([ADB, "-s", SERIAL, "shell", command], text=True,
                          capture_output=True, timeout=20, check=True).stdout


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
    output = ROOT / f"{args.tag}-matched.json"
    stdout_log = ROOT / f"{args.tag}-matched.stdout"
    stderr_log = ROOT / f"{args.tag}-matched.stderr"
    if any(path.exists() for path in (output, stdout_log, stderr_log)):
        ap.error("refusing to overwrite matched-memory artifacts")
    prior_state = state()
    if prior_state["screen_state"] == "ON":
        shell("input keyevent 223")
        time.sleep(2)
    start_state = state()
    remote = (f"cd {DEVICE} && timeout 180 env LD_LIBRARY_PATH=. "
              f"STARLING_ENGINE=fast STARLING_FAST_VERBOSE=1 "
              f"STARLING_FAST_CACHE_DIR={DEVICE} STARLING_GGML_THREADS=6 "
              f"./starling-bench --model parakeet --gguf {DEVICE}/{args.arm}.gguf "
              f"--warmup --runs 8 {DEVICE}/short.wav {DEVICE}/medium.wav")
    pid = None
    lines = []
    row = None
    with stderr_log.open("wb") as err:
        process = subprocess.Popen([ADB, "-s", SERIAL, "shell", remote],
                                   stdout=subprocess.PIPE, stderr=err, text=True)
        try:
            for line in process.stdout:
                lines.append(line)
                if "short.wav run=0" in line and row is None:
                    pid = shell("pidof starling-bench").strip()
                    if not pid.isdecimal():
                        raise RuntimeError(f"ambiguous bench PID {pid!r}")
                    meminfo = shell(f"dumpsys meminfo {pid}")
                    smaps = shell(f"cat /proc/{pid}/smaps_rollup")
                    battery = shell("dumpsys battery")
                    thermal = shell("dumpsys thermalservice")
                    row = {"arm": args.arm, "tag": args.tag, "pid": int(pid),
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
            if pid and process.poll() is None:
                shell(f"kill -TERM {pid}")
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    stdout_log.write_text("".join(lines))
    output.write_text(json.dumps(row, indent=2) + "\n")
    print(output.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
