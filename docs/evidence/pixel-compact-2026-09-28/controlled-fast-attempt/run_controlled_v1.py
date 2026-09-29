#!/usr/bin/env python3
"""Run the sealed Pixel ABBA protocol with raw battery and workload records."""

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"
ADB = "/home/m0hawk/android-sdk/platform-tools/adb"
SERIAL = "192.168.178.59:34113"
DEVICE = "/data/local/tmp/starling-issue-batch"
CASES = Path("/home/m0hawk/.t3/worktrees/starling/s1-quant-recipe/tests/fixtures/s1_quant_spans.json")
PORT = 18181


def adb(*args, timeout=30):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True,
                          text=True, check=True, timeout=timeout)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(pattern, raw):
    match = re.search(pattern, raw, re.MULTILINE)
    return match.group(1) if match else None


def sample_state(full=False):
    start_ns = time.monotonic_ns()
    raw = adb("shell", "dumpsys battery").stdout
    state = {
        "charge_uah": extract(r"^\s*Charge counter:\s*(-?\d+)", raw),
        "voltage_mv": extract(r"^\s*voltage:\s*(\d+)", raw),
        "battery_status": extract(r"^\s*status:\s*(\d+)", raw),
        "battery_percent": extract(r"^\s*level:\s*(\d+)", raw),
        "battery_temp_deci_c": extract(r"^\s*temperature:\s*(\d+)", raw),
        "ac_powered": extract(r"^\s*AC powered:\s*(true|false)", raw),
        "usb_powered": extract(r"^\s*USB powered:\s*(true|false)", raw),
        "wireless_powered": extract(r"^\s*Wireless powered:\s*(true|false)", raw),
    }
    for key in ("charge_uah", "voltage_mv", "battery_status", "battery_percent",
                "battery_temp_deci_c"):
        state[key] = int(state[key]) if state[key] is not None else None
    if full:
        thermal = adb("shell", "dumpsys thermalservice").stdout
        display = adb("shell", "dumpsys display").stdout
        power = adb("shell", "dumpsys power").stdout
        state.update({
            "thermal_status": extract(r"^Thermal Status:\s*(\d+)", thermal),
            "screen_state": extract(r"^\s*mScreenState=(\w+)", display),
            "wakefulness": extract(r"^\s*mWakefulness=(\w+)", power),
        })
        if state["thermal_status"] is not None:
            state["thermal_status"] = int(state["thermal_status"])
    end_ns = time.monotonic_ns()
    state.update({"mono_start_ns": start_ns, "mono_end_ns": end_ns,
                  "mono_mid_ns": (start_ns + end_ns) // 2,
                  "utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    return state


def bad_state(s):
    if s["charge_uah"] is None or s["voltage_mv"] is None:
        return "missing charge/voltage gauge"
    if s["battery_status"] != 3 or any(s[k] != "false" for k in
            ("ac_powered", "usb_powered", "wireless_powered")):
        return "phone is not discharging unplugged"
    if s["battery_temp_deci_c"] is not None and s["battery_temp_deci_c"] >= 420:
        return "battery temperature >=42C"
    if s.get("thermal_status") is not None and s["thermal_status"] >= 2:
        return "thermal status >=2"
    if s.get("screen_state") is not None and s["screen_state"] != "OFF":
        return "screen turned on"
    return None


class Recorder:
    def __init__(self, path):
        self.path = path
        self.file = path.open("w")
        self.lock = threading.Lock()
        self.samples = []
        self.failure = None

    def take(self, phase, full=False):
        with self.lock:
            s = sample_state(full)
            s["phase"] = phase
            self.samples.append(s)
            self.file.write(json.dumps(s) + "\n")
            self.file.flush()
            if bad_state(s):
                self.failure = bad_state(s)
            return s

    def idle(self, phase, seconds=45):
        origin = time.monotonic()
        rows = []
        for i in range(seconds // 5 + 1):
            delay = origin + 5 * i - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            rows.append(self.take(phase, full=(i % 3 == 0 or i == seconds // 5)))
            if self.failure:
                raise RuntimeError(self.failure)
        return rows

    def close(self):
        self.file.close()


def monitor(recorder, stop, process, remote_name):
    i = 0
    while not stop.wait(5):
        try:
            recorder.take("active", full=(i % 3 == 0))
        except Exception as exc:
            recorder.failure = f"sampler: {exc}"
        if recorder.failure:
            if process.poll() is None:
                try:
                    adb("shell", f"pkill -TERM {remote_name}")
                except Exception:
                    process.terminate()
            return
        i += 1


def wait_cool(block_dir):
    rows = []
    for attempt in range(7):
        s = sample_state(full=True)
        rows.append(s)
        reason = bad_state(s)
        if reason:
            raise RuntimeError(f"pre-block state: {reason}")
        if s["thermal_status"] is not None and s["thermal_status"] <= 1 and \
                s["battery_temp_deci_c"] < 370:
            break
        if attempt < 6:
            time.sleep(30)
    (block_dir / "cooldown.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows[-1]["thermal_status"] is not None and \
        rows[-1]["thermal_status"] <= 1 and rows[-1]["battery_temp_deci_c"] < 370


def run_fast(arm, block_dir):
    remote = (f"cd {DEVICE} && env LD_LIBRARY_PATH=. STARLING_ENGINE=fast "
              f"STARLING_FAST_VERBOSE=1 STARLING_FAST_CACHE_DIR={DEVICE} "
              f"STARLING_GGML_THREADS=6 ./starling-bench --model parakeet "
              f"--gguf {DEVICE}/{arm}.gguf --warmup --runs 72 "
              f"{DEVICE}/short.wav {DEVICE}/medium.wav")
    out = (block_dir / "bench.stdout").open("wb")
    err = (block_dir / "bench.stderr").open("wb")
    launch_ns = time.monotonic_ns()
    process = subprocess.Popen([ADB, "-s", SERIAL, "shell", remote],
                               stdout=out, stderr=err)
    return process, out, err, None, launch_ns


def run_s1(arm, block_dir):
    adb("forward", f"tcp:{PORT}", "tcp:8181")
    remote = (f"cd {DEVICE} && env LD_LIBRARY_PATH=. STARLING_ENGINE=ggml "
              f"STARLING_GGML_DEVICE=CPU STARLING_GGML_THREADS=6 "
              f"./starling-serve --model s1 --gguf {DEVICE}/s1-{arm}.gguf "
              "--warmup --host 127.0.0.1 --port 8181")
    out = (block_dir / "serve.stdout").open("wb")
    err = (block_dir / "serve.stderr").open("wb")
    launch_ns = time.monotonic_ns()
    process = subprocess.Popen([ADB, "-s", SERIAL, "shell", remote],
                               stdout=out, stderr=err)
    return process, out, err, PORT, launch_ns


def s1_requests(block_dir, process, launch_ns):
    deadline = time.monotonic() + 180
    ready_ms = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"S1 server exited before ready: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
                health = json.load(resp)
            if health.get("phase") == "ready" and health.get("model") == "s1":
                ready_ms = (time.monotonic_ns() - launch_ns) / 1e6
                break
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    if ready_ms is None:
        raise RuntimeError("S1 server did not become ready")
    cases = json.loads(CASES.read_text())
    rows = []
    for case in cases:
        payload = json.dumps({"transcript": case["transcript"]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/normalize", data=payload,
                                     headers={"Content-Type": "application/json"})
        start = time.monotonic_ns()
        with urllib.request.urlopen(req, timeout=180) as resp:
            obj = json.load(resp)
        end = time.monotonic_ns()
        if not isinstance(obj.get("text"), str):
            raise RuntimeError(f"{case['id']}: missing S1 text: {obj}")
        rows.append({"id": case["id"], "output": obj["text"],
                     "latency_ms": (end - start) / 1e6,
                     "mono_start_ns": start, "mono_end_ns": end})
        (block_dir / "responses.json").write_text(json.dumps(rows, indent=2,
                                                        ensure_ascii=False) + "\n")
        print(f"  {case['id']}: {rows[-1]['latency_ms']:.0f} ms", flush=True)
    return ready_ms, rows


def one_block(family, arm, index, protocol_sha):
    block_dir = ROOT / f"{family}-{index + 1}-{arm}"
    block_dir.mkdir(exist_ok=False)
    cooled = wait_cool(block_dir)
    recorder = Recorder(block_dir / "battery-state.jsonl")
    process = None
    out = err = None
    port = None
    result = {"family": family, "arm": arm, "index": index,
              "protocol_sha256": protocol_sha, "cooled_before_block": cooled}
    try:
        recorder.idle("pre_idle")
        start_sample = recorder.take("active_boundary_before_launch", full=False)
        if family == "parakeet_fast":
            process, out, err, port, launch_ns = run_fast(arm, block_dir)
        else:
            process, out, err, port, launch_ns = run_s1(arm, block_dir)
        stop = threading.Event()
        watcher = threading.Thread(target=monitor, args=(recorder, stop, process,
                                "starling-bench" if family == "parakeet_fast" else "starling-serve"),
                                daemon=True)
        watcher.start()
        try:
            if family == "parakeet_fast":
                rc = process.wait(timeout=600)
                if rc != 0:
                    raise RuntimeError(f"bench exit {rc}")
            else:
                ready_ms, rows = s1_requests(block_dir, process, launch_ns)
                result["launch_to_ready_ms"] = ready_ms
                result["responses"] = rows
                pid = adb("shell", "pidof starling-serve").stdout.strip()
                if not pid.isdecimal():
                    raise RuntimeError(f"ambiguous S1 PID: {pid!r}")
                adb("shell", f"kill -TERM {pid}")
                process.wait(timeout=20)
            exit_ns = time.monotonic_ns()
        finally:
            stop.set()
            watcher.join(timeout=10)
        if recorder.failure:
            raise RuntimeError(recorder.failure)
        end_sample = recorder.take("active_boundary_after_exit", full=False)
        result.update({"active_boundary_start": start_sample,
                       "process_launch_ns": launch_ns, "process_exit_ns": exit_ns,
                       "active_boundary_end": end_sample,
                       "process_duration_ms": (exit_ns - launch_ns) / 1e6})
        recorder.idle("post_idle")
        if recorder.failure:
            raise RuntimeError(recorder.failure)
        result["completed"] = True
    except Exception as exc:
        result["completed"] = False
        result["error"] = str(exc)
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                adb("shell", "pkill -TERM starling-bench" if family == "parakeet_fast"
                    else "pkill -TERM starling-serve")
                process.wait(timeout=15)
            except Exception:
                process.kill()
                process.wait()
        if port is not None:
            try:
                adb("forward", "--remove", f"tcp:{port}")
            except Exception:
                pass
        if out:
            out.close()
        if err:
            err.close()
        recorder.close()
        (block_dir / "result.json").write_text(json.dumps(result, indent=2,
                                                         ensure_ascii=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["parakeet_fast", "s1_cpu"])
    args = parser.parse_args()
    protocol_sha = sha(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text())
    family_spec = next(row for row in protocol["sequence"] if row["family"] == args.family)
    if args.family == "parakeet_fast" and family_spec["timed_runs"] != [72, 72]:
        raise RuntimeError("fast run count differs from protocol")
    if args.family == "s1_cpu" and sha(CASES) != family_spec["cases_sha256"]:
        raise RuntimeError("S1 case file hash differs from protocol")
    for index, arm in enumerate(family_spec["arms"]):
        print(f"START {args.family} block {index + 1}/4 {arm} UTC {dt.datetime.now(dt.timezone.utc).isoformat()}",
              flush=True)
        result = one_block(args.family, arm, index, protocol_sha)
        print(f"DONE {args.family} block {index + 1}/4 duration {result['process_duration_ms']/1000:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
