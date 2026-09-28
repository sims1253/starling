#!/usr/bin/env python3
"""Rerun the sealed Pixel ABBA protocol with owned-process cleanup.

The historical partial attempt used run_controlled_v1.py; keep that file intact.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"
ADB = os.environ.get("STARLING_ADB", "adb")
SERIAL = os.environ.get("STARLING_ADB_SERIAL", "")
DEVICE = os.environ.get("STARLING_DEVICE_DIR", "/data/local/tmp/starling-issue-batch")
CASES = Path(os.environ.get("STARLING_S1_CASES", str(
    Path(__file__).resolve().parents[4] / "tests/fixtures/s1_quant_spans.json")))
PORT = 18181


def adb(*args, timeout=30):
    if not SERIAL:
        raise RuntimeError("set STARLING_ADB_SERIAL to the intended device")
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True,
                          text=True, check=True, timeout=timeout)


def owned_pid(remote_name, run_id):
    """Find only a process carrying this launch's marker, never a name match alone."""
    try:
        listing = adb("shell", f"pidof {remote_name}").stdout.split()
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1:
            return None
        raise
    matches = []
    for pid in listing:
        if not pid.isdecimal():
            raise RuntimeError(f"invalid {remote_name} PID: {pid!r}")
        try:
            environ = adb("shell", f"cat /proc/{pid}/environ").stdout.split("\x00")
        except subprocess.CalledProcessError:
            continue  # Process may have exited after pidof.
        if f"STARLING_BENCH_RUN_ID={run_id}" in environ:
            matches.append(pid)
    if len(matches) > 1:
        raise RuntimeError(f"multiple owned {remote_name} processes: {matches}")
    return matches[0] if matches else None


def terminate_owned(remote_name, run_id):
    pid = owned_pid(remote_name, run_id)
    if pid is not None:
        adb("shell", f"kill -TERM {pid}")
    return pid


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
            reason = bad_state(s)
            if reason:
                self.failure = reason
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


def monitor(recorder, stop, process, remote_name, run_id):
    i = 0
    while not stop.wait(5):
        try:
            recorder.take("active", full=(i % 3 == 0))
        except Exception as exc:
            recorder.failure = f"sampler: {exc}"
        if recorder.failure:
            if process.poll() is None:
                try:
                    if terminate_owned(remote_name, run_id) is None:
                        process.terminate()
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
                s["battery_temp_deci_c"] is not None and s["battery_temp_deci_c"] < 370:
            break
        if attempt < 6:
            time.sleep(30)
    (block_dir / "cooldown.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows[-1]["thermal_status"] is not None and \
        rows[-1]["thermal_status"] <= 1 and \
        rows[-1]["battery_temp_deci_c"] is not None and \
        rows[-1]["battery_temp_deci_c"] < 370


def run_fast(arm, block_dir):
    run_id = uuid.uuid4().hex
    remote = (f"cd {shlex.quote(DEVICE)} && env LD_LIBRARY_PATH=. STARLING_ENGINE=fast "
              f"STARLING_BENCH_RUN_ID={run_id} "
              f"STARLING_FAST_VERBOSE=1 STARLING_FAST_CACHE_DIR={DEVICE} "
              f"STARLING_GGML_THREADS=6 ./starling-bench --model parakeet "
              f"--gguf {DEVICE}/{arm}.gguf --warmup --runs 72 "
              f"{DEVICE}/short.wav {DEVICE}/medium.wav")
    out = (block_dir / "bench.stdout").open("wb")
    err = (block_dir / "bench.stderr").open("wb")
    launch_ns = time.monotonic_ns()
    try:
        process = subprocess.Popen([ADB, "-s", SERIAL, "shell", remote],
                                   stdout=out, stderr=err)
    except Exception:
        out.close()
        err.close()
        raise
    return process, out, err, None, launch_ns, "starling-bench", run_id


def run_s1(arm, block_dir):
    run_id = uuid.uuid4().hex
    adb("forward", f"tcp:{PORT}", "tcp:8181")
    remote = (f"cd {shlex.quote(DEVICE)} && env LD_LIBRARY_PATH=. STARLING_ENGINE=ggml "
              f"STARLING_BENCH_RUN_ID={run_id} "
              f"STARLING_GGML_DEVICE=CPU STARLING_GGML_THREADS=6 "
              f"./starling-serve --model s1 --gguf {DEVICE}/s1-{arm}.gguf "
              "--warmup --host 127.0.0.1 --port 8181")
    out = (block_dir / "serve.stdout").open("wb")
    err = (block_dir / "serve.stderr").open("wb")
    launch_ns = time.monotonic_ns()
    try:
        process = subprocess.Popen([ADB, "-s", SERIAL, "shell", remote],
                                   stdout=out, stderr=err)
    except Exception:
        out.close()
        err.close()
        try:
            adb("forward", "--remove", f"tcp:{PORT}")
        except Exception:
            pass
        raise
    return process, out, err, PORT, launch_ns, "starling-serve", run_id


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


def one_block(family, arm, index, protocol_sha, output_root):
    block_dir = output_root / f"{family}-{index + 1}-{arm}"
    block_dir.mkdir(exist_ok=False)
    cooled = wait_cool(block_dir)
    recorder = Recorder(block_dir / "battery-state.jsonl")
    process = None
    remote_name = run_id = None
    out = err = None
    port = None
    result = {"family": family, "arm": arm, "index": index,
              "protocol_sha256": protocol_sha, "runner_sha256": sha(__file__),
              "runtime_device_serial": SERIAL, "runtime_device_dir": DEVICE,
              "cooled_before_block": cooled}
    try:
        result["runtime_hardware_serial"] = adb("shell", "getprop ro.serialno").stdout.strip()
        result["runtime_model"] = adb("shell", "getprop ro.product.model").stdout.strip()
        result["runtime_build_id"] = adb("shell", "getprop ro.build.id").stdout.strip()
        if result["runtime_model"] != "Pixel 10 Pro":
            raise RuntimeError(f"expected Pixel 10 Pro, got {result['runtime_model']!r}")
        recorder.idle("pre_idle")
        start_sample = recorder.take("active_boundary_before_launch", full=False)
        if family == "parakeet_fast":
            process, out, err, port, launch_ns, remote_name, run_id = run_fast(arm, block_dir)
        else:
            process, out, err, port, launch_ns, remote_name, run_id = run_s1(arm, block_dir)
        result["run_id"] = run_id
        stop = threading.Event()
        watcher = threading.Thread(target=monitor, args=(recorder, stop, process,
                                remote_name, run_id),
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
                pid = terminate_owned(remote_name, run_id)
                if pid is None:
                    raise RuntimeError("owned S1 server PID disappeared before shutdown")
                process.wait(timeout=20)
            exit_ns = time.monotonic_ns()
        finally:
            stop.set()
            watcher.join()
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
                if terminate_owned(remote_name, run_id) is None:
                    process.terminate()
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
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for this rerun; never overwrite sealed evidence")
    args = parser.parse_args()
    if not SERIAL:
        parser.error("set STARLING_ADB_SERIAL to the intended device")
    if not DEVICE.startswith("/") or not re.fullmatch(r"[A-Za-z0-9_./-]+", DEVICE):
        parser.error("STARLING_DEVICE_DIR must be an absolute shell-safe path")
    output_root = args.output_dir.resolve()
    if output_root == ROOT or ROOT in output_root.parents:
        parser.error("write reruns outside the sealed evidence directory")
    protocol_sha = sha(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text())
    family_spec = next(row for row in protocol["sequence"] if row["family"] == args.family)
    if args.family == "parakeet_fast" and family_spec["timed_runs"] != [72, 72]:
        raise RuntimeError("fast run count differs from protocol")
    if args.family == "s1_cpu":
        if not CASES.is_file():
            parser.error("S1 cases missing; set STARLING_S1_CASES to the pinned fixture")
        if sha(CASES) != family_spec["cases_sha256"]:
            raise RuntimeError("S1 case file hash differs from protocol")
    output_root.mkdir(parents=True, exist_ok=True)
    for index, arm in enumerate(family_spec["arms"]):
        print(f"START {args.family} block {index + 1}/4 {arm} UTC {dt.datetime.now(dt.timezone.utc).isoformat()}",
              flush=True)
        result = one_block(args.family, arm, index, protocol_sha, output_root)
        print(f"DONE {args.family} block {index + 1}/4 duration {result['process_duration_ms']/1000:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
