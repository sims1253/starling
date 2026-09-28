#!/usr/bin/env python3
"""Bounded CPU ASR-only and optional S1 co-residency observation."""

import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "native-copy-controlled"))
from run_copy import state, remote_server_pids  # noqa: E402

ADB = "/home/m0hawk/android-sdk/platform-tools/adb"
SERIAL = "192.168.178.59:34113"
DEVICE = "/data/local/tmp/starling-issue-batch"
PROTOCOL = ROOT / "protocol.json"
RECORD = ROOT / "record.json"
SAMPLES = ROOT / "samples.jsonl"


def adb(*args, timeout=30):
    return subprocess.run([ADB, "-s", SERIAL, *args], check=True,
                          capture_output=True, text=True, timeout=timeout).stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_field(raw, key):
    match = re.search(rf"^{re.escape(key)}:\s*(\d+)", raw, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing {key} in proc snapshot")
    return int(match.group(1))


def one_sample(stage, named_pids):
    s = state()
    mem_raw = adb("shell", "cat /proc/meminfo")
    mem = {key: read_field(mem_raw, key) for key in
           ("MemTotal", "MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree")}
    processes = {}
    for name, pid in named_pids.items():
        smaps = adb("shell", f"cat /proc/{pid}/smaps_rollup")
        status = adb("shell", f"cat /proc/{pid}/status")
        processes[name] = {"pid": pid, "pss_kb": read_field(smaps, "Pss"),
                           "rss_kb": read_field(smaps, "Rss"),
                           "private_clean_kb": read_field(smaps, "Private_Clean"),
                           "private_dirty_kb": read_field(smaps, "Private_Dirty"),
                           "vmhwm_kb": read_field(status, "VmHWM"),
                           "vmrss_kb": read_field(status, "VmRSS")}
        if processes[name]["pss_kb"] <= 0:
            raise RuntimeError(f"zero PSS for {name}")
    row = {"stage": stage, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "phone": s, "meminfo_kb": mem, "processes": processes}
    with SAMPLES.open("a") as file:
        file.write(json.dumps(row) + "\n")
    return row


def sample_stage(stage, named_pids):
    rows = []
    for index in range(3):
        if index:
            time.sleep(5)
        rows.append(one_sample(stage, named_pids))
    print(f"{stage}: MemAvailable {[r['meminfo_kb']['MemAvailable'] for r in rows]}", flush=True)
    return rows


def launch(name, slug, gguf, device_port, host_port, before, deadline):
    out = ROOT / f"{name}.log"
    adb("forward", f"tcp:{host_port}", f"tcp:{device_port}")
    remote = (f"cd {DEVICE} && env LD_LIBRARY_PATH=. STARLING_ENGINE=ggml "
              "STARLING_GGML_DEVICE=CPU STARLING_GGML_THREADS=6 "
              f"./starling-serve --model {slug} --gguf {DEVICE}/{gguf} "
              f"--warmup --host 127.0.0.1 --port {device_port}")
    start = time.monotonic_ns()
    log = out.open("wb")
    process = subprocess.Popen([ADB, "-s", SERIAL, "shell", remote],
                               stdout=log, stderr=subprocess.STDOUT)
    try:
        ready_deadline = min(deadline, time.monotonic() + 180)
        while time.monotonic() < ready_deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{name} server exited early: {process.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{host_port}/health", timeout=2) as response:
                    health = json.load(response)
                if health.get("phase") == "ready" and health.get("model") == slug:
                    break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError(f"{name} readiness exceeded 180 s or total bound")
        new = set(remote_server_pids()) - before
        if len(new) != 1:
            raise RuntimeError(f"{name} new PID ambiguous: {new}")
        pid = new.pop()
        log.flush()
        content = out.read_text(errors="replace")
        if (f"model={slug}, backend=CPU" not in content or
                "warmup complete" not in content or "warmup error:" in content):
            raise RuntimeError(f"{name} actual CPU backend/warmup log missing")
        return {"name": name, "pid": pid, "process": process, "log": log,
                "host_port": host_port, "ready_ms": (time.monotonic_ns()-start)/1e6,
                "health": health}
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        raise


def main():
    if RECORD.exists() or SAMPLES.exists():
        raise RuntimeError("existing data; refuse overwrite")
    cfg = json.loads(PROTOCOL.read_text())
    record = {"protocol_sha256": sha(PROTOCOL), "completed": False, "servers": {}}
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    deadline = time.monotonic() + 9 * 60
    owned = []
    forwards = []
    try:
        if remote_server_pids():
            raise RuntimeError("existing Starling server before baseline")
        sample_stage("idle", {})
        asr = launch("asr", "parakeet", "baseline.gguf", 8184, 18184,
                     set(), deadline)
        owned.append(asr)
        forwards.append(18184)
        record["servers"]["asr"] = {k: asr[k] for k in ("pid", "ready_ms", "health")}
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
        sample_stage("asr_only", {"asr": asr["pid"]})
        s1 = launch("s1", "s1", "s1-q4-k-m.gguf", 8185, 18185,
                    {asr["pid"]}, deadline)
        owned.append(s1)
        forwards.append(18185)
        record["servers"]["s1"] = {k: s1[k] for k in ("pid", "ready_ms", "health")}
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
        sample_stage("asr_plus_optional_s1", {"asr": asr["pid"], "s1": s1["pid"]})
        if time.monotonic() > deadline:
            raise TimeoutError("nine-minute total bound reached")
    except BaseException as error:
        record["failure"] = repr(error)
        raise
    finally:
        # The first process may have started before readiness exposed its PID.
        pids = remote_server_pids()
        for pid in pids:
            try:
                adb("shell", f"kill -TERM {pid}")
            except Exception as error:
                record.setdefault("cleanup_errors", []).append(repr(error))
        for item in owned:
            try:
                item["process"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait()
            item["log"].close()
        for port in forwards + [18184, 18185]:
            try:
                adb("forward", "--remove", f"tcp:{port}")
            except Exception:
                pass
        left = remote_server_pids()
        record["remote_pids_after_cleanup"] = left
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
        if left:
            raise RuntimeError(f"owned server survived cleanup: {left}")
        sample_stage("recovery", {})
        record["completed"] = not record.get("failure") and not record.get("cleanup_errors")
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print("RESIDENCY SNAPSHOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
