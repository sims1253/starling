#!/usr/bin/env python3
"""Run the eight protected-span cases through the Pixel's native S1 HTTP path."""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "benchmarks/s1"))
from quant_spans import audit_output, check_cases  # noqa: E402

ADB = os.environ.get("STARLING_ADB", "adb")
SERIAL = os.environ.get("STARLING_ADB_SERIAL", "")
DEVICE = "/data/local/tmp/starling-issue-batch"
PORT = 18181


def adb_command(*args: str) -> list[str]:
    if not SERIAL:
        raise RuntimeError("set STARLING_ADB_SERIAL to the intended device")
    return [ADB, "-s", SERIAL, *args]


def adb(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(adb_command(*args), text=True,
                          capture_output=True, timeout=30, check=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def post(transcript: str, req_id: str) -> tuple[str, float]:
    data = json.dumps({"transcript": transcript}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/normalize", data=data,
                                 headers={"Content-Type": "application/json",
                                          "X-Request-Id": req_id})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        obj = json.load(response)
    ms = (time.perf_counter() - start) * 1000
    if "error" in obj or not isinstance(obj.get("text"), str):
        raise RuntimeError(f"normalize failed: {obj}")
    return obj["text"], ms


def snapshot(pid: str) -> dict:
    battery = adb("shell", "dumpsys battery").stdout
    thermal = adb("shell", "dumpsys thermalservice").stdout
    power = adb("shell", "dumpsys power").stdout
    display = adb("shell", "dumpsys display").stdout
    meminfo = adb("shell", f"dumpsys meminfo {pid}").stdout
    smaps = adb("shell", f"cat /proc/{pid}/smaps_rollup").stdout
    def number(pat: str, text: str):
        found = re.search(pat, text, re.MULTILINE)
        return int(found.group(1)) if found else None
    def value(pat: str, text: str):
        found = re.search(pat, text, re.MULTILINE)
        return found.group(1) if found else None
    return {"charge_uah": number(r"^\s*Charge counter:\s*(-?\d+)", battery),
            "ac_powered": value(r"^\s*AC powered:\s*(true|false)", battery),
            "battery_status": number(r"^\s*status:\s*(\d+)", battery),
            "voltage_mv": number(r"^\s*voltage:\s*(\d+)", battery),
            "battery_percent": number(r"^\s*level:\s*(\d+)", battery),
            "battery_temp_deci_c": number(r"^\s*temperature:\s*(\d+)", battery),
            "thermal_status": number(r"^Thermal Status:\s*(\d+)", thermal),
            "wakefulness": value(r"^\s*mWakefulness=(\w+)", power),
            "screen_state": value(r"^\s*mScreenState=(\w+)", display),
            "smaps_rss_kb": number(r"^Rss:\s*(\d+) kB", smaps),
            "smaps_pss_kb": number(r"^Pss:\s*(\d+) kB", smaps),
            "graphics_kb": number(r"^\s*GL mtrack\s+(\d+)", meminfo),
            "total_pss_kb": number(r"^\s*TOTAL PSS:\s*(\d+)", meminfo)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["bf16", "q4-k-m"])
    ap.add_argument("--artifact-dir", required=True, type=Path,
                    help="directory containing both S1 GGUF arms")
    ap.add_argument("--binary", required=True, type=Path,
                    help="local starling-serve binary used to stage the device")
    ap.add_argument("--engine-source-commit", required=True,
                    help="source commit used to build the staged binary")
    args = ap.parse_args()
    if not SERIAL:
        ap.error("set STARLING_ADB_SERIAL to the intended device")
    cases_file = PROJECT / "tests/fixtures/s1_quant_spans.json"
    cases = json.loads(cases_file.read_text())
    check_cases(cases)
    model = args.artifact_dir / f"s1-{args.arm}.gguf"
    source = args.artifact_dir / "s1-bf16.gguf"
    binary = args.binary
    for path in (model, source, binary):
        if not path.is_file():
            ap.error(f"missing input: {path}")
    out = ROOT / f"s1-{args.arm}-pixel.json"
    log = ROOT / f"s1-{args.arm}-serve.log"
    if out.exists():
        ap.error("refusing to overwrite existing record")
    remote = (f"cd {DEVICE} && env LD_LIBRARY_PATH=. STARLING_ENGINE=ggml "
              f"STARLING_GGML_DEVICE=CPU STARLING_GGML_THREADS=6 "
              f"./starling-serve --model s1 --gguf {DEVICE}/s1-{args.arm}.gguf "
              "--warmup --host 127.0.0.1 --port 8181")
    adb("forward", f"tcp:{PORT}", "tcp:8181")
    pid = None
    with log.open("wb") as stream:
        try:
            process = subprocess.Popen(adb_command("shell", remote),
                                       stdout=stream, stderr=subprocess.STDOUT)
        except Exception:
            try:
                adb("forward", "--remove", f"tcp:{PORT}")
            except (OSError, subprocess.SubprocessError):
                pass
            raise
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see {log}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                        health = json.load(response)
                    if health.get("phase") == "ready":
                        break
                except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("server did not become ready")
            pid = adb("shell", "pidof starling-serve").stdout.strip()
            if not pid.isdecimal():
                raise RuntimeError(f"ambiguous server PID: {pid!r}")
            before = snapshot(pid)
            rows = []
            for case in cases:
                samples = [post(case["transcript"], f"{case['id']}-{repeat}")
                           for repeat in range(3)]
                if len({text for text, _ in samples}) != 1:
                    raise RuntimeError(f"nondeterministic text for {case['id']}")
                text = samples[0][0]
                times = [ms for _, ms in samples]
                rows.append({"id": case["id"], "output": text,
                             "spans_present": audit_output(case, text),
                             "times_ms": times, "median_ms": statistics.median(times)})
            after = snapshot(pid)
            record = {"schema": "s1-quant-spans-v1", "model_sha256": digest(model),
                      "model_bytes": model.stat().st_size, "source_sha256": digest(source),
                      "engine_sha256": digest(binary), "cases_sha256": digest(cases_file),
                      "device": "CPU/Pixel10Pro", "results": rows,
                      "memory_before": before, "memory_after": after,
                      "engine_source_commit": args.engine_source_commit,
                      "server_log_sha256": None}
            # Close the server first; hash its final log below.
        finally:
            cleanup_errors = []
            if pid:
                try:
                    adb("shell", f"kill -TERM {pid}")
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
            try:
                adb("forward", "--remove", f"tcp:{PORT}")
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_errors.append(repr(error))
            if cleanup_errors:
                if sys.exc_info()[0] is None:
                    raise RuntimeError(f"server cleanup failed: {cleanup_errors}")
                print(f"server cleanup also failed: {cleanup_errors}", file=sys.stderr)
    record["server_log_sha256"] = digest(log)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(out)
    print({r["id"]: r["spans_present"] for r in record["results"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
