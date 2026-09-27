"""Run the preregistered real-model Granite contention pilot.

This is a small diagnosis for #174/#178, not the full serving benchmark. It
requires a built native CPU server, Granite GGUF, the public test WAV fixtures,
and the project's ``server`` extra (websockets). No transcript is written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("pilot_spec.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def trace_events(log: Path, offset: int = 0) -> list[dict]:
    events = []
    with log.open("r", errors="replace") as src:
        src.seek(offset)
        for line in src:
            if not line.startswith("[trace] "):
                continue
            try:
                events.append(json.loads(line[len("[trace] "):]))
            except json.JSONDecodeError:
                pass
    return events


def wait_for_trace(log: Path, request_id: str, event: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(e.get("req") == request_id and e.get("ev") == event for e in trace_events(log)):
            return
        time.sleep(0.05)
    raise TimeoutError(f"missing trace {event} for {request_id}")


def wait_healthy(base_url: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load: {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200 and json.load(response).get("loaded"):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError("server did not become healthy")


def http_upload(base_url: str, audio: bytes, request_id: str, timeout_s: float) -> dict:
    boundary = f"starling-pilot-{request_id}"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngranite\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{base_url}/v1/audio/transcriptions", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "x-request-id": request_id},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
            text = payload.get("text")
            return {"status": response.status, "wall_ms": (time.perf_counter() - start) * 1000,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None,
                    "text_length": len(text) if isinstance(text, str) else None}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "wall_ms": (time.perf_counter() - start) * 1000,
                "error": exc.read(400).decode("utf-8", "replace")}


def ws_short_commit(ws, audio: bytes, timeout_s: float, retry_s: float,
                    long_future: concurrent.futures.Future | None = None) -> dict:
    busy = 0
    partials = 0
    try:
        ws.send(audio)
        start = time.perf_counter()  # stop/commit eligibility, after upload
        first_commit_before_long_done = long_future is not None and not long_future.done()
        ws.send('{"type":"commit"}')
        deadline = start + timeout_s
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            message = json.loads(ws.recv(timeout=max(0.1, remaining)))
            kind = message.get("type")
            if kind == "partial":
                partials += 1
            elif kind == "final":
                text = message.get("text")
                return {"status": "final", "stop_to_final_ms": (time.perf_counter() - start) * 1000,
                        "busy_responses": busy, "partials": partials,
                        "first_commit_before_long_done": first_commit_before_long_done,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None,
                        "text_length": len(text) if isinstance(text, str) else None}
            elif kind == "error" and message.get("message") == "server busy":
                busy += 1
                time.sleep(min(retry_s, max(0, deadline - time.perf_counter())))
                ws.send('{"type":"commit"}')
            elif kind == "error":
                return {"status": "error", "message": message.get("message"),
                        "busy_responses": busy, "partials": partials,
                        "first_commit_before_long_done": first_commit_before_long_done}
        return {"status": "timeout", "busy_responses": busy,
                "first_commit_before_long_done": first_commit_before_long_done}
    except (ConnectionClosed, TimeoutError) as exc:
        return {"status": "connection_lost", "error": str(exc),
                "busy_responses": busy, "partials": partials}


def verdict(trials: list[dict]) -> dict:
    if len(trials) != 6:
        return {"status": "inconclusive", "reason": "fewer than six completed trials"}
    checks = []
    for pair in range(3):
        idle, mixed = trials[2 * pair:2 * pair + 2]
        a, b = idle["short"], mixed["short"]
        long = mixed["long"]
        trace = mixed["long_trace"]
        complete = (
            a.get("status") == "final" and b.get("status") == "final"
            and long.get("status") == 200 and len(trace["service_ms"]) == 1
            and trace["chunks"] >= 2 and a.get("text_sha256") == b.get("text_sha256")
            and a.get("text_sha256") is not None
        )
        if not complete:
            return {"status": "inconclusive", "reason": f"pair {pair} lacked a matching successful final or trace"}
        latency_pass = (
            b["stop_to_final_ms"] - a["stop_to_final_ms"] >= 2000
            and b["stop_to_final_ms"] >= 2 * a["stop_to_final_ms"]
        )
        checks.append({"pair": pair, "latency_pass": latency_pass,
                       "busy_pass": b["busy_responses"] >= 1,
                       "overlap_pass": b["first_commit_before_long_done"]})
    material = all(all(value for key, value in check.items() if key != "pair") for check in checks)
    return {"status": "material_pilot_contention" if material else "no_material_pilot_result",
            "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--short-wav", type=Path, required=True)
    parser.add_argument("--long-wav", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "contention-pilot")
    parser.add_argument("--cpu-list", default="16-23", help="taskset CPU list for the server")
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="post-pilot serial long control and new-session admission probe")
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    for name, path, key in (
        ("server", args.binary, "server_binary_sha256"),
        ("model", args.gguf, "model_sha256"),
        ("short WAV", args.short_wav, "short_wav_sha256"),
        ("long WAV", args.long_wav, "long_wav_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != spec[key]:
            parser.error(f"{name} is missing or does not match the preregistered SHA-256: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "server.log"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["STARLING_TRACE"] = "1"
    env["OMP_NUM_THREADS"] = "8"
    command = ["taskset", "-c", args.cpu_list, str(args.binary), "--model", "granite",
               "--gguf", str(args.gguf), "--port", str(port)]
    summary = {
        "schema_version": 1, "spec_sha256": sha256_file(SPEC_PATH),
        "run_utc": datetime.now(timezone.utc).isoformat(), "cpu_affinity": args.cpu_list,
        "host_load_before": os.getloadavg(), "binary": str(args.binary),
        "model": str(args.gguf), "server_command": command, "trials": [],
    }
    short_audio = args.short_wav.read_bytes()
    long_audio = args.long_wav.read_bytes()
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_healthy(base_url, process, 180)
            warmup = http_upload(base_url, short_audio, "pilot-warmup", 300)
            if warmup.get("status") != 200 or warmup.get("text_sha256") is None:
                raise RuntimeError(f"warmup failed: {warmup}")
            summary["warmup"] = warmup
            if args.diagnostics_only:
                offset = log_path.stat().st_size
                control = http_upload(base_url, long_audio, "pilot-serial-long",
                                      spec["trial_timeout_seconds"])
                events = trace_events(log_path, offset)
                long_trace = {
                    "service_ms": [e["dur_ms"] for e in events
                                   if e.get("ev") == "request" and e.get("req") == "pilot-serial-long"],
                    "queue_wait_ms": [e["dur_ms"] for e in events
                                      if e.get("ev") == "queue_wait" and e.get("req") == "pilot-serial-long"],
                    "chunks": len([e for e in events if e.get("ev") == "chunk"
                                   and e.get("req") == "pilot-serial-long"]),
                }
                summary["serial_long_control"] = {"http": control, "trace": long_trace}
                print(f"[serial long] {control} trace={long_trace}", flush=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(http_upload, base_url, short_audio,
                                         "pilot-admission-http", spec["trial_timeout_seconds"])
                    wait_for_trace(log_path, "pilot-admission-http", "queue_wait", 30)
                    connect_start = time.perf_counter()
                    admission = {"http_active_at_connect_start": not future.done()}
                    try:
                        with connect(base_url.replace("http://", "ws://") + "/stream",
                                     ping_interval=None, proxy=None, open_timeout=20) as ws:
                            admission["connect_ms"] = (time.perf_counter() - connect_start) * 1000
                            admission["http_active_at_connect_end"] = not future.done()
                            ws.send('{"type":"ping"}')
                            pong_start = time.perf_counter()
                            reply = json.loads(ws.recv(timeout=30))
                            admission["ping_reply"] = reply.get("type")
                            admission["ping_ms"] = (time.perf_counter() - pong_start) * 1000
                            admission["http_active_at_pong"] = not future.done()
                    except (OSError, ConnectionClosed, TimeoutError, ValueError) as exc:
                        admission["error"] = str(exc)
                    admission["http"] = future.result(timeout=spec["trial_timeout_seconds"])
                    summary["new_session_probe"] = admission
                    print(f"[new session] {admission}", flush=True)
                summary["host_load_after"] = os.getloadavg()
                summary["verdict"] = {"status": "diagnostics_only"}
                (args.out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
                return 0 if control.get("status") == 200 and control.get("text_sha256") else 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                for index, scenario in enumerate(spec["sequence"]):
                    log.flush()
                    offset = log_path.stat().st_size
                    connect_start = time.perf_counter()
                    with connect(base_url.replace("http://", "ws://") + "/stream",
                                 ping_interval=None, proxy=None, max_size=4 * 1024 * 1024) as ws:
                        connect_ms = (time.perf_counter() - connect_start) * 1000
                        if scenario == "idle":
                            short = ws_short_commit(ws, short_audio,
                                                    spec["trial_timeout_seconds"],
                                                    spec["busy_retry_seconds"])
                            trial = {"scenario": scenario, "pair": index // 2,
                                     "ws_connect_ms": connect_ms, "short": short}
                        else:
                            request_id = f"pilot-long-{index // 2}"
                            print(f"[mixed {index // 2}] WS connected; starting long HTTP", flush=True)
                            long_future = pool.submit(http_upload, base_url, long_audio,
                                                      request_id, spec["trial_timeout_seconds"])
                            wait_for_trace(log_path, request_id, "queue_wait", 30)
                            short = ws_short_commit(ws, short_audio,
                                                    spec["trial_timeout_seconds"],
                                                    spec["busy_retry_seconds"], long_future)
                            long = long_future.result(timeout=spec["trial_timeout_seconds"])
                            trial = {"scenario": scenario, "pair": index // 2,
                                     "ws_connect_ms": connect_ms, "short": short, "long": long}
                    trial_events = trace_events(log_path, offset)
                    trial["short_trace"] = {
                        "service_ms": [e["dur_ms"] for e in trial_events
                                       if e.get("ev") == "request" and str(e.get("req", "")).startswith("#anon-")],
                        "busy_exits": len([e for e in trial_events if e.get("ev") == "queue_exit"
                                           and e.get("reason") == "server_busy"]),
                    }
                    if scenario == "mixed":
                        trial["long_request_id"] = request_id
                        trial["long_trace"] = {
                            "service_ms": [e["dur_ms"] for e in trial_events
                                           if e.get("ev") == "request" and e.get("req") == request_id],
                            "queue_wait_ms": [e["dur_ms"] for e in trial_events
                                              if e.get("ev") == "queue_wait" and e.get("req") == request_id],
                            "chunks": len([e for e in trial_events if e.get("ev") == "chunk"
                                           and e.get("req") == request_id]),
                            "ws_busy_exits": len([e for e in trial_events if e.get("ev") == "queue_exit"
                                                  and e.get("reason") == "server_busy"]),
                        }
                    summary["trials"].append(trial)
                    (args.out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
                    print(f"[{scenario} {index // 2}] short={short}", flush=True)
            summary["host_load_after"] = os.getloadavg()
            summary["verdict"] = verdict(summary["trials"])
            (args.out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
            return 0
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
