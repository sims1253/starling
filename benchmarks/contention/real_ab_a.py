"""Real Granite A/B/A cache-isolation check for opt-in chunk scheduling.

Two public, distinct 16 kHz WAVs with partial final chunks are decoded
serially, then A/B/A are interleaved in one fair server. Save hashes, not text.
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

from websockets.sync.client import connect


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", errors="replace") as src:
        for line in src:
            if line.startswith("[trace] "):
                try:
                    records.append(json.loads(line[8:]))
                except json.JSONDecodeError:
                    pass  # A concurrent writer may leave one incomplete line.
    return records


def wait_event(path: Path, request_id: str, kind: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 90
    offset = 0
    pending = b""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        if path.exists():
            with path.open("rb") as src:
                src.seek(offset)
                data = src.read()
                offset = src.tell()
            if data:
                lines = (pending + data).split(b"\n")
                pending = lines.pop()
                for line in lines:
                    if not line.startswith(b"[trace] "):
                        continue
                    try:
                        event = json.loads(line[8:])
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if event.get("req") == request_id and event.get("ev") == kind:
                        return
        time.sleep(0.05)
    raise TimeoutError(f"missing {kind} for {request_id}")


def wait_ready(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200 and json.load(response).get("loaded"):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError("server did not load")


def upload(url: str, audio: bytes, request_id: str) -> dict:
    boundary = f"starling-aba-{request_id}"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngranite\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{url}/v1/audio/transcriptions", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "x-request-id": request_id},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            text = json.load(response).get("text")
            return {"status": response.status,
                    "wall_ms": (time.perf_counter() - start) * 1000,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest()
                    if isinstance(text, str) else None,
                    "text_length": len(text) if isinstance(text, str) else None}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code,
                "wall_ms": (time.perf_counter() - start) * 1000,
                "error": exc.read(400).decode("utf-8", "replace")}


def run_arm(binary: Path, model: Path, audio_a: bytes, audio_b: bytes,
            warmup: bytes, out_dir: Path, cores: str, fair: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ("fair.log" if fair else "serial.log")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = ["taskset", "-c", cores, str(binary), "--model", "granite",
               "--gguf", str(model), "--port", str(port)]
    if fair:
        command.append("--granite-chunk-fairness")
    env = os.environ.copy()
    env["STARLING_TRACE"] = "1"
    env["OMP_NUM_THREADS"] = "8"
    arm: dict = {"command": command, "binary_sha256": sha256(binary)}
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(url, process)
            arm["warmup"] = upload(url, warmup, "aba-warmup")
            if arm["warmup"]["status"] != 200:
                raise RuntimeError(f"warmup failed: {arm['warmup']}")
            if fair:
                futures: dict[str, concurrent.futures.Future] = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    try:
                        futures["a1"] = pool.submit(upload, url, audio_a, "aba-a1")
                        wait_event(log_path, "aba-a1", "queue_wait", process)
                        # Probe a new session during A1's active first chunk.
                        connect_start = time.perf_counter()
                        with connect(url.replace("http://", "ws://") + "/stream",
                                     ping_interval=None, proxy=None, open_timeout=10) as ws:
                            connect_ms = (time.perf_counter() - connect_start) * 1000
                            ping_start = time.perf_counter()
                            ws.send('{"type":"ping"}')
                            reply = json.loads(ws.recv(timeout=5))
                            ping_ms = (time.perf_counter() - ping_start) * 1000
                        arm["new_session"] = {"connect_ms": connect_ms,
                                              "ping_ms": ping_ms,
                                              "reply": reply.get("type")}
                        futures["b"] = pool.submit(upload, url, audio_b, "aba-b")
                        wait_event(log_path, "aba-b", "queue_enter", process)
                        futures["a2"] = pool.submit(upload, url, audio_a, "aba-a2")
                        wait_event(log_path, "aba-a2", "queue_enter", process)
                        if any(e.get("req") == "aba-a1" and e.get("ev") == "chunk"
                               for e in events(log_path)):
                            raise RuntimeError("A1 completed its first chunk before B/A2 queued")
                    except Exception as exc:
                        arm["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        # Preserve every submitted request outcome even when
                        # admission or ordering fails before normal collection.
                        for name, future in futures.items():
                            try:
                                arm[name] = future.result(timeout=600)
                            except Exception as exc:
                                arm[name] = {"status": "client_error",
                                             "error": f"{type(exc).__name__}: {exc}"}
            else:
                arm["a1"] = upload(url, audio_a, "aba-serial-a")
                arm["b"] = upload(url, audio_b, "aba-serial-b")
        except Exception as exc:
            arm["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    record = events(log_path)
    for key, req in (("a1", "aba-a1" if fair else "aba-serial-a"),
                     ("b", "aba-b" if fair else "aba-serial-b"),
                     ("a2", "aba-a2")):
        if key in arm:
            arm[key]["chunks"] = [e["chunk"] for e in record
                                  if e.get("req") == req and e.get("ev") == "chunk"]
    arm["chunk_order"] = [(e.get("req"), e.get("chunk")) for e in record
                          if e.get("ev") == "chunk" and
                          e.get("req") in {"aba-a1", "aba-b", "aba-a2"}]
    return arm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-binary", type=Path, required=True)
    parser.add_argument("--fair-binary", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--a-wav", type=Path, required=True)
    parser.add_argument("--b-wav", type=Path, required=True)
    parser.add_argument("--warmup-wav", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpu-list", default="16-23")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    for name, path in (("serial_binary", args.serial_binary),
                       ("fair_binary", args.fair_binary), ("model", args.gguf),
                       ("a_wav", args.a_wav), ("b_wav", args.b_wav),
                       ("warmup_wav", args.warmup_wav)):
        if not path.is_file() or sha256(path) != spec[name + "_sha256"]:
            parser.error(f"{name} SHA-256 mismatch: {path}")
    a, b, warmup = args.a_wav.read_bytes(), args.b_wav.read_bytes(), args.warmup_wav.read_bytes()
    result = {"run_utc": datetime.now(timezone.utc).isoformat(),
              "spec_sha256": sha256(args.spec), "cpu_affinity": args.cpu_list,
              "host_load_before": os.getloadavg()}
    result["serial"] = run_arm(args.serial_binary, args.gguf, a, b, warmup,
                               args.out_dir, args.cpu_list, False)
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    result["fair"] = run_arm(args.fair_binary, args.gguf, a, b, warmup,
                             args.out_dir, args.cpu_list, True)
    result["host_load_after"] = os.getloadavg()
    serial, fair = result["serial"], result["fair"]
    sa, sb = serial.get("a1", {}), serial.get("b", {})
    fa, fb, fa2 = fair.get("a1", {}), fair.get("b", {}), fair.get("a2", {})
    requests = (sa, sb, fa, fb, fa2)
    status_ok = all(x.get("status") == 200 and x.get("text_sha256") for x in requests)
    hashes_ok = (status_ok and
                 sa["text_sha256"] == fa["text_sha256"] == fa2["text_sha256"] and
                 sb["text_sha256"] == fb["text_sha256"] and
                 sa["text_sha256"] != sb["text_sha256"])
    chunks_ok = all(x.get("chunks") == [1, 2, 3] for x in requests)
    order = fair["chunk_order"]
    a1_first = next((i for i, e in enumerate(order) if e == ("aba-a1", 1)), -1)
    a1_second = next((i for i, e in enumerate(order) if e == ("aba-a1", 2)), -1)
    b_first = next((i for i, e in enumerate(order) if e == ("aba-b", 1)), -1)
    a2_first = next((i for i, e in enumerate(order) if e == ("aba-a2", 1)), -1)
    interleaved = a1_first >= 0 and a1_first < b_first < a1_second and a1_first < a2_first < a1_second
    admission = fair.get("new_session", {})
    admission_ok = admission.get("reply") == "pong" and admission.get("ping_ms", 1e9) < 2000
    result["verdict"] = {"status": "pass" if all((status_ok, hashes_ok, chunks_ok, interleaved, admission_ok,
                                                    not serial.get("error"), not fair.get("error")))
                         else "no_go_or_inconclusive", "status_ok": status_ok,
                         "hashes_ok": hashes_ok, "chunks_ok": chunks_ok,
                         "interleaved": interleaved, "admission_ok": admission_ok}
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["verdict"], indent=2), flush=True)
    return 0 if result["verdict"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
