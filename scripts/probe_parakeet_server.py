"""Probe the persistent parakeet.cpp server: spawn it, warm up, time N requests.

Measures the REAL steady-state per-utterance latency of the ggml engine with
model load paid exactly once (the persistent-server architecture), vs the
per-process-spawn tax the benchmark harness currently pays via ParakeetCpp.

Single self-contained process: it spawns the server as a child, waits for
health, runs warmup + timed requests, then tears down. Run it under one Bash
call so the server stays alive for the whole measurement.

Usage:
    python scripts/probe_parakeet_server.py [--port 8765] [--reps 5] \\
        [--fixture short|medium|long|all] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER = Path("/home/m0hawk/Documents/parakeet.cpp/build-cuda/examples/server/parakeet-server")
MODEL = Path("/home/m0hawk/asr-bench/models/tdt-0.6b-v3-f16.gguf")
FIX = REPO / "tests" / "fixtures"


def wait_health(host: str, port: int, timeout: float = 90.0) -> bool:
    url = f"http://{host}:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def transcribe(host: str, port: int, wav: Path) -> tuple[str, float]:
    """POST a wav file; return (text, wall_clock_seconds)."""
    import uuid
    boundary = "----pk" + uuid.uuid4().hex
    data = wav.read_bytes()
    body = b"--" + boundary.encode() + b"\r\n" + \
           b'Content-Disposition: form-data; name="file"; filename="' + wav.name.encode() + \
           b'"\r\nContent-Type: audio/wav\r\n\r\n' + data + b"\r\n--" + boundary.encode() + \
           b"--\r\n"
    url = f"http://{host}:{port}/v1/audio/transcriptions"
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    return out.get("text", "").strip(), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fixture", default="all",
                    help="short|medium|long|all")
    args = ap.parse_args()

    if not SERVER.exists():
        print(f"ERROR: server binary missing: {SERVER}", file=sys.stderr); return 1
    if not MODEL.exists():
        print(f"ERROR: model missing: {MODEL}", file=sys.stderr); return 1

    fixtures = ["short", "medium", "long"] if args.fixture == "all" else [args.fixture]

    # spawn the server as a child; keep it alive for the whole run
    print(f"[probe] spawning server: {SERVER.name} port {args.port}")
    env = dict(os.environ)
    logf = open("/tmp/pkserver_probe.log", "w")
    proc = subprocess.Popen(
        [str(SERVER), "--model", str(MODEL),
         "--host", args.host, "--port", str(args.port)],
        stdout=logf, stderr=subprocess.STDOUT, env=env,
    )
    try:
        t_load0 = time.perf_counter()
        if not wait_health(args.host, args.port):
            print("[probe] server did not become healthy; log:", file=sys.stderr)
            logf.flush()
            print(Path("/tmp/pkserver_probe.log").read_text()[-2000:], file=sys.stderr)
            return 2
        load_s = time.perf_counter() - t_load0
        print(f"[probe] server up after {load_s:.1f}s (model load + bind)")

        for name in fixtures:
            wav = FIX / f"{name}.wav"
            # warmup (first real request pays encoder graph capture etc.)
            txt0, _ = transcribe(args.host, args.port, wav)
            samples = []
            for _ in range(args.reps):
                txt, ms = transcribe(args.host, args.port, wav)
                samples.append(ms * 1000.0)
            samples.sort()
            med = samples[len(samples) // 2]
            import statistics
            mean = statistics.mean(samples)
            audio_s = 7.43 if name == "short" else (22.30 if name == "medium" else 74.35)
            print(f"[probe] {name}: median {med:.1f}ms mean {mean:.1f}ms "
                  f"(min {min(samples):.1f} max {max(samples):.1f}) over {args.reps} reps; "
                  f"RTFx {audio_s/(med/1000):.0f}x")
            print(f"[probe]   text: {txt[:120]!r}")
        return 0
    finally:
        logf.flush()
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try: proc.kill()
            except Exception: pass
        logf.close()


if __name__ == "__main__":
    raise SystemExit(main())
