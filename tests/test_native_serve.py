#!/usr/bin/env python3
"""Integration tests for starling-serve (native HTTP/WS API).

Tests the native serving binary's HTTP/WebSocket API for compatibility with the
Python server contract. Runs against a live server instance.

Usage:
    # Build the binary first:
    cmake -B build -DSTARLING_SERVE=ON && cmake --build build --target starling-serve

    # Run tests (starts a server on port 18181 with --no-eager-load):
    uv run python tests/test_native_serve.py

    # Or against an already-running server:
    uv run python tests/test_native_serve.py --host 127.0.0.1 --port 8181
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "build-serve" / "starling-serve"
DEFAULT_PORT = 18181


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.2)
    return False


def make_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    """Create a minimal WAV file from float32 samples."""
    pcm16 = (samples * 32768).clip(-32768, 32767).astype("<i2")
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, cond: bool, detail: str = ""):
        if cond:
            self.passed += 1
            print(f"  PASS: {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  FAIL: {name} {detail}")


def test_health(base_url: str, tr: TestResults):
    """GET /health returns the expected JSON contract."""
    r = requests.get(f"{base_url}/health", timeout=5)
    tr.check("health status 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    tr.check("health has model", "model" in data, str(data))
    tr.check("health has loaded", "loaded" in data, str(data))
    tr.check("health has phase", "phase" in data, str(data))
    tr.check("health has queue_depth", "queue_depth" in data, str(data))


def test_health_alias(base_url: str, tr: TestResults):
    """GET / returns the same health response."""
    r1 = requests.get(f"{base_url}/health", timeout=5)
    r2 = requests.get(f"{base_url}/", timeout=5)
    tr.check("root is health alias", r1.json() == r2.json())


def test_warmup(base_url: str, tr: TestResults):
    """POST /warmup returns 202."""
    r = requests.post(f"{base_url}/warmup", timeout=5)
    tr.check("warmup returns 202", r.status_code == 202, f"got {r.status_code}")


def test_transcribe_empty(base_url: str, tr: TestResults):
    """POST /transcribe with empty body returns 400."""
    r = requests.post(f"{base_url}/transcribe", data=b"", timeout=5)
    tr.check("empty transcribe returns 400", r.status_code == 400,
             f"got {r.status_code}")


def test_cancel_notfound(base_url: str, tr: TestResults):
    """DELETE /inference/nonexistent returns 404."""
    r = requests.delete(f"{base_url}/inference/nonexistent", timeout=5)
    tr.check("cancel nonexistent returns 404", r.status_code == 404,
             f"got {r.status_code}")


def test_transcribe_wav(base_url: str, tr: TestResults):
    """POST /transcribe with a real WAV file returns proper JSON shape.

    Note: without a loaded model this will fail at inference; we test the
    error handling path (the server should return a structured error, not crash).
    """
    # 1 second of silence.
    samples = np.zeros(16000, dtype=np.float32)
    wav = make_wav(samples)
    r = requests.post(f"{base_url}/transcribe", data=wav,
                      headers={"Content-Type": "application/octet-stream"},
                      timeout=30)
    # Without a real model, the response should be a structured error (not a crash).
    tr.check("transcribe returns JSON", r.headers.get("content-type", "").startswith("application/json"),
             f"got {r.headers.get('content-type')}")
    data = r.json()
    if r.status_code == 200:
        tr.check("transcribe has text", "text" in data, str(data))
        tr.check("transcribe has duration_s", "duration_s" in data, str(data))
        tr.check("transcribe has request_id", "request_id" in data, str(data))
    else:
        tr.check("transcribe error is structured", "error" in data, str(data))


def test_request_id_passthrough(base_url: str, tr: TestResults):
    """X-Request-Id header is echoed back in the response."""
    rid = "test-rid-12345"
    samples = np.zeros(1600, dtype=np.float32)
    wav = make_wav(samples)
    r = requests.post(f"{base_url}/transcribe", data=wav,
                      headers={"Content-Type": "application/octet-stream",
                               "X-Request-Id": rid},
                      timeout=30)
    data = r.json()
    tr.check("request_id echoed", data.get("request_id") == rid,
             f"got {data.get('request_id')}")




# ---- WebSocket tests ------------------------------------------------------
def test_websocket_control_frames(port: int, tr: TestResults):
    """WS /stream control frames: ping→pong, reset→reset_ack, commit→final."""
    try:
        import asyncio
        import websockets
    except ImportError:
        tr.check("websockets module available", False, "not installed")
        return

    async def run():
        uri = f"ws://127.0.0.1:{port}/stream"
        async with websockets.connect(uri) as ws:
            # ping → pong
            await ws.send(json.dumps({"type": "ping"}))
            resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(resp)
            tr.check("ws ping→pong", data.get("type") == "pong", str(data))

            # reset → reset_ack
            await ws.send(json.dumps({"type": "reset"}))
            resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(resp)
            tr.check("ws reset→reset_ack", data.get("type") == "reset_ack", str(data))

            # commit with empty buffer → final
            await ws.send(json.dumps({"type": "commit"}))
            resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(resp)
            tr.check("ws commit→final", data.get("type") == "final", str(data))
            tr.check("ws final has text", "text" in data, str(data))
            tr.check("ws final has duration_s", "duration_s" in data, str(data))

    try:
        asyncio.run(run())
    except Exception as e:
        tr.check("ws test completed", False, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--binary", type=Path, default=BINARY)
    ap.add_argument("--no-server", action="store_true",
                    help="Don't start a server; test against an existing one")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    proc = None

    if not args.no_server:
        if not args.binary.exists():
            print(f"Binary not found: {args.binary}")
            print("Build first: cmake -B build-serve -DSTARLING_SERVE=ON && "
                  "cmake --build build-serve --target starling-serve")
            return 1
        print(f"Starting {args.binary} on {args.host}:{args.port} ...")
        proc = subprocess.Popen(
            [str(args.binary), "--model", "parakeet", "--gguf", "/dev/null",
             "--host", args.host, "--port", str(args.port), "--no-eager-load"],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if not _wait_for_port(args.host, args.port):
            print("ERROR: server did not start")
            proc.kill()
            return 1
        print("Server started.\n")

    tr = TestResults()
    try:
        print("Testing HTTP endpoints:")
        test_health(base_url, tr)
        test_health_alias(base_url, tr)
        test_warmup(base_url, tr)
        test_transcribe_empty(base_url, tr)
        test_cancel_notfound(base_url, tr)
        test_transcribe_wav(base_url, tr)
        test_request_id_passthrough(base_url, tr)
        print("\nTesting WebSocket control frames:")
        test_websocket_control_frames(args.port, tr)
    finally:
        if proc:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)

    print(f"\n{'='*60}")
    print(f"Results: {tr.passed} passed, {tr.failed} failed")
    if tr.failed:
        print("\nFailures:")
        for e in tr.errors:
            print(f"  - {e}")
        return 1
    print("All tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
