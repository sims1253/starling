#!/usr/bin/env python3
"""Integration tests for starling-serve (native HTTP/WS API).

Tests the native serving binary's HTTP/WebSocket API for compatibility with the
Python server contract. Runs against live server instances.

This is a standalone script, NOT a pytest module: the test_* functions below
take live-server arguments, so pytest collection is skipped explicitly.

Usage:
    # Build the binary first:
    cmake -B build -DSTARLING_SERVE=ON && cmake --build build --target starling-serve

    # Default run (server on port 18181 with --gguf /dev/null --no-eager-load;
    # exercises the transport contract without a model):
    uv run python tests/test_native_serve.py

    # Real-model run (eager load; true round trips incl. transcript text,
    # queue semantics, and WS streaming with audio):
    uv run python tests/test_native_serve.py --gguf /path/to/parakeet.gguf

    # Pin the audio + expected transcript (CI downloads its own clip):
    uv run python tests/test_native_serve.py --gguf model.gguf \
        --audio clip.wav --expected-text "..."
"""
from __future__ import annotations

import argparse
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

# Standalone script: its test_* functions need a live server and would only
# produce fixture errors under pytest. Skip collection when pytest is the
# runner (it has already imported itself by the time we execute). This check
# precedes the third-party imports so a pytest run skips cleanly even when
# requests/numpy are not installed.
if "pytest" in sys.modules:
    import pytest

    pytest.skip("integration script; run directly: python tests/test_native_serve.py",
                allow_module_level=True)

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]

# The canonical short fixture (LibriSpeech 2086-149220-0033, mono 16 kHz PCM16)
# is the audio behind golden/parakeet_tdt_short_*; it lives untracked next to
# tests/fixtures (audio is never committed — see .gitignore). Fall back to it
# when --audio is not given, else require an explicit clip.
DEFAULT_AUDIO = REPO / "tests" / "fixtures" / "2086-149220-0033.wav"
GOLDEN_SHORT_TEXT = REPO / "golden" / "parakeet_tdt_short_text.txt"
# Byte-exact golden for the default audio (parakeet-tdt greedy decode is
# deterministic); used when the golden file is absent.
DEFAULT_EXPECTED_TEXT = (
    "Well, I don't wish to see it any more, observed Phoebe, turning away her "
    "eyes. It is certainly very like the old portrait."
)
# Similarity floor for transcript checks: generous enough to absorb
# quantization drift (q8_0 vs bf16), strict enough that a broken decode
# (empty/garbage text) can never pass.
TEXT_SIMILARITY_MIN = 0.80


def _default_binary() -> Path:
    """Locate the default starling-serve binary across platforms."""
    if os.name == "nt":
        candidates = [
            REPO / "build" / "Release" / "starling-serve.exe",  # MSVC multi-config
            REPO / "build" / "starling-serve.exe",
        ]
    else:
        candidates = [REPO / "build" / "starling-serve"]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


BINARY = _default_binary()
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


def _free_port() -> int:
    """Pick a free TCP port for an auxiliary server instance."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    """Create a minimal WAV file from float32 samples."""
    pcm16 = (samples * 32768).clip(-32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono PCM16 WAV as float32 samples."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2, \
            "test audio must be mono PCM16"
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, sr


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return (samples * 32768).clip(-32768, 32767).astype("<i2").tobytes()


def text_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(
        None, a.strip().lower(), b.strip().lower()).ratio()


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


# ---- server process management ---------------------------------------------
#
# Every spawned server gets reader threads draining stdout+stderr into a ring
# buffer: server logging into an undrained subprocess.PIPE can fill the OS pipe
# buffer (~64 KB) and deadlock the server mid-test. stop() always runs the
# wait() under a timeout with a kill() fallback so a hung server can't wedge
# the suite (issue #22 script hygiene).

class ServeProc:
    def __init__(self, binary: Path, model: str, gguf: str, host: str,
                 port: int, extra_args: list[str] | None = None,
                 eager: bool = True, start_timeout: float = 10.0):
        self.host, self.port = host, port
        self.log: deque[str] = deque(maxlen=4000)
        args = [str(binary), "--model", model, "--gguf", gguf,
                "--host", host, "--port", str(port)]
        if not eager:
            args.append("--no-eager-load")
        args += extra_args or []
        print(f"Starting {binary.name} on {host}:{port} "
              f"({'eager' if eager else 'lazy'}) extra={extra_args or []}")
        self.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self._drain_threads = [
            threading.Thread(target=self._drain, args=(stream,), daemon=True)
            for stream in (self.proc.stdout, self.proc.stderr)
        ]
        for t in self._drain_threads:
            t.start()
        if not _wait_for_port(host, port, timeout=start_timeout):
            raise RuntimeError(
                f"server did not start on {host}:{port} within {start_timeout}s\n"
                + "\n".join(list(self.log)[-30:]))

    def _drain(self, stream):
        try:
            for line in iter(stream.readline, b""):
                self.log.append(line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self):
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"  (server on :{self.port} ignored SIGTERM; killing)")
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

    def dump_log_tail(self, n: int = 25):
        for line in list(self.log)[-n:]:
            print(f"    | {line}")


# ---- default-mode tests (no model: /dev/null placeholder GGUF) --------------

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


def test_transcribe_wrong_sample_rate(base_url: str, tr: TestResults):
    """Non-16 kHz WAV is rejected with 400 + sample-rate mismatch error."""
    samples = np.zeros(800, dtype=np.float32)
    wav = make_wav(samples, sr=8000)
    r = requests.post(f"{base_url}/transcribe", data=wav,
                      headers={"Content-Type": "application/octet-stream"},
                      timeout=10)
    tr.check("8kHz wav returns 400", r.status_code == 400, f"got {r.status_code}")
    tr.check("8kHz error mentions sample rate",
             "sample rate mismatch" in r.text, r.text)


def test_transcribe_model_not_loaded(base_url: str, tr: TestResults):
    """Valid 16 kHz WAV with an unloadable model returns a structured 503.

    The test server runs --no-eager-load with a placeholder --gguf, so the
    lazy load fails and the request must surface 'model not loaded' as JSON,
    not a crash or a dead socket.
    """
    samples = np.zeros(1600, dtype=np.float32)
    wav = make_wav(samples, sr=16000)
    r = requests.post(f"{base_url}/transcribe", data=wav,
                      headers={"Content-Type": "application/octet-stream"},
                      timeout=30)
    tr.check("16kHz wav on unloaded model returns 503", r.status_code == 503,
             f"got {r.status_code}")
    data = r.json()
    tr.check("503 error is model not loaded",
             data.get("error") == "model not loaded", str(data))


def test_transcribe_pcm16_fallback(base_url: str, tr: TestResults):
    """A non-WAV payload is treated as raw mono PCM16 @ 16 kHz.

    Raw PCM16 of silence is not valid WAV; if the fallback works, the request
    passes payload decoding and reaches inference (here: the 503 lazy-load
    failure, same as the WAV path) instead of a 400 malformed-audio error.
    """
    pcm16 = (np.zeros(1600, dtype=np.float32) * 32768).astype("<i2").tobytes()
    r = requests.post(f"{base_url}/transcribe", data=pcm16,
                      headers={"Content-Type": "application/octet-stream"},
                      timeout=30)
    tr.check("raw PCM16 reaches inference (not 400 malformed)",
             r.status_code != 400, f"got {r.status_code}")
    data = r.json()
    tr.check("raw PCM16 error is model not loaded",
             data.get("error") == "model not loaded", str(data))


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


def test_multipart_wav_decodes(base_url: str, tr: TestResults, endpoint: str):
    """multipart/form-data WAV uploads get parsed as WAV, not PCM16 garbage.

    Regression for the separator bug (issue #12): every multipart payload used
    to come out prefixed with a stray "\\r\\n", so WAV RIFF sniffing failed and
    the payload silently fell through to the raw-PCM16 fallback. An 8 kHz WAV
    can only produce the 400 sample-rate-mismatch response if the WAV header
    was actually parsed.
    """
    samples = np.zeros(800, dtype=np.float32)
    wav = make_wav(samples, sr=8000)
    r = requests.post(f"{base_url}/{endpoint}",
                      files={"audio": ("clip.wav", wav, "audio/wav")},
                      timeout=10)
    tr.check(f"multipart 8kHz wav on /{endpoint} returns 400",
             r.status_code == 400, f"got {r.status_code}: {r.text[:120]}")
    tr.check(f"multipart 8kHz wav on /{endpoint} mentions sample rate",
             "sample rate mismatch" in r.text, r.text[:200])


def test_multipart_wav_reaches_inference(base_url: str, tr: TestResults,
                                         endpoint: str):
    """A well-formed 16 kHz multipart WAV passes parsing and reaches inference.

    On the placeholder model that means the structured 503 'model not loaded'
    (identical to the raw-upload path) — proving the multipart payload was
    extracted byte-exact and decoded as WAV.
    """
    samples = np.zeros(1600, dtype=np.float32)
    wav = make_wav(samples, sr=16000)
    r = requests.post(f"{base_url}/{endpoint}",
                      files={"audio": ("clip.wav", wav, "audio/wav")},
                      timeout=30)
    data = r.json()
    tr.check(f"multipart 16kHz wav on /{endpoint} reaches inference",
             r.status_code == 503 and data.get("error") == "model not loaded",
             f"got {r.status_code}: {data}")


def test_malformed_wav_huge_frame_count(base_url: str, tr: TestResults):
    """A WAV header claiming billions of frames fails fast with 400.

    Crafted RF64: the ds64 sampleCount (0x100000000) is reported verbatim by
    dr_wav over an 8-byte payload. The decoder derives the frame cap from the
    payload, rejects the file, and the RIFF/RF64 magic keeps it out of the
    raw-PCM16 fallback (issue #12: no gigabyte allocations, no garbage 200s).
    """
    def put(fmt, *vals):
        return struct.pack(fmt, *vals)

    w = b"RF64" + put("<I", 0xFFFFFFFF) + b"WAVE"
    w += b"ds64" + put("<I", 28) + put("<QQQ", 100, 8, 0x100000000) + put("<I", 0)
    w += b"fmt " + put("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    w += b"data" + put("<I", 0xFFFFFFFF) + b"\x01" * 8
    w = w[:4] + put("<I", len(w) - 8) + w[8:]

    t0 = time.monotonic()
    r = requests.post(f"{base_url}/transcribe", data=w,
                      headers={"Content-Type": "application/octet-stream"},
                      timeout=10)
    dt = time.monotonic() - t0
    tr.check("huge-frame-count WAV returns 400", r.status_code == 400,
             f"got {r.status_code}: {r.text[:120]}")
    tr.check("huge-frame-count WAV rejected fast", dt < 5.0, f"took {dt:.2f}s")


# ---- WebSocket tests (default mode) -----------------------------------------

def test_websocket_control_frames(host: str, port: int, tr: TestResults):
    """WS /stream control frames: ping→pong, reset→reset_ack, commit→final."""
    try:
        import asyncio
        import websockets
    except ImportError:
        tr.check("websockets module available", False, "not installed")
        return

    async def run():
        uri = f"ws://{host}:{port}/stream"
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


def test_websocket_stream_cap(binary: Path, model: str, tr: TestResults):
    """--max-stream-seconds caps the per-connection buffer (issue #12).

    An oversized binary frame is refused with ONE error frame; further audio
    is ignored (but control frames still work), and reset re-enables audio.
    """
    try:
        import asyncio
        import websockets
    except ImportError:
        tr.check("websockets module available", False, "not installed")
        return

    port = _free_port()
    server = ServeProc(binary, model, "/dev/null", "127.0.0.1", port,
                       extra_args=["--max-stream-seconds", "2"],
                       eager=False)
    try:
        async def run():
            uri = f"ws://127.0.0.1:{port}/stream"
            async with websockets.connect(uri, max_size=None) as ws:
                second = pcm16_bytes(np.zeros(16000, dtype=np.float32))
                half = second[: 16000]  # 0.5 s of PCM16 bytes
                # 1.5 s (two frames): under the cap, accepted silently.
                await ws.send(second)
                await ws.send(half)
                await asyncio.sleep(0.2)
                # 1 s more: would buffer 2.5 s > 2.0 s cap → error frame.
                await ws.send(second)
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(resp)
                tr.check("ws cap sends error frame",
                         data.get("type") == "error"
                         and "limit" in data.get("message", ""), str(data))

                # Further audio is ignored — but the connection lives on.
                await ws.send(second)
                await asyncio.sleep(0.3)
                await ws.send(json.dumps({"type": "ping"}))
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(resp)
                tr.check("ws alive after cap (ping→pong)",
                         data.get("type") == "pong", str(data))

                # reset re-enables audio acceptance.
                await ws.send(json.dumps({"type": "reset"}))
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(resp)
                tr.check("ws reset after cap → reset_ack",
                         data.get("type") == "reset_ack", str(data))
                await ws.send(half)  # 0.5 s, accepted again (no error frame)
                await asyncio.sleep(0.3)
                await ws.send(json.dumps({"type": "commit"}))
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(resp)
                tr.check("ws commit after reset → final",
                         data.get("type") == "final", str(data))

        asyncio.run(run())
    except Exception as e:
        tr.check("ws cap test completed", False, str(e))
    finally:
        server.stop()


# ---- real-model tests (require --gguf) --------------------------------------

def test_real_roundtrip(base_url: str, tr: TestResults, samples: np.ndarray,
                        expected: str | None):
    """True round trip: 200 + transcript + duration + request id echo."""
    rid = "e2e-roundtrip-1"
    wav = make_wav(samples)
    t0 = time.monotonic()
    r = requests.post(f"{base_url}/transcribe", data=wav,
                      headers={"Content-Type": "application/octet-stream",
                               "X-Request-Id": rid},
                      timeout=300)
    dt = time.monotonic() - t0
    tr.check("roundtrip returns 200", r.status_code == 200,
             f"got {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        return
    data = r.json()
    text = data.get("text", "")
    print(f"    ({dt:.1f}s) text: {text[:100]!r}")
    tr.check("roundtrip text non-empty", len(text.strip()) > 0, repr(text))
    tr.check("roundtrip request_id echoed", data.get("request_id") == rid,
             str(data.get("request_id")))
    dur = data.get("duration_s", -1.0)
    want = len(samples) / 16000.0
    tr.check("roundtrip duration_s matches audio",
             abs(dur - want) < 0.1, f"got {dur}, want {want:.2f}")
    if expected:
        sim = text_similarity(text, expected)
        tr.check(f"roundtrip text matches expected (sim={sim:.2f})",
                 sim >= TEXT_SIMILARITY_MIN,
                 f"\n    got:      {text!r}\n    expected: {expected!r}")


def test_real_multipart_roundtrip(base_url: str, tr: TestResults,
                                  samples: np.ndarray, expected: str | None,
                                  endpoint: str):
    """multipart upload round-trips on /transcribe AND /inference (issue #22)."""
    wav = make_wav(samples)
    r = requests.post(f"{base_url}/{endpoint}",
                      files={"audio": ("clip.wav", wav, "audio/wav")},
                      headers={"X-Request-Id": f"e2e-mp-{endpoint}"},
                      timeout=300)
    tr.check(f"multipart roundtrip on /{endpoint} returns 200",
             r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        return
    data = r.json()
    text = data.get("text", "")
    tr.check(f"multipart text non-empty on /{endpoint}",
             len(text.strip()) > 0, repr(text))
    if expected:
        sim = text_similarity(text, expected)
        tr.check(f"multipart text matches on /{endpoint} (sim={sim:.2f})",
                 sim >= TEXT_SIMILARITY_MIN,
                 f"\n    got:      {text!r}\n    expected: {expected!r}")


def test_real_duplicate_request_id(base_url: str, tr: TestResults,
                                   samples: np.ndarray):
    """Two concurrent requests with the same X-Request-Id: 200 + 409."""
    wav = make_wav(samples)
    results: list[int] = []
    barrier = threading.Barrier(2)

    def post():
        barrier.wait()
        try:
            r = requests.post(f"{base_url}/transcribe", data=wav,
                              headers={"X-Request-Id": "dup-rid-1"},
                              timeout=300)
            results.append(r.status_code)
        except Exception as e:
            results.append(-1)
            print(f"    dup post error: {e}")

    for attempt in range(3):
        results.clear()
        threads = [threading.Thread(target=post) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if sorted(results) == [200, 409]:
            break
        time.sleep(0.2)  # retry: both finished before the second registered
    tr.check("duplicate X-Request-Id → one 200 + one 409",
             sorted(results) == [200, 409], f"got {sorted(results)}")


def test_real_server_busy(base_url: str, tr: TestResults, samples: np.ndarray):
    """Filling kMaxWaiters (8) rejects further requests with 503 busy."""
    wav = make_wav(samples)
    n = 16
    results: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(n)

    def post():
        barrier.wait()
        try:
            r = requests.post(f"{base_url}/transcribe", data=wav,
                              headers={"X-Request-Id": f"barrage-{time.time_ns()}"},
                              timeout=600)
            with lock:
                results.append(r.status_code)
        except Exception as e:
            with lock:
                results.append(-1)
            print(f"    barrage post error: {e}")

    saw_busy = False
    for attempt in range(3):
        results.clear()
        threads = [threading.Thread(target=post) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if 503 in results:
            saw_busy = True
            break
        # All 16 got through without ever exceeding 8 concurrent waiters —
        # the anchor decodes finished too fast. Retry with a bigger barrage.
        n = 24
        barrier = threading.Barrier(n)
    busy = results.count(503)
    ok = results.count(200)
    tr.check("concurrent barrage hits 503 server busy", saw_busy,
             f"statuses={sorted(set(results))}")
    tr.check("barrage still completes requests", ok >= 1,
             f"ok={ok}, statuses={sorted(set(results))}")


def test_real_cancel_queued(base_url: str, tr: TestResults,
                            anchor: np.ndarray):
    """DELETE /inference/<id> cancels a QUEUED request (499)."""
    anchor_wav = make_wav(anchor)
    victim_wav = make_wav(anchor[: 16000])
    victim_result: dict = {}

    def post_anchor():
        try:
            requests.post(f"{base_url}/transcribe", data=anchor_wav,
                          headers={"X-Request-Id": "cancel-anchor"},
                          timeout=600)
        except Exception:
            pass

    def post_victim():
        try:
            r = requests.post(f"{base_url}/transcribe", data=victim_wav,
                              headers={"X-Request-Id": "cancel-victim"},
                              timeout=600)
            victim_result["status"] = r.status_code
            victim_result["body"] = r.text[:200]
        except Exception as e:
            victim_result["status"] = -1
            victim_result["body"] = str(e)

    t_anchor = threading.Thread(target=post_anchor)
    t_anchor.start()
    # Wait until the anchor is in flight (busy → a transcribe is running),
    # then queue the victim behind it.
    if not _wait_busy_via_health(base_url, 30.0):
        tr.check("cancel test: anchor in flight", False, "server never busy")
        t_anchor.join()
        return
    t_victim = threading.Thread(target=post_victim)
    t_victim.start()
    if not _wait_queued_via_health(base_url, 10.0):
        tr.check("cancel test: victim queued", False,
                 "queue_depth never reached 1")
        requests.delete(f"{base_url}/inference/cancel-anchor", timeout=5)
        t_anchor.join(timeout=120)
        t_victim.join(timeout=120)
        return
    r = requests.delete(f"{base_url}/inference/cancel-victim", timeout=5)
    tr.check("DELETE queued request returns 200 cancelled",
             r.status_code == 200 and r.json().get("status") == "cancelled",
             f"got {r.status_code}: {r.text[:120]}")
    t_victim.join(timeout=30)
    tr.check("cancelled queued request returns 499",
             victim_result.get("status") == 499
             and "cancelled" in victim_result.get("body", ""),
             f"got {victim_result.get('status')}: {victim_result.get('body')}")
    # Reap the anchor (cancel it too so the queue drains fast).
    requests.delete(f"{base_url}/inference/cancel-anchor", timeout=5)
    t_anchor.join(timeout=120)


def _wait_busy_via_health(base_url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200 and r.json().get("busy"):
                return True
        except requests.RequestException:
            pass
        time.sleep(0.05)
    return False


def _wait_queued_via_health(base_url: str, timeout: float) -> bool:
    """Wait until a request is parked in the queue (depth >= 1)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200 and r.json().get("queue_depth", 0) >= 1:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.05)
    return False


def test_real_queue_timeout(binary: Path, model: str, gguf: str,
                            tr: TestResults, anchor: np.ndarray):
    """A queued request outlives --request-timeout-seconds → 504 (own server).

    The anchor is long (~1 min) so the victim stays parked in the queue past
    the 1 s timeout even on fast backends.
    """
    port = _free_port()
    server = ServeProc(binary, model, gguf, "127.0.0.1", port,
                       extra_args=["--request-timeout-seconds", "1"],
                       start_timeout=180.0)
    try:
        anchor_wav = make_wav(anchor)

        def post_anchor():
            try:
                requests.post(f"{server.base_url}/transcribe", data=anchor_wav,
                              headers={"X-Request-Id": "timeout-anchor"},
                              timeout=600)
            except Exception:
                pass

        threading.Thread(target=post_anchor, daemon=True).start()
        if not _wait_busy_via_health(server.base_url, 30.0):
            tr.check("timeout test: anchor in flight", False,
                     "server never busy")
            return
        victim: dict = {}

        def post_victim():
            try:
                r = requests.post(f"{server.base_url}/transcribe",
                                  data=make_wav(anchor[: 16000]),
                                  headers={"X-Request-Id": "timeout-victim"},
                                  timeout=30)
                victim["status"] = r.status_code
                victim["body"] = r.text[:200]
            except Exception as e:
                victim["status"] = -1
                victim["body"] = str(e)

        t0 = time.monotonic()
        tv = threading.Thread(target=post_victim)
        tv.start()
        if not _wait_queued_via_health(server.base_url, 10.0):
            tr.check("timeout test: victim queued", False,
                     "queue_depth never reached 1")
            tv.join(timeout=30)
            return
        tv.join(timeout=30)
        dt = time.monotonic() - t0
        tr.check("queued request times out with 504",
                 victim.get("status") == 504,
                 f"got {victim.get('status')}: {victim.get('body')}")
        tr.check("504 identifies the timeout",
                 "timed out" in victim.get("body", ""),
                 victim.get("body", ""))
        tr.check("504 arrives after ~1s, not instantly", dt >= 0.8,
                 f"took {dt:.2f}s")
    finally:
        server.stop()


def test_real_websocket_flow(host: str, port: int, tr: TestResults,
                             samples: np.ndarray, expected: str | None):
    """WS /stream with real audio: partials → commit → final; reset mid-stream."""
    try:
        import asyncio
        import websockets
    except ImportError:
        tr.check("websockets module available", False, "not installed")
        return

    pcm = pcm16_bytes(samples)
    chunk = 16000 // 2 * 2  # 0.5 s of PCM16 bytes per frame
    send_seconds = min(6.5, len(samples) / 16000.0)  # > min-chunk (5 s)

    async def run():
        uri = f"ws://{host}:{port}/stream"
        async with websockets.connect(uri, max_size=None) as ws:
            frames: list[dict] = []

            async def drain(timeout=0.5):
                try:
                    while True:
                        frames.append(
                            json.loads(await asyncio.wait_for(ws.recv(),
                                                              timeout=timeout)))
                except asyncio.TimeoutError:
                    pass

            sent = 0.0
            for off in range(0, int(send_seconds * 16000) * 2, chunk):
                await ws.send(pcm[off: off + chunk])
                sent += 0.5
                await drain(0.1)

            await drain(1.0)
            partials = [f for f in frames if f.get("type") == "partial"]
            tr.check("ws streaming produced partials", len(partials) >= 1,
                     f"frames={[f.get('type') for f in frames]}")
            if partials:
                tr.check("ws partial has non-empty text",
                         len(partials[-1].get("text", "").strip()) > 0,
                         str(partials[-1])[:200])

            await ws.send(json.dumps({"type": "commit"}))
            final = json.loads(await asyncio.wait_for(ws.recv(), timeout=60.0))
            tr.check("ws commit → final", final.get("type") == "final",
                     str(final)[:200])
            text = final.get("text", "")
            tr.check("ws final text non-empty", len(text.strip()) > 0,
                     repr(text))
            dur = final.get("duration_s", -1)
            tr.check("ws final duration matches sent audio",
                     abs(dur - sent) < 0.6, f"got {dur}, sent {sent:.1f}")
            if expected:
                sim = text_similarity(text, expected)
                tr.check(f"ws final text matches expected (sim={sim:.2f})",
                         sim >= TEXT_SIMILARITY_MIN,
                         f"\n    got:      {text!r}\n    expected: {expected!r}")

            # reset mid-stream: audio buffered before the reset is discarded.
            await ws.send(pcm[: chunk * 2])  # 1 s of audio
            await asyncio.sleep(0.2)
            await ws.send(json.dumps({"type": "reset"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            tr.check("ws mid-stream reset → reset_ack",
                     resp.get("type") == "reset_ack", str(resp))
            await ws.send(json.dumps({"type": "commit"}))
            final2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=60.0))
            tr.check("ws commit after reset → empty final",
                     final2.get("type") == "final"
                     and final2.get("duration_s", -1) == 0.0
                     and not final2.get("text", "").strip(),
                     str(final2)[:200])

    try:
        asyncio.run(run())
    except Exception as e:
        tr.check("ws flow test completed", False, str(e))


def test_real_idle_timeout_not_firing(binary: Path, model: str, gguf: str,
                                      tr: TestResults, samples: np.ndarray):
    """--idle-timeout must NOT fire during an active WS session (issue #22)."""
    try:
        import asyncio
        import websockets
    except ImportError:
        tr.check("websockets module available", False, "not installed")
        return

    port = _free_port()
    server = ServeProc(binary, model, gguf, "127.0.0.1", port,
                       extra_args=["--idle-timeout", "6"],
                       start_timeout=180.0)
    try:
        pcm = pcm16_bytes(samples)

        async def run():
            uri = f"ws://127.0.0.1:{port}/stream"
            async with websockets.connect(uri, max_size=None) as ws:
                # Stream for ~10 s (> idle-timeout 6 s) with frames every
                # ~1.5 s; every frame counts as activity.
                chunk = 16000  # 1 s of audio per frame
                for off in range(0, chunk * 10, chunk):
                    await ws.send(pcm[off: off + chunk])
                    await asyncio.sleep(1.5)
                # Partials may still be in flight; drain until the pong.
                await ws.send(json.dumps({"type": "ping"}))
                pong = False
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    try:
                        resp = json.loads(await asyncio.wait_for(ws.recv(),
                                                                 timeout=5.0))
                    except asyncio.TimeoutError:
                        break
                    if resp.get("type") == "pong":
                        pong = True
                        break
                tr.check("ws alive after streaming past idle-timeout",
                         pong, "no pong within 10s")

        asyncio.run(run())
        tr.check("server process still running after active WS session",
                 server.proc.poll() is None,
                 f"exit code {server.proc.poll()}")
    except Exception as e:
        tr.check("idle test completed", False, str(e))
    finally:
        server.stop()


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--binary", type=Path, default=BINARY)
    ap.add_argument("--no-server", action="store_true",
                    help="Don't start a server; test against an existing one")
    ap.add_argument("--gguf", type=Path, default=None,
                    help="Real-model GGUF: enables eager-load round-trip, "
                         "queue-semantics, and WS streaming tests "
                         "(env: STARLING_SERVE_TEST_GGUF)")
    ap.add_argument("--model", default="parakeet")
    ap.add_argument("--audio", type=Path, default=None,
                    help="16 kHz mono PCM16 WAV for real-model tests "
                         "(default: the tests/fixtures short clip if present)")
    ap.add_argument("--expected-text", default=None,
                    help="Expected transcript for --audio (default: the "
                         "parakeet short golden)")
    args = ap.parse_args()

    gguf = args.gguf or (Path(os.environ["STARLING_SERVE_TEST_GGUF"])
                         if os.environ.get("STARLING_SERVE_TEST_GGUF")
                         else None)

    if not args.no_server and not args.binary.exists():
        print(f"Binary not found: {args.binary}")
        print("Build first: cmake -B build -DSTARLING_SERVE=ON && "
              "cmake --build build --target starling-serve")
        return 1

    # Resolve real-model test inputs.
    samples = expected = None
    if gguf is not None:
        audio_path = args.audio or (DEFAULT_AUDIO if DEFAULT_AUDIO.exists() else None)
        if audio_path is None or not audio_path.exists():
            print(f"Real-model tests need audio: pass --audio (looked for "
                  f"{DEFAULT_AUDIO}); skipping real-model suite")
        else:
            samples, sr = read_wav(audio_path)
            if sr != 16000:
                print(f"--audio must be 16 kHz (got {sr} Hz); "
                      "skipping real-model suite")
                samples = None
            else:
                if args.expected_text is not None:
                    expected = args.expected_text
                elif audio_path == DEFAULT_AUDIO and GOLDEN_SHORT_TEXT.exists():
                    expected = GOLDEN_SHORT_TEXT.read_text().strip()
                elif audio_path == DEFAULT_AUDIO:
                    expected = DEFAULT_EXPECTED_TEXT
                # else: no expected text — shape-only assertions.
                print(f"Real-model audio: {audio_path} "
                      f"({len(samples) / 16000:.1f}s)")

    base_url = f"http://{args.host}:{args.port}"
    tr = TestResults()
    servers: list[ServeProc] = []
    main_server: ServeProc | None = None

    try:
        # ---- phase 1: transport contract on a placeholder model ----
        if not args.no_server:
            main_server = ServeProc(args.binary, args.model, "/dev/null",
                                    args.host, args.port, eager=False)
            servers.append(main_server)

        print("\nTesting HTTP endpoints (placeholder model):")
        test_health(base_url, tr)
        test_health_alias(base_url, tr)
        test_warmup(base_url, tr)
        test_transcribe_empty(base_url, tr)
        test_cancel_notfound(base_url, tr)
        test_transcribe_wrong_sample_rate(base_url, tr)
        test_transcribe_model_not_loaded(base_url, tr)
        test_transcribe_pcm16_fallback(base_url, tr)
        test_transcribe_wav(base_url, tr)
        test_request_id_passthrough(base_url, tr)
        print("\nTesting multipart WAV uploads (issue #12 regression):")
        for endpoint in ("transcribe", "inference"):
            test_multipart_wav_decodes(base_url, tr, endpoint)
        for endpoint in ("transcribe", "inference"):
            test_multipart_wav_reaches_inference(base_url, tr, endpoint)
        test_malformed_wav_huge_frame_count(base_url, tr)

        print("\nTesting WebSocket control frames:")
        test_websocket_control_frames(args.host, args.port, tr)
        if not args.no_server:
            print("\nTesting WebSocket stream cap (--max-stream-seconds):")
            test_websocket_stream_cap(args.binary, args.model, tr)

        # ---- phase 2: real-model suite ----
        if gguf is not None and samples is not None:
            print(f"\nTesting real-model suite (gguf={gguf}):")
            if args.no_server:
                health = requests.get(f"{base_url}/health", timeout=5).json()
                if not health.get("loaded"):
                    print("  (target server has no loaded model; "
                          "skipping real-model suite)")
                    gguf = None
            else:
                assert main_server is not None
                # Swap the placeholder server for an eager real-model server
                # on the same port.
                main_server.stop()
                servers.remove(main_server)
                main_server = ServeProc(args.binary, args.model, str(gguf),
                                        args.host, args.port, eager=True,
                                        start_timeout=300.0)
                servers.append(main_server)

            if gguf is not None:
                # Long anchor (~1 min): its decode outlasts the queue-wait
                # windows the cancel/timeout tests rely on, even on fast
                # backends with warm graphs.
                anchor = np.tile(samples,
                                 max(1, int(np.ceil(60.0 * 16000 / len(samples)))))
                test_real_roundtrip(base_url, tr, samples, expected)
                for endpoint in ("transcribe", "inference"):
                    test_real_multipart_roundtrip(base_url, tr, samples,
                                                  expected, endpoint)
                test_real_duplicate_request_id(base_url, tr, samples)
                test_real_server_busy(base_url, tr, anchor[: 4 * 16000])
                test_real_cancel_queued(base_url, tr, anchor)
                print("\nTesting WebSocket streaming with real audio:")
                test_real_websocket_flow(args.host, args.port, tr, samples,
                                         expected)
                test_real_queue_timeout(args.binary, args.model, str(gguf),
                                        tr, anchor)
                test_real_idle_timeout_not_firing(args.binary, args.model,
                                                  str(gguf), tr, samples)
        elif gguf is None:
            print("\n(real-model suite skipped: no --gguf)")
        else:
            print("\n(real-model suite skipped: no audio fixture)")
    finally:
        # Always reap every server: wait() under timeout + kill fallback.
        for s in servers:
            try:
                s.stop()
            except Exception as e:
                print(f"  (server stop error: {e})")

    print(f"\n{'='*60}")
    print(f"Results: {tr.passed} passed, {tr.failed} failed")
    if tr.failed:
        print("\nFailures:")
        for e in tr.errors:
            print(f"  - {e}")
        if main_server is not None:
            print("\nServer log tail:")
            main_server.dump_log_tail()
        return 1
    print("All tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
