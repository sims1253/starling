"""Stdlib-only test client for the Granite-Speech streaming ASR server.

Exercises both interfaces:

1. ``POST /inference``  - one-shot WAV upload -> text
2. ``WS   /stream``     - chunked streaming dictation -> partial + final

The script loads a WAV file (or synthesises a short tone if none is given),
uploads it to ``/inference``, and then replays it in ~``chunk-ms`` PCM chunks
over the WebSocket while printing partial and final results.

Zero third-party deps: HTTP uses :mod:`http.client`, and the WebSocket client
is a tiny RFC 6455 implementation on a raw socket. This matches the server's
own zero-dependency stdlib backend, so the client runs in the project venv
(which has torch/CUDA but no ``requests``/``websockets``).

Usage::

    python scripts/test_server.py --port 8181 --wav path/to/audio.wav
    python scripts/test_server.py --port 8181 --no-stream     # HTTP only
    python scripts/test_server.py --chunk-ms 500              # streaming only
    python scripts/test_server.py --health-only
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import socket
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Add src/ to sys.path so `import starling...` works when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from starling.granite.server import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    SAMPLE_RATE,
)

CHUNK_MS_DEFAULT: int = 500
WS_GUID: bytes = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ---------------------------------------------------------------------------
# Audio loading / encoding (numpy + stdlib wave)
# ---------------------------------------------------------------------------
def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file into a ``(N,)`` float32 mono numpy array + sample rate."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, framerate


def synth_tone(seconds: float = 4.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesise an amplitude-modulated 440Hz tone fallback."""
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 440.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t))).astype(np.float32)


def to_wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Encode float32 mono samples as 16-bit PCM WAV bytes."""
    import io

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Encode float32 mono samples as raw 16-bit little-endian PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# HTTP client (stdlib http.client)
# ---------------------------------------------------------------------------
def test_health(host: str, port: int) -> None:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/")
        res = conn.getresponse()
        body = res.read().decode(errors="replace").strip()
        print(f"[http] GET http://{host}:{port}/ -> {res.status} {res.reason} {body}")
    except Exception as exc:
        print(f"[http] health check failed: {exc}")
    finally:
        conn.close()


def _build_multipart(filename: str, wav_bytes: bytes) -> tuple[bytes, str]:
    """Build a multipart/form-data body for a single file upload.

    Returns ``(body, content_type_header)``.
    """
    boundary = "----granite-test-" + hashlib.sha1(os.urandom(16)).hexdigest()
    crlf = b"\r\n"
    parts: list[bytes] = []
    parts.append(b"--" + boundary.encode())
    parts.append(crlf + b'Content-Disposition: form-data; name="file"; filename="'
                 + filename.encode() + b'"' + crlf)
    parts.append(b"Content-Type: audio/wav" + crlf + crlf)
    parts.append(wav_bytes)
    parts.append(crlf + b"--" + boundary.encode() + b"--" + crlf)
    body = b"".join(parts)
    ctype = f"multipart/form-data; boundary={boundary}"
    return body, ctype


def test_inference(host: str, port: int, wav_bytes: bytes) -> None:
    body, ctype = _build_multipart("audio.wav", wav_bytes)
    conn = http.client.HTTPConnection(host, port, timeout=120)
    print(f"\n[http] POST http://{host}:{port}/inference  ({len(wav_bytes)} bytes)")
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/inference", body=body, headers={"Content-Type": ctype})
        res = conn.getresponse()
        ms = (time.perf_counter() - t0) * 1000.0
        text = res.read().decode(errors="replace")
        print(f"[http] status={res.status}  {ms:.1f}ms")
        if res.status == 503:
            print(f"[http] server busy: {text}")
            return
        if res.status != 200:
            print(f"[http] error: {text}")
            return
        data = json.loads(text)
        print(f"[http] text: {data.get('text', '')!r}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Minimal RFC 6455 WebSocket client (stdlib socket)
# ---------------------------------------------------------------------------
def _ws_connect(host: str, port: int, path: str) -> socket.socket:
    """Perform the WebSocket handshake and return the upgraded socket."""
    ws_handshake_nonce = ****************(os.urandom(16)).decode()
    sock = socket.create_connection((host, port), timeout=10)
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_handshake_nonce}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    ).encode()
    sock.sendall(req)

    # Read the HTTP/101 response headers.
    rfile = sock.makefile("rb")
    status_line = rfile.readline().decode(errors="replace").strip()
    if " 101 " not in status_line:
        rest = rfile.read(4096)
        sock.close()
        raise ConnectionError(f"websocket upgrade failed: {status_line!r} rest={rest!r}")
    headers: dict[str, str] = {}
    while True:
        line = rfile.readline().decode(errors="replace").strip()
        if not line:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    expected = base64.b64encode(hashlib.sha1(key.encode() + WS_GUID).digest()).decode()
    if headers.get("sec-websocket-accept") != expected:
        sock.close()
        raise ConnectionError("bad Sec-WebSocket-Accept")
    return sock


def _ws_send(sock: socket.socket, opcode: int, payload: bytes) -> None:
    """Send a masked client->server WebSocket frame."""
    b0 = 0x80 | (opcode & 0x0F)
    mask = os.urandom(4)
    n = len(payload)
    if n < 126:
        header = struct.pack(">BB", b0, 0x80 | n)
    elif n < 65536:
        header = struct.pack(">BBH", b0, 0x80 | 126, n)
    else:
        header = struct.pack(">BBQ", b0, 0x80 | 127, n)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    sock.sendall(header + mask + masked)


def _ws_recv(sock: socket.socket) -> tuple[int, bytes]:
    """Read one (unmasked) server->client frame -> ``(opcode, payload)``.

    Handles fragmentation and control frames (ping/pong) inline.
    """
    def _read_exact(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("websocket closed")
            buf.extend(chunk)
        return bytes(buf)

    pieces: list[bytes] = []
    final_opcode = 0x1
    while True:
        hdr = _read_exact(2)
        b0, b1 = hdr[0], hdr[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", _read_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _read_exact(8))[0]
        masked = bool(b1 & 0x80)
        mask = _read_exact(4) if masked else b""
        payload = _read_exact(length)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:  # close
            raise ConnectionError("server closed")
        if opcode == 0x9:  # ping -> respond pong, keep reading
            _ws_send(sock, 0xA, payload)
            continue
        if opcode == 0xA:  # pong
            continue
        if opcode in (0x1, 0x2):
            final_opcode = opcode
            pieces.append(payload)
        elif opcode == 0x0:
            pieces.append(payload)
        else:
            raise ConnectionError(f"unknown opcode {opcode}")
        if fin:
            return final_opcode, b"".join(pieces)


def test_stream(host: str, port: int, samples: np.ndarray, sr: int, chunk_ms: int) -> None:
    """Open WS /stream, replay audio in chunks, and print partial/final results."""
    print(f"\n[ws]   connect ws://{host}:{port}/stream  "
          f"({len(samples)/sr:.1f}s audio, {chunk_ms}ms chunks)")
    pcm = to_pcm16_bytes(samples)
    chunk_bytes = int(SAMPLE_RATE * chunk_ms / 1000.0) * 2  # 2 bytes/sample
    n_chunks = max(1, (len(pcm) + chunk_bytes - 1) // chunk_bytes)

    sock = _ws_connect(host, port, "/stream")
    sock.settimeout(120.0)
    t0 = time.perf_counter()
    try:
        for i in range(n_chunks):
            chunk = pcm[i * chunk_bytes : (i + 1) * chunk_bytes]
            _ws_send(sock, 0x2, chunk)  # binary frame
            # Non-blocking drain: print any partials the server sent back.
            sock.settimeout(0.01)
            try:
                while True:
                    opcode, payload = _ws_recv(sock)
                    if opcode != 0x1:
                        continue
                    try:
                        msg = json.loads(payload.decode())
                    except Exception:
                        continue
                    _print_ws_msg(msg, t0)
            except (BlockingIOError, socket.timeout):
                pass
            # Pace the send to simulate real-time mic input.
            time.sleep(chunk_ms / 1000.0)

        # Signal end-of-stream.
        _ws_send(sock, 0x1, json.dumps({"type": "commit"}).encode())
        print("[ws]   sent commit")

        # Drain remaining messages until we see final.
        sock.settimeout(60.0)
        try:
            while True:
                opcode, payload = _ws_recv(sock)
                if opcode != 0x1:
                    continue
                try:
                    msg = json.loads(payload.decode())
                except Exception:
                    continue
                if _print_ws_msg(msg, t0):
                    return
        except (socket.timeout, ConnectionError):
            print("[ws]   timed out / closed waiting for final")
    finally:
        try:
            _ws_send(sock, 0x8, b"")  # close frame
        except Exception:
            pass
        sock.close()


def _print_ws_msg(msg: dict, t0: float) -> bool:
    """Print a WS message; return True if it is the terminal ``final``."""
    mtype = msg.get("type")
    if mtype == "partial":
        print(
            f"[ws]   partial  [{msg.get('start_s', 0):.1f}-"
            f"{msg.get('end_s', 0):.1f}s]: {msg.get('text', '')!r}"
        )
    elif mtype == "final":
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"[ws]   FINAL    ({ms:.1f}ms): {msg.get('text', '')!r}")
        return True
    elif mtype == "error":
        print(f"[ws]   ERROR: {msg.get('message')}")
    elif mtype == "pong":
        pass
    else:
        print(f"[ws]   {mtype}: {msg}")
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stdlib test client for the Granite-Speech streaming ASR server."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--wav", default=None, help="path to a WAV file (else a tone is synthesised)")
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS_DEFAULT, help="streaming chunk size in ms")
    parser.add_argument("--no-http", action="store_true", help="skip POST /inference")
    parser.add_argument("--no-stream", action="store_true", help="skip WS /stream")
    parser.add_argument("--health-only", action="store_true", help="only run the health check")
    args = parser.parse_args()

    # ---- resolve audio ----
    if args.wav:
        samples, sr = load_wav(args.wav)
        print(f"[load] {args.wav}: {len(samples)/sr:.1f}s @ {sr}Hz")
    else:
        sr = SAMPLE_RATE
        samples = synth_tone(seconds=4.0)
        print(f"[load] synthesised 4.0s tone @ {sr}Hz (pass --wav for real audio)")
    wav_bytes = to_wav_bytes(samples, sr)

    # ---- health check ----
    test_health(args.host, args.port)
    if args.health_only:
        return 0

    # ---- HTTP /inference ----
    if not args.no_http:
        try:
            test_inference(args.host, args.port, wav_bytes)
        except Exception as exc:
            print(f"[http] FAILED: {exc}")

    # ---- WS /stream ----
    if not args.no_stream:
        try:
            test_stream(args.host, args.port, samples, sr, args.chunk_ms)
        except Exception as exc:
            print(f"[ws]   FAILED: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
