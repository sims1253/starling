"""HTTP/WebSocket server wrapping the Granite-Speech-4.1-2b mega pipeline.

This server runs as a long-lived local sidecar (for the freestyle Electron app
or any other client) that keeps the model resident in VRAM and exposes a
parakeet-server-compatible interface:

  * ``GET  /``            - health check (any status = alive)
  * ``GET  /health``      - health check alias
  * ``POST /inference``   - multipart WAV upload -> ``{"text": "..."}``
  * ``POST /transcribe``  - raw WAV bytes body -> ``{"text": "..."}`` (alias)
  * ``WS   /stream``      - real-time streaming dictation over WebSocket

A single worker serves one request at a time. Inference is serialised through
a ``threading.Lock`` and the GPU lock
(:mod:`starling.parakeet.gpu_lock`) is taken during each pass to avoid
contention with benchmarks. Concurrent requests are answered with HTTP 503.

Two transport backends are supported and chosen automatically at runtime:

  * **FastAPI + uvicorn** (preferred) when the optional deps are importable.
  * **stdlib-only** (``http.server`` + a minimal RFC 6455 WebSocket) as a
    zero-dependency fallback. The project venv ships torch/CUDA but no web
    framework, so the stdlib path is the one that works out of the box.

The heavy ``torch`` / ``transformers`` / starling imports are deferred to
:meth:`GraniteServer.load` so that ``--help``, app construction, and unit tests
of the audio helpers never touch CUDA.

Run with::

    python -m starling.granite.server --port 8181 --max-chunk-seconds 30
    python -m starling.granite.server --warmup
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import numpy as np

log = logging.getLogger("granite.server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8181
"""Default port (parakeet-server uses 8180; we sit next to it)."""

SAMPLE_RATE: int = 16000
"""Granite-Speech feature extractor sample rate (16 kHz mono)."""

DEFAULT_MAX_CHUNK_SECONDS: float = 30.0
"""Largest chunk transcribed in one shot (bounded by the 640-token KV cache:
30 s of audio ~ 300 audio tokens + 22 chat-template tokens)."""

DEFAULT_MIN_CHUNK_SECONDS: float = 5.0
"""Minimum accumulated audio before the first streaming partial is emitted."""

DEFAULT_PARTIAL_INTERVAL_SECONDS: float = 3.0
"""After the first partial, re-transcribe the growing buffer at most this often
(in wall-clock seconds) to throttle GPU work."""

WARMUP_SECONDS: float = 5.0
"""Length of the silent dummy clip used to capture CUDA graphs at startup."""

DEFAULT_MAX_NEW_TOKENS: int = 200
"""Greedy decode budget per chunk."""

GPU_LOCK_SESSION: str = "granite-server"
GPU_LOCK_MODEL: str = "granite-speech-4.1-2b"
GPU_LOCK_ETA_MIN: int = 1

WS_GUID: bytes = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
def _have_fastapi() -> bool:
    """True iff both ``fastapi`` and ``uvicorn`` are importable."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Audio helpers (numpy / stdlib only - no torch import required at module load)
# ---------------------------------------------------------------------------
def _wav_bytes_to_float32(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV byte string into a ``(N,)`` float32 mono array + sample rate.

    Uses the stdlib ``wave`` module so request handling stays dependency-light.
    """
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, framerate


def _pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Decode raw little-endian 16-bit PCM mono bytes into float32 samples."""
    if len(data) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2 == 1:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _resample_linear(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Cheap linear-interpolation resample so non-16kHz WAVs still work."""
    if sr_in == sr_out or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Server config + state container
# ---------------------------------------------------------------------------
@dataclass
class ServerConfig:
    """Runtime configuration for :class:`GraniteServer`."""

    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS
    min_chunk_seconds: float = DEFAULT_MIN_CHUNK_SECONDS
    partial_interval_seconds: float = DEFAULT_PARTIAL_INTERVAL_SECONDS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    speculative: bool = True
    warmup: bool = False
    encoder_mode: str = "cudagraph"
    use_fused_llm: bool = True
    attn_impl: str = "eager"


@dataclass
class GraniteServer:
    """Owns the loaded model + pipeline and serves transcription requests.

    Heavy imports are deferred to :meth:`load`.
    """

    config: ServerConfig = field(default_factory=ServerConfig)
    pipe: Any = None
    processor: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _busy: bool = False
    _loaded: bool = False

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the model, build the pipeline, and optionally warm up graphs.

        Idempotent: a second call is a no-op. Thread-safe.
        """
        with self._lock:
            if self._loaded:
                return
            from .loader import load_model_and_processor
            from .pipeline import MegaPipeline

            t0 = time.perf_counter()
            log.info("loading Granite-Speech model + processor ...")
            model, processor = load_model_and_processor(attn_impl=self.config.attn_impl)
            pipe = MegaPipeline(
                model,
                processor,
                encoder_mode=self.config.encoder_mode,
                use_fused_llm=self.config.use_fused_llm,
            )
            self.pipe = pipe
            self.processor = processor
            self._loaded = True
            log.info("model loaded in %.1fs", time.perf_counter() - t0)

        if self.config.warmup:
            self.warmup()

    def warmup(self) -> None:
        """Capture CUDA graphs on a short silent clip (no-op if not loaded)."""
        if not self._loaded:
            return
        from ..parakeet.gpu_lock import with_gpu_lock

        log.info("warming up CUDA graphs on %.1fs silent clip ...", WARMUP_SECONDS)
        n = int(WARMUP_SECONDS * SAMPLE_RATE)
        dummy = np.zeros(n, dtype=np.float32)
        with with_gpu_lock(
            session=GPU_LOCK_SESSION,
            model=GPU_LOCK_MODEL,
            eta_min=GPU_LOCK_ETA_MIN,
            note="warmup",
        ):
            self._transcribe_np(dummy)
        log.info("warmup complete")

    # ------------------------------------------------------------------ #
    # inference core (callers acquire the GPU lock)
    # ------------------------------------------------------------------ #
    def _transcribe_np(self, samples: np.ndarray) -> str:
        """Transcribe a 1-D float32 mono numpy array -> transcript text.

        Audio longer than ``max_chunk_seconds`` is split with
        :func:`starling.granite.long_audio.chunk_audio` and the per-chunk texts
        are concatenated. The GPU lock is NOT taken here; callers wrap the call
        so the lock scope stays tight.
        """
        import torch

        from .audio import build_inputs
        from .long_audio import DEFAULT_CHUNK_SECONDS, _join_chunk_texts, chunk_audio

        assert self._loaded and self.pipe is not None and self.processor is not None

        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = torch.from_numpy(np.ascontiguousarray(samples)).float().unsqueeze(0).contiguous()
        sr = SAMPLE_RATE

        max_chunk = self.config.max_chunk_seconds or DEFAULT_CHUNK_SECONDS
        audio_seconds = wav.shape[1] / sr

        if audio_seconds <= max_chunk:
            inputs = build_inputs(self.processor, wav)
            text, _ = self.pipe.transcribe(
                inputs["input_features"],
                inputs["input_ids"],
                inputs.get("input_features_mask"),
                max_new_tokens=self.config.max_new_tokens,
                speculative=self.config.speculative,
            )
            return text

        # Long audio: chunk it.
        texts: list[str] = []
        max_cache_len = int(getattr(self.pipe.llm, "max_cache_len", 640))
        dtype = self.pipe.dtype
        for chunk_wav, _start, _end, _idx in chunk_audio(wav, sr, max_chunk):
            inputs = build_inputs(self.processor, chunk_wav)
            feats = inputs["input_features"].to(dtype)
            ids = inputs["input_ids"]
            mask = inputs.get("input_features_mask")
            prompt_len = int(ids.shape[1])
            budget = max(1, min(self.config.max_new_tokens, max_cache_len - prompt_len - 1))
            text, _ = self.pipe.transcribe(
                feats, ids, mask,
                max_new_tokens=budget,
                speculative=self.config.speculative,
            )
            texts.append(text)
        return _join_chunk_texts(texts, 0.0)

    # ------------------------------------------------------------------ #
    # public async-ish entry points (synchronous; offload by caller)
    # ------------------------------------------------------------------ #
    def transcribe_bytes_sync(self, wav_bytes: bytes) -> str:
        """Decode WAV bytes, acquire GPU + busy locks, and transcribe."""
        self._ensure_loaded()
        samples, sr = _wav_bytes_to_float32(wav_bytes)
        if sr != SAMPLE_RATE:
            samples = _resample_linear(samples, sr, SAMPLE_RATE)
        return self._run_locked_sync(samples)

    def transcribe_pcm_sync(self, pcm16_bytes: bytes) -> str:
        """Decode raw 16-bit PCM mono bytes and transcribe."""
        self._ensure_loaded()
        samples = _pcm16_bytes_to_float32(pcm16_bytes)
        return self._run_locked_sync(samples)

    def _run_locked_sync(self, samples: np.ndarray) -> str:
        """Acquire the per-server busy flag and the GPU lock, then run."""
        from ..parakeet.gpu_lock import GpuLockBusy, acquire_gpu_lock, release_gpu_lock

        with self._lock:
            if self._busy:
                raise _Busy()
            self._busy = True
        try:
            try:
                acquire_gpu_lock(
                    session=GPU_LOCK_SESSION,
                    model=GPU_LOCK_MODEL,
                    eta_min=GPU_LOCK_ETA_MIN,
                    note="inference",
                    wait=False,
                )
            except GpuLockBusy as exc:
                raise _Busy() from exc
            try:
                return self._transcribe_np(samples)
            finally:
                release_gpu_lock()
        finally:
            with self._lock:
                self._busy = False

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


class _Busy(Exception):
    """Internal sentinel: server is busy with another request / GPU locked."""


# ---------------------------------------------------------------------------
# Streaming session (WS /stream)
# ---------------------------------------------------------------------------
@dataclass
class StreamSession:
    """Per-connection rolling audio buffer + streaming state."""

    server: GraniteServer
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    last_partial_ts: float = 0.0
    last_partial_text: str = ""

    def append_pcm(self, pcm16_bytes: bytes) -> None:
        s = _pcm16_bytes_to_float32(pcm16_bytes)
        if s.size > 0:
            self.samples = np.concatenate([self.samples, s]) if self.samples.size else s

    def append_wav(self, wav_bytes: bytes) -> None:
        try:
            s, _sr = _wav_bytes_to_float32(wav_bytes)
        except Exception:
            # Fall back to treating it as raw PCM if WAV parse fails.
            self.append_pcm(wav_bytes)
            return
        if s.size > 0:
            self.samples = np.concatenate([self.samples, s]) if self.samples.size else s

    @property
    def buffered_seconds(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    def should_emit_partial(self, now: float) -> bool:
        """True if we have enough audio AND enough wall-clock has elapsed."""
        if self.buffered_seconds < self.server.config.min_chunk_seconds:
            return False
        if (now - self.last_partial_ts) < self.server.config.partial_interval_seconds:
            return False
        return True

    def reset(self) -> None:
        self.samples = np.zeros(0, dtype=np.float32)
        self.last_partial_text = ""
        self.last_partial_ts = 0.0

    def transcribe_current_sync(self) -> str:
        """Transcribe the entire rolling buffer now (one GPU pass).

        A copy of the samples is taken so a concurrently-arriving chunk cannot
        mutate the array mid-decode.
        """
        snapshot = self.samples.copy()
        return self.server._run_locked_sync(snapshot)


# ===========================================================================
# BACKEND A: FastAPI + uvicorn (preferred, optional deps)
# ===========================================================================
def create_app(config: Optional[ServerConfig] = None) -> Any:
    """Build the FastAPI application bound to a :class:`GraniteServer`.

    Raises ``ImportError`` if fastapi/uvicorn are not installed. The model is
    loaded eagerly at startup (in a worker thread) so the first request is not
    penalised with a ~10s hit.
    """
    from fastapi import (  # type: ignore
        FastAPI,
        HTTPException,
        Request,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.responses import JSONResponse  # type: ignore

    config = config or ServerConfig()
    server = GraniteServer(config=config)
    app = FastAPI(title="granite-speech-server", version="1.0.0")
    app.state.granite_server = server  # type: ignore[attr-defined]

    @app.on_event("startup")
    async def _on_startup() -> None:  # pragma: no cover - exercised by run()
        await asyncio.to_thread(server.load)

    async def _decode_inference_body(request: "Request") -> bytes:
        """Extract the WAV payload from an /inference or /transcribe request.

        Handles both multipart/form-data (freestyle's upload) and a raw WAV
        body, without depending on python-multipart.
        """
        body = await request.body()
        if not body:
            return b""
        ctype = request.headers.get("content-type", "")
        if "multipart/form-data" in ctype:
            return _extract_multipart_payload(body, ctype)
        return body

    # ---------------------------- health ---------------------------- #
    @app.get("/")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": "granite-speech-4.1-2b",
                "loaded": server._loaded,
                "busy": server.is_busy(),
            }
        )

    @app.get("/health")
    async def health_alias() -> JSONResponse:
        return await health()

    # ------------------------ POST /inference ----------------------- #
    async def _inference(request):  # noqa: ANN001 - annotation set below
        payload = await _decode_inference_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="empty upload")
        try:
            text = await asyncio.to_thread(server.transcribe_bytes_sync, payload)
        except _Busy:
            return JSONResponse(status_code=503, content={"error": "server busy", "text": ""})
        return JSONResponse({"text": text})

    # ------------------------ POST /transcribe ---------------------- #
    async def _transcribe(request):  # noqa: ANN001 - annotation set below
        payload = await _decode_inference_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="empty request body")
        try:
            text = await asyncio.to_thread(server.transcribe_bytes_sync, payload)
        except _Busy:
            return JSONResponse(status_code=503, content={"error": "server busy", "text": ""})
        return JSONResponse({"text": text})

    # `from __future__ import annotations` stringifies annotations and the
    # `Request` class is only a local import here, so FastAPI cannot resolve the
    # forward ref. Assign the real class BEFORE registering the route so the
    # dependency scanner treats the param as a request injection.
    _inference.__annotations__["request"] = Request
    _transcribe.__annotations__["request"] = Request
    app.add_api_route("/inference", _inference, methods=["POST"])
    app.add_api_route("/transcribe", _transcribe, methods=["POST"])

    # --------------------------- WS /stream ------------------------- #
    async def _stream(ws):  # noqa: ANN001 - annotation set below
        await ws.accept()
        sess = StreamSession(server=server)
        log.info("WS /stream client connected")
        try:
            while True:
                msg = await ws.receive()
                # ---- text control messages ----
                text_msg = msg.get("text")
                if text_msg is not None:
                    try:
                        cmd = json.loads(text_msg)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "message": "bad json"})
                        continue
                    mtype = cmd.get("type")
                    if mtype == "commit":
                        if sess.buffered_seconds > 0.0:
                            try:
                                text = await asyncio.to_thread(sess.transcribe_current_sync)
                            except _Busy:
                                await ws.send_json({"type": "error", "message": "server busy"})
                                continue
                        else:
                            text = ""
                        await ws.send_json(
                            {"type": "final", "text": text, "duration_s": sess.buffered_seconds}
                        )
                        sess.reset()
                        continue
                    elif mtype == "ping":
                        await ws.send_json({"type": "pong"})
                        continue
                    elif mtype == "reset":
                        sess.reset()
                        await ws.send_json({"type": "reset_ack"})
                        continue
                    else:
                        await ws.send_json({"type": "error", "message": f"unknown type {mtype!r}"})
                        continue

                # ---- binary audio chunks ----
                bdata = msg.get("bytes")
                if not bdata:
                    continue
                if bdata[:4] == b"RIFF" and bdata[8:12] == b"WAVE":
                    sess.append_wav(bdata)
                else:
                    sess.append_pcm(bdata)

                now = time.monotonic()
                if sess.should_emit_partial(now):
                    try:
                        text = await asyncio.to_thread(sess.transcribe_current_sync)
                    except _Busy:
                        continue
                    sess.last_partial_ts = now
                    sess.last_partial_text = text
                    await ws.send_json(
                        {
                            "type": "partial",
                            "text": text,
                            "start_s": 0.0,
                            "end_s": sess.buffered_seconds,
                        }
                    )
        except WebSocketDisconnect:
            log.info("WS /stream client disconnected")
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("WS /stream error: %s", exc)
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # Same forward-ref fix as the POST routes: ``WebSocket`` is a local import
    # here, so assign the real class before registering the route.
    _stream.__annotations__["ws"] = WebSocket
    app.add_api_websocket_route("/stream", _stream)

    return app


# ===========================================================================
# BACKEND B: stdlib-only (http.server + minimal RFC 6455 WebSocket)
# ===========================================================================
def _extract_multipart_payload(body: bytes, content_type: str) -> bytes:
    """Return the file payload from a multipart/form-data body.

    freestyle uploads a single file named ``file``. We split on the boundary,
    skip the part headers, and return the payload bytes of the first (only)
    file part.
    """
    boundary = None
    for tok in content_type.split(";"):
        tok = tok.strip()
        if tok.lower().startswith("boundary="):
            boundary = tok[len("boundary="):].strip().strip('"')
            break
    if not boundary:
        return body
    delim = b"--" + boundary.encode()
    parts = body.split(delim)
    for part in parts:
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        # part starts with \r\n then headers, then \r\n\r\n then payload
        if part.startswith(b"\r\n"):
            part = part[2:]
        # remove trailing \r\n
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if b"\r\n\r\n" in part:
            _headers, payload = part.split(b"\r\n\r\n", 1)
            return payload
        return part
    return body


# ---- minimal RFC 6455 WebSocket framing (server side) ----
def _ws_accept_key(client_key: str) -> str:
    h = hashlib.sha1(client_key.encode() + WS_GUID).digest()
    return base64.b64encode(h).decode()


def _ws_read_frame(rfile) -> tuple[int, bytes]:
    """Read one WebSocket frame from ``rfile``. Returns ``(opcode, payload)``.

    Handles fragmentation (continuation frames) and control frames (ping/pong/
    close) transparently. Client->server frames must be masked per RFC 6455.
    """
    def _read_exact(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = rfile.read(n - len(buf))
            if not chunk:
                raise ConnectionError("websocket closed mid-frame")
            buf.extend(chunk)
        return bytes(buf)

    pieces: list[bytes] = []
    final_opcode = 0x1
    while True:
        hdr = _read_exact(2)
        b0, b1 = hdr[0], hdr[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", _read_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _read_exact(8))[0]
        mask = _read_exact(4) if masked else b""
        payload = _read_exact(length)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

        # Control frames are not fragmented and are handled inline.
        if opcode == 0x8:  # close
            raise ConnectionError("client closed")
        if opcode == 0x9:  # ping -> pong handled by caller loop; return as-is
            return 0x9, payload
        if opcode == 0xA:  # pong
            continue

        if opcode in (0x1, 0x2):
            final_opcode = opcode
            pieces.append(payload)
        elif opcode == 0x0:
            pieces.append(payload)
        else:
            raise ConnectionError(f"unknown ws opcode {opcode}")

        if fin:
            return final_opcode, b"".join(pieces)


def _ws_write_frame(wfile, opcode: int, payload: bytes) -> None:
    """Write a single (unmasked, FIN) server->client WebSocket frame."""
    b0 = 0x80 | (opcode & 0x0F)
    n = len(payload)
    if n < 126:
        header = struct.pack(">BB", b0, n)
    elif n < 65536:
        header = struct.pack(">BBH", b0, 126, n)
    else:
        header = struct.pack(">BBQ", b0, 127, n)
    wfile.write(header + payload)
    wfile.flush()


def _ws_send_json(wfile, obj: dict) -> None:
    _ws_write_frame(wfile, 0x1, json.dumps(obj).encode())


def _ws_send_pong(wfile, payload: bytes) -> None:
    _ws_write_frame(wfile, 0xA, payload)


def _serve_stream_session(
    rfile, wfile, server: GraniteServer, client_addr: tuple
) -> None:
    """Drive a single WS /stream connection (blocking, runs in a worker thread)."""
    sess = StreamSession(server=server)
    log.info("WS /stream client connected from %s", client_addr)
    try:
        while True:
            try:
                opcode, payload = _ws_read_frame(rfile)
            except ConnectionError:
                break

            if opcode == 0x9:  # ping
                _ws_send_pong(wfile, payload)
                continue
            if opcode == 0x1:  # text -> control JSON
                try:
                    cmd = json.loads(payload.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    _ws_send_json(wfile, {"type": "error", "message": "bad json"})
                    continue
                mtype = cmd.get("type")
                if mtype == "commit":
                    if sess.buffered_seconds > 0.0:
                        try:
                            text = sess.transcribe_current_sync()
                        except _Busy:
                            _ws_send_json(wfile, {"type": "error", "message": "server busy"})
                            continue
                    else:
                        text = ""
                    _ws_send_json(
                        wfile,
                        {"type": "final", "text": text, "duration_s": sess.buffered_seconds},
                    )
                    sess.reset()
                    continue
                elif mtype == "ping":
                    _ws_send_json(wfile, {"type": "pong"})
                    continue
                elif mtype == "reset":
                    sess.reset()
                    _ws_send_json(wfile, {"type": "reset_ack"})
                    continue
                else:
                    _ws_send_json(wfile, {"type": "error", "message": f"unknown type {mtype!r}"})
                    continue
            # opcode == 0x2 binary -> audio chunk
            if payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
                sess.append_wav(payload)
            else:
                sess.append_pcm(payload)

            now = time.monotonic()
            if sess.should_emit_partial(now):
                try:
                    text = sess.transcribe_current_sync()
                except _Busy:
                    continue
                sess.last_partial_ts = now
                sess.last_partial_text = text
                _ws_send_json(
                    wfile,
                    {
                        "type": "partial",
                        "text": text,
                        "start_s": 0.0,
                        "end_s": sess.buffered_seconds,
                    },
                )
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("WS /stream error: %s", exc)
        try:
            _ws_send_json(wfile, {"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        log.info("WS /stream client %s disconnected", client_addr)


def _build_stdlib_handler(server: GraniteServer):
    """Return a BaseHTTPRequestHandler subclass bound to ``server``."""

    class _Handler(BaseHTTPRequestHandler):
        # Quieter default logging.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            log.debug("http %s - %s", self.address_string(), fmt % args)

        server_version = "granite-speech-server/1.0"
        protocol_version = "HTTP/1.1"

        # -------- helpers --------
        def _send_json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # -------- inference / transcribe --------
        def do_POST(self) -> None:  # noqa: N802 - stdlib API
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            body = self.rfile.read(length) if length > 0 else b""

            if self.path == "/inference":
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    payload = _extract_multipart_payload(body, ctype)
                else:
                    # Treat the raw body as a WAV (curl --data-binary).
                    payload = body
                if not payload:
                    self._send_json(400, {"error": "empty upload", "text": ""})
                    return
                try:
                    text = server.transcribe_bytes_sync(payload)
                except _Busy:
                    self._send_json(503, {"error": "server busy", "text": ""})
                    return
                self._send_json(200, {"text": text})
                return

            if self.path == "/transcribe":
                if not body:
                    self._send_json(400, {"error": "empty body", "text": ""})
                    return
                try:
                    text = server.transcribe_bytes_sync(body)
                except _Busy:
                    self._send_json(503, {"error": "server busy", "text": ""})
                    return
                self._send_json(200, {"text": text})
                return

            self._send_json(404, {"error": "not found"})

        # -------- WebSocket upgrade (GET with Upgrade header) --------
        def do_GET_ws(self) -> bool:
            """If this is a WebSocket upgrade request, handle it and return True."""
            upgrade = self.headers.get("Upgrade", "").lower()
            if upgrade != "websocket" or self.path != "/stream":
                return False
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self.send_response(400)
                self.end_headers()
                return True
            accept = _ws_accept_key(key)
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            # Hand the raw socket to the streaming driver. The handler's rfile
            # is buffered; flush any pending write then serve.
            _serve_stream_session(self.rfile, self.wfile, server, self.client_address)
            return True

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            if self.do_GET_ws():
                return
            if self.path in ("/", "/health"):
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": "granite-speech-4.1-2b",
                        "loaded": server._loaded,
                        "busy": server.is_busy(),
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

    return _Handler


def _run_stdlib_server(server: GraniteServer, host: str, port: int) -> None:
    """Run the stdlib ThreadingHTTPServer forever (blocking)."""
    handler_cls = _build_stdlib_handler(server)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    httpd.daemon_threads = True
    log.info("stdlib server listening on %s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ===========================================================================
# CLI
# ===========================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m starling.granite.server",
        description="Granite-Speech-4.1-2b streaming ASR server (parakeet-server compatible).",
    )
    p.add_argument("--host", default=DEFAULT_HOST, help=f"bind host (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"bind port (default {DEFAULT_PORT})")
    p.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=DEFAULT_MAX_CHUNK_SECONDS,
        help=f"max audio chunk length per transcription (default {DEFAULT_MAX_CHUNK_SECONDS}s)",
    )
    p.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=DEFAULT_MIN_CHUNK_SECONDS,
        help=(
            "minimum buffered audio before the first WS /stream partial "
            f"(default {DEFAULT_MIN_CHUNK_SECONDS}s)"
        ),
    )
    p.add_argument(
        "--partial-interval-seconds",
        type=float,
        default=DEFAULT_PARTIAL_INTERVAL_SECONDS,
        help=(
            "minimum wall-clock gap between WS /stream partials "
            f"(default {DEFAULT_PARTIAL_INTERVAL_SECONDS}s)"
        ),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"greedy decode budget per chunk (default {DEFAULT_MAX_NEW_TOKENS})",
    )
    p.add_argument(
        "--no-speculative",
        action="store_true",
        help="disable self-speculative decoding (slower but avoids CTC draft setup)",
    )
    p.add_argument(
        "--encoder-mode",
        default="cudagraph",
        choices=["cudagraph", "eager", "compile", "triton"],
        help="fused encoder mode (default cudagraph)",
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        help="global attention implementation (default eager; q-former requires eager)",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="pre-capture CUDA graphs on a silent dummy clip at startup",
    )
    p.add_argument(
        "--no-eager-load",
        action="store_true",
        help="do not load the model at startup; load lazily on first request instead",
    )
    p.add_argument(
        "--stdlib",
        action="store_true",
        help="force the stdlib-only backend even if FastAPI/uvicorn are available",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="logging level (default info)",
    )
    return p


def run(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Loads the model, builds the app, and serves forever."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = ServerConfig(
        max_chunk_seconds=args.max_chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        partial_interval_seconds=args.partial_interval_seconds,
        max_new_tokens=args.max_new_tokens,
        speculative=not args.no_speculative,
        warmup=args.warmup,
        encoder_mode=args.encoder_mode,
        use_fused_llm=True,
        attn_impl=args.attn_impl,
    )

    use_fastapi = (not args.stdlib) and _have_fastapi()

    # Load the model eagerly (in the main thread) before serving so the first
    # request is fast. With --no-eager-load the GraniteServer loads lazily on
    # the first request instead.
    server = GraniteServer(config=config)
    if not args.no_eager_load:
        server.load()

    log.info(
        "starting granite-speech server on %s:%d (backend=%s, speculative=%s, warmup=%s)",
        args.host,
        args.port,
        "fastapi" if use_fastapi else "stdlib",
        config.speculative,
        config.warmup,
    )

    if use_fastapi:
        import uvicorn

        app = create_app(config)
        # The standalone ``server`` already loaded the model; reuse it by
        # replacing the app's lazily-constructed server with ours.
        app.state.granite_server = server  # type: ignore[attr-defined]
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    else:
        _run_stdlib_server(server, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
