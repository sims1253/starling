"""HTTP/WebSocket server wrapping the Granite-Speech-4.1-2b mega pipeline.

This server runs as a long-lived local sidecar (for the freestyle Electron app
or any other client) that keeps the model resident in VRAM and exposes a
parakeet-server-compatible interface:

  * ``GET  /``                - health check (any status = alive)
  * ``GET  /health``          - health check alias (also reports ``phase`` and
                                ``queue_depth`` so clients can render load state)
  * ``POST /inference``       - multipart/raw WAV upload -> ``{text, segments,
                                duration_s, request_id}``
  * ``POST /transcribe``      - raw WAV bytes body -> same shape as /inference
  * ``POST /warmup``          - pre-capture CUDA graphs on a silent clip
                                (idempotent; 202 Accepted while it runs)
  * ``DELETE /inference/<id>``- cancel a queued request by its ``X-Request-Id``
  * ``WS   /stream``          - real-time streaming dictation over WebSocket

Response shape. ``POST /inference`` returns chunk-level segment timestamps:

    {"text": "...", "duration_s": 12.3,
     "segments": [{"text": "...", "start_s": 0.0, "end_s": 12.3}]}

Granite-Speech has no per-token audio alignment (the LLM decodes text with no
duration head), so segments are at ``max_chunk_seconds`` granularity — one per
chunk window — rather than per-word. Shrink ``--max-chunk-seconds`` for finer
segments at the cost of more decode passes.

Request handling. A single GPU worker serves one request at a time. Inference
is serialised through the process-wide GPU lock
(:mod:`starling.parakeet.gpu_lock`); concurrent requests queue (up to
``MAX_WAITERS``) instead of being rejected, and are rejected with HTTP 503 only
once the queue is full. A client that supplies ``X-Request-Id`` on a POST may
``DELETE /inference/<id>`` to drop it from the queue. Cancellation is
best-effort for a request already on the GPU: CUDA-graph replays are not
preemptible, so an in-flight request still finishes its current ~5-16ms decode
step, but won't be returned (HTTP 499).

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

MAX_WAITERS: int = 8
"""Max requests waiting for the single GPU worker before we reject with HTTP
503 backpressure. Past this depth the client must retry rather than pile up."""

CANCEL_POLL_SECONDS: float = 0.1
"""How often a queued request checks its cancel event while waiting for the GPU."""

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
    _loaded: bool = False

    # --- request queueing -------------------------------------------------
    # The GPU can serve exactly one request at a time. Instead of answering
    # concurrent requests with HTTP 503 and forcing the client to retry-storm,
    # we serialise them here: at most one runs (guarded by the process-wide
    # ``acquire_gpu_lock``) while up to ``MAX_WAITERS`` others block waiting for
    # it. Each request carries a cancel event so a client abort
    # (DELETE /inference/<id>) can pull it from the queue. ``_n_waiters`` counts
    # everyone who has been admitted but not yet finished (including the
    # running one) and drives backpressure.
    _n_waiters: int = 0
    _requests: dict[str, "RequestContext"] = field(default_factory=dict)

    # --- lifecycle phase (reported by /health) ----------------------------
    # Coarse progress hint while the server is not yet ready, so clients can
    # render a meaningful status instead of a bare "not loaded". Transitions:
    #   unloaded -> loading_weights -> warming_up -> ready
    _phase: str = "unloaded"
    """One of ``unloaded``/``loading_weights``/``warming_up``/``ready``."""

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
            self._phase = "loading_weights"
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
            self._phase = "loaded"
            log.info("model loaded in %.1fs", time.perf_counter() - t0)

        if self.config.warmup:
            self.warmup()
        else:
            self._phase = "ready"

    def warmup(self) -> None:
        """Capture CUDA graphs on a short silent clip (no-op if not loaded)."""
        if not self._loaded:
            return
        from ..parakeet.gpu_lock import with_gpu_lock

        self._phase = "warming_up"
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
        self._phase = "ready"
        log.info("warmup complete")

    # ------------------------------------------------------------------ #
    # inference core (callers acquire the GPU lock)
    # ------------------------------------------------------------------ #
    def _transcribe_np(self, samples: np.ndarray) -> "TranscribeResult":
        """Transcribe a 1-D float32 mono numpy array -> transcript result.

        Audio longer than ``max_chunk_seconds`` is split with
        :func:`starling.granite.long_audio.chunk_audio` and the per-chunk texts
        are concatenated. The GPU lock is NOT taken here; callers wrap the call
        so the lock scope stays tight.

        Returns text plus chunk-level segment timestamps. The Granite-Speech
        architecture has no per-token audio alignment (the LLM decodes text with
        no duration head), so segments are at ``max_chunk_seconds`` granularity
        — one per chunk window — rather than per-word.
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
            return TranscribeResult(
                text=text,
                segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
                duration_s=audio_seconds,
            )

        # Long audio: chunk it. Track per-chunk (start_s, end_s) so callers can
        # align segment text against the source timeline.
        texts: list[str] = []
        segments: list[dict[str, Any]] = []
        max_cache_len = int(getattr(self.pipe.llm, "max_cache_len", 640))
        dtype = self.pipe.dtype
        for chunk_wav, start, end, _idx in chunk_audio(wav, sr, max_chunk):
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
            segments.append({"text": text, "start_s": start, "end_s": end})
        joined = _join_chunk_texts(texts, 0.0)
        return TranscribeResult(
            text=joined,
            segments=segments,
            duration_s=audio_seconds,
        )

    # ------------------------------------------------------------------ #
    # public entry points (synchronous; offload by caller)
    # ------------------------------------------------------------------ #
    def transcribe_bytes_sync(
        self, wav_bytes: bytes, request_id: Optional[str] = None
    ) -> "TranscribeResult":
        """Decode WAV bytes, queue for the GPU worker, and transcribe."""
        self._ensure_loaded()
        samples, sr = _wav_bytes_to_float32(wav_bytes)
        if sr != SAMPLE_RATE:
            samples = _resample_linear(samples, sr, SAMPLE_RATE)
        return self._run_queued_sync(samples, request_id)

    def transcribe_pcm_sync(
        self, pcm16_bytes: bytes, request_id: Optional[str] = None
    ) -> "TranscribeResult":
        """Decode raw 16-bit PCM mono bytes and transcribe."""
        self._ensure_loaded()
        samples = _pcm16_bytes_to_float32(pcm16_bytes)
        return self._run_queued_sync(samples, request_id)

    def _run_queued_sync(
        self, samples: np.ndarray, request_id: Optional[str]
    ) -> "TranscribeResult":
        """Admit to the bounded request queue, then run on the GPU worker.

        Up to ``MAX_WAITERS`` requests may be admitted; beyond that we reject
        with :class:`_Busy` so the client backs off rather than piling up. The
        actual GPU serialisation happens in :meth:`_serial_run` via the
        process-wide GPU lock; ``_n_waiters`` here is pure backpressure
        accounting.
        """
        ctx = RequestContext(request_id)
        with self._lock:
            if self._n_waiters >= MAX_WAITERS:
                raise _Busy()
            self._n_waiters += 1
            self._requests[ctx.id] = ctx
        try:
            return self._serial_run(ctx, samples)
        finally:
            with self._lock:
                self._n_waiters = max(0, self._n_waiters - 1)
                self._requests.pop(ctx.id, None)

    def _serial_run(
        self, ctx: "RequestContext", samples: np.ndarray
    ) -> "TranscribeResult":
        """Block until this is the sole GPU worker, then transcribe."""
        from ..parakeet.gpu_lock import GpuLockBusy, acquire_gpu_lock, release_gpu_lock

        # Wait for the GPU lock, polling for cancellation while queued.
        while True:
            if ctx.cancel.is_set():
                raise _Cancelled()
            try:
                acquire_gpu_lock(
                    session=GPU_LOCK_SESSION,
                    model=GPU_LOCK_MODEL,
                    eta_min=GPU_LOCK_ETA_MIN,
                    note="inference",
                    wait=False,
                )
                break
            except GpuLockBusy:
                if ctx.cancel.wait(CANCEL_POLL_SECONDS):
                    raise _Cancelled()

        ctx.state = "running"
        try:
            return self._transcribe_np(samples)
        finally:
            release_gpu_lock()

    # ------------------------------------------------------------------ #
    # request registry (cancellation + introspection)
    # ------------------------------------------------------------------ #
    def cancel_request(self, request_id: str) -> bool:
        """Signal cancellation of a queued or running request by id.

        Returns True if a request with that id was found. A queued request is
        dropped promptly; a running request cannot be preempted mid CUDA-graph
        replay (graphs are not interruptible), so cancellation is best-effort
        for the in-flight case and mainly saves queuing latency.
        """
        with self._lock:
            ctx = self._requests.get(request_id)
        if ctx is None:
            return False
        ctx.cancel.set()
        return True

    def queue_depth(self) -> int:
        """Number of requests waiting for the GPU worker (excludes running)."""
        with self._lock:
            return max(0, self._n_waiters - 1)

    def is_busy(self) -> bool:
        with self._lock:
            return self._n_waiters > 0

    def phase(self) -> str:
        with self._lock:
            return self._phase

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


class _Busy(Exception):
    """Internal sentinel: queue is full (backpressure) — client should retry."""


class _Cancelled(Exception):
    """Internal sentinel: request was cancelled via :meth:`cancel_request`."""


@dataclass
class TranscribeResult:
    """Transcription output with optional chunk-level timestamps."""

    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "segments": self.segments,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class RequestContext:
    """Per-request handle tracked in the server's registry for cancellation."""

    id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    state: str = "queued"


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

    def transcribe_current_sync(self) -> "TranscribeResult":
        """Transcribe the entire rolling buffer now (one GPU pass).

        A copy of the samples is taken so a concurrently-arriving chunk cannot
        mutate the array mid-decode.
        """
        snapshot = self.samples.copy()
        return self.server._run_queued_sync(snapshot, None)


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

    def _request_id(request: "Request") -> Optional[str]:
        """Client-supplied request id for cancellation, or None."""
        rid = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
        return rid or None

    # ---------------------------- health ---------------------------- #
    @app.get("/")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": "granite-speech-4.1-2b",
                "loaded": server._loaded,
                "busy": server.is_busy(),
                "phase": server.phase(),
                "queue_depth": server.queue_depth(),
            }
        )

    @app.get("/health")
    async def health_alias() -> JSONResponse:
        return await health()

    # --------------------------- POST /warmup ----------------------- #
    @app.post("/warmup")
    async def warmup_route() -> JSONResponse:
        """Pre-capture CUDA graphs on a silent clip (idempotent, async)."""
        await asyncio.to_thread(server.warmup)
        return JSONResponse({"status": "ok", "phase": server.phase()})

    # ------------------------ POST /inference ----------------------- #
    async def _inference(request):  # noqa: ANN001 - annotation set below
        payload = await _decode_inference_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="empty upload")
        rid = _request_id(request)
        try:
            result = await asyncio.to_thread(server.transcribe_bytes_sync, payload, rid)
        except _Busy:
            return JSONResponse(
                status_code=503,
                content={"error": "server busy", "text": "", "queue_depth": server.queue_depth()},
            )
        except _Cancelled:
            return JSONResponse(status_code=499, content={"error": "cancelled", "text": ""})
        resp = result.to_dict()
        resp["request_id"] = rid
        return JSONResponse(resp)

    # ------------------------ POST /transcribe ---------------------- #
    async def _transcribe(request):  # noqa: ANN001 - annotation set below
        payload = await _decode_inference_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="empty request body")
        rid = _request_id(request)
        try:
            result = await asyncio.to_thread(server.transcribe_bytes_sync, payload, rid)
        except _Busy:
            return JSONResponse(
                status_code=503,
                content={"error": "server busy", "text": "", "queue_depth": server.queue_depth()},
            )
        except _Cancelled:
            return JSONResponse(status_code=499, content={"error": "cancelled", "text": ""})
        resp = result.to_dict()
        resp["request_id"] = rid
        return JSONResponse(resp)

    # --------------------- DELETE /inference/{id} ------------------- #
    async def _abort(request):  # noqa: ANN001 - annotation set below
        rid = request.path_params.get("id")
        if not rid:
            return JSONResponse(status_code=400, content={"error": "missing request id"})
        cancelled = await asyncio.to_thread(server.cancel_request, str(rid))
        return JSONResponse(
            {"status": "cancelled" if cancelled else "not_found", "request_id": rid},
            status_code=200 if cancelled else 404,
        )

    # `from __future__ import annotations` stringifies annotations and the
    # `Request` class is only a local import here, so FastAPI cannot resolve the
    # forward ref. Assign the real class BEFORE registering the route so the
    # dependency scanner treats the param as a request injection.
    _inference.__annotations__["request"] = Request
    _transcribe.__annotations__["request"] = Request
    _abort.__annotations__["request"] = Request
    app.add_api_route("/inference", _inference, methods=["POST"])
    app.add_api_route("/transcribe", _transcribe, methods=["POST"])
    app.add_api_route("/inference/{id}", _abort, methods=["DELETE"])

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
                                result = await asyncio.to_thread(sess.transcribe_current_sync)
                            except _Busy:
                                await ws.send_json({"type": "error", "message": "server busy"})
                                continue
                            except _Cancelled:
                                await ws.send_json({"type": "error", "message": "cancelled"})
                                continue
                        else:
                            result = TranscribeResult(text="")
                        await ws.send_json(
                            {
                                "type": "final",
                                "text": result.text,
                                "segments": result.segments,
                                "duration_s": round(sess.buffered_seconds, 3),
                            }
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
                        result = await asyncio.to_thread(sess.transcribe_current_sync)
                    except _Busy:
                        continue
                    except _Cancelled:
                        continue
                    sess.last_partial_ts = now
                    sess.last_partial_text = result.text
                    await ws.send_json(
                        {
                            "type": "partial",
                            "text": result.text,
                            "segments": result.segments,
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
                            result = sess.transcribe_current_sync()
                        except _Busy:
                            _ws_send_json(wfile, {"type": "error", "message": "server busy"})
                            continue
                        except _Cancelled:
                            _ws_send_json(wfile, {"type": "error", "message": "cancelled"})
                            continue
                    else:
                        result = TranscribeResult(text="")
                    _ws_send_json(
                        wfile,
                        {
                            "type": "final",
                            "text": result.text,
                            "segments": result.segments,
                            "duration_s": round(sess.buffered_seconds, 3),
                        },
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
                    result = sess.transcribe_current_sync()
                except _Busy:
                    continue
                except _Cancelled:
                    continue
                sess.last_partial_ts = now
                sess.last_partial_text = result.text
                _ws_send_json(
                    wfile,
                    {
                        "type": "partial",
                        "text": result.text,
                        "segments": result.segments,
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

            if self.path == "/warmup":
                threading.Thread(target=server.warmup, daemon=True).start()
                self._send_json(202, {"status": "warmup started", "phase": server.phase()})
                return

            rid = self.headers.get("X-Request-Id") or self.headers.get("X-Correlation-Id")

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
                    result = server.transcribe_bytes_sync(payload, rid)
                except _Busy:
                    self._send_json(
                        503,
                        {"error": "server busy", "text": "", "queue_depth": server.queue_depth()},
                    )
                    return
                except _Cancelled:
                    self._send_json(499, {"error": "cancelled", "text": ""})
                    return
                resp = result.to_dict()
                resp["request_id"] = rid
                self._send_json(200, resp)
                return

            if self.path == "/transcribe":
                if not body:
                    self._send_json(400, {"error": "empty body", "text": ""})
                    return
                try:
                    result = server.transcribe_bytes_sync(body, rid)
                except _Busy:
                    self._send_json(
                        503,
                        {"error": "server busy", "text": "", "queue_depth": server.queue_depth()},
                    )
                    return
                except _Cancelled:
                    self._send_json(499, {"error": "cancelled", "text": ""})
                    return
                resp = result.to_dict()
                resp["request_id"] = rid
                self._send_json(200, resp)
                return

            self._send_json(404, {"error": "not found"})

        # -------- abort (DELETE /inference/<id>) --------
        def do_DELETE(self) -> None:  # noqa: N802 - stdlib API
            if self.path.startswith("/inference/"):
                rid = self.path[len("/inference/"):]
                cancelled = server.cancel_request(rid) if rid else False
                self._send_json(
                    200 if cancelled else 404,
                    {"status": "cancelled" if cancelled else "not_found", "request_id": rid},
                )
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
                        "phase": server.phase(),
                        "queue_depth": server.queue_depth(),
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
