"""Unified HTTP/WebSocket ASR server for every starling model.

This replaces the former granite-only ``starling.granite.server``. One process
serves ONE model at a time, selected by ``--model`` (default ``granite``). The
model is kept resident in VRAM and exposed via a
parakeet-server-compatible interface:

  * ``GET  /``                - health check (reports ``model``, ``phase``, ``queue_depth``)
  * ``GET  /health``          - health alias
  * ``POST /inference``       - multipart/raw WAV upload -> ``{text, segments, duration_s, request_id}``
  * ``POST /transcribe``      - raw WAV bytes -> same shape as /inference
  * ``POST /warmup``          - pre-capture CUDA graphs on a silent clip (idempotent; 202)
  * ``DELETE /inference/<id>``- cancel a queued request by ``X-Request-Id``
  * ``WS   /stream``          - real-time streaming dictation

The model pipelines have incompatible ``transcribe`` signatures, so the
per-model differences (input building, long-audio chunking, the granite-only
speculative path) are isolated behind a :class:`ModelBackend` with one subclass
per model. Everything else -- request queue, cancellation, lifecycle phase,
streaming session, the dual FastAPI/stdlib transport, WAV/PCM decoding -- is
model-agnostic and shared.

Run with::

    python -m starling.server --model granite --port 8181
    python -m starling.server --model parakeet --warmup
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import socket
import struct
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import numpy as np

log = logging.getLogger("starling.server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8181
"""Default port (parakeet-server uses 8180; we sit next to it)."""

SAMPLE_RATE: int = 16000
"""Feature extractor sample rate (16 kHz mono) for every supported model."""

DEFAULT_MAX_CHUNK_SECONDS: float = 30.0
"""Largest chunk transcribed in one shot for chunked backends.

Granite is bounded by its 640-token KV cache (30 s ~ 300 audio + 22 prompt
tokens). Parakeet-unified uses this to avoid O(N^2) full-attention encoder
memory on long utterances.
"""

DEFAULT_MIN_CHUNK_SECONDS: float = 5.0
"""Minimum accumulated audio before the first streaming partial is emitted."""

DEFAULT_PARTIAL_INTERVAL_SECONDS: float = 3.0
"""After the first partial, re-transcribe the growing buffer at most this often."""

WARMUP_SECONDS: float = 5.0
"""Length of the silent dummy clip used to capture CUDA graphs at startup."""

DEFAULT_MAX_NEW_TOKENS: int = 200
"""Greedy decode budget per chunk (LLM-decoder backends)."""

GPU_LOCK_SESSION: str = "starling-server"
GPU_LOCK_ETA_MIN: int = 1

MAX_WAITERS: int = 8
"""Max requests waiting for the single GPU worker before HTTP 503 backpressure."""

CANCEL_POLL_SECONDS: float = 0.1

DEFAULT_MAX_UPLOAD_BYTES: int = 256 * 1024 * 1024
"""Maximum accepted HTTP request body (256 MiB, before multipart decoding)."""

DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 10 * 60.0
"""Wall-clock deadline covering both queueing and model execution."""

WS_GUID: bytes = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

MAX_WS_FRAME_BYTES: int = 16 * 1024 * 1024
"""Maximum accepted single WebSocket frame payload (16 MiB cap).

A client that claims a gigantic 64-bit payload length would otherwise block the
receiver (``_read_exact``) indefinitely. Frames larger than this raise and tear
the connection down instead.
"""

WS_SOCKET_TIMEOUT_SECONDS: float = 120.0
"""Idle timeout (seconds) on a stdlib WebSocket session.

A long dictation can stay silent for many seconds between utterances, so this is
generous; a totally dead/silent connection still times out rather than hanging a
handler thread forever.
"""

STREAM_TRIM_MIN_SAMPLES: int = SAMPLE_RATE
"""Minimum committed-prefix length (in samples) to drop from a stream buffer.

``StreamSession`` trims the chunker's finalized prefix once it crosses this size,
bounding per-append ``np.concatenate`` copying and RAM. 1 second of audio.
"""

# Supported model slugs -> (backend class, display name, gpu-lock model label).
# Built lazily as backend classes are defined below.
MODEL_SLUGS = ("granite", "parakeet", "parakeet_unified", "moss", "qwen3", "ark", "cohere", "higgs", "audex")


def _gpu_lock_model(slug: str) -> str:
    return {
        "granite": "granite-speech-4.1-2b",
        "parakeet": "parakeet-tdt-0.6b-v3",
        "parakeet_unified": "parakeet-unified-en-0.6b",
        "moss": "moss-transcribe-preview-2b",
        "qwen3": "qwen3-asr-1.7b",
        "ark": "ark-asr-3b",
        "cohere": "cohere-transcribe-03-2026",
        "higgs": "higgs-audio-v3-stt",
        "audex": "nemotron-labs-audex-2b",
    }.get(slug, slug)


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------
class ModelBackend:
    """Loads one model pipeline and transcribes 1-D float32 mono audio.

    Subclasses override :meth:`load`, :meth:`transcribe`, and :meth:`prewarm`.
    Audio arrives as a contiguous float32 numpy array at ``SAMPLE_RATE`` Hz;
    the return is a :class:`TranscribeResult` (text + chunk-level segments).
    Heavy ``torch`` / model imports happen inside :meth:`load` so ``--help``
    and CPU-only tests never touch CUDA.
    """

    slug: str = ""

    def __init__(self, config: "ServerConfig") -> None:
        self.config = config
        self.pipe: Any = None
        self.processor: Any = None
        self.chunker: Any = None
        self._cancel_event: Optional[threading.Event] = None
        self._deadline: float = float("inf")

    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    def load(self) -> None:
        raise NotImplementedError

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        raise NotImplementedError

    def prewarm(self, samples: np.ndarray) -> None:
        """Default warmup: one transcribe on a short silent clip."""
        self.transcribe(samples)

    def set_graph_mode(self, *, streaming: bool, duration_s: float = 0.0) -> None:
        """Pick graphed vs eager for the coming transcribe. No-op by default;
        backends whose pipelines expose the toggle (ark/qwen3/cohere) override."""

    def _want_graphed(self, *, streaming: bool, duration_s: float, chunked: bool) -> bool:
        """Resolve the adaptive graph policy (see ServerConfig.graph_mode)."""
        mode = getattr(self.config, "graph_mode", "auto")
        if mode == "graphed":
            return True
        if mode == "eager":
            return False
        if streaming:
            return True  # fixed-window stream -> one recurring shape -> amortises
        if chunked and duration_s >= float(getattr(self.config, "file_graph_min_seconds", 60.0)):
            return True  # long file -> many same-size chunks -> amortises
        return False     # one-shot / short file -> capture never amortises

    def _check_stopped(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise _Cancelled()
        if time.monotonic() >= self._deadline:
            raise _DeadlineExceeded()

    def _decode_budget(self, duration_s: float) -> int:
        """Scale the decode cap down for short chunks without exceeding the CLI cap."""
        estimated = max(1, math.ceil(duration_s * 5.0) + 32)
        return min(max(1, int(self.config.max_new_tokens)), estimated)

    def _effective_chunk_seconds(self, fallback: float = DEFAULT_MAX_CHUNK_SECONDS) -> float:
        configured = max(0.1, float(self.config.max_chunk_seconds or fallback))
        token_limited = max(0.1, (int(self.config.max_new_tokens) - 32) / 5.0)
        return min(configured, token_limited)

    def _configured_chunk_seconds(self) -> float:
        return max(
            0.1,
            float(self.config.max_chunk_seconds or DEFAULT_MAX_CHUNK_SECONDS),
        )

    def _transcribe_chunked(
        self,
        samples: np.ndarray,
        transcribe_chunk,
    ) -> "TranscribeResult":
        """Run a single-shot backend over bounded waveform chunks."""
        audio = np.ascontiguousarray(samples.reshape(-1), dtype=np.float32)
        audio_seconds = len(audio) / SAMPLE_RATE
        # A user can raise --max-chunk-seconds, but the token cap must never
        # make that silently truncate output. Split again when the estimated
        # transcript would consume the configured decode budget.
        chunk_seconds = self._effective_chunk_seconds()
        chunk_size = max(1, int(round(chunk_seconds * SAMPLE_RATE)))
        texts: list[str] = []
        segments: list[dict[str, Any]] = []
        for start in range(0, len(audio), chunk_size):
            self._check_stopped()
            end = min(start + chunk_size, len(audio))
            chunk = np.ascontiguousarray(audio[start:end])
            text = transcribe_chunk(chunk, self._decode_budget(len(chunk) / SAMPLE_RATE))
            texts.append(text)
            segments.append(
                {"text": text, "start_s": start / SAMPLE_RATE, "end_s": end / SAMPLE_RATE}
            )
        self._check_stopped()
        joined = " ".join(" ".join(texts).split())
        return TranscribeResult(text=joined, segments=segments, duration_s=audio_seconds)


class GraniteBackend(ModelBackend):
    """granite-speech-4.1-2b: chunked transcribe + optional self-speculation."""

    slug = "granite"

    def load(self) -> None:
        from .granite.loader import load_model_and_processor
        from .granite.pipeline import MegaPipeline

        model, processor = load_model_and_processor(attn_impl=self.config.attn_impl)
        self.pipe = MegaPipeline(
            model,
            processor,
            encoder_mode=self.config.encoder_mode,
            use_fused_llm=self.config.use_fused_llm,
        )
        self.processor = processor

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        import torch

        from .granite.audio import build_inputs
        from .granite.long_audio import DEFAULT_CHUNK_SECONDS, _join_chunk_texts, chunk_audio

        assert self.pipe is not None and self.processor is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = torch.from_numpy(np.ascontiguousarray(samples)).float().unsqueeze(0).contiguous()
        sr = SAMPLE_RATE
        max_chunk = self._effective_chunk_seconds(DEFAULT_CHUNK_SECONDS)
        audio_seconds = wav.shape[1] / sr

        if audio_seconds <= max_chunk:
            self._check_stopped()
            inputs = build_inputs(self.processor, wav)
            text, _ = self.pipe.transcribe(
                inputs["input_features"],
                inputs["input_ids"],
                inputs.get("input_features_mask"),
                max_new_tokens=self._decode_budget(audio_seconds),
                speculative=self.config.speculative,
            )
            return TranscribeResult(
                text=text,
                segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
                duration_s=audio_seconds,
            )

        texts: list[str] = []
        segments: list[dict[str, Any]] = []
        max_cache_len = int(getattr(self.pipe.llm, "max_cache_len", 640))
        dtype = self.pipe.dtype
        for chunk_wav, start, end, _idx in chunk_audio(wav, sr, max_chunk):
            self._check_stopped()
            inputs = build_inputs(self.processor, chunk_wav)
            feats = inputs["input_features"].to(dtype)
            ids = inputs["input_ids"]
            mask = inputs.get("input_features_mask")
            prompt_len = int(ids.shape[1])
            budget = max(
                1,
                min(
                    self._decode_budget(end - start),
                    max_cache_len - prompt_len - 1,
                ),
            )
            text, _ = self.pipe.transcribe(
                feats, ids, mask,
                max_new_tokens=budget,
                speculative=self.config.speculative,
            )
            texts.append(text)
            segments.append({"text": text, "start_s": start, "end_s": end})
        return TranscribeResult(
            text=_join_chunk_texts(texts, 0.0),
            segments=segments,
            duration_s=audio_seconds,
        )


class ParakeetBackend(ModelBackend):
    """parakeet-tdt-0.6b-v3: raw-audio transcribe, long audio handled internally."""

    slug = "parakeet"

    def load(self) -> None:
        from .parakeet.pipeline import MegaParakeetPipeline

        self.pipe = MegaParakeetPipeline()
        self.processor = self.pipe.processor

    def _get_chunker(self):
        assert self.pipe is not None
        if self.chunker is None:
            from .parakeet.chunking import ChunkedTranscriber

            chunk_seconds = self._configured_chunk_seconds()
            self.chunker = ChunkedTranscriber(
                self.pipe,
                chunk_seconds=chunk_seconds,
                overlap_seconds=min(2.0, chunk_seconds / 4.0),
            )
        return self.chunker

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        audio_seconds = len(audio) / SAMPLE_RATE
        max_chunk_seconds = self._configured_chunk_seconds()
        if audio_seconds > max_chunk_seconds:
            text = self._get_chunker().transcribe(
                audio, sr=SAMPLE_RATE, should_stop=self._check_stopped
            )
        else:
            self._check_stopped()
            texts = self.pipe.transcribe([audio])
            text = texts[0] if texts else ""
        # Parakeet's TDT decoder has no chunk-window segment contract exposed
        # here, so we return a single whole-utterance segment (the server's
        # streaming partials still carve the timeline via the rolling buffer).
        return TranscribeResult(
            text=text,
            segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
            duration_s=audio_seconds,
        )


class ParakeetUnifiedBackend(ModelBackend):
    """parakeet-unified-en-0.6b: NeMo-free megakernel (FastConformer-RNN-T)."""

    slug = "parakeet_unified"

    def load(self) -> None:
        from .parakeet_unified.pipeline import MegaParakeetUnifiedPipeline

        self.pipe = MegaParakeetUnifiedPipeline()
        # No HF processor for this model (the .nemo ships only weights + the
        # sentencepiece model inside the zip); the pipeline exposes its
        # tokenizer directly.
        self.processor = self.pipe.tokenizer
        self.chunker = None

    def _get_chunker(self):
        assert self.pipe is not None
        if self.chunker is None:
            from .parakeet_unified.chunking import ChunkedTranscriber

            chunk_seconds = self._configured_chunk_seconds()
            overlap_seconds = min(2.0, chunk_seconds / 4.0)
            self.chunker = ChunkedTranscriber(
                self.pipe,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
        return self.chunker

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        audio_seconds = len(audio) / SAMPLE_RATE
        max_chunk_seconds = self._configured_chunk_seconds()
        if audio_seconds > max_chunk_seconds:
            text = self._get_chunker().transcribe(
                audio, sr=SAMPLE_RATE, should_stop=self._check_stopped
            )
        else:
            self._check_stopped()
            texts = self.pipe.transcribe([audio])
            text = texts[0] if texts else ""
        return TranscribeResult(
            text=text,
            segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
            duration_s=audio_seconds,
        )


class MossBackend(ModelBackend):
    """moss-transcribe-preview-2b: processor-built inputs, chunked transcribe."""

    slug = "moss"

    def load(self) -> None:
        from .moss.loader import load_model_and_processor
        from .moss.pipeline import MossMegaPipeline

        # Streaming uses the adaptive cudagraph encoder: in the /stream path the
        # transcribed windows are bounded (fixed chunk + short tails, prompt
        # T<=~150), so few graphs are captured and they replay in a narrow
        # position range -- the regime where the graphs are robust (validated by
        # the 27-cell streaming bench + 400-iter stress).  The pipeline default
        # is eager because *batch* transcription of wildly varying clip lengths
        # in one process can hit a graph-accumulation limit in the megakernel
        # decode graph (bench_leaderboard shards per dataset to avoid it).
        #
        # Optional FP8 decode uses the graph-safe fused dequant-GEMV; unlike the
        # old torch._scaled_mm path it has no cuBLASLt workspace to alias across
        # the recurring decode graphs used here.
        model, processor = load_model_and_processor()
        self.pipe = MossMegaPipeline(
            model, processor, max_cache_len=2048, encoder_mode="cudagraph",
        )
        self.processor = processor
        self._proc_call = processor  # the MossProcessor is callable on a wav

    def _build(self, wav_np: np.ndarray) -> dict:
        import torch

        inp = self._proc_call(wav_np.astype("float32"))
        return {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inp.items()
        }

    def _transcribe_chunk(self, audio: np.ndarray, budget: int) -> str:
        assert self.pipe is not None and self.processor is not None
        inp = self._build(audio)
        text, _ = self.pipe.transcribe(
            inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
            inp["audio_input_mask"], max_new_tokens=budget,
        )
        return text

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        return self._transcribe_chunked(samples, self._transcribe_chunk)


class Qwen3Backend(ModelBackend):
    """qwen3-asr-1.7b: processor-built inputs, chunked transcribe."""

    slug = "qwen3"

    def load(self) -> None:
        from .qwen3.loader import load_model_and_processor
        from .qwen3.pipeline import MegaPipeline

        model, processor = load_model_and_processor(attn_impl=self.config.attn_impl)
        self.pipe = MegaPipeline.from_pretrained()
        self.processor = processor

    def set_graph_mode(self, *, streaming: bool, duration_s: float = 0.0) -> None:
        # Long files are chunked into recurring shapes, so graph capture can
        # amortise there as well as in the fixed-window streaming path.
        self.pipe.set_prefill_use_graph(
            self._want_graphed(streaming=streaming, duration_s=duration_s, chunked=True)
        )

    def _transcribe_chunk(self, samples: np.ndarray, budget: int) -> str:
        import torch

        from .qwen3.audio import build_inputs

        assert self.pipe is not None and self.processor is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = torch.from_numpy(np.ascontiguousarray(samples)).float().unsqueeze(0).contiguous()
        inp = build_inputs(self.processor, wav, sr=SAMPLE_RATE)
        text, _ = self.pipe.transcribe(
            inp["input_features"], inp["input_ids"],
            inp.get("input_features_mask"),
            max_new_tokens=budget,
        )
        return text

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        return self._transcribe_chunked(samples, self._transcribe_chunk)


class ArkBackend(ModelBackend):
    """ark-asr-3b: Whisper+adapter encoder + Qwen2.5 decoder megakernel.

    The pipeline's ``transcribe`` takes a 1-D float32 waveform directly (no
    processor-built input dict), so this backend is the thinnest of the LLM
    backends. File input is waveform-chunked to bound the 4096-token static KV
    cache, like qwen3.
    """

    slug = "ark"

    def load(self) -> None:
        from .ark.pipeline import MegaPipeline

        self.pipe = MegaPipeline.from_pretrained()

    def set_graph_mode(self, *, streaming: bool, duration_s: float = 0.0) -> None:
        self.pipe.set_prefill_use_graph(
            self._want_graphed(streaming=streaming, duration_s=duration_s, chunked=True)
        )

    def _transcribe_chunk(self, samples: np.ndarray, budget: int) -> str:
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = np.ascontiguousarray(samples, dtype=np.float32)
        text, _ids = self.pipe.transcribe(
            wav, max_new_tokens=budget,
        )
        return text

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        return self._transcribe_chunked(samples, self._transcribe_chunk)


class CohereBackend(ModelBackend):
    """cohere-transcribe-03-2026: seq2seq enc-dec megakernel.

    ``CohereMegaPipeline.transcribe`` accepts a 1-D waveform and returns a
    list of transcripts (one per processor chunk); we join them for the
    single-clip server path. Single-shot per chunk.
    """

    slug = "cohere"

    def load(self) -> None:
        from .cohere.pipeline import CohereMegaPipeline

        self.pipe = CohereMegaPipeline.from_pretrained()

    def set_graph_mode(self, *, streaming: bool, duration_s: float = 0.0) -> None:
        # Chunks internally, so a long file yields many same-size encoder shapes;
        # graph the encoder for streaming or long files, eager otherwise. The
        # byte-exact cross-attn (decoder-graph) bucketing stays on regardless.
        self.pipe.set_graphed_encoder(
            self._want_graphed(streaming=streaming, duration_s=duration_s, chunked=True)
        )

    def _transcribe_chunk(self, samples: np.ndarray, budget: int) -> str:
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = np.ascontiguousarray(samples, dtype=np.float32)
        texts, _ids = self.pipe.transcribe(
            wav, max_new_tokens=budget,
        )
        return " ".join(t.strip() for t in texts if t.strip())

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        return self._transcribe_chunked(samples, self._transcribe_chunk)


class HiggsBackend(ModelBackend):
    """higgs-audio-v3-stt: Whisper-large-v3 mel + Qwen3-1.7B decoder megakernel.

    NOTE: like the benchmark adapter, this only loads under the isolated
    ``.venv-higgs`` (``transformers==4.51``); see ``higgs/UV_NOTES.md``.
    """

    slug = "higgs"

    def load(self) -> None:
        try:
            from .higgs.pipeline import HiggsMega
        except Exception as exc:
            raise RuntimeError(
                "the higgs backend requires the isolated .venv-higgs environment; "
                "see src/starling/higgs/UV_NOTES.md"
            ) from exc

        self.pipe = HiggsMega.from_pretrained()

    def _transcribe_chunk(self, samples: np.ndarray, budget: int) -> str:
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = np.ascontiguousarray(samples, dtype=np.float32)
        text = self.pipe.transcribe(
            wav, sample_rate=SAMPLE_RATE, max_new_tokens=budget,
        )
        return text

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        return self._transcribe_chunked(samples, self._transcribe_chunk)


class AudexBackend(ModelBackend):
    """nemotron-labs-audex-2b: Whisper encoder + Nemotron-Dense 2B decoder (ASR).

    The pipeline's ``transcribe`` takes a 1-D float32 waveform. Audio is
    chunked to 30 s clips (the Whisper window), each producing 750 audio
    embeddings. File input is waveform-chunked to bound the static KV cache.
    """

    slug = "audex"

    def load(self) -> None:
        from .audex.pipeline import MegaPipeline

        self.pipe = MegaPipeline.from_pretrained()

    def set_graph_mode(self, *, streaming: bool, duration_s: float = 0.0) -> None:
        self.pipe.set_prefill_use_graph(
            self._want_graphed(streaming=streaming, duration_s=duration_s, chunked=True)
        )

    def _transcribe_chunk(self, samples: np.ndarray, budget: int) -> str:
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = np.ascontiguousarray(samples, dtype=np.float32)
        text, _ = self.pipe.transcribe(wav, max_new_tokens=budget)
        return text

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        return self._transcribe_chunked(samples, self._transcribe_chunk)


_BACKENDS: dict[str, type[ModelBackend]] = {
    "granite": GraniteBackend,
    "parakeet": ParakeetBackend,
    "parakeet_unified": ParakeetUnifiedBackend,
    "moss": MossBackend,
    "qwen3": Qwen3Backend,
    "ark": ArkBackend,
    "cohere": CohereBackend,
    "higgs": HiggsBackend,
    "audex": AudexBackend,
}


def get_backend(slug: str, config: "ServerConfig") -> ModelBackend:
    """Resolve a model slug to a backend instance (pure, no GPU work)."""
    if slug not in _BACKENDS:
        raise ValueError(
            f"unknown model {slug!r}; choose one of {', '.join(MODEL_SLUGS)}"
        )
    return _BACKENDS[slug](config)


# ---------------------------------------------------------------------------
# Result + request context
# ---------------------------------------------------------------------------
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


class RequestState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"


@dataclass
class RequestContext:
    """Per-request handle tracked in the server's registry for cancellation."""

    id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    state: RequestState = RequestState.QUEUED
    deadline: float = float("inf")


class _Busy(Exception):
    """Internal sentinel: queue is full (backpressure) — client should retry."""


class _Cancelled(Exception):
    """Internal sentinel: request was cancelled via :meth:`StarlingServer.cancel_request`."""


class _DeadlineExceeded(Exception):
    """Internal sentinel: the request exceeded its wall-clock deadline."""


class _DuplicateRequest(Exception):
    """Internal sentinel: a caller reused an active request id."""


# ---------------------------------------------------------------------------
# Server config + state container
# ---------------------------------------------------------------------------
@dataclass
class ServerConfig:
    """Runtime configuration for :class:`StarlingServer`."""

    model: str = "granite"
    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS
    min_chunk_seconds: float = DEFAULT_MIN_CHUNK_SECONDS
    partial_interval_seconds: float = DEFAULT_PARTIAL_INTERVAL_SECONDS
    # Fixed-window overlapping-chunk streaming (see starling.stream_chunk).  When
    # ``stream_chunk_seconds > 0``, the /stream buffer is finalized in constant
    # ``stream_chunk_seconds`` windows overlapping by ``stream_overlap_seconds``:
    # bounds the per-transcribe prompt (no long-dictation KV overflow), keeps
    # work O(N), and -- being a constant mel length -- reuses the cudagraph
    # encoder.  Set to 0 for the legacy re-transcribe-the-whole-buffer behavior.
    stream_chunk_seconds: float = 12.0
    stream_overlap_seconds: float = 3.0
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    speculative: bool = True
    warmup: bool = False
    encoder_mode: str = "cudagraph"
    use_fused_llm: bool = True
    attn_impl: str = "eager"
    # Adaptive CUDA-graph policy (backends that support it: ark, qwen3, cohere).
    # A graph capture only pays off when the same shape recurs enough to amortise
    # it, so the server picks per request mode:
    #   * ``/stream`` (fixed stream_chunk_seconds windows -> one recurring shape)
    #     uses the graphed path.
    #   * ``/inference`` / ``/transcribe`` (one-shot file) uses eager, unless the
    #     file is long enough to be chunked into many same-size windows
    #     (``duration >= file_graph_min_seconds``), in which case graphed wins.
    # ``graph_mode`` overrides the auto policy: "auto" (default) | "graphed" |
    # "eager". Both paths are byte-exact; this only trades capture cost vs reuse.
    graph_mode: str = "auto"
    file_graph_min_seconds: float = 60.0
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    opt_flags: Any = None


@dataclass
class StarlingServer:
    """Owns the loaded model backend and serves transcription requests.

    Heavy imports are deferred to :meth:`load`.
    """

    config: ServerConfig = field(default_factory=ServerConfig)
    backend: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _loaded: bool = False

    # --- request queueing -------------------------------------------------
    _n_waiters: int = 0
    _requests: dict[str, "RequestContext"] = field(default_factory=dict)
    _request_order: list[str] = field(default_factory=list)
    _queue_changed: threading.Condition = field(init=False, repr=False)

    # --- lifecycle phase (reported by /health) ----------------------------
    #   unloaded -> loading_weights -> warming_up -> ready
    _phase: str = "unloaded"

    # --- warmup dedup -----------------------------------------------------
    #   ``/warmup`` and ``warmup()`` may be called concurrently; without a guard
    #   each call captures a fresh CUDA graph on the GPU. The lock + flag make
    #   warmup idempotent: a second caller no-ops while one is in flight.
    _warmup_lock: threading.Lock = field(default_factory=threading.Lock)
    _warmup_in_progress: bool = False

    def __post_init__(self) -> None:
        self._queue_changed = threading.Condition(self._lock)

    @property
    def model_slug(self) -> str:
        return self.config.model

    @property
    def loaded(self) -> bool:
        """True once the model backend has been loaded into VRAM."""
        with self._lock:
            return self._loaded

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Build the backend, load the model, and optionally warm up graphs.

        Idempotent: a second call is a no-op. Thread-safe.
        """
        with self._lock:
            if self._loaded:
                return
            self._phase = "loading_weights"
            if self.config.opt_flags is not None:
                from .flags import set_default_flags

                set_default_flags(self.config.opt_flags)
            backend = get_backend(self.config.model, self.config)
            t0 = time.perf_counter()
            log.info("loading %s model ...", self.config.model)
            backend.load()
            self.backend = backend
            self._loaded = True
            self._phase = "loaded"
            log.info("%s model loaded in %.1fs", self.config.model, time.perf_counter() - t0)

        if self.config.warmup:
            self.warmup()
        else:
            self._phase = "ready"

    def warmup(self) -> None:
        """Capture CUDA graphs on a short silent clip (no-op if not loaded or already warming)."""
        if not self._loaded or self.backend is None:
            return
        # Dedup: if another thread is already capturing graphs, no-op. The flag
        # is checked/released under the lock but the GPU work runs unlocked so a
        # long warmup never blocks other callers.
        with self._warmup_lock:
            if self._warmup_in_progress:
                return
            self._warmup_in_progress = True
        try:
            from .parakeet.gpu_lock import with_gpu_lock

            self._phase = "warming_up"
            log.info("warming up CUDA graphs on %.1fs silent clip ...", WARMUP_SECONDS)
            n = int(WARMUP_SECONDS * SAMPLE_RATE)
            dummy = np.zeros(n, dtype=np.float32)
            with with_gpu_lock(
                session=GPU_LOCK_SESSION,
                model=_gpu_lock_model(self.config.model),
                eta_min=GPU_LOCK_ETA_MIN,
                note="warmup",
            ):
                self._transcribe_np(dummy)
            self._phase = "ready"
            log.info("warmup complete")
        finally:
            with self._warmup_lock:
                self._warmup_in_progress = False

    # ------------------------------------------------------------------ #
    # inference core (callers acquire the GPU lock)
    # ------------------------------------------------------------------ #
    def _transcribe_np(self, samples: np.ndarray, *, streaming: bool = False) -> TranscribeResult:
        """Transcribe a 1-D float32 mono numpy array via the loaded backend.

        The GPU lock is NOT taken here; callers wrap the call so the lock scope
        stays tight. ``streaming`` selects the backend's adaptive CUDA-graph
        policy (fixed stream chunks -> graphed; one-shot file -> eager unless
        long); a no-op for backends without the toggle.
        """
        assert self._loaded and self.backend is not None
        self.backend.set_graph_mode(
            streaming=streaming, duration_s=len(samples) / SAMPLE_RATE
        )
        return self.backend.transcribe(samples)

    # ------------------------------------------------------------------ #
    # public entry points (synchronous; offload by caller)
    # ------------------------------------------------------------------ #
    def transcribe_bytes_sync(
        self, wav_bytes: bytes, request_id: Optional[str] = None
    ) -> TranscribeResult:
        self._ensure_loaded()
        samples, sr = _wav_bytes_to_float32(wav_bytes)
        if sr != SAMPLE_RATE:
            samples = _resample_audio(samples, sr, SAMPLE_RATE)
        return self._run_queued_sync(samples, request_id)

    def transcribe_pcm_sync(
        self, pcm16_bytes: bytes, request_id: Optional[str] = None
    ) -> TranscribeResult:
        self._ensure_loaded()
        samples = _pcm16_bytes_to_float32(pcm16_bytes)
        return self._run_queued_sync(samples, request_id)

    def _run_queued_sync(
        self, samples: np.ndarray, request_id: Optional[str], *, streaming: bool = False
    ) -> TranscribeResult:
        rid = request_id or uuid.uuid4().hex
        timeout = float(self.config.request_timeout_seconds)
        deadline = time.monotonic() + timeout if timeout > 0 else float("inf")
        ctx = RequestContext(rid, deadline=deadline)
        with self._queue_changed:
            if self._n_waiters >= MAX_WAITERS:
                raise _Busy()
            if ctx.id in self._requests:
                raise _DuplicateRequest(ctx.id)
            self._n_waiters += 1
            self._requests[ctx.id] = ctx
            self._request_order.append(ctx.id)
            self._queue_changed.notify_all()
        try:
            return self._serial_run(ctx, samples, streaming=streaming)
        finally:
            with self._queue_changed:
                self._n_waiters = max(0, self._n_waiters - 1)
                self._requests.pop(ctx.id, None)
                if ctx.id in self._request_order:
                    self._request_order.remove(ctx.id)
                self._queue_changed.notify_all()

    def _serial_run(
        self, ctx: "RequestContext", samples: np.ndarray, *, streaming: bool = False
    ) -> TranscribeResult:
        from .parakeet.gpu_lock import GpuLockBusy, acquire_gpu_lock, release_gpu_lock

        # Preserve arrival order inside this server process. The head waiter is
        # the only thread allowed to contend for the cross-process GPU lock.
        with self._queue_changed:
            while not self._request_order or self._request_order[0] != ctx.id:
                self._raise_if_stopped(ctx)
                wait_for = min(CANCEL_POLL_SECONDS, max(0.0, ctx.deadline - time.monotonic()))
                self._queue_changed.wait(wait_for)

        while True:
            self._raise_if_stopped(ctx)
            try:
                lock_owner = acquire_gpu_lock(
                    session=GPU_LOCK_SESSION,
                    model=_gpu_lock_model(self.config.model),
                    eta_min=GPU_LOCK_ETA_MIN,
                    note="inference",
                    wait=False,
                )
                break
            except GpuLockBusy:
                if ctx.cancel.wait(CANCEL_POLL_SECONDS):
                    raise _Cancelled()

        ctx.state = RequestState.RUNNING
        try:
            self.backend._cancel_event = ctx.cancel
            self.backend._deadline = ctx.deadline
            result = self._transcribe_np(samples, streaming=streaming)
            self._raise_if_stopped(ctx)
            return result
        finally:
            if self.backend is not None:
                self.backend._cancel_event = None
                self.backend._deadline = float("inf")
            release_gpu_lock(lock_owner)

    @staticmethod
    def _raise_if_stopped(ctx: "RequestContext") -> None:
        if ctx.cancel.is_set():
            raise _Cancelled()
        if time.monotonic() >= ctx.deadline:
            raise _DeadlineExceeded()

    # ------------------------------------------------------------------ #
    # request registry (cancellation + introspection)
    # ------------------------------------------------------------------ #
    def cancel_request(self, request_id: str) -> bool:
        with self._lock:
            ctx = self._requests.get(request_id)
        if ctx is None:
            return False
        ctx.cancel.set()
        return True

    def queue_depth(self) -> int:
        with self._lock:
            return sum(ctx.state is RequestState.QUEUED for ctx in self._requests.values())

    def is_busy(self) -> bool:
        with self._lock:
            return self._n_waiters > 0

    def phase(self) -> str:
        with self._lock:
            return self._phase

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


def _transcribe_payload_sync(
    server: StarlingServer, payload: bytes, request_id: str
) -> tuple[int, dict[str, Any]]:
    """Shared HTTP transport adapter for one WAV transcription request."""
    try:
        result = server.transcribe_bytes_sync(payload, request_id)
    except _Busy:
        return 503, {
            "error": "server busy",
            "text": "",
            "queue_depth": server.queue_depth(),
        }
    except _Cancelled:
        return 499, {"error": "cancelled", "text": ""}
    except _DeadlineExceeded:
        return 504, {"error": "request timed out", "text": ""}
    except _DuplicateRequest:
        return 409, {"error": "request id already active", "text": ""}
    except (ValueError, wave.Error):
        # Malformed/truncated WAV or unsupported encoding from the decoder.
        # Map to 400 (client error) instead of propagating as a 500 / dead socket.
        return 400, {"error": "malformed audio payload", "text": ""}
    response = result.to_dict()
    response["request_id"] = request_id
    return 200, response


# ---------------------------------------------------------------------------
# Audio helpers (numpy / stdlib only - no torch import required at module load)
# ---------------------------------------------------------------------------
def _wav_bytes_to_float32(data: bytes) -> tuple[np.ndarray, int]:
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
    if len(data) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2 == 1:
        # Odd byte count can't be whole int16 samples; drop the trailing byte
        # rather than silently misaligning the whole stream.
        log.warning("dropping odd trailing PCM byte (len=%d)", len(data))
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _resample_audio(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited polyphase resampling, including anti-alias filtering."""
    if sr_in == sr_out or len(samples) == 0:
        return samples
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sample rates must be positive")
    from scipy.signal import resample_poly

    divisor = math.gcd(int(sr_in), int(sr_out))
    out = resample_poly(samples, sr_out // divisor, sr_in // divisor)
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Streaming session (WS /stream)
# ---------------------------------------------------------------------------
@dataclass
class StreamSession:
    """Per-connection rolling audio buffer + streaming state.

    With ``config.stream_chunk_seconds > 0`` the buffer is finalized in fixed
    overlapping windows (:class:`starling.stream_chunk.ChunkStreamer`): bounded
    per-transcribe prompt, O(N) work, and cudagraph-encoder reuse.  Otherwise the
    legacy whole-buffer re-transcribe path is used.
    """

    server: StarlingServer
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    last_partial_ts: float = 0.0
    chunker: Any = None

    def __post_init__(self) -> None:
        cfg = self.server.config
        if getattr(cfg, "stream_chunk_seconds", 0.0) and cfg.stream_chunk_seconds > 0:
            from .stream_chunk import ChunkStreamer

            self.chunker = ChunkStreamer(
                sample_rate=SAMPLE_RATE,
                chunk_seconds=cfg.stream_chunk_seconds,
                overlap_seconds=cfg.stream_overlap_seconds,
                min_seconds=cfg.min_chunk_seconds,
                partial_interval_seconds=cfg.partial_interval_seconds,
            )

    def _tx(self, window: np.ndarray) -> Optional[str]:
        """Transcribe one window to text; ``None`` if the server is busy/cancelled."""
        try:
            res = self.server._run_queued_sync(
                np.ascontiguousarray(window, dtype=np.float32), None, streaming=True
            )
        except (_Busy, _Cancelled):
            return None
        return res.text

    def stream_step(self, now: float) -> Optional[str]:
        """Advance the chunked stream; returns text to emit as a partial, or None."""
        return self.chunker.step(self.samples, now, self._tx)

    def stream_flush(self) -> str:
        """Finalize all buffered audio (on commit) and return the full text."""
        return self.chunker.flush(self.samples, self._tx)

    def _maybe_trim_samples(self) -> None:
        """Drop the chunker's committed prefix from the rolling buffer.

        ``ChunkStreamer.boundary`` is the sample index up to which audio is fully
        finalized (advanced only at whole-window boundaries). Everything below it
        is dead weight we keep re-copying on every append and re-passing to the
        chunker. Trimming it keeps the buffer (and per-append ``concatenate`` cost)
        bounded over a long dictation.

        Index-shift invariant: after we drop the first ``boundary`` samples, the
        chunker's absolute ``boundary`` index points into dropped territory, so we
        reset ``chunker.boundary = 0``. The chunker then sees the trimmed array as
        fresh from index 0, and its committed-text state stays consistent because
        audio before ``boundary`` was already stitched into ``committed`` -- it is
        never read again. We only trim once the prefix is substantial (>= 1s and
        a meaningful fraction of the buffer) to avoid trimming on every tiny chunk.
        """
        chunker = self.chunker
        if chunker is None:
            return
        b = chunker.boundary
        if b <= 0 or b >= len(self.samples):
            return
        if b < STREAM_TRIM_MIN_SAMPLES and b < len(self.samples) // 2:
            return
        self.samples = np.ascontiguousarray(self.samples[b:], dtype=np.float32)
        chunker.boundary = 0

    def append_pcm(self, pcm16_bytes: bytes) -> None:
        s = _pcm16_bytes_to_float32(pcm16_bytes)
        if s.size > 0:
            self.samples = np.concatenate([self.samples, s]) if self.samples.size else s
        self._maybe_trim_samples()

    def append_wav(self, wav_bytes: bytes) -> None:
        # Only treat as WAV if it has a RIFF/WAVE header; otherwise it's raw PCM16.
        if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            self.append_pcm(wav_bytes)
            return
        try:
            s, sr = _wav_bytes_to_float32(wav_bytes)
        except (ValueError, wave.Error):
            # Malformed WAV despite the header -- don't silently reinterpret as
            # PCM16 (that decodes header bytes as audio samples -> garbage).
            # Drop just this chunk.
            return
        if sr != SAMPLE_RATE:
            s = _resample_audio(s, sr, SAMPLE_RATE)
        if s.size > 0:
            self.samples = np.concatenate([self.samples, s]) if self.samples.size else s
        self._maybe_trim_samples()

    @property
    def buffered_seconds(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    def should_emit_partial(self, now: float) -> bool:
        if self.buffered_seconds < self.server.config.min_chunk_seconds:
            return False
        if (now - self.last_partial_ts) < self.server.config.partial_interval_seconds:
            return False
        return True

    def reset(self) -> None:
        self.samples = np.zeros(0, dtype=np.float32)
        self.last_partial_ts = 0.0
        if self.chunker is not None:
            self.chunker.reset()

    def transcribe_current_sync(self) -> TranscribeResult:
        snapshot = self.samples.copy()
        return self.server._run_queued_sync(snapshot, None)


# ===========================================================================
# BACKEND A: FastAPI + uvicorn (preferred, optional deps)
# ===========================================================================
def create_app(
    config: Optional[ServerConfig] = None,
    *,
    server: Optional[StarlingServer] = None,
    load_on_startup: bool = True,
) -> Any:
    """Build the FastAPI application bound to a :class:`StarlingServer`."""
    from fastapi import (  # type: ignore
        FastAPI,
        HTTPException,
        Request,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.responses import JSONResponse  # type: ignore

    if server is not None and config is not None and server.config is not config:
        raise ValueError("pass either config or server, not both")
    server = server or StarlingServer(config=config or ServerConfig())

    @asynccontextmanager
    async def _lifespan(_app):  # noqa: ANN001
        if load_on_startup:
            await asyncio.to_thread(server.load)
        yield

    app = FastAPI(title="starling-server", version="2.0.0", lifespan=_lifespan)
    app.state.starling_server = server  # type: ignore[attr-defined]

    # Strong references for fire-and-forget background tasks. asyncio only holds
    # a weak reference to tasks created via create_task, so an unreferenced task
    # can be garbage-collected before it ever runs. Each entry is dropped again
    # as soon as its task completes (see add_done_callback) so the set cannot
    # grow without bound.
    _background_tasks: set = set()

    async def _decode_inference_body(request: "Request") -> bytes:
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > server.config.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="request body too large")
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid Content-Length") from None
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > server.config.max_upload_bytes:
                raise HTTPException(status_code=413, detail="request body too large")
            chunks.append(chunk)
        body = b"".join(chunks)
        if not body:
            return b""
        ctype = request.headers.get("content-type", "")
        if "multipart/form-data" in ctype:
            return _extract_multipart_payload(body, ctype)
        return body

    def _request_id(request: "Request") -> str:
        rid = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
        return rid or uuid.uuid4().hex

    def _health_body() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": server.model_slug,
            "loaded": server.loaded,
            "busy": server.is_busy(),
            "phase": server.phase(),
            "queue_depth": server.queue_depth(),
        }

    @app.get("/")
    async def health() -> JSONResponse:
        return JSONResponse(_health_body())

    @app.get("/health")
    async def health_alias() -> JSONResponse:
        return await health()

    @app.post("/warmup")
    async def warmup_route() -> JSONResponse:
        task = asyncio.create_task(asyncio.to_thread(server.warmup))
        # Retain a strong reference until completion (see _background_tasks).
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return JSONResponse(
            {"status": "warmup started", "phase": server.phase()}, status_code=202
        )

    async def _inference(request):  # noqa: ANN001
        payload = await _decode_inference_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="empty upload")
        rid = _request_id(request)
        status, response = await asyncio.to_thread(
            _transcribe_payload_sync, server, payload, rid
        )
        return JSONResponse(response, status_code=status)

    async def _transcribe(request):  # noqa: ANN001
        payload = await _decode_inference_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="empty request body")
        rid = _request_id(request)
        status, response = await asyncio.to_thread(
            _transcribe_payload_sync, server, payload, rid
        )
        return JSONResponse(response, status_code=status)

    async def _abort(request):  # noqa: ANN001
        rid = request.path_params.get("id")
        if not rid:
            return JSONResponse(status_code=400, content={"error": "missing request id"})
        cancelled = await asyncio.to_thread(server.cancel_request, str(rid))
        return JSONResponse(
            {"status": "cancelled" if cancelled else "not_found", "request_id": rid},
            status_code=200 if cancelled else 404,
        )

    _inference.__annotations__["request"] = Request
    _transcribe.__annotations__["request"] = Request
    _abort.__annotations__["request"] = Request
    app.add_api_route("/inference", _inference, methods=["POST"])
    app.add_api_route("/transcribe", _transcribe, methods=["POST"])
    app.add_api_route("/inference/{id}", _abort, methods=["DELETE"])

    async def _stream(ws):  # noqa: ANN001
        await ws.accept()
        sess = StreamSession(server=server)
        log.info("WS /stream client connected")
        try:
            while True:
                msg = await ws.receive()
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
                                if sess.chunker is not None:
                                    text = await asyncio.to_thread(sess.stream_flush)
                                    result = TranscribeResult(
                                        text=text,
                                        segments=[{"text": text, "start_s": 0.0,
                                                   "end_s": sess.buffered_seconds}],
                                        duration_s=sess.buffered_seconds,
                                    )
                                else:
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

                bdata = msg.get("bytes")
                if not bdata:
                    continue
                if bdata[:4] == b"RIFF" and bdata[8:12] == b"WAVE":
                    sess.append_wav(bdata)
                else:
                    sess.append_pcm(bdata)

                now = time.monotonic()
                if sess.chunker is not None:
                    # Chunked path: finalize full windows + emit committed+tail.
                    # ChunkStreamer handles throttling and busy (-> None) itself.
                    text = await asyncio.to_thread(sess.stream_step, now)
                    if text is not None:
                        sess.last_partial_ts = now
                        await ws.send_json(
                            {
                                "type": "partial",
                                "text": text,
                                "segments": [{"text": text, "start_s": 0.0,
                                              "end_s": sess.buffered_seconds}],
                                "start_s": 0.0,
                                "end_s": sess.buffered_seconds,
                            }
                        )
                elif sess.should_emit_partial(now):
                    try:
                        result = await asyncio.to_thread(sess.transcribe_current_sync)
                    except _Busy:
                        continue
                    except _Cancelled:
                        continue
                    sess.last_partial_ts = now
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

    _stream.__annotations__["ws"] = WebSocket
    app.add_api_websocket_route("/stream", _stream)

    return app


# ===========================================================================
# BACKEND B: stdlib-only (http.server + minimal RFC 6455 WebSocket)
# ===========================================================================
def _parse_content_disposition(header_value: str) -> dict[str, str]:
    """Parse a ``Content-Disposition`` value into its parameters.

    Returns a dict that always has a ``disposition`` key (e.g. ``form-data``)
    and any ``name``/``filename`` params (values are unquoted if quoted).
    """
    out: dict[str, str] = {}
    # Split on ';' but be tolerant: the first token is the disposition type.
    pieces = [p.strip() for p in header_value.split(";") if p.strip()]
    if not pieces:
        return out
    out["disposition"] = pieces[0].lower()
    for tok in pieces[1:]:
        if "=" not in tok:
            continue
        key, _, val = tok.partition("=")
        key = key.strip().lower()
        val = val.strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        out[key] = val
    return out


def _extract_multipart_payload(body: bytes, content_type: str) -> bytes:
    """Pull the audio bytes out of a ``multipart/form-data`` upload.

    Handles the convention where the file is the last part AND where it is named
    explicitly: parts are scored and the best match is returned. Selection order
    (most specific to least):

      1. a part named ``audio`` or ``file`` (explicit audio field), then
      2. a part with a non-empty ``filename``, then
      3. a part whose ``Content-Type`` starts with ``audio/``, then
      4. the last non-empty part (multipart form convention puts the file last).

    If no boundary is present the body is returned unchanged (callers may post
    raw WAV with ``Content-Type: application/octet-stream`` -- that path is
    valid and must keep working).
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
    # (name, filename, content_type, payload) per candidate part.
    candidates: list[tuple[Optional[str], Optional[str], Optional[str], bytes]] = []
    last_payload: Optional[bytes] = None
    for raw in body.split(delim):
        # Drop the surrounding CRLF wrappers and the closing ``--`` marker.
        if raw in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        if raw.startswith(b"\r\n"):
            raw = raw[2:]
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        if not raw:
            continue
        if b"\r\n\r\n" in raw:
            header_bytes, payload = raw.split(b"\r\n\r\n", 1)
        else:
            header_bytes, payload = b"", raw
        name: Optional[str] = None
        filename: Optional[str] = None
        ctype: Optional[str] = None
        for line in header_bytes.split(b"\r\n"):
            if b":" not in line:
                continue
            hkey, _, hval = line.partition(b":")
            hkey = hkey.strip().lower()
            hval = hval.strip().decode("latin-1", "replace")
            if hkey == b"content-disposition":
                params = _parse_content_disposition(hval)
                name = params.get("name")
                filename = params.get("filename")
            elif hkey == b"content-type":
                ctype = hval
        last_payload = payload
        candidates.append((name, filename, ctype, payload))

    if not candidates:
        return body

    def _score(c: tuple[Optional[str], Optional[str], Optional[str], bytes]) -> int:
        name, filename, ctype, _payload = c
        if name is not None and name.lower() in ("audio", "file"):
            return 4
        if filename:
            return 3
        if ctype and ctype.lower().startswith("audio/"):
            return 2
        return 0

    best = max(candidates, key=_score)
    # If nothing scored (only bare text fields), fall back to the last part.
    if _score(best) == 0:
        return last_payload if last_payload is not None else body
    return best[3]



def _ws_accept_key(client_key: str) -> str:
    h = hashlib.sha1(client_key.encode() + WS_GUID).digest()
    return base64.b64encode(h).decode()


def _ws_read_frame(rfile) -> tuple[int, bytes]:
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
        # RFC 6455 §5.1: client->server frames MUST be masked. A missing mask is
        # a protocol error; the spec wants a close with code 1002, but tearing
        # the connection down here is the practical security-equivalent measure.
        if not masked:
            raise ConnectionError("unmasked client frame (RFC 6455 violation)")
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", _read_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _read_exact(8))[0]
        # Cap the claimed length so a bogus 2^63 can't wedge _read_exact forever.
        if length > MAX_WS_FRAME_BYTES:
            raise ValueError(f"websocket frame too large: {length} bytes")
        mask = _read_exact(4) if masked else b""
        payload = _read_exact(length)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

        if opcode == 0x8:
            raise ConnectionError("client closed")
        if opcode == 0x9:
            return 0x9, payload
        if opcode == 0xA:
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
    rfile, wfile, server: StarlingServer, client_addr: tuple
) -> None:
    sess = StreamSession(server=server)
    log.info("WS /stream client connected from %s", client_addr)
    try:
        while True:
            try:
                opcode, payload = _ws_read_frame(rfile)
            except ConnectionError:
                break
            except socket.timeout:
                # Idle beyond WS_SOCKET_TIMEOUT_SECONDS -- close cleanly.
                log.info("WS /stream client %s timed out (idle)", client_addr)
                break

            if opcode == 0x9:
                _ws_send_pong(wfile, payload)
                continue
            if opcode == 0x1:
                try:
                    cmd = json.loads(payload.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    _ws_send_json(wfile, {"type": "error", "message": "bad json"})
                    continue
                mtype = cmd.get("type")
                if mtype == "commit":
                    if sess.buffered_seconds > 0.0:
                        try:
                            if sess.chunker is not None:
                                _t = sess.stream_flush()
                                result = TranscribeResult(
                                    text=_t,
                                    segments=[{"text": _t, "start_s": 0.0,
                                               "end_s": sess.buffered_seconds}],
                                    duration_s=sess.buffered_seconds,
                                )
                            else:
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
            if payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
                sess.append_wav(payload)
            else:
                sess.append_pcm(payload)

            now = time.monotonic()
            if sess.chunker is not None:
                text = sess.stream_step(now)
                if text is not None:
                    sess.last_partial_ts = now
                    _ws_send_json(
                        wfile,
                        {
                            "type": "partial",
                            "text": text,
                            "segments": [{"text": text, "start_s": 0.0,
                                          "end_s": sess.buffered_seconds}],
                            "start_s": 0.0,
                            "end_s": sess.buffered_seconds,
                        },
                    )
            elif sess.should_emit_partial(now):
                try:
                    result = sess.transcribe_current_sync()
                except _Busy:
                    continue
                except _Cancelled:
                    continue
                sess.last_partial_ts = now
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


def _build_stdlib_handler(server: StarlingServer):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            log.debug("http %s - %s", self.address_string(), fmt % args)

        server_version = "starling-server/2.0"
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length", "text": ""})
                return

            if length > server.config.max_upload_bytes:
                self.close_connection = True
                self._send_json(413, {"error": "request body too large", "text": ""})
                return

            if self.path == "/warmup":
                threading.Thread(target=server.warmup, daemon=True).start()
                self._send_json(202, {"status": "warmup started", "phase": server.phase()})
                return

            body = self.rfile.read(length) if length > 0 else b""

            rid = (
                self.headers.get("X-Request-Id")
                or self.headers.get("X-Correlation-Id")
                or uuid.uuid4().hex
            )

            if self.path == "/inference":
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    payload = _extract_multipart_payload(body, ctype)
                else:
                    payload = body
                if not payload:
                    self._send_json(400, {"error": "empty upload", "text": ""})
                    return
                status, response = _transcribe_payload_sync(server, payload, rid)
                self._send_json(status, response)
                return

            if self.path == "/transcribe":
                if not body:
                    self._send_json(400, {"error": "empty body", "text": ""})
                    return
                status, response = _transcribe_payload_sync(server, body, rid)
                self._send_json(status, response)
                return

            self._send_json(404, {"error": "not found"})

        def do_DELETE(self) -> None:  # noqa: N802
            if self.path.startswith("/inference/"):
                rid = self.path[len("/inference/"):]
                cancelled = server.cancel_request(rid) if rid else False
                self._send_json(
                    200 if cancelled else 404,
                    {"status": "cancelled" if cancelled else "not_found", "request_id": rid},
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_GET_ws(self) -> bool:
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
            # Bound the session so a dead/silent client can't pin a handler
            # thread forever. Setting the socket timeout propagates to the
            # rfile/wfile buffered file objects used by the frame reader.
            self.request.settimeout(WS_SOCKET_TIMEOUT_SECONDS)
            _serve_stream_session(self.rfile, self.wfile, server, self.client_address)
            return True

        def do_GET(self) -> None:  # noqa: N802
            if self.do_GET_ws():
                return
            if self.path in ("/", "/health"):
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": server.model_slug,
                        "loaded": server.loaded,
                        "busy": server.is_busy(),
                        "phase": server.phase(),
                        "queue_depth": server.queue_depth(),
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

    return _Handler


def _run_stdlib_server(server: StarlingServer, host: str, port: int) -> None:
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
        prog="python -m starling.server",
        description="Unified starling ASR server (granite/parakeet/moss/qwen3/ark/cohere/higgs/audex).",
    )
    p.add_argument(
        "--model",
        default="granite",
        choices=list(MODEL_SLUGS),
        help="model to load (default granite)",
    )
    p.add_argument("--host", default=DEFAULT_HOST, help=f"bind host (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"bind port (default {DEFAULT_PORT})")
    p.add_argument(
        "--profile",
        choices=["file", "realtime", "batch", "accuracy"],
        default="file",
        help="named serving profile (default file)",
    )
    p.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=DEFAULT_MAX_CHUNK_SECONDS,
        help=f"max audio chunk length per transcription for chunked backends "
             f"(default {DEFAULT_MAX_CHUNK_SECONDS}s; ignored by parakeet)",
    )
    p.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=DEFAULT_MIN_CHUNK_SECONDS,
        help=f"minimum buffered audio before the first WS /stream partial (default {DEFAULT_MIN_CHUNK_SECONDS}s)",
    )
    p.add_argument(
        "--partial-interval-seconds",
        type=float,
        default=DEFAULT_PARTIAL_INTERVAL_SECONDS,
        help=f"minimum wall-clock gap between WS /stream partials (default {DEFAULT_PARTIAL_INTERVAL_SECONDS}s)",
    )
    p.add_argument(
        "--stream-chunk-seconds", type=float, default=12.0,
        help="fixed WS stream window in seconds; 0 restores whole-buffer mode",
    )
    p.add_argument(
        "--stream-overlap-seconds", type=float, default=3.0,
        help="overlap between fixed WS stream windows (default 3)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"greedy decode budget per chunk for LLM backends (default {DEFAULT_MAX_NEW_TOKENS})",
    )
    p.add_argument(
        "--no-speculative",
        action="store_true",
        help="granite-only: disable self-speculative decoding",
    )
    p.add_argument(
        "--encoder-mode",
        default="cudagraph",
        choices=["cudagraph", "eager", "compile", "triton"],
        help="fused encoder mode (granite; default cudagraph)",
    )
    p.add_argument(
        "--attn-impl",
        default="eager",
        help="global attention implementation (default eager; granite/qwen3 q-former requires eager)",
    )
    p.add_argument(
        "--graph-mode", choices=["auto", "graphed", "eager"], default=None,
        help="override adaptive CUDA graph selection",
    )
    p.add_argument(
        "--file-graph-min-seconds", type=float, default=60.0,
        help="minimum file duration for graph capture in auto mode",
    )
    p.add_argument(
        "--sdpa-attention", action="store_true",
        help="enable the nearly byte-exact shared SDPA attention path",
    )
    p.add_argument(
        "--fp8-weights", action="store_true",
        help="enable tolerance-mode fused fp8 decoder weights",
    )
    p.add_argument(
        "--tolerance-mode", action="store_true",
        help="allow validated non-byte-exact optimizations",
    )
    p.add_argument(
        "--max-upload-mb", type=float, default=DEFAULT_MAX_UPLOAD_BYTES / (1024 * 1024),
        help="maximum HTTP request body in MiB (default 256)",
    )
    p.add_argument(
        "--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="wall-clock request deadline including queue time (default 600; "
             "0 or negative disables)",
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


def _have_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except Exception:
        return False
    return True


def run(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Loads the model, builds the app, and serves forever."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.no_speculative and args.model != "granite":
        log.warning("--no-speculative is granite-only; ignored for model=%s", args.model)

    from .flags import OptFlags

    fp8_models = {"granite", "moss"}
    if args.profile == "batch":
        opt_flags = OptFlags(
            tolerance_mode=True,
            sdpa_attention=True,
            fp8_weights=args.model in fp8_models,
        )
        profile_graph_mode = "graphed"
    elif args.profile == "realtime":
        opt_flags = OptFlags(sdpa_attention=True)
        profile_graph_mode = "graphed"
    else:
        opt_flags = OptFlags()
        profile_graph_mode = "auto"

    if args.fp8_weights and not (args.tolerance_mode or opt_flags.tolerance_mode):
        raise SystemExit("--fp8-weights requires --tolerance-mode (or --profile batch)")
    if args.fp8_weights and args.model not in fp8_models:
        raise SystemExit("--fp8-weights is currently implemented only for granite and moss")
    if args.sdpa_attention:
        opt_flags.sdpa_attention = True
    if args.tolerance_mode:
        opt_flags.tolerance_mode = True
    if args.fp8_weights:
        opt_flags.fp8_weights = True

    config = ServerConfig(
        model=args.model,
        max_chunk_seconds=args.max_chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        partial_interval_seconds=args.partial_interval_seconds,
        stream_chunk_seconds=args.stream_chunk_seconds,
        stream_overlap_seconds=args.stream_overlap_seconds,
        max_new_tokens=args.max_new_tokens,
        speculative=not args.no_speculative,
        warmup=args.warmup,
        encoder_mode=args.encoder_mode,
        use_fused_llm=True,
        attn_impl=args.attn_impl,
        graph_mode=args.graph_mode or profile_graph_mode,
        file_graph_min_seconds=args.file_graph_min_seconds,
        max_upload_bytes=max(1, int(args.max_upload_mb * 1024 * 1024)),
        request_timeout_seconds=args.request_timeout_seconds,
        opt_flags=opt_flags,
    )

    use_fastapi = (not args.stdlib) and _have_fastapi()

    server = StarlingServer(config=config)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "binding unauthenticated ASR endpoints to public/non-loopback host %s",
            args.host,
        )
    if not args.no_eager_load:
        server.load()

    if args.warmup and args.no_eager_load:
        log.warning(
            "--warmup is set but --no-eager-load skips startup warmup; "
            "warmup will run on the first request instead"
        )

    log.info(
        "starting starling server on %s:%d (model=%s, backend=%s, warmup=%s)",
        args.host,
        args.port,
        args.model,
        "fastapi" if use_fastapi else "stdlib",
        config.warmup,
    )

    if use_fastapi:
        import uvicorn

        app = create_app(server=server, load_on_startup=not args.no_eager_load)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    else:
        _run_stdlib_server(server, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
