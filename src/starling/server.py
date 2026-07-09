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
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
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

WS_GUID: bytes = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Supported model slugs -> (backend class, display name, gpu-lock model label).
# Built lazily as backend classes are defined below.
MODEL_SLUGS = ("granite", "parakeet", "parakeet_unified", "moss", "qwen3", "ark", "cohere", "higgs")


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

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        audio_seconds = len(audio) / SAMPLE_RATE
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

            chunk_seconds = max(
                0.1,
                float(self.config.max_chunk_seconds or DEFAULT_MAX_CHUNK_SECONDS),
            )
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
        max_chunk_seconds = float(
            self.config.max_chunk_seconds or DEFAULT_MAX_CHUNK_SECONDS
        )
        if audio_seconds > max_chunk_seconds:
            text = self._get_chunker().transcribe(audio, sr=SAMPLE_RATE)
        else:
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
        # fp8 decode (+20%%, ~0%% WER) is deliberately NOT enabled: torch._scaled_mm
        # in a captured graph corrupts under sustained streaming churn.  It stays
        # opt-in for one-off batch jobs, where it is stable.
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

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        assert self.pipe is not None and self.processor is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        audio_seconds = len(audio) / SAMPLE_RATE

        inp = self._build(audio)
        text, _ = self.pipe.transcribe(
            inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
            inp["audio_input_mask"], max_new_tokens=self.config.max_new_tokens,
        )
        return TranscribeResult(
            text=text,
            segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
            duration_s=audio_seconds,
        )


class Qwen3Backend(ModelBackend):
    """qwen3-asr-1.7b: processor-built inputs, chunked transcribe."""

    slug = "qwen3"

    def load(self) -> None:
        from .qwen3.audio import build_inputs  # noqa: F401
        from .qwen3.loader import load_model_and_processor
        from .qwen3.pipeline import MegaPipeline

        model, processor = load_model_and_processor(attn_impl=self.config.attn_impl)
        self.pipe = MegaPipeline.from_pretrained()
        self.processor = processor
        self._build_inputs = build_inputs

    def set_graph_mode(self, *, streaming: bool, duration_s: float = 0.0) -> None:
        # Single-shot per request (like ark): graphed prefill only for streaming.
        self.pipe.set_prefill_use_graph(
            self._want_graphed(streaming=streaming, duration_s=duration_s, chunked=False)
        )

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        import torch

        from .qwen3.audio import build_inputs

        assert self.pipe is not None and self.processor is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = torch.from_numpy(np.ascontiguousarray(samples)).float().unsqueeze(0).contiguous()
        audio_seconds = wav.shape[1] / SAMPLE_RATE
        inp = build_inputs(self.processor, wav, sr=SAMPLE_RATE)
        text, _ = self.pipe.transcribe(
            inp["input_features"], inp["input_ids"],
            inp.get("input_features_mask"),
            max_new_tokens=self.config.max_new_tokens,
        )
        return TranscribeResult(
            text=text,
            segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
            duration_s=audio_seconds,
        )


class ArkBackend(ModelBackend):
    """ark-asr-3b: Whisper+adapter encoder + Qwen2.5 decoder megakernel.

    The pipeline's ``transcribe`` takes a 1-D float32 waveform directly (no
    processor-built input dict), so this backend is the thinnest of the LLM
    backends. Single-shot (no chunker): bounded by the 4096-token static KV
    cache, like qwen3.
    """

    slug = "ark"

    def load(self) -> None:
        from .ark.pipeline import MegaPipeline

        self.pipe = MegaPipeline.from_pretrained()

    def set_graph_mode(self, *, streaming: bool, duration_s: float = 0.0) -> None:
        # Single-shot per request: graphed prefill only amortises across repeated
        # stream chunks, so file requests stay eager regardless of length.
        self.pipe.set_prefill_use_graph(
            self._want_graphed(streaming=streaming, duration_s=duration_s, chunked=False)
        )

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = np.ascontiguousarray(samples, dtype=np.float32)
        audio_seconds = wav.shape[0] / SAMPLE_RATE
        text, _ids = self.pipe.transcribe(
            wav, max_new_tokens=self.config.max_new_tokens,
        )
        return TranscribeResult(
            text=text,
            segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
            duration_s=audio_seconds,
        )


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

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = np.ascontiguousarray(samples, dtype=np.float32)
        audio_seconds = wav.shape[0] / SAMPLE_RATE
        texts, _ids = self.pipe.transcribe(
            wav, max_new_tokens=self.config.max_new_tokens,
        )
        text = " ".join(t.strip() for t in texts if t.strip())
        return TranscribeResult(
            text=text,
            segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
            duration_s=audio_seconds,
        )


class HiggsBackend(ModelBackend):
    """higgs-audio-v3-stt: Whisper-large-v3 mel + Qwen3-1.7B decoder megakernel.

    NOTE: like the benchmark adapter, this only loads under the isolated
    ``.venv-higgs`` (``transformers==4.51``); see ``higgs/UV_NOTES.md``.
    """

    slug = "higgs"

    def load(self) -> None:
        from .higgs.pipeline import HiggsMega

        self.pipe = HiggsMega.from_pretrained()

    def transcribe(self, samples: np.ndarray) -> "TranscribeResult":
        assert self.pipe is not None
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        wav = np.ascontiguousarray(samples, dtype=np.float32)
        audio_seconds = wav.shape[0] / SAMPLE_RATE
        text = self.pipe.transcribe(
            wav, sample_rate=SAMPLE_RATE, max_new_tokens=self.config.max_new_tokens,
        )
        return TranscribeResult(
            text=text,
            segments=[{"text": text, "start_s": 0.0, "end_s": audio_seconds}],
            duration_s=audio_seconds,
        )


_BACKENDS: dict[str, type[ModelBackend]] = {
    "granite": GraniteBackend,
    "parakeet": ParakeetBackend,
    "parakeet_unified": ParakeetUnifiedBackend,
    "moss": MossBackend,
    "qwen3": Qwen3Backend,
    "ark": ArkBackend,
    "cohere": CohereBackend,
    "higgs": HiggsBackend,
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


@dataclass
class RequestContext:
    """Per-request handle tracked in the server's registry for cancellation."""

    id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    state: str = "queued"


class _Busy(Exception):
    """Internal sentinel: queue is full (backpressure) — client should retry."""


class _Cancelled(Exception):
    """Internal sentinel: request was cancelled via :meth:`StarlingServer.cancel_request`."""


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

    # --- lifecycle phase (reported by /health) ----------------------------
    #   unloaded -> loading_weights -> warming_up -> ready
    _phase: str = "unloaded"

    @property
    def model_slug(self) -> str:
        return self.config.model

    @property
    def loaded(self) -> bool:
        """True once the model backend has been loaded into VRAM."""
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
        """Capture CUDA graphs on a short silent clip (no-op if not loaded)."""
        if not self._loaded or self.backend is None:
            return
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
            samples = _resample_linear(samples, sr, SAMPLE_RATE)
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
        ctx = RequestContext(request_id)
        with self._lock:
            if self._n_waiters >= MAX_WAITERS:
                raise _Busy()
            self._n_waiters += 1
            self._requests[ctx.id] = ctx
        try:
            return self._serial_run(ctx, samples, streaming=streaming)
        finally:
            with self._lock:
                self._n_waiters = max(0, self._n_waiters - 1)
                self._requests.pop(ctx.id, None)

    def _serial_run(
        self, ctx: "RequestContext", samples: np.ndarray, *, streaming: bool = False
    ) -> TranscribeResult:
        from .parakeet.gpu_lock import GpuLockBusy, acquire_gpu_lock, release_gpu_lock

        while True:
            if ctx.cancel.is_set():
                raise _Cancelled()
            try:
                acquire_gpu_lock(
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

        ctx.state = "running"
        try:
            return self._transcribe_np(samples, streaming=streaming)
        finally:
            release_gpu_lock()

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
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _resample_linear(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)


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
    last_partial_text: str = ""
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

    def append_pcm(self, pcm16_bytes: bytes) -> None:
        s = _pcm16_bytes_to_float32(pcm16_bytes)
        if s.size > 0:
            self.samples = np.concatenate([self.samples, s]) if self.samples.size else s

    def append_wav(self, wav_bytes: bytes) -> None:
        try:
            s, _sr = _wav_bytes_to_float32(wav_bytes)
        except Exception:
            self.append_pcm(wav_bytes)
            return
        if s.size > 0:
            self.samples = np.concatenate([self.samples, s]) if self.samples.size else s

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
        self.last_partial_text = ""
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
    app = FastAPI(title="starling-server", version="2.0.0")
    app.state.starling_server = server  # type: ignore[attr-defined]

    @app.on_event("startup")
    async def _on_startup() -> None:  # pragma: no cover - exercised by run()
        if load_on_startup:
            await asyncio.to_thread(server.load)

    async def _decode_inference_body(request: "Request") -> bytes:
        body = await request.body()
        if not body:
            return b""
        ctype = request.headers.get("content-type", "")
        if "multipart/form-data" in ctype:
            return _extract_multipart_payload(body, ctype)
        return body

    def _request_id(request: "Request") -> Optional[str]:
        rid = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
        return rid or None

    def _health_body() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": server.model_slug,
            "loaded": server._loaded,
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
        await asyncio.to_thread(server.warmup)
        return JSONResponse({"status": "ok", "phase": server.phase()})

    async def _inference(request):  # noqa: ANN001
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

    async def _transcribe(request):  # noqa: ANN001
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
                        sess.last_partial_text = text
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

    _stream.__annotations__["ws"] = WebSocket
    app.add_api_websocket_route("/stream", _stream)

    return app


# ===========================================================================
# BACKEND B: stdlib-only (http.server + minimal RFC 6455 WebSocket)
# ===========================================================================
def _extract_multipart_payload(body: bytes, content_type: str) -> bytes:
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
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if b"\r\n\r\n" in part:
            _headers, payload = part.split(b"\r\n\r\n", 1)
            return payload
        return part
    return body


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
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", _read_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _read_exact(8))[0]
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
                    sess.last_partial_text = text
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
                        "loaded": server._loaded,
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
        description="Unified starling ASR server (granite/parakeet/moss/qwen3/ark/cohere/higgs).",
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

    config = ServerConfig(
        model=args.model,
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

    server = StarlingServer(config=config)
    if not args.no_eager_load:
        server.load()

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
