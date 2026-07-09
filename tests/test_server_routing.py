"""CPU-safe tests for the unified server's model routing.

These exercise the pure, GPU-free parts of :mod:`starling.server`: the slug ->
backend class resolution, config plumbing, the CLI arg parser, and error
handling for unknown models. No model is loaded and no CUDA is required, so the
test runs in any environment (CI / CPU-only).

The end-to-end transcribe paths are covered by the per-model pipeline tests
(``test_smoke``, ``test_qwen3_pipeline``, ...) and the live ``test_server.py``
script; here we only assert routing wires up the right backend for each slug.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from starling.server import (  # noqa: E402
    MODEL_SLUGS,
    ArkBackend,
    CohereBackend,
    GraniteBackend,
    HiggsBackend,
    MossBackend,
    ModelBackend,
    ParakeetBackend,
    ParakeetUnifiedBackend,
    Qwen3Backend,
    SAMPLE_RATE,
    ServerConfig,
    StarlingServer,
    create_app,
    get_backend,
)


def test_model_slugs_are_the_supported_set() -> None:
    assert set(MODEL_SLUGS) == {
        "granite", "parakeet", "parakeet_unified", "moss", "qwen3", "ark",
        "cohere", "higgs",
    }


def test_get_backend_resolves_each_slug_to_the_right_class() -> None:
    cfg = ServerConfig()
    assert isinstance(get_backend("granite", cfg), GraniteBackend)
    assert isinstance(get_backend("parakeet", cfg), ParakeetBackend)
    assert isinstance(get_backend("parakeet_unified", cfg), ParakeetUnifiedBackend)
    assert isinstance(get_backend("moss", cfg), MossBackend)
    assert isinstance(get_backend("qwen3", cfg), Qwen3Backend)
    assert isinstance(get_backend("ark", cfg), ArkBackend)
    assert isinstance(get_backend("cohere", cfg), CohereBackend)
    assert isinstance(get_backend("higgs", cfg), HiggsBackend)


def test_get_backend_unknown_slug_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_backend("whisper", ServerConfig())


def test_backend_slug_round_trips() -> None:
    """Each backend class advertises the slug that resolves to it."""
    cfg = ServerConfig()
    for slug in MODEL_SLUGS:
        assert get_backend(slug, cfg).slug == slug


def test_server_config_carries_model_slug() -> None:
    for slug in MODEL_SLUGS:
        srv = StarlingServer(config=ServerConfig(model=slug))
        assert srv.model_slug == slug
        # not loaded yet -> backend is None and load() has not run
        assert srv.backend is None
        assert srv.loaded is False


def test_cli_arg_parser_accepts_each_model() -> None:
    from starling.server import _build_arg_parser

    parser = _build_arg_parser()
    for slug in MODEL_SLUGS:
        args = parser.parse_args(["--model", slug])
        assert args.model == slug
    # default is granite
    assert parser.parse_args([]).model == "granite"


def test_cli_arg_parser_rejects_unknown_model() -> None:
    from starling.server import _build_arg_parser

    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "nope"])


def test_parakeet_unified_long_audio_uses_chunker() -> None:
    class _Pipe:
        tokenizer = object()

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_list):  # noqa: ANN001
            self.calls += 1
            return ["one-shot"]

    class _Chunker:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio, sr=SAMPLE_RATE, should_stop=None):  # noqa: ANN001
            self.calls += 1
            assert sr == SAMPLE_RATE
            assert should_stop is not None
            return "chunked"

    backend = ParakeetUnifiedBackend(ServerConfig(max_chunk_seconds=1.0))
    pipe = _Pipe()
    chunker = _Chunker()
    backend.pipe = pipe
    backend.chunker = chunker

    short = backend.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
    long = backend.transcribe(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))

    assert short.text == "one-shot"
    assert long.text == "chunked"
    assert pipe.calls == 1
    assert chunker.calls == 1


def test_create_app_reuses_existing_server_without_startup_load() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")

    server = StarlingServer(config=ServerConfig(model="granite"))
    app = create_app(server=server, load_on_startup=False)

    assert app.state.starling_server is server


@pytest.mark.parametrize("backend_cls", [MossBackend, Qwen3Backend, ArkBackend, HiggsBackend])
def test_single_shot_llm_backends_chunk_long_audio(backend_cls) -> None:
    backend = backend_cls(ServerConfig(max_chunk_seconds=1.0, max_new_tokens=200))
    calls: list[tuple[int, int]] = []

    def fake_chunk(audio, budget):  # noqa: ANN001
        calls.append((len(audio), budget))
        return f"chunk-{len(calls)}"

    backend._transcribe_chunk = fake_chunk
    result = backend.transcribe(np.zeros(int(2.5 * SAMPLE_RATE), dtype=np.float32))

    assert [size for size, _ in calls] == [SAMPLE_RATE, SAMPLE_RATE, SAMPLE_RATE // 2]
    assert [segment["end_s"] for segment in result.segments] == [1.0, 2.0, 2.5]
    assert result.text == "chunk-1 chunk-2 chunk-3"
    assert calls[-1][1] < calls[0][1]


def test_chunked_backend_checks_cancellation_between_chunks() -> None:
    backend = ModelBackend(ServerConfig(max_chunk_seconds=1.0))
    backend._cancel_event = threading.Event()

    def cancel_after_first(_audio, _budget):  # noqa: ANN001
        backend._cancel_event.set()
        return "first"

    from starling.server import _Cancelled

    with pytest.raises(_Cancelled):
        backend._transcribe_chunked(
            np.zeros(2 * SAMPLE_RATE, dtype=np.float32), cancel_after_first
        )


def test_anonymous_requests_receive_unique_registry_ids(monkeypatch) -> None:
    server = StarlingServer()
    seen: list[str] = []

    def fake_run(ctx, _samples, *, streaming=False):  # noqa: ANN001, ARG001
        seen.append(ctx.id)
        return __import__("starling.server", fromlist=["TranscribeResult"]).TranscribeResult(text="ok")

    monkeypatch.setattr(server, "_serial_run", fake_run)
    samples = np.zeros(1, dtype=np.float32)
    server._run_queued_sync(samples, None)
    server._run_queued_sync(samples, None)

    assert len(seen) == 2
    assert all(isinstance(rid, str) and rid for rid in seen)
    assert seen[0] != seen[1]


def test_running_cancellation_returns_cancelled(monkeypatch) -> None:
    from starling import server as server_module

    class FakeBackend:
        _cancel_event = None

    server = StarlingServer(backend=FakeBackend(), _loaded=True)

    def finish_after_cancel(self, samples, streaming=False):  # noqa: ANN001, ARG001
        self._requests["request"].cancel.set()
        return server_module.TranscribeResult(text="too late")

    monkeypatch.setattr(
        server_module.StarlingServer,
        "_transcribe_np",
        finish_after_cancel,
    )
    import starling.parakeet.gpu_lock as gpu_lock

    monkeypatch.setattr(gpu_lock, "acquire_gpu_lock", lambda **kwargs: "owner")
    monkeypatch.setattr(gpu_lock, "release_gpu_lock", lambda owner=None: True)

    with pytest.raises(server_module._Cancelled):
        server._run_queued_sync(np.zeros(1, dtype=np.float32), "request")


def test_request_deadline_is_checked_before_gpu_work() -> None:
    from starling.server import RequestContext, _DeadlineExceeded

    server = StarlingServer()
    ctx = RequestContext("late", deadline=time.monotonic() - 1.0)
    with pytest.raises(_DeadlineExceeded):
        server._raise_if_stopped(ctx)


def test_resampler_filters_aliases_when_downsampling() -> None:
    from starling.server import _resample_audio

    sr_in = 48000
    t = np.arange(sr_in, dtype=np.float32) / sr_in
    above_nyquist = np.sin(2 * np.pi * 12000 * t).astype(np.float32)
    resampled = _resample_audio(above_nyquist, sr_in, SAMPLE_RATE)

    assert len(resampled) == SAMPLE_RATE
    assert float(np.sqrt(np.mean(resampled**2))) < 0.05


def test_cli_exposes_server_tuning_and_limits() -> None:
    from starling.server import _build_arg_parser

    args = _build_arg_parser().parse_args(
        [
            "--profile", "realtime",
            "--graph-mode", "eager",
            "--file-graph-min-seconds", "90",
            "--stream-chunk-seconds", "10",
            "--stream-overlap-seconds", "2",
            "--max-upload-mb", "12",
            "--request-timeout-seconds", "30",
        ]
    )
    assert args.profile == "realtime"
    assert args.graph_mode == "eager"
    assert args.file_graph_min_seconds == 90
    assert args.stream_chunk_seconds == 10
    assert args.stream_overlap_seconds == 2
    assert args.max_upload_mb == 12
    assert args.request_timeout_seconds == 30
