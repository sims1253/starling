"""CPU-only tests for the server load/warmup lifecycle (issue #20 hardening).

No GPU, no model download, no network, no live socket: backend loading and GPU
work are faked/monkeypatched, so these run in the CPU CI matrix alongside
``test_server_routing.py`` / ``test_server_robustness.py``. Guards against
regressions of the lifecycle fixes:

* 1. ``Qwen3Backend.load`` constructs the pipeline from the model it just
     loaded instead of re-loading via ``MegaPipeline.from_pretrained()``.
* 2. Health accessors (``loaded`` / ``phase`` / ``queue_depth`` / ``is_busy``)
     never block behind an in-flight model load.
* 3. Concurrent ``load()`` calls still run the backend load exactly once.
* 4. A failed warmup resets ``phase`` to ``ready`` and is swallowed+logged;
     ``load()`` with ``--warmup`` survives a warmup failure.
* 5. ``POST /warmup`` (FastAPI + stdlib transports) rejects an unloaded model
     with 409 instead of a no-op 202.
"""

from __future__ import annotations

import asyncio
from email.message import Message
import json
import os
import sys
import threading
import time
import types
from typing import Any, NoReturn

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from starling.server import (  # noqa: E402
    Qwen3Backend,
    ServerConfig,
    StarlingServer,
    _build_stdlib_handler,
    create_app,
)
import starling.server as server_module  # noqa: E402


# ---------------------------------------------------------------------------
# shared fakes
# ---------------------------------------------------------------------------
class _FakeBackend:
    """Minimal loaded-backend stand-in (no GPU surface beyond the test needs)."""

    _cancel_event: threading.Event | None = None
    _deadline: float = float("inf")

    def load(self) -> None:
        pass

    def set_graph_mode(self, **_kwargs: Any) -> None:
        pass


def _patch_gpu_lock_noop(monkeypatch) -> None:  # noqa: ANN001
    """Make the cross-process GPU lock a fast no-op (no flock / nvidia-smi)."""
    import starling.parakeet.gpu_lock as gpu_lock

    monkeypatch.setattr(gpu_lock, "acquire_gpu_lock", lambda **kwargs: "owner")
    monkeypatch.setattr(gpu_lock, "release_gpu_lock", lambda owner=None: True)


# ---------------------------------------------------------------------------
# 1. Qwen3Backend loads the model exactly once
# ---------------------------------------------------------------------------
def test_qwen3_backend_load_reuses_loaded_model(monkeypatch) -> None:  # noqa: ANN001
    """load() must build the pipeline from the (model, processor) pair it just
    loaded, forwarding encoder_mode/use_fused_llm -- NOT call
    MegaPipeline.from_pretrained(), which discarded that pair and re-loaded the
    model a second time with default attn/encoder settings.

    The real starling.qwen3 modules are replaced via sys.modules: importing
    them here would load model dependencies into a CPU-only test process
    for no coverage gain.
    """
    model_sentinel = object()
    processor_sentinel = object()
    constructs: list[tuple[Any, Any, dict[str, Any]]] = []
    loader_calls: list[dict[str, Any]] = []

    class _FakeMegaPipeline:
        def __init__(self, model: Any, processor: Any, **kwargs: Any) -> None:
            constructs.append((model, processor, kwargs))
            self.built = True

        @classmethod
        def from_pretrained(cls, **_kwargs: Any) -> NoReturn:
            raise AssertionError("from_pretrained() must not be called (double load)")

    def fake_load_model_and_processor(**kwargs: Any) -> tuple[Any, Any]:
        loader_calls.append(kwargs)
        return model_sentinel, processor_sentinel

    fake_pkg = types.ModuleType("starling.qwen3")
    fake_loader = types.ModuleType("starling.qwen3.loader")
    fake_loader.load_model_and_processor = fake_load_model_and_processor
    fake_pipeline = types.ModuleType("starling.qwen3.pipeline")
    fake_pipeline.MegaPipeline = _FakeMegaPipeline
    monkeypatch.setitem(sys.modules, "starling.qwen3", fake_pkg)
    monkeypatch.setitem(sys.modules, "starling.qwen3.loader", fake_loader)
    monkeypatch.setitem(sys.modules, "starling.qwen3.pipeline", fake_pipeline)

    # Non-default values so forwarding regressions cannot hide behind defaults.
    backend = Qwen3Backend(
        ServerConfig(model="qwen3", attn_impl="sdpa", encoder_mode="eager", use_fused_llm=False)
    )
    backend.load()

    assert backend.pipe is not None and backend.pipe.built
    assert backend.processor is processor_sentinel
    assert len(constructs) == 1  # the model was loaded exactly once
    assert loader_calls == [{"attn_impl": "sdpa"}]  # --attn-impl forwarded
    model, processor, kwargs = constructs[0]
    assert model is model_sentinel
    assert processor is processor_sentinel
    assert kwargs["encoder_mode"] == "eager"
    assert kwargs["use_fused_llm"] is False


# ---------------------------------------------------------------------------
# 2. Health accessors never block behind a model load
# ---------------------------------------------------------------------------
def test_health_accessors_respond_during_model_load(monkeypatch) -> None:  # noqa: ANN001
    """loaded()/phase() must answer while load() is mid-flight.

    Regression: load() ran under ``_lock``, the same lock the health accessors
    take -- on the FastAPI transport the health body runs on the event loop,
    so an in-flight (lazy) load froze the whole server, health checks included.
    """
    server = StarlingServer(config=ServerConfig(model="granite", warmup=False))
    in_load = threading.Event()
    release_load = threading.Event()

    class _BlockingBackend:
        def load(self) -> None:
            in_load.set()
            assert release_load.wait(timeout=10.0), "test never released the load"

    monkeypatch.setattr(
        server_module, "get_backend", lambda slug, cfg: _BlockingBackend()
    )

    loader = threading.Thread(target=server.load, daemon=True)
    loader.start()
    assert in_load.wait(timeout=5.0), "backend.load() never started"

    # Run the accessors in their own threads so a regression manifests as a
    # failed bounded wait (clear failure) instead of a hung test process.
    observed: dict[str, Any] = {}
    accessors_done = threading.Event()

    def probe_accessors() -> None:
        observed["phase"] = server.phase()
        observed["loaded"] = server.loaded
        observed["busy"] = server.is_busy()
        observed["queue_depth"] = server.queue_depth()
        accessors_done.set()

    probe = threading.Thread(target=probe_accessors, daemon=True)
    probe.start()
    assert accessors_done.wait(timeout=2.0), (
        "health accessors blocked behind the in-flight model load"
    )
    assert observed["phase"] == "loading_weights"
    assert observed["loaded"] is False

    release_load.set()
    loader.join(timeout=5.0)
    assert not loader.is_alive()

    assert server.loaded is True
    assert server.phase() == "ready"


# ---------------------------------------------------------------------------
# 3. Concurrent load() calls run the backend load exactly once
# ---------------------------------------------------------------------------
def test_concurrent_loads_run_backend_load_once(monkeypatch) -> None:  # noqa: ANN001
    """Two overlapping load() calls serialize on the load lock: the backend's
    (heavy) load runs exactly once and both callers observe a loaded server."""
    server = StarlingServer(config=ServerConfig(model="granite", warmup=False))
    in_load = threading.Event()
    release_load = threading.Event()
    loads = {"n": 0}

    class _CountingBackend:
        def load(self) -> None:
            loads["n"] += 1
            in_load.set()
            assert release_load.wait(timeout=10.0), "test never released the load"

    monkeypatch.setattr(server_module, "get_backend", lambda slug, cfg: _CountingBackend())

    # Hold the first load in-flight long enough for the second caller to pile
    # up behind the load lock (deterministic, not timing-dependent).
    first = threading.Thread(target=server.load, daemon=True)
    first.start()
    assert in_load.wait(timeout=5.0)
    second = threading.Thread(target=server.load, daemon=True)
    second.start()
    time.sleep(0.1)  # let the second caller actually reach the lock
    release_load.set()

    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive() and not second.is_alive()

    assert loads["n"] == 1  # idempotent under concurrency
    assert server.loaded is True
    assert server.phase() == "ready"


# ---------------------------------------------------------------------------
# 4. Warmup failure handling
# ---------------------------------------------------------------------------
def test_warmup_failure_resets_phase_and_is_logged(monkeypatch, caplog) -> None:  # noqa: ANN001
    """A warmup exception must not escape, must leave phase at ``ready`` (not
    stuck at ``warming_up``), and must clear the in-progress dedup flag."""
    _patch_gpu_lock_noop(monkeypatch)
    server = StarlingServer(backend=_FakeBackend(), _loaded=True)

    def failing_transcribe(
        self, _samples: np.ndarray, *, _streaming: bool = False  # noqa: ANN001, ARG001
    ) -> None:
        raise RuntimeError("simulated graph-capture OOM")

    monkeypatch.setattr(StarlingServer, "_transcribe_np", failing_transcribe)

    with caplog.at_level("ERROR", logger="starling.server"):
        server.warmup()  # must not raise

    assert server.phase() == "ready"
    assert server._warmup_in_progress is False
    assert any("warmup failed" in rec.message for rec in caplog.records)


def test_warmup_success_still_reaches_ready(monkeypatch) -> None:  # noqa: ANN001
    """Sanity guard for the surrounding refactor: a successful warmup keeps the
    established phase transition warming_up -> ready."""
    _patch_gpu_lock_noop(monkeypatch)
    server = StarlingServer(backend=_FakeBackend(), _loaded=True)

    def ok_transcribe(
        self, _samples: np.ndarray, *, _streaming: bool = False  # noqa: ANN001, ARG001
    ) -> "server_module.TranscribeResult":
        return server_module.TranscribeResult(text="warm")

    monkeypatch.setattr(StarlingServer, "_transcribe_np", ok_transcribe)
    server.warmup()

    assert server.phase() == "ready"


def test_load_survives_warmup_failure(monkeypatch) -> None:  # noqa: ANN001
    """load() with config.warmup=True completes even when graph capture fails:
    the weights ARE loaded, so the server must come up usable (phase ready),
    not crash at startup."""
    _patch_gpu_lock_noop(monkeypatch)
    server = StarlingServer(config=ServerConfig(model="granite", warmup=True))
    monkeypatch.setattr(
        server_module, "get_backend", lambda slug, cfg: _FakeBackend()
    )

    def failing_transcribe(
        self, _samples: np.ndarray, *, _streaming: bool = False  # noqa: ANN001, ARG001
    ) -> None:
        raise RuntimeError("simulated graph-capture OOM")

    monkeypatch.setattr(StarlingServer, "_transcribe_np", failing_transcribe)

    server.load()  # must not raise

    assert server.loaded is True
    assert server.phase() == "ready"


# ---------------------------------------------------------------------------
# 5. POST /warmup on an unloaded model -> 409 (both transports)
# ---------------------------------------------------------------------------
def _warmup_route(app: Any) -> Any:
    return next(r for r in app.routes if getattr(r, "path", None) == "/warmup")


def test_fastapi_warmup_route_rejects_unloaded_model() -> None:
    """The FastAPI /warmup endpoint returns 409 (not a no-op 202) while the
    model is unloaded. Invokes the endpoint coroutine directly: the response
    object carries status + body without needing an HTTP client."""
    pytest.importorskip("fastapi")

    server = StarlingServer(config=ServerConfig(model="granite"))
    app = create_app(server=server, load_on_startup=False)

    response = asyncio.run(_warmup_route(app).endpoint())

    assert response.status_code == 409
    body = json.loads(response.body)
    assert body["error"] == "model not loaded"
    assert body["phase"] == "unloaded"


def test_fastapi_warmup_route_dispatches_warmup_when_loaded() -> None:
    """Loaded server: the endpoint answers 202 and the fire-and-forget task
    really runs warmup. The scenario keeps the loop alive until the dispatched
    task completes, so asyncio.run() cannot cancel it mid-flight."""
    pytest.importorskip("fastapi")

    server = StarlingServer(backend=_FakeBackend(), _loaded=True)
    warmup_calls: list[None] = []
    server.warmup = lambda: warmup_calls.append(None)  # type: ignore[method-assign]
    app = create_app(server=server, load_on_startup=False)

    async def scenario() -> Any:
        response = await _warmup_route(app).endpoint()
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*pending)
        return response

    response = asyncio.run(scenario())

    assert response.status_code == 202
    body = json.loads(response.body)
    assert body["status"] == "warmup started"
    assert len(warmup_calls) == 1


def _stdlib_handler_for(server: StarlingServer, path: str) -> Any:  # noqa: ANN001
    """Instantiate the stdlib POST handler without a socket: do_POST only
    touches self.path / self.headers / self._send_json."""
    handler = object.__new__(_build_stdlib_handler(server))
    handler.path = path
    handler.headers = Message()  # Content-Length absent -> body length 0
    return handler


def test_stdlib_warmup_route_rejects_unloaded_model() -> None:
    """The stdlib transport mirrors the FastAPI 409 for /warmup on an unloaded
    model instead of answering a misleading 202."""
    server = StarlingServer(config=ServerConfig(model="granite"))
    handler = _stdlib_handler_for(server, "/warmup")

    sent: list[tuple[int, dict[str, Any]]] = []
    handler._send_json = lambda status, obj: sent.append((status, obj))  # type: ignore[method-assign]

    handler.do_POST()

    assert sent == [(409, {"error": "model not loaded", "phase": "unloaded"})]


def test_stdlib_warmup_route_starts_warmup_when_loaded(monkeypatch) -> None:  # noqa: ANN001
    """Loaded server: the 202 path still dispatches a real warmup thread."""
    server = StarlingServer(backend=_FakeBackend(), _loaded=True)
    warmup_called = threading.Event()
    monkeypatch.setattr(server, "warmup", warmup_called.set)

    handler = _stdlib_handler_for(server, "/warmup")
    sent: list[tuple[int, dict[str, Any]]] = []
    handler._send_json = lambda status, obj: sent.append((status, obj))  # type: ignore[method-assign]

    handler.do_POST()

    assert sent[0][0] == 202
    assert sent[0][1]["status"] == "warmup started"
    assert warmup_called.wait(timeout=2.0), "warmup was never dispatched"
