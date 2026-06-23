"""Shared pytest configuration for the starling test suite.

Three concerns are handled here:

1. **Per-test GPU teardown** (autouse, function scope).
   After every test we ``gc.collect()`` + ``torch.cuda.empty_cache()`` so
   intermediate tensors / CUDA-graph capture buffers / Triton autotune caches
   do not accumulate across tests within a module.

2. **Per-module model-cache teardown** (autouse, module scope).
   Almost every heavy test file stashes the model it loads in module-level
   globals (``_MODEL``/``_PROC`` for granite, ``_MODEL_CACHE`` for the encoder
   test, ``_PIPELINES``/``_PIPE``/``_PIPES`` for the parakeet pipeline tests,
   ``_STATE``/``_RUNNER`` for the decoder/mel/smoke tests). Because pytest keeps
   collected modules alive in ``sys.modules`` for the whole session, those refs
   pin a ~5GB model per file (and ~2GB per parakeet pipeline) until the process
   exits -- which is exactly what produced the 32GB VRAM bleed.

   The fix: after a module's last test finishes, nullify the known cache
   globals for that module and free the cache. The next test file then reloads
   its own model fresh, so only one heavy model is resident at a time.

3. **Markers**.
   * ``compile`` -- slow ``torch.compile(mode="max-autotune")`` benchmark
     tests. Skipped by default; opt in with ``--runcompile``.
   * ``slow``    -- perf-gate tests that are flaky under GPU contention
     (e.g. multi-step vs single-step, which is only ~2% apart). Skipped by
     default; opt in with ``--runslow``.

These two markers keep the default correctness suite fast (~minutes, not 10+)
and green under GPU contention, while leaving the expensive checks runnable on
an idle GPU via the opt-in flags.
"""

from __future__ import annotations

import gc

import pytest

# ``torch`` is imported lazily so that simply collecting the suite (e.g.
# ``pytest --co``) does not hard-require a working CUDA runtime.
_TORCH = None  # type: object | None  # cached module or False once tried


def _import_torch():
    """Return the ``torch`` module, or ``None`` if it / CUDA is unavailable."""
    global _TORCH
    if _TORCH is None:
        try:
            import torch as _t

            _TORCH = _t
        except Exception:  # noqa: BLE001 -- collection must never crash
            _TORCH = False
    return _TORCH if _TORCH is not False else None


def _free_gpu() -> None:
    """Aggressively release GPU memory (gc + cuda empty_cache + peak reset)."""
    gc.collect()
    t = _import_torch()
    if t is not None and t.cuda.is_available():
        t.cuda.synchronize()
        t.cuda.empty_cache()
        t.cuda.reset_peak_memory_stats()


# Module-level globals that the test files use to cache loaded models / heavy
# state across the tests in that module. Nullifying them after a module's last
# test lets ``gc`` reclaim the model before the next test file loads its own.
_MODULE_CACHE_GLOBALS = (
    # granite model + processor (per-file caches; each holds its own ~5GB copy)
    "_MODEL",
    "_PROC",
    # encoder mega test
    "_MODEL_CACHE",
    "_INPUT_FEATURES",
    "_GOLDEN",
    # end-to-end pipeline / speculative / batched tests
    "_INPUTS",
    "_WAV",
    "_SR",
    # parakeet pipeline tests (each pipeline ~2GB)
    "_PIPELINES",
    "_PIPE",
    "_PIPES",
    # decoder / mel / smoke heavy state
    "_STATE",
    "_RUNNER",
)


# --------------------------------------------------------------------------- #
# GPU teardown fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _gpu_teardown_per_test():
    """Free intermediate GPU tensors / graphs after every test."""
    yield
    _free_gpu()


@pytest.fixture(scope="module", autouse=True)
def _drop_module_model_cache(request):
    """After a test module finishes, drop its model-caching module globals.

    This is the actual fix for the cross-file VRAM leak: without it, each
    granite test file pins its own ~5GB model in a module global for the whole
    session (8 files -> ~40GB if all alive). We nullify the known cache names
    after the module's last test so the next file reloads fresh.
    """
    yield
    mod = getattr(request, "module", None)
    if mod is not None:
        for attr in _MODULE_CACHE_GLOBALS:
            if hasattr(mod, attr):
                setattr(mod, attr, None)
    _free_gpu()


# --------------------------------------------------------------------------- #
# markers + opt-in flags
# --------------------------------------------------------------------------- #
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "compile: slow torch.compile / max-autotune benchmark tests "
        "(skipped by default; run with --runcompile)",
    )
    config.addinivalue_line(
        "markers",
        "slow: slow or contention-flaky perf-gate tests "
        "(skipped by default; run with --runslow)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runcompile",
        action="store_true",
        default=False,
        help="run slow torch.compile / max-autotune benchmark tests",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow / contention-flaky perf-gate tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``compile`` / ``slow`` tests unless their opt-in flag is passed."""
    run_compile = bool(config.getoption("--runcompile"))
    run_slow = bool(config.getoption("--runslow"))
    skip_compile = pytest.mark.skip(
        reason="needs --runcompile (slow torch.compile benchmark)"
    )
    skip_slow = pytest.mark.skip(
        reason="needs --runslow (slow / contention-flaky perf gate)"
    )
    for item in items:
        # Use get_closest_marker (not ``in item.keywords``) so we only skip
        # explicitly-marked tests, never tests whose name merely contains the
        # word "compile"/"slow".
        if not run_compile and item.get_closest_marker("compile") is not None:
            item.add_marker(skip_compile)
        if not run_slow and item.get_closest_marker("slow") is not None:
            item.add_marker(skip_slow)
