"""C API hygiene tests for libstarling_ggml (the cpp/capi.cpp surface).

Covers the capi-error-hygiene fixes:

* ``starling_ggml_load`` failure messages must be readable, stable text. The
  parakeet load path used to hand cpp/capi.cpp a ``*err_out`` pointing into a
  destroyed ``ParakeetCtx`` (or a dead exception's ``what()``), so
  ``starling_ggml_last_error`` returned freed-heap garbage bytes.
* ``starling_ggml_backend_name`` must report the RUNTIME-selected device once
  a model load has created the global backend (it used to always report the
  compile-time backend family, e.g. "vulkan" on a build with Vulkan even when
  ``STARLING_GGML_DEVICE=cpu`` pinned the CPU device).

Both checks run in a subprocess: the ggml Backend is process-global, so an
in-process test cannot control which device an already-created backend picked.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from starling._ggml import _native

REPO = Path(__file__).resolve().parent.parent
# Same override knob as test_ggml_parity.py (benchmarks/engines.py reads it
# the same way), so hosts that keep the GGUF elsewhere still get the
# runtime-device assertion instead of a skip.
PARAKEET_GGUF = Path(os.environ.get(
    "STARLING_GGML_PARAKEET_MODEL",
    str(REPO / "models" / "parakeet-tdt-0.6b-v3-bf16-exact.gguf"))).expanduser()

pytestmark = pytest.mark.skipif(not _native.available(),
                                reason="libstarling_ggml not built")

# Compile-time family names backend_name() may report before any load
# (cpp/capi.cpp backend_name_for_build).
_BUILD_FAMILIES = {"cpu", "cuda", "metal", "vulkan", "hip"}

_CHILD = textwrap.dedent("""\
    import sys
    sys.path.insert(0, {src!r})
    from starling._ggml._native import backend_name, GgmlModel, PARAKEET_TDT

    print("PRE:" + backend_name())
    try:
        GgmlModel(PARAKEET_TDT, "/nonexistent/parakeet.gguf")
        raise SystemExit("bad-path load unexpectedly succeeded")
    except RuntimeError as e:
        print("ERR:" + str(e))
    if {with_model!r}:
        m = GgmlModel(PARAKEET_TDT, {gguf!r})
        print("POST:" + backend_name())
        m.close()
""")


def _run_child(with_model: bool) -> dict[str, str]:
    env = dict(os.environ)
    # Pin the CPU device so the runtime-device assertion is deterministic on
    # any host (and never touches GPU memory).
    env["STARLING_GGML_DEVICE"] = "cpu"
    code = _CHILD.format(src=str(REPO / "src"), with_model=with_model,
                         gguf=str(PARAKEET_GGUF))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        for tag in ("PRE:", "ERR:", "POST:"):
            if line.startswith(tag):
                out[tag[:-1]] = line[len(tag):]
    return out


def test_load_error_message_is_stable_text() -> None:
    """A failed parakeet load reports readable text, not freed-heap garbage."""
    out = _run_child(with_model=False)
    assert out["PRE"] in _BUILD_FAMILIES
    msg = out["ERR"]
    # The pre-fix bug copied bytes from a destroyed ParakeetCtx::err —
    # typically invalid UTF-8 or recycled heap contents, and never mentioning
    # the path. Require the real message.
    msg.encode("utf-8")  # raises on garbage bytes
    assert "/nonexistent/parakeet.gguf" in msg


def test_backend_name_reports_runtime_device() -> None:
    """backend_name() switches to the runtime device after a real load."""
    if not PARAKEET_GGUF.exists():
        pytest.skip("parakeet GGUF not present")
    out = _run_child(with_model=True)
    assert out["PRE"] in _BUILD_FAMILIES  # before load: compile-time family
    assert out["POST"] == "CPU"           # after load: STARLING_GGML_DEVICE=cpu
