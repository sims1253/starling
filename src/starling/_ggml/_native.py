"""ctypes binding to libstarling_ggml — Starling's in-tree ggml engine.

In-process transcription of parakeet-tdt and moss via the shared ggml/CUDA
engine built from ``cpp/`` (``libstarling_ggml``). No HTTP, no WAV encode/
decode, no subprocess: a model loads once into an opaque context and each call
feeds raw float32 PCM straight to the C API. This is the lowest-overhead path
to the byte-exact ggml engine.

OPTIONAL: the binding activates only if ``libstarling_ggml`` is found. The
pure-Python Starling package keeps working without it — callers gate on
:func:`available` and skip the ggml engine if it returns False (mirroring the
proven ``_ggml_parakeet_native.py`` pattern).

Build (from the Starling repo root, optional)::

    cmake -B build -DSTARLING_GGML_CUDA=ON -DSTARLING_GGML_SHARED=ON
    cmake --build build -j

Then this module discovers ``build/libstarling_ggml.so`` automatically, or set
``STARLING_GGML_LIB`` to point at an explicit path.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# The ABI version this binding expects. Bumped in lockstep with
# STARLING_GGML_ABI_VERSION in cpp/include/starling_ggml.h.
_EXPECTED_ABI_VERSION = 1


def _candidate_lib_paths() -> list[Path]:
    """Search order for the built shared library.

    1. ``$STARLING_GGML_LIB`` (explicit override, must point at the file).
    2. Package-relative (``src/starling/_ggml/../../../build/...``) — for an
       in-place built wheel or editable install that ships the .so alongside.
    3. ``<repo-root>/build/libstarling_ggml.{so,dylib,dll}`` — the dev-build
       output location from ``cmake --build build``.
    """
    exts = (".so", ".dylib", ".dll") if sys.platform == "win32" else (
        (".dylib",) if sys.platform == "darwin" else (".so",)
    )
    candidates: list[Path] = []
    env = os.environ.get("STARLING_GGML_LIB")
    if env:
        candidates.append(Path(env).expanduser())
    # Package-relative: this file is src/starling/_ggml/_native.py; the repo
    # root is three levels up in a source checkout.
    pkg_root = Path(__file__).resolve().parent
    for up in (pkg_root.parent.parent.parent, pkg_root.parent.parent):
        for ext in exts:
            candidates.append(up / "build" / f"libstarling_ggml{ext}")
    return candidates


_LIB = None  # cached ctypes.CDLL, or False once tried-and-failed
_CAPI_OK = None  # True once the symbols resolved and ABI matched


def _load_lib():
    """Return the loaded ctypes.CDLL, or None if unavailable/broken.

    Cached: the first call probes; subsequent calls return the cached result
    (the .so or None). Never raises — callers use :func:`available`.
    """
    global _LIB, _CAPI_OK
    if _LIB is not None:
        return _LIB if (_LIB is not False and _CAPI_OK) else None
    # Probe candidates in order; load the first that exists + resolves symbols.
    for path in _candidate_lib_paths():
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError:
            continue
        if not _resolve_symbols(lib):
            continue
        _LIB = lib
        _CAPI_OK = True
        return lib
    _LIB = False
    _CAPI_OK = False
    return None


def _resolve_symbols(lib: ctypes.CDLL) -> bool:
    """Resolve the C-API symbols + check the ABI version. Returns True on match."""
    try:
        lib.starling_ggml_abi_version.restype = ctypes.c_int
        lib.starling_ggml_abi_version.argtypes = []
        lib.starling_ggml_backend_name.restype = ctypes.c_char_p
        lib.starling_ggml_backend_name.argtypes = []
        lib.starling_ggml_load.restype = ctypes.c_void_p
        lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
        lib.starling_ggml_free.argtypes = [ctypes.c_void_p]
        lib.starling_ggml_free.restype = None
        lib.starling_ggml_shutdown.argtypes = []
        lib.starling_ggml_shutdown.restype = None
        lib.starling_ggml_last_error.restype = ctypes.c_char_p
        lib.starling_ggml_last_error.argtypes = [ctypes.c_void_p]
        lib.starling_ggml_transcribe_pcm.restype = ctypes.POINTER(ctypes.c_char)
        lib.starling_ggml_transcribe_pcm.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64, ctypes.c_int,
        ]
        lib.starling_ggml_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
        lib.starling_ggml_free_string.restype = None
    except (AttributeError, OSError):
        return False
    # ABI check: refuse to load a mismatched .so (prevents cryptic arg
    # misalignment after an API bump the binding hasn't tracked).
    got = lib.starling_ggml_abi_version()
    return got == _EXPECTED_ABI_VERSION


def available() -> bool:
    """True iff libstarling_ggml loads, exposes the C API, and matches ABI."""
    return _load_lib() is not None


def backend_name() -> str:
    """The active ggml backend ('cuda'/'metal'/'vulkan'/'cpu').

    Returns 'unavailable' if the library isn't loaded (for diagnostics).
    """
    lib = _load_lib()
    if lib is None:
        return "unavailable"
    return lib.starling_ggml_backend_name().decode("utf-8", "replace")


# Model selector constants (mirror the C enum starling_ggml_model).
PARAKEET_TDT = 1
MOSS = 2


class GgmlModel:
    """An in-process Starling ggml model: load once, transcribe many.

    Holds the opaque ``starling_ggml_ctx``. One instance per engine; freed in
    :meth:`close`. NOT thread-safe (the ggml backend is process-global); the
    engine layer serialises calls.
    """

    def __init__(self, model: int, model_path: str) -> None:
        lib = _load_lib()
        if lib is None:
            raise RuntimeError(
                "libstarling_ggml not available — build it with "
                "`cmake -B build -DSTARLING_GGML_CUDA=ON -DSTARLING_GGML_SHARED=ON "
                "&& cmake --build build -j`, or set STARLING_GGML_LIB"
            )
        self._lib = lib
        self._ctx = lib.starling_ggml_load(model, model_path.encode("utf-8"))
        if not self._ctx:
            err = self._last_error()
            raise RuntimeError(f"starling_ggml_load failed: {err}")

    def _last_error(self) -> str:
        try:
            return self._lib.starling_ggml_last_error(self._ctx).decode("utf-8", "replace")
        except Exception:
            return ""

    def transcribe_pcm(self, samples, n: int, sample_rate: int = 16000) -> str:
        """Transcribe ``n`` mono float32 samples; return the UTF-8 transcript."""
        ptr = self._lib.starling_ggml_transcribe_pcm(
            self._ctx, samples, n, sample_rate)
        if not ptr:
            err = self._last_error()
            raise RuntimeError(f"starling_ggml_transcribe_pcm failed: {err}")
        raw = ctypes.cast(ptr, ctypes.c_char_p).value
        text = (raw or b"").decode("utf-8", "replace")
        self._lib.starling_ggml_free_string(ptr)
        return text

    def transcribe_pcm_ids(self, samples, n: int, sample_rate: int = 16000):
        """Transcribe ``n`` mono float32 samples; return the raw token-id stream.

        Parakeet-only. Returns the emitted id stream INCLUDING blanks, prefixed
        with the decoder-start (blank) token, matching the format of the golden
        ``parakeet_tdt_*_ids.pt`` (HF ``model.generate().sequences[0]``). This
        is the strictest in-tree parity gate (greedy TDT is deterministic).

        Lazily resolves ``starling_ggml_parakeet_decode_ids_pub`` so it does not
        gate :func:`available` for builds that pre-date the symbol.
        """
        lib = self._lib
        fn = getattr(lib, "starling_ggml_parakeet_decode_ids_pub", None)
        if fn is None:
            raise RuntimeError(
                "libstarling_ggml has no starling_ggml_parakeet_decode_ids_pub "
                "symbol (rebuild cpp/)")
        fn.restype = ctypes.POINTER(ctypes.c_int64)
        fn.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
        ]
        out_n = ctypes.c_int64(0)
        ptr = fn(self._ctx, samples, n, ctypes.byref(out_n))
        if not ptr:
            err = self._last_error()
            raise RuntimeError(f"starling_ggml_parakeet_decode_ids failed: {err}")
        try:
            ids = ptr[: out_n.value]  # list[int]
        finally:
            # The buffer was std::malloc'd in C; starling_ggml_free_string does
            # std::free, which is type-agnostic (element type is irrelevant to
            # free), so reusing it is correct and avoids a new C entry point.
            lib.starling_ggml_free_string(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char)))
        return ids

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.starling_ggml_free(self._ctx)
            self._ctx = None
