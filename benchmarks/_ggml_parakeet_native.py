"""ctypes binding to libparakeet.so (mudler's parakeet.cpp C API).

In-process transcription with NO HTTP, NO WAV encode/decode, NO subprocess:
the model loads once into an opaque ``parakeet_ctx`` and each call feeds raw
float32 PCM directly to the C API. This is the lowest-overhead way to drive
the byte-exact ggml/CUDA engine from Python -- it removes the ~25 ms+ HTTP +
WAV tax of the server wrapper and matches the in-process CLI bench numbers.

The binding is optional: it activates only if ``libparakeet.so`` exists (built
with ``cmake -DPARAKEET_SHARED=ON`` in the parakeet.cpp repo). If absent, the
:class:`GgmlParakeet` engine falls back to the persistent HTTP server.

Process teardown note: parakeet.cpp's global ggml ``Backend`` destructor runs
at process exit and crashes if CUDA has already torn down (a known ggml
issue). Callers that hold a context MUST call :func:`shutdown` (which calls
``parakeet_capi_free`` and then ``os._exit``-style hard exit is NOT used here;
instead we keep the context alive for the process lifetime and rely on the
engine ``close()`` to free it before any other engine loads).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

# Build artifact: libparakeet.so lives in the parakeet.cpp build tree. Override
# with GGML_PARAKEET_LIB. The model path is shared with the server engine.
LIBPARAKEET_SO = Path(os.environ.get(
    "GGML_PARAKEET_LIB",
    Path.home() / "Documents" / "parakeet.cpp" / "build-cuda" / "libparakeet.so",
)).expanduser()

_LIB = None  # cached ctypes.CDLL or False once tried
_CAPI_OK = None  # True once the C API symbols resolved


def _load_lib():
    """Return the ctypes.CDLL for libparakeet, or None if unavailable/broken."""
    global _LIB, _CAPI_OK
    if _LIB is not None:
        return _LIB if (_LIB is not False and _CAPI_OK) else None
    if not LIBPARAKEET_SO.exists():
        _LIB = False
        return None
    try:
        lib = ctypes.CDLL(str(LIBPARAKEET_SO))
        # resolve the C API symbols we need
        lib.parakeet_capi_abi_version.restype = ctypes.c_int
        lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        lib.parakeet_capi_load.restype = ctypes.c_void_p
        lib.parakeet_capi_transcribe_pcm.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.parakeet_capi_transcribe_pcm.restype = ctypes.POINTER(ctypes.c_char)
        lib.parakeet_capi_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
        lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_last_error.restype = ctypes.c_char_p
        lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]
        # pk::shutdown_backend() -- frees the process-global ggml Backend NOW
        # (while the CUDA driver is alive) instead of at process exit. The C++
        # mangled name is the only exported form; calling it avoids the
        # "driver shutting down" abort in the static destructor at teardown.
        try:
            lib._ZN2pk16shutdown_backendEv.argtypes = []
            lib._ZN2pk16shutdown_backendEv.restype = None
            lib._shutdown_backend = lib._ZN2pk16shutdown_backendEv
        except (AttributeError, OSError):
            lib._shutdown_backend = None
        _LIB = lib
        _CAPI_OK = True
        return lib
    except OSError:
        _LIB = False
        return None


def available() -> bool:
    """True iff libparakeet.so loads and exposes the C API."""
    return _load_lib() is not None


class NativeParakeet:
    """An in-process parakeet.cpp context: load once, transcribe many.

    Holds the opaque ``parakeet_ctx`` (a loaded ggml model). One instance per
    engine; freed in :meth:`close`. NOT thread-safe (the ggml backend is
    process-global); the engine serialises calls.
    """

    def __init__(self, model_path: str) -> None:
        self._lib = _load_lib()
        if self._lib is None:
            raise RuntimeError(f"libparakeet.so not available at {LIBPARAKEET_SO}")
        self._ctx = self._lib.parakeet_capi_load(model_path.encode())
        if not self._ctx:
            err = ""
            try:
                err = self._lib.parakeet_capi_last_error(self._ctx).decode()
            except Exception:
                pass
            raise RuntimeError(f"parakeet_capi_load failed: {err}")

    def transcribe_pcm(self, samples, n: int, sample_rate: int = 16000,
                       decoder: int = 0) -> str:
        """Transcribe mono float32 PCM; return the UTF-8 transcript.

        ``decoder``: 0=default, 1=ctc, 2=tdt/rnnt (the parakeet-tdt model uses
        default -> transducer/TDT).
        """
        ptr = self._lib.parakeet_capi_transcribe_pcm(
            self._ctx, samples, n, sample_rate, decoder)
        if not ptr:
            err = ""
            try:
                err = self._lib.parakeet_capi_last_error(self._ctx).decode()
            except Exception:
                pass
            raise RuntimeError(f"parakeet_capi_transcribe_pcm failed: {err}")
        text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", "replace")
        self._lib.parakeet_capi_free_string(ptr)
        return text

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.parakeet_capi_free(self._ctx)
            self._ctx = None
        # Free the global ggml backend explicitly so its device buffers are
        # released while the CUDA driver is still alive (avoids the abort in
        # the Backend static destructor at process exit). NOTE: with the
        # persistent PredictionNet/Joint decode graphs (parakeet.cpp 5a98272),
        # the Model destructor (via parakeet_capi_free above) frees the nets'
        # ReplayGraph device buffers first; shutdown_backend then frees the
        # global backend. If shutdown_backend is called it can double-free / race
        # with the nets' teardown under some build states, so we call it best-
        # effort and swallow any error -- the OS reclaims the rest at exit.
        shutdown = getattr(self._lib, "_shutdown_backend", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass
