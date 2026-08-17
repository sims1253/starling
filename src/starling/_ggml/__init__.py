"""Starling's in-tree ggml engine (libstarling_ggml).

Optional C++ sibling to the pure-Python Starling package: a ggml/CUDA inference
engine for parakeet-tdt and moss, built from ``cpp/`` and driven in-process via
ctypes (no HTTP, no subprocess). Byte-exact against the goldens, universal-
backend (CUDA/Metal/Vulkan/CPU).

The engine is OPTIONAL — the pure-Python package works without it. Importing
this subpackage never triggers a build or a load; everything is lazy and gated
on :func:`available`.

Example::

    from starling._ggml import available, GgmlModel, PARAKEET_TDT
    if available():
        m = GgmlModel(PARAKEET_TDT, "/path/to/tdt-0.6b-v3-f16.gguf")
        text = m.transcribe_pcm(pcm_ptr, pcm.size)
        m.close()
"""

from ._native import (
    ARK,
    GRANITE,
    HIGGS,
    HOJO,
    MOSS,
    PARAKEET_TDT,
    GgmlModel,
    available,
    backend_name,
)

__all__ = [
    "ARK",
    "GRANITE",
    "HIGGS",
    "HOJO",
    "MOSS",
    "PARAKEET_TDT",
    "GgmlModel",
    "available",
    "backend_name",
]
