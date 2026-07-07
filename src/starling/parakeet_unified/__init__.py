"""starling.parakeet_unified -- megakernel pipeline for
nvidia/parakeet-unified-en-0.6b (Unified FastConformer-RNN-T).

NeMo-free port: the .nemo checkpoint is loaded directly (no nemo_toolkit), the
Conformer encoder + RNNT prediction net + joint are hand-built in PyTorch, and
the encoder + greedy RNNT decode are captured into CUDA graphs (mirroring the
sibling ``starling.parakeet`` TDT pipeline). Output is byte-exact with the
NeMo/sherpa-onnx greedy transcript.

Public API (re-exports the GPU-lock helpers from the sibling parakeet module):
    MegaParakeetUnifiedPipeline    -- end-to-end audio->text megakernel
    acquire_gpu_lock / release_gpu_lock / with_gpu_lock / GpuLockBusy
"""

from starling.parakeet.gpu_lock import (
    GpuLockBusy,
    acquire_gpu_lock,
    release_gpu_lock,
    with_gpu_lock,
)

__all__ = [
    "MegaParakeetUnifiedPipeline",
    "acquire_gpu_lock",
    "release_gpu_lock",
    "with_gpu_lock",
    "GpuLockBusy",
]


def __getattr__(name: str):  # PEP 562: lazy import the pipeline
    if name == "MegaParakeetUnifiedPipeline":
        from .pipeline import MegaParakeetUnifiedPipeline

        return MegaParakeetUnifiedPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
