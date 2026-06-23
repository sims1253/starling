"""starling.parakeet — megakernel components for nvidia/parakeet-tdt-0.6b-v3.

High-performance inference pipeline for the parakeet-tdt-0.6b-v3
(FastConformer-TDT) speech-to-text model: GPU mel extraction, CUDA-graphed
encoder, multi-step graphed TDT decode, and memory-bounded chunked long-audio.
"""

from .gpu_lock import acquire_gpu_lock, release_gpu_lock, with_gpu_lock, GpuLockBusy

__all__ = ["acquire_gpu_lock", "release_gpu_lock", "with_gpu_lock", "GpuLockBusy"]
