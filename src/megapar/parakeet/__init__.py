"""megapar.parakeet — megakernel components for nvidia/parakeet-tdt-0.6b-v3.

Subpackage for the parakeet-tdt-0.6b-v3 (FastConformer-TDT) optimization track.
Granite-speech-4.1-2b lives at the top-level megapar package; do not cross-import.
See ../../../../comms.md for the multi-model coordination contract.
"""

from .gpu_lock import acquire_gpu_lock, release_gpu_lock, with_gpu_lock, GpuLockBusy

__all__ = ["acquire_gpu_lock", "release_gpu_lock", "with_gpu_lock", "GpuLockBusy"]
