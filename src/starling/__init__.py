"""starling: hand-tuned Triton megakernels for fast on-device ASR.

Hosts two model tracks sharing the same CUDA-graph / fused-kernel toolkit:
  * granite-speech-4.1-2b (``starling.granite`` subpackage: encoder + LLM decode + spec)
  * parakeet-tdt-0.6b-v3 (``starling.parakeet`` subpackage)
"""

from .granite.encoder_mega import FusedEncoder

__all__ = ["FusedEncoder"]
