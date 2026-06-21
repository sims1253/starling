"""megapar: high-performance inference megakernel for granite-speech-4.1-2b on RTX 5090."""

from .encoder_mega import FusedEncoder

__all__ = ["FusedEncoder"]
