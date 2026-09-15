"""starling.voxtral — pipeline components for mistralai/Voxtral-Mini-4B-Realtime-2602.

Voxtral Realtime transcription: a causal Whisper-style audio encoder whose
per-step slice is a fixed 4 embeds (one audio token), an additive audio
injection into the Ministral-3-class text decoder, and per-request-constant
AdaRMSNorm delay modulation. The v1 eager loop mirrors stock ``generate``
byte-for-byte; the fixed per-step shapes (encoder slice 4x1280, text step
1x3072) let a CUDA-graphed decode path slot in later.

Shared architecture constants live in :mod:`starling.voxtral.config`.
"""

from __future__ import annotations
