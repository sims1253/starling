"""starling.nar — CUDA-graph megakernel for Granite-Speech-4.1-2b-NAR.

Non-autoregressive speech recognition: a single bidirectional forward pass
(encoder conformer + Q-Former projector + bidirectional granite-4.0-1b LLM
editor). No decode loop, so the optimization is graph capture of the dense
forwards + ``torch.compile`` of the LLM editor, rather than a K-step decode
graph. Output is byte-identical to the eager ``transformers`` reference.
"""

from .mega import NarMega

__all__ = ["NarMega"]
