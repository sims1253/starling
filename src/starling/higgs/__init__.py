"""starling.higgs: CUDA-graph megakernel for bosonai/higgs-audio-v3-stt.

Whisper-large-v3 mel encoder + MLP projector + Qwen3-1.7B decoder. The Qwen3
decode is the launch-bound bottleneck; it is CUDA-graph-captured (single- or
multi-step) and byte-exact with the eager ``model.generate()`` reference.

Runs under its own isolated venv ``.venv-higgs`` (transformers 4.51) because the
model's trust_remote_code modeling breaks under the repo's transformers 5.13.
See ``loader.py`` for details.
"""

from .llm_mega import BenchReport, GenerateResult, LLMMega
from .pipeline import HiggsMega

__all__ = ["LLMMega", "HiggsMega", "GenerateResult", "BenchReport"]
