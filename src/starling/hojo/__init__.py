"""starling.hojo: CUDA megakernel package for HojoAI/Hojo-ASR-V1.

Hojo-ASR-V1 is a from-scratch speech-to-text model:

``Whisper-large-v3 mel -> Qwen3-Omni audio tower (32 layers, d=1280) ->
WeNet Conformer bottleneck (2 layers, d=2560) -> ln_speech ->
Qwen3-4B decoder (beam-4)``. ~5.19 B params.

Design (mirrors ``starling.higgs``, the closest existing template):
* The **audio encoder** (tower + Conformer) runs **eager**. Graphing it is not
  byte-exact -- the tower builds a block-diagonal 4D attention mask from
  ``cu_seqlens`` via host-side ``.item()`` syncs and dynamic boolean indexing,
  and the Conformer's conv module masks padding with a dynamic boolean fill.
  This is the same finding as higgs (eager-encoder + graphed-decode is a valid,
  performant design; the encoder runs once per clip).
* The **Qwen3 decoder** is beam-4. This first landing drives the stock
  ``decoder_model.generate(num_beams=4, ...)`` for byte-exactness with the
  golden oracle; a custom CUDA-graph-captured beam loop is future work (the
  repo has no prior beam-search megakernel).

Runs under its own isolated venv ``.venv-hojo`` (transformers 4.57.6, torch
2.7.1+cu128) because the model depends on ``hojo-asr`` and the Qwen3-Omni /
Qwen3 modeling that ships with that transformers version. See ``loader.py``.
"""

from .llm_mega import GenerateResult, LLMMega
from .pipeline import HojoMega

__all__ = ["LLMMega", "HojoMega", "GenerateResult"]
