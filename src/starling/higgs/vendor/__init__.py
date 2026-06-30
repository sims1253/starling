"""Vendored higgs-audio preprocessing code, decoupled from ``boson_multimodal``.

See ``higgs_audio_collator.py`` for why this is vendored rather than imported
from the upstream package.
"""

from .higgs_audio_collator import (
    ChatMLDatasetSample,
    HiggsAudioBatchInput,
    HiggsAudioSampleCollator,
)

__all__ = ["ChatMLDatasetSample", "HiggsAudioBatchInput", "HiggsAudioSampleCollator"]
