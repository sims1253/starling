"""Shim so the upstream ``transcribe.py`` finds the collator.

Re-exports our transformers-version-independent vendored copy (no boson_multimodal
dependency, which would pin transformers<4.47 and conflict with Qwen3's 4.51 need).
"""
from starling.higgs.vendor.higgs_audio_collator import (  # noqa: F401
    HiggsAudioSampleCollator,
    HiggsAudioBatchInput,
)
from starling.higgs.vendor.chatml_dataset import ChatMLDatasetSample  # noqa: F401
