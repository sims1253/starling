"""Vendored ``bosonai/higgs-audio-v3-stt`` remote modeling code.

Loaded directly (instead of via ``trust_remote_code=True``) so we can patch the
transformers-5.x incompatibilities (Whisper encoder mask plumbing, removed
``GenerationConfig.generation_kwargs``) without affecting the shared venv or
other model worktrees. The files here are byte-for-byte the upstream remote
code except for the documented patches (search for "transformers 5.x").
"""

# Import order matters: config first, then modeling (modeling imports config).
from .configuration_higgs_audio import HiggsAudio3Config, HiggsAudioEncoderConfig
from .modeling_higgs_audio import HiggsAudio3Model

__all__ = ["HiggsAudio3Config", "HiggsAudioEncoderConfig", "HiggsAudio3Model"]
