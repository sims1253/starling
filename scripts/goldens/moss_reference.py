"""Stock-transformers reference path for MOSS-Transcribe-preview-2B.

Shared by ``moss_golden.py`` (token ids + text) and
``moss_golden_components.py`` (staged component tensors). The model and the
processor come from the upstream repository's own remote code
(``modeling_Moss.py`` / ``processing_Moss.py`` / ``chat_template_default.py``)
pinned to ``MODEL_REVISION``; nothing is vendored.

The decode is an eager greedy loop over an exact-width ``DynamicCache``
(the stock ``generate`` path rejects this model's kwargs on current
transformers, but the loop calls the identical model modules). Exact-width
attention keeps bf16 near-ties independent of any cache capacity: a padded
StaticCache flipped golden medium/long token 21.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = ("short", "medium", "long")

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-preview-2B"
MODEL_REVISION = "c98175cb20e48bd9be4e95f6c85f2af18899f780"
EOS_TOKEN_ID = 151645  # <|im_end|>
REFERENCE_PATH = "scripts/goldens/moss_reference.py (eager greedy, exact-width DynamicCache)"


def capture_device() -> str:
    """CUDA when available, else CPU; ``STARLING_GOLDEN_DEVICE`` overrides."""
    env_dev = os.environ.get("STARLING_GOLDEN_DEVICE")
    if env_dev in ("cpu", "cuda"):
        return env_dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def repo_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = wav.mean(1)
    return torch.from_numpy(wav.astype("float32")), sr


def _remote_module(filename: str):
    """Import one of the upstream remote-code files at the pinned revision."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_ID, filename, revision=MODEL_REVISION)
    name = Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def load_model_and_processor(*, device: str, dtype: torch.dtype = torch.bfloat16) -> tuple[Any, Any]:
    """Load the eager bf16 model and the upstream MOSS processor."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        dtype=dtype, attn_implementation="eager",
    ).to(device)
    # from_pretrained(attn_implementation=...) only reaches the top-level
    # config; the LLM/audio sub-configs keep their default ("sdpa"), which
    # would silently put the eager reference on an SDPA kernel. Propagate.
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
    model.eval()

    processing, _ = _remote_module("processing_Moss.py")
    template = hf_hub_download(MODEL_ID, "chat_template_default.py", revision=MODEL_REVISION)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    proc = processing.MossProcessor(tok, config=processing.MelConfig(mel_dim=128),
                                    template_path=template)
    return model, proc


def audio_features(model: Any, audio_data: torch.Tensor, seqlens: torch.Tensor) -> torch.Tensor:
    """Stock audio encoder -> last_hidden_state (the adapter input)."""
    return model.model.audio_model(input_features=audio_data, feature_lens=seqlens).last_hidden_state


def build_inputs_embeds(model: Any, input_ids: torch.Tensor, features: torch.Tensor,
                        audio_input_mask: torch.Tensor) -> torch.Tensor:
    """Merge audio features into the token embeddings exactly as MossModel.forward."""
    lm = model.model.language_model
    inputs_embeds = lm.get_input_embeddings()(input_ids)
    audio_embeds = model.model.audio_adapter(features)
    mask = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(mask, audio_embeds)


def _prefill(model: Any, inputs_embeds: torch.Tensor, cache: Any) -> torch.Tensor:
    lm = model.model.language_model
    T = inputs_embeds.shape[1]
    device = inputs_embeds.device
    # attention_mask=None: the model builds its own causal mask (bit-identical
    # to an explicit 0/-inf mask on the eager path).
    out = lm(inputs_embeds=inputs_embeds, attention_mask=None,
             position_ids=torch.arange(T, device=device).unsqueeze(0),
             past_key_values=cache, use_cache=True,
             cache_position=torch.arange(T, device=device))
    return model.lm_head(out.last_hidden_state[:, -1:, :])


def last_prefill_logits(model: Any, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """The prefill distribution whose argmax seeds :func:`greedy_generate`."""
    from transformers.cache_utils import DynamicCache

    return _prefill(model, inputs_embeds, DynamicCache(config=model.model.language_model.config))


def greedy_generate(model: Any, inputs_embeds: torch.Tensor, *, max_new_tokens: int = 200,
                    max_cache_len: int = 2048) -> torch.Tensor:
    """Eager greedy decode; returns (1, n_new) int64 on CPU (EOS included)."""
    from transformers.cache_utils import DynamicCache

    lm = model.model.language_model
    device = inputs_embeds.device
    T = inputs_embeds.shape[1]
    assert T + max_new_tokens <= max_cache_len, f"cache overflow: {T}+{max_new_tokens}>{max_cache_len}"
    cache = DynamicCache(config=lm.config)
    next_token = _prefill(model, inputs_embeds, cache).argmax(dim=-1)  # (1, 1)
    gen = [int(next_token.item())]
    step_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
    for i in range(max_new_tokens - 1):
        cur = T + i
        step_ids.copy_(next_token)
        out = lm(input_ids=step_ids, attention_mask=None,
                 position_ids=torch.tensor([[cur]], device=device),
                 past_key_values=cache, use_cache=True,
                 cache_position=torch.tensor([cur], device=device))
        next_token = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1)
        gen.append(int(next_token.item()))
        if gen[-1] == EOS_TOKEN_ID:
            break
    return torch.tensor(gen, dtype=torch.int64).unsqueeze(0)
