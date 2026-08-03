"""NeMo-free loader for nvidia/parakeet-unified-en-0.6b.

The HF repo ships a single ``parakeet-unified-en-0.6b.nemo`` (2.47 GB) which is
a **zip containing only the torch flat-checkpoint** ``model_weights/`` -- no
``config.yaml``, no tokenizer. (NeMo normally embeds those in the zip, but this
checkpoint was saved as a bare ``state_dict``.) So:

* weights: ``unzip`` the ``.nemo`` -> ``torch.load('model_weights')`` (the flat
  checkpoint is a *directory* inside the zip; torch >= 2.12 reads it directly
  once the zip is unpacked). Returns an ``OrderedDict[str, Tensor]`` of 989
  keys, no ``state_dict`` wrapper, no ``model.`` prefix.
* tokenizer: the sentencepiece model from
  ``eschmidbauer/parakeet-unified-en-0.6b-c`` (byte-identical to the original;
  cross-checked against the sherpa-onnx unified export).

Everything else (encoder/decoder/joint dims, mel params) is in ``config.py``,
locked from the tensor shapes.

The downloaded ``.nemo`` is cached under the HF hub cache; the extracted
``model_weights`` dir under ``~/.cache/starling/parakeet_unified/`` so the
one-off unzip is amortised across loads.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict

import torch

from . import config as C

# Where the unzipped flat checkpoint lives (amortise the one-off unzip).
_CACHE_DIR = Path(
    os.environ.get("STARLING_CACHE", str(Path.home() / ".cache" / "starling"))
) / "parakeet_unified"


def _nemo_local_path() -> Path:
    """Resolve (downloading if needed) the ``.nemo`` file to a local path."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=C.MODEL_ID, filename=C.NEMO_FILENAME)
    return Path(p)


def _tokenizer_local_path() -> Path:
    """Resolve (downloading if needed) the sentencepiece model path."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=C.TOKENIZER_HF_REPO, filename=C.TOKENIZER_HF_FILE
    )
    return Path(p)


def _weights_zip_path(nemo_path: Path) -> Path:
    """Build a torch-loadable zip of the ``model_weights/`` flat checkpoint.

    The ``.nemo`` is itself a zip, but it nests the torch flat checkpoint under
    a ``model_weights/`` subdir alongside other content, so ``torch.load`` on
    the ``.nemo`` directly lands in the wrong (legacy) reader branch. And the
    torch flat format (``.format_version == 1``) is read by ``PyTorchFileReader``
    as a zip whose entries live under exactly one top-level subdir -- *not* as
    an extracted directory on disk (this torch build rejects dir paths).

    So: copy the ``model_weights/`` entries out of the ``.nemo`` into a
    standalone ``model_weights.zip`` (preserving the single ``model_weights/``
    top-level subdir) once, cache it, and ``torch.load`` that. Re-uses the same
    zip torch would have written had it saved the checkpoint itself.
    """
    target = _CACHE_DIR / "model_weights.zip"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(nemo_path) as src, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_STORED
    ) as dst:
        for member in src.namelist():
            if not member.startswith("model_weights/"):
                continue
            # Keep the leading "model_weights/" so all entries sit under one
            # top-level subdir (what PyTorchFileReader requires).
            with src.open(member) as f:
                dst.writestr(member, f.read())
    tmp.replace(target)
    return target


def load_state_dict(
    *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None,
) -> Dict[str, torch.Tensor]:
    """Load and return the parakeet-unified state_dict.

    The checkpoint is bf16 on disk; pass ``dtype`` to cast (e.g.
    ``torch.bfloat16`` keeps it, ``torch.float32`` upcasts for the eager
    numerical reference). Keys are **as stored** (``encoder.layers.0...``,
    ``decoder.prediction...``, ``joint...``, ``preprocessor...``) -- the
    hand-built modules in :mod:`modeling` use the same names so
    ``load_state_dict(strict=True)`` is the byte-exact gate.
    """
    nemo = _nemo_local_path()
    weights_zip = _weights_zip_path(nemo)
    sd = torch.load(
        str(weights_zip), map_location=device, weights_only=False
    )
    if not isinstance(sd, dict):
        raise RuntimeError(f"unexpected checkpoint type: {type(sd)}")
    if dtype is not None:
        sd = {
            k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in sd.items()
        }
    return sd


def load_tokenizer_path() -> Path:
    """Local path to the sentencepiece ``tokenizer.model``."""
    return _tokenizer_local_path()


__all__ = ["load_state_dict", "load_tokenizer_path"]
