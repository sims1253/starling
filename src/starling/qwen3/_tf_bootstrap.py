"""Ensure the shared ``transformers`` install exposes ``qwen3_asr``.

Context
-------
This repo is shared across several agent worktrees via a single symlinked
``.venv``. The pinned ``transformers`` (git main, commit ``957e6032``) ships the
``qwen3_asr`` model source, but at that commit the model is NOT yet registered
in the package's auto-mappings / ``__init__`` exports (the registration landed
in a later commit). On top of that, ``setup.py find_packages`` drops four
not-yet-registered model dirs (``minicpm3``, ``nemotron3_5_asr``,
``qwen3_asr``, ``xcodec2``) from the built wheel, and any ``uv pip install`` of
another package reinstalls a transformers wheel that is again missing them. So
``qwen3_asr`` vanishes unpredictably whenever a sibling worktree touches the
venv.

This module makes the Qwen3-ASR model importable regardless of that churn, by:

1. Restoring the four pure-Python model dirs from the stable uv git checkout
   (idempotent copy if absent).
2. Registering ``Qwen3ASRConfig`` / ``Qwen3ASREncoderConfig`` in
   ``transformers.models.auto.CONFIG_MAPPING``.
3. Exposing ``Qwen3ASRForConditionalGeneration`` / ``Qwen3ASRProcessor`` at the
   ``transformers`` top level.

Imported (best-effort) by ``starling.qwen3.loader`` before the first
``from transformers import ...``.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from pathlib import Path

# Model dirs that find_packages drops at this transformers commit.
_MISSING_DIRS = ("minicpm3", "nemotron3_5_asr", "qwen3_asr", "xcodec2")


def _site_packages_models_dir() -> Path | None:
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent / "models"


def _git_checkout_models_dir() -> Path | None:
    """The uv git checkout for transformers @ 957e6032 (stable across reinstalls)."""
    base = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
    if not base.exists():
        return None
    for checkout in base.iterdir():
        cand = checkout / "957e6032a2" / "src" / "transformers" / "models"
        if (cand / "qwen3_asr" / "__init__.py").exists():
            return cand
    return None


def _restore_dirs() -> bool:
    inst = _site_packages_models_dir()
    if inst is None:
        return False
    src = _git_checkout_models_dir()
    if src is None:
        return False
    for m in _MISSING_DIRS:
        dst = inst / m
        if not dst.exists():
            shutil.copytree(src / m, dst)
    ok_dirs = all((inst / m).exists() for m in _MISSING_DIRS)

    # processing_qwen3_asr.py imports symbols added in the same PR across a few
    # base files (audio_utils.make_list_of_audio_chat_template,
    # import_utils.is_nagisa_available / is_soynlp_available, ...). These base
    # files in the checkout are strict supersets of the installed versions
    # (purely additive for the qwen3_asr / optional-dep features). Sync them if
    # the qwen3_asr symbol is missing, and drop their sys.modules cache so the
    # next import re-reads the restored file.
    _sync_base_files(src.parent)
    return ok_dirs


def _sync_base_files(src_tf_dir: Path) -> None:
    """Copy additive base files needed by qwen3_asr if their symbols are missing."""
    import sys

    checks = {
        "audio_utils.py": ("transformers.audio_utils", "make_list_of_audio_chat_template"),
        "utils/import_utils.py": (
            "transformers.utils.import_utils",
            "is_nagisa_available",
        ),
    }
    import importlib

    for rel, (mod_name, probe) in checks.items():
        src_file = src_tf_dir / rel
        if not src_file.exists():
            continue
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, probe):
                continue  # already has the symbol
        except Exception:
            continue
        try:
            dst_file = Path(mod.__file__)
            shutil.copy2(src_file, dst_file)
        except Exception:
            continue
        # invalidate cache so the next import re-reads
        for k in list(sys.modules):
            if k == mod_name or k.startswith(mod_name + "."):
                del sys.modules[k]
        importlib.invalidate_caches()


def _register() -> None:
    """Register qwen3_asr configs + model classes in the transformers namespace."""
    import transformers

    # The pinned transformers commit's ``utils/auto_docstring.py`` prints a
    # harmless ``[ERROR] Config not found for qwen3-asr ...`` to stdout while
    # decorating the Qwen3-ASR model classes (the docstring generator runs at
    # class-definition time and does not know the not-yet-upstream config). It
    # is purely cosmetic and fires on EVERY import, so we swallow just those
    # lines around the submodule imports below.
    import contextlib

    class _SwallowDocstringNoise:
        def __enter__(self):
            import sys
            self._real, self._buf = sys.stdout, []
            sys.stdout = self
            return self

        def write(self, s):
            if "Config not found for qwen3-asr" not in s and s.strip():
                self._real.write(s)
            elif s and not s.strip():
                self._real.write(s)  # preserve bare newlines
            return len(s)

        def flush(self):
            self._real.flush()

        def __exit__(self, *exc):
            import sys
            sys.stdout = self._real

    # Import the submodules (now that the dirs are restored).
    with _SwallowDocstringNoise():
        cfg_mod = importlib.import_module("transformers.models.qwen3_asr.configuration_qwen3_asr")
        modeling_mod = importlib.import_module("transformers.models.qwen3_asr.modeling_qwen3_asr")

    # (1) CONFIG_MAPPING registration so config_dict["model_type"]="qwen3_asr*"
    # resolves. _LazyConfigMapping.register adds to _extra_content (no conflict).
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for model_type, cls_name in (
        ("qwen3_asr", "Qwen3ASRConfig"),
        ("qwen3_asr_encoder", "Qwen3ASREncoderConfig"),
    ):
        cls = getattr(cfg_mod, cls_name, None)
        if cls is not None and model_type not in CONFIG_MAPPING:
            CONFIG_MAPPING.register(model_type, cls, exist_ok=True)

    # (2) Expose model + processor at the transformers top level so
    # `from transformers import Qwen3ASRForConditionalGeneration` works.
    for attr in ("Qwen3ASRForConditionalGeneration", "Qwen3ASRModel", "Qwen3ASREncoder"):
        cls = getattr(modeling_mod, attr, None)
        if cls is not None and not hasattr(transformers, attr):
            setattr(transformers, attr, cls)

    # (3) Register the audio encoder with AutoModel so
    # AutoModel.from_config(audio_config) resolves Qwen3ASREncoder. The text
    # decoder (qwen3) is already registered upstream.
    from transformers import AutoModel

    enc_cfg = getattr(cfg_mod, "Qwen3ASREncoderConfig", None)
    enc_cls = getattr(modeling_mod, "Qwen3ASREncoder", None)
    if enc_cfg is not None and enc_cls is not None:
        try:
            AutoModel.register(enc_cfg, enc_cls, exist_ok=True)
        except Exception:
            pass
    try:
        with _SwallowDocstringNoise():
            proc_mod = importlib.import_module("transformers.models.qwen3_asr.processing_qwen3_asr")
        for attr in ("Qwen3ASRProcessor",):
            cls = getattr(proc_mod, attr, None)
            if cls is not None and not hasattr(transformers, attr):
                setattr(transformers, attr, cls)
    except Exception:
        pass


def ensure_qwen3_asr() -> bool:
    if not _restore_dirs():
        return False
    try:
        _register()
    except Exception:
        pass
    import transformers

    return hasattr(transformers, "Qwen3ASRForConditionalGeneration")


ensure_qwen3_asr()
