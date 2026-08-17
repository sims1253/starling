#!/usr/bin/env python3
"""Cross-platform reproducible environment rebuild for starling.

Replaces the old bash-only ``setup_env.sh`` (now in ``scripts/attic/``) so the
same setup works on Linux/macOS and on
native Windows (where bash and ``.venv/bin/python`` are unavailable -- the
Windows venv interpreter lives at ``.venv/Scripts/python.exe``).

The venv is pinned to CUDA 13.0 wheels (RTX 5090 / sm_120) via ``pyproject.toml``
(``[tool.uv.sources]``); this script just recreates the venv and verifies the
core imports.  On Windows ``triton`` is not installed (it has no official
Windows wheel), so it is omitted from the verification import -- the
:mod:`starling._kernels` package auto-selects the torch backend there.

Usage::

    python scripts/setup_env.py            # create .venv + uv sync + verify
    python scripts/setup_env.py --python 3.11

Requires ``uv`` on PATH.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt" or platform.system() == "Windows"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _venv_python() -> Path:
    """Path to the venv interpreter (``.venv/bin/python`` or ``.venv/Scripts/python.exe``)."""
    if IS_WINDOWS:
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def _run(cmd: list[str], **kw) -> None:
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), **kw)


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("error: 'uv' not found on PATH; install from https://docs.astral.sh/uv/", file=sys.stderr)
        return 1

    py_version = "3.10"
    if "--python" in sys.argv:
        py_version = sys.argv[sys.argv.index("--python") + 1]

    print(">> recreating venv (self-contained, cu130-pinned)")
    _run([uv, "venv", "--python", py_version, ".venv"])
    _run([uv, "sync"])

    # Verify core imports.  Triton has no official Windows wheel, so on Windows
    # it is simply not installed and the starling._kernels package uses the
    # torch backend.  Skip it from the verification there.
    print(">> verifying imports")
    imports = ["torch", "transformers", "accelerate", "soundfile"]
    if not IS_WINDOWS:
        imports.append("triton")
    check = (
        "import {mods}; "
        "import torch; "
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"
    ).format(mods=",".join(imports))
    _run([str(_venv_python()), "-c", check])

    # Verify starling itself imports and reports its active kernel backend.
    print(">> verifying starling import")
    _run([
        str(_venv_python()), "-c",
        "import starling._kernels as K; "
        "print('starling kernel backend:', K.get_backend_name(), "
        "'(triton' if K.have_triton() else '(torch fallback)')",
    ])

    print(">> done. venv at", (REPO_ROOT / ".venv").relative_to(Path.cwd())
          if (REPO_ROOT / ".venv").is_relative_to(Path.cwd()) else ".venv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
