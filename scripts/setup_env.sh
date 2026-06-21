#!/usr/bin/env bash
# Reproducible environment rebuild for megapar.
# The venv is pinned to CUDA 13.0 wheels (RTX 5090 / sm_120).
set -euo pipefail
cd "$(dirname "$0")/.."
echo ">> recreating venv (self-contained, cu130-pinned)"
uv venv --python 3.10 .venv
uv sync
echo ">> verifying"
.venv/bin/python -c "import torch,triton,transformers,accelerate,soundfile;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"
echo ">> done. venv at .venv"
