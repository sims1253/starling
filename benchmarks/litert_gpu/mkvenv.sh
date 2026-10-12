#!/bin/bash
# Host venv for exporting the micro models (torch CPU + litert-torch, Apache-2.0).
# Same pins as the NPU spike (benchmarks/npu). No Google Tensor SDK packages.
set -euo pipefail
V=${1:-/tmp/litert-gpu/venv}
uv venv -p 3.12 "$V"
export VIRTUAL_ENV=$V
uv pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.10.*" 2>&1 | tail -2
uv pip install "litert-torch==0.9.4" "transformers==5.19.*" 2>&1 | tail -2
"$V/bin/python" -I -c "import torch, litert_torch, ai_edge_litert; print('ok', torch.__version__)"
uv pip list 2>/dev/null | grep -iE "litert|torch|ai-edge|tensorflow" || true
