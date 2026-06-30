"""Install a sitecustomize .pth hook so qwen3_asr survives `uv sync` reinstalls.

The pinned transformers (git main) ships qwen3_asr source but, at this commit,
find_packages drops 4 model dirs (minicpm3, nemotron3_5_asr, qwen3_asr,
xcodec2) from the wheel AND they aren't registered in the auto-mappings. Any
`uv sync` / `uv pip install` reinstalls a transformers wheel missing them, so
`from transformers import Qwen3ASRForConditionalGeneration` breaks.

This writes a `.pth` file into the venv site-packages whose `import` line runs
at interpreter startup (before any user import), restoring the 4 dirs from the
stable uv git checkout and registering qwen3_asr. Idempotent; safe to re-run.

Run:  uv run python scripts/setup_qwen3_tf.py
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

PTH_NAME = "zzz_starling_qwen3_tf_bootstrap.pth"
PTH_BODY = (
    "# Restores qwen3_asr in transformers at interpreter startup.\n"
    "import starling.qwen3._tf_bootstrap as _sb; _sb.ensure_qwen3_asr()\n"
)


def main() -> int:
    # starling is installed editable by uv, so it's importable at .pth time.
    site_dir = Path(site.getsitepackages()[0])
    pth = site_dir / PTH_NAME
    pth.write_text(PTH_BODY)
    print(f"[setup_qwen3_tf] wrote {pth}")
    # Also do an immediate restore + verify.
    import starling.qwen3._tf_bootstrap as sb  # noqa: F401

    ok = False
    try:
        from transformers import Qwen3ASRForConditionalGeneration  # noqa: F401

        ok = True
    except Exception as e:
        print(f"[setup_qwen3_tf] verify failed: {e!r}", file=sys.stderr)
    print(f"[setup_qwen3_tf] Qwen3ASRForConditionalGeneration importable: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
