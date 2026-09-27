"""Collect S1-mini decoder activation importance on the existing text fixtures.

Use a Q8_0 linear-weight GGUF so ggml feeds F32 activations to quantized
matmuls; the BF16-exact model does not expose them to the imatrix hook. The
five quality prompts, sixteen control combinations, and three length tiers
are calibration inputs. The protected-span cases in s1_quant_spans.json are
held out from this collection.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--library", required=True, type=Path)
    args = ap.parse_args()
    for path in (args.model, args.library):
        if not path.is_file():
            ap.error(f"missing input: {path}")
    os.environ["STARLING_GGML_DEVICE"] = "cpu"
    os.environ["STARLING_GGML_LIB"] = str(args.library.resolve())
    os.environ["STARLING_IMATRIX"] = str(args.output.resolve())

    spec = importlib.util.spec_from_file_location(
        "s1_transcripts", ROOT / "tests/fixtures/s1_transcripts.py")
    fixture = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(fixture)

    from starling._ggml import GgmlModel, S1
    from starling._ggml._native import _load_lib, backend_name
    model = GgmlModel(S1, str(args.model.resolve()))
    try:
        if backend_name().lower() != "cpu":
            raise RuntimeError("S1 imatrix collection requires CPU activation memory")
        inputs = [(t, s, st, c) for t, s, st, c, _ in fixture.QUALITY_CASES]
        inputs.extend(fixture.CONTROL_MATRIX)
        inputs.extend((t, "semi-formal", "prose", "general")
                      for t in fixture.LENGTH_TIERS.values())
        for index, (transcript, styling, structure, context) in enumerate(inputs, 1):
            model.normalize_text(transcript, styling, structure, context)
            print(f"[{index}/{len(inputs)}] {len(transcript)} input chars", flush=True)
        lib = _load_lib()
        flush = getattr(lib, "starling_ggml_imatrix_flush_pub")
        flush.argtypes = []
        flush.restype = None
        flush()
    finally:
        model.close()
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("imatrix collector did not write an output file")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
