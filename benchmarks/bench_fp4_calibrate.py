"""Calibrate NVFP4 weight quantization for the Granite LLM.

Measures three things:

(a) Per-layer FP4 quantization error (max-abs and mean-abs of
    ``dequant_fp4(quant_fp4(w)) - w``) for every GEMM weight in the Granite
    decoder (qkv, o_proj, gate-up, down_proj, lm_head).
(b) The weight-byte reduction (the bandwidth win that motivates FP4).
(c) The theoretical decode-step speedup ceiling, derived from
    ``benchmarks/bench_decode_profile.py``'s finding that GEMVs are ~51% of
    decode-step time and are weight-bandwidth-bound. Idealized speedup
    ~= 1 / (1 - 0.51 + 0.51/compression_ratio).

The REAL speedup needs a fused dequant-GEMV kernel (stream packed FP4 +
per-block fp8 scale, dequant in registers). This script does not measure
that; it only quantifies the upper bound that such a kernel could reach.

Usage::

    uv run python benchmarks/bench_fp4_calibrate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.granite.fp4 import (  # noqa: E402
    dequantize_fp4,
    fp4_weight_bytes,
    quantize_fp4,
)
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402

# From benchmarks/bench_decode_profile.py: GEMVs are 51% of decode-step time.
GEMV_FRACTION_OF_STEP = 0.51


def _err_report(name: str, w: torch.Tensor) -> dict:
    codes, scales, meta = quantize_fp4(w)
    wd = dequantize_fp4(codes, scales, meta)
    e = (wd.float() - w.float()).abs()
    n = w.numel()
    return {
        "name": name,
        "nelem": n,
        "bf16_bytes": 2 * n,
        "fp4_bytes": fp4_weight_bytes(n),
        "max_abs_err": e.max().item(),
        "mean_abs_err": e.mean().item(),
        "mean_abs_w": w.abs().float().mean().item(),
        "rel_err": (e.mean() / w.abs().float().mean()).item(),
    }


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    print("loading granite model ...", flush=True)
    model, processor = load_model_and_processor(attn_impl="eager")
    comps = get_components(model)
    lm = comps["language_model"]

    rows = []
    total_bf16 = 0
    total_fp4 = 0
    for i, layer in enumerate(lm.layers):
        sa = layer.self_attn
        mlp = layer.mlp
        for sub, w in (
            (f"L{i:02d}.qkv_w",   torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight])),
            (f"L{i:02d}.o_proj",  sa.o_proj.weight),
            (f"L{i:02d}.gu_w",    torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight])),
            (f"L{i:02d}.down",    mlp.down_proj.weight),
        ):
            r = _err_report(sub, w.detach())
            total_bf16 += r["bf16_bytes"]
            total_fp4 += r["fp4_bytes"]
            if i == 0:  # print layer 0 in full; rest aggregated
                rows.append(r)
    r = _err_report("lm_head", model.lm_head.weight.detach())
    total_bf16 += r["bf16_bytes"]
    total_fp4 += r["fp4_bytes"]
    rows.append(r)

    print("\n== Per-weight FP4 quantization error (layer 0 representative) ==")
    print(f'{"weight":18s} {"nelem":>12s} {"bf16_B":>12s} {"fp4_B":>12s} '
          f'{"max_abs":>10s} {"mean_abs":>10s} {"rel_err":>9s}')
    for r in rows:
        print(f'{r["name"]:18s} {r["nelem"]:12d} {r["bf16_bytes"]:12d} '
              f'{r["fp4_bytes"]:12d} {r["max_abs_err"]:10.5f} '
              f'{r["mean_abs_err"]:10.6f} {r["rel_err"]:9.4f}')

    compression = total_bf16 / total_fp4
    # Roofline: the non-GEMV fraction (49%) is unchanged; the GEMV fraction
    # could ideally shrink by `compression`. UPPER BOUND that assumes a fused
    # dequant-GEMV kernel with zero dequant overhead.
    ideal_step_speedup = 1.0 / (1.0 - GEMV_FRACTION_OF_STEP
                                + GEMV_FRACTION_OF_STEP / compression)
    print("\n== Weight-byte reduction (the bandwidth win) ==")
    print(f"  total bf16 bytes : {total_bf16:,}")
    print(f"  total fp4  bytes : {total_fp4:,}")
    print(f"  compression      : {compression:.2f}x")
    print("\n== Theoretical decode-step speedup ceiling (UPPER BOUND) ==")
    print(f"  GEMV fraction of decode step : {GEMV_FRACTION_OF_STEP:.2f}")
    print(f"  (from benchmarks/bench_decode_profile.py)")
    print(f"  idealized step speedup       : {ideal_step_speedup:.2f}x")
    print(f"  NOTE: assumes fused dequant-GEMV kernel with ZERO dequant overhead.")
    print(f"  The scaffold's _fp4_linear dequantizes the FULL bf16 weight first,")
    print(f"  so it will be SLOWER than bf16. Fused kernel + QAD = future work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
