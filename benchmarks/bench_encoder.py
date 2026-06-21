"""Benchmark: stock vs fused encoder megakernel.

Produces a comparison table (median ms + correctness diff vs golden) for:
  * stock eager encoder (transformers GraniteSpeechCTCEncoder)
  * FusedEncoder mode="eager"    (clean reimplementation, byte-exact)
  * FusedEncoder mode="cudagraph" (manual CUDA-graph capture, byte-exact) *** winner ***
  * FusedEncoder mode="triton"    (triton elementwise kernels, byte-exact)
  * FusedEncoder mode="compile"   (torch.compile max-autotune, numerically close)

Run:  uv run python benchmarks/bench_encoder.py
"""

from __future__ import annotations

import json
import statistics
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from megapar.audio import build_inputs, load_sample_audio  # noqa: E402
from megapar.config import ENCODER_ATOL, TRACES_DIR  # noqa: E402
from megapar.encoder_mega import FusedEncoder  # noqa: E402
from megapar.golden import load_golden  # noqa: E402
from megapar.loader import get_components, load_model_and_processor  # noqa: E402

OUTPUTS = _REPO_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def cuda_timer(fn, warmup: int = 3, iters: int = 20) -> tuple[float, float]:
    """Return (median_ms, min_ms) using CUDA events."""
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    return statistics.median(times), min(times)


def diff_vs_golden(out: torch.Tensor, golden: torch.Tensor) -> tuple[float, float]:
    d = (out.float() - golden.float()).abs()
    return float(d.max().item()), float(d.mean().item())


def main() -> int:
    print("=" * 72)
    print("megapar encoder megakernel benchmark")
    print("=" * 72)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"tolerance: ENCODER_ATOL={ENCODER_ATOL} (max), 5e-3 (mean)")

    # ---- load model + sample input ----
    print("\n[bench] loading model + sample audio ...")
    model, processor = load_model_and_processor(attn_impl="eager")
    encoder = get_components(model)["encoder"]
    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    feats = inputs["input_features"].to(torch.bfloat16).cuda()
    golden = load_golden("encoder_last_hidden.pt").cuda()
    print(f"[bench] input_features: {tuple(feats.shape)} {feats.dtype}")
    print(f"[bench] golden:         {tuple(golden.shape)} {golden.dtype}")

    results = []

    # ---- stock baseline ----
    print("\n[bench] timing stock encoder ...")
    with torch.inference_mode():
        stock_out = encoder(feats, return_dict=True).last_hidden_state
    stock_d = diff_vs_golden(stock_out, golden)
    stock_med, stock_min = cuda_timer(lambda: encoder(feats, return_dict=True))
    results.append({
        "name": "stock (transformers)",
        "median_ms": round(stock_med, 3),
        "min_ms": round(stock_min, 3),
        "max_diff": stock_d[0],
        "mean_diff": stock_d[1],
        "passes": stock_d[0] < ENCODER_ATOL,
        "notes": "GraniteSpeechCTCEncoder.forward",
    })

    # ---- fused modes ----
    fused_configs = [
        ("eager", {}, "clean reimplementation (byte-exact)"),
        ("cudagraph", {}, "manual CUDA-graph capture (byte-exact) *** WINNER ***"),
        ("triton", {"_no_compile": True}, "triton elementwise kernels, no compile (byte-exact)"),
        ("triton", {"compile_mode": "max-autotune-no-cudagraphs"},
         "triton kernels + torch.compile (byte-exact)"),
        ("compile", {"compile_mode": "max-autotune-no-cudagraphs"},
         "torch.compile max-autotune (fp32 attn intermediates)"),
    ]

    for mode, kw, notes in fused_configs:
        no_compile = kw.pop("_no_compile", False)
        compile_tag = kw.get("compile_mode", "")
        label = f"{mode}" + (f"({compile_tag})" if compile_tag else ("(no-compile)" if no_compile else ""))
        print(f"[bench] timing FusedEncoder mode={label} ...")
        try:
            fe = FusedEncoder(encoder, mode=mode, **kw).cuda()
            if no_compile:
                fe._compiled_forward = None  # pure triton, bypass torch.compile
            with torch.inference_mode():
                out = fe(feats)
                torch.cuda.synchronize()
                med, mn = cuda_timer(lambda: fe(feats))
                d = diff_vs_golden(out, golden)
            results.append({
                "name": f"fused:{label}",
                "median_ms": round(med, 3),
                "min_ms": round(mn, 3),
                "max_diff": d[0],
                "mean_diff": d[1],
                "passes": d[0] < ENCODER_ATOL and d[1] < 5e-3,
                "notes": notes,
            })
            del fe
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            results.append({
                "name": f"fused:{label}",
                "median_ms": float("nan"),
                "min_ms": float("nan"),
                "max_diff": float("nan"),
                "mean_diff": float("nan"),
                "passes": False,
                "notes": f"FAILED: {type(exc).__name__}: {str(exc)[:120]}",
            })

    # ---- print table ----
    print("\n" + "=" * 90)
    print(f"{'mode':<42} {'median':>8} {'min':>8} {'max_diff':>10} {'mean_diff':>10} {'pass':>5}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['name']:<42} {r['median_ms']:>7.2f}m {r['min_ms']:>7.2f}m "
            f"{r['max_diff']:>10.2e} {r['mean_diff']:>10.2e} {'Y' if r['passes'] else 'N':>5}"
        )
    print("=" * 90)
    best = min(
        (r for r in results if r["passes"] and r["median_ms"] == r["median_ms"]),
        key=lambda r: r["median_ms"],
        default=None,
    )
    if best is not None:
        speedup = stock_med / best["median_ms"]
        print(f"\nbest (fastest passing): {best['name']} @ {best['median_ms']:.2f}ms "
              f"({speedup:.2f}x vs stock {stock_med:.2f}ms)")
    else:
        print("\nno passing configuration!")

    # ---- save ----
    payload = {
        "device": torch.cuda.get_device_name(0),
        "input_shape": list(feats.shape),
        "golden_shape": list(golden.shape),
        "tolerance": {"max": ENCODER_ATOL, "mean": 5e-3},
        "stock_baseline_ms": round(stock_med, 3),
        "results": results,
    }
    out_path = OUTPUTS / "encoder_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
