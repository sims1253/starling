"""A/B: Qwen3 windowed attention -- manual matmul/softmax vs SDPA-MATH.

The Qwen3 encoder's windowed attention (24 layers, ``ws x ws`` windows) uses a
manual ``matmul -> +mask -> softmax -> matmul`` recipe that materialises the
full ``(W, nh, ws, ws)`` scores + attn tensors. SDPA's MATH backend does the
same fp32 softmax over the same scores but in one fused kernel -- byte-exact
in principle (only the Q@K^T cuBLAS algorithm may differ by ~1 ULP).

Builds the GraphedEncoder both ways (``sdpa_attention`` flag), runs the same
input_features, and reports:
  * max-abs / mean-abs diff of the encoder output (manual vs sdpa)
  * median graph-replay time for each variant

Usage: uv run python benchmarks/bench_qwen3_attn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.qwen3.audio import build_inputs, load_wav  # noqa: E402
from starling.qwen3.encoder_mega import GraphedEncoder  # noqa: E402
from starling.qwen3.golden import _fixture_wav  # noqa: E402
from starling.qwen3.loader import get_components, load_model_and_processor  # noqa: E402


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()

    print("loading qwen3 model ...", flush=True)
    model, processor = load_model_and_processor()
    comps = get_components(model)
    encoder = comps["encoder"]

    # Build input_features from the sample clip via the qwen3 audio helper.
    wav, sr = load_wav(_fixture_wav())
    inputs = build_inputs(processor, wav, sr=sr)
    input_features = inputs["input_features"].to(model.dtype).cuda()
    input_features_mask = inputs.get("input_features_mask")
    if input_features_mask is not None:
        input_features_mask = input_features_mask.cuda()
    print("input_features:", tuple(input_features.shape), flush=True)

    # Two fresh GraphedEncoders -- one manual, one sdpa -- over the same weights.
    print("building manual-attention encoder ...", flush=True)
    enc_manual = GraphedEncoder(encoder, mode="cudagraph", sdpa_attention=False)
    print("building sdpa-attention encoder ...", flush=True)
    enc_sdpa = GraphedEncoder(encoder, mode="cudagraph", sdpa_attention=True)

    out_base = enc_manual(input_features, input_features_mask)
    out_sdpa = enc_sdpa(input_features, input_features_mask)
    print("baseline out:", tuple(out_base.shape), flush=True)

    diff = (out_sdpa.float() - out_base.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    base_abs = float(out_base.float().abs().mean().item())
    print(f"\nsdpa-vs-manual max_abs={max_abs:.6e}  "
          f"mean_abs={mean_abs:.6e}  (out mean_abs={base_abs:.6e})", flush=True)
    print(f"relative mean diff: {mean_abs/max(base_abs,1e-9):.6e}", flush=True)

    def time_replay(enc, label: str) -> float:
        trial_us = []
        CHUNK = 50
        for t in range(args.trials):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            done = 0
            while done < args.reps:
                n = min(CHUNK, args.reps - done)
                for _ in range(n):
                    # replay the captured graph for this shape
                    key = next(iter(enc._graphs.keys()))
                    enc._graphs[key]["graph"].replay()
                torch.cuda.synchronize()
                done += n
            e.record()
            torch.cuda.synchronize()
            total_ms = s.elapsed_time(e)
            trial_us.append(total_ms / args.reps * 1000.0)
            print(f"  [{label}] trial {t+1}: {trial_us[-1]:.2f} us/replay", flush=True)
        return median(trial_us)

    base_us = time_replay(enc_manual, "manual")
    sdpa_us = time_replay(enc_sdpa, "sdpa")
    speedup = base_us / sdpa_us if sdpa_us > 0 else float("nan")

    print(f"\n## Qwen3 windowed-attention A/B "
          f"(median of {args.trials}x{args.reps} replays)\n")
    print("| variant | us/replay | speedup | max-abs diff | mean-abs diff |")
    print("|---------|-----------|---------|--------------|---------------|")
    print(f"| manual matmul/softmax | {base_us:.2f} | 1.00x | - | - |")
    print(f"| SDPA math backend     | {sdpa_us:.2f} | {speedup:.2f}x | "
          f"{max_abs:.2e} | {mean_abs:.2e} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
