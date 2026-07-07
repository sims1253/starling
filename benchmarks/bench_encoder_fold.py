"""A/B: Granite conformer encoder with vs without BatchNorm fold.

Builds two fresh ``FusedEncoder`` instances from the same stock encoder
weights -- one without folding (baseline) and one with ``fold_bn=True`` --
then for each:

  1. runs a forward on the golden ``input_features`` and records the
     ``last_hidden_state``,
  2. times the captured graph replay (median of N trials),
  3. reports max-abs and mean-abs diff vs the unfused reference.

The fold is byte-exact in fp32 (per-channel affine) so the diff should be
sub-ULP bf16 noise from re-casting the folded weight.

Usage: uv run python benchmarks/bench_encoder_fold.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import median

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.granite.encoder_mega import FusedEncoder  # noqa: E402
from starling.granite.golden import load_golden  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()

    print("loading model ...", flush=True)
    model, processor = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    feats = load_golden("audio_embeds.pt")  # not used; we need input_features
    # The encoder takes input_features (mel). Load from golden.
    from starling.granite.golden import load_golden as _lg
    # input_features is stored as part of golden capture? Check what's there.
    # We'll rebuild from audio instead.
    from starling.granite.audio import build_inputs, load_sample_audio
    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    input_features = inputs["input_features"].to(model.dtype)
    print("input_features:", tuple(input_features.shape), flush=True)

    stock_encoder = components["encoder"]

    # Baseline encoder (no fold). Build on a copy of stock weights so the fold
    # path gets pristine weights.
    print("\nbuilding baseline encoder (no fold) ...", flush=True)
    enc_base = FusedEncoder(stock_encoder, mode="cudagraph", fold_bn=False)
    out_base = enc_base(input_features)
    print("baseline out:", tuple(out_base.shape), flush=True)

    # Folded encoder. Reload model so fold mutates fresh weights.
    print("reloading model for folded encoder ...", flush=True)
    model2, _ = load_model_and_processor(attn_impl="eager")
    components2 = get_components(model2)
    enc_folded = FusedEncoder(components2["encoder"], mode="cudagraph", fold_bn=True)
    out_folded = enc_folded(input_features.to(model2.dtype))

    diff = (out_folded.float() - out_base.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    base_abs = float(out_base.float().abs().mean().item())
    print(f"\nfolded-vs-baseline max_abs={max_abs:.6e}  "
          f"mean_abs={mean_abs:.6e}  (out mean_abs={base_abs:.6e})", flush=True)
    print(f"relative mean diff: {mean_abs/max(base_abs,1e-9):.6e}", flush=True)

    # Time both via graph replay
    def time_replay(enc, label: str) -> float:
        trial_us: list[float] = []
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
                    enc._graph.replay() if hasattr(enc, "_graph") else enc(input_features)
                torch.cuda.synchronize()
                done += n
            e.record()
            torch.cuda.synchronize()
            total_ms = s.elapsed_time(e)
            trial_us.append(total_ms / args.reps * 1000.0)
            print(f"  [{label}] trial {t+1}: {trial_us[-1]:.2f} us/replay", flush=True)
        return median(trial_us)

    base_us = time_replay(enc_base, "baseline")
    folded_us = time_replay(enc_folded, "folded")
    speedup = base_us / folded_us if folded_us > 0 else float("nan")

    print(f"\n## Granite encoder BatchNorm-fold A/B "
          f"(median of {args.trials}x{args.reps} replays)\n")
    print("| variant | us/replay | speedup | max-abs diff | mean-abs diff |")
    print("|---------|-----------|---------|--------------|---------------|")
    print(f"| baseline (BN unfused) | {base_us:.2f} | 1.00x | - | - |")
    print(f"| folded (BN into conv) | {folded_us:.2f} | {speedup:.2f}x | "
          f"{max_abs:.2e} | {mean_abs:.2e} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
