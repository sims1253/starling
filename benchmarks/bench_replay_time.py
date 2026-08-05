"""Steady-state decode-step timing via raw CUDA-graph replay.

Isolates the per-token decode cost from prefill, the Python generate loop, and
host syncs (``.item()``). Captures the single-step decode graph, then replays
it ``reps`` times back-to-back under CUDA events. This is the truest measure
of "how fast is one decode step on the GPU".

Used to A/B code changes that touch the captured decode forward (RoPE,
lm_head, attention backend): toggle the implementation in the source, rerun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.granite.golden import load_golden  # noqa: E402
from starling.granite.llm_mega import FusedLLMMega  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="current")
    ap.add_argument("--reps", type=int, default=400, help="graph replays per measurement")
    ap.add_argument("--trials", type=int, default=5, help="independent trials")
    args = ap.parse_args()

    print(f"[{args.label}] loading model ...", flush=True)
    model, processor = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)

    dec = FusedLLMMega(components["language_model"], model.lm_head, max_cache_len=896)

    # Capture the decode graph: prefill then capture()
    print(f"[{args.label}] prefill ...", flush=True)
    next_token = dec.prefill(inputs_embeds)
    T = inputs_embeds.shape[1]
    print(f"[{args.label}] capture ...", flush=True)
    dec.capture(next_token, T)
    print(f"[{args.label}] captured, timing ...", flush=True)

    # Correctness check is done separately in tests/test_llm_mega.py -- this
    # bench focuses purely on steady-state replay latency. We assume the
    # captured graph is correct (verified before/after the change via pytest).
    byte_exact = "see pytest"

    # Use the graph that capture() already built. Time ``reps`` back-to-back
    # replays -- each replay = one decode step on the GPU. This isolates the
    # captured forward from prefill and from the host-side Python loop.
    # Replay in chunks with a sync between them so the CUDA queue never fills
    # to the point of deadlocking under sustained launch pressure.
    CHUNK = 50
    trial_us: list[float] = []
    for trial in range(args.trials):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        done = 0
        while done < args.reps:
            n = min(CHUNK, args.reps - done)
            for _ in range(n):
                dec._graph.replay()
            torch.cuda.synchronize()
            done += n
        e.record()
        torch.cuda.synchronize()
        total_ms = s.elapsed_time(e)
        per_us = total_ms / args.reps * 1000.0
        trial_us.append(per_us)
        print(f"  trial {trial+1}/{args.trials}: {per_us:.2f} us/step", flush=True)

    med = median(trial_us)
    print(f"[{args.label}] decode_step_us (median of {args.trials}x{args.reps} replays) "
          f"= {med:.2f} us/step  ({1e6/med:.1f}k step/s)", flush=True)
    print(f"RESULT\t{args.label}\t{med:.2f}\t{byte_exact}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
