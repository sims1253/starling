"""Contention-robust decode-step measurement via torch.profiler.

The profiler's self_device_time_total sums per-kernel GPU execution time and is
robust to other processes on the GPU (it measures kernel runtime, not wall/queue
time). This builds a decoder, captures its graph, and reports:
  - total us/step (sum of all kernel self-times / N steps)
  - per-category breakdown
  - launch count/step

Run before AND after a change in separate invocations; compare the totals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.flags import get_default_flags  # noqa: E402
from starling.granite.golden import load_golden  # noqa: E402
from starling.granite.llm_mega import FusedLLMMega  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402


@torch.inference_mode()
def main() -> int:
    print(f"flags: {get_default_flags()}", flush=True)
    print("loading ...", flush=True)
    model, _ = load_model_and_processor(attn_impl="eager")
    comps = get_components(model)
    ie = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
    T = ie.shape[1]

    dec = FusedLLMMega(comps["language_model"], model.lm_head, max_cache_len=896)
    nt = dec.prefill(ie)
    dec.capture(nt, T)

    N = 100
    for _ in range(20):
        dec._graph.replay()
    torch.cuda.synchronize()

    # 3 profiler trials, take min total (least contended)
    best_total = None
    best_ka = None
    for _ in range(3):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(N):
                dec._graph.replay()
            torch.cuda.synchronize()
        ka = prof.key_averages()
        total = sum(e.self_device_time_total for e in ka) / N  # us/step
        if best_total is None or total < best_total:
            best_total = total
            best_ka = ka

    launches = sum(e.count for e in best_ka) / N
    print(f"\n=== decode step: {best_total:.1f} us/step, {launches:.0f} launches/step, "
          f"{1e6/best_total:.0f} tok/s (GPU-time, best of 3) ===\n")

    # category breakdown
    rows = sorted(best_ka, key=lambda e: e.self_device_time_total, reverse=True)[:10]
    print(f"{'us/step':>9} {'calls/step':>11} {'us/call':>9}  kernel")
    for e in rows:
        print(f"{e.self_device_time_total/N:>9.1f} {e.count/N:>11.1f} "
              f"{e.self_device_time_total/e.count:>9.2f}  {e.key[:54]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
