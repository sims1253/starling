"""Measure fp8 weight-quant decode speedup vs the bf16 default.

Builds both decoders in one process, profiles each (GPU-time, best of 3),
and reports the per-step delta + the category breakdown so we can see the
GEMV cost drop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.flags import OptFlags, set_default_flags, get_default_flags  # noqa: E402
from starling.granite.golden import load_golden  # noqa: E402
from starling.granite.llm_mega import FusedLLMMega  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402


def profile_step(dec, N=100, warmup=20, trials=3):
    for _ in range(warmup):
        dec._graph.replay()
    torch.cuda.synchronize()
    best = None
    for _ in range(trials):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(N):
                dec._graph.replay()
            torch.cuda.synchronize()
        ka = prof.key_averages()
        total = sum(e.self_device_time_total for e in ka) / N
        launches = sum(e.count for e in ka) / N
        if best is None or total < best[0]:
            best = (total, launches, ka)
    return best


@torch.inference_mode()
def main() -> int:
    print("loading ...", flush=True)
    model, _ = load_model_and_processor(attn_impl="eager")
    comps = get_components(model)
    ie = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
    T = ie.shape[1]
    saved = get_default_flags()

    # bf16 baseline
    set_default_flags(OptFlags())
    d0 = FusedLLMMega(comps["language_model"], model.lm_head, max_cache_len=896)
    nt = d0.prefill(ie); d0.capture(nt, T)
    t0, l0, ka0 = profile_step(d0)

    # fp8
    set_default_flags(OptFlags(fp8_weights=True, tolerance_mode=True))
    d1 = FusedLLMMega(comps["language_model"], model.lm_head, max_cache_len=896)
    d1.prefill(ie); d1.capture(nt, T)
    t1, l1, ka1 = profile_step(d1)

    set_default_flags(saved)

    print(f"\n{'':>12} {'bf16':>10} {'fp8':>10} {'speedup':>9}")
    print(f"{'us/step':>12} {t0:>10.0f} {t1:>10.0f} {t0/t1:>8.2f}x")
    print(f"{'tok/s':>12} {1e6/t0:>10.0f} {1e6/t1:>10.0f}")
    print(f"{'launches':>12} {l0:>10.0f} {l1:>10.0f}")
    print(f"\nsaves {t0-t1:.0f} us/step ({(t0-t1)/t0*100:.1f}% faster)")

    # GEMV breakdown
    def gemv_us(ka):
        return sum(e.self_device_time_total for e in ka if "gemvx" in e.key) / 100  # N=100
    g0 = gemv_us(ka0); g1 = gemv_us(ka1)
    print(f"\nGEMV total:  bf16={g0:.0f} us/step  fp8={g1:.0f} us/step  ({g0/g1:.2f}x, saves {g0-g1:.0f} us)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
