"""Profile one granite decode step to see where the ~4.5ms actually goes.

Captures a CUDA-graph decode step, then runs it under torch.profiler to get a
per-kernel breakdown. This answers: is the bottleneck the GEMMs (cuBLAS), the
attention, the Triton elementwise kernels, or pure launch overhead?

Usage: uv run python benchmarks/bench_decode_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.granite.golden import load_golden  # noqa: E402
from starling.granite.llm_mega import FusedLLMMega  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402


@torch.inference_mode()
def main() -> int:
    print("loading model ...", flush=True)
    model, _ = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
    dec = FusedLLMMega(components["language_model"], model.lm_head, max_cache_len=896)
    nt = dec.prefill(inputs_embeds)
    T = inputs_embeds.shape[1]
    dec.capture(nt, T)

    # warmup the profiler caches
    for _ in range(5):
        dec._graph.replay()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    print("\nprofiling 200 decode-step replays ...", flush=True)
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(200):
            dec._graph.replay()
        torch.cuda.synchronize()

    print("\n=== top CUDA kernels by total GPU time ===\n", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
