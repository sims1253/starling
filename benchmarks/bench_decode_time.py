"""Time the granite FusedLLMMega decode loop against the golden inputs.

Standalone timing harness (no flag sweep) -- used with ``git stash`` to A/B
a code change against the committed baseline.

Usage:
  uv run python benchmarks/bench_decode_time.py            # times whatever code is checked out
  uv run python benchmarks/bench_decode_time.py --reps 30 --tokens 150
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

from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.granite.golden import load_golden  # noqa: E402
from starling.granite.llm_mega import FusedLLMMega  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--label", default="current")
    args = ap.parse_args()

    print(f"[{args.label}] loading model ...", flush=True)
    model, processor = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    tokenizer = processor.tokenizer
    inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
    golden_ids = load_golden("greedy_ids.pt")

    dec = FusedLLMMega(components["language_model"], model.lm_head, max_cache_len=896)

    # one capture + correctness check
    res0 = dec.generate(
        inputs_embeds, max_new_tokens=args.tokens,
        eos_token_id=LLM_EOS_TOKEN_ID, tokenizer=tokenizer, capture=True,
    )
    T = inputs_embeds.shape[1]
    golden_gen = golden_ids[0, T:T + res0.n_tokens]
    min_len = min(golden_gen.numel(), res0.ids.numel())
    byte_exact = bool(torch.equal(golden_gen[:min_len], res0.ids[0, :min_len].cpu()))
    print(f"[{args.label}] byte_exact={byte_exact}  n_tokens={res0.n_tokens}", flush=True)

    times_ms: list[float] = []
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec.generate(
            inputs_embeds, max_new_tokens=args.tokens,
            eos_token_id=LLM_EOS_TOKEN_ID, tokenizer=tokenizer, capture=False,
        )
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    med = median(times_ms)
    tps = res0.n_tokens / (med / 1000.0)
    print(f"[{args.label}] decode_ms(median of {args.reps})={med:.2f}  "
          f"tok/s={tps:.1f}  min={min(times_ms):.2f}  max={max(times_ms):.2f}",
          flush=True)
    # machine-readable single line for easy diffing
    print(f"RESULT\t{args.label}\t{med:.2f}\t{tps:.1f}\t{byte_exact}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
