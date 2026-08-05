"""A/B benchmark for optional decode/encoder optimisations.

Loads the Granite model + golden ``inputs_embeds`` ONCE, then for each flag
combination:

  1. builds a fresh ``LLMMega`` (so the decode CUDA graph is captured cleanly),
  2. greedy-decodes ``max_new_tokens`` tokens,
  3. times the steady-state decode loop (median of ``reps`` runs),
  4. compares the generated ids to the golden greedy ids (byte-exactness check),
  5. compares the transcript to the golden transcript.

Output: a single markdown table to stdout + ``outputs/opt_ab.json``.

Usage:
  uv run python benchmarks/bench_opt_ab.py
  uv run python benchmarks/bench_opt_ab.py --reps 30 --tokens 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.flags import OptFlags, set_default_flags  # noqa: E402
from starling.granite.golden import load_golden, load_golden_text  # noqa: E402
from starling.granite.llm_mega import FusedLLMMega  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402

OUTPUTS = REPO_ROOT / "outputs"

# Default flag combos to sweep. Each entry: (label, OptFlags kwargs).
# All combos below enable tolerance_mode so the validator doesn't reject
# sdpa/flash combos -- we judge by the WER-style transcript + id match below.
DEFAULT_COMBOS: list[tuple[str, dict]] = [
    ("baseline (manual attn)",       dict()),
    ("sdpa_attention (math+gqa)",    dict(sdpa_attention=True, tolerance_mode=True)),
    ("flash_attention",              dict(flash_attention=True, tolerance_mode=True)),
]


@torch.inference_mode()
def bench_one(
    *,
    components,
    lm_head,
    inputs_embeds: torch.Tensor,
    golden_ids: torch.Tensor,
    golden_text: str,
    combo_flags: dict,
    max_new_tokens: int,
    reps: int,
    tokenizer,
) -> dict:
    """Build a fresh decoder under ``combo_flags`` and time it.

    Returns dict with: label, decode_ms (median), tok_per_s, n_tokens,
    byte_exact (bool), text_exact (bool), text.
    """
    of = OptFlags(**combo_flags)
    set_default_flags(of)  # FusedLLMMega reads get_default_flags() at __init__
    # Force single-step FusedLLMMega path (multistep is its own thing) so the
    # flag under test (attention backend) is the only variable.
    dec = FusedLLMMega(
        components["language_model"],
        lm_head,
        max_cache_len=896,
    )

    # Prime: prefill + capture once.  Generate() does both.
    res0 = dec.generate(
        inputs_embeds, max_new_tokens=max_new_tokens,
        eos_token_id=LLM_EOS_TOKEN_ID, tokenizer=tokenizer, capture=True,
    )
    n_gen = res0.n_tokens
    gen_ids_cpu = res0.ids.cpu()

    # Byte-exactness: compare generated ids to golden ids[:, T:] where T is
    # the prompt length baked into the golden ids. The golden greedy_ids.pt
    # stores prompt+generated, so slice the generated tail.
    T_prompt = inputs_embeds.shape[1]
    golden_gen = golden_ids[0, T_prompt:T_prompt + n_gen]
    min_len = min(golden_gen.numel(), gen_ids_cpu[0].numel())
    byte_exact = bool(
        torch.equal(golden_gen[:min_len], gen_ids_cpu[0, :min_len])
    )
    text = res0.text.strip()
    text_exact = (text == golden_text.strip())

    # Steady-state timing: re-run generate ``reps`` times. Each run reuses the
    # captured decode graph (capture is idempotent). Time only the call.
    times_ms: list[float] = []
    for _ in range(reps):
        # Reset cache state by re-prefilling inside generate(); generate()
        # calls prefill() which resets the cache. Good.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = dec.generate(
            inputs_embeds, max_new_tokens=max_new_tokens,
            eos_token_id=LLM_EOS_TOKEN_ID, tokenizer=tokenizer, capture=False,
        )
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    decode_ms = median(times_ms)
    tok_per_s = r.n_tokens / max(decode_ms / 1000.0, 1e-9)

    # Free decoder tensors before next combo
    del dec
    torch.cuda.empty_cache()

    return {
        "n_tokens": n_gen,
        "decode_ms": decode_ms,
        "tok_per_s": tok_per_s,
        "byte_exact": byte_exact,
        "text_exact": text_exact,
        "text": text[:120],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--tokens", type=int, default=150)
    args = ap.parse_args()

    print("loading granite model + golden artefacts ...", flush=True)
    model, processor = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    lm_head = model.lm_head
    tokenizer = processor.tokenizer
    inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
    golden_ids = load_golden("greedy_ids.pt")
    golden_text = load_golden_text()

    results: list[dict] = []
    baseline_ms = None
    for label, combo in DEFAULT_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        r = bench_one(
            components=components,
            lm_head=lm_head,
            inputs_embeds=inputs_embeds,
            golden_ids=golden_ids,
            golden_text=golden_text,
            combo_flags=combo,
            max_new_tokens=args.tokens,
            reps=args.reps,
            tokenizer=tokenizer,
        )
        if baseline_ms is None:
            baseline_ms = r["decode_ms"]
        speedup = baseline_ms / r["decode_ms"] if r["decode_ms"] > 0 else float("nan")
        r["label"] = label
        r["speedup_vs_baseline"] = speedup
        print(
            f"  decode_ms={r['decode_ms']:.1f}  tok/s={r['tok_per_s']:.1f}  "
            f"speedup={speedup:.2f}x  byte_exact={r['byte_exact']}  "
            f"text_exact={r['text_exact']}",
            flush=True,
        )
        if not r["text_exact"]:
            print(f"  transcript: {r['text']!r}", flush=True)
        results.append(r)

    # Restore process defaults
    set_default_flags(OptFlags())

    # Print markdown table
    print("\n## Granite decode A/B (steady-state, median of "
          f"{args.reps}, {args.tokens} tokens)\n")
    print("| variant | decode ms | tok/s | speedup | byte-exact | text-exact |")
    print("|---------|-----------|-------|---------|------------|------------|")
    for r in results:
        print(
            f"| {r['label']} | {r['decode_ms']:.1f} | {r['tok_per_s']:.1f} | "
            f"{r['speedup_vs_baseline']:.2f}x | {'yes' if r['byte_exact'] else 'NO'} | "
            f"{'yes' if r['text_exact'] else 'NO'} |"
        )

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "opt_ab.json").write_text(json.dumps({
        "tokens": args.tokens,
        "reps": args.reps,
        "results": results,
    }, indent=2))
    print(f"\nwrote {OUTPUTS / 'opt_ab.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
