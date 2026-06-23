"""Quick correctness probe: QuantLLMMega vs golden transcript.

Not a formal test (see tests/test_quant.py); a fast diagnostic to confirm INT8
quantisation preserves the transcript.  (The full logit-diff + speed numbers
live in scripts/bench_quant.py.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from starling.config import LLM_EOS_TOKEN_ID
from starling.golden import load_golden, load_golden_text
from starling.loader import get_components, load_model_and_processor
from starling.parakeet.gpu_lock import with_gpu_lock
from starling.quant import QuantLLMMega


def wer_simple(ref: str, hyp: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def main() -> int:
    with with_gpu_lock(
        session="granite-mega",
        model="granite-speech-4.1-2b",
        eta_min=5,
        note="probe_quant_correctness: INT8 decode vs golden",
    ):
        print("loading model ...", flush=True)
        model, proc = load_model_and_processor(attn_impl="eager")
        comps = get_components(model)
        lm = comps["language_model"]
        lm_head = model.lm_head
        inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)

        golden_gen = load_golden("greedy_ids.pt")[0, 271:]
        golden_resp = load_golden_text().strip().split("ASSISTANT:", 1)[1].strip()

        print("INT8 quant decode ...", flush=True)
        qdec = QuantLLMMega(lm, lm_head, max_cache_len=640)
        res_q = qdec.generate(inputs_embeds, max_new_tokens=100, eos_token_id=LLM_EOS_TOKEN_ID)

        tok_match = int((res_q.ids[0] == golden_gen).sum().item())
        tok_pct = tok_match / golden_gen.shape[0] * 100.0
        text_q = proc.tokenizer.decode(res_q.ids[0], skip_special_tokens=True)
        w = wer_simple(golden_resp, text_q)

        print("\n===== RESULTS =====")
        print(f"token match: {tok_match}/{golden_gen.shape[0]} ({tok_pct:.1f}%)")
        print(f"WER vs golden: {w:.4f}")
        print(f"\ngolden : {golden_resp[:160]!r}")
        print(f"quant  : {text_q[:160]!r}")
        torch.cuda.synchronize()
        print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
