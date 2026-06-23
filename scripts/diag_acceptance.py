#!/usr/bin/env python3
"""Diagnose draft acceptance: compare draft tokens vs golden greedy tokens."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.golden import load_golden  # noqa: E402
from starling.loader import get_components, load_model_and_processor  # noqa: E402
from starling.pipeline import MegaPipeline  # noqa: E402
from starling.speculative import CTCBPEDraft, load_out_llm  # noqa: E402


def main() -> int:
    model, processor = load_model_and_processor(attn_impl="eager")
    pipe = MegaPipeline(model, processor, encoder_mode="cudagraph", use_fused_llm=True)
    tokenizer = processor.tokenizer

    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    feats = inputs["input_features"]

    out_llm = load_out_llm(device="cuda", dtype=torch.bfloat16)
    draft_ext = CTCBPEDraft(pipe.fused_encoder, out_llm)

    with torch.inference_mode():
        mid_h, enc_hidden = draft_ext.encode_with_mid(feats)
        draft = draft_ext.draft(enc_hidden, mid_h)

        golden_gen = load_golden("greedy_ids.pt")[0, 271:]  # (100,)
        golden_list = golden_gen.tolist()

        print(f"draft len={len(draft)}, golden len={len(golden_list)}")
        print(f"\ndraft[0:15]  = {draft[:15]}")
        print(f"golden[0:15] = {golden_list[:15]}")
        print(f"\ndraft[0] decoded:  {tokenizer.decode([draft[0]])!r}")
        print(f"golden[0] decoded: {tokenizer.decode([golden_list[0]])!r}")
        print(f"golden[1] decoded: {tokenizer.decode([golden_list[1]])!r}")

        # position-by-position comparison
        n = min(len(draft), len(golden_list))
        matches = sum(1 for i in range(n) if draft[i] == golden_list[i])
        print(f"\nposition-aligned matches: {matches}/{n} ({matches/n:.1%})")

        # first few comparisons
        print("\nfirst 20 position comparisons:")
        for i in range(min(20, n)):
            d = tokenizer.decode([draft[i]])
            g = tokenizer.decode([golden_list[i]])
            m = "MATCH" if draft[i] == golden_list[i] else "DIFF"
            print(f"  [{i:2d}] {m} draft={draft[i]:6d} {d!r:15s}  golden={golden_list[i]:6d} {g!r}")

        # set overlap (how many draft tokens appear ANYWHERE in golden)
        golden_set = set(golden_list)
        draft_in_golden = sum(1 for t in draft if t in golden_set)
        print(f"\ndraft tokens appearing in golden (any pos): {draft_in_golden}/{len(draft)} ({draft_in_golden/len(draft):.1%})")

        # check if golden tokens appear in draft as a subsequence
        # (i.e., can we find golden tokens in order within the draft?)
        gi = 0
        for dt in draft:
            if gi < len(golden_list) and dt == golden_list[gi]:
                gi += 1
        print(f"golden tokens findable as subsequence in draft: {gi}/{len(golden_list)} ({gi/len(golden_list):.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
