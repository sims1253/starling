#!/usr/bin/env python3
"""Compute LCS and match runs between draft and golden to assess max acceptance."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.audio import build_inputs, load_sample_audio
from starling.golden import load_golden
from starling.loader import get_components, load_model_and_processor
from starling.pipeline import MegaPipeline
from starling.speculative import CTCBPEDraft, load_out_llm


def main():
    model, processor = load_model_and_processor(attn_impl="eager")
    pipe = MegaPipeline(model, processor, encoder_mode="cudagraph", use_fused_llm=True)
    tokenizer = processor.tokenizer
    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    out_llm = load_out_llm(device="cuda", dtype=torch.bfloat16)
    draft_ext = CTCBPEDraft(pipe.fused_encoder, out_llm)
    with torch.inference_mode():
        mid_h, enc_hidden = draft_ext.encode_with_mid(inputs["input_features"])
        draft = draft_ext.draft(enc_hidden, mid_h)
        golden = load_golden("greedy_ids.pt")[0, 271:].tolist()

    m, n = len(draft), len(golden)
    # LCS DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if draft[i - 1] == golden[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    print(f"draft={m} golden={n} LCS={lcs} ({lcs / min(m, n):.1%} of shorter)")
    print(f"Max multi-round acceptance: {lcs}/{m} = {lcs / m:.1%}")
    print(f"Golden tokens covered by draft: {lcs}/{n} = {lcs / n:.1%}")

    # Longest consecutive run
    best_run = 0
    best_pos = (-1, -1)
    for i in range(m):
        for j in range(n):
            k = 0
            while i + k < m and j + k < n and draft[i + k] == golden[j + k]:
                k += 1
            if k > best_run:
                best_run = k
                best_pos = (i, j)
    print(f"\nLongest consecutive run: {best_run} at draft[{best_pos[0]}], golden[{best_pos[1]}]")

    # All consecutive runs >= 3
    print("\nConsecutive runs >= 3:")
    seen = set()
    for i in range(m):
        for j in range(n):
            k = 0
            while i + k < m and j + k < n and draft[i + k] == golden[j + k]:
                k += 1
            if k >= 3 and (i, j) not in seen:
                seen.add((i, j))
                toks = [draft[i + x] for x in range(k)]
                dec = tokenizer.decode(toks)
                print(f"  draft[{i}:{i+k}] golden[{j}:{j+k}] len={k}: {dec!r}")

    # Count total tokens in runs >= 3
    in_run = [False] * m
    for i in range(m):
        for j in range(n):
            k = 0
            while i + k < m and j + k < n and draft[i + k] == golden[j + k]:
                k += 1
            if k >= 3:
                for x in range(k):
                    in_run[i + x] = True
    total_in_runs = sum(in_run)
    print(f"\nDraft tokens in runs >= 3: {total_in_runs}/{m} = {total_in_runs / m:.1%}")


if __name__ == "__main__":
    main()
