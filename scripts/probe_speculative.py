#!/usr/bin/env python3
"""Validate the speculative decoder against golden greedy_ids (byte-exact)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.golden import load_golden, load_golden_text  # noqa: E402
from starling.loader import get_components, load_model_and_processor  # noqa: E402
from starling.pipeline import MegaPipeline  # noqa: E402
from starling.speculative import CTCBPEDraft, SpeculativeDecoder, load_out_llm  # noqa: E402


def main() -> int:
    print("[spec-test] loading model ...")
    model, processor = load_model_and_processor(attn_impl="eager")
    pipe = MegaPipeline(model, processor, encoder_mode="cudagraph", use_fused_llm=True)
    comps = get_components(model)
    tokenizer = processor.tokenizer

    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    feats = inputs["input_features"]
    input_ids = inputs["input_ids"]
    ifm = inputs.get("input_features_mask")

    # Load out_llm and build the draft extractor.
    out_llm = load_out_llm(device="cuda", dtype=torch.bfloat16)
    draft_ext = CTCBPEDraft(pipe.fused_encoder, out_llm)

    with torch.inference_mode():
        # (1) Encode + draft.
        mid_h, enc_hidden = draft_ext.encode_with_mid(feats)
        draft = draft_ext.draft(enc_hidden, mid_h)
        draft_text = tokenizer.decode(draft, skip_special_tokens=True)
        print(f"[spec-test] draft: {len(draft)} tokens")
        print(f"[spec-test] draft text: {draft_text[:120]}...")

        # (2) Build inputs_embeds (reuse enc_hidden for the projector).
        audio_embeds = pipe.projector(enc_hidden)
        inputs_embeds = pipe.build_inputs_embeds(input_ids, audio_embeds, ifm)
        print(f"[spec-test] inputs_embeds: {tuple(inputs_embeds.shape)}")

        # (3) Speculative decode.
        spec_dec = SpeculativeDecoder(pipe.llm, pipe.embed_tokens)
        res = spec_dec.generate(
            inputs_embeds, draft, max_new_tokens=100, eos_token_id=100257
        )
        print(f"\n[spec-test] speculative: {res.n_tokens} tokens, "
              f"{res.total_ms:.1f}ms, {res.tok_per_s:.1f} tok/s")
        print(f"[spec-test] draft_count={res.draft_count} accepted={res.accepted} "
              f"acceptance_rate={res.acceptance_rate:.1%} verify_forwards={res.verify_forwards}")

        # (4) Compare with golden.
        golden_gen = load_golden("greedy_ids.pt")[0, 271:]  # (100,)
        spec_ids = res.ids[0]

        # Find common length (spec might stop early at EOS).
        n = min(len(spec_ids), len(golden_gen))
        match = (spec_ids[:n] == golden_gen[:n]).all().item()
        first_diff = -1
        if not match:
            diffs = (spec_ids[:n] != golden_gen[:n]).nonzero()
            first_diff = int(diffs[0].item()) if len(diffs) > 0 else -1

        print(f"\n[spec-test] golden len={len(golden_gen)}, spec len={len(spec_ids)}")
        print(f"[spec-test] EXACT TOKEN MATCH (first {n}): {'YES' if match else 'NO'}")
        if not match:
            print(f"[spec-test] first diff at position {first_diff}")
            print(f"  golden[{first_diff}]={int(golden_gen[first_diff])} "
                  f"spec[{first_diff}]={int(spec_ids[first_diff])}")
            lo = max(0, first_diff - 3)
            hi = min(n, first_diff + 5)
            print(f"  golden[{lo}:{hi}] = {golden_gen[lo:hi].tolist()}")
            print(f"  spec  [{lo}:{hi}] = {spec_ids[lo:hi].tolist()}")

        spec_text = tokenizer.decode(spec_ids, skip_special_tokens=True)
        golden_text = load_golden_text().strip()
        golden_resp = golden_text.split("ASSISTANT:", 1)[1].strip()
        text_match = spec_text.strip() == golden_resp
        print(f"\n[spec-test] TEXT MATCH: {'YES' if text_match else 'NO'}")
        if not text_match:
            print(f"  golden: {golden_resp[:150]!r}")
            print(f"  spec:   {spec_text.strip()[:150]!r}")

        # (5) Also compare with non-speculative for cross-check.
        _, nonspec_ids = pipe.transcribe(feats, input_ids, ifm, max_new_tokens=100)
        nonspec_match = (nonspec_ids[0] == spec_ids).all().item()
        print(f"\n[spec-test] spec vs non-spec EXACT MATCH: {'YES' if nonspec_match else 'NO'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
