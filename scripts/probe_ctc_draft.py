#!/usr/bin/env python3
"""Probe: validate the BPE CTC draft algorithm from the IBM notebook.

Reconstructs the draft pipeline:
  1. Run encoder eagerly, capture mid-layer (layer num_layers//2 - 1 = 7) output.
  2. importance = 1 - blank_prob  (grapheme head `encoder.out` on mid-layer, blank=idx 0).
  3. Posterior-weighted pool of LAST-layer hidden with importance, window=4.
  4. out_llm (1024->100353) on pooled -> softmax -> argmax.
  5. CTC collapse: unique_consecutive, drop blank (label 0), map label i -> token i-1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from megapar.audio import build_inputs, load_sample_audio  # noqa: E402
from megapar.config import MODEL_ID  # noqa: E402
from megapar.encoder_mega import FusedEncoder  # noqa: E402
from megapar.loader import get_components, load_model_and_processor  # noqa: E402


def main() -> int:
    print("[probe] loading model ...")
    model, processor = load_model_and_processor(attn_impl="eager")
    comps = get_components(model)
    enc_module = comps["encoder"]
    tokenizer = processor.tokenizer

    # Build FusedEncoder (eager mode is fine; we call _block_eager directly)
    fused = FusedEncoder(enc_module, mode="eager")

    # Load out_llm head from the model repo
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo_id=MODEL_ID, filename="out_llm.safetensors"))
    out_llm = nn.Linear(1024, 100353, bias=True)
    with torch.no_grad():
        out_llm.weight.copy_(sd["weight"])
        out_llm.bias.copy_(sd["bias"])
    out_llm = out_llm.to(torch.bfloat16).cuda().eval()

    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    feats = inputs["input_features"].to(torch.bfloat16).cuda()

    window = 4
    mid_idx = fused.mid_idx  # 8

    with torch.inference_mode():
        # Prepare the block-attention mask on GPU (forward() does this; we bypass it).
        fused._prepare_block_mask(int(feats.shape[1]), feats.device)
        # Run encoder block-by-block (eager), capture mid_h before CTC feedback
        x = fused.input_linear(feats)
        mid_h = None
        for idx in range(fused.num_layers):
            x = fused._block_eager(idx, x)
            if (idx + 1) == mid_idx:
                mid_h = x  # output of block (mid_idx-1)=7, BEFORE ctc feedback
                mid_logits = fused.out(x)
                x = x + fused.out_mid(F.softmax(mid_logits, dim=-1))
        enc_hidden = x  # last layer output (1, T, 1024)
        print(f"[probe] enc_hidden={tuple(enc_hidden.shape)} mid_h={tuple(mid_h.shape)}")

        # importance from mid-layer grapheme blank prob
        mid_grapheme_logits = fused.out(mid_h)  # (1, T, 348)
        mid_probs = F.softmax(mid_grapheme_logits.float(), dim=-1)
        importance = 1.0 - mid_probs[:, :, 0]  # (1, T)
        print(f"[probe] importance shape={tuple(importance.shape)} "
              f"min={importance.min():.3f} max={importance.max():.3f} mean={importance.mean():.3f}")

        # posterior-weighted pool of enc_hidden
        B, T, D = enc_hidden.shape
        pad = (window - T % window) % window
        if pad > 0:
            eh = F.pad(enc_hidden, (0, 0, 0, pad))
            imp = F.pad(importance, (0, pad))
        else:
            eh, imp = enc_hidden, importance
        nw = eh.shape[1] // window
        eh_v = eh.view(B, nw, window, D)
        imp_v = imp.view(B, nw, window)
        weights = imp_v / (imp_v.sum(dim=-1, keepdim=True) + 1e-8)
        pooled = (eh_v * weights.unsqueeze(-1)).sum(dim=2)  # (1, nw, 1024)
        pooled = pooled.to(torch.bfloat16)
        print(f"[probe] pooled shape={tuple(pooled.shape)}")

        # BPE head
        bpe_logits = out_llm(pooled)  # (1, nw, 100353)
        bpe_probs = F.softmax(bpe_logits.float(), dim=-1)
        idx_lab = bpe_probs.argmax(dim=-1)[0]  # (nw,)
        print(f"[probe] label histogram: blank(0)={int((idx_lab==0).sum())}/{len(idx_lab)} "
              f"unique_nonblank={int((idx_lab>0).sum())}")

        # CTC collapse
        dedup = torch.unique_consecutive(idx_lab)
        non_blank = dedup[dedup > 0]
        token_ids = [int(t.item()) - 1 for t in non_blank]  # label i -> token i-1
        print(f"[probe] draft token count = {len(token_ids)}")
        print(f"[probe] draft token ids (first 40): {token_ids[:40]}")

        draft_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"\n[probe] DRAFT TRANSCRIPT:\n{draft_text}\n")

        # also print golden for comparison
        from megapar.golden import load_golden_text
        golden = load_golden_text()
        golden_resp = golden.split("ASSISTANT:", 1)[1].strip()
        print(f"[probe] GOLDEN RESPONSE:\n{golden_resp}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
