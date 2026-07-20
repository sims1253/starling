"""Probe: dump per-layer Qwen3 hidden states for the MOSS golden short prefill.

Feeds golden/raw/moss_short_inputs_embeds.f32 through the reference
language_model with the EXACT mask/positions of reference.py's prefill and
captures every decoder layer's output hidden state via forward hooks.

Output: golden/raw/moss_short_llm_hidden.f32 — float32, layout
[29][107][2048] row-major: slice 0 = the input embeddings, slice L = output
of decoder layer L-1 (so slice 28 = final-norm INPUT... no: layer 27 output).
Slices are bf16-computed values upcast to float32 on save.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from starling.moss.loader import load_model_and_processor

    ge = np.fromfile(REPO / "golden/raw/moss_short_inputs_embeds.f32", dtype=np.float32)
    T = ge.size // 2048
    embeds = torch.from_numpy(ge.reshape(1, T, 2048)).to(torch.bfloat16).cuda()

    model, _proc = load_model_and_processor()
    lm = model.model.language_model
    lm_head = model.lm_head

    max_cache_len = 2048
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=lm.config)
    dtype = embeds.dtype
    device = embeds.device
    neg = torch.finfo(dtype).min
    ar = torch.arange(max_cache_len, device=device)
    q = torch.arange(T, device=device).unsqueeze(1)
    mask4 = torch.where(ar[None, None, None, :] <= q[None, None, :, :], 0.0, neg).to(dtype)
    pos = torch.arange(T, device=device).unsqueeze(0)
    cp = torch.arange(T, device=device)

    layers_out: list[torch.Tensor] = [embeds[0].float().cpu()]
    hooks = []
    for layer in lm.layers:
        def hook(_m, _args, out, _acc=layers_out):
            hidden = out[0] if isinstance(out, tuple) else out
            _acc.append(hidden[0].float().cpu())
        hooks.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        out = lm(
            inputs_embeds=embeds,
            attention_mask=mask4,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            cache_position=cp,
        )
    for h in hooks:
        h.remove()

    last = out.last_hidden_state[:, -1:, :]
    logits = lm_head(last).detach().float().cpu().numpy()
    golden_logits = np.fromfile(REPO / "golden/raw/moss_short_prefill_logits.f32", dtype=np.float32)
    print("sanity: prefill logits max-abs vs golden:", float(np.abs(logits.ravel() - golden_logits).max()))
    print("sanity: argmax", int(logits.argmax()), "golden argmax", int(golden_logits.argmax()))

    arr = np.stack([t.numpy() for t in layers_out])  # [29, T, 2048]
    out_path = REPO / "golden/raw/moss_short_llm_hidden.f32"
    arr.astype(np.float32).tofile(out_path)
    (REPO / "golden/raw/moss_short_llm_hidden.json").write_text(
        json.dumps({"shape": list(arr.shape), "dtype": "float32",
                    "layout": "slice0=input embeds; slice L=layer L-1 output"})
    )
    print("wrote", out_path, arr.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
