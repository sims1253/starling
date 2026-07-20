"""Probe: dump TRUE-path layer-0 intermediates for the MOSS golden short prefill.

Unlike probe_moss_llm_layer0.py (manual module calls, which diverges from the
real model by up to 0.125 at layer 0), this runs the real Qwen3Model.forward
with exact-width DynamicCache (the verified eager golden path) and
captures intermediates via monkeypatches/hooks:

  n (post input_layernorm [T,2048]), qn/kn (post q/k norm [H,T,D]),
  qr/kr (post-rope [H,T,D]), scores_h0/probs_h0 ([T,T], head 0),
  attn_out ([T,2048] post o_proj), down ([T,2048]), layer_out ([T,2048]).

Output: golden/raw/l0true/<name>.f32 (row-major, shapes as above).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "golden/raw/l0true"


def save(name: str, t: torch.Tensor) -> None:
    OUT.mkdir(exist_ok=True)
    t.detach().float().cpu().contiguous().numpy().astype(np.float32).tofile(OUT / f"{name}.f32")
    print(name, tuple(t.shape))


def main() -> int:
    from starling.moss.loader import load_model_and_processor

    fx = sys.argv[1] if len(sys.argv) > 1 else "short"
    ge = np.fromfile(REPO / f"golden/raw/moss_{fx}_inputs_embeds.f32", dtype=np.float32)
    T = ge.size // 2048
    embeds = torch.from_numpy(ge.reshape(1, T, 2048)).to(torch.bfloat16).cuda()

    model, _ = load_model_and_processor()
    lm = model.model.language_model

    import transformers.models.qwen3.modeling_qwen3 as mq

    captured = {}

    real_rope = mq.apply_rotary_pos_emb
    def rope_spy(q, k, cos, sin, **kw):
        qr, kr = real_rope(q, k, cos, sin, **kw)
        if q.shape[1] == 16 and q.shape[2] == T and "qr" not in captured:
            captured["qr"] = qr[0].clone()   # [H,T,D]
            captured["kr"] = kr[0].clone()   # [KV,T,D]
        return qr, kr
    mq.apply_rotary_pos_emb = rope_spy

    layer0 = lm.layers[0]
    hooks = []
    hooks.append(layer0.input_layernorm.register_forward_hook(lambda _m, _i, o: save("n", o[0])))
    def grab(name, t):
        captured.setdefault(name, t)
        return None  # a forward hook's non-None return REPLACES the module output
    hooks.append(layer0.self_attn.q_norm.register_forward_hook(lambda _m, _i, o: grab("qn", o[0].transpose(0, 1).clone())))
    hooks.append(layer0.self_attn.k_norm.register_forward_hook(lambda _m, _i, o: grab("kn", o[0].transpose(0, 1).clone())))
    hooks.append(layer0.self_attn.o_proj.register_forward_hook(lambda _m, _i, o: save("attn_out", o[0])))
    hooks.append(layer0.mlp.down_proj.register_forward_hook(lambda _m, _i, o: save("down", o[0])))
    hooks.append(layer0.register_forward_hook(lambda _m, _i, o: save("layer_out", (o[0] if isinstance(o, tuple) else o)[0])))

    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=lm.config)
    pos = torch.arange(T, device=embeds.device).unsqueeze(0)

    with torch.no_grad():
        lm(inputs_embeds=embeds, attention_mask=None, position_ids=pos,
           past_key_values=cache, use_cache=True,
           cache_position=torch.arange(T, device=embeds.device))
    for h in hooks:
        h.remove()
    mq.apply_rotary_pos_emb = real_rope

    for name in ("qn", "kn", "qr", "kr"):
        save(name, captured[name])
    # sanity vs previously captured model layer outputs
    ref = np.fromfile(REPO / f"golden/raw/moss_{fx}_llm_hidden.f32", dtype=np.float32).reshape(29, T, 2048)
    mine = np.fromfile(OUT / "layer_out.f32", dtype=np.float32)
    print("sanity layer0 out vs model layers ref max-abs:", float(np.abs(mine - ref[1].ravel()).max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
