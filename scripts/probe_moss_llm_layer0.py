"""Probe: dump reference layer-0 INTERMEDIATES for the MOSS golden short prefill.

Replicates Qwen3 layer 0 manually from the reference model's modules on the
golden inputs_embeds, saving each intermediate as f32 raw for C++ comparison:
  n (post input_layernorm), q/k post q_norm+k_norm (pre-rope, [H,T,D]),
  q/k post-rope ([H,T,D]), scores head0 ([T,T]), probs head0,
  attn_out ([T,2048] post o_proj), x_mid (residual after attn),
  gate/up/silu*up/down outs, layer_out.
Layout: all f32 row-major [rows, cols] with rows=T unless noted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "golden/raw/l0"


def save(name: str, t: torch.Tensor) -> None:
    OUT.mkdir(exist_ok=True)
    t.detach().float().cpu().numpy().astype(np.float32).tofile(OUT / f"{name}.f32")
    print(name, tuple(t.shape))


def main() -> int:
    from starling.moss.loader import load_model_and_processor

    ge = np.fromfile(REPO / "golden/raw/moss_short_inputs_embeds.f32", dtype=np.float32)
    T = ge.size // 2048
    x = torch.from_numpy(ge.reshape(1, T, 2048)).to(torch.bfloat16).cuda()

    model, _ = load_model_and_processor()
    lm = model.model.language_model
    layer = lm.layers[0]
    attn = layer.self_attn
    H, KV, D = 16, 8, 128

    pos = torch.arange(T, device=x.device).unsqueeze(0)
    neg = torch.finfo(x.dtype).min
    ar = torch.arange(T, device=x.device)
    q_ = torch.arange(T, device=x.device).unsqueeze(1)
    mask2 = torch.where(ar[None, None, None, :] <= q_[None, None, :, :], 0.0, neg).to(x.dtype)

    with torch.no_grad():
        n = layer.input_layernorm(x)
        save("n", n[0])
        q = attn.q_proj(n).view(1, T, H, D)
        k = attn.k_proj(n).view(1, T, KV, D)
        v = attn.v_proj(n).view(1, T, KV, D)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        save("q_prenorm_rope", q[0].permute(1, 0, 2).reshape(H * T, D) if False else q[0].transpose(0, 1))
        save("k_pre_rope", k[0].transpose(0, 1))
        qt = q.transpose(1, 2)  # [1,H,T,D]
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        cos, sin = lm.rotary_emb(vt, pos)
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        qr, kr = apply_rotary_pos_emb(qt, kt, cos, sin)
        save("q_rope", qr[0].transpose(0, 1))
        save("k_rope", kr[0].transpose(0, 1))
        kfull = kr.repeat_interleave(2, dim=1)
        vfull = vt.repeat_interleave(2, dim=1)
        scores = torch.matmul(qr, kfull.transpose(-1, -2)) * attn.scaling
        scores = scores + mask2
        save("scores_h0", scores[0, 0])
        probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        save("probs_h0", probs[0, 0])
        ctx = torch.matmul(probs, vfull)  # [1,H,T,D]
        co = ctx.transpose(1, 2).reshape(1, T, H * D)
        attn_out = attn.o_proj(co)
        save("attn_out", attn_out[0])
        x_mid = x + attn_out
        save("x_mid", x_mid[0])
        n2 = layer.post_attention_layernorm(x_mid)
        g = layer.mlp.gate_proj(n2)
        u = layer.mlp.up_proj(n2)
        import torch.nn.functional as F
        z = F.silu(g) * u
        down = layer.mlp.down_proj(z)
        save("down", down[0])
        out = x_mid + down
        save("layer_out", out[0])

        # sanity vs the real layer forward through the model path (already verified
        # byte-exact vs golden at the model level in probe_moss_llm_layers)
        ref = np.fromfile(REPO / "golden/raw/moss_short_llm_hidden.f32", dtype=np.float32).reshape(29, 107, 2048)
        mine = out[0].float().cpu().numpy()
        print("manual layer0 vs model layer0 max-abs:", float(np.abs(mine - ref[1]).max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
