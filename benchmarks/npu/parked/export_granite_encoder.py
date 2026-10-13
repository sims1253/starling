# PARKED (2026-10-13): the Pixel 10 Pro TPU compiler cannot build these graphs
# (30 s full-length tensors overflow on-chip memory; sequence slices fail or
# exhaust the phone's memory). Kept for reference only, not wired into the
# engine or the compile scripts. See docs/fast-engine.md, "Pixel TPU".
"""Export one segment of Granite Speech's audio encoder (CTC conformer with
block-local Shaw attention, mid-CTC self-conditioning, BLIP-2 Q-Former
projector) as a TPU artifact source graph, for a fixed bucket of P stacked
mel frames (P a multiple of the attention block, 200 = 4 s; the projector
covers the first P // window * window frames, e.g. P = 1600: 1590 = 31.8 s):

  inputs:  x [1, P, input_dim] f32 stacked frames (segment L0 = 0) or
           h [1, P, d] f32 (later segments),
           fmask [1, P, 1] f32 (1 on real frames, 0 past them),
           kbias [P / C, 1, 1, C] f32 (0 real / -1e4 padding frames, per
           attention block of C = context_size)
  output:  h [1, P, d] f32, or, for the last segment, the projector rows
           [1, P // window * queries, llm hidden] f32 (before the LLM's
           embedding multiplier).

Exact against the reference on the real frames: the attention block
holding the end masks the padding keys (like the reference's padded block),
the depthwise conv sees zeros past the audio (fmask), and the Q-Former's
last window sees zero frames past it. fp16 weights (int8 with
NPU_WEIGHTS=int8, tpu_export.py).

python -I export_granite_encoder.py MODEL P L0 L1 OUT.tflite
"""
import glob
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

model, P, L0, L1, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
snap = model if os.path.isdir(model) else glob.glob(os.path.expanduser(
    f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpu_export  # noqa: E402

torch.set_grad_enabled(False)
cfg = json.load(open(os.path.join(snap, "config.json")))
ec, pc = cfg["encoder_config"], cfg["projector_config"]
D, H, DH, NL, C = ec["hidden_dim"], ec["num_heads"], ec["dim_head"], ec["num_layers"], ec["context_size"]
IN, KS, MAXPOS, VOC = ec["input_dim"], ec["conv_kernel_size"], ec["max_pos_emb"], ec["output_dim"]
FF = D * ec["feedforward_mult"]
WIN, DS = cfg["window_size"], cfg["downsample_rate"]
NQ = WIN // DS
QH, QL, QEPS = pc["num_attention_heads"], pc["num_hidden_layers"], pc["layer_norm_eps"]
OUT = cfg["text_config"]["hidden_size"]
MID = NL // 2                      # self-conditioning after this many layers
assert P % C == 0, f"P={P} must be a multiple of {C}"
NB, NW = P // C, P // WIN
last = L1 == NL
EPS = 1e-5                         # nn.LayerNorm / BatchNorm1d defaults
W = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("encoder.layers."):
                if L0 <= int(k.split(".")[2]) < L1:
                    W[k] = h.get_tensor(k).float()
            elif k.startswith("encoder.input_linear.") and L0 == 0:
                W[k] = h.get_tensor(k).float()
            elif k.startswith("encoder.out") and L0 < MID <= L1:
                W[k] = h.get_tensor(k).float()
            elif k.startswith("projector.") and last:
                W[k] = h.get_tensor(k).float()
assert W, "no encoder.* tensors (wrong checkpoint?)"
seq = torch.arange(C)
DIST = torch.clamp(seq[:, None] - seq[None, :], -C, C) + MAXPOS    # [C, C] rel-pos table rows


def Pm(n):
    return nn.Parameter(W.pop(n), requires_grad=False)


def ln(x, m, p):
    return F.layer_norm(x, (x.shape[-1],), getattr(m, p + "_w"), getattr(m, p + "_b"), EPS)


class Seg(nn.Module):
    def __init__(self):
        super().__init__()
        if L0 == 0:
            self.in_w, self.in_b = Pm("encoder.input_linear.weight"), Pm("encoder.input_linear.bias")
        ly = []
        for li in range(L0, L1):
            p = f"encoder.layers.{li}."
            m = nn.Module()
            for a in ("ff1", "ff2"):
                for n, t in (("pre_norm", "w"), ("up_proj", "w"), ("down_proj", "w")):
                    setattr(m, f"{a}_{n}_w", Pm(p + f"{a}.{n}.weight"))
                    setattr(m, f"{a}_{n}_b", Pm(p + f"{a}.{n}.bias"))
            m.attn_norm_w, m.attn_norm_b = Pm(p + "attn.pre_norm.weight"), Pm(p + "attn.pre_norm.bias")
            # to_kv and the GLU's up_conv split into two weights each: the
            # phone's compiler rejects last-dim slices of activations.
            m.q_w = Pm(p + "attn.to_q.weight")
            kv = W.pop(p + "attn.to_kv.weight")
            m.k_w = nn.Parameter(kv[:H * DH].contiguous(), requires_grad=False)
            m.v_w = nn.Parameter(kv[H * DH:].contiguous(), requires_grad=False)
            m.o_w, m.o_b = Pm(p + "attn.to_out.weight"), Pm(p + "attn.to_out.bias")
            # Shaw rel-pos as a per-query-row matrix: E[c] = table[DIST[c]]^T [DH, C].
            rel = W.pop(p + "attn.rel_pos_emb.weight")
            m.rel = nn.Parameter(rel[DIST].transpose(1, 2).contiguous(), requires_grad=False)
            m.conv_norm_w, m.conv_norm_b = Pm(p + "conv.norm.weight"), Pm(p + "conv.norm.bias")
            up, ub = W.pop(p + "conv.up_conv.weight")[:, :, 0], W.pop(p + "conv.up_conv.bias")
            ci = up.shape[0] // 2
            m.ua_w = nn.Parameter(up[:ci].contiguous(), requires_grad=False)
            m.ua_b = nn.Parameter(ub[:ci].contiguous(), requires_grad=False)
            m.ug_w = nn.Parameter(up[ci:].contiguous(), requires_grad=False)
            m.ug_b = nn.Parameter(ub[ci:].contiguous(), requires_grad=False)
            m.dw_w = Pm(p + "conv.depth_conv.conv.weight")             # [CI, 1, KS]
            g, b = W.pop(p + "conv.batch_norm.weight"), W.pop(p + "conv.batch_norm.bias")
            mu, var = W.pop(p + "conv.batch_norm.running_mean"), W.pop(p + "conv.batch_norm.running_var")
            W.pop(p + "conv.batch_norm.num_batches_tracked", None)
            sc = g / torch.sqrt(var + EPS)
            m.bn_s = nn.Parameter(sc, requires_grad=False)
            m.bn_t = nn.Parameter(b - mu * sc, requires_grad=False)
            m.down_w = nn.Parameter(W.pop(p + "conv.down_conv.weight")[:, :, 0], requires_grad=False)
            m.down_b = Pm(p + "conv.down_conv.bias")
            m.post_w, m.post_b = Pm(p + "post_norm.weight"), Pm(p + "post_norm.bias")
            ly.append(m)
        self.ly = nn.ModuleList(ly)
        if L0 < MID <= L1:
            self.out_w, self.out_b = Pm("encoder.out.weight"), Pm("encoder.out.bias")
            self.mid_w, self.mid_b = Pm("encoder.out_mid.weight"), Pm("encoder.out_mid.bias")
        if last:
            q = "projector.qformer."
            self.query = nn.Parameter(F.layer_norm(W.pop("projector.query"), (D,), W.pop(q + "layernorm.weight"),
                                                   W.pop(q + "layernorm.bias"), QEPS), requires_grad=False)
            qf = []
            for li in range(QL):
                p = f"{q}encoder.layer.{li}."
                m = nn.Module()
                for a, src in (("s", "attention"), ("c", "crossattention")):
                    for n in ("query", "key", "value"):
                        setattr(m, f"{a}{n[0]}_w", Pm(p + f"{src}.attention.{n}.weight"))
                        setattr(m, f"{a}{n[0]}_b", Pm(p + f"{src}.attention.{n}.bias"))
                    setattr(m, f"{a}o_w", Pm(p + f"{src}.output.dense.weight"))
                    setattr(m, f"{a}o_b", Pm(p + f"{src}.output.dense.bias"))
                    setattr(m, f"{a}n_w", Pm(p + f"{src}.output.LayerNorm.weight"))
                    setattr(m, f"{a}n_b", Pm(p + f"{src}.output.LayerNorm.bias"))
                m.up_w, m.up_b = Pm(p + "intermediate_query.dense.weight"), Pm(p + "intermediate_query.dense.bias")
                m.dn_w, m.dn_b = Pm(p + "output_query.dense.weight"), Pm(p + "output_query.dense.bias")
                m.fn_w, m.fn_b = Pm(p + "output_query.LayerNorm.weight"), Pm(p + "output_query.LayerNorm.bias")
                qf.append(m)
            self.qf = nn.ModuleList(qf)
            self.lin_w, self.lin_b = Pm("projector.linear.weight"), Pm("projector.linear.bias")

    def ff(self, h, m, a):
        y = F.layer_norm(h, (D,), getattr(m, f"{a}_pre_norm_w"), getattr(m, f"{a}_pre_norm_b"), EPS)
        y = F.linear(F.silu(F.linear(y, getattr(m, f"{a}_up_proj_w"), getattr(m, f"{a}_up_proj_b"))),
                     getattr(m, f"{a}_down_proj_w"), getattr(m, f"{a}_down_proj_b"))
        return h + 0.5 * y

    def attn(self, h, m, kbias):
        y = F.layer_norm(h, (D,), m.attn_norm_w, m.attn_norm_b, EPS)
        q = F.linear(y, m.q_w).reshape(NB, C, H, DH).transpose(1, 2)               # [NB, H, C, DH]
        k = F.linear(y, m.k_w).reshape(NB, C, H, DH).transpose(1, 2)
        v = F.linear(y, m.v_w).reshape(NB, C, H, DH).transpose(1, 2)
        # pos[b, h, c, r] = q[b, h, c] . table[DIST[c, r]]: batched over c.
        qc = q.permute(2, 0, 1, 3).reshape(C, NB * H, DH)
        pos = torch.matmul(qc, m.rel).reshape(C, NB, H, C).permute(1, 2, 0, 3)
        s = (torch.matmul(q, k.transpose(2, 3)) + pos) * DH ** -0.5 + kbias
        o = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(1, P, H * DH)
        return h + F.linear(o, m.o_w, m.o_b)

    def conv(self, h, m, fmask):
        y = F.layer_norm(h, (D,), m.conv_norm_w, m.conv_norm_b, EPS)
        y = F.linear(y, m.ua_w, m.ua_b) * torch.sigmoid(F.linear(y, m.ug_w, m.ug_b)) * fmask   # GLU, [1, P, CI]
        y = F.conv1d(F.pad(y.transpose(1, 2), (KS // 2, KS // 2)), m.dw_w, groups=y.shape[-1])
        y = F.silu(y.transpose(1, 2) * m.bn_s + m.bn_t)
        return h + F.linear(y, m.down_w, m.down_b)

    def mha(self, x, kv, m, a, nq, nk):
        """BLIP-2 attention: x [NW|1, nq, D] queries, kv [NW, nk, D]."""
        hd = D // QH
        q = F.linear(x, getattr(m, f"{a}q_w"), getattr(m, f"{a}q_b")).reshape(-1, nq, QH, hd).transpose(1, 2)
        k = F.linear(kv, getattr(m, f"{a}k_w"), getattr(m, f"{a}k_b")).reshape(-1, nk, QH, hd).transpose(1, 2)
        v = F.linear(kv, getattr(m, f"{a}v_w"), getattr(m, f"{a}v_b")).reshape(-1, nk, QH, hd).transpose(1, 2)
        o = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(2, 3)) * hd ** -0.5, -1), v)
        o = o.transpose(1, 2).reshape(-1, nq, D)
        return F.layer_norm(F.linear(o, getattr(m, f"{a}o_w"), getattr(m, f"{a}o_b")) + x, (D,),
                            getattr(m, f"{a}n_w"), getattr(m, f"{a}n_b"), QEPS)

    def forward(self, x, fmask, kbias):
        h = F.linear(x, self.in_w, self.in_b) if L0 == 0 else x
        for li, m in zip(range(L0, L1), self.ly):
            h = self.ff(h, m, "ff1")
            h = self.attn(h, m, kbias)
            h = self.conv(h, m, fmask)
            h = self.ff(h, m, "ff2")
            h = F.layer_norm(h, (D,), m.post_w, m.post_b, EPS)
            if li + 1 == MID:
                h = h + F.linear(torch.softmax(F.linear(h, self.out_w, self.out_b), -1), self.mid_w, self.mid_b)
        if not last:
            return h
        e = (h * fmask)[:, :NW * WIN].reshape(NW, WIN, D)
        qx = self.query.expand(NW, NQ, D)
        for m in self.qf:
            qx = self.mha(qx, qx, m, "s", NQ, NQ)
            qx = self.mha(qx, e, m, "c", NQ, WIN)
            qx = F.layer_norm(F.linear(F.gelu(F.linear(qx, m.up_w, m.up_b)), m.dn_w, m.dn_b) + qx, (D,),
                              m.fn_w, m.fn_b, QEPS)
        return F.linear(qx.reshape(1, NW * NQ, D), self.lin_w, self.lin_b)


if __name__ == "__main__":
    seg = Seg().eval()
    assert not W, f"unused tensors: {sorted(W)[:5]}"
    args = (torch.randn(1, P, IN) if L0 == 0 else torch.randn(1, P, D), torch.ones(1, P, 1), torch.zeros(NB, 1, 1, C))
    ins, outs = tpu_export.export(seg, args, out_path)
    ckey = (f"granite-enc/1 P={P} layers={L0}-{L1} d={D} L={NL} H={H} dh={DH} ctx={C} ks={KS} in={IN} "
            f"win={WIN} ds={DS} qL={QL} out={OUT}")
    print(json.dumps({"key": ckey, "in": ins, "out": outs, "tflite": out_path, "bytes": os.path.getsize(out_path)}))
