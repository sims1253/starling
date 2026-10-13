# PARKED (2026-10-13): the Pixel 10 Pro TPU compiler cannot build these graphs
# (30 s full-length tensors overflow on-chip memory; sequence slices fail or
# exhaust the phone's memory). Kept for reference only, not wired into the
# engine or the compile scripts. See docs/fast-engine.md, "Pixel TPU".
"""Self-attention in a form the Pixel TPU compiler accepts, for the encoder
exports (export_ark_encoder.py).

The phone's compiler rejects batched matmuls whose key length exceeds 256
("Reshape input/output slice mismatch"), last-dim slices of 64-wide heads,
and slices of the head-transposed [1, H, P, hd] tensors ("Does not support
overlap"). So the P positions are cut into n blocks of c <= 256 rows while
still in the [1, P, d] layout, each block is split into heads on its own,
RoPE is a constant [hd, hd] rotate-half matmul, and each query block runs
an exact online softmax over the key blocks it may see.
"""
import torch


def blocks(P, cap=256):
    n = -(-P // cap)
    assert P % n == 0, f"P={P} must split into equal blocks of <= {cap}"
    return n, P // n


def rope_tables(pos, cos, sin):
    """cos/sin [P, hd] -> per-position rows as [1, 1, len, hd] constants."""
    return cos[pos][None, None], sin[pos][None, None]


def attention(q, k, v, H, hd, scale, rope=None, bias=None, mask=None, cap=256):
    """q, k, v [1, P, H*hd] (projected, before RoPE). rope = (cos [P, hd],
    sin [P, hd], rot [hd, hd]) or None. bias: additive key bias [1, 1, 1, P]
    (a graph input) or None. mask(i, j, c): None = block pair fully visible,
    False = invisible (skipped), or an additive [c, c] constant.
    Returns [1, P, H*hd]."""
    P = q.shape[1]
    n, c = blocks(P, cap)

    def heads(t, i, roped):
        t = t[:, i * c:(i + 1) * c].reshape(1, c, H, hd).transpose(1, 2)
        if roped and rope is not None:
            cos, sin, rot = rope
            t = t * cos[i * c:(i + 1) * c][None, None] + torch.matmul(t, rot) * sin[i * c:(i + 1) * c][None, None]
        return t

    ks = [heads(k, j, True) for j in range(n)]
    vs = [heads(v, j, False) for j in range(n)]
    out = []
    for i in range(n):
        qi = heads(q, i, True) * scale
        ss, vv = [], []
        for j in range(n):
            mk = mask(i, j, c) if mask is not None else None
            if mk is False:
                continue
            s = torch.matmul(qi, ks[j].transpose(2, 3))
            if mk is not None:
                s = s + mk[None, None]
            if bias is not None:
                s = s + bias[..., j * c:(j + 1) * c]
            ss.append(s)
            vv.append(vs[j])
        m = ss[0].amax(-1, keepdim=True)
        for s in ss[1:]:
            m = torch.maximum(m, s.amax(-1, keepdim=True))
        es = [torch.exp(s - m) for s in ss]
        den = es[0].sum(-1, keepdim=True)
        o = torch.matmul(es[0], vv[0])
        for e, vj in zip(es[1:], vv[1:]):
            den = den + e.sum(-1, keepdim=True)
            o = o + torch.matmul(e, vj)
        out.append((o / den).transpose(1, 2).reshape(1, c, H * hd))
    return torch.cat(out, 1) if n > 1 else out[0]
