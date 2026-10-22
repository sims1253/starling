"""Export the Parakeet-TDT FastConformer encoder of a Starling GGUF as the
TPU artifacts' source graphs (cpp/npu/encoder_contract.hpp, parakeet-enc):
one encoder window of TPU_ROWS (416) subsampled frames, ~33 s of audio, in
five graphs the Pixel's on-device compiler accepts:

  sub      one subsampling chunk: 26 output frames from a 215-row mel window
           (8 * 26 + 7 rows, the 7-row halos included) with VALID convs in
           time; inputs mel [1, 1, 215, n_mels], stage masks m1 [1, 1, 107, 1]
           and m2 [1, 1, 53, 1] (1 for rows inside the clip's stage-1/2
           lengths, 0 for the zero padding the GPU convs read), row mask
           rm [1, 26, 1]; output [1, 26, d]. The engine runs it per chunk.
  l0-6 .. l18-24
           six conformer layers each over all 416 rows, the last one with
           the joint encoder projection; inputs x [4, 104, d], row mask
           rm [4, 104, 1] (1 for the valid frames) and additive key mask
           km [1, 1, 1, 416] (0 / -1e4); output [4, 104, d] (last:
           [4, 104, joint_hidden]).

What the compiler needs, and why the graphs look like this (firmware 0.21,
Pixel 10 Pro): row-wise work in batch layout [4, 104, d] (a LayerNorm or a
linear over [1, 416, 1024] cannot be sharded); the feed-forward up/down
projections split into four 1024-column pieces (a 1024 x 4096 linear does
not fit); attention over all 416 keys in one batched matmul and softmax;
the relative-position term without NeMo's rel_shift (an activation slice):
bd[i, j] = u_i . pe(i - j) with pe(r) = (sin r w_k, cos r w_k) is
sum_k cos(j w_k) (a_k sin i w_k + b_k cos i w_k) + sin(j w_k) (b_k sin i w_k -
a_k cos i w_k), two constant matmuls; the subsampling one 26-frame chunk per
call (batched chunks compile but compute wrong rows). The TPU computes in
fp16: the residual streams run scaled (the subsampling output and layer 0
by 1/256, later layers by 1/16; scales folded into the weights) so the
LayerNorms' squared deviations stay finite, which unscaled they do not
(Parakeet's residuals reach ~7000 after the subsampling and ~1500 later).

Every weight comes from the GGUF (dequantized from its storage type, stored
as fp16), and the same masks reproduce the GPU engine's semantics exactly
(cpp/fast/parakeet_engine.cpp: zero conv padding past each stage's length,
rows past the valid length bias-only, masked as keys, zero attention and
GLU output), so in f32 the five graphs give the GPU path's encoder output.

python -I export_parakeet.py MODEL.gguf OUTDIR [--expect-hash 0xHASH]
    Writes OUTDIR/<piece>.tflite and prints one JSON line per piece:
    {"piece", "key", "in", "out", "model_hash", "gguf", "tflite", "bytes"}.
python -I export_parakeet.py --model-hash MODEL.gguf ...
    Prints "0xHASH PATH" per file; fails for a GGUF that is not Parakeet-TDT.
python -I export_parakeet.py --find 0xHASH DIR_OR_FILE ...
    Prints the first Parakeet-TDT GGUF (searched recursively) with that
    model hash, or exits 1 (hashes cached in ~/.cache/starling-npu).
python -I export_parakeet.py --check MODEL.gguf MEL.f32
    Host check: the five graphs in torch (f32) against a direct port of
    the GPU engine's encoder, on a mel dump ([n_mels, T] f32, mel_dump).
"""
import importlib.util
import json
import math
import os
import sys
import tempfile

_spec = importlib.util.spec_from_file_location(
    "export_encoder", os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_encoder.py"))
ee = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ee)

CONTRACT = "parakeet-enc/1"      # cpp/npu/encoder_contract.hpp kParakeetEncContract
CHUNK = 26                       # subsampled frames per sub call
ROWS = 416                       # frames per encoder window (16 chunks)
ROW_BLOCKS = 4                   # batch layout [4, 104, d]
SEG_LAYERS = 6                   # conformer layers per segment graph
X_SCALE = 1.0 / 256              # subsampling output / layer-0 residual scale
R_SCALE = 1.0 / 16               # residual scale of layers 1..L-1
NEG = -1e4                       # key mask
# LayerNorm epsilon floor in scaled units: fp16's smallest normal number, so a
# zero row (padding) normalizes to finite values. In true units it is
# LN_EPS_MIN / scale^2 (4.0 at layer 0, 0.016 later) where NeMo has 1e-5, against
# residual variances of ~1e5 (layer 0) and ~1e2 (later) in Parakeet: a 1e-4
# relative change at most.
LN_EPS_MIN = 2.0 ** -14


def is_encoder_tensor(name):
    return name.startswith(("encoder.", "joint.enc."))


def is_encoder_key(key):
    return key.startswith("parakeet.encoder.") or key == "parakeet.preprocessor.n_mels"


def encoder_meta(r):
    meta = []
    for key, t in r.kv_type.items():
        if not is_encoder_key(key) or t == ee.Gguf.BOOL:
            continue
        if t in ee.Gguf.INTS:
            meta.append((key, "i", int(r.kv[key])))
        elif t in ee.Gguf.FLOATS:
            meta.append((key, "f", float(r.kv[key])))
        elif t == ee.Gguf.STRING:
            meta.append((key, "s", r.kv[key]))
        else:
            raise ValueError(f"{r.path}: encoder metadata {key} is an array (not covered by the model hash)")
    return meta


def _is_parakeet(r):
    return r.kv.get("general.architecture") == "parakeet_tdt" and "parakeet.encoder.d_model" in r.kv


def model_hash(path, reader=None):
    """The model hash of a Parakeet-TDT GGUF (0x + 16 hex digits); ValueError for any other file."""
    r = reader or ee.Gguf(path)
    if not _is_parakeet(r):
        raise ValueError(f"{path}: not a Parakeet-TDT GGUF")
    ts = [(n, t, ne, nb, ee.xxh64(r.raw(off, nb))) for n, t, ne, off, nb in r.tensors if is_encoder_tensor(n)]
    if not ts:
        raise ValueError(f"{path}: no encoder tensors (encoder.*, joint.enc.*)")
    return f"0x{ee.encoder_digest(ts, encoder_meta(r)):016x}"


def find(want, roots):
    """The first Parakeet-TDT GGUF under roots with model hash want, or None
    (hashes cached like export_encoder.py's, under their own keys)."""
    want = int(want, 16)
    cache = ee._load_cache()
    try:
        for path in ee._ggufs(roots):
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = "parakeet:" + os.path.realpath(path)
            stamp = [st.st_size, st.st_mtime_ns]
            hit = cache.get(key)
            if isinstance(hit, dict) and hit.get("stamp") == stamp and hit.get("v") == ee.DIGEST_VERSION:
                h = hit.get("hash")
            else:
                try:
                    h = model_hash(path)
                except ValueError:
                    h = None
                cache[key] = {"stamp": stamp, "v": ee.DIGEST_VERSION, "hash": h}
            if h is not None and int(h, 16) == want:
                return path
    finally:
        ee._save_cache(cache)
    return None


def stage_len(x):
    return (x - 1) // 2 + 1


def load_gguf(path, expect=None):
    """(config, {name: f32 array in torch layout}, model hash) from one read of the file."""
    import numpy as np
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    r = ee.Gguf(path)
    if not _is_parakeet(r):
        raise SystemExit(f"{path}: not a Parakeet-TDT GGUF (no TPU graph for it)")
    ts = [(n, t, ne, bytes(r.raw(off, nb))) for n, t, ne, off, nb in r.tensors if is_encoder_tensor(n)]
    mh = f"0x{ee.encoder_digest([(n, t, ne, len(b), ee.xxh64(b)) for n, t, ne, b in ts], encoder_meta(r)):016x}"
    if expect is not None and int(mh, 16) != int(expect, 16):
        raise SystemExit(f"{path}: model hash {mh}, not {expect}: not the GGUF the app loads")

    def i(k):
        t = r.kv_type.get(k)
        if t not in ee.Gguf.INTS:
            raise SystemExit(f"{path}: {k} missing or not an integer")
        return int(r.kv[k])

    cfg = dict(D=i("parakeet.encoder.d_model"), L=i("parakeet.encoder.n_layers"), H=i("parakeet.encoder.n_heads"),
               FF=i("parakeet.encoder.ff_dim"), CK=i("parakeet.encoder.conv_kernel"),
               SC=i("parakeet.encoder.subsampling_conv_channels"), NM=i("parakeet.preprocessor.n_mels"))
    if r.kv.get("parakeet.encoder.conv_norm_type") != "batch_norm" or i("parakeet.encoder.xscaling") != 0:
        raise SystemExit(f"{path}: only batch_norm conv modules without xscaling have a TPU graph")
    w = {}
    for n, t, ne, raw in ts:
        rows = ee._prod(ne[1:])
        a = np.frombuffer(raw, np.uint8).reshape(rows, len(raw) // rows)
        w[n] = dequantize(a, GGMLQuantizationType(t)).reshape(tuple(reversed(ne))).astype(np.float32)
    cfg["JH"] = int(w["joint.enc.weight"].shape[0])
    D, H, CK = cfg["D"], cfg["H"], cfg["CK"]
    if D % H or cfg["L"] % SEG_LAYERS or CK % 2 == 0 or cfg["NM"] % 8 or cfg["FF"] % 4:
        raise SystemExit(f"{path}: encoder shape {cfg} has no TPU graph")
    if stage_len(stage_len(stage_len(cfg["NM"]))) * cfg["SC"] != w["encoder.pre_encode.out.weight"].shape[1]:
        raise SystemExit(f"{path}: pre_encode.out does not match the subsampling")
    for li in range(cfg["L"]):
        for b in ("feed_forward1.linear1.bias", "self_attn.linear_q.bias", "self_attn.linear_pos.bias",
                  "conv.pointwise_conv1.bias"):
            if f"encoder.layers.{li}.{b}" in w:
                raise SystemExit(f"{path}: encoder.layers.{li}.{b} is not supported by the TPU graph")
    return cfg, w, mh


def piece_names(cfg):
    return ["sub"] + [f"l{a}-{a + SEG_LAYERS}" for a in range(0, cfg["L"], SEG_LAYERS)]


def piece_key(cfg, piece):
    return (f"{CONTRACT} piece={piece} mels={cfg['NM']} chunk={CHUNK} rows={ROWS} d={cfg['D']} L={cfg['L']} "
            f"H={cfg['H']} ff={cfg['FF']} ck={cfg['CK']} sc={cfg['SC']} jh={cfg['JH']}")


def piece_io(cfg, piece):
    """Byte sizes of a piece's f32 inputs and output (the artifact header's)."""
    D, NM = cfg["D"], cfg["NM"]
    if piece == "sub":
        return [(8 * CHUNK + 7) * NM * 4, (4 * CHUNK + 3) * 4, (2 * CHUNK + 1) * 4, CHUNK * 4], [CHUNK * D * 4]
    last = piece.endswith(f"-{cfg['L']}")
    return [ROWS * D * 4, ROWS * 4, ROWS * 4], [ROWS * (cfg["JH"] if last else D) * 4]


def build(cfg, w, piece):
    """The torch module of one piece, every parameter from the GGUF."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_grad_enabled(False)
    def T(a):
        return torch.from_numpy(np.ascontiguousarray(a))
    D, H, FF, CK, SC = cfg["D"], cfg["H"], cfg["FF"], cfg["CK"], cfg["SC"]
    dk = D // H

    def mk(W, b=None, scale=1.0):
        m = nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
        m.weight.data = (W * scale).contiguous()
        if b is not None:
            m.bias.data = (b * scale).contiguous()
        return m

    if piece == "sub":
        class Sub(nn.Module):
            def __init__(self):
                super().__init__()
                def g(n):
                    return T(w[f"encoder.pre_encode.{n}"])
                for n in ("0", "2", "3", "5"):
                    self.register_buffer(f"w{n}", g(f"conv.{n}.weight"))
                    self.register_buffer(f"b{n}", g(f"conv.{n}.bias"))
                self.register_buffer("w6", g("conv.6.weight") * X_SCALE)
                self.register_buffer("b6", g("conv.6.bias") * X_SCALE)
                self.out = mk(g("out.weight"))
                self.register_buffer("ob", g("out.bias") * X_SCALE)

            def forward(self, mel, m1, m2, rm):
                x = F.relu(F.conv2d(mel, self.w0, self.b0, 2, (0, 1))) * m1
                x = F.conv2d(x, self.w2, self.b2, 2, (0, 1), groups=SC)
                x = F.relu(F.conv2d(x, self.w3, self.b3)) * m2
                x = F.conv2d(x, self.w5, self.b5, 2, (0, 1), groups=SC)
                x = F.relu(F.conv2d(x, self.w6, self.b6))                  # [1, SC, CHUNK, F3]
                x = x.permute(0, 2, 1, 3).reshape(1, CHUNK, -1)
                return self.out(x) * rm + self.ob

        return Sub().eval()

    l0, l1 = (int(v) for v in piece[1:].split("-"))
    L = cfg["L"]
    half = D // 2
    div = np.exp(np.arange(half, dtype=np.float64) * 2 * -(math.log(10000.0) / D))
    ang = np.arange(ROWS, dtype=np.float64)[:, None] * div[None]

    class Seg(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("Sq", T(np.sin(ang).astype(np.float32)))                 # [ROWS, half]
            self.register_buffer("Cq", T(np.cos(ang).astype(np.float32)))
            self.register_buffer("kc", T(np.cos(ang).T.astype(np.float32)).reshape(1, 1, half, ROWS).contiguous())
            self.register_buffer("ks", T(np.sin(ang).T.astype(np.float32)).reshape(1, 1, half, ROWS).contiguous())
            self.layers = nn.ModuleList()
            s = 1.0 / math.sqrt(dk)
            n4 = FF // 4
            for li in range(l0, l1):
                p = f"encoder.layers.{li}."
                def g(n):
                    return T(w[p + n])
                def o(n):
                    return T(w[p + n]) if p + n in w else None
                rs = X_SCALE if li == 0 else R_SCALE
                rs_next = 1.0 if li + 1 == L else R_SCALE
                Y = nn.Module()
                b2 = o("feed_forward1.linear2.bias")
                Y.ff1u = nn.ModuleList(mk(g("feed_forward1.linear1.weight")[k * n4:(k + 1) * n4]) for k in range(4))
                Y.ff1d = nn.ModuleList(mk(g("feed_forward1.linear2.weight")[:, k * n4:(k + 1) * n4],
                                          b2 if k == 0 else None, 0.5 * rs) for k in range(4))
                b1, b2 = o("feed_forward2.linear1.bias"), o("feed_forward2.linear2.bias")
                Y.ff2u = nn.ModuleList(mk(g("feed_forward2.linear1.weight")[k * n4:(k + 1) * n4],
                                          None if b1 is None else b1[k * n4:(k + 1) * n4]) for k in range(4))
                Y.ff2d = nn.ModuleList(mk(g("feed_forward2.linear2.weight")[:, k * n4:(k + 1) * n4],
                                          b2 if k == 0 else None, 0.5 * rs) for k in range(4))
                Y.lns = nn.ModuleList()
                for j, n in enumerate(("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2",
                                       "norm_out")):
                    ln = nn.LayerNorm(D, max(1e-5 * rs * rs, LN_EPS_MIN))
                    f = rs_next if j == 4 else 1.0
                    ln.weight.data, ln.bias.data = g(n + ".weight") * f, g(n + ".bias") * f
                    Y.lns.append(ln)
                bu, bv = g("self_attn.pos_bias_u").reshape(-1), g("self_attn.pos_bias_v").reshape(-1)
                Y.q = mk(g("self_attn.linear_q.weight"), bu, s)               # (q + u) / sqrt(dk)
                Y.register_buffer("dvu", ((bv - bu) * s).reshape(1, H, 1, dk).clone())
                Y.k = mk(g("self_attn.linear_k.weight"), o("self_attn.linear_k.bias"))
                Y.v = mk(g("self_attn.linear_v.weight"), o("self_attn.linear_v.bias"))
                Wp = g("self_attn.linear_pos.weight")                          # [D out, D in]
                Y.register_buffer("pa", torch.stack([Wp[h * dk:(h + 1) * dk, 0::2] for h in range(H)])[None].contiguous())
                Y.register_buffer("pb", torch.stack([Wp[h * dk:(h + 1) * dk, 1::2] for h in range(H)])[None].contiguous())
                Y.o = mk(g("self_attn.linear_out.weight"), o("self_attn.linear_out.bias"), rs)
                pw1 = g("conv.pointwise_conv1.weight").reshape(2 * D, D)
                Y.pw1a, Y.pw1g = mk(pw1[:D]), mk(pw1[D:])
                sc = g("conv.batch_norm.weight") / torch.sqrt(g("conv.batch_norm.running_var") + 1e-5)
                sh = g("conv.batch_norm.bias") - g("conv.batch_norm.running_mean") * sc
                if o("conv.depthwise_conv.bias") is not None:
                    sh = sh + o("conv.depthwise_conv.bias") * sc
                Y.dw = nn.Conv2d(D, D, (CK, 1), padding=(CK // 2, 0), groups=D, bias=True)
                Y.dw.weight.data = (g("conv.depthwise_conv.weight").reshape(D, CK) * sc[:, None]).reshape(D, 1, CK, 1)
                Y.dw.bias.data = sh.clone()
                Y.pw2 = mk(g("conv.pointwise_conv2.weight").reshape(D, D), o("conv.pointwise_conv2.bias"), rs)
                self.layers.append(Y)
            self.jenc = mk(T(w["joint.enc.weight"]), T(w["joint.enc.bias"])) if l1 == L else None

        def heads(self, t):   # [4, 104, D] -> [1, H, ROWS, dk]
            return t.reshape(1, ROWS, H, dk).transpose(1, 2)

        def forward(self, x, rm, km):
            RB = ROWS // ROW_BLOCKS
            for Y in self.layers:
                h = Y.lns[0](x)
                for u, d in zip(Y.ff1u, Y.ff1d):
                    x = x + d(F.silu(u(h)))
                h = Y.lns[1](x)
                qu = self.heads(Y.q(h))
                qv = qu + Y.dvu
                k, v = self.heads(Y.k(h)), self.heads(Y.v(h))
                ac = torch.matmul(qu, k.transpose(2, 3))
                a, b = torch.matmul(qv, Y.pa), torch.matmul(qv, Y.pb)
                al = a * self.Sq + b * self.Cq
                be = b * self.Sq - a * self.Cq
                bd = torch.matmul(al, self.kc) + torch.matmul(be, self.ks)
                pr = torch.softmax(ac + bd + km, -1)
                att = torch.matmul(pr, v).transpose(1, 2).reshape(ROW_BLOCKS, RB, D)
                x = x + Y.o(att * rm)
                h = Y.lns[2](x)
                gl = Y.pw1a(h) * torch.sigmoid(Y.pw1g(h)) * rm
                c = Y.dw(gl.reshape(1, ROWS, D).permute(0, 2, 1)[..., None])
                c = c[..., 0].permute(0, 2, 1).reshape(ROW_BLOCKS, RB, D)
                x = x + Y.pw2(F.silu(c))
                h = Y.lns[3](x)
                for u, d in zip(Y.ff2u, Y.ff2d):
                    x = x + d(F.silu(u(h)))
                x = Y.lns[4](x)
            return self.jenc(x) if self.jenc is not None else x

    return Seg().eval()


# ---- host-side inputs (the engine builds the same: cpp/fast/parakeet_engine.cpp) ----
def lengths(T):
    T1 = stage_len(T)
    T2 = stage_len(T1)
    T3 = stage_len(T2)
    valid = T - 1
    for _ in range(3):
        valid = stage_len(valid)
    return T1, T2, T3, min(valid, T3)


def sub_inputs(mel_tm, k):
    """Inputs of sub for chunk k of a time-major [T, n_mels] mel."""
    import numpy as np
    T, NM = mel_tm.shape
    T1, T2, _, valid = lengths(T)
    g0 = 8 * CHUNK * k - 7
    mel = np.zeros((1, 1, 8 * CHUNK + 7, NM), np.float32)
    lo, hi = max(0, g0), min(T, g0 + 8 * CHUNK + 7)
    if hi > lo:
        mel[0, 0, lo - g0:hi - g0] = mel_tm[lo:hi]
    r1 = 4 * CHUNK * k - 3 + np.arange(4 * CHUNK + 3)
    r2 = 2 * CHUNK * k - 1 + np.arange(2 * CHUNK + 1)
    m1 = ((r1 >= 0) & (r1 < T1)).astype(np.float32).reshape(1, 1, -1, 1)
    m2 = ((r2 >= 0) & (r2 < T2)).astype(np.float32).reshape(1, 1, -1, 1)
    rm = (CHUNK * k + np.arange(CHUNK) < valid).astype(np.float32).reshape(1, CHUNK, 1)
    return mel, m1, m2, rm


def run_torch(cfg, w, mel_tm):
    """The five graphs in torch (f32) as the engine chains them: [T3, JH]."""
    import numpy as np
    import torch
    T = mel_tm.shape[0]
    _, _, T3, valid = lengths(T)
    if T3 > ROWS:
        raise SystemExit(f"{T} mel frames: more than one TPU window ({ROWS} frames)")
    D = cfg["D"]
    x = np.zeros((ROWS, D), np.float32)
    sub = build(cfg, w, "sub")
    for k in range(ROWS // CHUNK):
        if CHUNK * k >= T3:
            break
        x[CHUNK * k:CHUNK * (k + 1)] = sub(*(torch.from_numpy(a) for a in sub_inputs(mel_tm, k)))[0].numpy()
    rm = torch.from_numpy((np.arange(ROWS) < valid).astype(np.float32).reshape(ROW_BLOCKS, -1, 1))
    km = torch.from_numpy(np.where(np.arange(ROWS) < valid, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, ROWS))
    x = torch.from_numpy(x.reshape(ROW_BLOCKS, -1, D))
    for piece in piece_names(cfg)[1:]:
        x = build(cfg, w, piece)(x, rm, km)
    return x.reshape(ROWS, -1)[:T3].numpy()


def reference(cfg, w, mel_tm):
    """A direct port of the GPU engine's encoder (NeMo's rel_shift, unscaled), float64: [T3, JH]."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    dt = torch.float64
    def g(n):
        return torch.from_numpy(w[n]).to(dt)
    def o(n):
        return g(n) if n in w else None
    D, H, SC = cfg["D"], cfg["H"], cfg["SC"]
    dk = D // H
    T = mel_tm.shape[0]
    _, _, T3, valid = lengths(T)
    x = torch.from_numpy(mel_tm).to(dt)[None, None]
    pe_ = "encoder.pre_encode."
    x = F.relu(F.conv2d(x, g(pe_ + "conv.0.weight"), g(pe_ + "conv.0.bias"), 2, 1))
    x = F.conv2d(x, g(pe_ + "conv.2.weight"), g(pe_ + "conv.2.bias"), 2, 1, groups=SC)
    x = F.relu(F.conv2d(x, g(pe_ + "conv.3.weight"), g(pe_ + "conv.3.bias")))
    x = F.conv2d(x, g(pe_ + "conv.5.weight"), g(pe_ + "conv.5.bias"), 2, 1, groups=SC)
    x = F.relu(F.conv2d(x, g(pe_ + "conv.6.weight"), g(pe_ + "conv.6.bias")))
    x = x.permute(0, 2, 1, 3).reshape(T3, -1)
    rm = (torch.arange(T3) < valid).to(dt)[:, None]
    x = F.linear(x, g(pe_ + "out.weight")) * rm + g(pe_ + "out.bias")
    half = D // 2
    div = np.exp(np.arange(half, dtype=np.float64) * 2 * -(math.log(10000.0) / D))
    pos = (T3 - 1) - np.arange(2 * T3 - 1, dtype=np.float64)
    pe = np.zeros((2 * T3 - 1, D))
    pe[:, 0::2], pe[:, 1::2] = np.sin(pos[:, None] * div), np.cos(pos[:, None] * div)
    pe = torch.from_numpy(pe.astype(np.float32)).to(dt)
    km = torch.where(torch.arange(T3) < valid, 0.0, -float("inf")).to(dt)
    idx = torch.arange(T3)[None, :] - torch.arange(T3)[:, None] + T3 - 1
    def ln(t, n):
        return F.layer_norm(t, (D,), g(n + ".weight"), g(n + ".bias"), 1e-5)
    for li in range(cfg["L"]):
        p = f"encoder.layers.{li}."
        def ff(t, k):
            return F.linear(F.silu(F.linear(t, g(p + k + ".linear1.weight"), o(p + k + ".linear1.bias"))),
                                           g(p + k + ".linear2.weight"), o(p + k + ".linear2.bias"))
        x = x + 0.5 * ff(ln(x, p + "norm_feed_forward1"), "feed_forward1")
        h = ln(x, p + "norm_self_att")
        q = F.linear(h, g(p + "self_attn.linear_q.weight"))
        def heads(t):
            return t.reshape(-1, H, dk).transpose(0, 1)
        k = heads(F.linear(h, g(p + "self_attn.linear_k.weight"), o(p + "self_attn.linear_k.bias")))
        v = heads(F.linear(h, g(p + "self_attn.linear_v.weight"), o(p + "self_attn.linear_v.bias")))
        qu = heads(q + g(p + "self_attn.pos_bias_u").reshape(-1))
        qv = heads(q + g(p + "self_attn.pos_bias_v").reshape(-1))
        pp = heads(F.linear(pe, g(p + "self_attn.linear_pos.weight")))
        bd = torch.gather(qv @ pp.transpose(1, 2), 2, idx[None].expand(H, T3, T3))
        a = torch.softmax((qu @ k.transpose(1, 2) + bd) / math.sqrt(dk) + km, -1) @ v
        x = x + F.linear(a.transpose(0, 1).reshape(T3, D) * rm, g(p + "self_attn.linear_out.weight"),
                         o(p + "self_attn.linear_out.bias"))
        h = ln(x, p + "norm_conv")
        c = F.linear(h, g(p + "conv.pointwise_conv1.weight").reshape(2 * D, D))
        c = c[:, :D] * torch.sigmoid(c[:, D:]) * rm
        sc = g(p + "conv.batch_norm.weight") / torch.sqrt(g(p + "conv.batch_norm.running_var") + 1e-5)
        sh = g(p + "conv.batch_norm.bias") - g(p + "conv.batch_norm.running_mean") * sc
        c = F.conv1d(c.t()[None], g(p + "conv.depthwise_conv.weight").reshape(D, 1, -1), o(p + "conv.depthwise_conv.bias"),
                     1, cfg["CK"] // 2, groups=D)[0].t()
        x = x + F.linear(F.silu(c * sc + sh), g(p + "conv.pointwise_conv2.weight").reshape(D, D),
                         o(p + "conv.pointwise_conv2.bias"))
        x = x + 0.5 * ff(ln(x, p + "norm_feed_forward2"), "feed_forward2")
        x = ln(x, p + "norm_out")
    return F.linear(x, g("joint.enc.weight"), g("joint.enc.bias")).float().numpy()


def export(path, outdir, expect=None):
    import numpy as np
    import torch
    import litert_torch
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    cfg, w, mh = load_gguf(path, expect)
    os.makedirs(outdir, exist_ok=True)
    for piece in piece_names(cfg):
        mod = build(cfg, w, piece)
        ins, outs = piece_io(cfg, piece)
        if piece == "sub":
            args = (torch.randn(1, 1, 8 * CHUNK + 7, cfg["NM"]) * 0.5, torch.ones(1, 1, 4 * CHUNK + 3, 1),
                    torch.ones(1, 1, 2 * CHUNK + 1, 1), torch.ones(1, CHUNK, 1))
        else:
            args = (torch.randn(ROW_BLOCKS, ROWS // ROW_BLOCKS, cfg["D"]) * 0.1,
                    torch.ones(ROW_BLOCKS, ROWS // ROW_BLOCKS, 1), torch.zeros(1, 1, 1, ROWS))
        assert [int(np.prod(a.shape)) * 4 for a in args] == ins
        key = piece_key(cfg, piece)
        out_path = os.path.join(outdir, f"{piece}.tflite")
        with tempfile.TemporaryDirectory() as td:
            f32 = os.path.join(td, "f32.tflite")
            litert_torch.convert(mod, args).export(f32)
            del mod
            rm_ = recipe_manager.RecipeManager()
            rm_.add_weight_only_config(".*", qtyping.TFLOperationName.ALL_SUPPORTED, 16,
                                       algorithm_key=AlgorithmName.FLOAT_CASTING)
            src = os.path.join(td, "fp16.tflite")
            quantizer.Quantizer(f32, rm_.get_quantization_recipe()).quantize(None, serialize_to_path=src)
            b = bytearray(open(src, "rb").read())
            k = b"keep_stablehlo_constant"
            assert b.count(k) <= 1, b.count(k)
            b = b.replace(k, b"x" * len(k))
            b = ee.embed_provenance(b, mh, key)
            assert ee.read_provenance(b) == (f"0x{int(mh, 16):016x}", key)
            open(out_path, "wb").write(b)
        print(json.dumps({"piece": piece, "key": key, "in": ins, "out": outs, "model_hash": mh,
                          "gguf": os.path.abspath(path), "tflite": out_path, "bytes": os.path.getsize(out_path)}),
              flush=True)


def check(path, mel_path):
    import numpy as np
    cfg, w, _ = load_gguf(path)
    mel = np.fromfile(mel_path, np.float32).reshape(cfg["NM"], -1).T.copy()
    ref = reference(cfg, w, mel)
    got = run_torch(cfg, w, mel)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    print(json.dumps({"frames": int(ref.shape[0]), "rel_l2": rel, "max_abs": float(np.abs(got - ref).max())}))
    return 0 if rel < 1e-3 else 1


def main(argv):
    a = argv[1:]
    try:
        if a[:1] == ["--model-hash"] and len(a) >= 2:
            for p in a[1:]:
                print(model_hash(p), p)
            return 0
        if a[:1] == ["--check"] and len(a) == 3:
            return check(a[1], a[2])
        if a[:1] == ["--find"] and len(a) >= 3:
            hit = find(a[1], a[2:])
            if hit is None:
                return 1
            print(hit)
            return 0
        if len(a) in (2, 4) and not a[0].startswith("--") and (len(a) == 2 or a[2] == "--expect-hash"):
            export(a[0], a[1], a[3] if len(a) == 4 else None)
            return 0
    except ValueError as e:
        raise SystemExit(str(e)) from None
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
