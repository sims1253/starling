"""Export the Qwen3-ASR audio encoder (+ projector) of a Starling GGUF as the
TPU artifact's source graph: one attention window of n_window_infer mel
frames, inputs mel [1, n_mels, frames] f32 and key mask [1, 1, 1, tokens] f32
(0 valid / -1e4 padding), output [1, tokens, output_dim] f32 — the contract in
cpp/npu/encoder_contract.hpp.

The graph is built from the GGUF the app loads, nothing else: its metadata
gives the shapes and its own enc.* / proj.* tensors, dequantized from
whatever they are stored as (Q4_0, Q8_0, K-quants, BF16, F16, F32), give the
weights, so the TPU computes with the GPU engine's encoder weights. Weights
are stored as fp16 (the TPU computes in fp16 either way); the converter's
`keep_stablehlo_constant` metadata key, which the phone's on-device compiler
rejects, is renamed in place. The export also computes the GGUF's model hash
(the digest cpp/npu computes, below): the artifact is named by the hash of
the file its weights came from.

python -I export_encoder.py MODEL.gguf OUT.tflite [--expect-hash 0xHASH]
    Prints one JSON line: {"key", "in", "out", "model_hash", "gguf", "tflite",
    "bytes"}. With --expect-hash it refuses, before exporting, a GGUF whose
    model hash is another.
python -I export_encoder.py --model-hash MODEL.gguf ...
    Prints "0xHASH PATH" per file.
python -I export_encoder.py --find 0xHASH [--hd N] DIR_OR_FILE ...
    Prints the first Qwen3-ASR GGUF (searched recursively; --hd: only those
    whose LLM hidden size is N) with that model hash, or exits 1. Hashes are
    cached by (path, size, mtime) in ~/.cache/starling-npu/model-hashes.json.

Needs the gguf package; torch, litert-torch and ai-edge-quantizer for the
export. The xxhash package, where installed, only speeds the hashing up.
"""
import json
import os
import struct
import sys
import tempfile

# ---- the model hash (cpp/npu/encoder_contract.hpp, qwen3_encoder_digest) ----
FNV_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
M64 = 0xFFFFFFFFFFFFFFFF
XP1, XP2, XP3, XP4, XP5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                           0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5)


def fnv1a64(data, h=FNV_BASIS):
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & M64
    return h


def xxh64_pure(data, seed=0):
    """XXH64 in plain Python (the same function as cpp/npu's xxh64)."""
    data = bytes(data)
    n = len(data)
    p = 0

    def rotl(x, r):
        return ((x << r) | (x >> (64 - r))) & M64

    def rnd(acc, w):
        acc = (acc + w * XP2) & M64
        return (((acc << 31) | (acc >> 33)) & M64) * XP1 & M64

    if n >= 32:
        v1, v2, v3, v4 = (seed + XP1 + XP2) & M64, (seed + XP2) & M64, seed, (seed - XP1) & M64
        stripes = n // 32
        words = struct.unpack_from(f"<{4 * stripes}Q", data)
        for i in range(0, 4 * stripes, 4):
            v1 = rnd(v1, words[i])
            v2 = rnd(v2, words[i + 1])
            v3 = rnd(v3, words[i + 2])
            v4 = rnd(v4, words[i + 3])
        p = 32 * stripes
        h = (rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18)) & M64
        for v in (v1, v2, v3, v4):
            h = ((h ^ rnd(0, v)) * XP1 + XP4) & M64
    else:
        h = (seed + XP5) & M64
    h = (h + n) & M64
    while n - p >= 8:
        h = (rotl(h ^ rnd(0, struct.unpack_from("<Q", data, p)[0]), 27) * XP1 + XP4) & M64
        p += 8
    if n - p >= 4:
        h = (rotl(h ^ (struct.unpack_from("<I", data, p)[0] * XP1 & M64), 23) * XP2 + XP3) & M64
        p += 4
    while p < n:
        h = (rotl(h ^ (data[p] * XP5 & M64), 11) * XP1) & M64
        p += 1
    h ^= h >> 33
    h = h * XP2 & M64
    h ^= h >> 29
    h = h * XP3 & M64
    h ^= h >> 32
    return h


try:
    import xxhash

    def xxh64(data, seed=0):
        return xxhash.xxh64_intdigest(data, seed)
except ImportError:
    xxh64 = xxh64_pure


def _prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


def is_encoder_tensor(name):
    return name.startswith(("enc.", "proj."))


def encoder_digest(tensors):
    """tensors: (name, ggml type, ne (up to 4 dims), byte size, xxh64 of the bytes).
    FNV-1a 64 over, per tensor in name order: the name and a NUL, the type
    (u32), ne[4] (u64, unused dims 1), the byte size and the xxh64 (u64)."""
    h = FNV_BASIS
    for name, typ, ne, nbytes, data_hash in sorted(tensors, key=lambda t: t[0].encode()):
        ne4 = [int(d) for d in ne] + [1] * (4 - len(ne))
        h = fnv1a64(name.encode() + b"\0" + struct.pack("<I6Q", typ, *ne4, nbytes, data_hash), h)
    return h


class Gguf:
    """The parts of a GGUF file the export needs: scalar and string metadata
    (arrays are skipped: the tokenizer's would take seconds to parse) and the
    tensor table, over a read-only mapping of the file."""

    _SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def __init__(self, path):
        import mmap

        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

        self.path = path
        try:
            with open(path, "rb") as f:
                self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as e:
            raise ValueError(f"{path}: cannot read ({e})") from None
        self.pos = 0
        if self._get("<4s") != b"GGUF" or self._get("<I") not in (2, 3):
            raise ValueError(f"{path}: not a GGUF v2/v3 file")
        n_tensors, n_kv = self._get("<Q"), self._get("<Q")
        self.kv = {}
        for _ in range(n_kv):
            key = self._str()
            self.kv[key] = self._value(self._get("<I"))
        self.tensors = []   # (name, ggml type, ne, offset in the file, byte size)
        infos = []
        for _ in range(n_tensors):
            name = self._str()
            ne = [self._get("<Q") for _ in range(self._get("<I"))]
            infos.append((name, ne, self._get("<I"), self._get("<Q")))
        align = int(self.kv.get("general.alignment", 32))
        data0 = (self.pos + align - 1) // align * align
        for name, ne, typ, off in infos:
            try:
                block, size = GGML_QUANT_SIZES[GGMLQuantizationType(typ)]
            except (ValueError, KeyError):
                raise ValueError(f"{path}: {name} has unknown ggml type {typ}") from None
            if not ne or ne[0] % block:
                raise ValueError(f"{path}: {name} shape {ne} does not fit its type")
            nbytes = _prod(ne) // block * size
            if data0 + off + nbytes > len(self.mm):
                raise ValueError(f"{path}: {name} runs past the end of the file")
            self.tensors.append((name, typ, ne, data0 + off, nbytes))

    def _get(self, fmt):
        try:
            v = struct.unpack_from(fmt, self.mm, self.pos)[0]
        except struct.error:
            raise ValueError(f"{self.path}: truncated GGUF header") from None
        self.pos += struct.calcsize(fmt)
        return v

    def _str(self):
        n = self._get("<Q")
        if self.pos + n > len(self.mm):
            raise ValueError(f"{self.path}: truncated GGUF header")
        self.pos += n
        return bytes(self.mm[self.pos - n:self.pos]).decode("utf-8", "replace")

    def _value(self, t):
        if t in self._SCALAR:
            return self._get(self._SCALAR[t])
        if t == 8:
            return self._str()
        if t == 9:   # array: skipped
            et, n = self._get("<I"), self._get("<Q")
            if et in self._SCALAR:
                self.pos += n * struct.calcsize(self._SCALAR[et])
            else:
                for _ in range(n):
                    self._value(et)
            return None
        raise ValueError(f"{self.path}: unknown GGUF value type {t}")

    def raw(self, off, nbytes):
        return memoryview(self.mm)[off:off + nbytes]


def _open(path):
    return Gguf(path)


def _field(r, key, default=None):
    v = r.kv.get(key)
    return default if v is None else v


def _hash_of(path, tensors):
    """tensors: (name, ggml type, ne, bytes) of every encoder tensor."""
    if not tensors:
        raise ValueError(f"{path}: no audio-encoder tensors (enc.*, proj.*)")
    return f"0x{encoder_digest([(n, t, ne, len(b), xxh64(b)) for n, t, ne, b in tensors]):016x}"


def model_hash(path, reader=None):
    """The model hash of a Qwen3-ASR GGUF (0x + 16 hex digits)."""
    r = reader or _open(path)
    return _hash_of(path, [(name, typ, ne, r.raw(off, nbytes))
                           for name, typ, ne, off, nbytes in r.tensors if is_encoder_tensor(name)])


def _is_qwen3_asr(r):
    return _field(r, "general.architecture") == "qwen3" and "qwen3.enc.hidden" in r.kv


# ---- finding the GGUF with a given hash, with a cache ----
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "starling-npu", "model-hashes.json")


def _load_cache():
    try:
        with open(CACHE) as f:
            c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(c):
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        tmp = f"{CACHE}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(c, f, indent=0, sort_keys=True)
        os.replace(tmp, CACHE)
    except OSError:
        pass  # a cache only


def _describe(path, cache):
    """(model hash or None, LLM hidden size or None) of path, cached by (size, mtime)."""
    try:
        st = os.stat(path)
    except OSError:   # a dangling link
        return None, None
    key = os.path.realpath(path)
    stamp = [st.st_size, st.st_mtime_ns]
    hit = cache.get(key)
    if isinstance(hit, dict) and hit.get("stamp") == stamp:
        return hit.get("hash"), hit.get("hd")
    h = hd = None
    try:
        r = _open(path)
        if _is_qwen3_asr(r):
            hd = _field(r, "qwen3.llm.hidden_size")
            h = model_hash(path, r)
    except ValueError:
        pass
    cache[key] = {"stamp": stamp, "hash": h, "hd": hd}
    return h, hd


def _ggufs(roots):
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for d, dirs, files in os.walk(root):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(".gguf"):
                    yield os.path.join(d, f)


def find(want, roots, hd=None):
    want = int(want, 16)
    cache = _load_cache()
    try:
        for path in _ggufs(roots):
            h, ghd = _describe(path, cache)
            if h is not None and int(h, 16) == want and (hd is None or ghd == hd):
                return path
    finally:
        _save_cache(cache)
    return None


# ---- the graph ----
def load_gguf(path, expect=None):
    """(config, {GGUF tensor name: f32 numpy array in torch layout}, model hash).
    The weights and the hash come from the same copy of the tensors' bytes;
    with expect (0x...), a GGUF with another hash is refused first."""
    import numpy as np
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    r = _open(path)
    if not _is_qwen3_asr(r):
        raise SystemExit(f"{path}: not a Qwen3-ASR GGUF (no TPU graph for it)")
    cfg = {k: int(_field(r, f"qwen3.{k}")) for k in (
        "enc.n_mel", "enc.hidden", "enc.layers", "enc.heads", "enc.head_dim", "enc.ffn_dim",
        "enc.downsample_hidden", "enc.n_window", "enc.n_window_infer", "enc.max_pos_emb",
        "enc.output_dim", "proj.hidden", "proj.output_dim")}
    cfg["enc.layer_norm_eps"] = float(_field(r, "qwen3.enc.layer_norm_eps", 1e-5))
    ts = [(name, typ, ne, bytes(r.raw(off, nbytes)))
          for name, typ, ne, off, nbytes in r.tensors if is_encoder_tensor(name)]
    mh = _hash_of(path, ts)
    if expect is not None and int(mh, 16) != int(expect, 16):
        raise SystemExit(f"{path}: model hash {mh}, not {expect}: not the GGUF the app loads")
    w = {}
    for name, typ, ne, raw in ts:
        # ne is innermost-first: the reversed shape is torch's layout
        # (Linear [out, in], Conv2d [out, in, kh, kw]).
        rows = _prod(ne[1:])
        a = np.frombuffer(raw, np.uint8).reshape(rows, len(raw) // rows)
        w[name] = dequantize(a, GGMLQuantizationType(typ)).reshape(tuple(reversed(ne))).astype(np.float32)
    return cfg, w, mh


# GGUF name (scripts/convert_qwen3_gguf.py) -> parameter of the modules below,
# which keep the HF Qwen3-ASR audio tower's structure and names.
def module_name(g):
    top = {"enc.conv1.": "enc.conv2d1.", "enc.conv2.": "enc.conv2d2.", "enc.conv3.": "enc.conv2d3.",
           "enc.out.": "enc.conv_out.", "enc.ln_post.": "enc.ln_post.", "proj.linear_1.": "proj.linear_1.",
           "proj.linear_2.": "proj.linear_2."}
    for a, b in top.items():
        if g.startswith(a):
            return b + g[len(a):]
    if g.startswith("enc.blk."):
        i, rest = g[len("enc.blk."):].split(".", 1)
        part, leaf = rest.rsplit(".", 1)
        sub = {"attn_norm": "self_attn_layer_norm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
               "attn_v": "self_attn.v_proj", "attn_o": "self_attn.out_proj", "ffn_norm": "final_layer_norm",
               "ff_up": "fc1", "ff_down": "fc2"}.get(part)
        if sub:
            return f"enc.layers.{i}.{sub}.{leaf}"
    return None


def build_window(cfg, w):
    """The encoder window as a torch module with the GGUF's weights loaded."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_grad_enabled(False)
    D, L, H, FF = cfg["enc.hidden"], cfg["enc.layers"], cfg["enc.heads"], cfg["enc.ffn_dim"]
    NM, DS, eps = cfg["enc.n_mel"], cfg["enc.downsample_hidden"], cfg["enc.layer_norm_eps"]
    PH, OUT = cfg["proj.hidden"], cfg["proj.output_dim"]
    assert cfg["enc.head_dim"] * H == D and PH == D and OUT == cfg["enc.output_dim"], cfg

    def half(x):
        return (x - 1) // 2 + 1

    FR = cfg["enc.n_window_infer"]               # mel frames per attention window
    CH = cfg["enc.n_window"] * 2                 # frames per conv chunk (100)
    NCH = FR // CH                               # chunks per window
    TPC = half(half(half(CH)))                   # tokens per chunk (13)
    TOK = TPC * NCH                              # tokens per window
    FQ = half(half(half(NM)))                    # frequency bins after the convs (16)
    assert FR % CH == 0 and TPC <= cfg["enc.max_pos_emb"], cfg

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj = nn.Linear(D, D), nn.Linear(D, D)
            self.v_proj, self.out_proj = nn.Linear(D, D), nn.Linear(D, D)
            self.scaling = (D // H) ** -0.5

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn_layer_norm = nn.LayerNorm(D, eps)
            self.self_attn = Attention()
            self.final_layer_norm = nn.LayerNorm(D, eps)
            self.fc1, self.fc2 = nn.Linear(D, FF), nn.Linear(FF, D)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2d1 = nn.Conv2d(1, DS, 3, 2, padding=1)
            self.conv2d2 = nn.Conv2d(DS, DS, 3, 2, padding=1)
            self.conv2d3 = nn.Conv2d(DS, DS, 3, 2, padding=1)
            self.conv_out = nn.Linear(DS * FQ, D, bias=False)
            self.layers = nn.ModuleList(Layer() for _ in range(L))
            self.ln_post = nn.LayerNorm(D, eps)

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1, self.linear_2 = nn.Linear(D, PH), nn.Linear(PH, OUT)

    class Window(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.proj = Encoder(), Projector()
            self.register_buffer("pos", torch.zeros(TPC, D))

        def forward(self, mel, kbias):
            e = self.enc
            x = mel.reshape(NM, NCH, CH).permute(1, 0, 2).reshape(NCH, 1, NM, CH)
            x = F.gelu(e.conv2d1(x))
            x = F.gelu(e.conv2d2(x))
            x = F.gelu(e.conv2d3(x))
            x = x.permute(0, 3, 1, 2).reshape(NCH, TPC, DS * FQ)
            h = (e.conv_out(x) + self.pos).reshape(1, TOK, D)
            for ly in e.layers:
                a = ly.self_attn
                r = h
                y = ly.self_attn_layer_norm(h)
                q = a.q_proj(y).reshape(1, TOK, H, D // H).transpose(1, 2)
                k = a.k_proj(y).reshape(1, TOK, H, D // H).transpose(1, 2)
                v = a.v_proj(y).reshape(1, TOK, H, D // H).transpose(1, 2)
                s = torch.matmul(q * a.scaling, k.transpose(2, 3)) + kbias
                o = torch.matmul(torch.softmax(s, dim=-1), v).transpose(1, 2).reshape(1, TOK, D)
                h = r + a.out_proj(o)
                h = h + ly.fc2(F.gelu(ly.fc1(ly.final_layer_norm(h))))
            h = e.ln_post(h)
            return self.proj.linear_2(F.gelu(self.proj.linear_1(h)))

    win = Window().eval()
    sd = {}
    for g, a in w.items():
        if g == "enc.pos_embed":
            sd["pos"] = torch.from_numpy(a[:TPC].copy())
            continue
        m = module_name(g)
        if m is None:
            raise SystemExit(f"unknown encoder tensor {g}: this exporter does not know the model")
        sd[m] = torch.from_numpy(a)
    if "pos" not in sd:
        raise SystemExit("no enc.pos_embed")
    win.load_state_dict(sd, strict=True)   # every parameter from the GGUF, every GGUF tensor used
    shapes = dict(NM=NM, FR=FR, TOK=TOK, D=D, L=L, H=H, FF=FF, DS=DS, OUT=OUT)
    return win, shapes


def export(path, out_path, expect=None):
    cfg, w, mh = load_gguf(path, expect)
    win, s = build_window(cfg, w)
    del w
    import torch
    import litert_torch
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    mel = torch.randn(1, s["NM"], s["FR"]) * 0.5
    kb = torch.zeros(1, 1, 1, s["TOK"])
    with tempfile.TemporaryDirectory() as td:
        f32 = os.path.join(td, "f32.tflite")
        # Big encoders (1.7B: ~0.3 B parameters) only convert within 8 GB with
        # lightweight_conversion; the fp16 weight casting below cannot read that
        # output, so those are compiled from f32 weights (the TPU computes in
        # fp16 either way and stores fp16 bytecode; only the .tflite is bigger).
        light = sum(p.numel() for p in win.parameters()) > 200_000_000
        litert_torch.convert(win, (mel, kb), lightweight_conversion=light).export(f32)
        del win
        src = f32
        if not light:
            rm = recipe_manager.RecipeManager()
            rm.add_weight_only_config(".*", qtyping.TFLOperationName.ALL_SUPPORTED, 16,
                                      algorithm_key=AlgorithmName.FLOAT_CASTING)
            src = os.path.join(td, "fp16.tflite")
            quantizer.Quantizer(f32, rm.get_quantization_recipe()).quantize(None, serialize_to_path=src)
        b = bytearray(open(src, "rb").read())
        key = b"keep_stablehlo_constant"
        assert b.count(key) <= 1, b.count(key)
        b = b.replace(key, b"x" * len(key))
        open(out_path, "wb").write(b)

    ckey = (f"qwen3-asr-enc/1 mels={s['NM']} frames={s['FR']} tokens={s['TOK']} d={s['D']} L={s['L']} "
            f"H={s['H']} ff={s['FF']} ds={s['DS']} out={s['OUT']}")
    print(json.dumps({"key": ckey, "in": [s["NM"] * s["FR"] * 4, s["TOK"] * 4], "out": [s["TOK"] * s["OUT"] * 4],
                      "model_hash": mh, "gguf": os.path.abspath(path), "tflite": out_path,
                      "bytes": os.path.getsize(out_path)}))


def main(argv):
    a = argv[1:]
    try:
        if a[:1] == ["--model-hash"] and len(a) >= 2:
            for p in a[1:]:
                print(model_hash(p), p)
            return 0
        if a[:1] == ["--find"] and len(a) >= 3:
            want, rest, hd = a[1], a[2:], None
            if rest[0] == "--hd" and len(rest) >= 3:
                hd, rest = int(rest[1]), rest[2:]
            hit = find(want, rest, hd)
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
