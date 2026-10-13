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
(the digest cpp/npu computes, below, of the encoder tensors and the encoder
metadata the graph is built from): the artifact is named by the hash of the
file its weights and shapes came from, and the .tflite carries that hash and
the contract key as a `starling_npu_provenance` metadata entry, which the
phone-side compile checks.

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


def is_encoder_key(key):
    return key.startswith(("qwen3.enc.", "qwen3.proj."))


def encoder_digest(tensors, meta=()):
    """tensors: (name, ggml type, ne (up to 4 dims), byte size, xxh64 of the bytes);
    meta: (key, kind, value) of the encoder metadata (encoder_meta).
    FNV-1a 64 over the tensor count (u64), then per tensor in name order: the
    name and a NUL, the type (u32), ne[4] (u64, unused dims 1), the byte size
    and the xxh64 (u64); then the metadata count (u64) and per key in key
    order: the key and a NUL, the kind (b"i", b"f" or b"s") and the value
    (int64; the double's bits; or the length (u64) and the UTF-8 bytes)."""
    h = fnv1a64(struct.pack("<Q", len(tensors)))
    for name, typ, ne, nbytes, data_hash in sorted(tensors, key=lambda t: t[0].encode()):
        ne4 = [int(d) for d in ne] + [1] * (4 - len(ne))
        h = fnv1a64(name.encode() + b"\0" + struct.pack("<I6Q", typ, *ne4, nbytes, data_hash), h)
    h = fnv1a64(struct.pack("<Q", len(meta)), h)
    for key, kind, value in sorted(meta, key=lambda m: m[0].encode()):
        if kind == "i":
            v = struct.pack("<Q", value & M64)
        elif kind == "f":
            v = struct.pack("<d", value)
        else:
            b = value.encode()
            v = struct.pack("<Q", len(b)) + b
        h = fnv1a64(key.encode() + b"\0" + kind.encode() + v, h)
    return h


class Gguf:
    """The parts of a GGUF file the export needs: scalar and string metadata
    (arrays are skipped: the tokenizer's would take seconds to parse) and the
    tensor table, over a read-only mapping of the file."""

    _SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    INTS, FLOATS, BOOL, STRING, ARRAY = (0, 1, 2, 3, 4, 5, 10, 11), (6, 12), 7, 8, 9

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
        self.kv_type = {}
        for _ in range(n_kv):
            key = self._str()
            self.kv_type[key] = self._get("<I")
            self.kv[key] = self._value(self.kv_type[key])
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


def encoder_meta(r):
    """(key, kind, value) of every encoder metadata key, as the engine's
    ModelLoader reads it: integers ("i"), floats ("f", f32 widened exactly to
    a double), strings ("s"); booleans are skipped (ModelLoader does not read
    them); an array is an error (the digest has no form for it)."""
    meta = []
    for key, t in r.kv_type.items():
        if not is_encoder_key(key) or t == Gguf.BOOL:
            continue
        if t in Gguf.INTS:
            meta.append((key, "i", int(r.kv[key])))
        elif t in Gguf.FLOATS:
            meta.append((key, "f", float(r.kv[key])))
        elif t == Gguf.STRING:
            meta.append((key, "s", r.kv[key]))
        else:
            raise ValueError(f"{r.path}: encoder metadata {key} is an array (not covered by the model hash)")
    return meta


def _hash_of(r, tensors):
    """tensors: (name, ggml type, ne, bytes) of every encoder tensor of r."""
    if not tensors:
        raise ValueError(f"{r.path}: no audio-encoder tensors (enc.*, proj.*)")
    return f"0x{encoder_digest([(n, t, ne, len(b), xxh64(b)) for n, t, ne, b in tensors], encoder_meta(r)):016x}"


def model_hash(path, reader=None):
    """The model hash of a Qwen3-ASR GGUF (0x + 16 hex digits)."""
    r = reader or _open(path)
    return _hash_of(r, [(name, typ, ne, r.raw(off, nbytes))
                        for name, typ, ne, off, nbytes in r.tensors if is_encoder_tensor(name)])


def _is_qwen3_asr(r):
    return _field(r, "general.architecture") == "qwen3" and "qwen3.enc.hidden" in r.kv


# ---- finding the GGUF with a given hash, with a cache ----
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "starling-npu", "model-hashes.json")
# Bumped with every change to the digest, so a cache entry of an older one is
# recomputed (2: the encoder metadata is part of it).
DIGEST_VERSION = 2


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
    if isinstance(hit, dict) and hit.get("stamp") == stamp and hit.get("v") == DIGEST_VERSION:
        return hit.get("hash"), hit.get("hd")
    h = hd = None
    try:
        r = _open(path)
        if _is_qwen3_asr(r):
            hd = _field(r, "qwen3.llm.hidden_size")
            h = model_hash(path, r)
    except ValueError:
        pass
    cache[key] = {"stamp": stamp, "v": DIGEST_VERSION, "hash": h, "hd": hd}
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


# ---- provenance stamped into the .tflite (cpp/npu/encoder_contract.hpp) ----
PROVENANCE_NAME = "starling_npu_provenance"
CONTRACT = "qwen3-asr-enc/2"   # cpp/npu/encoder_contract.hpp kQwen3EncContract


def provenance_text(mh, key):
    return f"model_hash=0x{int(mh, 16):016x}\nkey={key}\n".encode()


class _Fb:
    """Bounds-checked reads of a .tflite flatbuffer (TFLite schema field
    numbers: Model.{version 0, operator_codes 1, subgraphs 2, description 3,
    buffers 4, metadata_buffer 5, metadata 6, signature_defs 7},
    Metadata.{name 0, buffer 1}, Buffer.{data 0, offset 1, size 2})."""

    def __init__(self, b):
        self.b = b
        if len(b) < 8 or bytes(b[4:8]) != b"TFL3":
            raise ValueError("not a .tflite")
        self.root = self.u32(0)

    def get(self, fmt, o):
        if o < 0 or o + struct.calcsize(fmt) > len(self.b):
            raise ValueError("malformed .tflite")
        return struct.unpack_from(fmt, self.b, o)[0]

    def u32(self, o):
        return self.get("<I", o)

    def field(self, t, i):   # position of field i of the table at t, or None
        vt = t - self.get("<i", t)
        if 4 + 2 * i >= self.get("<H", vt):
            return None
        o = self.get("<H", vt + 4 + 2 * i)
        return t + o if o else None

    def deref(self, p):
        return p + self.u32(p)

    def tables(self, p):   # positions of the tables of the vector the field at p points to
        if p is None:
            return []
        v = self.deref(p)
        n = self.u32(v)
        if v + 4 + 4 * n > len(self.b):
            raise ValueError("malformed .tflite")
        return [self.deref(v + 4 + 4 * k) for k in range(n)]

    def bytes_at(self, p):   # the bytes of the string or [ubyte] the field at p points to
        v = self.deref(p)
        n = self.u32(v)
        if v + 4 + n > len(self.b):
            raise ValueError("malformed .tflite")
        return bytes(self.b[v + 4:v + 4 + n])


def read_provenance(tfl):
    """(model hash "0x...", contract key) a .tflite is stamped with; None
    without a stamp; ValueError for a malformed or repeated one."""
    import re

    fb = _Fb(tfl)
    bufs = fb.tables(fb.field(fb.root, 4))
    found = []
    for m in fb.tables(fb.field(fb.root, 6)):
        nm = fb.field(m, 0)
        if nm is None or fb.bytes_at(nm) != PROVENANCE_NAME.encode():
            continue
        bi = fb.field(m, 1)
        idx = fb.u32(bi) if bi is not None else 0
        if idx >= len(bufs):
            raise ValueError("the provenance stamp names a missing buffer")
        d = fb.field(bufs[idx], 0)
        found.append(fb.bytes_at(d) if d is not None else b"")
    if not found:
        return None
    if len(found) > 1:
        raise ValueError("more than one provenance stamp")
    m = re.fullmatch(rb"model_hash=(0x[0-9a-f]{16})\nkey=([^\n]*)\n", found[0])
    if not m:
        raise ValueError("malformed provenance stamp")
    return m.group(1).decode(), m.group(2).decode()


def embed_provenance(tfl, mh, key):
    """tfl (a .tflite's bytes) with one more metadata entry, PROVENANCE_NAME,
    whose buffer is provenance_text(mh, key). The original flatbuffer is kept
    byte for byte, moved back by a multiple of 64 bytes (so every alignment in
    it holds); in front of it go a new root table with the original's fields
    (offsets re-pointed) and buffers / metadata vectors extended by the new
    entry. Flatbuffer offsets are relative, so nothing inside the flatbuffer
    changes but the file offsets of buffers stored after it (Buffer.offset,
    as ai-edge-quantizer writes them), which move by the same amount (then a
    multiple of 4096, keeping their alignment)."""
    fb = _Fb(tfl)
    if read_provenance(tfl) is not None:
        raise ValueError("already stamped")
    root = fb.root
    vt = root - fb.get("<i", root)
    if any(fb.field(root, i) for i in range(8, (fb.get("<H", vt) - 4) // 2)):
        raise ValueError("the .tflite has Model fields this stamp does not know")
    pos = {i: fb.field(root, i) for i in range(8)}
    bufs = fb.tables(pos[4])
    mds = fb.tables(pos[6])
    external = []   # positions of Buffer.offset values that are file offsets
    for t in bufs:
        off = fb.field(t, 1)
        if off is not None and fb.get("<Q", off) > 1:
            if fb.get("<Q", off) >= len(tfl):
                raise ValueError("malformed .tflite")
            external.append(off)
    name = PROVENANCE_NAME.encode()
    data = provenance_text(mh, key)

    def al(x, a):
        return (x + a - 1) // a * a

    # Absolute positions of what goes in front; the original's bytes 8.. move to 8 + n.
    a_vt = 8                                     # root vtable: 8 fields
    a_rt = a_vt + 20                             # root table: soffset + 8 four-byte slots
    a_bv = a_rt + 36                             # buffers vector
    a_mv = a_bv + 4 + 4 * (len(bufs) + 1)        # metadata vector
    a_mvt = a_mv + 4 + 4 * (len(mds) + 1)        # Metadata vtable (2 fields)
    a_mt = a_mvt + 8                             # Metadata table: soffset, name, buffer
    a_bvt = a_mt + 12                            # Buffer vtable (1 field, padded)
    a_bt = a_bvt + 8                             # Buffer table: soffset, data
    a_ns = a_bt + 8                              # the name string
    a_dv = al(a_ns + 4 + len(name) + 1 + 4, 16) - 4   # the data vector, its bytes 16-aligned
    end = a_dv + 4 + len(data)
    n = al(end - 8, 4096 if external else 64)
    out = bytearray(8 + n)

    def put(fmt, o, *v):
        struct.pack_into(fmt, out, o, *v)

    put("<I", 0, a_rt)
    out[4:8] = b"TFL3"
    put("<HH", a_vt, 20, 36)
    put("<i", a_rt, a_rt - a_vt)
    for i in range(8):
        slot = a_rt + 4 + 4 * i
        if i == 4:
            put("<I", slot, a_bv - slot)
        elif i == 6:
            put("<I", slot, a_mv - slot)
        elif pos[i] is None:
            continue
        elif i == 0:
            put("<I", slot, fb.u32(pos[0]))       # version, a scalar
        else:
            put("<I", slot, fb.deref(pos[i]) + n - slot)
        put("<H", a_vt + 4 + 2 * i, 4 + 4 * i)
    for vec_at, targets, new in ((a_bv, bufs, a_bt), (a_mv, mds, a_mt)):
        put("<I", vec_at, len(targets) + 1)
        for k, t in enumerate([t + n for t in targets] + [new]):
            e = vec_at + 4 + 4 * k
            put("<I", e, t - e)
    put("<HHHH", a_mvt, 8, 12, 4, 8)
    put("<i", a_mt, a_mt - a_mvt)
    put("<I", a_mt + 4, a_ns - (a_mt + 4))
    put("<I", a_mt + 8, len(bufs))
    put("<HHH", a_bvt, 6, 8, 4)
    put("<i", a_bt, a_bt - a_bvt)
    put("<I", a_bt + 4, a_dv - (a_bt + 4))
    put("<I", a_ns, len(name))
    out[a_ns + 4:a_ns + 4 + len(name)] = name
    put("<I", a_dv, len(data))
    out[a_dv + 4:end] = data
    out += tfl[8:]
    for off in external:
        put("<Q", off + n, fb.get("<Q", off) + n)
    return bytes(out)


# ---- the graph ----
CONFIG_INTS = ("enc.n_mel", "enc.hidden", "enc.layers", "enc.heads", "enc.head_dim", "enc.ffn_dim",
               "enc.downsample_hidden", "enc.n_window", "enc.n_window_infer", "enc.max_pos_emb",
               "enc.output_dim", "proj.hidden", "proj.output_dim")


def encoder_config(r):
    """The encoder settings the graph is built from, read with the engine's
    types (cpp/qwen3/loader.cpp): the engine ignores a key of another GGUF
    type and keeps its default, so such a key would make the graph differ from
    the engine under the same model hash; it is refused instead."""
    def typed(key, kinds, what):
        t = r.kv_type.get(key)
        if t is not None and t not in kinds:
            raise SystemExit(f"{r.path}: {key} is not {what} (GGUF type {t}); the engine would ignore it")
        return r.kv.get(key)
    cfg = {}
    for k in CONFIG_INTS:
        v = typed(f"qwen3.{k}", Gguf.INTS, "an integer")
        if v is None:
            raise SystemExit(f"{r.path}: no qwen3.{k}")
        cfg[k] = int(v)
    eps = typed("qwen3.enc.layer_norm_eps", Gguf.FLOATS, "a float")
    cfg["enc.layer_norm_eps"] = 1e-5 if eps is None else float(eps)   # the engine's default
    return cfg


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
    cfg = encoder_config(r)
    ts = [(name, typ, ne, bytes(r.raw(off, nbytes)))
          for name, typ, ne, off, nbytes in r.tensors if is_encoder_tensor(name)]
    mh = _hash_of(r, ts)
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
        ckey = (f"{CONTRACT} mels={s['NM']} frames={s['FR']} tokens={s['TOK']} d={s['D']} L={s['L']} "
                f"H={s['H']} ff={s['FF']} ds={s['DS']} out={s['OUT']}")
        b = bytearray(open(src, "rb").read())
        key = b"keep_stablehlo_constant"
        assert b.count(key) <= 1, b.count(key)
        b = b.replace(key, b"x" * len(key))
        b = embed_provenance(b, mh, ckey)
        assert read_provenance(b) == (f"0x{int(mh, 16):016x}", ckey)
        open(out_path, "wb").write(b)

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
