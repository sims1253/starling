"""The model hash of scripts/npu/export_encoder.py against the C++ side
(cpp/npu/encoder_contract.hpp): the artifact is named by the hash the export
computes from the GGUF it read, and the engine only uses an artifact whose
hash it computes from the GGUF it loaded, so both must hash the same bytes
the same way. cpp/tests/npu_fingerprint_test.cpp writes the same synthesized
GGUFs with ggml and pins the same constants.
"""

from __future__ import annotations

import importlib.util
import os
import struct
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

EXPORTER = Path(__file__).resolve().parents[1] / "scripts" / "npu" / "export_encoder.py"

# kDigestVector / kDigestTweakedVector / kDigestEpsVector in cpp/tests/npu_fingerprint_test.cpp.
DIGEST_VECTOR = "0x9b4f28654f4e15b6"
DIGEST_TWEAKED_VECTOR = "0x66351613e1266017"
DIGEST_EPS_VECTOR = "0xf237d4837eb57312"


def _exporter():
    spec = importlib.util.spec_from_file_location("export_encoder", EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # numpy/gguf/xxhash at most at import; torch is imported by export()
    return mod


def _fill(name: str, n: int) -> np.ndarray:
    # write_model's byte pattern: byte i of a tensor is (i * 37 + 11 + strlen(name)) & 255.
    return ((np.arange(n, dtype=np.int64) * 37 + 11 + len(name)) & 255).astype(np.uint8)


def _write_model(path: Path, tweak: int) -> None:
    """The GGUF of write_model() in cpp/tests/npu_fingerprint_test.cpp (tweak:
    1 an encoder weight byte, 2 a non-encoder tensor byte, 3 the LayerNorm
    epsilon, 4 a non-encoder key, 5 an array under an encoder key)."""
    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_uint32("qwen3.enc.heads", 14)
    w.add_float32("qwen3.enc.layer_norm_eps", 1e-6 if tweak == 3 else 1e-5)
    w.add_uint32("qwen3.proj.output_dim", 1024)
    w.add_string("qwen3.enc.activation", "gelu")
    w.add_bool("qwen3.enc.flag", True)
    w.add_uint32("qwen3.llm.hidden_size", 2048 if tweak == 4 else 1024)
    if tweak == 5:
        w.add_array("qwen3.enc.windows", [1, 2])
    vectors = ["enc.conv1.bias", "enc.ln_post.weight", "enc.ln_post.bias", "proj.linear_2.bias",
               "enc.blk.0.attn_norm.weight"]
    for k, name in enumerate(vectors):
        w.add_tensor(name, np.array([(16 * k + i) * 0.25 - 1.0 for i in range(4 + k)], np.float32))
    q = _fill("enc.blk.0.attn_q.weight", 68)            # Q8_0, ne [32, 2]: 2 blocks of 34 bytes
    if tweak == 1:
        q[40] ^= 1
    w.add_tensor("enc.blk.0.attn_q.weight", q.reshape(2, 34), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    conv = _fill("enc.conv1.weight", 72)                 # BF16, ne [3, 3, 1, 4]
    w.add_tensor("enc.conv1.weight", conv.reshape(4, 1, 3, 6), raw_dtype=gguf.GGMLQuantizationType.BF16)
    llm = _fill("llm.blk.0.attn_q.weight", 32)          # F32, ne [4, 2]
    if tweak == 2:
        llm[3] ^= 1
    w.add_tensor("llm.blk.0.attn_q.weight", llm.view(np.float32).reshape(2, 4))
    w.add_tensor("audio.mel_filters", _fill("audio.mel_filters", 16).view(np.float32).reshape(2, 2))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_fnv1a64_is_the_standard_hash():
    ee = _exporter()
    assert ee.FNV_BASIS == 0xCBF29CE484222325
    assert ee.fnv1a64(b"") == 0xCBF29CE484222325
    assert ee.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert ee.fnv1a64(b"foobar") == 0x85944171F73967E8


def _xxh_vectors():
    v37 = bytes((i * 7 + 3) & 255 for i in range(37))
    v1032 = bytes(range(256)) * 4 + b"starling"
    # The same reference values as test_vectors() in npu_fingerprint_test.cpp (xxhsum -H1).
    return [(b"", 0xEF46DB3751D8E999), (b"abc", 0x44BC2CF5AD770999), (v37, 0xE32EF63802F5A3FD),
            (v1032, 0x89D35540496386B9)]


def test_xxh64_matches_the_reference_vectors():
    ee = _exporter()
    for data, want in _xxh_vectors():
        assert ee.xxh64_pure(data) == want
        assert ee.xxh64(data) == want  # the xxhash package where installed


def test_model_hash_matches_the_cpp_vectors(tmp_path, monkeypatch):
    ee = _exporter()
    paths = {}
    for tweak in range(6):
        paths[tweak] = tmp_path / f"m{tweak}.gguf"
        _write_model(paths[tweak], tweak)
    for impl in (ee.xxh64, ee.xxh64_pure):
        monkeypatch.setattr(ee, "xxh64", impl)
        assert ee.model_hash(str(paths[0])) == DIGEST_VECTOR
        assert ee.model_hash(str(paths[1])) == DIGEST_TWEAKED_VECTOR   # one weight byte: another hash
        assert ee.model_hash(str(paths[2])) == DIGEST_VECTOR           # outside the encoder: the same
        assert ee.model_hash(str(paths[4])) == DIGEST_VECTOR           # a non-encoder key: the same
    with pytest.raises(ValueError, match="qwen3.enc.windows is an array"):
        ee.model_hash(str(paths[5]))


def test_another_layer_norm_eps_is_another_model(tmp_path):
    """Two GGUFs with identical encoder tensors but another epsilon build
    different graphs, so they must not share an artifact."""
    ee = _exporter()
    base, eps = tmp_path / "base.gguf", tmp_path / "eps.gguf"
    _write_model(base, 0)
    _write_model(eps, 3)
    assert ee.model_hash(str(eps)) == DIGEST_EPS_VECTOR
    assert ee.model_hash(str(eps)) != ee.model_hash(str(base))
    meta = dict((k, (kind, v)) for k, kind, v in ee.encoder_meta(ee.Gguf(str(eps))))
    assert meta == {"qwen3.enc.heads": ("i", 14), "qwen3.enc.layer_norm_eps": ("f", float(np.float32(1e-6))),
                    "qwen3.proj.output_dim": ("i", 1024), "qwen3.enc.activation": ("s", "gelu")}


@pytest.mark.parametrize("add", ["add_uint32", "add_bool", "add_string"])
def test_a_mistyped_encoder_setting_is_refused(tmp_path, add):
    """The engine reads enc.layer_norm_eps only as a float and keeps 1e-5
    otherwise; the exporter must not build the graph with another value."""
    ee = _exporter()
    p = tmp_path / "eps.gguf"
    w = gguf.GGUFWriter(str(p), "qwen3")
    for k in ee.CONFIG_INTS:
        w.add_uint32(f"qwen3.{k}", 4)
    getattr(w, add)("qwen3.enc.layer_norm_eps", {"add_uint32": 1, "add_bool": True, "add_string": "1"}[add])
    w.add_tensor("enc.blk.0.attn_q.weight", np.zeros((2, 4), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(SystemExit, match="layer_norm_eps is not a float"):
        ee.encoder_config(ee.Gguf(str(p)))


def test_encoder_config_reads_the_engine_types(tmp_path):
    ee = _exporter()
    p = tmp_path / "ok.gguf"
    w = gguf.GGUFWriter(str(p), "qwen3")
    for k in ee.CONFIG_INTS:
        w.add_uint32(f"qwen3.{k}", 4)
    w.add_tensor("enc.blk.0.attn_q.weight", np.zeros((2, 4), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    cfg = ee.encoder_config(ee.Gguf(str(p)))
    assert cfg["enc.layer_norm_eps"] == 1e-5 and cfg["enc.hidden"] == 4


def test_model_hash_needs_encoder_tensors(tmp_path):
    ee = _exporter()
    p = tmp_path / "llm-only.gguf"
    w = gguf.GGUFWriter(str(p), "qwen3")
    w.add_tensor("llm.embed.weight", np.zeros((2, 4), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="no audio-encoder tensors"):
        ee.model_hash(str(p))
    with pytest.raises(ValueError, match="not a GGUF"):
        ee.model_hash(str(EXPORTER))


def _write_qwen3_asr(path: Path, hd: int, byte: int = 0) -> None:
    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_uint32("qwen3.enc.hidden", 4)
    w.add_uint32("qwen3.llm.hidden_size", hd)
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])   # arrays are skipped, not parsed
    a = np.arange(8, dtype=np.float32)
    a.view(np.uint8)[0] ^= byte
    w.add_tensor("enc.blk.0.attn_q.weight", a.reshape(2, 4))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_find_picks_the_gguf_with_the_hash_and_caches_it(tmp_path, monkeypatch):
    ee = _exporter()
    monkeypatch.setattr(ee, "CACHE", str(tmp_path / "cache" / "model-hashes.json"))
    models = tmp_path / "models"
    (models / "sub").mkdir(parents=True)
    _write_qwen3_asr(models / "a.gguf", 1024)
    _write_qwen3_asr(models / "sub" / "b.gguf", 2048, byte=1)   # a "fine-tune": one weight byte differs
    (models / "notes.gguf").write_bytes(b"not a gguf")
    ha, hb = ee.model_hash(str(models / "a.gguf")), ee.model_hash(str(models / "sub" / "b.gguf"))
    assert ha != hb
    assert ee.find(hb, [str(models)]) == str(models / "sub" / "b.gguf")
    assert ee.find(ha, [str(models)], hd=1024) == str(models / "a.gguf")
    assert ee.find(ha, [str(models)], hd=2048) is None
    assert ee.find("0x1", [str(models)]) is None
    assert os.path.exists(ee.CACHE)

    # Cached: found again without rehashing ...
    def no_hashing(*a, **k):
        raise AssertionError("rehashed a cached file")

    monkeypatch.setattr(ee, "model_hash", no_hashing)
    assert ee.find(hb, [str(models)]) == str(models / "sub" / "b.gguf")
    # ... until the file changes (size or mtime).
    _write_qwen3_asr(models / "sub" / "b.gguf", 2048, byte=2)
    with pytest.raises(AssertionError, match="rehashed"):
        ee.find(hb, [str(models)])


def test_every_encoder_tensor_maps_to_one_parameter():
    ee = _exporter()
    names = ["enc.conv1.weight", "enc.conv1.bias", "enc.conv2.weight", "enc.conv2.bias", "enc.conv3.weight",
             "enc.conv3.bias", "enc.out.weight", "enc.ln_post.weight", "enc.ln_post.bias",
             "proj.linear_1.weight", "proj.linear_1.bias", "proj.linear_2.weight", "proj.linear_2.bias"]
    for part in ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_norm", "ff_up", "ff_down"):
        names += [f"enc.blk.17.{part}.weight", f"enc.blk.17.{part}.bias"]
    mapped = [ee.module_name(n) for n in names]
    assert None not in mapped and len(set(mapped)) == len(mapped)
    assert ee.module_name("enc.blk.0.attn_q.weight") == "enc.layers.0.self_attn.q_proj.weight"
    assert ee.module_name("enc.out.weight") == "enc.conv_out.weight"
    assert ee.module_name("enc.blk.0.unknown.weight") is None


def test_encoder_digest_layout():
    ee = _exporter()
    # FNV-1a over the tensor count, per tensor name + NUL, type u32, ne[4] u64,
    # size u64, xxh64 u64; then the metadata count, per key key + NUL, kind and value.
    body = struct.pack("<Q", 1) + b"enc.x\0" + struct.pack("<I6Q", 8, 32, 2, 1, 1, 68, 0x1234)
    assert ee.encoder_digest([("enc.x", 8, [32, 2], 68, 0x1234)]) == ee.fnv1a64(body + struct.pack("<Q", 0))
    meta = [("qwen3.enc.s", "s", "ab"), ("qwen3.enc.f", "f", 0.5), ("qwen3.enc.i", "i", -1)]
    tail = (struct.pack("<Q", 3) + b"qwen3.enc.f\0f" + struct.pack("<d", 0.5) + b"qwen3.enc.i\0i"
            + struct.pack("<q", -1) + b"qwen3.enc.s\0s" + struct.pack("<Q", 2) + b"ab")
    assert ee.encoder_digest([("enc.x", 8, [32, 2], 68, 0x1234)], meta) == ee.fnv1a64(body + tail)


# ---- the provenance stamp (cpp/tests/npu_provenance_test.cpp pins the same bytes) ----
def _minimal_tflite() -> bytes:
    """A minimal TFLite flatbuffer: Model.version 3, buffers [empty, "1.14.0"],
    metadata [min_runtime_version -> buffer 1] (the bytes of kBaseTflite in
    npu_provenance_test.cpp)."""
    b = bytearray(152)

    def put(fmt, o, *v):
        struct.pack_into(fmt, b, o, *v)

    put("<I", 0, 28)
    b[4:8] = b"TFL3"
    put("<9H", 8, 18, 16, 4, 0, 0, 0, 8, 0, 12)     # Model vtable: version, buffers, metadata
    put("<i", 28, 20)
    put("<I", 32, 3)
    put("<I", 36, 44 - 36)
    put("<I", 40, 56 - 40)
    put("<I", 44, 2)
    put("<I", 48, 68 - 48)
    put("<I", 52, 80 - 52)
    put("<I", 56, 1)
    put("<I", 60, 96 - 60)
    put("<HH", 64, 4, 4)                            # empty Buffer
    put("<i", 68, 68 - 64)
    put("<HHH", 72, 6, 8, 4)                        # Buffer with data
    put("<i", 80, 80 - 72)
    put("<I", 84, 140 - 84)
    put("<HHHH", 88, 8, 12, 4, 8)                   # Metadata
    put("<i", 96, 96 - 88)
    put("<I", 100, 108 - 100)
    put("<I", 104, 1)
    put("<I", 108, 19)
    b[112:131] = b"min_runtime_version"
    put("<I", 140, 6)
    b[144:150] = b"1.14.0"
    return bytes(b)


PROVENANCE_KEY = "qwen3-asr-enc/2 test"
PROVENANCE_HASH = "0x0123456789abcdef"
# embed_provenance(_minimal_tflite(), PROVENANCE_HASH, PROVENANCE_KEY): kStampedTflite in npu_provenance_test.cpp.
STAMPED_SHA = "2a328bdd6196a8ad4b4d82d95c858000eda497db5489d72ffaa0fef383474740"


def test_provenance_round_trips_and_keeps_the_original():
    import hashlib

    ee = _exporter()
    base = _minimal_tflite()
    assert ee.read_provenance(base) is None
    stamped = ee.embed_provenance(base, PROVENANCE_HASH, PROVENANCE_KEY)
    assert hashlib.sha256(stamped).hexdigest() == STAMPED_SHA
    assert ee.read_provenance(stamped) == (PROVENANCE_HASH, PROVENANCE_KEY)
    # The original flatbuffer follows, byte for byte, moved by a multiple of 64.
    shift = len(stamped) - len(base)
    assert shift % 64 == 0 and stamped[8 + shift:] == base[8:]
    # Every original field is still there: version 3, its buffers and its metadata entry.
    fb = ee._Fb(stamped)
    assert fb.u32(fb.field(fb.root, 0)) == 3
    bufs = fb.tables(fb.field(fb.root, 4))
    mds = fb.tables(fb.field(fb.root, 6))
    assert len(bufs) == 3 and len(mds) == 2
    assert fb.bytes_at(fb.field(mds[0], 0)) == b"min_runtime_version"
    assert fb.bytes_at(fb.field(bufs[fb.u32(fb.field(mds[0], 1))], 0)) == b"1.14.0"
    assert fb.field(bufs[0], 0) is None
    # The stamp's bytes are 16-aligned, like every TFLite buffer.
    d = fb.field(bufs[2], 0)
    assert (fb.deref(d) + 4) % 16 == 0
    with pytest.raises(ValueError, match="already stamped"):
        ee.embed_provenance(stamped, PROVENANCE_HASH, PROVENANCE_KEY)


def test_provenance_moves_buffers_stored_after_the_flatbuffer():
    """ai-edge-quantizer writes weights after the flatbuffer (Buffer.offset /
    size are file offsets): the stamp moves them with the file."""
    ee = _exporter()
    b = bytearray(4096 + 16)

    def put(fmt, o, *v):
        struct.pack_into(fmt, b, o, *v)

    put("<I", 0, 24)
    b[4:8] = b"TFL3"
    put("<7H", 8, 14, 12, 4, 0, 0, 0, 8)            # Model vtable: version, buffers
    put("<i", 24, 24 - 8)
    put("<I", 28, 3)
    put("<I", 32, 36 - 32)
    put("<I", 36, 1)
    put("<I", 40, 56 - 40)
    put("<5H", 44, 10, 24, 0, 8, 16)                # Buffer vtable: offset, size
    put("<i", 56, 56 - 44)
    put("<QQ", 64, 4096, 8)
    b[4096:4104] = b"EXTERNAL"
    stamped = ee.embed_provenance(bytes(b), PROVENANCE_HASH, PROVENANCE_KEY)
    assert ee.read_provenance(stamped) == (PROVENANCE_HASH, PROVENANCE_KEY)
    fb = ee._Fb(stamped)
    buf = fb.tables(fb.field(fb.root, 4))[0]
    off, size = fb.get("<Q", fb.field(buf, 1)), fb.get("<Q", fb.field(buf, 2))
    assert off % 4096 == 0 and stamped[off:off + size] == b"EXTERNAL"
    with pytest.raises(ValueError, match="already stamped"):
        ee.embed_provenance(stamped, PROVENANCE_HASH, PROVENANCE_KEY)
    with pytest.raises(ValueError, match="not a .tflite"):
        ee.read_provenance(b"GGUF" * 4)
