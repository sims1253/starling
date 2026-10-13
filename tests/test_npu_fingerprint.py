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

