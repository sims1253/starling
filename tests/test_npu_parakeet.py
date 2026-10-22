"""The Parakeet TPU encoder export (scripts/npu/export_parakeet.py) against
the C++ side (cpp/npu/encoder_contract.hpp, cpp/tests/npu_fingerprint_test.cpp):
the model hash of a synthesized Parakeet GGUF, the pieces' contract keys and
I/O sizes; and, with torch installed, the five TPU graphs against a direct
port of the GPU engine's encoder on a tiny random model (the rel-pos
identity, the masks, the residual scales, the chunked subsampling).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

EXPORTER = Path(__file__).resolve().parents[1] / "scripts" / "npu" / "export_parakeet.py"

# kParakeetDigestVector in cpp/tests/npu_fingerprint_test.cpp.
PARAKEET_DIGEST_VECTOR = "0x29faef0bc7c1f073"


def _exporter():
    spec = importlib.util.spec_from_file_location("export_parakeet", EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fill(name: str, n: int) -> np.ndarray:
    return ((np.arange(n, dtype=np.int64) * 37 + 11 + len(name)) & 255).astype(np.uint8)


def _write_model(path: Path, tweak: int) -> None:
    """write_parakeet_model() of cpp/tests/npu_fingerprint_test.cpp."""
    w = gguf.GGUFWriter(str(path), "parakeet_tdt")
    w.add_uint32("parakeet.encoder.d_model", 1024)
    w.add_uint32("parakeet.encoder.n_layers", 24)
    w.add_string("parakeet.encoder.conv_norm_type", "batch_norm")
    w.add_uint32("parakeet.preprocessor.n_mels", 80 if tweak == 3 else 128)
    w.add_uint32("parakeet.preprocessor.hop_length", 320 if tweak == 4 else 160)
    w.add_uint32("parakeet.decoder.pred_hidden", 640)
    q = _fill("encoder.layers.0.self_attn.linear_q.weight", 68)     # Q8_0, ne [32, 2]
    if tweak == 1:
        q[40] ^= 1
    w.add_tensor("encoder.layers.0.self_attn.linear_q.weight", q.reshape(2, 34),
                 raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    for name, n in (("encoder.pre_encode.out.bias", 4), ("encoder.layers.0.norm_out.weight", 4),
                    ("joint.enc.bias", 3)):
        w.add_tensor(name, _fill(name, 4 * n).view(np.float32))
    dec = _fill("decoder.prediction.embed.weight", 32)
    if tweak == 2:
        dec[3] ^= 1
    w.add_tensor("decoder.prediction.embed.weight", dec.view(np.float32).reshape(2, 4))
    w.add_tensor("joint.joint_net.2.bias", _fill("joint.joint_net.2.bias", 20).view(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_model_hash_matches_the_cpp_vector(tmp_path):
    ep = _exporter()
    h = {}
    for tweak in range(5):
        p = tmp_path / f"pk{tweak}.gguf"
        _write_model(p, tweak)
        h[tweak] = ep.model_hash(str(p))
    assert h[0] == PARAKEET_DIGEST_VECTOR
    assert h[1] != h[0]          # an encoder weight byte
    assert h[2] == h[0]          # a decoder tensor: outside the encoder
    assert h[3] != h[0]          # n_mels: the graph depends on it
    assert h[4] == h[0]          # the hop length: outside


def test_piece_contract_matches_the_cpp_side():
    ep = _exporter()
    cfg = dict(D=1024, L=24, H=8, FF=4096, CK=9, SC=256, NM=128, JH=640)
    assert ep.piece_names(cfg) == ["sub", "l0-6", "l6-12", "l12-18", "l18-24"]
    assert ep.piece_key(cfg, "l0-6") == (
        "parakeet-enc/1 piece=l0-6 mels=128 chunk=26 rows=416 d=1024 L=24 H=8 ff=4096 ck=9 sc=256 jh=640")
    assert ep.piece_io(cfg, "sub") == ([110080, 428, 212, 104], [106496])
    assert ep.piece_io(cfg, "l6-12") == ([1703936, 1664, 1664], [1703936])
    assert ep.piece_io(cfg, "l18-24") == ([1703936, 1664, 1664], [1064960])


def _tiny_model(seed: int):
    rng = np.random.default_rng(seed)
    D, H, L, FF, CK, SC, NM, JH = 32, 4, 6, 64, 5, 8, 16, 12
    f3 = 2   # 16 -> 8 -> 4 -> 2 frequency bins
    def r(*s, sc=0.2):
        return (rng.standard_normal(s) * sc).astype(np.float32)
    w = {
        "encoder.pre_encode.conv.0.weight": r(SC, 1, 3, 3, sc=0.5), "encoder.pre_encode.conv.0.bias": r(SC),
        "encoder.pre_encode.conv.2.weight": r(SC, 1, 3, 3, sc=0.5), "encoder.pre_encode.conv.2.bias": r(SC),
        "encoder.pre_encode.conv.3.weight": r(SC, SC, 1, 1, sc=0.5), "encoder.pre_encode.conv.3.bias": r(SC),
        "encoder.pre_encode.conv.5.weight": r(SC, 1, 3, 3, sc=0.5), "encoder.pre_encode.conv.5.bias": r(SC),
        "encoder.pre_encode.conv.6.weight": r(SC, SC, 1, 1, sc=0.5), "encoder.pre_encode.conv.6.bias": r(SC),
        # Parakeet-like magnitudes: the subsampling output in the hundreds, the
        # residual stream in the tens (norm_out gains ~8).
        "encoder.pre_encode.out.weight": r(D, SC * f3, sc=100.0), "encoder.pre_encode.out.bias": r(D, sc=10.0),
        "joint.enc.weight": r(JH, D), "joint.enc.bias": r(JH),
    }
    for li in range(L):
        p = f"encoder.layers.{li}."
        for n in ("feed_forward1", "feed_forward2"):
            w[p + n + ".linear1.weight"] = r(FF, D)
            w[p + n + ".linear2.weight"] = r(D, FF)
        for n in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
            w[p + "self_attn." + n + ".weight"] = r(D, D)
        w[p + "self_attn.pos_bias_u"] = r(H, D // H)
        w[p + "self_attn.pos_bias_v"] = r(H, D // H)
        w[p + "conv.pointwise_conv1.weight"] = r(2 * D, D, 1)
        w[p + "conv.pointwise_conv2.weight"] = r(D, D, 1)
        w[p + "conv.depthwise_conv.weight"] = r(D, 1, CK, sc=0.5)
        w[p + "conv.batch_norm.weight"] = 1 + r(D)
        w[p + "conv.batch_norm.bias"] = r(D)
        w[p + "conv.batch_norm.running_mean"] = r(D)
        w[p + "conv.batch_norm.running_var"] = (1 + np.abs(r(D))).astype(np.float32)
        for n in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            w[p + n + ".weight"] = (8.0 if n == "norm_out" else 1.0) * (1 + r(D))
            w[p + n + ".bias"] = r(D)
    return dict(D=D, L=L, H=H, FF=FF, CK=CK, SC=SC, NM=NM, JH=JH), w


@pytest.mark.parametrize("frames", [93, 1000, 3201])
def test_tpu_graphs_compute_the_gpu_engines_encoder(frames):
    pytest.importorskip("torch")
    ep = _exporter()
    cfg, w = _tiny_model(frames)
    mel = np.random.default_rng(1).standard_normal((frames, cfg["NM"])).astype(np.float32)
    ref = ep.reference(cfg, w, mel)
    got = ep.run_torch(cfg, w, mel)
    assert got.shape == ref.shape == (ep.lengths(frames)[2], cfg["JH"])
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 2e-3   # the LayerNorm epsilon floor (export_parakeet.py LN_EPS_MIN); a wrong graph is ~1
