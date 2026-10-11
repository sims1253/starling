"""Per-architecture opt-in tensors of starling-quantize (CPU only).

Embeddings and conv-named tensors stay at the source dtype unless an explicit
recipe rule selects them, and only for the architectures and types the
engines read them with (kOptIns in cpp/tools/starling_quantize.cpp).
Build starling-quantize first, or set STARLING_QUANTIZE_BIN to its location.
"""

import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
ROOT = Path(__file__).resolve().parents[1]
TYPES = gguf.GGMLQuantizationType
PW1 = "encoder.layers.0.conv.pointwise_conv1.weight"


def imatrix_entry(name: str, width: int) -> bytes:
    raw = name.encode()
    return (struct.pack("<I", len(raw)) + raw + struct.pack("<IQ", width, 1)
            + np.ones(width, dtype="<f4").tobytes())


@pytest.fixture
def quantize(tmp_path):
    binary = Path(os.environ.get("STARLING_QUANTIZE_BIN", ROOT / "build-cpu/starling-quantize"))
    if not binary.is_file():
        if "STARLING_QUANTIZE_BIN" in os.environ:
            pytest.fail(f"STARLING_QUANTIZE_BIN does not exist: {binary}")
        pytest.skip("build starling-quantize or set STARLING_QUANTIZE_BIN")
    # The collector records a weight the engine casts/reshapes first under
    # ggml's derived name; the quantizer must use it for the weight itself.
    imatrix = tmp_path / "imatrix.imx"
    imatrix.write_bytes(b"STLGIMX1" + struct.pack("<II", 1, 1)
                        + imatrix_entry(PW1 + " (copy) (reshaped)", 256))
    calls = 0

    def run(arch, tensors, recipe):
        nonlocal calls
        calls += 1
        source, output = tmp_path / f"in{calls}.gguf", tmp_path / f"out{calls}.gguf"
        writer = gguf.GGUFWriter(str(source), arch)
        rng = np.random.default_rng(7)
        for name, shape in tensors.items():
            writer.add_tensor(name, rng.standard_normal(shape).astype(np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        recipe_path = tmp_path / f"recipe{calls}"
        recipe_path.write_text(recipe)
        done = subprocess.run([str(binary), "--input", str(source), "--output", str(output),
                               "--recipe", str(recipe_path), "--imatrix", str(imatrix)],
                              check=True, capture_output=True, text=True)
        reader = gguf.GGUFReader(str(output))
        return {t.name: (t.tensor_type, tuple(int(x) for x in t.shape)) for t in reader.tensors}, done.stderr

    return run


LLM = {"llm.embed.weight": (64, 256), "enc.pos_embed": (64, 256), "llm.blk.0.ffn.up.weight": (64, 256)}


@pytest.mark.parametrize("arch", ["qwen3", "moss_transcribe", "granite", "ark", "voxtral"])
def test_llm_embedding_needs_an_explicit_rule(quantize, arch):
    implicit, _ = quantize(arch, LLM, "default q4_0\n")
    explicit, _ = quantize(arch, LLM, "default q4_0\n^llm\\.embed\\.weight$ q4_0\n")
    broad, _ = quantize(arch, LLM, "default q4_0\n.* q8_0\n")
    assert implicit["llm.embed.weight"][0] == TYPES.F32
    assert implicit["llm.blk.0.ffn.up.weight"][0] == TYPES.Q4_0
    assert explicit["llm.embed.weight"][0] == TYPES.Q4_0
    assert broad["llm.embed.weight"][0] == TYPES.Q8_0
    # Positional tables are never opt-in, whatever a rule says.
    for result in (implicit, explicit, broad):
        assert result["enc.pos_embed"][0] == TYPES.F32


@pytest.mark.parametrize("arch", ["s1", "parakeet_tdt", ""])
def test_llm_embedding_stays_for_other_architectures(quantize, arch):
    result, _ = quantize(arch, LLM, "default q4_0\n^llm\\.embed\\.weight$ q4_0\n")
    assert result["llm.embed.weight"][0] == TYPES.F32


def test_llm_embedding_rejects_unlisted_types(quantize):
    with pytest.raises(subprocess.CalledProcessError) as error:
        quantize("qwen3", LLM, "default q4_0\n^llm\\.embed\\.weight$ iq4_nl\n")
    assert "embedding recipe supports only" in error.value.stderr


PARAKEET = {PW1: (512, 256, 1), "encoder.layers.0.feed_forward1.linear1.weight": (64, 256)}


def test_pointwise_conv_is_flattened_only_when_quantized(quantize):
    kept, _ = quantize("parakeet_tdt", PARAKEET, "default q4_0\n")
    f16, _ = quantize("parakeet_tdt", PARAKEET, f"default q4_0\n{PW1}$ f16\n")
    q4, stderr = quantize("parakeet_tdt", PARAKEET, f"default q4_0\n{PW1}$ q4_0\n")
    assert kept[PW1] == (TYPES.F32, (1, 256, 512))
    assert f16[PW1] == (TYPES.F16, (1, 256, 512))
    assert q4[PW1] == (TYPES.Q4_0, (256, 512))
    assert "(1 by derived name)" in stderr


def test_conv_named_linears_opt_in_per_architecture(quantize):
    tensors = {"enc.blk.3.conv_up.weight": (64, 256), "enc.conv_out.weight": (64, 256)}
    recipe = "default q4_0\nconv_up q8_0\nconv_out q8_0\n"
    granite, _ = quantize("granite", tensors, recipe)
    moss, _ = quantize("moss_transcribe", tensors, recipe)
    assert granite["enc.blk.3.conv_up.weight"][0] == TYPES.Q8_0
    assert granite["enc.conv_out.weight"][0] == TYPES.F32
    assert moss["enc.conv_out.weight"][0] == TYPES.Q8_0
    assert moss["enc.blk.3.conv_up.weight"][0] == TYPES.F32
