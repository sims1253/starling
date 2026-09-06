"""Exercise the quantizer CLI and inspect its written GGUFs (CPU only).

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
EMBED = "decoder.prediction.embed.weight"
LINEARS = [
    "joint.pred.weight", "joint.joint_net.2.weight",
    *[f"decoder.prediction.dec_rnn.lstm.weight_{kind}_l{layer}"
      for kind in ("ih", "hh") for layer in (0, 1)],
]


@pytest.fixture
def quantize(tmp_path):
    binary = Path(os.environ.get("STARLING_QUANTIZE_BIN", ROOT / "build-cpu/starling-quantize"))
    if not binary.is_file():
        if "STARLING_QUANTIZE_BIN" in os.environ:
            pytest.fail(f"STARLING_QUANTIZE_BIN does not exist: {binary}")
        pytest.skip("build starling-quantize or set STARLING_QUANTIZE_BIN")
    imatrix = tmp_path / "imatrix.bin"
    name = b"encoder.weight"
    imatrix.write_bytes(b"STLGIMX1" + struct.pack("<III", 1, 1, len(name)) + name
                       + struct.pack("<IQ", 256, 1) + np.ones(256, dtype="<f4").tobytes())
    calls = 0

    def run(recipe=None, arch="parakeet_tdt", level="iq2_xxs"):
        nonlocal calls
        calls += 1
        source = tmp_path / f"source{calls}.gguf"
        output = tmp_path / f"output{calls}.gguf"
        writer = gguf.GGUFWriter(str(source), arch)
        rng = np.random.default_rng(42)
        for name in [EMBED, *LINEARS, "encoder.weight", "other.embed.weight"]:
            width = 256 if name == "encoder.weight" else 640
            writer.add_tensor(name, rng.standard_normal((4, width)).astype(np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        command = [str(binary), "--input", str(source), "--output", str(output),
                   "--quant", level, "--imatrix", str(imatrix), "--quiet"]
        if recipe is not None:
            recipe_path = tmp_path / f"recipe{calls}"
            recipe_path.write_text(recipe)
            command += ["--recipe", str(recipe_path)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        reader = gguf.GGUFReader(str(output))
        return {t.name: (t.tensor_type, t.n_bytes) for t in reader.tensors}

    return run


def test_compact_recipe_changes_only_embedding_and_six_linears(quantize):
    baseline = quantize()
    compact = quantize((ROOT / "benchmarks/recipes/parakeet-iq2-compact.recipe").read_text())
    types = gguf.GGMLQuantizationType
    assert baseline[EMBED] == (types.F32, 4 * 640 * 4)
    assert compact[EMBED] == (types.Q8_0, 4 * 640 // 32 * 34)
    for name in LINEARS:
        assert baseline[name] == (types.Q8_0, 4 * 640 // 32 * 34)
        assert compact[name] == (types.IQ4_NL, 4 * 640 // 32 * 18)
    for name in ("encoder.weight", "other.embed.weight"):
        assert compact[name] == baseline[name]
    assert sum(n for _, n in compact.values()) < sum(n for _, n in baseline.values())


@pytest.mark.parametrize("arch", ["parakeet_tdt", "moss", ""])
def test_embedding_requires_parakeet_and_explicit_rule(quantize, arch):
    types = gguf.GGMLQuantizationType
    implicit = quantize("default q8_0\n", arch=arch)
    explicit = quantize("default q8_0\n^decoder\\.prediction\\.embed\\.weight$ q8_0\n", arch=arch)
    assert implicit[EMBED][0] == types.F32
    assert explicit[EMBED][0] == (types.Q8_0 if arch == "parakeet_tdt" else types.F32)
    assert explicit["other.embed.weight"][0] == types.F32


def test_first_matching_precision_override_is_preserved(quantize):
    recipe = (ROOT / "benchmarks/recipes/parakeet-iq2-compact.recipe").read_text()
    recipe = "^decoder\\.prediction\\.embed\\.weight$ f32\n^joint\\.pred\\.weight$ q8_0\n" + recipe
    result = quantize(recipe)
    assert result[EMBED][0] == gguf.GGMLQuantizationType.F32
    assert result["joint.pred.weight"][0] == gguf.GGMLQuantizationType.Q8_0
    assert result["joint.joint_net.2.weight"][0] == gguf.GGMLQuantizationType.IQ4_NL


def test_high_precision_named_level_keeps_original_fallback(quantize):
    result = quantize(level="q6_k")
    assert result[EMBED][0] == gguf.GGMLQuantizationType.F32
    for name in LINEARS:
        assert result[name][0] == gguf.GGMLQuantizationType.Q8_0


@pytest.mark.parametrize("dtype", ["iq2_xxs", "iq4_nl", "q6_k"])
def test_embedding_rejects_unvalidated_recipe_precision(quantize, dtype):
    with pytest.raises(subprocess.CalledProcessError) as error:
        quantize(f"default q8_0\n^decoder\\.prediction\\.embed\\.weight$ {dtype}\n")
    assert "embedding recipe supports only q8_0 or f32" in error.value.stderr
