"""Native benchmark adapters preserve routing and the C API input contract."""

from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import engines
from starling import _ggml


@pytest.mark.parametrize("slug,cls,kind", [
    ("parakeet", engines.StarlingGgmlParakeet, _ggml.PARAKEET_TDT),
    ("moss", engines.StarlingGgmlMoss, _ggml.MOSS),
    ("ark", engines.StarlingGgmlArk, _ggml.ARK),
    ("higgs", engines.StarlingGgmlHiggs, _ggml.HIGGS),
    ("hojo", engines.StarlingGgmlHojo, _ggml.HOJO),
    ("granite", engines.StarlingGgmlGranite, _ggml.GRANITE),
    ("qwen3", engines.StarlingGgmlQwen3, _ggml.QWEN3),
    ("s1", engines.StarlingGgmlS1, _ggml.S1),
    ("audex", engines.StarlingGgmlAudex, _ggml.AUDEX),
])
def test_native_adapter_contract(monkeypatch, tmp_path, slug, cls, kind):
    model_path = tmp_path / "model.gguf"
    monkeypatch.setattr(engines, f"STARLING_GGML_{slug.upper()}_MODEL", model_path)
    monkeypatch.delenv(f"STARLING_GGML_{slug.upper()}_MODEL", raising=False)
    monkeypatch.setattr(_ggml, "available", lambda: True)
    native = Mock()
    factory = Mock(return_value=native)
    monkeypatch.setattr(_ggml, "GgmlModel", factory)
    monkeypatch.setattr(engines, "available_keys", lambda: [f"starling-ggml-{slug}"])
    engine = engines.build_engines([slug], ["starling-ggml"])[slug][0]
    assert isinstance(engine, cls)
    assert not engine.available
    model_path.touch()
    assert engine.available
    factory.assert_not_called()  # discovery must not load weights

    # The shared adapter must convert a strided float64 view to contiguous f32.
    audio = np.arange(10, dtype=np.float64)[::2]

    def transcribe(pcm, count, sample_rate):
        assert sample_rate == 16000
        np.testing.assert_array_equal(np.ctypeslib.as_array(pcm, shape=(count,)), audio)
        return " transcript "

    native.transcribe_pcm.side_effect = transcribe
    native.normalize_text.return_value = "normalized"
    expected = "normalized" if slug == "s1" else "transcript"
    assert engine.transcribe(audio, B=2) == [expected, expected]
    factory.assert_called_once_with(kind, str(model_path))
    if slug == "s1":
        native.transcribe_pcm.assert_not_called()
        native.normalize_text.assert_called_with(engines._s1_tier_transcript(audio))
    else:
        native.normalize_text.assert_not_called()
        assert native.transcribe_pcm.call_count == 2
    engine.close()
    engine.close()
    native.close.assert_called_once()


def test_parakeet_reloads_model_path_for_quantization_sweeps(monkeypatch, tmp_path):
    factory = Mock()
    monkeypatch.setattr(_ggml, "GgmlModel", factory)
    engine = engines.StarlingGgmlParakeet()
    for name in ("reference.gguf", "quantized.gguf"):
        path = tmp_path / name
        monkeypatch.setenv("STARLING_GGML_PARAKEET_MODEL", str(path))
        engine.load()
        factory.assert_called_with(_ggml.PARAKEET_TDT, str(path))
        engine.close()
    assert factory.call_count == 2
