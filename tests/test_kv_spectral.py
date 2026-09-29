"""The spectral probe must expose error that training-only PCA would hide."""

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


@lru_cache(maxsize=1)
def _probe():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_kv_spectral.py"
    spec = importlib.util.spec_from_file_location("bench_kv_spectral", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_held_out_error_detects_new_direction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    probe = _probe()
    g = torch.Generator().manual_seed(23)
    train = torch.zeros(256, 1, 4)
    train[:, 0, :2] = torch.randn(256, 2, generator=g)
    matching = torch.zeros(64, 1, 4)
    matching[:, 0, :2] = torch.randn(64, 2, generator=g)
    shifted = matching.clone()
    shifted[:, 0, 2] = torch.randn(64, generator=g) * 4

    same = probe.pca_layer(train, matching, (0.99,))
    different = probe.pca_layer(train, shifted, (0.99,))

    assert same["d_eff_mean@0.99"] == different["d_eff_mean@0.99"] == 2
    assert same["held_out_relative_mse_mean@d_eff_0.99"] < 1e-6
    assert different["held_out_relative_mse_mean@d_eff_0.99"] > 0.5
    assert different["held_out_relative_mse_mean@d_eff_0.99"] <= 1.00001
    assert different["held_out_relative_mse_mean_by_rank"]["4"] < 1e-6


def test_non_16k_wav_rejected_before_capture(tmp_path):
    probe = _probe()
    path = tmp_path / "48k.wav"
    sf.write(path, np.zeros(480, dtype=np.float32), 48000)
    with pytest.raises(ValueError, match="sample rate 48000; expected 16000"):
        probe.gather_calibration_clips(2, [path, path])

    # Direct callers of measure_granite cannot bypass the WAV reader's guard.
    with pytest.raises(ValueError, match="sample rate 48000; expected 16000"):
        probe.measure_granite([(np.zeros(480, dtype=np.float32), 48000, str(path))])


def test_explicit_audio_cannot_be_silently_truncated(tmp_path):
    probe = _probe()
    paths = [tmp_path / f"{i}.wav" for i in range(3)]
    with pytest.raises(ValueError, match="exceeding --clips 2"):
        probe.gather_calibration_clips(2, paths)


def test_silent_held_out_head_cannot_score_perfect_reconstruction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    probe = _probe()
    train = torch.randn(8, 1, 4)
    held_out = train.mean(dim=0, keepdim=True).expand(4, -1, -1).clone()
    with pytest.raises(ValueError, match="relative error is undefined"):
        probe.pca_layer(train, held_out, (0.99,))
