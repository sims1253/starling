"""Corpus helpers feeding imatrix calibration (benchmarks/fleurs_*.py)."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import fleurs_download  # noqa: E402
import fleurs_util  # noqa: E402


def test_fleurs_requires_full_requested_budget(monkeypatch):
    clip = {"audio": {"array": np.zeros(1600), "sampling_rate": 16000},
            "transcription": "words"}
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        load_dataset=lambda *args, **kwargs: [clip]))
    with pytest.raises(RuntimeError, match="requested 2 clips, found 1"):
        list(fleurs_util.fleurs_clips({"en_us": 2}))

    def broken_stream():
        yield clip
        raise OSError("download failed")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        load_dataset=lambda *args, **kwargs: broken_stream()))
    assert len(list(fleurs_util.fleurs_clips({"en_us": 1}))) == 1
    with pytest.raises(OSError, match="download failed"):
        list(fleurs_util.fleurs_clips({"en_us": 2}))


def test_download_checks_files_instead_of_stale_marker(monkeypatch, tmp_path):
    import subprocess

    (tmp_path / "en_us_train.done").touch()
    calls = []

    def populate(count):
        for i in range(count):
            for suffix in (".wav", ".txt"):
                (tmp_path / f"en_us_train_{i}{suffix}").write_bytes(b"fixture")

    def run(command, **kwargs):
        count = int(command[-2])
        calls.append(count)
        populate(count)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    populate(2)
    for count in (2, 4, 4):
        monkeypatch.setattr(sys, "argv", ["fleurs_download", "--out", str(tmp_path),
                                          "--fleurs", f"en_us:{count}"])
        assert fleurs_download.main() == 0
    assert calls == [4]
    (tmp_path / "en_us_train_3.txt").unlink()
    assert fleurs_download.main() == 0
    assert calls == [4, 4]
