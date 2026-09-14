"""CPU checks for comparable quantization inputs and complete corpus requests."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import fleurs_download
import fleurs_util
import wer_quant


def test_corpus_json_retains_ordered_clips_for_paired_comparisons(monkeypatch, tmp_path):
    import hashlib
    import json
    import soundfile as sf

    model = tmp_path / "model.gguf"
    model.touch()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index, reference in enumerate(("one two", "three four")):
        wav = corpus / f"en_us_test_{index}.wav"
        sf.write(wav, np.zeros(1600, dtype=np.float32), 16000)
        wav.with_suffix(".txt").write_text(reference)
    hypotheses = iter(("one", "three four", "one two", "three"))
    monkeypatch.setattr(wer_quant, "StarlingGgmlParakeet", lambda: SimpleNamespace(
        available=True, load=lambda: None, close=lambda: None,
        transcribe=lambda audio: [next(hypotheses)]))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["wer_quant", "--tiers", "", "--models",
                                      f"first={model}", f"second={model}",
                                      "--corpus", str(corpus), "--include-clips",
                                      "--json", str(output)])
    assert wer_quant.main() == 0
    row, second = json.loads(output.read_text())
    audio_hash = hashlib.sha256(np.zeros(1600, dtype=np.float32).tobytes()).hexdigest()
    assert row["clips"]["en_us_test"] == [
        {"id": "en_us_test_0.wav", "audio_sha256": audio_hash,
         "reference": "one two", "hypothesis": "one", "wer": 50.0},
        {"id": "en_us_test_1.wav", "audio_sha256": audio_hash,
         "reference": "three four", "hypothesis": "three four", "wer": 0.0},
    ]
    assert row["wer"]["en_us_test"] == 25.0
    first_clips, second_clips = row["clips"]["en_us_test"], second["clips"]["en_us_test"]
    assert [(c["id"], c["audio_sha256"], c["reference"]) for c in first_clips] == [
        (c["id"], c["audio_sha256"], c["reference"]) for c in second_clips]
    assert [c["wer"] for c in second_clips] == [0.0, 50.0]


def test_variants_receive_identical_noise_regardless_of_model_order(monkeypatch, tmp_path):
    models = [tmp_path / name for name in ("a.gguf", "b.gguf")]
    for path in models:
        path.touch()
    source = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    monkeypatch.setattr(wer_quant.mkfx, "load_fixtures", lambda: {"short": source.copy()})
    seen = []

    def transcribe(audio):
        seen.append(audio.copy())
        return [wer_quant.REFERENCE_TRANSCRIPTS["short"]]

    monkeypatch.setattr(wer_quant, "StarlingGgmlParakeet", lambda: SimpleNamespace(
        available=True, load=lambda: None, close=lambda: None, transcribe=transcribe))
    for order in (models, models[::-1]):
        monkeypatch.setattr(sys, "argv", ["wer_quant", "--tiers", "short",
                                          "--snr-db", "5", "--models", *map(str, order)])
        assert wer_quant.main() == 0
    assert len(seen) == 4
    assert not np.array_equal(seen[0], source)
    for audio in seen[1:]:
        np.testing.assert_array_equal(audio, seen[0])


@pytest.mark.parametrize("failure", ["model", "engine", "corpus", "sidecar", "blank", "no-cohort"])
def test_incomplete_evaluation_fails(monkeypatch, tmp_path, capsys, failure):
    model = tmp_path / "model.gguf"
    if failure != "model":
        model.touch()
    monkeypatch.setattr(wer_quant.mkfx, "load_fixtures", lambda: {"short": np.zeros(1600)})
    monkeypatch.setattr(wer_quant, "StarlingGgmlParakeet", lambda: SimpleNamespace(
        available=failure != "engine", load=lambda: None, close=lambda: None,
        transcribe=lambda audio: ["words"]))
    argv = ["wer_quant", "--tiers", "" if failure == "no-cohort" else "short",
            "--models", str(model)]
    if failure in ("corpus", "sidecar", "blank"):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        if failure in ("sidecar", "blank"):
            (corpus / "en_us_test_0.wav").touch()
        if failure == "blank":
            (corpus / "en_us_test_0.txt").write_text("  ")
        argv += ["--corpus", str(corpus)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        wer_quant.main()
    assert error.value.code == 2
    messages = {"model": "does not exist", "engine": "native engine unavailable",
                "corpus": "no WAV files found", "sidecar": "sidecar is missing",
                "blank": "sidecar is empty", "no-cohort": "select at least one"}
    assert messages[failure] in capsys.readouterr().err


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
