"""The Pixel runner must verify the staged inputs before starting inference."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER_PATH = (Path(__file__).resolve().parents[1] / "docs/evidence" /
               "pixel-s1-2026-09-28/run_s1.py")
spec = importlib.util.spec_from_file_location("s1_pixel_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inputs(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    source = artifact_dir / "s1-bf16.gguf"
    model = artifact_dir / "s1-q4-k-m.gguf"
    binary = tmp_path / "starling-serve"
    source.write_bytes(b"bf16 model")
    model.write_bytes(b"q4 model")
    binary.write_bytes(b"server binary")
    return artifact_dir, source, model, binary


def test_staged_hashes_match_recorded_local_inputs(tmp_path, monkeypatch):
    _, source, model, binary = _inputs(tmp_path)
    expected = {source.name: _hash(source.read_bytes()),
                model.name: _hash(model.read_bytes()),
                binary.name: _hash(binary.read_bytes())}
    checked = []

    def fake_adb(*args, **kwargs):
        assert args[:2] == ("shell", "sha256sum")
        assert kwargs == {"timeout": 300}
        name = Path(args[2]).name
        checked.append(name)
        return SimpleNamespace(stdout=f"{expected[name]}  {args[2]}\n")

    monkeypatch.setattr(runner, "adb", fake_adb)
    verified = runner.verify_staged_artifacts("q4-k-m", model, source, binary)
    assert checked == ["s1-bf16.gguf", "s1-q4-k-m.gguf", "starling-serve"]
    assert verified == {"model_sha256": expected[model.name],
                        "model_bytes": model.stat().st_size,
                        "source_sha256": expected[source.name],
                        "engine_sha256": expected[binary.name]}


def test_remote_hash_must_identify_the_requested_file(monkeypatch):
    def fake_adb(*args, **kwargs):
        return SimpleNamespace(stdout=f"{_hash(b'other file')}  /tmp/other-file\n")

    monkeypatch.setattr(runner, "adb", fake_adb)
    with pytest.raises(RuntimeError, match="invalid remote SHA-256 output"):
        runner.staged_digest("starling-serve")


@pytest.mark.parametrize("bad_name", ["s1-bf16.gguf", "s1-q4-k-m.gguf", "starling-serve"])
def test_staged_artifact_mismatch_stops_before_launch(tmp_path, monkeypatch, bad_name):
    artifact_dir, source, model, binary = _inputs(tmp_path)
    expected = {source.name: _hash(source.read_bytes()),
                model.name: _hash(model.read_bytes()),
                binary.name: _hash(binary.read_bytes())}
    expected[bad_name] = _hash(b"other staged bytes")

    def fake_adb(*args, **kwargs):
        if args[:2] == ("shell", "getprop"):
            value = "Google" if args[2] == "ro.product.manufacturer" else "Pixel 10 Pro"
            return SimpleNamespace(stdout=value + "\n")
        if args[:2] == ("shell", "sha256sum"):
            assert kwargs == {"timeout": 300}
            return SimpleNamespace(stdout=f"{expected[Path(args[2]).name]}  {args[2]}\n")
        raise AssertionError(f"unexpected device action: {args}")

    monkeypatch.setattr(runner, "adb", fake_adb)
    monkeypatch.setattr(runner, "SERIAL", "test-device")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "q4-k-m", "--artifact-dir",
                                  str(artifact_dir), "--binary", str(binary),
                                  "--engine-source-commit", "test"])
    with pytest.raises(RuntimeError, match=f"staged {bad_name} differs"):
        runner.main()


def test_wrong_device_stops_before_artifact_or_launch_commands(tmp_path, monkeypatch):
    artifact_dir, _, _, binary = _inputs(tmp_path)

    def fake_adb(*args, **kwargs):
        if args == ("shell", "getprop", "ro.product.manufacturer"):
            return SimpleNamespace(stdout="Other\n")
        if args == ("shell", "getprop", "ro.product.model"):
            return SimpleNamespace(stdout="Other Phone\n")
        raise AssertionError(f"unexpected device action: {args}")

    monkeypatch.setattr(runner, "adb", fake_adb)
    monkeypatch.setattr(runner, "SERIAL", "test-device")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "bf16", "--artifact-dir",
                                  str(artifact_dir), "--binary", str(binary),
                                  "--engine-source-commit", "test"])
    with pytest.raises(RuntimeError, match="expected Google Pixel 10 Pro"):
        runner.main()
