"""Cache-consistency guard for the real-corpus fixture.

The bug this locks down: ``load_real_corpus`` used to accept a cache whenever it
held *at least* ``n`` wavs and *at least* ``n`` references. A cache left behind
by a larger earlier call (``utterance_000..031``) beside a smaller
``reference.json`` (0..7) satisfied that check, so the loader served real audio
paired with the wrong transcript and silently poisoned every WER computed from
it. ``_pick_varied`` picks different items for different ``n``, so the numeric
indices lined up while the audio/text pairs did not.

These tests exercise the guard and the orphan cleanup without touching the
network or the real cache directory.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))

import get_real_corpus as grc  # noqa: E402


def _point_cache_at(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(grc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(grc, "REF_TEXT_PATH", tmp_path / "reference.json")


def _touch_wavs(directory: Path, indices) -> None:
    for i in indices:
        (directory / f"utterance_{i:03d}.wav").touch()


def _fake_items(n: int) -> list:
    """Varied-length fake utterances shaped like the HF audio rows."""
    return [
        {
            "audio": {
                "array": np.full(1600 + i, float(i), dtype=np.float32),
                "sampling_rate": 16000,
            },
            "text": f"utt {i}",
        }
        for i in range(n)
    ]


def test_accepts_consistent_cache(monkeypatch, tmp_path):
    _point_cache_at(monkeypatch, tmp_path)
    _touch_wavs(tmp_path, range(8))
    ref = {i: f"ref {i}" for i in range(8)}
    assert grc._cache_is_consistent(grc._cache_paths(), ref, 8)


def test_rejects_orphan_wavs_from_larger_run(monkeypatch, tmp_path):
    """The production bug: 32 cached wavs beside an 8-entry reference.json."""
    _point_cache_at(monkeypatch, tmp_path)
    _touch_wavs(tmp_path, range(32))
    ref = {i: f"ref {i}" for i in range(8)}
    assert not grc._cache_is_consistent(grc._cache_paths(), ref, 8)


def test_rejects_missing_or_blank_reference(monkeypatch, tmp_path):
    _point_cache_at(monkeypatch, tmp_path)
    _touch_wavs(tmp_path, range(8))
    assert not grc._cache_is_consistent(grc._cache_paths(), {}, 8)
    blank = {i: ("" if i == 3 else f"ref {i}") for i in range(8)}
    assert not grc._cache_is_consistent(grc._cache_paths(), blank, 8)


def test_rejects_cache_smaller_than_request(monkeypatch, tmp_path):
    _point_cache_at(monkeypatch, tmp_path)
    _touch_wavs(tmp_path, range(4))
    ref = {i: f"ref {i}" for i in range(4)}
    assert not grc._cache_is_consistent(grc._cache_paths(), ref, 8)


def test_load_reference_map_tolerates_invalid_json(monkeypatch, tmp_path):
    _point_cache_at(monkeypatch, tmp_path)
    grc.REF_TEXT_PATH.write_text("{not json")
    assert grc._load_reference_map() == {}


def test_mismatched_cache_downloads_and_removes_orphans(monkeypatch, tmp_path):
    """A mismatched cache re-downloads and leaves wavs/refs in lockstep."""
    _point_cache_at(monkeypatch, tmp_path)
    _touch_wavs(tmp_path, range(32))
    grc.REF_TEXT_PATH.write_text(json.dumps({str(i): f"old {i}" for i in range(8)}))

    monkeypatch.setattr(grc.sf, "write", lambda path, *a, **k: Path(path).touch())
    monkeypatch.setattr(grc.sf, "read", lambda path, **k: (np.zeros(1600, dtype=np.float32), 16000))
    # Inject a fake `datasets` so the function's local import never hits the
    # network (and the test does not require the [bench] extra).
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: _fake_items(20)
    monkeypatch.setitem(sys.modules, "datasets", fake)

    items = grc.load_real_corpus(8)

    assert len(items) == 8
    remaining = {grc._wav_index(p) for p in grc._cache_paths()}
    assert remaining == set(range(8))
    ref = json.loads(grc.REF_TEXT_PATH.read_text())
    assert set(map(int, ref)) == set(range(8))
