"""Open ASR Leaderboard English short-form evaluation corpus.

Faithful reproduction of the dataset side of the Hugging Face Open ASR
Leaderboard (https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
English short-form eval. Every clip comes from one Hub repo
``hf-audio/open-asr-leaderboard``, configured per dataset, exactly as the
leaderboard's ``run_whisper.sh`` does:

    DATASET_PATH="hf-audio/open-asr-leaderboard"
    DATASET_CONFIGS=(
        "voxpopuli test"
        "ami test"
        "earnings22 test"
        "gigaspeech test"
        "librispeech test.clean"
        "librispeech test.other"
        "spgispeech test"
    )

Subsampling
-----------
The leaderboard runs the *full* test splits (no subsample;
``--max_eval_samples=-1``). Those are large (spgispeech 39k, gigaspeech 20k,
ami 12k clips), and the autoregressive models here are slow, so this loader
takes a deterministic ``num_samples`` cap with the SAME first-N semantics as
the leaderboard's ``--max_eval_samples N`` (``dataset.select(range(N))`` /
``.take(N)``). ``num_samples=0`` (or ``None``) means the full split -- the
genuine reproduction.

Filtering follows ``is_target_text_in_range`` (drop clips whose reference is
empty or the literal ``"ignore time segment in scoring"`` placeholder) AFTER
normalization, matching the leaderboard.

Caching
-------
Each (dataset, config, split, num_samples) is cached as PCM_16 wav under
``tests/fixtures/leaderboard_corpus/<key>/clip_*.wav`` plus a
``reference.json`` (id -> reference text) so the benchmark is reproducible
without re-downloading. Mirrors the cache pattern in ``get_real_corpus.py``.

Public API
----------
``DATASETS``                              -- the 7 (key, config, split) tuples
``LeaderboardClip``                       -- (audio_float32, sample_rate, ref_text)
``load_dataset_split(key, num_samples)``  -- list[LeaderboardClip], cached
``load_all(num_samples)``                 -- {key: list[LeaderboardClip]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

CACHE_DIR = Path(__file__).parent / "leaderboard_corpus"
SAMPLE_RATE = 16000
DATASET_PATH = "hf-audio/open-asr-leaderboard"

# The 7 English short-form configs the leaderboard evaluates (config, split).
# Key is the short label used in tables / cache dirs.
DATASETS: list[tuple[str, str, str]] = [
    ("voxpopuli", "voxpopuli", "test"),
    ("ami", "ami", "test"),
    ("earnings22", "earnings22", "test"),
    ("gigaspeech", "gigaspeech", "test"),
    ("librispeech_clean", "librispeech", "test.clean"),
    ("librispeech_other", "librispeech", "test.other"),
    ("spgispeech", "spgispeech", "test"),
]

# Placeholder reference the leaderboard drops during scoring.
_IGNORE = "ignore time segment in scoring"


@dataclass
class LeaderboardClip:
    """One cached evaluation utterance."""

    audio: np.ndarray   # float32 mono @16kHz
    sample_rate: int
    reference: str      # raw (un-normalized) ground-truth transcript

    @property
    def duration_s(self) -> float:
        return len(self.audio) / float(self.sample_rate)


def _cache_dir(key: str, num_samples: Optional[int]) -> Path:
    tag = "full" if not num_samples else f"n{num_samples}"
    return CACHE_DIR / f"{key}__{tag}"


def _is_cached(d: Path, num_samples: Optional[int]) -> bool:
    if not (d / "reference.json").exists():
        return False
    ref = json.loads((d / "reference.json").read_text())
    return len(ref) >= (num_samples or len(ref))


def _serve_cached(d: Path) -> list[LeaderboardClip]:
    """Read cached clips back in clip-index order (stable across runs)."""
    ref = {int(k): v for k, v in json.loads((d / "reference.json").read_text()).items()}
    paths = {int(p.stem.split("_")[-1]): p for p in d.glob("clip_*.wav")}
    clips: list[LeaderboardClip] = []
    for idx in sorted(paths):
        a, sr = sf.read(str(paths[idx]))
        if a.ndim != 1:
            a = a[:, 0]
        a = np.ascontiguousarray(a, dtype=np.float32)
        clips.append(LeaderboardClip(a, int(sr), ref.get(idx, "")))
    return clips


def load_dataset_split(
    key: str, num_samples: Optional[int] = None, *, token: Optional[str] = None,
) -> list[LeaderboardClip]:
    """Load (and cache) one dataset's clips.

    Args:
        key: short label in :data:`DATASETS` (e.g. ``"librispeech_clean"``).
        num_samples: deterministic first-N cap (``0``/``None`` = full split).
        token: optional HF token for the (gated) dataset repo.

    Returns the cached/downloaded clips in dataset order. Empty-reference /
    ``ignore time segment`` clips are dropped *after* they are read, matching
    the leaderboard, so a ``num_samples`` cap is a cap on the *raw* split read
    (i.e. before filtering) -- the returned list may be slightly shorter.
    """
    config = split = None
    for k, cfg, sp in DATASETS:
        if k == key:
            config, split = cfg, sp
            break
    if config is None:
        raise KeyError(f"unknown leaderboard dataset key {key!r}; "
                       f"choose from {[k for k, _, _ in DATASETS]}")

    d = _cache_dir(key, num_samples)
    if _is_cached(d, num_samples):
        return _serve_cached(d)

    d.mkdir(parents=True, exist_ok=True)
    from datasets import Audio, load_dataset
    from whisper_normalizer.english import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    ds = load_dataset(DATASET_PATH, config, split=split, token=token)
    # force 16kHz mono float32 decoding (the repo is already 16kHz, but be explicit)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    if num_samples and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    ref_out: dict[int, str] = {}
    out: list[LeaderboardClip] = []
    drop = 0
    for i in range(len(ds)):
        ex = ds[i]
        a = ex["audio"]["array"]
        if a.ndim != 1:
            a = a[:, 0]
        a = np.ascontiguousarray(a, dtype=np.float32)
        sr = int(ex["audio"].get("sampling_rate", SAMPLE_RATE)) if isinstance(ex["audio"], dict) else SAMPLE_RATE
        text = str(ex.get("text", "")).strip()
        # leaderboard filter: drop empty / placeholder refs AFTER normalization
        if normalizer(text).strip() == "" or normalizer(text).strip() == _IGNORE:
            drop += 1
            continue
        idx = len(out)
        sf.write(str(d / f"clip_{idx:05d}.wav"), a, sr, subtype="PCM_16")
        ref_out[idx] = text
        out.append(LeaderboardClip(a, sr, text))
    (d / "reference.json").write_text(json.dumps(ref_out, indent=2))
    print(f"[leaderboard_corpus] {key}: cached {len(out)} clips "
          f"(read {len(ds)}, dropped {drop} empty/placeholder)", flush=True)
    return out


def load_all(
    num_samples: Optional[int] = None,
    keys: Optional[list[str]] = None,
    *,
    token: Optional[str] = None,
) -> dict[str, list[LeaderboardClip]]:
    """Load every (or a subset of) dataset splits.

    Args:
        num_samples: deterministic first-N cap per dataset (``0``/``None`` = full).
        keys: subset of :data:`DATASETS` keys; default = all 7.
        token: optional HF token.

    Returns ``{key: [LeaderboardClip, ...]}``.
    """
    keys = keys or [k for k, _, _ in DATASETS]
    return {k: load_dataset_split(k, num_samples, token=token) for k in keys}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pre-download/cache the OAL corpus.")
    ap.add_argument("--num-samples", type=int, default=50,
                    help="first-N clips per dataset (0 = full split)")
    ap.add_argument("--keys", default=",".join(k for k, _, _ in DATASETS),
                    help="comma list of dataset keys to fetch")
    args = ap.parse_args()
    n = args.num_samples or None
    for k in [x.strip() for x in args.keys.split(",") if x.strip()]:
        load_dataset_split(k, n)
    print("done.")
