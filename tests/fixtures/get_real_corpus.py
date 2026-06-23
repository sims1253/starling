"""Download a small, real, varied English speech test set for the robustness benchmark.

The headline RTF (1749x) was measured on synthetic-repeated clips (one utterance
concatenated N times). Real speech differs from that in three load-relevant ways:

  * token density varies (some utterances are dense, some sparse),
  * blank-skip frequency varies (silence gaps change the decode loop length),
  * padding waste varies (a batch of different-length utterances pads to the
    longest, so the effective batch is less uniform than the synthetic case).

This module fetches a small set of *real* varied utterances so the robustness
benchmark can characterise those effects. Primary source is the HuggingFace
``hf-internal-testing/librispeech_asr_dummy`` "clean" "validation" split (the
same one used in the parakeet model card example): 73 real LibriSpeech
utterances, ~2-15s each, different content. Each downloaded utterance is cached
as PCM_16 .wav under ``tests/fixtures/real_corpus/`` so the benchmark is
reproducible without re-downloading.

Public surface
--------------
:func:`load_real_corpus`
    Returns a list of ``(audio_float32, sample_rate, reference_text)`` tuples
    for ``n`` varied real utterances (different content, different lengths).
:func:`load_real_corpus_batch`
    Convenience: returns just the audio arrays (for direct pipeline feeding).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

CACHE_DIR = Path(__file__).parent / "real_corpus"
SAMPLE_RATE = 16000
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "validation"

# A small LibriSpeech-style reference; written next to the wavs so the bench can
# report WER-ish sanity (not used for any correctness gate, just for reporting).
REF_TEXT_PATH = CACHE_DIR / "reference.json"


def _pick_varied(items: List[dict], n: int) -> List[int]:
    """Pick ``n`` indices spread across the duration distribution (varied lengths).

    Sorts the dataset by audio duration and picks indices at quantile points so
    the chosen batch spans short -> long rather than clustering near the median.
    """
    durs = []
    for i, ex in enumerate(items):
        arr = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        durs.append(len(arr) / float(sr))
    order = sorted(range(len(items)), key=lambda i: durs[i])
    if n >= len(order):
        return order
    # quantile-spaced picks: 0th, 1/n, 2/n, ..., (n-1)/n of the sorted-by-dur list
    picks = [order[int(round(k * (len(order) - 1) / (n - 1)))] if n > 1 else order[0]
             for k in range(n)]
    # de-dup while preserving order (rare collisions at small n)
    seen = set()
    out = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    # if dedup shrank the list, top up with unused shortest remaining
    i = 0
    while len(out) < n and i < len(order):
        if order[i] not in seen:
            seen.add(order[i])
            out.append(order[i])
        i += 1
    return out[:n]


def _cache_paths() -> List[Path]:
    return sorted(CACHE_DIR.glob("utterance_*.wav"))


def load_real_corpus(n: int = 8) -> List[Tuple[np.ndarray, int, str]]:
    """Return ``n`` real varied LibriSpeech utterances as ``(audio, sr, text)``.

    Downloads + caches on first call; later calls read the cache. The cache key
    is per-utterance wav filenames ``utterance_000.wav`` ... and a
    ``reference.json`` mapping index -> reference transcript.

    Args:
        n: number of varied utterances to return (default 8).

    Returns:
        list of ``(audio_float32_mono_16k, sample_rate, reference_text)`` tuples,
        ordered from shortest to longest duration.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _cache_paths()
    ref = {}
    if REF_TEXT_PATH.exists():
        import json
        try:
            ref = {int(k): v for k, v in json.loads(REF_TEXT_PATH.read_text()).items()}
        except (json.JSONDecodeError, OSError):
            ref = {}

    if len(cached) >= n and len(ref) >= n:
        # serve from cache
        out = []
        for p in cached[:n]:
            idx = int(p.stem.split("_")[-1])
            a, sr = sf.read(str(p))
            if a.ndim != 1:
                a = a[:, 0]
            a = np.ascontiguousarray(a, dtype=np.float32)
            out.append((a, int(sr), ref.get(idx, "")))
        # sort by duration for stable, readable reporting
        out.sort(key=lambda t: len(t[0]) / float(t[1]))
        return out

    # ---- download ----
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    items = list(ds)
    picks = _pick_varied(items, n)

    import json
    ref_out = {}
    out: List[Tuple[np.ndarray, int, str]] = []
    for new_idx, ds_idx in enumerate(picks):
        ex = items[ds_idx]
        arr = np.ascontiguousarray(ex["audio"]["array"], dtype=np.float32)
        if arr.ndim != 1:
            arr = arr[:, 0]
        sr = int(ex["audio"]["sampling_rate"])
        text = str(ex.get("text", "")).strip()
        wav_path = CACHE_DIR / f"utterance_{new_idx:03d}.wav"
        sf.write(str(wav_path), arr, sr, subtype="PCM_16")
        ref_out[new_idx] = text
        out.append((arr, sr, text))

    REF_TEXT_PATH.write_text(json.dumps(ref_out, indent=2))
    out.sort(key=lambda t: len(t[0]) / float(t[1]))
    return out


def load_real_corpus_batch(n: int = 8) -> List[np.ndarray]:
    """Convenience: return just the audio arrays for ``n`` varied utterances."""
    return [a for (a, _sr, _t) in load_real_corpus(n)]


if __name__ == "__main__":
    items = load_real_corpus(8)
    print(f"cached {len(items)} real utterances under {CACHE_DIR}:")
    for i, (a, sr, t) in enumerate(items):
        print(f"  [{i}] {len(a)/sr:5.2f}s  sr={sr}  text={t[:60]!r}")
