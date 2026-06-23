"""Deterministic fixture generator for the starling baseline harness.

The correctness oracle and the RTF benchmark must be reproducible across runs and
machines, so this module builds the test utterances from a SINGLE downloaded sample
by plain concatenation (no RNG, no external corpora).

Fixtures (mono, 16 kHz, float32 in-memory / PCM_16 on disk):
    short  = the raw sample                    (~one utterance)
    medium = sample repeated 3x (no gap)
    long   = sample repeated 10x

The downloaded source sample lives next to this file and IS committed; the
regenerated {short,medium,long}.wav files are git-ignored (see ../.gitignore) but
kept under fixtures/ so they can be regenerated at any time via:

    uv run python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_WAV = Path(__file__).parent / "2086-149220-0033.wav"
SAMPLE_RATE = 16000

# (name, repeat_count). Deterministic -- no RNG anywhere.
FIXTURE_REPETITIONS = {
    "short": 1,
    "medium": 3,
    "long": 10,
}


def load_sample() -> np.ndarray:
    """Load the canonical source sample as float32 (mono, 16 kHz)."""
    audio, sr = sf.read(str(SAMPLE_WAV))
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz sample, got {sr} Hz")
    if audio.ndim != 1:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio, dtype=np.float32)


def make_fixtures(write: bool = True) -> dict[str, np.ndarray]:
    """Build the {short, medium, long} fixture arrays deterministically.

    Args:
        write: if True, also persist each fixture as PCM_16 .wav next to this file.
    """
    base = load_sample()
    fixtures: dict[str, np.ndarray] = {}
    for name, reps in FIXTURE_REPETITIONS.items():
        arr = np.concatenate([base] * reps)
        # Defensive normalization back into [-1, 1] range; the source is already
        # normalized but a no-op clip keeps concatenation artifacts impossible.
        arr = np.clip(arr, -1.0, 1.0).astype(np.float32)
        fixtures[name] = arr
        if write:
            out_path = Path(__file__).parent / f"{name}.wav"
            sf.write(str(out_path), arr, SAMPLE_RATE, subtype="PCM_16")
    return fixtures


def load_fixtures() -> dict[str, np.ndarray]:
    """Return the fixture arrays, regenerating the .wav files if any are missing."""
    fixtures: dict[str, np.ndarray] = {}
    regenerated = False
    for name in FIXTURE_REPETITIONS:
        wav_path = Path(__file__).parent / f"{name}.wav"
        if wav_path.exists():
            audio, sr = sf.read(str(wav_path))
            if sr != SAMPLE_RATE:
                raise ValueError(f"{wav_path}: expected {SAMPLE_RATE} Hz, got {sr}")
            fixtures[name] = np.ascontiguousarray(audio, dtype=np.float32)
        else:
            fixtures = make_fixtures(write=True)
            regenerated = True
            break
    if regenerated:
        return fixtures
    return fixtures


def build_batch(fixtures: dict[str, np.ndarray], batch_size: int) -> list[np.ndarray]:
    """Fill a batch of `batch_size` utterances by cycling [short, medium, long].

    Padding to the max length within the batch is handled later by the processor
    (padding="longest"); here we only select the per-element audio arrays.
    """
    order = ["short", "medium", "long"]
    return [fixtures[order[i % len(order)]] for i in range(batch_size)]


def build_uniform_batch(audio: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """A batch of `batch_size` copies of the same utterance (clean per-length scaling)."""
    return [audio for _ in range(batch_size)]


if __name__ == "__main__":
    fx = make_fixtures(write=True)
    for name, arr in fx.items():
        print(f"{name:7s}: {len(arr)} samples = {len(arr) / SAMPLE_RATE:.2f}s")
