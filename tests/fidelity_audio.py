"""Tiny WAV synthesizer for fidelity-corpus transport fixtures (E06).

Stdlib only (``wave`` + ``math`` + ``array``). Speech stand-ins are amplitude
modulated sine tones; silence is digital zero. This exists purely as
**transport and stitch evidence**: it proves that region annotations map to
frame ranges, that chunk splitting at silence boundaries is lossless, and
that reassembly restores the original span count. It is never ASR-quality
evidence — no recognizer is run on this audio, by design.
"""

from __future__ import annotations

import array
import io
import math
import sys
import wave
from typing import Any

SAMPLE_RATE = 16000
TONE_HZ = 220.0
ENVELOPE_HZ = 10.0
AMPLITUDE = 12000.0


def synthesize_samples(
    segments: list[dict[str, Any]], sample_rate: int = SAMPLE_RATE
) -> array.array:
    """int16 mono samples for the synthesis segments of a corpus fixture."""
    samples = array.array("h")
    for segment in segments:
        count = int(round(segment["duration_seconds"] * sample_rate))
        if segment["kind"] == "silence":
            samples.extend(array.array("h", bytes(2 * count)))
            continue
        for i in range(count):
            t = i / sample_rate
            envelope = 0.6 + 0.4 * math.sin(2 * math.pi * ENVELOPE_HZ * t)
            value = AMPLITUDE * envelope * math.sin(2 * math.pi * TONE_HZ * t)
            samples.append(int(max(-32768, min(32767, value))))
    return samples


def write_wav(
    fileobj: io.BytesIO, segments: list[dict[str, Any]], sample_rate: int = SAMPLE_RATE
) -> int:
    """Write the synthesized audio as a 16-bit mono WAV; returns frame count."""
    samples = synthesize_samples(segments, sample_rate)
    if sys.byteorder == "big":  # wave writes little-endian; array("h") is native
        samples = array.array("h", samples)
        samples.byteswap()
    writer = wave.open(fileobj, "wb")
    try:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    finally:
        writer.close()
    return len(samples)


def read_wav(
    fileobj: io.BytesIO, sample_rate: int = SAMPLE_RATE
) -> tuple[int, array.array]:
    """(frame_count, samples) read back from a 16-bit mono WAV."""
    reader = wave.open(fileobj, "rb")
    try:
        assert reader.getnchannels() == 1 and reader.getsampwidth() == 2
        rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
        assert rate == sample_rate
    finally:
        reader.close()
    samples = array.array("h", raw)
    if sys.byteorder == "big":
        samples.byteswap()
    return len(raw) // 2, samples


def speech_frame_bounds(
    segments: list[dict[str, Any]], sample_rate: int = SAMPLE_RATE
) -> list[tuple[int, int]]:
    """(start_frame, end_frame) for every speech segment, in order."""
    bounds = []
    frame = 0
    for segment in segments:
        count = int(round(segment["duration_seconds"] * sample_rate))
        if segment["kind"] == "speech":
            bounds.append((frame, frame + count))
        frame += count
    return bounds


def split_at_silences(
    samples: array.array, segments: list[dict[str, Any]], sample_rate: int = SAMPLE_RATE
) -> list[array.array]:
    """Split samples into per-speech-segment chunks (silence dropped).

    This is the transport-level stitching fixture: concatenating the chunks
    must restore exactly the speech frames of the original stream.
    """
    chunks = []
    for start, end in speech_frame_bounds(segments, sample_rate):
        chunks.append(samples[start:end])
    return chunks
