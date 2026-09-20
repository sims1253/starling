import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "vite-plus/test";

import {
  AudioFormatError,
  decodePcm16Wav,
  encodeWav16k,
  prepareWav16k,
  STARLING_SAMPLE_RATE,
} from "../src/audio.js";

describe("canonical WAV preparation", () => {
  it("mixes stereo and resamples to the native server's required format", async () => {
    const frames = 4_800;
    const stereo = new Float32Array(frames * 2);

    for (let index = 0; index < frames; index += 1) {
      stereo[index * 2] = Math.sin(index / 10);
      stereo[index * 2 + 1] = Math.cos(index / 10);
    }

    const prepared = await prepareWav16k({ samples: stereo, sampleRate: 48_000, channels: 2 });
    const decoded = decodePcm16Wav(prepared.wav);
    assert.equal(decoded.sampleRate, STARLING_SAMPLE_RATE);
    assert.equal(decoded.channels, 1);
    assert.equal(decoded.samples.length, 1_600);
    assert.equal(prepared.durationSeconds, 0.1);
    assert.equal(prepared.blob.type, "audio/wav");
  });

  it("clamps non-finite and out-of-range samples", () => {
    const bytes = encodeWav16k({
      samples: Float32Array.from([-2, -1, Number.NaN, 1, 2]),
      sampleRate: 16_000,
    });

    const decoded = decodePcm16Wav(bytes);
    assert.deepEqual(
      [...decoded.samples].map((sample) => Math.round(sample * 32_768)),
      [-32_768, -32_768, 0, 32_767, 32_767],
    );
  });

  it("rejects truncated WAV chunks instead of treating them as PCM", () => {
    const bytes = encodeWav16k({ samples: new Float32Array([0, 0]), sampleRate: 16_000 });
    new DataView(bytes.buffer).setUint32(40, 2_000, true);
    assert.throws(() => decodePcm16Wav(bytes), AudioFormatError);
  });
});

describe("PCM16 rounding contract (shared fixture)", () => {
  it("matches the fixture also consumed by the Rust port", () => {
    // G07: this package is the semantic source of the quantization contract;
    // apps/desktop-gpui/crates/dictation runs the identical fixture. The
    // negative half-tie cases (e.g. -2^-16 scaling to exactly -0.5 -> 0)
    // fail under an implementation that rounds ties away from zero.
    const fixtureUrl = new URL(
      "../../../apps/desktop-gpui/test-fixtures/pcm-rounding.json",
      import.meta.url,
    );

    const fixture = parsePcmRoundingFixture(readFileSync(fixtureUrl, "utf8"));

    assert.ok(fixture.length >= 15, "fixture must keep its coverage");

    const samples = Float32Array.from(fixture.map((row) => row.input));

    const bytes = encodeWav16k({ samples, sampleRate: 16_000 });
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    fixture.forEach(({ input, expected }, index) => {
      assert.equal(view.getInt16(44 + index * 2, true), expected, `case ${index}: ${input}`);
    });
  });
});

// Parse the fixture at its I/O boundary: accept numeric inputs as-is and
// numeric strings (the JSON encoding of NaN / Infinity), reject anything
// else loudly instead of narrowing by representation at the use site.
function parsePcmRoundingFixture(raw: string): Array<{ input: number; expected: number }> {
  const parsed: unknown = JSON.parse(raw);

  if (typeof parsed !== "object" || parsed === null || !("cases" in parsed)) {
    throw new Error("pcm-rounding fixture: expected an object with a cases array");
  }

  const { cases } = parsed;

  if (!Array.isArray(cases)) {
    throw new Error("pcm-rounding fixture: cases must be an array");
  }

  return cases.map((entry) => {
    if (typeof entry !== "object" || entry === null || !("input" in entry) || !("expected" in entry)) {
      throw new Error("pcm-rounding fixture: each case needs input and expected");
    }

    const { input, expected } = entry;

    if (typeof input !== "number" && typeof input !== "string") {
      throw new Error("pcm-rounding fixture: input must be a number or numeric string");
    }

    if (typeof expected !== "number") {
      throw new Error("pcm-rounding fixture: expected must be a number");
    }

    return { input: Number(input), expected };
  });
}
