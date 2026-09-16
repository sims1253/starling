import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";

import {
  AudioFormatError,
  decodePcm16Wav,
  encodeWav16k,
  prepareWav16k,
  resampleTo16k,
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

function sine(
  sampleCount: number,
  sampleRate: number,
  frequency: number,
  amplitude = 0.5,
): Float32Array {
  const samples = new Float32Array(sampleCount);

  for (let index = 0; index < sampleCount; index += 1) {
    samples[index] = amplitude * Math.sin((2 * Math.PI * frequency * index) / sampleRate);
  }

  return samples;
}

// Amplitude of `frequency` in a mono signal via the Goertzel algorithm (the
// probe tones used here are far apart, so rectangular-window leakage is
// negligible at these lengths).
function amplitudeAt(samples: Float32Array, sampleRate: number, frequency: number): number {
  const coefficient = 2 * Math.cos((2 * Math.PI * frequency) / sampleRate);
  let s1 = 0;
  let s2 = 0;

  for (const sample of samples) {
    const s0 = sample + coefficient * s1 - s2;
    s2 = s1;
    s1 = s0;
  }

  const power = s1 * s1 + s2 * s2 - coefficient * s1 * s2;

  return (2 * Math.sqrt(Math.max(power, 0))) / samples.length;
}

describe("anti-aliased resampling (issue #122)", () => {
  // Stopband budget: an out-of-band tone must land at least 40 dB below the
  // 0.5 input amplitude after resampling (0.5 * 10^(-40/20) = 0.005). Plain
  // linear interpolation folded such tones into the recognition band at
  // full amplitude.
  const ALIAS_CEILING = 0.005;

  it("attenuates a 12 kHz tone that would fold to 4 kHz (48 kHz input)", () => {
    const output = resampleTo16k({ samples: sine(48_000, 48_000, 12_000), sampleRate: 48_000 });
    assert.equal(output.length, 16_000);
    const alias = amplitudeAt(output, STARLING_SAMPLE_RATE, 4_000);
    assert.ok(
      alias <= ALIAS_CEILING,
      `4 kHz alias amplitude ${alias} exceeds the 40 dB floor of ${ALIAS_CEILING}`,
    );
  });

  it("attenuates a 15 kHz tone that would fold to 1 kHz (44.1 kHz input)", () => {
    const output = resampleTo16k({ samples: sine(44_100, 44_100, 15_000), sampleRate: 44_100 });
    assert.ok(amplitudeAt(output, STARLING_SAMPLE_RATE, 1_000) <= ALIAS_CEILING);
  });

  it("keeps in-band tones at full level", () => {
    const at48k = resampleTo16k({ samples: sine(48_000, 48_000, 1_000), sampleRate: 48_000 });
    assert.ok(Math.abs(amplitudeAt(at48k, STARLING_SAMPLE_RATE, 1_000) - 0.5) <= 0.01);

    const at44k1 = resampleTo16k({ samples: sine(44_100, 44_100, 3_000), sampleRate: 44_100 });
    assert.ok(Math.abs(amplitudeAt(at44k1, STARLING_SAMPLE_RATE, 3_000) - 0.5) <= 0.01);
  });

  it("passes DC exactly", () => {
    const output = resampleTo16k({
      samples: new Float32Array(48_000).fill(0.5),
      sampleRate: 48_000,
    });

    for (const sample of output) assert.ok(Math.abs(sample - 0.5) <= 1e-6);
  });

  it("returns 16 kHz input untouched and still mixes channels", () => {
    const mono = sine(16_000, 16_000, 500);
    assert.deepEqual(
      Array.from(resampleTo16k({ samples: mono, sampleRate: 16_000 })),
      Array.from(mono),
    );

    const stereo = new Float32Array(2_000 * 2);

    for (let frame = 0; frame < 2_000; frame += 1) {
      stereo[frame * 2] = 0.25;
      stereo[frame * 2 + 1] = -0.25;
    }

    const mixed = resampleTo16k({ samples: stereo, sampleRate: 16_000, channels: 2 });
    assert.equal(mixed.length, 2_000);

    for (const sample of mixed) assert.ok(Math.abs(sample) <= 1e-6);
  });

  it("preserves duration and localizes an impulse", () => {
    // Odd input length exercises the output-length rounding.
    const odd = resampleTo16k({ samples: new Float32Array(10_001), sampleRate: 48_000 });
    assert.equal(odd.length, Math.round((10_001 * STARLING_SAMPLE_RATE) / 48_000));

    const impulse = new Float32Array(48_000);
    impulse[24_000] = 1;
    const output = resampleTo16k({ samples: impulse, sampleRate: 48_000 });
    let peak = 0;
    let peakIndex = -1;

    for (let index = 0; index < output.length; index += 1) {
      const value = output[index] ?? 0;
      assert.ok(Number.isFinite(value));

      if (Math.abs(value) > peak) {
        peak = Math.abs(value);
        peakIndex = index;
      }
    }

    assert.equal(peakIndex, 8_000); // 24_000 input samples / 3 = 8_000 output
    assert.ok(peak > 0 && peak <= 1);
  });
});
