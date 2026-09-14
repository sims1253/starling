import assert from "node:assert/strict";
import { describe, it } from "node:test";

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
