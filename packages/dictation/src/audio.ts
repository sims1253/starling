import { Schema } from "effect";

export const STARLING_SAMPLE_RATE = 16_000;

export interface PcmAudio {
  /** Interleaved floating-point samples in the range -1..1. */
  readonly samples: Float32Array;
  readonly sampleRate: number;
  readonly channels?: number;
}

export interface PreparedWav {
  /** Canonical WAV bytes. `bytes` is retained as a descriptive alias. */
  readonly wav: Uint8Array;
  readonly bytes: Uint8Array;
  readonly blob: Blob;
  readonly sampleRate: typeof STARLING_SAMPLE_RATE;
  readonly channels: 1;
  readonly durationMs: number;
  readonly durationSeconds: number;
}

export type AudioSource = PcmAudio | Blob | ArrayBuffer | Uint8Array;

export class AudioFormatError extends Schema.TaggedError<AudioFormatError>()("AudioFormatError", {
  message: Schema.String,
}) {}

export const PcmAudioSchema = Schema.Struct({
  samples: Schema.instanceOf(Float32Array),
  sampleRate: Schema.Finite,
  channels: Schema.optionalKey(Schema.Finite),
});

function assertPcm(audio: PcmAudio): number {
  if (!Number.isInteger(audio.sampleRate) || audio.sampleRate <= 0) {
    throw new AudioFormatError({ message: "sampleRate must be a positive integer" });
  }

  const channels = audio.channels ?? 1;

  if (!Number.isInteger(channels) || channels < 1 || channels > 64) {
    throw new AudioFormatError({ message: "channels must be an integer between 1 and 64" });
  }

  if (audio.samples.length === 0) {
    throw new AudioFormatError({ message: "audio contains no samples" });
  }

  if (audio.samples.length % channels !== 0) {
    throw new AudioFormatError({
      message: "interleaved sample count is not divisible by channels",
    });
  }

  return channels;
}

/** Mix interleaved PCM to mono without modifying the source array. */
export function mixToMono(audio: PcmAudio): Float32Array {
  const channels = assertPcm(audio);

  if (channels === 1) return audio.samples.slice();

  const frames = audio.samples.length / channels;
  const mono = new Float32Array(frames);

  for (let frame = 0; frame < frames; frame += 1) {
    let sum = 0;

    for (let channel = 0; channel < channels; channel += 1) {
      sum += audio.samples[frame * channels + channel] ?? 0;
    }

    mono[frame] = sum / channels;
  }

  return mono;
}

/**
 * Resample interleaved floating-point PCM to mono 16 kHz.
 *
 * Linear interpolation is intentionally dependency-free and deterministic.
 * Native recording APIs should still request 16 kHz directly when possible.
 */
export function resampleTo16k(audio: PcmAudio): Float32Array {
  const mono = mixToMono(audio);

  if (audio.sampleRate === STARLING_SAMPLE_RATE) return mono;

  const outputLength = Math.max(
    1,
    Math.round((mono.length * STARLING_SAMPLE_RATE) / audio.sampleRate),
  );

  const output = new Float32Array(outputLength);
  const ratio = audio.sampleRate / STARLING_SAMPLE_RATE;

  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio;
    const left = Math.min(Math.floor(position), mono.length - 1);
    const right = Math.min(left + 1, mono.length - 1);
    const fraction = position - left;
    const leftSample = mono[left] ?? 0;
    output[index] = leftSample + ((mono[right] ?? leftSample) - leftSample) * fraction;
  }

  return output;
}

function pcm16(sample: number): number {
  const finite = Number.isFinite(sample) ? sample : 0;
  const clamped = Math.max(-1, Math.min(1, finite));

  return clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);
}

/** Encode arbitrary floating-point PCM as mono 16 kHz PCM16 WAV. */
export function encodeWav16k(audio: PcmAudio): Uint8Array {
  const samples = resampleTo16k(audio);
  const dataSize = samples.length * 2;

  if (dataSize > 0xffff_ffff - 36) {
    throw new AudioFormatError({ message: "audio is too large for a WAV file" });
  }

  const bytes = new Uint8Array(44 + dataSize);
  const view = new DataView(bytes.buffer);

  const writeAscii = (offset: number, value: string): void => {
    for (let index = 0; index < value.length; index += 1) {
      bytes[offset + index] = value.charCodeAt(index);
    }
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, STARLING_SAMPLE_RATE, true);
  view.setUint32(28, STARLING_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, dataSize, true);

  for (let index = 0; index < samples.length; index += 1) {
    view.setInt16(44 + index * 2, pcm16(samples[index] ?? 0), true);
  }

  return bytes;
}

interface DecodedWav extends PcmAudio {}

function ascii(bytes: Uint8Array, offset: number, length: number): string {
  let value = "";

  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index] ?? 0);
  }

  return value;
}

/** Decode the PCM16 WAV subset accepted by both serving backends. */
export function decodePcm16Wav(input: ArrayBuffer | Uint8Array): DecodedWav {
  const bytes =
    input instanceof Uint8Array
      ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
      : new Uint8Array(input);

  if (bytes.length < 44 || ascii(bytes, 0, 4) !== "RIFF" || ascii(bytes, 8, 4) !== "WAVE") {
    throw new AudioFormatError({ message: "expected a RIFF/WAVE file" });
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 12;
  let format: { channels: number; sampleRate: number; bitsPerSample: number } | undefined;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= bytes.length) {
    const id = ascii(bytes, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;

    if (size > bytes.length - body) {
      throw new AudioFormatError({ message: "WAV chunk exceeds the available payload" });
    }

    if (id === "fmt ") {
      if (size < 16) throw new AudioFormatError({ message: "WAV fmt chunk is truncated" });
      const encoding = view.getUint16(body, true);

      if (encoding !== 1) {
        throw new AudioFormatError({ message: "only PCM WAV audio is supported" });
      }

      format = {
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bitsPerSample: view.getUint16(body + 14, true),
      };
    } else if (id === "data") {
      dataOffset = body;
      dataSize = size;
      break;
    }

    offset = body + size + (size % 2);
  }

  if (!format || dataOffset < 0) {
    throw new AudioFormatError({ message: "WAV is missing fmt or data" });
  }

  if (format.bitsPerSample !== 16) {
    throw new AudioFormatError({ message: "only 16-bit WAV audio is supported" });
  }

  if (format.channels < 1 || format.channels > 64 || format.sampleRate < 1) {
    throw new AudioFormatError({ message: "WAV has invalid channel or sample-rate metadata" });
  }

  const bytesPerFrame = format.channels * 2;

  if (dataSize === 0 || dataSize % bytesPerFrame !== 0) {
    throw new AudioFormatError({ message: "WAV PCM data is empty or has an incomplete frame" });
  }

  const samples = new Float32Array(dataSize / 2);

  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(dataOffset + index * 2, true) / 0x8000;
  }

  return { samples, sampleRate: format.sampleRate, channels: format.channels };
}

/** Validate/decode an audio source and return the canonical upload format. */
export async function prepareWav16k(source: AudioSource): Promise<PreparedWav> {
  let pcm: PcmAudio;

  if (source instanceof Blob) {
    pcm = decodePcm16Wav(new Uint8Array(await source.arrayBuffer()));
  } else if (source instanceof Uint8Array) {
    pcm = decodePcm16Wav(source);
  } else if (source instanceof ArrayBuffer) {
    pcm = decodePcm16Wav(new Uint8Array(source));
  } else {
    pcm = source;
  }

  const encoded = encodeWav16k(pcm);
  const frameCount = (encoded.length - 44) / 2;

  const prepared: PreparedWav = {
    wav: encoded,
    bytes: encoded,
    blob: new Blob([encoded.slice().buffer], { type: "audio/wav" }),
    sampleRate: STARLING_SAMPLE_RATE,
    channels: 1,
    durationMs: (frameCount / STARLING_SAMPLE_RATE) * 1_000,
    durationSeconds: frameCount / STARLING_SAMPLE_RATE,
  };

  return Object.freeze(prepared);
}
