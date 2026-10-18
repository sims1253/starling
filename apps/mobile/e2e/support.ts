import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

export const packageId = 'dev.starling.mobile.debug';
export const artifacts = resolve(import.meta.dirname, 'artifacts');
export const fixture = resolve(import.meta.dirname, '../../../tests/fixtures/2086-149220-0033.wav');
export const model = {
  name: 'parakeet-tdt-0.6b-v3-q4pw-fullimx.gguf',
  url: 'https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf/resolve/07008403b0ae5e49632e5d65df264933d494a281/parakeet-tdt-0.6b-v3-q4pw-fullimx.gguf',
  sha256: '6975b243883552a41abac56fd4021aef8ec57489de8fb8854c464fb41a551e26',
};
export const reference = "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.";

export function command(binary: string, args: string[], input?: Buffer): Buffer {
  return execFileSync(binary, args, { input, timeout: 120_000, maxBuffer: 64 * 1024 * 1024 });
}

export function adb(...args: string[]): Buffer {
  const serial = process.env.ANDROID_SERIAL;
  assert.match(serial ?? '', /^emulator-\d+$/, 'Select a dedicated emulator with ANDROID_SERIAL');
  return command('adb', ['-s', serial!, ...args]);
}

export function privateFile(name: string): Buffer {
  return adb('exec-out', 'run-as', packageId, 'cat', `files/${name}`);
}

export function hash(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

export function jsonFile(path: string, value: unknown): void {
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
}

export function pcm(file: string): Buffer {
  const probe = JSON.parse(command('ffprobe', [
    '-v', 'error', '-show_entries', 'stream=sample_rate,channels', '-of', 'json', file,
  ]).toString());
  assert.equal(probe.streams.length, 1);
  assert.equal(probe.streams[0].sample_rate, '16000');
  assert.equal(probe.streams[0].channels, 1);
  return command('ffmpeg', ['-v', 'error', '-i', file, '-f', 's16le', '-c:a', 'pcm_s16le', '-']);
}

export async function until<T>(read: () => T, ready: (value: T) => boolean, timeout = 120_000): Promise<T> {
  const deadline = Date.now() + timeout;
  let lastError: unknown;
  while (Date.now() < deadline) {
    try {
      const value = read();
      if (ready(value)) return value;
    } catch (error) { lastError = error; }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error(`Condition did not settle within ${timeout} ms`, { cause: lastError });
}

export interface Recording {
  id: string;
  status: string;
  duration_s: number;
  raw_transcript: string | null;
  revisions: { text: string; model: string }[];
}

export function recordings(): Recording[] {
  const names = adb('shell', 'run-as', packageId, 'ls', 'files/recordings').toString().trim().split(/\s+/);
  return names.filter(name => /^[a-f0-9-]+\.json$/.test(name))
    .map(name => JSON.parse(privateFile(`recordings/${name}`).toString()));
}

export function snapshot(label: string): Recording {
  const dir = resolve(artifacts, label);
  mkdirSync(dir, { recursive: true });
  const names = adb('shell', 'run-as', packageId, 'ls', 'files/recordings').toString().trim().split(/\s+/);
  for (const name of names.filter(name => /^[a-f0-9-]+\.(json|wav|flac)$/.test(name))) {
    writeFileSync(resolve(dir, name), privateFile(`recordings/${name}`));
  }
  const rows = readdirSync(dir).filter(name => name.endsWith('.json'));
  assert.equal(rows.length, 1, 'A clean run must contain exactly one recording');
  return JSON.parse(readFileSync(resolve(dir, rows[0]), 'utf8'));
}

export function verifyAudio(file: string, source: Buffer): Buffer {
  const audio = pcm(file);
  assert(audio.length >= source.length, 'Capture must contain the complete fixture');
  assert.equal(audio.length % 2, 0);
  for (let offset = 0; offset < audio.length; offset += source.length) {
    assert(audio.subarray(offset, offset + source.length).equals(source.subarray(0, Math.min(source.length, audio.length - offset))),
      `Captured samples differ from the test microphone at byte ${offset}`);
  }
  return audio;
}
