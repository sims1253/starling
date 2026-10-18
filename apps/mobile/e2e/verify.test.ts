import assert from 'node:assert/strict';
import { copyFileSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, resolve } from 'node:path';
import { test } from 'node:test';
import { command, fixture, hash, jsonFile, model, pcm, reference } from './support.ts';
import { verify } from './verify.ts';

// These fixtures exercise the evidence checker. Only the device journey proves
// app behavior. Mutations below must be rejected even with a fresh checksum.
test('independently verifies audio and rejects incomplete or altered evidence', () => {
  const dir = mkdtempSync(resolve(tmpdir(), 'starling-evidence-'));
  const id = 'a0000000-0000-0000-0000-000000000000';
  const audio = pcm(fixture);
  const row = { id, status: 'TRANSCRIBED', duration_s: audio.length / 32_000,
    raw_transcript: reference, revisions: [{ text: reference, model: model.name }] };
  const report = { run: { exitCode: 0, results: [{ status: 'passed' }] } };
  const manifest = { model, files: {} as Record<string, string> };
  const outside = resolve(dir, '..', `${basename(dir)}-outside.txt`);
  try {
    copyFileSync(fixture, resolve(dir, 'input.wav'));
    jsonFile(resolve(dir, 'journey.json'), { recordingId: id, transcript: reference });
    mkdirSync(resolve(dir, 'runner'));
    // Presence fixtures only. The E2E runner produces real images and video.
    writeFileSync(resolve(dir, 'runner/screen.png'), 'test screenshot');
    writeFileSync(resolve(dir, 'runner/video.mp4'), 'test video');
    jsonFile(resolve(dir, 'runner/report.json'), report);
    for (const phase of ['failed', 'transcribed', 'restarted']) {
      mkdirSync(resolve(dir, phase));
      jsonFile(resolve(dir, phase, `${id}.json`), phase === 'failed'
        ? { ...row, status: 'FAILED', raw_transcript: null, revisions: [] } : row);
      // Check WAV capture and FLAC retention with the independent decoder.
      if (phase === 'failed') copyFileSync(fixture, resolve(dir, phase, `${id}.wav`));
      else command('ffmpeg', ['-v', 'error', '-i', fixture, resolve(dir, phase, `${id}.flac`)]);
    }
    copyFileSync(resolve(dir, 'transcribed', `${id}.flac`), resolve(dir, 'export.flac'));
    manifest.files['input.wav'] = hash(readFileSync(resolve(dir, 'input.wav')));
    const reset = () => {
      jsonFile(resolve(dir, 'manifest.json'), manifest);
      jsonFile(resolve(dir, 'runner/report.json'), report);
      jsonFile(resolve(dir, 'journey.json'), { recordingId: id, transcript: reference });
      jsonFile(resolve(dir, 'restarted', `${id}.json`), row);
    };
    reset();
    assert.doesNotThrow(() => verify(dir));

    writeFileSync(outside, 'outside the evidence bundle');
    const escape = `../${basename(outside)}`;
    manifest.files[escape] = hash(readFileSync(outside));
    jsonFile(resolve(dir, 'manifest.json'), manifest);
    assert.throws(() => verify(dir), /escapes evidence directory/);
    delete manifest.files[escape];
    symlinkSync(outside, resolve(dir, 'outside-link'));
    manifest.files['outside-link'] = hash(readFileSync(outside));
    jsonFile(resolve(dir, 'manifest.json'), manifest);
    assert.throws(() => verify(dir), /escapes evidence directory/);
    delete manifest.files['outside-link'];
    reset();

    jsonFile(resolve(dir, 'runner/report.json'), { run: { exitCode: 0, results: [{ status: 'skipped' }] } });
    assert.throws(() => verify(dir), /Skipped or filtered/);
    reset();
    jsonFile(resolve(dir, 'restarted', `${id}.json`), { ...row, raw_transcript: 'lost text' });
    assert.throws(() => verify(dir), /Restart must retain/);
    reset();
    const altered = Buffer.from(readFileSync(resolve(dir, 'input.wav')));
    altered[altered.length - 2] ^= 1;
    writeFileSync(resolve(dir, 'input.wav'), altered);
    assert.throws(() => verify(dir), /hash differs/);
    // Even updating the hash cannot hide a change to the recorded samples.
    manifest.files['input.wav'] = hash(altered);
    jsonFile(resolve(dir, 'manifest.json'), manifest);
    assert.throws(() => verify(dir), /Captured samples differ/);
    copyFileSync(fixture, resolve(dir, 'input.wav'));
    manifest.files['input.wav'] = hash(readFileSync(fixture));
    reset();
    rmSync(resolve(dir, 'runner/video.mp4'));
    assert.throws(() => verify(dir), /Keep video/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outside, { force: true });
  }
});
