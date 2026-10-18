import assert from 'node:assert/strict';
import { readdirSync, readFileSync, realpathSync } from 'node:fs';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { artifacts, hash, model, pcm, reference, verifyAudio, type Recording } from './support.ts';

export function verify(directory: string): unknown {
  const root = realpathSync(directory);
  const inside = (name: string): string => {
    const contained = (path: string) => {
      const rel = relative(root, path);
      assert(rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel),
        `${name} escapes evidence directory`);
    };
    assert(!isAbsolute(name), `${name} escapes evidence directory`);
    const path = resolve(root, name);
    contained(path);
    const target = realpathSync(path);
    contained(target);
    return target;
  };
  const manifest = JSON.parse(readFileSync(inside('manifest.json'), 'utf8'));
  assert.deepEqual(manifest.model, model, 'Evidence must use the pinned real model');
  for (const [name, digest] of Object.entries(manifest.files as Record<string, string>)) {
    assert.equal(hash(readFileSync(inside(name))), digest, `${name} hash differs`);
  }
  const report = JSON.parse(readFileSync(inside('runner/report.json'), 'utf8'));
  assert.equal(report.run.exitCode, 0, 'E2E runner must pass');
  assert.equal(report.run.results.length, 1, 'Exactly one acceptance journey must execute');
  assert.equal(report.run.results[0].status, 'passed', 'Skipped or filtered tests are not proof');
  const evidence = readdirSync(inside('runner'), { recursive: true }).map(String);
  assert(evidence.some(name => name.endsWith('.png')), 'Keep screenshots on passing runs');
  assert(evidence.some(name => name.endsWith('.mp4')), 'Keep video on passing runs');
  const journey = JSON.parse(readFileSync(inside('journey.json'), 'utf8'));
  const source = pcm(inside('input.wav'));
  const phases = ['failed', 'transcribed', 'restarted'];
  const rows: Recording[] = [];
  const captured: Buffer[] = [];
  for (const phase of phases) {
    const dir = inside(phase);
    const files = readdirSync(dir);
    const metadata = files.filter(name => name.endsWith('.json'));
    assert.equal(metadata.length, 1);
    const row: Recording = JSON.parse(readFileSync(inside(`${phase}/${metadata[0]}`), 'utf8'));
    assert.equal(row.id, journey.recordingId);
    const audioFiles = files.filter(name => name === `${row.id}.wav` || name === `${row.id}.flac`);
    assert(audioFiles.length > 0, `${phase} must retain audio`);
    for (const name of audioFiles) {
      const audio = verifyAudio(inside(`${phase}/${name}`), source);
      assert.equal(audio.length / 32_000, row.duration_s, `${phase} duration must match samples`);
      captured.push(audio);
    }
    rows.push(row);
  }
  assert.equal(rows[0].status, 'FAILED');
  assert.equal(rows[0].raw_transcript, null);
  assert.equal(rows[0].revisions.length, 0);
  assert.equal(rows[1].status, 'TRANSCRIBED');
  assert.equal(rows[1].revisions.length, 1);
  assert.equal(rows[1].revisions[0].model, manifest.model.name);
  assert.equal(rows[1].raw_transcript, journey.transcript);
  assert.equal(rows[1].revisions[0].text, journey.transcript);
  assert.deepEqual(rows[2], rows[1], 'Restart must retain the same metadata');
  const normalize = (text: string) => text.toLowerCase().replace(/[^a-z']/g, '');
  assert(normalize(journey.transcript).includes(normalize(reference)), 'Real inference must recognize the complete reference');
  const exports = readdirSync(root).filter(name => /^export\.(wav|flac)$/.test(name));
  assert.equal(exports.length, 1);
  const exported = verifyAudio(inside(exports[0]), source);
  for (const audio of captured) assert(audio.equals(exported), 'Failed, retried, restarted, and exported audio must match');
  return { recordingId: journey.recordingId, transcript: journey.transcript, pcmSha256: hash(exported), samples: exported.length / 2 };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  console.log(JSON.stringify(verify(resolve(process.argv[2] ?? artifacts)), null, 2));
}
