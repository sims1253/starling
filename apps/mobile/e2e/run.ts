import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { relative, resolve } from 'node:path';
import { adb, artifacts, command, fixture, hash, jsonFile, model, packageId, pcm, snapshot } from './support.ts';
import { verify } from './verify.ts';

process.chdir(import.meta.dirname);
process.env.E2E_TELEMETRY_DISABLED = '1';
rmSync(artifacts, { recursive: true, force: true });
mkdirSync(artifacts, { recursive: true });
const manifest: Record<string, unknown> = {
  schema: 1,
  status: 'setup',
  startedAt: new Date().toISOString(),
  model,
  rerun: 'cd apps/mobile/e2e && npm ci && npm test',
  verify: 'cd apps/mobile/e2e && npm ci && npm run verify -- /path/to/downloaded/artifacts',
  fixtureLicense: 'LibriSpeech 2086-149220-0033, CC BY 4.0, https://www.openslr.org/12',
};
let failure: unknown;
let staged = false;
try {
  const devices = command('adb', ['devices']).toString().split('\n')
    .map(line => line.trim().split(/\s+/)).filter(parts => parts[1] === 'device').map(parts => parts[0]);
  if (!process.env.ANDROID_SERIAL) {
    assert.equal(devices.length, 1, 'Boot one dedicated emulator or set ANDROID_SERIAL');
    process.env.ANDROID_SERIAL = devices[0];
  }
  assert(devices.includes(process.env.ANDROID_SERIAL!), 'Selected emulator is not online');
  manifest.serial = process.env.ANDROID_SERIAL;
  manifest.device = adb('shell', 'getprop', 'ro.build.fingerprint').toString().trim();
  manifest.abi = adb('shell', 'getprop', 'ro.product.cpu.abi').toString().trim();
  assert.equal(adb('shell', 'getprop', 'ro.build.version.sdk').toString().trim(), '35', 'Use the API 35 emulator');
  const apk = resolve(process.env.STARLING_E2E_APK ?? '../app/build/outputs/apk/debug/app-debug.apk');
  manifest.apkSha256 = hash(readFileSync(apk));
  manifest.commit = command('git', ['rev-parse', 'HEAD']).toString().trim();
  manifest.dirty = command('git', ['status', '--porcelain', '--untracked-files=no']).toString().trim() !== '';
  manifest.node = process.version;
  manifest.dependencies = JSON.parse(readFileSync('package.json', 'utf8')).devDependencies;
  manifest.lockSha256 = hash(readFileSync('package-lock.json'));
  copyFileSync(fixture, resolve(artifacts, 'input.wav'));
  assert(pcm(fixture).length > 0, 'Audio tools and fixture must work before changing the app');

  const cache = resolve(homedir(), '.cache/starling-e2e');
  mkdirSync(cache, { recursive: true });
  const modelPath = resolve(cache, model.name);
  if (!existsSync(modelPath) || hash(readFileSync(modelPath)) !== model.sha256) {
    execFileSync('curl', ['--fail', '--location', '--retry', '3', '--output', `${modelPath}.part`, model.url],
      { stdio: 'inherit', timeout: 600_000 });
    assert.equal(hash(readFileSync(`${modelPath}.part`)), model.sha256, 'Downloaded model checksum');
    renameSync(`${modelPath}.part`, modelPath);
  }
  adb('install', '-r', apk);
  assert.equal(adb('shell', 'pm', 'clear', packageId).toString().trim(), 'Success');
  adb('shell', 'mkdir', '-p', '/data/local/tmp/starling-e2e');
  adb('push', fixture, '/data/local/tmp/starling-e2e/input.wav');
  adb('push', modelPath, '/data/local/tmp/starling-e2e/model.gguf');
  adb('shell', 'run-as', packageId, 'mkdir', '-p', 'files/debug', 'files/models');
  adb('shell', 'run-as', packageId, 'cp', '/data/local/tmp/starling-e2e/input.wav', 'files/debug/test-mic.wav');
  adb('shell', 'run-as', packageId, 'cp', '/data/local/tmp/starling-e2e/model.gguf', `files/models/${model.name}`);
  assert.equal(adb('shell', 'run-as', packageId, 'sha256sum', `files/models/${model.name}`).toString().split(/\s+/)[0], model.sha256);
  adb('shell', 'run-as', packageId, 'touch', 'files/debug/fail-transcription');
  // Disable repeated live inference for this batch-retry journey on CPU CI.
  adb('shell', 'run-as', packageId, 'sh', '-c', "'echo 3600 3600 > files/debug/stream-cadence'");
  staged = true;
  adb('logcat', '-c');
  manifest.status = 'running';
  jsonFile(resolve(artifacts, 'manifest.json'), manifest);
  const result = spawnSync(process.execPath, ['node_modules/e2e/dist/cli/bin.js', 'run', '--reporter', 'list,junit'],
    { stdio: 'inherit', env: process.env, timeout: 900_000 });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `E2E runner exited with ${result.status} (${result.signal})`);
} catch (error) {
  failure = error;
  manifest.error = String(error);
} finally {
  if (staged) {
    try { writeFileSync(resolve(artifacts, 'logcat.txt'), adb('logcat', '-d', '-v', 'threadtime')); }
    catch (error) { manifest.logcatError = String(error); }
    if (failure) {
      try { snapshot('failure-state'); } catch (error) { manifest.snapshotError = String(error); }
    }
    try {
      adb('shell', 'am', 'force-stop', packageId);
      adb('shell', 'run-as', packageId, 'rm', '-f', 'files/debug/test-mic.wav', 'files/debug/fail-transcription', 'files/debug/stream-cadence');
      adb('shell', 'rm', '-rf', '/data/local/tmp/starling-e2e');
    } catch (error) { manifest.cleanupError = String(error); failure ??= error; }
  }
  manifest.finishedAt = new Date().toISOString();
  manifest.files = Object.fromEntries(readdirSync(artifacts, { recursive: true, withFileTypes: true })
    .filter(entry => entry.isFile() && entry.name !== 'manifest.json')
    .map(entry => {
      const path = resolve(entry.parentPath, entry.name);
      return [relative(artifacts, path), hash(readFileSync(path))];
    }));
  manifest.status = failure ? 'failed' : 'verifying';
  jsonFile(resolve(artifacts, 'manifest.json'), manifest);
}
if (!failure) {
  try {
    manifest.proof = verify(artifacts);
    manifest.status = 'passed';
  } catch (error) {
    failure = error;
    manifest.status = 'failed';
    manifest.error = String(error);
  }
  jsonFile(resolve(artifacts, 'manifest.json'), manifest);
}
console.log(`Evidence: ${artifacts}`);
if (failure) { console.error(failure); process.exitCode = 1; }
