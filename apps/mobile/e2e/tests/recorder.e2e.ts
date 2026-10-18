import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test } from '@e2e-dev/mobile';
import { expect } from 'e2e';
import { adb, artifacts, jsonFile, packageId, recordings, snapshot, until } from '../support.ts';

test('retains failed capture, retries with real inference, restarts, and exports', async ({ app, screen, device }) => {
  await app.open();
  const record = screen.getByTestId('record_button');
  await screen.scrollUntilVisible(record);
  await expect(screen.getByText('No recordings yet.')).toBeVisible();
  await record.tap();
  await expect(record).toHaveText('Stop and transcribe');
  const [take] = await until(recordings, rows => rows.length === 1);
  // Wait for durable samples, not a sleep that assumes capture started.
  await until(() => Number(adb('shell', 'run-as', packageId, 'stat', '-c', '%s', `files/recordings/${take.id}.wav.part`).toString().trim()),
    bytes => bytes >= 44 + 16_000 * 2 * 8);
  await record.tap();
  const status = screen.getByTestId('recording_status');
  await screen.scrollUntilVisible(status);
  await expect(status).toContainText('Injected transcription failure');
  await until(recordings, rows => rows[0]?.status === 'FAILED');
  // Audio upkeep may replace the WAV with FLAC between listing and reading.
  const failed = await until(() => snapshot('failed'), () => true);
  assert.equal(failed.raw_transcript, null);
  assert.equal(failed.revisions.length, 0);
  await app.screenshot('failed-with-audio-retained');

  adb('shell', 'run-as', packageId, 'rm', 'files/debug/fail-transcription');
  const retry = screen.getByTestId('retry_recording_button');
  await screen.scrollUntilVisible(retry);
  await retry.tap();
  await until(recordings, rows => rows[0]?.status === 'TRANSCRIBED', 300_000);
  const transcript = screen.getByTestId('recording_transcript');
  await screen.scrollUntilVisible(transcript, { direction: 'up' });
  await expect(transcript).toContainText(/don['’]t wish to see it any\s*more/i);
  const text = await transcript.textContent();
  const success = await until(() => snapshot('transcribed'), () => true);
  assert.equal(success.id, failed.id);
  assert.equal(success.raw_transcript, text);
  assert.equal(success.revisions.length, 1);
  await app.screenshot('real-transcript');

  await app.restart();
  await screen.scrollUntilVisible(transcript);
  await expect(transcript).toHaveText(text!);
  const restored = await until(() => snapshot('restarted'), () => true);
  assert.deepEqual(restored, success);
  await app.screenshot('retained-after-process-restart');

  const exportButton = screen.getByTestId('export_recording_button');
  await screen.scrollUntilVisible(exportButton);
  await exportButton.tap();
  const filename = screen.getByRole('textbox');
  await expect(filename).toBeVisible();
  const name = await filename.inputValue();
  assert.match(name, /^starling-[\d-]+\.(wav|flac)$/);
  // The dedicated API 35 emulator opens CREATE_DOCUMENT in Downloads.
  await device.dismissKeyboard();
  await screen.getByRole('button', /^save$/i).tap();
  await screen.scrollUntilVisible(screen.getByTestId('recording_message'), { direction: 'up' });
  await expect(screen.getByTestId('recording_message')).toHaveText('Recording exported.');
  const exportedPath = `/sdcard/Download/${name}`;
  writeFileSync(resolve(artifacts, `export.${name.split('.').at(-1)}`), adb('exec-out', 'cat', exportedPath));
  jsonFile(resolve(artifacts, 'journey.json'), { recordingId: take.id, transcript: text, exportedPath });
  await app.screenshot('export-complete');
});
