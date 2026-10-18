# Run the Android acceptance journey

This pilot drives the Android recorder through its UI with
[TesterArmy e2e](https://github.com/tester-army/e2e). It uses the existing debug
WAV microphone, the real on-device model, and the app's capture, storage, and
transcription paths. It requires no AI provider or API key.

## Failure cases

These cases define the checks before implementation:

- A missing device, fixture, model, or APK must fail setup, not skip the test.
- A stale app install or recording must not supply the expected result.
- Record must start capture, and Stop must finalize a nonempty recording.
- Inference must produce the fixture's speech, including its negation.
- Restart must retain the same recording, audio, and raw transcript.
- Export must produce playable audio whose decoded samples match capture.
- An injected transcription failure must retain audio for a successful UI retry.
- A failed test or evidence check must fail the command and CI job.
- Passing runs must retain evidence too, including inputs and a rerun command.

The fixture replaces microphone hardware only. This pilot does not verify a
physical microphone, keyboard insertion, interrupted capture, iOS, or GPUI
desktop. Existing device, recovery, and numerical tests remain in place.

## Run the journey

Use Node 24.8 or newer, Java 17, the Android SDK and NDK, `adb`, `curl`,
`ffmpeg`, and `ffprobe`. Boot a dedicated API 35 emulator with at least 4 GB
of RAM. The command clears **Starling Debug** data on that emulator. It refuses
physical devices. It overwrites this directory's `artifacts/` on each run.

1. From `apps/mobile`, build the debug APK:

   ```bash
   ./gradlew --no-daemon assembleDebug
   ```

2. Install the pinned test dependencies:

   ```bash
   cd e2e
   npm ci
   npm run check
   ```

3. Select the emulator and run the journey:

   ```bash
   ANDROID_SERIAL=emulator-5554 npm test
   ```

   With exactly one connected emulator, you can omit `ANDROID_SERIAL`.
   `STARLING_E2E_APK` selects another debug APK by absolute path. The runner
   downloads the pinned 364 MB model into `~/.cache/starling-e2e/`, checks its
   SHA-256, and copies it into the debug app. Later runs reuse the cache.

The test drives Record, Stop, Retry, restart, and Export through the UI. Setup
stages model and audio files through `run-as`. Read-only inspection checks the
saved recording without replacing capture, persistence, or inference. The debug
cadence hook defers live preview so CPU CI tests the batch retry path.

The Android emulator workflow runs this journey after the instrumented tests.
Both suites retain reports on success and failure. No provider credentials or
external AI judgments are used.

## Verify the evidence

A passing command writes `artifacts/manifest.json` with `status: "passed"`.
The bundle contains the input WAV, failed and successful recording snapshots,
the snapshot after restart, the exported audio, screenshots, video, runner
reports, and logcat. The manifest records the source commit, APK and model
hashes, device fingerprint, dependency versions, file hashes, and rerun command.
The model and APK are not copied into the bundle.

To independently verify a downloaded `android-acceptance-evidence` artifact,
run this command from `apps/mobile/e2e`:

```bash
npm run verify -- /absolute/path/to/extracted/artifact
```

The verifier checks file hashes, execution of the journey, transcript fidelity,
recording identity, persistence, duration, and every decoded audio sample. It
prints the recording ID, raw transcript, PCM hash, and sample count. Capture
duration depends on UI timing. Its samples must be an exact prefix of the
looped fixture, and all saved and exported versions must contain identical PCM.
This makes the assertions repeatable without requiring identical container
bytes, timestamps, or ASR punctuation on every machine.

To reproduce the run, check out the manifest's commit, build its APK, boot the
same API 35 emulator configuration, and run `npm ci && npm test` here. A dirty
working tree is recorded in the manifest. Hardware microphone and platform
coverage require separate device journeys.
