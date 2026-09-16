# `@starling/dictation`

Shared browser-compatible primitives for Starling's desktop app and browser preview.

The package uses TypeScript 7 and Effect v4. `StarlingClient` exposes
`transcribeEffect()`, `healthEffect()`, and `cancelEffect()` with typed failures
and cancellation. The Promise methods shown below remain available. Server
responses and saved IndexedDB sessions are decoded with Effect Schema.
See [workspace tooling](../../docs/typescript.md) for versions and checks.

The package keeps four boundaries explicit:

- audio is converted to mono 16 kHz PCM16 WAV before upload, so it works with
  both the Python and native servers;
- inference returns the server's raw text unchanged;
- any fidelity warning or edit is advisory data and never mutates raw text;
- every request is sent with `redirect: "manual"` and 3xx responses fail with
  the same "Server redirect blocked" wording as the Electron bridge, so audio
  and credentials are never re-sent to an origin the user did not configure.

Captured WAVs can be written to `IndexedDbSessionStore` before inference. The
store retains audio through attempts, failures, and successful transcription;
only `delete()` removes it.

```ts
import {
  IndexedDbSessionStore,
  StarlingClient,
  analyzeTranscript,
  prepareWav16k,
} from "@starling/dictation";

const wav = await prepareWav16k({ samples, sampleRate: audioContext.sampleRate });
const sessions = new IndexedDbSessionStore();
const session = await sessions.create({ wav: wav.blob, durationMs: wav.durationMs });

await sessions.markAttempt(session.id);
try {
  const result = await new StarlingClient({ baseUrl: "http://127.0.0.1:8181" }).transcribe(wav);
  await sessions.saveTranscript(session.id, result);
  const review = analyzeTranscript(result.text, { expectedTerms: ["auth"] });
  console.log(review.rawText, review.warnings);
} catch (error) {
  await sessions.saveFailure(session.id, error);
}
```
