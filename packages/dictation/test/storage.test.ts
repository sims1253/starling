import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";
import { Schema } from "effect";
import { IDBFactory, IDBObjectStore } from "fake-indexeddb";

import {
  DictationSessionNotFoundError,
  DictationStorageError,
  IndexedDbSessionStore,
  MemorySessionStore,
  exportDictationSession,
} from "../src/storage.js";
import { decodePcm16Wav } from "../src/audio.js";

const wav = new Blob([new Uint8Array([82, 73, 70, 70]).buffer], { type: "audio/wav" });

async function overwriteStoredSession<Value>(
  factory: IDBFactory,
  databaseName: string,
  value: Value,
): Promise<void> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    // Versionless open attaches at whatever version the store created.
    const request = factory.open(databaseName);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction("sessions", "readwrite");

    transaction.objectStore("sessions").put(value);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });

  database.close();
}

describe("retry-safe session storage", () => {
  it("retains audio through attempts, failures, and successful retry", async () => {
    const store = new MemorySessionStore();
    const captured = await store.create({ id: "one", wav, durationMs: 100 });
    assert.equal(captured.status, "captured");

    const attempting = await store.markAttempt("one");
    assert.equal(attempting.attemptCount, 1);
    assert.equal(attempting.wav, wav);

    const failed = await store.saveFailure("one", new Error("offline"));
    assert.equal(failed.status, "failed");
    assert.equal(failed.wav, wav);

    await store.markAttempt("one");

    const complete = await store.saveTranscript("one", {
      text: "agreed",
      segments: [],
      requestId: "request-one",
    });

    assert.equal(complete.status, "transcribed");
    assert.equal(complete.attemptCount, 2);
    assert.equal(complete.wav, wav);
    assert.equal((await store.get("one"))?.wav, wav);

    await store.delete("one");
    assert.equal(await store.get("one"), undefined);
  });

  it("makes duplicate and unavailable persistence failures explicit", async () => {
    const memory = new MemorySessionStore();

    await memory.create({ id: "duplicate", wav });
    await assert.rejects(memory.create({ id: "duplicate", wav }), DictationStorageError);
    await assert.rejects(
      memory.markAttempt("missing"),
      (cause) =>
        cause instanceof DictationSessionNotFoundError && cause instanceof DictationStorageError,
    );

    const unavailable = new IndexedDbSessionStore({ indexedDB: undefined });

    await assert.rejects(unavailable.list(), /IndexedDB is unavailable/);
  });

  it("persists retained WAVs in IndexedDB across store instances", async () => {
    const indexedDB = new IDBFactory();
    const options = { databaseName: "dictation-test", indexedDB };
    const first = new IndexedDbSessionStore(options);
    await first.create({ id: "durable", wav, durationMs: 25 });
    await first.markAttempt("durable");
    await first.saveFailure("durable", "network unavailable");

    const reopened = new IndexedDbSessionStore(options);
    const restored = await reopened.get("durable");
    assert.ok(restored);
    assert.equal(restored?.status, "failed");
    assert.equal(restored?.attemptCount, 1);
    assert.deepEqual(
      new Uint8Array(await restored.wav.arrayBuffer()),
      new Uint8Array([82, 73, 70, 70]),
    );

    await reopened.delete("durable");
    assert.equal(await first.get("durable"), undefined);
    first.close();
    reopened.close();
  });

  it("retains the original recognition when a later retry changes meaning", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbSessionStore({ databaseName: "retry-history", indexedDB: factory });

    await store.create({ id: "revision", wav });
    await store.markAttempt("revision");
    await store.saveTranscript("revision", { text: "never merge this", segments: [] });
    await store.markAttempt("revision");
    await store.saveTranscript("revision", { text: "merge this", segments: [] });

    store.close();

    const reopened = new IndexedDbSessionStore({
      databaseName: "retry-history",
      indexedDB: factory,
    });

    const restored = await reopened.get("revision");

    assert.ok(restored);
    assert.equal(restored.transcript?.text, "merge this");
    assert.equal(restored.transcriptHistory?.[0]?.text, "never merge this");
    assert.equal(
      exportDictationSession(restored).manifest.transcriptHistory?.[0]?.text,
      "never merge this",
    );
    reopened.close();
  });

  it("rejects corrupted IndexedDB records at the schema boundary", async () => {
    const factory = new IDBFactory();
    const databaseName = "corrupted-session";
    const store = new IndexedDbSessionStore({ databaseName, indexedDB: factory });
    const session = await store.create({ id: "corrupted", wav });

    await overwriteStoredSession(factory, databaseName, {
      ...session,
      attemptCount: "not-a-number",
    });

    await assert.rejects(
      store.get("corrupted"),
      (cause) =>
        cause instanceof DictationStorageError && cause.cause instanceof Schema.SchemaError,
    );

    store.close();
  });

  it("exports a versioned manifest beside retained audio", async () => {
    const session = await new MemorySessionStore().create({ id: "export", wav });
    const bundle = exportDictationSession(session);
    assert.equal(bundle.manifest.schemaVersion, 1);
    assert.equal(bundle.manifest.audioFile, "recording.wav");
    assert.equal(bundle.wav, wav);
  });

  it("keeps streamed and streamError as optional review metadata", async () => {
    const memory = new MemorySessionStore();
    const session = await memory.create({ id: "streamed", wav });

    assert.equal(session.streamed, undefined);
    assert.equal(session.streamError, undefined);

    const noted = await memory.noteStreamError(session.id, "live transcription unavailable");
    assert.equal(noted.status, "captured");
    assert.equal(noted.streamError, "live transcription unavailable");

    await memory.markAttempt(session.id);

    const saved = await memory.saveTranscript(
      session.id,
      { text: "kept", segments: [] },
      { streamed: true },
    );

    assert.equal(saved.status, "transcribed");
    assert.equal(saved.streamed, true);
    assert.equal(saved.streamError, "live transcription unavailable");
    assert.equal(exportDictationSession(saved).manifest.streamed, true);
  });
});

function chunkOf(frames: number, fill = 1): Uint8Array {
  const bytes = new Uint8Array(frames * 2);
  const view = new DataView(bytes.buffer);

  for (let index = 0; index < bytes.byteLength; index += 2) {
    view.setInt16(index, fill, true);
  }

  return bytes;
}

async function wavBytes(wav: Blob): Promise<Uint8Array> {
  return new Uint8Array(await wav.arrayBuffer());
}

describe("streaming capture journal", () => {
  it("assembles journaled PCM16 chunks into a canonical WAV session", async () => {
    const store = new MemorySessionStore();
    const capture = await store.beginStreamCapture({ id: "live" });

    await capture.append(chunkOf(1_600));
    await capture.append(chunkOf(2_400, 2));
    assert.equal(capture.appendedFrames(), 4_000);

    const finished = await capture.finish(250);

    assert.ok(finished.session);
    assert.equal(finished.session.id, "live");
    assert.equal(finished.session.status, "captured");
    assert.equal(finished.durationMs, 250);
    assert.equal(finished.wav, finished.session.wav);

    const decoded = decodePcm16Wav(await wavBytes(finished.wav));
    assert.equal(decoded.sampleRate, 16_000);
    assert.equal(decoded.channels, 1);
    assert.equal(decoded.samples.length, 4_000);
    assert.equal(decoded.samples[0], 1 / 0x8000);
    assert.equal(decoded.samples[1_600], 2 / 0x8000);
  });

  it("keeps the assembled WAV when durable persistence fails", async () => {
    const store = new MemorySessionStore();
    const capture = await store.beginStreamCapture({ id: "parked" });
    await capture.append(chunkOf(10));

    // Consume the id so the finish create() collides, like a full store.
    await store.create({ id: "parked", wav });

    const finished = await capture.finish();

    assert.equal(finished.session, undefined);
    assert.ok(finished.failure instanceof DictationStorageError);
    assert.ok(finished.wav.size > 44);
  });

  it("abandons without creating a session", async () => {
    const store = new MemorySessionStore();
    const capture = await store.beginStreamCapture();
    await capture.append(chunkOf(10));
    await capture.abandon();

    assert.equal((await store.list()).length, 0);
    await assert.rejects(capture.append(chunkOf(2)), DictationStorageError);
  });

  it("rejects chunks that are not PCM16 frames", async () => {
    const store = new MemorySessionStore();
    const capture = await store.beginStreamCapture();

    await assert.rejects(capture.append(new Uint8Array(3)), TypeError);
    await assert.rejects(capture.append(new Uint8Array(0)), TypeError);
  });

  it("persists journals in IndexedDB and recovers orphaned captures as retryable sessions", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "stream-capture-test", indexedDB: factory };
    const first = new IndexedDbSessionStore(options);
    const capture = await first.beginStreamCapture({ id: "interrupted" });

    await capture.append(chunkOf(1_000));
    await capture.append(chunkOf(500, 2));
    first.close();
    // No finish(): the app died mid-recording.

    const reopened = new IndexedDbSessionStore(options);
    const recovered = await reopened.recoverStreamCaptures();

    assert.equal(recovered.length, 1);
    assert.equal(recovered[0]?.id, "interrupted");
    assert.equal(recovered[0]?.status, "failed");
    assert.match(recovered[0]?.lastError ?? "", /Recovered after the app closed/);

    const restored = await reopened.get("interrupted");

    if (!restored) throw new Error("recovered session missing");

    const decoded = decodePcm16Wav(await wavBytes(restored.wav));
    assert.equal(decoded.samples.length, 1_500);
    assert.equal(decoded.samples[999], 1 / 0x8000);
    assert.equal(decoded.samples[1_000], 2 / 0x8000);

    // Recovery consumes the journal; a second pass finds nothing.
    assert.equal((await reopened.recoverStreamCaptures()).length, 0);
    reopened.close();
  });

  it("clears the journal after a successful finish", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "stream-finish-test", indexedDB: factory };
    const store = new IndexedDbSessionStore(options);
    const capture = await store.beginStreamCapture({ id: "done" });

    await capture.append(chunkOf(10));
    const finished = await capture.finish();

    assert.ok(finished.session);
    assert.equal((await store.recoverStreamCaptures()).length, 0);
    assert.ok(await store.get("done"));
    store.close();
  });

  it("keeps assembling from memory when durable journaling fails mid-recording", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "stream-degraded-test", indexedDB: factory };
    const store = new IndexedDbSessionStore(options);
    const capture = await store.beginStreamCapture({ id: "degraded" });

    await capture.append(chunkOf(10));

    const originalPut = IDBObjectStore.prototype.put;

    IDBObjectStore.prototype.put = function patchedPut(
      this: IDBObjectStore,
      value: { captureId?: string },
    ): IDBRequest {
      if (value?.captureId) {
        throw new DOMException("Test journal is full", "QuotaExceededError");
      }

      return originalPut.call(this, value);
    };

    try {
      await assert.rejects(capture.append(chunkOf(10)), DictationStorageError);
      await capture.append(chunkOf(10, 2)); // memory-only after the first failure

      const finished = await capture.finish();

      assert.ok(finished.session);
      assert.equal(decodePcm16Wav(await wavBytes(finished.wav)).samples.length, 30);
    } finally {
      IDBObjectStore.prototype.put = originalPut;
      store.close();
    }
  });

  it("opens v1 databases at the journal schema version without touching sessions", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "stream-upgrade-test", indexedDB: factory };
    const first = new IndexedDbSessionStore(options);

    await first.create({ id: "v1-session", wav });
    first.close();

    // A store that predates streaming (opened before the version bump) wrote
    // plain v1 session records; reopening must read them back unchanged.
    const second = new IndexedDbSessionStore(options);
    const restored = await second.get("v1-session");

    assert.ok(restored);
    assert.equal(restored?.streamed, undefined);
    assert.equal(restored?.streamError, undefined);
    assert.equal((await second.list()).length, 1);
    second.close();
  });
});
