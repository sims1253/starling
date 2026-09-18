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
  invalidSessionWav,
  type GrantedWebLock,
  type WebLockRequestOptions,
  type WebLockSnapshot,
  type WebLocksLike,
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

/** Raw stored-record shape for fixtures that damage one field at a time. */
type DamagedStoredRecord = { wav?: Blob };

/** A representative v1 session record, as written before the journal stores existed. */
type VersionOneSession = {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: string;
  wav: Blob;
  durationMs?: number;
  attemptCount: number;
  transcript?: { text: string; segments: [] };
};

/**
 * Build a genuine version-1 database: `sessions` plus its indexes only, one
 * session inserted through the raw connection, exactly like the store did
 * before streaming captures raised the version to 2.
 */
async function openVersionOneSessionDatabase(
  factory: IDBFactory,
  databaseName: string,
  session: VersionOneSession,
): Promise<IDBDatabase> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = factory.open(databaseName, 1);

    request.onupgradeneeded = () => {
      const store = request.result.createObjectStore("sessions", { keyPath: "id" });

      store.createIndex("updatedAt", "updatedAt");
      store.createIndex("status", "status");
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction("sessions", "readwrite");

    transaction.objectStore("sessions").put(session);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });

  return database;
}

/** Versionless open attaches at whatever version the store last created. */
function openAtCurrentVersion(factory: IDBFactory, databaseName: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(databaseName);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
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

  it("skips the session write when the take is cancelled before finishing", async () => {
    // The #160 probe: a close-guard Discard that lands while the streamed
    // finalize is in flight must leave no session row behind.
    const store = new MemorySessionStore();
    const capture = await store.beginStreamCapture({ id: "discarded-live" });
    await capture.append(chunkOf(10));

    const finished = await capture.finish(250, () => true);

    assert.equal(finished.session, undefined);
    assert.equal(finished.failure, undefined);
    assert.equal(finished.durationMs, 250);
    assert.ok(finished.wav.size > 44);
    assert.equal(await store.get("discarded-live"), undefined);
    assert.equal((await store.list()).length, 0);
  });

  it("still persists the session when the take is not cancelled", async () => {
    const store = new MemorySessionStore();
    const capture = await store.beginStreamCapture({ id: "kept-live" });
    await capture.append(chunkOf(10));

    const finished = await capture.finish(250, () => false);

    assert.ok(finished.session);
    assert.equal(finished.session.id, "kept-live");
    assert.ok(await store.get("kept-live"));
  });

  it("clears the IndexedDB journal and writes no session once cancelled", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "stream-cancel-test", indexedDB: factory };
    const store = new IndexedDbSessionStore(options);
    const capture = await store.beginStreamCapture({ id: "discarded" });

    await capture.append(chunkOf(10));

    const finished = await capture.finish(250, () => true);

    assert.equal(finished.session, undefined);
    assert.equal(await store.get("discarded"), undefined);
    // The journal is gone, so the discarded take cannot resurrect on start.
    assert.equal((await store.recoverStreamCaptures()).length, 0);
    assert.equal(await store.get("discarded"), undefined);
    store.close();
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

  it("clears every journal row on abandon even after journaling failed mid-take", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "stream-abandon-test", indexedDB: factory };
    const store = new IndexedDbSessionStore(options);
    const capture = await store.beginStreamCapture({ id: "abandoned" });

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

      // The first chunk is still journaled; a discard must not resurrect it.
      await capture.abandon();
    } finally {
      IDBObjectStore.prototype.put = originalPut;
    }

    assert.equal((await store.recoverStreamCaptures()).length, 0);
    assert.equal(await store.get("abandoned"), undefined);
    store.close();
  });

  it("upgrades real version-1 databases to the journal schema without touching sessions", async () => {
    const factory = new IDBFactory();
    const databaseName = "v1-upgrade-test";

    const v1 = await openVersionOneSessionDatabase(factory, databaseName, {
      id: "v1-session",
      createdAt: "2025-09-01T10:00:00.000Z",
      updatedAt: "2025-09-01T10:00:01.000Z",
      status: "transcribed",
      wav,
      durationMs: 25,
      attemptCount: 1,
      transcript: { text: "written before streaming existed", segments: [] },
    });

    // The fixture is a genuine pre-journal database: version 1 with only the
    // sessions store, exactly as the store created it before the bump.
    assert.equal(v1.version, 1);
    assert.deepEqual([...v1.objectStoreNames], ["sessions"]);
    assert.deepEqual([...v1.transaction("sessions").objectStore("sessions").indexNames].sort(), [
      "status",
      "updatedAt",
    ]);
    v1.close();

    const store = new IndexedDbSessionStore({ databaseName, indexedDB: factory });
    const restored = await store.get("v1-session");

    assert.ok(restored);
    assert.equal(restored.createdAt, "2025-09-01T10:00:00.000Z");
    assert.equal(restored.updatedAt, "2025-09-01T10:00:01.000Z");
    assert.equal(restored.status, "transcribed");
    assert.equal(restored.durationMs, 25);
    assert.equal(restored.transcript?.text, "written before streaming existed");
    assert.deepEqual(
      new Uint8Array(await restored.wav.arrayBuffer()),
      new Uint8Array([82, 73, 70, 70]),
    );
    assert.equal(restored.streamed, undefined);
    assert.equal(restored.streamError, undefined);

    // The journal stores and the captureId index arrive with the upgrade.
    const upgraded = await openAtCurrentVersion(factory, databaseName);
    const upgradedChunks = upgraded.transaction("stream-chunks", "readonly");

    assert.equal(upgraded.version, 2);
    assert.ok(upgraded.objectStoreNames.contains("stream-captures"));
    assert.ok(upgraded.objectStoreNames.contains("stream-chunks"));
    assert.ok(upgradedChunks.objectStore("stream-chunks").indexNames.contains("captureId"));
    upgraded.close();

    // They also work: a capture journaled after the upgrade assembles into a
    // canonical WAV session through the new stores.
    const capture = await store.beginStreamCapture({ id: "post-upgrade" });

    await capture.append(chunkOf(1_000));

    const finished = await capture.finish();

    assert.ok(finished.session);
    assert.equal(finished.session.id, "post-upgrade");
    assert.equal(decodePcm16Wav(await wavBytes(finished.wav)).samples.length, 1_000);
    assert.equal((await store.get("post-upgrade"))?.status, "captured");
    store.close();
  });

  it("keeps a stalled v1 connection from blocking the upgrade silently", async () => {
    const factory = new IDBFactory();
    const databaseName = "v1-blocked-upgrade";

    const v1 = await openVersionOneSessionDatabase(factory, databaseName, {
      id: "stuck",
      createdAt: "2025-09-01T10:00:00.000Z",
      updatedAt: "2025-09-01T10:00:01.000Z",
      status: "captured",
      wav,
      attemptCount: 0,
    });

    // Deliberately no versionchange handler and no close: the open v1
    // connection pins the database at version 1.
    const store = new IndexedDbSessionStore({ databaseName, indexedDB: factory });

    await assert.rejects(
      store.get("stuck"),
      (cause) => cause instanceof DictationStorageError && /blocked/.test(cause.message),
    );
    v1.close();
    store.close();
  });

  it("reopens current-version databases without re-running the upgrade", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "stream-reopen-test", indexedDB: factory };
    const first = new IndexedDbSessionStore(options);

    await first.create({ id: "current-session", wav });
    first.close();

    // A same-version reopen must read plain session records back unchanged.
    const second = new IndexedDbSessionStore(options);
    const restored = await second.get("current-session");

    assert.ok(restored);
    assert.equal(restored?.streamed, undefined);
    assert.equal(restored?.streamError, undefined);
    assert.equal((await second.list()).length, 1);
    second.close();
  });
});

describe("damaged history isolation", () => {
  it("keeps healthy sessions visible while quarantining damaged records", async () => {
    const factory = new IDBFactory();
    const databaseName = "damaged-history";
    const store = new IndexedDbSessionStore({ databaseName, indexedDB: factory });

    await store.create({ id: "healthy-one", wav, durationMs: 10 });
    await store.create({ id: "healthy-two", wav, durationMs: 20 });
    const damaged = await store.create({ id: "damaged", wav, durationMs: 30 });

    await overwriteStoredSession(factory, databaseName, {
      ...damaged,
      status: "uploading",
    });

    const report = await store.listReport();

    assert.deepEqual(report.sessions.map((session) => session.id).sort(), [
      "healthy-one",
      "healthy-two",
    ]);
    assert.equal(report.invalid.length, 1);
    assert.equal(report.invalid[0]?.id, "damaged");

    const failure = report.invalid[0]?.cause;

    assert.ok(failure instanceof DictationStorageError);
    assert.ok(failure.cause instanceof Schema.SchemaError);

    // list() keeps resolving with the healthy history instead of rejecting.
    assert.deepEqual((await store.list()).map((session) => session.id).sort(), [
      "healthy-one",
      "healthy-two",
    ]);

    // Quarantine means neither deletion nor silent acceptance: the raw record
    // is still there and get() still refuses to decode it.
    await assert.rejects(
      store.get("damaged"),
      (cause) =>
        cause instanceof DictationStorageError && cause.cause instanceof Schema.SchemaError,
    );

    // Healthy entries keep their single-record reads working.
    assert.equal((await store.get("healthy-one"))?.id, "healthy-one");
    store.close();
  });

  it("reports damaged records with a missing field and unknown ids", async () => {
    const factory = new IDBFactory();
    const databaseName = "missing-wav-history";
    const store = new IndexedDbSessionStore({ databaseName, indexedDB: factory });
    const session = await store.create({ id: "missing-wav", wav });

    const withoutWav: DamagedStoredRecord = { ...session };

    delete withoutWav.wav;

    await overwriteStoredSession(factory, databaseName, withoutWav);
    // Empty string is a valid IndexedDB key but not a NonEmptyString id.
    await overwriteStoredSession(factory, databaseName, { id: "" });

    const report = await store.listReport();

    assert.equal(report.sessions.length, 0);
    assert.equal(report.invalid.length, 2);
    assert.deepEqual(report.invalid.map((entry) => entry.id).sort(), [
      "(unknown id)",
      "missing-wav",
    ]);

    // A record without a usable wav offers no audio to rescue.
    for (const entry of report.invalid) {
      assert.equal(invalidSessionWav(entry), undefined);
    }

    store.close();
  });

  it("keeps the damaged record's audio exportable for rescue", async () => {
    const factory = new IDBFactory();
    const databaseName = "rescuable-history";
    const store = new IndexedDbSessionStore({ databaseName, indexedDB: factory });
    const session = await store.create({ id: "rescuable", wav });

    // Damage a metadata field; the retained wav is untouched and identical.
    await overwriteStoredSession(factory, databaseName, {
      ...session,
      attemptCount: "not-a-number",
    });

    const [entry] = (await store.listReport()).invalid;

    assert.ok(entry);
    assert.equal(entry.id, "rescuable");

    // The raw record round-trips through IndexedDB, so rescue compares bytes.
    const rescued = invalidSessionWav(entry);

    assert.ok(rescued);
    assert.deepEqual(
      new Uint8Array(await rescued.arrayBuffer()),
      new Uint8Array(await wav.arrayBuffer()),
    );

    // Empty or missing blobs offer no audio worth offering.
    assert.equal(
      invalidSessionWav({ id: "x", cause: new Error(), record: { wav: new Blob() } }),
      undefined,
    );
    assert.equal(invalidSessionWav({ id: "y", cause: new Error(), record: {} }), undefined);
    store.close();
  });

  it("recovers orphaned journals even when a history record is damaged", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "recovery-vs-damage", indexedDB: factory };
    const first = new IndexedDbSessionStore(options);
    const capture = await first.beginStreamCapture({ id: "orphaned" });

    await capture.append(chunkOf(1_000));

    const damaged = await first.create({ id: "damaged", wav });

    await overwriteStoredSession(factory, options.databaseName, {
      ...damaged,
      status: "wedged",
    });
    first.close();

    const reopened = new IndexedDbSessionStore(options);

    // The damaged entry surfaces through listReport() instead of rejecting,
    // and the orphan journal still recovers into a retryable session.
    const report = await reopened.listReport();
    const recovered = await reopened.recoverStreamCaptures();

    assert.equal(report.invalid.length, 1);
    assert.equal(report.invalid[0]?.id, "damaged");
    assert.equal(recovered.length, 1);
    assert.equal(recovered[0]?.id, "orphaned");
    assert.equal(recovered[0]?.status, "failed");

    const restored = await reopened.get("orphaned");

    if (!restored) throw new Error("recovered session missing");

    assert.equal(decodePcm16Wav(await wavBytes(restored.wav)).samples.length, 1_000);
    reopened.close();
  });

  it("reports an empty invalid list for healthy memory-backed history", async () => {
    const store = new MemorySessionStore();

    await store.create({ id: "healthy", wav });

    const report = await store.listReport();

    assert.deepEqual(
      report.sessions.map((session) => session.id),
      ["healthy"],
    );
    assert.deepEqual(report.invalid, []);
  });
});

/**
 * In-memory stand-in for navigator.locks with the semantics the store relies
 * on: one exclusive holder per name, and `ifAvailable` reporting contention
 * as a null grant instead of queueing.
 */
class MemoryWebLocks implements WebLocksLike {
  private readonly held = new Set<string>();

  async request<Result>(
    name: string,
    options: WebLockRequestOptions,
    granted: (lock: GrantedWebLock | null) => Promise<Result> | Result,
  ): Promise<Result> {
    if (options.ifAvailable && this.held.has(name)) {
      return granted(null);
    }

    this.held.add(name);

    try {
      return await granted({ name });
    } finally {
      this.held.delete(name);
    }
  }

  async query(): Promise<WebLockSnapshot> {
    return Object.freeze({
      held: Object.freeze([...this.held].map((name) => Object.freeze({ name }))),
    });
  }
}

describe("cross-window capture ownership", () => {
  it("treats an explicit webLocks: undefined as the forced fallback, not an omitted option", async () => {
    const factory = new IDBFactory();
    const requested: string[] = [];
    const inner = new MemoryWebLocks();

    const hostLocks: WebLocksLike = {
      request: (name, options, granted) => {
        requested.push(name);

        return inner.request(name, options, granted);
      },
      query: () => inner.query(),
    };

    const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");

    Object.defineProperty(globalThis, "navigator", {
      value: { locks: hostLocks },
      configurable: true,
    });

    try {
      // Omitted property: the host's lock manager is used.
      const defaulted = new IndexedDbSessionStore({
        databaseName: "host-locks",
        indexedDB: factory,
      });

      const withHost = await defaulted.beginStreamCapture({ id: "host-owned" });

      await withHost.abandon();
      defaulted.close();

      assert.equal(requested.length > 0, true);

      // Explicitly undefined: the unlocked fallback runs even though the
      // host provides navigator.locks — the host manager stays untouched.
      requested.length = 0;

      const forced = new IndexedDbSessionStore({
        databaseName: "forced-fallback",
        indexedDB: factory,
        webLocks: undefined,
      });

      const unlocked = await forced.beginStreamCapture({ id: "unlocked-owned" });

      await unlocked.abandon();
      forced.close();

      assert.deepEqual(requested, []);
    } finally {
      if (navigatorDescriptor) {
        Object.defineProperty(globalThis, "navigator", navigatorDescriptor);
      } else {
        // SAFETY: the descriptor was absent, so this environment had no
        // navigator to begin with — removing the stub restores that.
        delete (globalThis as { navigator?: unknown }).navigator;
      }
    }
  });

  it("does not consume a journal another window is still recording into", async () => {
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-live", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);
    const capture = await owner.beginStreamCapture({ id: "owned-live" });

    await capture.append(chunkOf(1_000));

    const second = new IndexedDbSessionStore(options);

    // Tab B's startup recovery leaves the live journal alone: no session is
    // created under the capture id, and no journal row is removed.
    assert.deepEqual(await second.recoverStreamCaptures(), []);
    assert.equal(await second.get("owned-live"), undefined);

    // The owner keeps journaling and finishes with every frame durable.
    await capture.append(chunkOf(500, 2));

    const finished = await capture.finish();

    if (!finished.session) throw new Error("finish failed under a concurrent sweep");

    const decoded = decodePcm16Wav(await wavBytes(finished.session.wav));

    assert.equal(decoded.samples.length, 1_500);
    assert.equal(decoded.samples[999], 1 / 0x8000);
    assert.equal(decoded.samples[1_000], 2 / 0x8000);

    // History shows the owner's own take, not a recovery-marked partial.
    const listed = await second.listReport();

    assert.deepEqual(
      listed.sessions.map((session) => [session.id, session.status]),
      [["owned-live", "captured"]],
    );
    assert.deepEqual(await second.recoverStreamCaptures(), []);
    owner.close();
    second.close();
  });

  it("keeps hands off a live journal when web locks are unavailable", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "ownership-fallback", indexedDB: factory, webLocks: undefined };
    const owner = new IndexedDbSessionStore(options);
    const capture = await owner.beginStreamCapture({ id: "fallback-live" });

    await capture.append(chunkOf(1_000));

    // Unlocked fallback: ownership is tracked within the environment, which
    // still keeps two live stores from consuming each other's journals.
    const second = new IndexedDbSessionStore(options);

    assert.deepEqual(await second.recoverStreamCaptures(), []);
    assert.equal(await second.get("fallback-live"), undefined);

    await capture.append(chunkOf(500, 2));

    const finished = await capture.finish();

    if (!finished.session) throw new Error("finish failed in the unlocked fallback");

    assert.equal(decodePcm16Wav(await wavBytes(finished.session.wav)).samples.length, 1_500);
    owner.close();
    second.close();
  });

  it("recovers a journal once its owner is terminated", async () => {
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-terminated", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);
    const capture = await owner.beginStreamCapture({ id: "terminated" });

    await capture.append(chunkOf(1_000));
    await capture.append(chunkOf(500, 2));

    // The owning window goes away; its lock and registry claims end with it.
    owner.close();

    // A real window's lock release settles asynchronously with its close;
    // give the claim a tick to end before judging the journal abandoned.
    await new Promise((resolve) => setTimeout(resolve, 0));

    const second = new IndexedDbSessionStore(options);
    const recovered = await second.recoverStreamCaptures();

    assert.equal(recovered.length, 1);
    assert.equal(recovered[0]?.id, "terminated");
    assert.equal(recovered[0]?.status, "failed");
    assert.match(recovered[0]?.lastError ?? "", /Recovered after the app closed/);

    const restored = await second.get("terminated");

    if (!restored) throw new Error("recovered session missing");

    const decoded = decodePcm16Wav(await wavBytes(restored.wav));

    assert.equal(decoded.samples.length, 1_500);
    assert.equal(decoded.samples[999], 1 / 0x8000);
    assert.equal(decoded.samples[1_000], 2 / 0x8000);

    // The journal was consumed exactly once.
    assert.deepEqual(await second.recoverStreamCaptures(), []);
    second.close();
  });

  it("keeps an active empty capture discoverable instead of deleting it", async () => {
    const factory = new IDBFactory();
    const options = { databaseName: "ownership-empty", indexedDB: factory, webLocks: undefined };
    const owner = new IndexedDbSessionStore(options);
    const capture = await owner.beginStreamCapture({ id: "empty-live" });

    const second = new IndexedDbSessionStore(options);

    // Zero chunks, live owner: the metadata must survive so chunks appended
    // afterwards stay recoverable.
    assert.deepEqual(await second.recoverStreamCaptures(), []);

    await capture.append(chunkOf(1_000));
    owner.close();

    const third = new IndexedDbSessionStore(options);
    const recovered = await third.recoverStreamCaptures();

    assert.equal(recovered.length, 1);
    assert.equal(recovered[0]?.id, "empty-live");

    const restored = await third.get("empty-live");

    if (!restored) throw new Error("recovered session missing");

    assert.equal(decodePcm16Wav(await wavBytes(restored.wav)).samples.length, 1_000);
    second.close();
    third.close();
  });

  it("leaves a journal recoverable when promotion crashes mid-transaction", async () => {
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "promotion-crash", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);
    const capture = await owner.beginStreamCapture({ id: "crashed" });

    await capture.append(chunkOf(1_000));
    owner.close();

    const originalAdd = IDBObjectStore.prototype.add;

    IDBObjectStore.prototype.add = function patchedAdd(
      this: IDBObjectStore,
      value: { id?: string; status?: string },
    ): IDBRequest {
      const request = originalAdd.call(this, value);

      if (value?.id === "crashed" && value?.status === "failed") {
        // Die right after the recovered session was queued: the insert and
        // the journal delete must roll back together.
        this.transaction.abort();
      }

      return request;
    };

    const sweeper = new IndexedDbSessionStore(options);

    try {
      assert.deepEqual(await sweeper.recoverStreamCaptures(), []);
    } finally {
      IDBObjectStore.prototype.add = originalAdd;
    }

    // Neither half of the aborted promotion survived.
    assert.equal(await sweeper.get("crashed"), undefined);

    // The intact journal still promotes cleanly on the next sweep.
    const recovered = await sweeper.recoverStreamCaptures();

    assert.equal(recovered.length, 1);
    assert.equal(recovered[0]?.id, "crashed");

    const restored = await sweeper.get("crashed");

    if (!restored) throw new Error("recovered session missing");

    assert.equal(decodePcm16Wav(await wavBytes(restored.wav)).samples.length, 1_000);
    sweeper.close();
  });

  it("refuses to begin a capture another window already owns", async () => {
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-begin", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);

    await owner.beginStreamCapture({ id: "claimed" });

    const second = new IndexedDbSessionStore(options);

    await assert.rejects(
      second.beginStreamCapture({ id: "claimed" }),
      (cause) =>
        cause instanceof DictationStorageError && /owned by another window/.test(cause.message),
    );
    owner.close();
    second.close();
  });

  it("does not treat another window's in-flight transcription as interrupted", async () => {
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-attempt", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);

    await owner.create({ id: "attempt", wav });
    await owner.markAttempt("attempt");

    const second = new IndexedDbSessionStore(options);

    assert.equal(await second.transcriptionInFlight("attempt"), true);

    // What App.tsx's startup sweep does: only an owner whose signal is gone
    // gets interrupted, so the live attempt is left alone.
    if (!(await second.transcriptionInFlight("attempt"))) {
      await second.saveFailure("attempt", "Interrupted before the server returned a transcript.");
    }

    assert.equal((await second.get("attempt"))?.status, "transcribing");

    // The owner completes and its in-flight signal ends with the attempt.
    await owner.saveTranscript("attempt", { text: "done live", segments: [] });

    assert.equal(await second.transcriptionInFlight("attempt"), false);
    assert.equal((await second.get("attempt"))?.status, "transcribed");
    assert.equal((await second.get("attempt"))?.transcript?.text, "done live");
    owner.close();
    second.close();
  });

  it("interrupts a transcribing session once its owner is gone", async () => {
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-dead-attempt", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);

    await owner.create({ id: "late-attempt", wav });
    await owner.markAttempt("late-attempt");
    owner.close();

    // A real window's lock release settles asynchronously with its close;
    // give the claim a tick to end before judging the attempt abandoned.
    await new Promise((resolve) => setTimeout(resolve, 0));

    const second = new IndexedDbSessionStore(options);

    assert.equal(await second.transcriptionInFlight("late-attempt"), false);

    await second.saveFailure(
      "late-attempt",
      "Interrupted before the server returned a transcript.",
    );

    assert.equal((await second.get("late-attempt"))?.status, "failed");
    second.close();
  });

  it("keeps a contended second attempt visible after the lock winner settles", async () => {
    // Edge 1 (#162): two windows transcribe the same session. Lock
    // contention admits both, but only the winner holds a cross-window lock;
    // when the winner settles first, the loser's own signal must still read
    // as in-flight so no startup sweep marks it interrupted.
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-contended", indexedDB: factory, webLocks };
    const first = new IndexedDbSessionStore(options);
    const second = new IndexedDbSessionStore(options);

    await first.create({ id: "contended", wav });
    await first.markAttempt("contended");
    await second.markAttempt("contended");

    // The second window's own admission left the session transcribing; the
    // winner completing must not make the live contender look abandoned to
    // a fresh sweep window holding no signal of its own.
    await first.saveTranscript("contended", { text: "winner lands", segments: [] });

    const sweeper = new IndexedDbSessionStore(options);

    assert.equal(await sweeper.transcriptionInFlight("contended"), true);

    // What App.tsx's startup sweep does with a still-live signal: nothing —
    // the contender's session is never marked interrupted.
    if (!(await sweeper.transcriptionInFlight("contended"))) {
      await sweeper.saveFailure(
        "contended",
        "Interrupted before the server returned a transcript.",
      );
    }

    assert.equal((await sweeper.get("contended"))?.status, "transcribed");
    assert.equal((await sweeper.get("contended"))?.transcript?.text, "winner lands");

    await second.saveTranscript("contended", { text: "contender lands", segments: [] });

    assert.equal(await sweeper.transcriptionInFlight("contended"), false);
    assert.equal((await sweeper.get("contended"))?.transcript?.text, "contender lands");
    first.close();
    second.close();
    sweeper.close();
  });

  it("never lets a late interruption overwrite a settled transcript", async () => {
    // Edge 2 (#162): the sweep's transcriptionInFlight probe can pass while
    // a live owner still holds its signal, and the owner's saveTranscript
    // can then commit before the sweep's saveFailure. The marking write is
    // conditional, so the settled transcript survives the race either way.
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-settled-race", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);
    const sweeper = new IndexedDbSessionStore(options);

    await owner.create({ id: "racing", wav });
    await owner.markAttempt("racing");

    // The sweep probed while the owner was live, then the owner settled
    // before the marking write ran — the exact #162 interleave.
    assert.equal(await sweeper.transcriptionInFlight("racing"), true);
    await owner.saveTranscript("racing", { text: "settled first", segments: [] });

    const marked = await sweeper.saveFailure(
      "racing",
      "Interrupted before the server returned a transcript.",
    );

    assert.equal(marked.status, "transcribed");
    assert.equal(marked.transcript?.text, "settled first");

    const stored = await owner.get("racing");

    assert.equal(stored?.status, "transcribed");
    assert.equal(stored?.transcript?.text, "settled first");
    assert.equal(stored?.lastError, undefined);
    owner.close();
    sweeper.close();
  });

  it("releases the attempt signal when the settling write fails", async () => {
    const factory = new IDBFactory();
    const webLocks = new MemoryWebLocks();
    const options = { databaseName: "ownership-failed-save", indexedDB: factory, webLocks };
    const owner = new IndexedDbSessionStore(options);

    await owner.create({ id: "doomed", wav });
    await owner.markAttempt("doomed");

    assert.equal(await owner.transcriptionInFlight("doomed"), true);

    // The session disappears from another window (or the settling write
    // aborts): the attempt signal must still be released, or every other
    // window's liveness probe reports true forever.
    const second = new IndexedDbSessionStore(options);
    await second.delete("doomed");

    await assert.rejects(owner.saveTranscript("doomed", { text: "never lands", segments: [] }));

    assert.equal(await owner.transcriptionInFlight("doomed"), false);

    await assert.rejects(owner.saveFailure("doomed", "also never lands"));

    assert.equal(await owner.transcriptionInFlight("doomed"), false);
    owner.close();
    second.close();
  });
});
