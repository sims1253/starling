import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Schema } from "effect";
import { IDBFactory } from "fake-indexeddb";

import {
  DictationSessionNotFoundError,
  DictationStorageError,
  IndexedDbSessionStore,
  MemorySessionStore,
  exportDictationSession,
} from "../src/storage.js";

const wav = new Blob([new Uint8Array([82, 73, 70, 70]).buffer], { type: "audio/wav" });

async function overwriteStoredSession<Value>(
  factory: IDBFactory,
  databaseName: string,
  value: Value,
): Promise<void> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = factory.open(databaseName, 1);

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
});
