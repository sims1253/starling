import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vite-plus/test";

import basicSession from "../../../../packages/contracts/insight-events/fixtures/basic-session.json";
import deletionPropagation from "../../../../packages/contracts/insight-events/fixtures/deletion-propagation.json";
import generatedOutput from "../../../../packages/contracts/insight-events/fixtures/generated-output.json";
import negativeProxy from "../../../../packages/contracts/insight-events/fixtures/negative-proxy.json";
import syncReplay from "../../../../packages/contracts/insight-events/fixtures/sync-replay.json";
import timezoneShift from "../../../../packages/contracts/insight-events/fixtures/timezone-shift.json";
import {
  IndexedDbInsightEventStore,
  MemoryInsightEventStore,
  insightEventProblems,
  isCaptureFinalized,
  isRecognitionSelected,
  sameEventPayload,
  type CaptureFinalizedEvent,
  type InsightEvent,
  type RecognitionSelectedEvent,
} from "./insightEvents";

/**
 * Port of the E28 contract tests (`tests/test_insight_events.py`, schema
 * layer): every fixture event conforms to the frozen schema, and events
 * carrying raw-text/selection/path/window-title/secret-shaped extra fields
 * are REJECTED — the privacy contract is structural, not advisory. Plus the
 * store behaviors the idempotency rules imply: identical replays dedupe,
 * differing payloads under a known event_id conflict, damaged records
 * quarantine.
 */

function events(value: unknown): readonly InsightEvent[] {
  // SAFETY: fixture JSON from the frozen contract, validated by the tests
  // below before anything else reads it.
  return value as readonly InsightEvent[];
}

const BASIC = events(basicSession);

function basicEvent(index: number): InsightEvent {
  const event = BASIC[index];

  if (event === undefined) throw new Error("basic-session fixture shape changed");

  return event;
}

function eventOfKind(kind: InsightEvent["type"], pool: readonly InsightEvent[]): InsightEvent {
  const found = pool.find((event) => event.type === kind);

  if (found === undefined) throw new Error(`fixture lacks a ${kind} event`);

  return found;
}

function basicCaptureFinalized(): CaptureFinalizedEvent {
  const event = basicEvent(0);

  if (!isCaptureFinalized(event)) throw new Error("basic-session fixture shape changed");

  return event;
}

function basicRecognitionSelected(): RecognitionSelectedEvent {
  const event = eventOfKind("recognition_selected", BASIC);

  if (!isRecognitionSelected(event)) throw new Error("basic-session fixture shape changed");

  return event;
}

const FIXTURES = [
  ["basic-session", BASIC],
  ["sync-replay", events(syncReplay)],
  ["deletion-propagation", events(deletionPropagation)],
  ["timezone-shift", events(timezoneShift)],
  ["negative-proxy", events(negativeProxy)],
  ["generated-output", events(generatedOutput)],
] as const;

describe("schema conformance", () => {
  it("accepts every event of every fixture", () => {
    expect(FIXTURES.length).toBeGreaterThan(0);

    for (const [name, fixture] of FIXTURES) {
      expect(fixture.length, name).toBeGreaterThan(0);

      for (const event of fixture) {
        expect(insightEventProblems(event), `${name} ${event.event_id}`).toEqual([]);
      }
    }
  });

  it("rejects a missing required field", () => {
    const { sample_rate: _dropped, ...poisoned } = basicCaptureFinalized();

    expect(insightEventProblems(poisoned).length).toBeGreaterThan(0);
  });

  it("rejects an unknown event type", () => {
    expect(
      insightEventProblems({ ...basicCaptureFinalized(), type: "telemetry_uploaded" }).length,
    ).toBeGreaterThan(0);
  });

  it("rejects an unknown delivery status", () => {
    const delivery = eventOfKind("delivery_recorded", BASIC);

    expect(
      insightEventProblems({ ...delivery, status: "probably_inserted" }).length,
    ).toBeGreaterThan(0);
  });

  const FORBIDDEN_FIELDS = [
    "transcript",
    "raw_text",
    "selected_text",
    "clipboard_content",
    "file_path",
    "window_title",
    "app_title",
    "api_key",
    "oauth_token",
    "url",
  ] as const;

  for (const field of FORBIDDEN_FIELDS) {
    it(`rejects the forbidden extra field "${field}" on every event kind`, () => {
      // One representative event per kind: no event type may grow a free-form
      // string field carrying raw text, selections, paths, titles or secrets.
      const kinds: readonly InsightEvent[] = [
        eventOfKind("capture_finalized", BASIC),
        eventOfKind("recognition_selected", BASIC),
        eventOfKind("transformation_completed", BASIC),
        eventOfKind("delivery_recorded", BASIC),
        eventOfKind("capture_deleted", events(deletionPropagation)),
      ];

      for (const sample of kinds) {
        const poisoned = { ...sample, [field]: "C:/Users/me/secret-diary.txt — never share this" };

        expect(insightEventProblems(poisoned).length, sample.type).toBeGreaterThan(0);
      }
    });
  }

  it("treats payloads with reordered keys as the same payload", () => {
    const first = basicCaptureFinalized();
    const second = basicRecognitionSelected();
    const reordered: CaptureFinalizedEvent = {
      occurred_at: first.occurred_at,
      reporting_timezone: first.reporting_timezone,
      mode_id: first.mode_id,
      complete_audio: first.complete_audio,
      sample_rate: first.sample_rate,
      sample_count: first.sample_count,
      type: first.type,
      capture_id: first.capture_id,
      event_id: first.event_id,
      schema_version: first.schema_version,
    };

    expect(sameEventPayload(reordered, first)).toBe(true);
    expect(sameEventPayload({ ...second, event_id: first.event_id }, first)).toBe(false);
  });
});

describe("MemoryInsightEventStore", () => {
  it("dedupes identical replays and conflicts on differing payloads", async () => {
    const store = new MemoryInsightEventStore();
    const event = basicCaptureFinalized();

    await store.append(event);
    await store.append(event); // idempotent replay

    const replayed = await store.load();

    expect(replayed.events).toHaveLength(1);
    expect(replayed.invalidCount).toBe(0);

    await expect(store.append({ ...event, sample_count: 1 })).rejects.toThrow(
      /conflicting payloads/,
    );
  });

  it("persists tombstones and clears on demand", async () => {
    const store = new MemoryInsightEventStore();
    const tombstone = eventOfKind("capture_deleted", events(deletionPropagation));

    await store.append(tombstone);

    let log = await store.load();

    expect(log.events).toHaveLength(1);

    await store.clear();
    log = await store.load();

    expect(log.events).toHaveLength(0);
  });
});

describe("IndexedDbInsightEventStore", () => {
  it("persists events across store instances and dedupes replays", async () => {
    const factory = new IDBFactory();
    const first = new IndexedDbInsightEventStore({
      databaseName: "insights-a",
      indexedDB: factory,
    });
    const event = basicCaptureFinalized();

    await first.append(event);
    await first.append(event);

    const second = new IndexedDbInsightEventStore({
      databaseName: "insights-a",
      indexedDB: factory,
    });
    const log = await second.load();

    expect(log.events).toHaveLength(1);
    expect(log.events[0]?.event_id).toBe(event.event_id);
    expect(log.invalidCount).toBe(0);
  });

  it("conflicts on a differing payload under a known event id", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbInsightEventStore({
      databaseName: "insights-b",
      indexedDB: factory,
    });
    const event = basicRecognitionSelected();

    await store.append(event);

    await expect(store.append({ ...event, lexical_words: 5 })).rejects.toThrow(
      /conflicting payloads/,
    );
  });

  it("quarantines damaged records instead of failing the whole log", async () => {
    const factory = new IDBFactory();
    const databaseName = "insights-c";
    const store = new IndexedDbInsightEventStore({ databaseName, indexedDB: factory });
    const event = basicEvent(0);

    await store.append(event);
    await writeRawCaptureRecord(factory, databaseName, {
      event_id: "poisoned",
      capture_id: "x",
      occurred_at: "2026-09-20T12:00:00Z",
      type: "capture_finalized",
      sample_count: 1,
      sample_rate: 16000,
      complete_audio: true,
      mode_id: "faithful",
      reporting_timezone: "UTC",
      transcript: "raw text must never live in an event",
    });

    const log = await store.load();

    expect(log.events).toHaveLength(1);
    expect(log.events[0]?.event_id).toBe(event.event_id);
    expect(log.invalidCount).toBe(1);
  });

  it("clears the log for a reset", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbInsightEventStore({
      databaseName: "insights-d",
      indexedDB: factory,
    });

    await store.append(basicCaptureFinalized());
    await store.clear();

    const log = await store.load();

    expect(log.events).toHaveLength(0);
  });
});

/** A capture_finalized-shaped record plus one forbidden field. */
interface PoisonedCaptureRecord {
  event_id: string;
  capture_id: string;
  occurred_at: string;
  type: string;
  sample_count: number;
  sample_rate: number;
  complete_audio: boolean;
  mode_id: string;
  reporting_timezone: string;
  transcript: string;
}

/** Write a record into the events store directly, bypassing the append contract. */
function writeRawCaptureRecord(
  factory: IDBFactory,
  databaseName: string,
  record: PoisonedCaptureRecord,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = factory.open(databaseName, 1);

    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains("events")) {
        request.result.createObjectStore("events", { keyPath: "event_id" });
      }
    };
    request.onsuccess = () => {
      const database = request.result;
      const transaction = database.transaction("events", "readwrite");

      transaction.objectStore("events").put(record);
      transaction.oncomplete = () => {
        database.close();
        resolve();
      };
      transaction.onerror = () => reject(transaction.error ?? new Error("raw write failed"));
      transaction.onabort = () => reject(transaction.error ?? new Error("raw write aborted"));
    };
    request.onerror = () => reject(request.error ?? new Error("raw open failed"));
  });
}
