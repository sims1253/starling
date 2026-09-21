import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vite-plus/test";

import { TOKENIZER_ID } from "./insightEmitter";
import { DEFAULT_INSIGHT_CONSENT, type InsightConsent } from "./insightConsent";
import {
  IndexedDbInsightTermStore,
  InsightTermRecorder,
  InsightTermValidationError,
  MAX_TERMS_PER_KIND,
  MemoryInsightTermStore,
  captureTerms,
  insightTermRecordProblems,
  type InsightTermRecord,
} from "./insightTerms";

/**
 * Content-derived aggregate coverage (E29 phase 2): the record shape is the
 * privacy contract — short whitespace-free tokens with counts, and nothing
 * else fits — stores replace per capture, tombstone against resurrecting a
 * deleted take's phrases, consent gates every write, and withdrawal purges
 * what a grant had retained.
 */

const T0 = Date.parse("2026-09-20T12:00:00Z");

function recordOf(
  captureId: string,
  text: string,
  options: { readonly at?: number; readonly kinds?: ReadonlySet<"terms" | "phrases"> } = {},
): InsightTermRecord {
  const derived = captureTerms(text, { kinds: options.kinds });

  return {
    schema_version: 1,
    capture_id: captureId,
    occurred_at: new Date(options.at ?? T0).toISOString(),
    tokenizer: TOKENIZER_ID,
    terms: derived.terms,
    phrases: derived.phrases,
  };
}

describe("captureTerms", () => {
  it("counts word-like tokens and two/three-word phrases", () => {
    const derived = captureTerms("Deploy the server, deploy the server now");

    // case-folded vocabulary counts; short function words count too — the
    // declared vocabulary filter lives in the card layer, not here
    expect(derived.terms).toContainEqual({ text: "deploy", count: 2 });
    expect(derived.terms).toContainEqual({ text: "server", count: 2 });
    expect(derived.terms).toContainEqual({ text: "the", count: 2 });
    // bigrams and trigrams over consecutive word-like tokens
    expect(derived.phrases).toContainEqual({ text: "deploy the", count: 2 });
    expect(derived.phrases).toContainEqual({ text: "deploy the server", count: 2 });
    // punctuation is not word-like and never enters a phrase
    expect(derived.phrases.some((entry) => entry.text.includes(","))).toBe(false);
  });

  it("derives only the kinds the consent grants", () => {
    const kinds = new Set<"terms" | "phrases">(["terms"]);
    const derived = captureTerms("one two three", { kinds });

    expect(derived.terms.length).toBe(3);
    expect(derived.phrases).toEqual([]);
  });

  it("truncates transcripts whose aggregates exceed the bounded size", () => {
    // 5000 distinct tokens overrun MAX_TERMS_PER_KIND; the most frequent
    // entry must survive the deterministic truncation and the take must
    // never fail because of the term write.
    const many = [
      ...Array.from({ length: 4_999 }, (_, index) => `token${index}`),
      ...Array.from({ length: 100 }, () => "repeated"),
    ].join(" ");

    const derived = captureTerms(many);

    expect(derived.terms.length).toBe(MAX_TERMS_PER_KIND);
    expect(derived.terms.some((entry) => entry.text === "repeated")).toBe(true);
    expect(derived.terms.some((entry) => entry.text === "token0")).toBe(true);
    expect(derived.terms.some((entry) => entry.text === "token4998")).toBe(false);
  });

  it("store put still refuses a non-conforming record", async () => {
    // Derivation truncates instead of throwing, so the validation error now
    // guards only the store boundary against records that bypassed it.
    const store = new MemoryInsightTermStore();

    // SAFETY: deliberately non-conforming record (an unknown field) — the
    // JSON round-trip cast exists to bypass the type check precisely so
    // put's boundary validation can be exercised with this shape.
    const smuggled = JSON.parse(
      JSON.stringify({ ...recordOf("take-1", "hello world"), extra: "not allowed" }),
    ) as InsightTermRecord;

    await expect(store.put(smuggled)).rejects.toThrow(InsightTermValidationError);
  });
});

describe("insightTermRecordProblems", () => {
  it("accepts a well-formed record", () => {
    expect(insightTermRecordProblems(recordOf("take-1", "hello world"))).toEqual([]);
  });

  it("rejects a transcript smuggled into a term or an extra field", () => {
    const raw = {
      ...recordOf("take-1", "hello world"),
      terms: [{ text: "the full transcript text with spaces", count: 1 }],
    };

    expect(insightTermRecordProblems(raw).length).toBeGreaterThan(0);

    const poisoned = { ...recordOf("take-1", "hello world"), transcript: "hello world" };

    expect(insightTermRecordProblems(poisoned).length).toBeGreaterThan(0);
  });

  it("rejects unsorted or duplicated counts", () => {
    const base = recordOf("take-1", "alpha beta");

    // ["beta", "alpha"] violates the canonical sort order.
    expect(insightTermRecordProblems({ ...base, terms: [...base.terms].reverse() })).toContain(
      "terms are not sorted, unique and bounded",
    );

    const duplicated = {
      ...base,
      terms: [...base.terms, base.terms[0] ?? { text: "alpha", count: 1 }],
    };

    expect(insightTermRecordProblems(duplicated)).toContain(
      "terms are not sorted, unique and bounded",
    );
  });

  it("rejects zero or negative counts and non-token phrases", () => {
    const base = recordOf("take-1", "alpha beta");

    expect(
      insightTermRecordProblems({ ...base, terms: [{ text: "alpha", count: 0 }] }).length,
    ).toBe(1);

    expect(
      insightTermRecordProblems({ ...base, phrases: [{ text: "four word phrase here", count: 1 }] })
        .length,
    ).toBeGreaterThan(0);
  });
});

describe("MemoryInsightTermStore", () => {
  it("replaces a capture's record instead of adding a second one", async () => {
    const store = new MemoryInsightTermStore();

    await store.put(recordOf("take-1", "alpha beta alpha"));
    await store.put(recordOf("take-1", "gamma delta"));

    const log = await store.load();

    expect(log.records).toHaveLength(1);
    expect(log.records[0]?.terms).toContainEqual({ text: "gamma", count: 1 });
    expect(log.records[0]?.terms.some((entry) => entry.text === "alpha")).toBe(false);
  });

  it("tombstones a capture and refuses a late re-record", async () => {
    const store = new MemoryInsightTermStore();

    await store.put(recordOf("take-1", "alpha"));
    await store.tombstone("take-1");

    const afterDelete = await store.load();

    expect(afterDelete.records).toHaveLength(0);
    expect(afterDelete.tombstones).toEqual(["take-1"]);
    await expect(store.put(recordOf("take-1", "alpha"))).rejects.toThrow(/deleted/);
  });

  it("purges only the withdrawn kind and clears on demand", async () => {
    const store = new MemoryInsightTermStore();

    await store.put(recordOf("take-1", "alpha beta gamma"));

    await store.purgeKinds(new Set(["phrases"]));

    let log = await store.load();

    expect(log.records[0]?.phrases).toEqual([]);
    expect(log.records[0]?.terms?.length).toBe(3);

    await store.clear();
    log = await store.load();

    expect(log.records).toHaveLength(0);
    expect(log.tombstones).toHaveLength(0);
  });
});

describe("IndexedDbInsightTermStore", () => {
  it("persists records and tombstones across store instances", async () => {
    const factory = new IDBFactory();

    const first = new IndexedDbInsightTermStore({ databaseName: "terms-a", indexedDB: factory });

    await first.put(recordOf("take-1", "alpha beta"));
    await first.tombstone("take-2");

    const second = new IndexedDbInsightTermStore({ databaseName: "terms-a", indexedDB: factory });

    const log = await second.load();

    expect(log.records).toHaveLength(1);
    expect(log.invalidCount).toBe(0);
    expect(log.tombstones).toEqual(["take-2"]);
    await expect(second.put(recordOf("take-2", "alpha"))).rejects.toThrow(/deleted/);
  });

  it("quarantines damaged records instead of failing the whole log", async () => {
    const factory = new IDBFactory();
    const databaseName = "terms-b";
    const store = new IndexedDbInsightTermStore({ databaseName, indexedDB: factory });

    await store.put(recordOf("take-1", "alpha beta"));

    const poisoned: PoisonedTermRecord = {
      schema_version: 1,
      capture_id: "take-9",
      occurred_at: new Date(T0).toISOString(),
      tokenizer: TOKENIZER_ID,
      terms: [{ text: "a whole transcript cannot fit here because spaces", count: 1 }],
      phrases: [],
      transcript: "hello world",
    };

    await writeRawRecord(factory, databaseName, poisoned);

    const log = await store.load();

    expect(log.records).toHaveLength(1);
    expect(log.invalidCount).toBe(1);
  });
});

/** A term-record-shaped value plus one forbidden free-text field. */
interface PoisonedTermRecord {
  readonly schema_version: number;
  readonly capture_id: string;
  readonly occurred_at: string;
  readonly tokenizer: string;
  readonly terms: ReadonlyArray<{ readonly text: string; readonly count: number }>;
  readonly phrases: ReadonlyArray<{ readonly text: string; readonly count: number }>;
  readonly transcript: string;
}

/** Place a raw (unvalidated) record into the store's database, as damage would. */
async function writeRawRecord(
  factory: IDBFactory,
  databaseName: string,
  record: PoisonedTermRecord,
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const request = factory.open(databaseName, 1);

    request.onsuccess = () => {
      const database = request.result;
      const transaction = database.transaction("records", "readwrite");

      transaction.objectStore("records").put(record);
      transaction.oncomplete = () => {
        database.close();
        resolve();
      };

      transaction.onerror = () => reject(transaction.error);
    };

    request.onerror = () => reject(request.error);
  });
}

describe("InsightTermRecorder", () => {
  function recorderWith(consent: () => InsightConsent) {
    const store = new MemoryInsightTermStore();

    return {
      store,
      recorder: new InsightTermRecorder(store, {
        now: () => new Date(T0),
        consent,
      }),
    };
  }

  it("writes nothing while the content-derived consent is off", async () => {
    const { store, recorder } = recorderWith(() => DEFAULT_INSIGHT_CONSENT);

    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha beta" });

    expect((await store.load()).records).toHaveLength(0);
  });

  it("retains only the granted kind per write", async () => {
    const { store, recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      recurringPhrases: true,
    }));

    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha beta gamma" });

    const log = await store.load();

    expect(log.records[0]?.phrases.length).toBeGreaterThan(0);
    expect(log.records[0]?.terms).toEqual([]);
  });

  it("replaces a capture's aggregates on retranscription", async () => {
    const { store, recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      vocabularyPatterns: true,
    }));

    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha beta" });
    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "delta epsilon" });

    const log = await store.load();

    expect(log.records).toHaveLength(1);
    expect(log.records[0]?.terms.some((entry) => entry.text === "alpha")).toBe(false);
    expect(log.records[0]?.terms).toContainEqual({ text: "delta", count: 1 });
  });

  it("deletes the aggregates with the take, and dominates a stale re-record", async () => {
    const { store, recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      vocabularyPatterns: true,
    }));

    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha beta" });
    await recorder.captureDeleted("take-1");

    let log = await store.load();

    expect(log.records).toHaveLength(0);

    // A stale sync replay of the take's transcript cannot resurrect it.
    await expect(
      recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha beta" }),
    ).rejects.toThrow(/deleted/);

    log = await store.load();

    expect(log.records).toHaveLength(0);
  });

  it("purges retained data when a grant is withdrawn", async () => {
    let consent: InsightConsent = {
      ...DEFAULT_INSIGHT_CONSENT,
      recurringPhrases: true,
      vocabularyPatterns: true,
    };

    const { store, recorder } = recorderWith(() => consent);

    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha beta gamma" });

    consent = { ...consent, recurringPhrases: false };
    await recorder.withdrawKinds(new Set(["phrases"]));

    const log = await store.load();

    expect(log.records[0]?.phrases).toEqual([]);
    expect(log.records[0]?.terms.length).toBe(3);
  });
});
