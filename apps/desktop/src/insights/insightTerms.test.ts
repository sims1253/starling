import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vite-plus/test";

import { TOKENIZER_ID } from "./insightEmitter";
import { DEFAULT_INSIGHT_CONSENT, type InsightConsent } from "./insightConsent";
import {
  IndexedDbInsightTermStore,
  InsightTermRecorder,
  InsightTermTombstoneError,
  InsightTermValidationError,
  MAX_TERMS_PER_KIND,
  MemoryInsightTermStore,
  analyzedKinds,
  applyTermWrite,
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
    derived_kinds: derived.derived_kinds,
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

    // The stamp names exactly what was derived, sorted: a later reader can
    // tell an analyzed-but-empty kind from one nobody analyzed.
    expect(derived.derived_kinds).toEqual(["terms"]);
    expect(captureTerms("one two three").derived_kinds).toEqual(["phrases", "terms"]);
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

describe("analyzedKinds", () => {
  it("reads the stamp, with a stamped-but-empty kind still a real zero", () => {
    const stamped = recordOf("take-1", "hello world", {
      kinds: new Set(["terms"]),
    });

    expect(analyzedKinds(stamped)).toEqual({ terms: true, phrases: false });

    // A stamp with no labels at all is an analyzed take that found nothing.
    const analyzedEmpty = { ...stamped, terms: [], phrases: [] };

    expect(analyzedKinds(analyzedEmpty)).toEqual({ terms: true, phrases: false });
  });

  it("falls back to label evidence for unstamped legacy records", () => {
    const { derived_kinds: _stamp, ...legacy } = recordOf("take-1", "alpha beta gamma");

    expect(analyzedKinds(legacy)).toEqual({ terms: true, phrases: true });

    // Empty and unstamped is the one honest unknown: no stamp, no evidence.
    expect(analyzedKinds({ ...legacy, terms: [], phrases: [] })).toEqual({
      terms: false,
      phrases: false,
    });
  });
});

describe("insightTermRecordProblems", () => {
  it("accepts a well-formed record", () => {
    expect(insightTermRecordProblems(recordOf("take-1", "hello world"))).toEqual([]);
  });

  it("accepts a record without the derived_kinds stamp, rejects a bogus one", () => {
    const { derived_kinds: _stamp, ...legacy } = recordOf("take-1", "hello world");

    // Pre-stamp dev data still decodes — it is an unknown for per-kind
    // denominators, not damage — while a stamp naming an unknown kind is
    // exactly the closed-branch violation the schema exists to catch.
    expect(insightTermRecordProblems(legacy)).toEqual([]);

    // SAFETY: deliberately bogus stamp ("moods" is not a kind) — the JSON
    // round-trip cast bypasses the type check precisely so the schema
    // boundary can be exercised with a value the types forbid.
    const bogus = JSON.parse(
      JSON.stringify({ ...recordOf("take-1", "hello world"), derived_kinds: ["moods"] }),
    ) as InsightTermRecord;

    expect(insightTermRecordProblems(bogus).length).toBeGreaterThan(0);
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

  it("purges only the withdrawn kind, narrows its stamp, and clears on demand", async () => {
    const store = new MemoryInsightTermStore();

    await store.put(recordOf("take-1", "alpha beta gamma"));

    const purged = await store.purgeKinds(new Set(["phrases"]));

    let log = await store.load();

    expect(log.records[0]?.phrases).toEqual([]);
    expect(log.records[0]?.terms?.length).toBe(3);

    // The purged kind is no longer stamped: an unknown again, not a zero.
    expect(log.records[0]?.derived_kinds).toEqual(["terms"]);
    expect(purged[0]?.derived_kinds).toEqual(["terms"]);

    // A second withdrawal of the same kind changes nothing — the returned
    // delta stays honest to what actually changed.
    expect(await store.purgeKinds(new Set(["phrases"]))).toEqual([]);

    await store.clear();
    log = await store.load();

    expect(log.records).toHaveLength(0);
    expect(log.tombstones).toHaveLength(0);
  });

  it("stamps an unstamped record from what a purge leaves behind", async () => {
    const store = new MemoryInsightTermStore();

    // Pre-derived_kinds dev data: a withdrawal must not erase the
    // provenance of a kind the record still holds data for — the retained
    // aggregates are the only evidence it has, so they become the stamp,
    // while the emptied kind stays an unknown.
    const { derived_kinds: _stamp, ...legacy } = recordOf("take-1", "alpha beta gamma");

    await store.put(legacy);

    const purged = await store.purgeKinds(new Set(["phrases"]));

    expect(purged[0]?.derived_kinds).toEqual(["terms"]);

    const log = await store.load();

    expect(log.records[0]?.phrases).toEqual([]);
    expect(log.records[0]?.derived_kinds).toEqual(["terms"]);
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

  it("rejects a put over a tombstone with the tombstone error, deterministically", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbInsightTermStore({ databaseName: "terms-c", indexedDB: factory });

    await store.tombstone("take-1");

    // The abort the refusal triggers must not race a second, generic
    // "put was aborted" rejection onto the same promise: the failure reason
    // is the tombstone, every time.
    await expect(store.put(recordOf("take-1", "alpha"))).rejects.toBeInstanceOf(
      InsightTermTombstoneError,
    );
  });

  it("purges one transaction atomically and returns the rewritten records", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbInsightTermStore({ databaseName: "terms-d", indexedDB: factory });

    await store.put(recordOf("take-1", "alpha beta gamma"));

    const purged = await store.purgeKinds(new Set(["phrases"]));

    expect(purged).toHaveLength(1);
    expect(purged[0]?.phrases).toEqual([]);
    expect(purged[0]?.terms.length).toBe(3);

    const log = await store.load();

    expect(log.records[0]?.phrases).toEqual([]);
    expect(log.records[0]?.terms.length).toBe(3);
  });

  it("never reverts a concurrent re-put to the purge's stale snapshot", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbInsightTermStore({ databaseName: "terms-e", indexedDB: factory });

    await store.put(recordOf("take-1", "alpha beta gamma"));

    // The re-put is issued while the purge is in flight, between its read
    // and its write under the old two-transaction implementation — the
    // purge then re-persisted the stale pre-purge copy and the re-put's
    // content was lost. One readwrite cursor transaction cannot see or
    // clobber a write it never read.
    const purge = store.purgeKinds(new Set(["phrases"]));
    const reput = store.put(recordOf("take-1", "delta epsilon zeta"));

    await Promise.all([purge, reput]);

    const log = await store.load();

    expect(log.records[0]?.terms.some((entry) => entry.text === "delta")).toBe(true);
    expect(log.records[0]?.terms.some((entry) => entry.text === "alpha")).toBe(false);
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

  it("leaves damaged records untouched when the cursor purges", async () => {
    const factory = new IDBFactory();
    const databaseName = "terms-f";
    const store = new IndexedDbInsightTermStore({ databaseName, indexedDB: factory });

    await store.put(recordOf("take-1", "alpha beta gamma"));

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

    const purged = await store.purgeKinds(new Set(["terms"]));

    // The conforming record is purged; the damaged one is quarantined, not
    // repaired or deleted by the withdrawal.
    expect(purged).toHaveLength(1);
    expect(purged[0]?.capture_id).toBe("take-1");

    const log = await store.load();

    expect(log.invalidCount).toBe(1);
  });

  it("skips records the withdrawal does not change, keeping the delta honest", async () => {
    const factory = new IDBFactory();
    const store = new IndexedDbInsightTermStore({ databaseName: "terms-g", indexedDB: factory });

    await store.put(recordOf("take-1", "alpha beta gamma"));

    // First withdrawal rewrites the record (phrases held data); a second
    // finds nothing held and nothing stamped, so it rewrites and returns
    // nothing — no no-op IndexedDB writes, no phantom purge delta.
    expect(await store.purgeKinds(new Set(["phrases"]))).toHaveLength(1);
    expect(await store.purgeKinds(new Set(["phrases"]))).toEqual([]);

    const log = await store.load();

    expect(log.records[0]?.phrases).toEqual([]);
    expect(log.records[0]?.derived_kinds).toEqual(["terms"]);
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

    const write = await recorder.recognitionSelected({
      captureId: "take-1",
      transcriptText: "alpha beta",
    });

    // "none" is the honest outcome: no store I/O happened at all, so a
    // caller's mirror must not re-read on it either.
    expect(write).toEqual({ kind: "none" });
    expect((await store.load()).records).toHaveLength(0);
  });

  it("resolves with the written record so the mirror applies the delta", async () => {
    const { recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      vocabularyPatterns: true,
    }));

    const write = await recorder.recognitionSelected({
      captureId: "take-1",
      transcriptText: "alpha beta",
    });

    expect(write.kind).toBe("record");

    if (write.kind !== "record") return;

    expect(write.record.capture_id).toBe("take-1");
    expect(write.record.terms).toContainEqual({ text: "alpha", count: 1 });
  });

  it("anchors the record to the capture's finalization instant, not the write clock", async () => {
    const { store, recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      vocabularyPatterns: true,
    }));

    const finalizedAt = new Date(T0 - 14 * 24 * 60 * 60 * 1000).toISOString();

    await recorder.recognitionSelected({
      captureId: "take-1",
      transcriptText: "alpha beta",
      finalizedAt,
    });

    expect((await store.load()).records[0]?.occurred_at).toBe(finalizedAt);

    // Without a known finalization the write-time clock is the fallback.
    const fallback = await recorder.recognitionSelected({
      captureId: "take-2",
      transcriptText: "alpha",
    });

    if (fallback.kind !== "record") throw new Error("expected a record write");

    expect(fallback.record.occurred_at).toBe(new Date(T0).toISOString());
  });

  it("falls back to the write clock for an anchor that is not schema-shaped", async () => {
    const { store, recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      vocabularyPatterns: true,
    }));

    // A malformed anchor must cost its precision, never the whole write:
    // flowing it into occurred_at would fail the store's validation and
    // silently discard the take's aggregates.
    const write = await recorder.recognitionSelected({
      captureId: "take-1",
      transcriptText: "alpha beta",
      finalizedAt: "yesterday, probably",
    });

    if (write.kind !== "record") throw new Error("expected a record write");

    expect(write.record.occurred_at).toBe(new Date(T0).toISOString());
    expect((await store.load()).records).toHaveLength(1);
  });

  it("stamps the granted kinds on the written record", async () => {
    const { store, recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      recurringPhrases: true,
    }));

    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha beta" });

    expect((await store.load()).records[0]?.derived_kinds).toEqual(["phrases"]);
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
    const write = await recorder.withdrawKinds(new Set(["phrases"]));

    const log = await store.load();

    expect(log.records[0]?.phrases).toEqual([]);
    expect(log.records[0]?.terms.length).toBe(3);

    // The purge write names the records the store rewrote, stamp narrowed.
    expect(write.kind).toBe("purge");

    if (write.kind !== "purge") return;

    expect(write.records[0]?.phrases).toEqual([]);
    expect(write.records[0]?.derived_kinds).toEqual(["terms"]);
  });

  it("names the deletion and the reset in their writes", async () => {
    const { recorder } = recorderWith(() => ({
      ...DEFAULT_INSIGHT_CONSENT,
      vocabularyPatterns: true,
    }));

    await recorder.recognitionSelected({ captureId: "take-1", transcriptText: "alpha" });

    const deleted = await recorder.captureDeleted("take-1");
    const reset = await recorder.reset();

    expect(deleted).toEqual({ kind: "tombstone", captureId: "take-1" });
    expect(reset).toEqual({ kind: "clear" });
  });
});

describe("applyTermWrite", () => {
  it("upserts a record in place and appends a new capture", () => {
    const base = [recordOf("take-1", "alpha"), recordOf("take-2", "beta")];
    const revised = recordOf("take-1", "delta");
    const fresh = recordOf("take-3", "epsilon");

    const applied = applyTermWrite(applyTermWrite(base, { kind: "record", record: revised }), {
      kind: "record",
      record: fresh,
    });

    expect(applied.map((record) => record.capture_id)).toEqual(["take-1", "take-2", "take-3"]);
    expect(applied[0]?.terms).toContainEqual({ text: "delta", count: 1 });
  });

  it("drops the tombstoned capture, upserts a purge, clears on reset", () => {
    const base = [recordOf("take-1", "alpha beta gamma"), recordOf("take-2", "beta")];
    const original = base[0];

    if (original === undefined) throw new Error("fixture record missing");

    const withoutDeleted = applyTermWrite(base, { kind: "tombstone", captureId: "take-2" });

    expect(withoutDeleted.map((record) => record.capture_id)).toEqual(["take-1"]);

    const purged = applyTermWrite(withoutDeleted, {
      kind: "purge",
      records: [{ ...original, phrases: [] }],
    });

    expect(purged[0]?.phrases).toEqual([]);
    expect(applyTermWrite(purged, { kind: "clear" })).toEqual([]);

    // "none" changes nothing — the consent answer, not an omission.
    expect(applyTermWrite(purged, { kind: "none" })).toBe(purged);
  });
});
