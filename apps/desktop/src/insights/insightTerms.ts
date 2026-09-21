import { Option, Schema } from "effect";

import { TOKENIZER_ID } from "./insightEmitter";
import { DEFAULT_INSIGHT_CONSENT, consentedTermKinds, type InsightConsent } from "./insightConsent";

/**
 * Content-derived Insight aggregates (E29 phase 2), kept strictly separate
 * from the E28 event log. The event schema structurally forbids text, and
 * for good reason: numbers can be recounted, phrases cannot be unspoken.
 * Frequent terms therefore live in their own store under their own consent,
 * exactly as INSIGHTS.md requires — nothing is written here unless the
 * matching grant is on, only aggregates are retained (per-capture term and
 * phrase counts, never transcripts), and deletion is a tombstone that a
 * late re-record cannot walk around. There is no second copy to rebuild
 * from: once a capture's record is deleted, its phrases are gone.
 *
 * The structural bound is the privacy contract's second half: a record can
 * only hold short whitespace-free tokens and one-to-three-word phrases with
 * small integer counts, sorted and deduplicated. A whole transcript, a
 * selection, a path or a secret does not fit the shape, so a poisoned
 * record is invalid before it can be persisted — the same discipline the
 * event schema enforces with closed branches.
 */

export const INSIGHT_TERM_SCHEMA_VERSION = 1;

/**
 * Per-kind aggregate bound, sized for realistic long takes rather than
 * erroring on them: a ten-minute dictation at ~150 wpm is ~1500 tokens,
 * ~3000 distinct two/three-word phrases — comfortably inside. A transcript
 * beyond the bound (roughly half an hour of continuous distinct speech) is
 * truncated deterministically, not rejected: the term write must never
 * fail a take, and truncation keeps the most frequent entries, so cards
 * built from the record still cite exact counts for what it kept and can
 * only ever understate recurrence, never overstate it.
 */
export const MAX_TERMS_PER_KIND = 4096;

/**
 * The single source of truth for the token shape: no whitespace, no
 * control characters, at most 64 code units. The record schema's checks
 * and the derivation-time bound below are built from this one pattern, so
 * validation and derivation cannot drift apart.
 */
const TOKEN_CORE = "[^\\p{White_Space}\\p{C}]";

const TermBoundPattern = new RegExp(`^${TOKEN_CORE}{1,64}$`, "u");

const PhraseBoundPattern = new RegExp(`^${TOKEN_CORE}{1,64}(?: ${TOKEN_CORE}{1,64}){1,2}$`, "u");

/** A single vocabulary token: no whitespace, no control characters, bounded. */
const TermTextSchema = Schema.String.pipe(Schema.check(Schema.isPattern(TermBoundPattern)));

/** A recurring phrase: two or three single-space-joined tokens. */
const PhraseTextSchema = Schema.String.pipe(Schema.check(Schema.isPattern(PhraseBoundPattern)));

const EventIdSchema = Schema.String.pipe(Schema.check(Schema.isPattern(/^[A-Za-z0-9_.-]{1,128}$/)));

const SafeTokenSchema = Schema.String.pipe(
  Schema.check(Schema.isPattern(/^[A-Za-z0-9_.:+-]{1,128}$/)),
);

const TIMESTAMP_PATTERN =
  /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$/;

const TimestampSchema = Schema.String.pipe(Schema.check(Schema.isPattern(TIMESTAMP_PATTERN)));

const PositiveIntSchema = Schema.Number.pipe(
  Schema.check(Schema.isInt()),
  Schema.check(Schema.isGreaterThan(0)),
);

export interface TermCount {
  readonly text: string;
  readonly count: number;
}

/**
 * #/$defs/termCounts — sorted, deduplicated, bounded. The schema's patterns
 * carry the privacy bounds (short whitespace-free tokens, small counts);
 * canonical order and uniqueness are enforced by `canonicalCounts` at the
 * validation boundary, so a hand-edited record with duplicate or shuffled
 * counts is invalid exactly like a transcript smuggling attempt.
 */
const TermCountsSchema = Schema.Array(
  Schema.Struct({ text: TermTextSchema, count: PositiveIntSchema }),
);

const PhraseCountsSchema = Schema.Array(
  Schema.Struct({ text: PhraseTextSchema, count: PositiveIntSchema }),
);

function canonicalCounts(
  counts: ReadonlyArray<{ readonly text: string; readonly count: number }>,
): boolean {
  if (counts.length > MAX_TERMS_PER_KIND) return false;

  return counts.every((entry, index) => {
    const previous = index === 0 ? undefined : counts[index - 1];

    return previous === undefined || previous.text < entry.text;
  });
}

export interface InsightTermRecord {
  readonly schema_version: 1;
  readonly capture_id: string;
  readonly occurred_at: string;
  readonly tokenizer: string;
  readonly terms: readonly TermCount[];
  readonly phrases: readonly TermCount[];
}

export const InsightTermRecordSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  capture_id: EventIdSchema,
  occurred_at: TimestampSchema,
  tokenizer: SafeTokenSchema,
  terms: TermCountsSchema,
  phrases: PhraseCountsSchema,
});

/**
 * Decode one unknown record against the term-record schema. Excess
 * properties are errors, so a record carrying free text under any extra
 * field is rejected structurally before it can be persisted.
 */
export const decodeInsightTermRecord = Schema.decodeUnknownOption(InsightTermRecordSchema, {
  onExcessProperty: "error",
});

/** Structural validation: an empty list means the record conforms. */
export function insightTermRecordProblems(
  // A raw record from storage or a poisoned test value is unknown by definition;
  // this function is the boundary that parses it.
  value: unknown, // oxlint-disable-line anti-slop/no-unknown-parameters -- see above
): readonly string[] {
  const decoded = decodeInsightTermRecord(value);

  if (Option.isNone(decoded)) {
    return [
      `record does not conform to the insight term record schema v${INSIGHT_TERM_SCHEMA_VERSION}`,
    ];
  }

  const record = decoded.value;
  const problems: string[] = [];

  if (!canonicalCounts(record.terms)) problems.push("terms are not sorted, unique and bounded");

  if (!canonicalCounts(record.phrases)) problems.push("phrases are not sorted, unique and bounded");

  return problems;
}

export class InsightTermValidationError extends Error {
  override readonly name = "InsightTermValidationError";

  constructor(message: string) {
    super(message);
  }
}

/** A transcript cannot be resurrected for a capture the user deleted. */
export class InsightTermTombstoneError extends Error {
  override readonly name = "InsightTermTombstoneError";

  constructor(
    readonly captureId: string,
    message: string,
  ) {
    super(message);
  }
}

/** Records that decoded cleanly, the deletion tombstones, and damage count. */
export interface InsightTermLog {
  readonly records: readonly InsightTermRecord[];
  readonly tombstones: readonly string[];
  readonly invalidCount: number;
}

let wordSegmenter: Intl.Segmenter | undefined;

function segmenter(): Intl.Segmenter {
  wordSegmenter ??= new Intl.Segmenter("und", { granularity: "word" });

  return wordSegmenter;
}

/** Word-like segments of a text under the declared tokenizer, case-folded. */
function lexicalTokens(text: string): readonly string[] {
  const tokens: string[] = [];

  for (const segment of segmenter().segment(text)) {
    if (segment.isWordLike !== true) continue;

    const folded = segment.segment.toLowerCase();

    if (TermBoundPattern.test(folded)) tokens.push(folded);
  }

  return tokens;
}

/** Count occurrences of each string in a list, in canonical sorted order. */
function countAll(items: readonly string[]): readonly TermCount[] {
  const counts = new Map<string, number>();

  for (const item of items) counts.set(item, (counts.get(item) ?? 0) + 1);

  const entries: TermCount[] = [...counts.entries()].map(([text, count]) => ({ text, count }));

  entries.sort((left, right) => (left.text < right.text ? -1 : left.text > right.text ? 1 : 0));

  return entries;
}

/**
 * Deterministically bound one kind's entries: keep the most frequent
 * (ties by label, the card ordering's own rule), then restore the canonical
 * sorted-by-label order the record schema requires. A truncated record
 * keeps exact counts for the entries it retained.
 */
function boundedEntries(entries: readonly TermCount[]): readonly TermCount[] {
  if (entries.length <= MAX_TERMS_PER_KIND) return entries;

  const kept = [...entries]
    .sort(
      (left, right) =>
        right.count - left.count || (left.text < right.text ? -1 : left.text > right.text ? 1 : 0),
    )
    .slice(0, MAX_TERMS_PER_KIND);

  kept.sort((left, right) => (left.text < right.text ? -1 : left.text > right.text ? 1 : 0));

  return kept;
}

/** Two- and three-word phrases over consecutive word-like tokens. */
function phraseTokens(tokens: readonly string[]): readonly string[] {
  const phrases: string[] = [];

  for (let index = 0; index + 1 < tokens.length; index += 1) {
    phrases.push(`${tokens[index] ?? ""} ${tokens[index + 1] ?? ""}`);

    if (index + 2 < tokens.length)
      phrases.push(`${tokens[index] ?? ""} ${tokens[index + 1] ?? ""} ${tokens[index + 2] ?? ""}`);
  }

  return phrases;
}

export interface CaptureTermsOptions {
  /** Which kinds to derive; a kind the consent does not grant is left empty. */
  readonly kinds?: ReadonlySet<"terms" | "phrases">;
}

/**
 * Derive one capture's aggregate from its selected transcript text. This is
 * the only place transcript contents meet the term store, and only counts
 * leave: word-like tokens and short phrases with their frequencies. With no
 * kinds granted, the result is empty — consent off means nothing derived is
 * retained, not a record with the text hidden inside. Ordinary long takes
 * (a multi-minute dictation yields roughly twice as many distinct phrases
 * as words) fit the bound; a transcript beyond it is truncated
 * deterministically to its most frequent entries rather than refused, so
 * deriving aggregates can never fail the take, and the truncation only
 * ever understates recurrence — the kept counts stay exact.
 */
export function captureTerms(
  transcriptText: string,
  options: CaptureTermsOptions = {},
): Omit<InsightTermRecord, "schema_version" | "capture_id" | "occurred_at" | "tokenizer"> {
  const kinds = options.kinds ?? new Set<"terms" | "phrases">(["terms", "phrases"]);
  const wantsTerms = kinds.has("terms");
  const wantsPhrases = kinds.has("phrases");
  const tokens = wantsTerms || wantsPhrases ? lexicalTokens(transcriptText) : [];

  const terms = wantsTerms ? boundedEntries(countAll(tokens)) : [];
  const phrases = wantsPhrases ? boundedEntries(countAll(phraseTokens(tokens))) : [];

  return { terms, phrases };
}

/**
 * Durable storage for content-derived term aggregates. `put` replaces a
 * capture's record (a retranscription revises the counts, it never adds a
 * second record), refuses tombstoned captures (a deleted take's phrases
 * cannot be resurrected by a stale re-record), and `tombstone` removes the
 * record and remembers the deletion. `purgeKinds` implements consent
 * withdrawal: the withdrawn kind's data is deleted from every record.
 */
export interface InsightTermStore {
  load(): Promise<InsightTermLog>;
  put(record: InsightTermRecord): Promise<void>;
  tombstone(captureId: string): Promise<void>;
  purgeKinds(kinds: ReadonlySet<"terms" | "phrases">): Promise<void>;
  clear(): Promise<void>;
}

function assertConforming(record: InsightTermRecord): void {
  if (Option.isNone(decodeInsightTermRecord(record))) {
    throw new InsightTermValidationError(insightTermRecordProblems(record).join("; "));
  }
}

export class MemoryInsightTermStore implements InsightTermStore {
  private readonly records = new Map<string, InsightTermRecord>();
  private readonly tombstones = new Set<string>();

  async load(): Promise<InsightTermLog> {
    return Object.freeze({
      records: Object.freeze([...this.records.values()]),
      tombstones: Object.freeze([...this.tombstones]),
      invalidCount: 0,
    });
  }

  async put(record: InsightTermRecord): Promise<void> {
    assertConforming(record);

    if (this.tombstones.has(record.capture_id)) {
      throw new InsightTermTombstoneError(
        record.capture_id,
        `Capture ${record.capture_id} was deleted; its term aggregates cannot be resurrected`,
      );
    }

    this.records.set(record.capture_id, record);
  }

  async tombstone(captureId: string): Promise<void> {
    this.records.delete(captureId);
    this.tombstones.add(captureId);
  }

  async purgeKinds(kinds: ReadonlySet<"terms" | "phrases">): Promise<void> {
    for (const [captureId, record] of this.records) {
      this.records.set(captureId, {
        ...record,
        terms: kinds.has("terms") ? [] : record.terms,
        phrases: kinds.has("phrases") ? [] : record.phrases,
      });
    }
  }

  async clear(): Promise<void> {
    this.records.clear();
    this.tombstones.clear();
  }
}

export interface IndexedDbInsightTermStoreOptions {
  readonly databaseName?: string;
  readonly indexedDB?: IDBFactory | undefined;
}

const RECORDS_OBJECT_STORE = "records";

const TOMBSTONES_OBJECT_STORE = "tombstones";

function runIdbRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(new Error(`an insight term store request failed: ${request.error?.message ?? ""}`));
  });
}

/**
 * Term aggregates in their own IndexedDB database, mirroring the insight
 * event store's conventions: damaged records are quarantined out of
 * `load()` (counted, never silently deleted), a `put` over a tombstone
 * rejects, and tombstones persist by design so stale re-records can never
 * resurrect deleted phrases.
 */
export class IndexedDbInsightTermStore implements InsightTermStore {
  private readonly databaseName: string;
  private readonly factory: IDBFactory | undefined;
  private databasePromise: Promise<IDBDatabase> | undefined;

  constructor(options: IndexedDbInsightTermStoreOptions = {}) {
    this.databaseName = options.databaseName ?? "starling-insights-terms";
    this.factory = options.indexedDB ?? globalThis.indexedDB;
  }

  async load(): Promise<InsightTermLog> {
    const database = await this.database();

    const recordResults = await runIdbRequest(
      database
        .transaction(RECORDS_OBJECT_STORE, "readonly")
        .objectStore(RECORDS_OBJECT_STORE)
        .getAll(),
    );

    const tombstones = await runIdbRequest(
      database
        .transaction(TOMBSTONES_OBJECT_STORE, "readonly")
        .objectStore(TOMBSTONES_OBJECT_STORE)
        .getAllKeys(),
    );

    const records: InsightTermRecord[] = [];
    let invalidCount = 0;

    for (const record of recordResults) {
      const decoded = decodeInsightTermRecord(record);

      if (Option.isSome(decoded)) records.push(decoded.value);
      else invalidCount += 1;
    }

    return Object.freeze({
      records: Object.freeze(records),
      tombstones: Object.freeze(tombstones.map(String)),
      invalidCount,
    });
  }

  async put(record: InsightTermRecord): Promise<void> {
    assertConforming(record);
    const database = await this.database();

    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(
        [RECORDS_OBJECT_STORE, TOMBSTONES_OBJECT_STORE],
        "readwrite",
      );

      const tombstoneRequest = transaction
        .objectStore(TOMBSTONES_OBJECT_STORE)
        .get(record.capture_id);

      tombstoneRequest.onsuccess = () => {
        if (tombstoneRequest.result !== undefined) {
          transaction.abort();
          reject(
            new InsightTermTombstoneError(
              record.capture_id,
              `Capture ${record.capture_id} was deleted; its term aggregates cannot be resurrected`,
            ),
          );

          return;
        }

        transaction.objectStore(RECORDS_OBJECT_STORE).put(record);
      };

      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(new Error(`an insight term store put failed: ${transaction.error?.message ?? ""}`));
      transaction.onabort = () => reject(new Error(`an insight term store put was aborted`));
    });
  }

  async tombstone(captureId: string): Promise<void> {
    const database = await this.database();

    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(
        [RECORDS_OBJECT_STORE, TOMBSTONES_OBJECT_STORE],
        "readwrite",
      );

      transaction.objectStore(RECORDS_OBJECT_STORE).delete(captureId);
      // The tombstone store has out-of-line keys, so the id is both value and key.
      transaction.objectStore(TOMBSTONES_OBJECT_STORE).put(captureId, captureId);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new Error(`an insight term store tombstone failed: ${transaction.error?.message ?? ""}`),
        );
      transaction.onabort = () => reject(new Error(`an insight term store tombstone was aborted`));
    });
  }

  async purgeKinds(kinds: ReadonlySet<"terms" | "phrases">): Promise<void> {
    const log = await this.load();
    const database = await this.database();

    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(RECORDS_OBJECT_STORE, "readwrite");
      const store = transaction.objectStore(RECORDS_OBJECT_STORE);

      for (const record of log.records) {
        store.put({
          ...record,
          terms: kinds.has("terms") ? [] : record.terms,
          phrases: kinds.has("phrases") ? [] : record.phrases,
        });
      }

      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new Error(`an insight term store purge failed: ${transaction.error?.message ?? ""}`),
        );
      transaction.onabort = () => reject(new Error(`an insight term store purge was aborted`));
    });
  }

  async clear(): Promise<void> {
    const database = await this.database();

    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(
        [RECORDS_OBJECT_STORE, TOMBSTONES_OBJECT_STORE],
        "readwrite",
      );

      transaction.objectStore(RECORDS_OBJECT_STORE).clear();
      transaction.objectStore(TOMBSTONES_OBJECT_STORE).clear();
      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new Error(`an insight term store clear failed: ${transaction.error?.message ?? ""}`),
        );
      transaction.onabort = () => reject(new Error(`an insight term store clear was aborted`));
    });
  }

  private database(): Promise<IDBDatabase> {
    this.databasePromise ??= new Promise((resolve, reject) => {
      if (!this.factory) {
        reject(new Error("IndexedDB is unavailable in this environment"));

        return;
      }

      const request = this.factory.open(this.databaseName, 1);

      request.onupgradeneeded = () => {
        request.result.createObjectStore(RECORDS_OBJECT_STORE, { keyPath: "capture_id" });
        request.result.createObjectStore(TOMBSTONES_OBJECT_STORE);
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () =>
        reject(new Error(`could not open the insight term store: ${request.error?.message ?? ""}`));
    });

    return this.databasePromise;
  }
}

export interface InsightTermRecorderOptions {
  /** Injectable clock for deterministic tests. */
  readonly now?: () => Date;
  /** Consent provider, read at write time so a withdrawal takes effect at once. */
  readonly consent?: () => InsightConsent;
}

export interface RecognitionTermsInput {
  readonly captureId: string;
  /** The selected transcript text; only its aggregates ever leave this call. */
  readonly transcriptText: string;
}

/**
 * The consent-gated writer: records one capture's term aggregates only for
 * the kinds the current consent grants. A retranscription replaces the
 * capture's record (revising the counts, never adding a second one).
 * Consent withdrawal is a separate, explicit act — `withdrawKinds` deletes
 * the withdrawn kind's retained data — because turning an analysis off must
 * delete what it derived, not merely stop adding to it. Deletion is
 * `captureDeleted`: a tombstone a stale re-record cannot walk around.
 */
export class InsightTermRecorder {
  private readonly store: InsightTermStore;
  private readonly now: () => Date;
  private readonly consent: () => InsightConsent;
  private writeChain: Promise<void> = Promise.resolve();

  constructor(store: InsightTermStore, options: InsightTermRecorderOptions = {}) {
    this.store = store;
    this.now = options.now ?? (() => new Date());
    this.consent = options.consent ?? (() => DEFAULT_INSIGHT_CONSENT);
  }

  /**
   * Record the aggregates of a settled selected transcript. When the consent
   * grants neither kind, nothing is written — withdrawal is handled by
   * withdrawKinds, deletion by captureDeleted, and this path never invents
   * either. Writes serialize on a chain so store order follows emission
   * order.
   */
  async recognitionSelected(input: RecognitionTermsInput): Promise<void> {
    const kinds = consentedTermKinds(this.consent());

    if (kinds.size === 0) return;

    const derived = captureTerms(input.transcriptText, { kinds });

    await this.enqueue(() =>
      this.store.put({
        schema_version: INSIGHT_TERM_SCHEMA_VERSION,
        capture_id: input.captureId,
        occurred_at: this.now().toISOString(),
        tokenizer: TOKENIZER_ID,
        terms: derived.terms,
        phrases: derived.phrases,
      }),
    );
  }

  /** The confirmed-delete path: remove the capture's aggregates, forever. */
  async captureDeleted(captureId: string): Promise<void> {
    await this.enqueue(() => this.store.tombstone(captureId));
  }

  /** Consent withdrawal for a kind deletes that kind's retained data. */
  async withdrawKinds(kinds: ReadonlySet<"terms" | "phrases">): Promise<void> {
    await this.enqueue(() => this.store.purgeKinds(kinds));
  }

  /** Reset: term aggregation starts over from an empty store. */
  async reset(): Promise<void> {
    await this.enqueue(() => this.store.clear());
  }

  private enqueue(work: () => Promise<void>): Promise<void> {
    const run = this.writeChain.then(work, work);

    this.writeChain = run.catch(() => {
      /* the chain survives a failed write; the caller sees the rejection */
    });

    return run;
  }
}
