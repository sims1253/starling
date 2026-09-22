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
 * event schema enforces with closed branches. Each record also stamps
 * `derived_kinds`, the kinds its write actually derived, so a later reader
 * can tell an analyzed-but-empty take from one nobody analyzed — a
 * distinction an empty label list alone cannot make.
 */

export const INSIGHT_TERM_SCHEMA_VERSION = 1;

/** The two content-derived kinds a record can retain aggregates of. */
export type InsightTermKind = "terms" | "phrases";

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
  /**
   * The kinds this record actually derived and still retains, stamped at
   * write time from the granted kinds and narrowed by withdrawal purges.
   * A kind listed here with an empty label list is an analyzed take that
   * yielded nothing of that kind — a real zero, not an unknown. A record
   * without the stamp (pre-`derived_kinds` data) has only its labels as
   * evidence: `analyzedKinds` counts a kind whose labels survive and
   * treats a fully empty unstamped record as an unknown.
   */
  readonly derived_kinds?: readonly InsightTermKind[];
}

export const InsightTermRecordSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  capture_id: EventIdSchema,
  occurred_at: TimestampSchema,
  tokenizer: SafeTokenSchema,
  terms: TermCountsSchema,
  phrases: PhraseCountsSchema,
  derived_kinds: Schema.optional(Schema.Array(Schema.Literals(["terms", "phrases"]))),
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

/**
 * What one withdrawal purge did: the records it rewrote and only those, plus
 * how many damaged records it had to quarantine. Quarantined records may
 * still retain the withdrawn kind's aggregates — the count exists so the
 * caller can state the weaker guarantee instead of reporting the withdrawal
 * as complete.
 */
export interface InsightTermPurgeResult {
  readonly updated: readonly InsightTermRecord[];
  readonly skippedInvalid: number;
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
  readonly kinds?: ReadonlySet<InsightTermKind>;
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
 *
 * The result names the kinds it derived in `derived_kinds`, so a stored
 * record can always say which kinds an empty list is an honest zero for.
 */
export function captureTerms(
  transcriptText: string,
  options: CaptureTermsOptions = {},
): Omit<InsightTermRecord, "schema_version" | "capture_id" | "occurred_at" | "tokenizer"> {
  const kinds = options.kinds ?? new Set<InsightTermKind>(["terms", "phrases"]);
  const wantsTerms = kinds.has("terms");
  const wantsPhrases = kinds.has("phrases");
  const tokens = wantsTerms || wantsPhrases ? lexicalTokens(transcriptText) : [];

  const terms = wantsTerms ? boundedEntries(countAll(tokens)) : [];
  const phrases = wantsPhrases ? boundedEntries(countAll(phraseTokens(tokens))) : [];

  return { terms, phrases, derived_kinds: [...kinds].sort() };
}

/**
 * Durable storage for content-derived term aggregates. `put` replaces a
 * capture's record (a retranscription revises the counts, it never adds a
 * second record), refuses tombstoned captures (a deleted take's phrases
 * cannot be resurrected by a stale re-record), and `tombstone` removes the
 * record and remembers the deletion. `purgeKinds` implements consent
 * withdrawal: the withdrawn kind's data is deleted from every record that
 * still held or stamped it.
 *
 * The mutating methods resolve with what they changed — the written record,
 * the purged records and only those — so a caller holding a mirror of
 * `load()` can apply the delta without re-reading the store on every write
 * and without re-upserting records nothing touched.
 */
export interface InsightTermStore {
  load(): Promise<InsightTermLog>;
  put(record: InsightTermRecord): Promise<InsightTermRecord>;
  tombstone(captureId: string): Promise<void>;
  purgeKinds(kinds: ReadonlySet<InsightTermKind>): Promise<InsightTermPurgeResult>;
  clear(): Promise<void>;
}

function assertConforming(record: InsightTermRecord): void {
  if (Option.isNone(decodeInsightTermRecord(record))) {
    throw new InsightTermValidationError(insightTermRecordProblems(record).join("; "));
  }
}

/** Whether a withdrawal of `kind` would change this record at all. */
function purgeTouches(record: InsightTermRecord, kind: InsightTermKind): boolean {
  const labels = kind === "terms" ? record.terms : record.phrases;

  return labels.length > 0 || record.derived_kinds?.includes(kind) === true;
}

/**
 * The kinds a record's retained aggregates prove it derived — the only
 * evidence an unstamped (pre-`derived_kinds`) record has. Empty lists prove
 * nothing either way, so they stamp nothing.
 */
function evidencedKinds(record: InsightTermRecord): readonly InsightTermKind[] {
  return (["terms", "phrases"] as const).filter((kind) =>
    kind === "terms" ? record.terms.length > 0 : record.phrases.length > 0,
  );
}

/**
 * The kinds a record is analyzed for, as the one shared per-kind predicate
 * every denominator derives from — the per-kind card counts and the panel's
 * any-kind count can never drift apart. A stamped record reads its stamp (a
 * kind stamped with an empty label list is a real zero: analyzed, nothing
 * found); an unstamped record falls back to label evidence
 * (`evidencedKinds`): labels of the kind present prove the take was
 * analyzed for it, which beats treating pre-stamp data as an unknown.
 */
export function analyzedKinds(
  record: InsightTermRecord,
): Readonly<Record<InsightTermKind, boolean>> {
  const stamped = record.derived_kinds;

  if (stamped !== undefined) {
    return { terms: stamped.includes("terms"), phrases: stamped.includes("phrases") };
  }

  return { terms: record.terms.length > 0, phrases: record.phrases.length > 0 };
}

/**
 * One record after a withdrawal purge of `kinds`: the withdrawn kinds'
 * aggregates are deleted and their stamp is removed from `derived_kinds` —
 * a purged kind becomes an unknown again, not a zero. A stamped record
 * narrows its own stamp; an unstamped one is stamped from what the purge
 * left behind, so the withdrawal never erases provenance for a kind the
 * record still holds data for. Undefined when the purge changes nothing
 * (neither retained nor stamped for any of the kinds), so a store can skip
 * the write and keep its returned delta to what actually changed.
 */
function purgedRecord(
  record: InsightTermRecord,
  kinds: ReadonlySet<InsightTermKind>,
): InsightTermRecord | undefined {
  if (![...kinds].some((kind) => purgeTouches(record, kind))) return undefined;

  const emptied: InsightTermRecord = {
    ...record,
    terms: kinds.has("terms") ? [] : record.terms,
    phrases: kinds.has("phrases") ? [] : record.phrases,
  };

  // A stamped record narrows its own stamp. An unstamped one is stamped
  // only from what the purge left behind — and when no evidence survives,
  // it stays unstamped: an unknown, never a fabricated "analyzed, found
  // nothing" stamp the labels never supported.
  if (emptied.derived_kinds === undefined) {
    const evidenced = evidencedKinds(emptied).filter((kind) => !kinds.has(kind));

    return evidenced.length === 0 ? emptied : { ...emptied, derived_kinds: evidenced };
  }

  return {
    ...emptied,
    derived_kinds: emptied.derived_kinds.filter((kind) => !kinds.has(kind)),
  };
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

  async put(record: InsightTermRecord): Promise<InsightTermRecord> {
    assertConforming(record);

    if (this.tombstones.has(record.capture_id)) {
      throw new InsightTermTombstoneError(
        record.capture_id,
        `Capture ${record.capture_id} was deleted; its term aggregates cannot be resurrected`,
      );
    }

    this.records.set(record.capture_id, record);

    return record;
  }

  async tombstone(captureId: string): Promise<void> {
    this.records.delete(captureId);
    this.tombstones.add(captureId);
  }

  async purgeKinds(kinds: ReadonlySet<InsightTermKind>): Promise<InsightTermPurgeResult> {
    const updated: InsightTermRecord[] = [];

    for (const [captureId, record] of this.records) {
      const purged = purgedRecord(record, kinds);

      if (purged === undefined) continue;

      this.records.set(captureId, purged);
      updated.push(purged);
    }

    return Object.freeze({ updated: Object.freeze(updated), skippedInvalid: 0 });
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

  async put(record: InsightTermRecord): Promise<InsightTermRecord> {
    assertConforming(record);
    const database = await this.database();

    await new Promise<void>((resolve, reject) => {
      // The tombstone refusal settles the promise itself; onabort/onerror
      // must not race a second, generic rejection onto the same settled
      // promise (engines may fire either after an abort()).
      let refused: InsightTermTombstoneError | undefined;

      const transaction = database.transaction(
        [RECORDS_OBJECT_STORE, TOMBSTONES_OBJECT_STORE],
        "readwrite",
      );

      const tombstoneRequest = transaction
        .objectStore(TOMBSTONES_OBJECT_STORE)
        .get(record.capture_id);

      tombstoneRequest.onsuccess = () => {
        if (tombstoneRequest.result !== undefined) {
          refused = new InsightTermTombstoneError(
            record.capture_id,
            `Capture ${record.capture_id} was deleted; its term aggregates cannot be resurrected`,
          );

          reject(refused);
          transaction.abort();

          return;
        }

        transaction.objectStore(RECORDS_OBJECT_STORE).put(record);
      };

      transaction.oncomplete = () => resolve();

      transaction.onerror = () => {
        if (refused !== undefined) return;

        reject(new Error(`an insight term store put failed: ${transaction.error?.message ?? ""}`));
      };

      transaction.onabort = () => {
        if (refused !== undefined) return;

        reject(new Error(`an insight term store put was aborted`));
      };
    });

    return record;
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

  /**
   * Withdrawal is one atomic read-modify-write: a single readwrite
   * transaction walks the records store with a cursor and rewrites exactly
   * the conforming records the withdrawal changes, in place, so a write
   * that lands mid-purge (a recognition settlement from the serialized
   * chain, another window) can never be clobbered by a stale pre-purge
   * snapshot — the deleted kind is gone from every record the transaction
   * actually saw, and nothing else changes. Damaged records are
   * quarantined, not repaired: the cursor skips what does not decode
   * (exactly like `load()` would) and the skip is counted, because a
   * quarantined record may still retain the withdrawn kind's data and the
   * caller must be able to say so.
   */
  async purgeKinds(kinds: ReadonlySet<InsightTermKind>): Promise<InsightTermPurgeResult> {
    const database = await this.database();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(RECORDS_OBJECT_STORE, "readwrite");
      const cursorRequest = transaction.objectStore(RECORDS_OBJECT_STORE).openCursor();
      const updated: InsightTermRecord[] = [];
      let skippedInvalid = 0;

      cursorRequest.onsuccess = () => {
        const cursor = cursorRequest.result;

        if (cursor === null) return;

        const decoded = decodeInsightTermRecord(cursor.value);

        if (Option.isSome(decoded)) {
          const purged = purgedRecord(decoded.value, kinds);

          // Only records the withdrawal actually changed are rewritten and
          // returned — a second withdrawal of the same kind (or a record
          // that never held or stamped it) is a no-op, so the delta callers
          // apply stays honest and the store keeps its writes minimal.
          if (purged !== undefined) {
            cursor.update(purged);
            updated.push(purged);
          }
        } else {
          skippedInvalid += 1;
        }

        cursor.continue();
      };

      transaction.oncomplete = () =>
        resolve(Object.freeze({ updated: Object.freeze(updated), skippedInvalid }));
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
  /**
   * The capture's finalization instant — its E28 capture_finalized event's
   * occurred_at — when the caller knows it. The record is anchored to the
   * take's own time so a delayed retranscription cannot move a take across
   * a card-window boundary. Only a schema-shaped timestamp anchors; the
   * write-time clock is the fallback for a finalization event still in
   * flight or a value that is not schema-shaped — an anchor must never
   * cost the write its aggregates.
   */
  readonly finalizedAt?: string;
}

/**
 * Only a schema-shaped timestamp may anchor a record; anything else falls
 * back to the write-time clock instead of failing the store's validation.
 */
function anchorableFinalizedAt(value: string | undefined): string | undefined {
  return value !== undefined && TIMESTAMP_PATTERN.test(value) ? value : undefined;
}

/**
 * What one recorder write did to the store, so a caller holding a mirror of
 * `load()` can apply the change without re-reading everything: the write
 * already knows exactly what changed. `none` is not an omission — it is the
 * consent answer "no kinds granted", where nothing was written at all.
 */
export type InsightTermWrite =
  | { readonly kind: "none" }
  | { readonly kind: "record"; readonly record: InsightTermRecord }
  | { readonly kind: "tombstone"; readonly captureId: string }
  | {
      readonly kind: "purge";
      readonly records: readonly InsightTermRecord[];
      /** Damaged records the purge quarantined; their data may survive. */
      readonly skippedInvalid: number;
    }
  | { readonly kind: "clear" };

/**
 * Apply one write's delta to a records mirror: the upserts replace in place
 * (a capture keeps its position, a new capture appends), a tombstone drops
 * its capture, a purge upserts the records the store rewrote, and a clear
 * empties the mirror. Pure — the caller's state update stays functional.
 *
 * Deltas are recorder output, and the recorder owns the tombstone
 * invariant: `put` refuses a tombstoned capture before any delta exists, so
 * a "record" delta for a capture whose "tombstone" delta already ran cannot
 * be produced (the write rejects to its caller instead). A caller applying
 * raw writes from anywhere else takes custody of that invariant itself.
 */
export function applyTermWrite(
  records: readonly InsightTermRecord[],
  write: InsightTermWrite,
): readonly InsightTermRecord[] {
  switch (write.kind) {
    case "none":
      return records;
    case "record":
      return upsertRecords(records, [write.record]);
    case "tombstone":
      return records.filter((record) => record.capture_id !== write.captureId);
    case "purge":
      return upsertRecords(records, write.records);
    case "clear":
      return [];
  }
}

function upsertRecords(
  records: readonly InsightTermRecord[],
  updates: readonly InsightTermRecord[],
): readonly InsightTermRecord[] {
  const byCapture = new Map(records.map((record) => [record.capture_id, record] as const));

  for (const update of updates) byCapture.set(update.capture_id, update);

  return Object.freeze([...byCapture.values()]);
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
  private writeChain: Promise<unknown> = Promise.resolve();

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
   * order, and the resolved write names exactly what changed so the caller's
   * mirror can apply it without re-reading the store. The record stamps the
   * granted kinds (`derived_kinds`), so a later reader can tell an
   * analyzed-but-empty take from one nobody analyzed.
   */
  async recognitionSelected(input: RecognitionTermsInput): Promise<InsightTermWrite> {
    const kinds = consentedTermKinds(this.consent());

    if (kinds.size === 0) return { kind: "none" };

    const derived = captureTerms(input.transcriptText, { kinds });

    const record: InsightTermRecord = {
      schema_version: INSIGHT_TERM_SCHEMA_VERSION,
      capture_id: input.captureId,
      occurred_at: anchorableFinalizedAt(input.finalizedAt) ?? this.now().toISOString(),
      tokenizer: TOKENIZER_ID,
      terms: derived.terms,
      phrases: derived.phrases,
      derived_kinds: derived.derived_kinds,
    };

    await this.enqueue(() => this.store.put(record));

    return { kind: "record", record };
  }

  /** The confirmed-delete path: remove the capture's aggregates, forever. */
  async captureDeleted(captureId: string): Promise<InsightTermWrite> {
    await this.enqueue(() => this.store.tombstone(captureId));

    return { kind: "tombstone", captureId };
  }

  /** Consent withdrawal for a kind deletes that kind's retained data. */
  async withdrawKinds(kinds: ReadonlySet<InsightTermKind>): Promise<InsightTermWrite> {
    const { updated, skippedInvalid } = await this.enqueue(() => this.store.purgeKinds(kinds));

    return { kind: "purge", records: updated, skippedInvalid };
  }

  /** Reset: term aggregation starts over from an empty store. */
  async reset(): Promise<InsightTermWrite> {
    await this.enqueue(() => this.store.clear());

    return { kind: "clear" };
  }

  private enqueue<T>(work: () => Promise<T>): Promise<T> {
    const run = this.writeChain.then(work, work);

    this.writeChain = run.catch(() => {
      /* the chain survives a failed write; the caller sees the rejection */
    });

    return run;
  }
}
