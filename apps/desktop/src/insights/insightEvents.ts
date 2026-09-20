import { Option, Predicate, Schema } from "effect";

/**
 * Starling local Insight events, schema v1 (issue draft E28, consumed by E29).
 *
 * This module is the TypeScript port of the frozen contract at
 * `packages/contracts/insight-events/schema.json`: the same five event kinds,
 * the same closed property sets, and the same idempotency rule. The privacy
 * contract is structural, not advisory — every branch closes its properties
 * (`onExcessProperty: "error"` here, `additionalProperties: false` there), so
 * an event carrying raw text, a selection, a path, a window title or a secret
 * is invalid before it can ever be persisted. Events carry counts, IDs,
 * versions and constrained tokens only.
 *
 * Idempotency: `event_id` is the dedupe key. A retry or sync replay must
 * resend a payload equal under `sameEventPayload` under the same `event_id`;
 * a differing payload under a known `event_id` is a conflict (a hard error),
 * never a harmless replay. Deletion: `capture_deleted` is a tombstone that
 * dominates stale replays — replayed pre-deletion events can never resurrect
 * the capture's totals (enforced by the aggregator in insightMetrics.ts).
 */

export const INSIGHT_EVENT_SCHEMA_VERSION = 1;

export const DELIVERY_STATUSES = [
  "confirmed",
  "submitted_unconfirmed",
  "failed",
  "conflict",
  "cancelled",
] as const;

export type DeliveryStatus = (typeof DELIVERY_STATUSES)[number];

export const CHANGE_KINDS = ["structural", "user", "dictionary", "snippet", "style"] as const;

export type ChangeKind = (typeof CHANGE_KINDS)[number];

export const TRANSFORMATION_KINDS = [
  "model_authoring",
  "snippet_expansion",
  "user_edit",
  "dictionary_substitution",
] as const;

export type TransformationKind = (typeof TRANSFORMATION_KINDS)[number];

const TIMESTAMP_PATTERN =
  /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$/;

/** #/$defs/eventId — also the shape of `capture_id` references. */
const EventIdSchema = Schema.String.pipe(Schema.check(Schema.isPattern(/^[A-Za-z0-9_.-]{1,128}$/)));

/** #/$defs/safeToken — constrained tokens (mode ids, tokenizer ids). */
const SafeTokenSchema = Schema.String.pipe(
  Schema.check(Schema.isPattern(/^[A-Za-z0-9_.:+-]{1,128}$/)),
);

/** #/$defs/tzName — IANA-style timezone names. */
const TzNameSchema = Schema.String.pipe(Schema.check(Schema.isPattern(/^[A-Za-z0-9_+/-]{1,64}$/)));

/** #/$defs/timestamp — an instant with an explicit UTC offset. */
const TimestampSchema = Schema.String.pipe(Schema.check(Schema.isPattern(TIMESTAMP_PATTERN)));

const NonNegativeIntSchema = Schema.Number.pipe(
  Schema.check(Schema.isInt()),
  Schema.check(Schema.isGreaterThanOrEqualTo(0)),
);

const PositiveIntSchema = Schema.Number.pipe(
  Schema.check(Schema.isInt()),
  Schema.check(Schema.isGreaterThan(0)),
);

/** `["number","null"]` with `minimum: 0` — decimals allowed, unknown waits are null. */
const NonNegativeNumberSchema = Schema.Number.pipe(Schema.check(Schema.isGreaterThanOrEqualTo(0)));

export interface ChangeCounts {
  readonly structural: number;
  readonly user: number;
  readonly dictionary: number;
  readonly snippet: number;
  readonly style: number;
}

const ChangeCountsSchema = Schema.Struct({
  structural: NonNegativeIntSchema,
  user: NonNegativeIntSchema,
  dictionary: NonNegativeIntSchema,
  snippet: NonNegativeIntSchema,
  style: NonNegativeIntSchema,
});

export interface CaptureFinalizedEvent {
  readonly schema_version: 1;
  readonly event_id: string;
  readonly capture_id: string;
  readonly occurred_at: string;
  readonly type: "capture_finalized";
  readonly sample_count: number;
  readonly sample_rate: number;
  readonly complete_audio: boolean;
  readonly mode_id: string;
  readonly reporting_timezone: string;
}

export interface RecognitionSelectedEvent {
  readonly schema_version: 1;
  readonly event_id: string;
  readonly capture_id: string;
  readonly occurred_at: string;
  readonly type: "recognition_selected";
  readonly attempt_id: string;
  readonly selection_seq: number;
  readonly lexical_words: number;
  readonly raw_words: number;
  readonly tokenizer: string;
  readonly post_stop_ready_ms: number | null;
}

export interface TransformationCompletedEvent {
  readonly schema_version: 1;
  readonly event_id: string;
  readonly capture_id: string;
  readonly occurred_at: string;
  readonly type: "transformation_completed";
  readonly revision_id: string;
  readonly revision_seq: number;
  readonly transformation_kind: TransformationKind;
  readonly formula_version: 1;
  readonly change_counts: ChangeCounts;
}

export interface DeliveryRecordedEvent {
  readonly schema_version: 1;
  readonly event_id: string;
  readonly capture_id: string;
  readonly occurred_at: string;
  readonly type: "delivery_recorded";
  readonly delivery_id: string;
  readonly delivery_seq: number;
  readonly status: DeliveryStatus;
  readonly output_words: number;
  readonly generated_words: number;
}

export interface CaptureDeletedEvent {
  readonly schema_version: 1;
  readonly event_id: string;
  readonly capture_id: string;
  readonly occurred_at: string;
  readonly type: "capture_deleted";
}

export type InsightEvent =
  | CaptureFinalizedEvent
  | RecognitionSelectedEvent
  | TransformationCompletedEvent
  | DeliveryRecordedEvent
  | CaptureDeletedEvent;

export const CaptureFinalizedSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  event_id: EventIdSchema,
  capture_id: EventIdSchema,
  occurred_at: TimestampSchema,
  type: Schema.Literals(["capture_finalized"]),
  sample_count: NonNegativeIntSchema,
  sample_rate: PositiveIntSchema,
  complete_audio: Schema.Boolean,
  mode_id: SafeTokenSchema,
  reporting_timezone: TzNameSchema,
});

export const RecognitionSelectedSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  event_id: EventIdSchema,
  capture_id: EventIdSchema,
  occurred_at: TimestampSchema,
  type: Schema.Literals(["recognition_selected"]),
  attempt_id: EventIdSchema,
  selection_seq: NonNegativeIntSchema,
  lexical_words: NonNegativeIntSchema,
  raw_words: NonNegativeIntSchema,
  tokenizer: SafeTokenSchema,
  post_stop_ready_ms: Schema.NullOr(NonNegativeNumberSchema),
});

export const TransformationCompletedSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  event_id: EventIdSchema,
  capture_id: EventIdSchema,
  occurred_at: TimestampSchema,
  type: Schema.Literals(["transformation_completed"]),
  revision_id: EventIdSchema,
  revision_seq: NonNegativeIntSchema,
  transformation_kind: Schema.Literals([...TRANSFORMATION_KINDS]),
  formula_version: Schema.Literal(1),
  change_counts: ChangeCountsSchema,
});

export const DeliveryRecordedSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  event_id: EventIdSchema,
  capture_id: EventIdSchema,
  occurred_at: TimestampSchema,
  type: Schema.Literals(["delivery_recorded"]),
  delivery_id: EventIdSchema,
  delivery_seq: NonNegativeIntSchema,
  status: Schema.Literals([...DELIVERY_STATUSES]),
  output_words: NonNegativeIntSchema,
  generated_words: NonNegativeIntSchema,
});

export const CaptureDeletedSchema = Schema.Struct({
  schema_version: Schema.Literal(1),
  event_id: EventIdSchema,
  capture_id: EventIdSchema,
  occurred_at: TimestampSchema,
  type: Schema.Literals(["capture_deleted"]),
});

export const InsightEventSchema = Schema.Union([
  CaptureFinalizedSchema,
  RecognitionSelectedSchema,
  TransformationCompletedSchema,
  DeliveryRecordedSchema,
  CaptureDeletedSchema,
]);

/**
 * Decode one unknown record against the frozen event schema. Excess properties
 * are errors, exactly like the contract's `additionalProperties: false`, so a
 * poisoned record (raw text, a path, a secret) is rejected structurally.
 */
export const decodeInsightEvent = Schema.decodeUnknownOption(InsightEventSchema, {
  onExcessProperty: "error",
});

export function isCaptureFinalized(event: InsightEvent): event is CaptureFinalizedEvent {
  return event.type === "capture_finalized";
}

export function isRecognitionSelected(event: InsightEvent): event is RecognitionSelectedEvent {
  return event.type === "recognition_selected";
}

export function isTransformationCompleted(
  event: InsightEvent,
): event is TransformationCompletedEvent {
  return event.type === "transformation_completed";
}

export function isDeliveryRecorded(event: InsightEvent): event is DeliveryRecordedEvent {
  return event.type === "delivery_recorded";
}

export function isCaptureDeleted(event: InsightEvent): event is CaptureDeletedEvent {
  return event.type === "capture_deleted";
}

/**
 * Structural validation result for one candidate event: an empty list means
 * the event conforms to the frozen schema; otherwise the strings name the
 * violations (used by tests and by the store's write boundary).
 */
export function insightEventProblems(
  // A raw record from storage or a poisoned test value is unknown by definition;
  // this function is the boundary that parses it.
  value: unknown, // oxlint-disable-line anti-slop/no-unknown-parameters -- see above
): readonly string[] {
  const decoded = decodeInsightEvent(value);

  if (Option.isSome(decoded)) return [];

  // The decode message of a union is verbose; the short form keeps test
  // failures and quarantine notes readable.
  return [`event does not conform to the insight event schema v${INSIGHT_EVENT_SCHEMA_VERSION}`];
}

/**
 * Canonical JSON for payload equality: object keys sorted recursively, so two
 * records with the same content compare equal regardless of key order — the
 * same equality the Python oracle expresses with dict comparison.
 */
function canonicalJson(
  // Any JSON value can appear in an event payload; canonicalization is exactly
  // the walk over that unknown.
  value: unknown, // oxlint-disable-line anti-slop/no-unknown-parameters -- see above
): string {
  if (value === null) return "null";

  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;

  if (Predicate.isObject(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }

  return JSON.stringify(value);
}

/** Same `event_id` plus equal payload = idempotent replay; otherwise a conflict. */
export function sameEventPayload(left: InsightEvent, right: InsightEvent): boolean {
  return canonicalJson(left) === canonicalJson(right);
}

export class InsightEventValidationError extends Error {
  override readonly name = "InsightEventValidationError";

  constructor(message: string) {
    super(message);
  }
}

export class InsightEventConflictError extends Error {
  override readonly name = "InsightEventConflictError";

  constructor(
    readonly eventId: string,
    message: string,
  ) {
    super(message);
  }
}

/** Events that decoded cleanly, plus how many stored records were damaged. */
export interface InsightEventLog {
  readonly events: readonly InsightEvent[];
  readonly invalidCount: number;
}

/**
 * Durable storage for insight events. The append contract is the schema's
 * idempotency rule: a byte-equal replay of a known `event_id` is a no-op, a
 * differing payload under a known `event_id` rejects, and nothing else is
 * ever overwritten — sequence fields, not rewrites, carry revisions.
 */
export interface InsightEventStore {
  load(): Promise<InsightEventLog>;
  append(event: InsightEvent): Promise<void>;
  clear(): Promise<void>;
}

export class MemoryInsightEventStore implements InsightEventStore {
  private readonly events = new Map<string, InsightEvent>();

  async load(): Promise<InsightEventLog> {
    return Object.freeze({
      events: Object.freeze([...this.events.values()]),
      invalidCount: 0,
    });
  }

  async append(event: InsightEvent): Promise<void> {
    assertConforming(event);
    const existing = this.events.get(event.event_id);

    if (existing !== undefined) {
      if (!sameEventPayload(existing, event)) {
        throw new InsightEventConflictError(
          event.event_id,
          `Same event ID ${event.event_id} has conflicting payloads`,
        );
      }

      return;
    }

    this.events.set(event.event_id, event);
  }

  async clear(): Promise<void> {
    this.events.clear();
  }
}

function assertConforming(event: InsightEvent): void {
  if (Option.isNone(decodeInsightEvent(event))) {
    throw new InsightEventValidationError(insightEventProblems(event).join("; "));
  }
}

export interface IndexedDbInsightEventStoreOptions {
  readonly databaseName?: string;
  readonly indexedDB?: IDBFactory | undefined;
}

const EVENTS_OBJECT_STORE = "events";

function runIdbRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(new Error(`an insight event store request failed: ${request.error?.message ?? ""}`));
  });
}

/**
 * Insight events in their own IndexedDB database, mirroring the dictation
 * store's conventions: a versioned database name, records keyed by
 * `event_id`, damaged records quarantined out of `load()` (counted, never
 * silently deleted), and appends that enforce replay-vs-conflict before the
 * write. Tombstones persist forever by design — they must dominate stale
 * replays, so there is deliberately no tombstone garbage collection.
 */
export class IndexedDbInsightEventStore implements InsightEventStore {
  private readonly databaseName: string;
  private readonly factory: IDBFactory | undefined;
  private databasePromise: Promise<IDBDatabase> | undefined;

  constructor(options: IndexedDbInsightEventStoreOptions = {}) {
    this.databaseName = options.databaseName ?? "starling-insights";
    this.factory = options.indexedDB ?? globalThis.indexedDB;
  }

  async load(): Promise<InsightEventLog> {
    const database = await this.database();

    const records = await runIdbRequest(
      database
        .transaction(EVENTS_OBJECT_STORE, "readonly")
        .objectStore(EVENTS_OBJECT_STORE)
        .getAll(),
    );

    const events: InsightEvent[] = [];
    let invalidCount = 0;

    for (const record of records) {
      const decoded = decodeInsightEvent(record);

      if (Option.isSome(decoded)) events.push(decoded.value);
      else invalidCount += 1;
    }

    return Object.freeze({ events: Object.freeze(events), invalidCount });
  }

  async append(event: InsightEvent): Promise<void> {
    assertConforming(event);
    const database = await this.database();

    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(EVENTS_OBJECT_STORE, "readwrite");
      const store = transaction.objectStore(EVENTS_OBJECT_STORE);
      const request = store.get(event.event_id);

      request.onsuccess = () => {
        const existing = request.result;

        if (existing !== undefined) {
          const decoded = decodeInsightEvent(existing);

          if (Option.isNone(decoded) || !sameEventPayload(decoded.value, event)) {
            transaction.abort();
            reject(
              new InsightEventConflictError(
                event.event_id,
                `Same event ID ${event.event_id} has conflicting payloads`,
              ),
            );

            return;
          }

          // An equal payload under a known event id is an idempotent replay.
          return;
        }

        store.put(event);
      };

      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new Error(`an insight event store append failed: ${transaction.error?.message ?? ""}`),
        );
      transaction.onabort = () => reject(new Error(`an insight event store append was aborted`));
    });
  }

  async clear(): Promise<void> {
    const database = await this.database();

    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(EVENTS_OBJECT_STORE, "readwrite");
      transaction.objectStore(EVENTS_OBJECT_STORE).clear();
      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new Error(`an insight event store clear failed: ${transaction.error?.message ?? ""}`),
        );
      transaction.onabort = () => reject(new Error(`an insight event store clear was aborted`));
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
        request.result.createObjectStore(EVENTS_OBJECT_STORE, { keyPath: "event_id" });
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () =>
        reject(
          new Error(`could not open the insight event store: ${request.error?.message ?? ""}`),
        );
    });

    return this.databasePromise;
  }
}
