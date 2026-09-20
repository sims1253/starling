import {
  CHANGE_KINDS,
  DELIVERY_STATUSES,
  INSIGHT_EVENT_SCHEMA_VERSION,
  isCaptureFinalized,
  isDeliveryRecorded,
  isRecognitionSelected,
  isTransformationCompleted,
  sameEventPayload,
  type ChangeKind,
  type DeliveryRecordedEvent,
  type DeliveryStatus,
  type InsightEvent,
  type TransformationCompletedEvent,
  type TransformationKind,
} from "./insightEvents";

/**
 * TypeScript port of the executable metric contract (E28) —
 * `tests/insight_metrics.py` is the oracle; this module reproduces its frozen
 * semantics for the desktop Insights surface without importing the Python:
 *
 * * Dedupe key: `event_id`. Same ID + equal payload = idempotent replay
 *   (retry/sync). Same ID + different payload = conflict (hard error).
 * * Recognition selection: latest `selection_seq` per capture wins; a retry
 *   *replaces* the word count, it never adds.
 * * Tombstones: any `capture_deleted` for a capture_id removes every
 *   attributable event of that capture, regardless of arrival order. Stale
 *   sync replays cannot resurrect deleted totals.
 * * WPM denominator: silence-inclusive captured seconds of takes that have a
 *   selected recognition ("recognized words per captured minute"), summed over
 *   the population (weighted totals, never an average of per-take rates).
 * * Generated/snippet words are counted from delivery events, separately from
 *   recognized speech words, and never enter the WPM or time proxy numerators.
 * * Time-saved proxy: typing estimate at a user baseline minus eligible
 *   capture seconds minus non-overlapping post-Stop wait. `null` when the
 *   baseline is unset or any wait is unknown; may be negative and is reported
 *   as such.
 *
 * Output field names keep the oracle's snake_case so an exported aggregate is
 * directly comparable against the reference implementation's JSON.
 */

export const FORMULA_VERSION = 1;

const TIME_COMPARISON_CAVEAT =
  "Estimate excludes unobserved correction time; generated words are not speech and never enter this estimate.";

export class InsightContractError extends Error {
  override readonly name = "InsightContractError";

  constructor(message: string) {
    super(message);
  }
}

/**
 * Reject non-finite and negative values. The Python oracle additionally
 * rejects bools and non-numbers before this check; here the event types carry
 * numbers, so the runtime guard has to catch what a cast can smuggle in:
 * NaN and the infinities (a plain `< 0` test alone lets NaN through).
 */
function nonnegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new InsightContractError(`${label} must be finite and nonnegative`);
  }

  return value;
}

/**
 * Latest event by a monotone sequence field; equal seq + different payload =
 * conflict. Mirrors the oracle's `latest_by_sequence`, including the refusal
 * of fractional, negative or non-integer sequence values.
 */
function latestBySequence<T extends InsightEvent>(
  items: readonly T[],
  sequenceOf: (item: T) => number,
): T | undefined {
  const seen = new Map<number, T>();
  let highest: number | undefined;

  for (const item of items) {
    const seq = sequenceOf(item);

    if (!Number.isInteger(seq) || seq < 0) {
      throw new InsightContractError(`Invalid sequence value ${seq}`);
    }

    const existing = seen.get(seq);

    if (existing !== undefined && !sameEventPayload(existing, item)) {
      throw new InsightContractError("Conflicting sequence values");
    }

    seen.set(seq, item);

    if (highest === undefined || seq > highest) highest = seq;
  }

  if (highest === undefined) return undefined;

  const latest = seen.get(highest);

  return latest;
}

/**
 * Idempotency dedupe by event_id, then drop tombstoned captures entirely.
 * Tombstones dominate stale replays; explicit restore is not part of v1.
 */
function dedupeAndTombstone(events: readonly InsightEvent[]): Map<string, InsightEvent[]> {
  const unique = new Map<string, InsightEvent>();

  for (const event of events) {
    if (event.schema_version !== INSIGHT_EVENT_SCHEMA_VERSION) {
      throw new InsightContractError("Unsupported event schema_version");
    }

    const existing = unique.get(event.event_id);

    if (existing !== undefined && !sameEventPayload(existing, event)) {
      throw new InsightContractError("Same event ID has conflicting payloads");
    }

    unique.set(event.event_id, event);
  }

  const deleted = new Set<string>();

  for (const event of unique.values()) {
    if (event.type === "capture_deleted") deleted.add(event.capture_id);
  }

  const grouped = new Map<string, InsightEvent[]>();

  for (const event of unique.values()) {
    if (deleted.has(event.capture_id)) continue;

    const items = grouped.get(event.capture_id);

    if (items === undefined) grouped.set(event.capture_id, [event]);
    else items.push(event);
  }

  return grouped;
}

interface CanonicalCapture {
  readonly sampleCount: number;
  readonly sampleRate: number;
  readonly completeAudio: boolean;
  readonly occurredAt: string;
}

/**
 * The one canonical finalization of a take, under the oracle's exact rules:
 * undefined when the capture has no finalization event (orphan metadata is
 * not evidence of a captured take), an error when two claim to be canonical.
 */
function canonicalCapture(items: readonly InsightEvent[]): CanonicalCapture | undefined {
  const captures = items.filter(isCaptureFinalized);

  if (captures.length === 0) return undefined;

  if (captures.length !== 1) {
    throw new InsightContractError("Multiple canonical capture finalizations");
  }

  const capture = captures[0];

  if (capture === undefined) {
    throw new InsightContractError("Multiple canonical capture finalizations");
  }

  return {
    sampleCount: capture.sample_count,
    sampleRate: capture.sample_rate,
    completeAudio: capture.complete_audio,
    occurredAt: capture.occurred_at,
  };
}

/** Capture duration in seconds; refuses impossible rates and counts. */
function captureSeconds(capture: CanonicalCapture): number {
  const sampleRate = nonnegative(capture.sampleRate, "sample_rate");

  if (sampleRate <= 0) throw new InsightContractError("sample_rate must be positive");

  return nonnegative(capture.sampleCount, "sample_count") / sampleRate;
}

export interface InsightAggregate {
  readonly formula_version: typeof FORMULA_VERSION;
  readonly unique_takes: number;
  readonly selected_recognitions: number;
  readonly incomplete_captures: number;
  readonly recognized_words: number;
  readonly raw_recognized_words: number;
  readonly captured_seconds: number;
  readonly eligible_capture_seconds: number;
  /** Silence-inclusive capture-normalized rate; null rather than corrupted. */
  readonly recognized_words_per_captured_minute: number | null;
  readonly tokenizers: readonly string[];
  readonly delivery_counts: Readonly<Record<DeliveryStatus, number>>;
  readonly output_words_by_status: Readonly<Record<DeliveryStatus, number>>;
  readonly generated_words_by_status: Readonly<Record<DeliveryStatus, number>>;
  /** Changes by type; NOT certified error corrections. */
  readonly change_counts: Readonly<Record<ChangeKind, number>>;
  readonly transformation_counts: Readonly<Record<TransformationKind, number>>;
  readonly typing_time_comparison_seconds: number | null;
  readonly time_comparison_caveat: string;
}

function zeroedStatusCounts(): Record<DeliveryStatus, number> {
  return {
    confirmed: 0,
    submitted_unconfirmed: 0,
    failed: 0,
    conflict: 0,
    cancelled: 0,
  };
}

function zeroedChangeCounts(): Record<ChangeKind, number> {
  return { structural: 0, user: 0, dictionary: 0, snippet: 0, style: 0 };
}

function zeroedTransformationCounts(): Record<TransformationKind, number> {
  return {
    model_authoring: 0,
    snippet_expansion: 0,
    user_edit: 0,
    dictionary_substitution: 0,
  };
}

const DELIVERY_STATUS_SET: ReadonlySet<string> = new Set(DELIVERY_STATUSES);

/**
 * Aggregate a local event log into the v1 metric contract. Throws
 * InsightContractError on structurally impossible input (conflicting event
 * IDs or sequences, impossible counts, non-positive sample rates) so bugs can
 * never silently degrade into plausible numbers.
 */
export function aggregate(
  events: readonly InsightEvent[],
  typingWpm?: number | null,
): InsightAggregate {
  if (typingWpm !== undefined && typingWpm !== null && nonnegative(typingWpm, "typing_wpm") === 0) {
    throw new InsightContractError("typing_wpm must be positive");
  }

  const grouped = dedupeAndTombstone(events);

  let count = 0;
  let words = 0;
  let rawWords = 0;
  let selectedCount = 0;
  let incomplete = 0;
  let capturedSeconds = 0;
  let eligibleSeconds = 0;
  let waitSeconds = 0;
  let waitsKnown = true;
  const tokenizers = new Set<string>();
  const deliveryCounts = zeroedStatusCounts();
  const deliveredWords = zeroedStatusCounts();
  const generatedWords = zeroedStatusCounts();
  const changeCounts = zeroedChangeCounts();
  const transformationCounts = zeroedTransformationCounts();

  for (const captureId of [...grouped.keys()].sort()) {
    const items = grouped.get(captureId) ?? [];
    const capture = canonicalCapture(items);

    // Orphan metadata is not evidence of a captured take.
    if (capture === undefined) continue;

    const duration = captureSeconds(capture);

    capturedSeconds += duration;
    count += 1;
    incomplete += capture.completeAudio ? 0 : 1;

    const selected = latestBySequence(
      items.filter(isRecognitionSelected),
      (event) => event.selection_seq,
    );

    if (selected !== undefined) {
      selectedCount += 1;

      const lexical = nonnegative(selected.lexical_words, "lexical_words");
      const raw = nonnegative(selected.raw_words, "raw_words");

      if (lexical > raw) {
        throw new InsightContractError("Lexical count cannot exceed raw ASR count");
      }

      words += Math.trunc(lexical);
      rawWords += Math.trunc(raw);
      eligibleSeconds += duration;
      tokenizers.add(selected.tokenizer);

      if (selected.post_stop_ready_ms === null) {
        waitsKnown = false;
      } else {
        waitSeconds += nonnegative(selected.post_stop_ready_ms, "post_stop_ready_ms") / 1000;
      }
    }

    // Transformation revisions: latest revision_seq per revision_id wins
    // (a refinement retry replaces, it does not add); distinct revision IDs
    // are distinct passes and both are counted. Change counts are labelled
    // "changes", never "corrected errors".
    const revisions = new Map<string, TransformationCompletedEvent[]>();

    for (const item of items) {
      if (!isTransformationCompleted(item)) continue;

      const list = revisions.get(item.revision_id);

      if (list === undefined) revisions.set(item.revision_id, [item]);
      else list.push(item);
    }

    for (const revisionItems of revisions.values()) {
      const revision = latestBySequence(revisionItems, (event) => event.revision_seq);

      if (revision === undefined) continue;

      transformationCounts[revision.transformation_kind] += 1;

      for (const kind of CHANGE_KINDS) {
        changeCounts[kind] += Math.trunc(
          nonnegative(revision.change_counts[kind], `change_counts.${kind}`),
        );
      }
    }

    const deliveries = new Map<string, DeliveryRecordedEvent[]>();

    for (const item of items) {
      if (!isDeliveryRecorded(item)) continue;

      const list = deliveries.get(item.delivery_id);

      if (list === undefined) deliveries.set(item.delivery_id, [item]);
      else list.push(item);
    }

    for (const deliveryItems of deliveries.values()) {
      const delivery = latestBySequence(deliveryItems, (event) => event.delivery_seq);

      if (delivery === undefined) continue;

      if (!DELIVERY_STATUS_SET.has(delivery.status)) {
        throw new InsightContractError("Unknown delivery acknowledgement");
      }

      const output = nonnegative(delivery.output_words, "output_words");
      const generated = nonnegative(delivery.generated_words, "generated_words");

      if (generated > output) {
        throw new InsightContractError("Generated words cannot exceed output words");
      }

      deliveryCounts[delivery.status] += 1;
      deliveredWords[delivery.status] += Math.trunc(output);
      generatedWords[delivery.status] += Math.trunc(generated);
    }
  }

  const comparable = tokenizers.size === 1;
  const wpm = eligibleSeconds > 0 && comparable ? (words * 60) / eligibleSeconds : null;

  const proxy =
    typingWpm !== undefined && typingWpm !== null && waitsKnown && selectedCount > 0 && comparable
      ? (words * 60) / typingWpm - eligibleSeconds - waitSeconds
      : null;

  return {
    formula_version: FORMULA_VERSION,
    unique_takes: count,
    selected_recognitions: selectedCount,
    incomplete_captures: incomplete,
    recognized_words: words,
    raw_recognized_words: rawWords,
    captured_seconds: capturedSeconds,
    eligible_capture_seconds: eligibleSeconds,
    recognized_words_per_captured_minute: wpm,
    tokenizers: [...tokenizers].sort(),
    delivery_counts: deliveryCounts,
    output_words_by_status: deliveredWords,
    generated_words_by_status: generatedWords,
    change_counts: changeCounts,
    transformation_counts: transformationCounts,
    typing_time_comparison_seconds: proxy,
    time_comparison_caveat: TIME_COMPARISON_CAVEAT,
  };
}

export interface DayActivity {
  readonly takes: number;
  readonly captured_seconds: number;
}

/** Selection-derived quality counts, on the same dedupe/tombstone rules. */
export interface SelectionStats {
  /** Non-deleted captures with at least one selected recognition. */
  readonly capturesWithSelection: number;
  /** Selections a newer selection replaced — retries, not extra takes. */
  readonly supersededSelections: number;
  /** Captures whose selected recognition recognized zero words. */
  readonly emptySelections: number;
}

export function selectionStats(events: readonly InsightEvent[]): SelectionStats {
  const grouped = dedupeAndTombstone(events);
  let capturesWithSelection = 0;
  let supersededSelections = 0;
  let emptySelections = 0;

  for (const items of grouped.values()) {
    const selections = items.filter(isRecognitionSelected);

    if (selections.length === 0) continue;

    capturesWithSelection += 1;
    supersededSelections += selections.length - 1;

    const selected = latestBySequence(selections, (event) => event.selection_seq);

    if (selected !== undefined && selected.lexical_words === 0) emptySelections += 1;
  }

  return { capturesWithSelection, supersededSelections, emptySelections };
}

/** The local calendar-day key of an instant, in the declared timezone. */
export function localDayKey(ms: number, timezoneName: string): string {
  return localDayFormatter(timezoneName).format(ms);
}

/**
 * Group attributable takes into local calendar days of a declared timezone.
 * DST-aware through the host's IANA timezone database — a naive wall-clock
 * conversion would silently misplace the fallback hour, so an unknown
 * timezone refuses to run rather than guessing.
 */
export function activityByDay(
  events: readonly InsightEvent[],
  timezoneName: string,
): Map<string, DayActivity> {
  const dayFormatter = localDayFormatter(timezoneName);
  const grouped = dedupeAndTombstone(events);
  const days = new Map<string, DayBucket>();

  for (const captureId of [...grouped.keys()].sort()) {
    const items = grouped.get(captureId) ?? [];
    const capture = canonicalCapture(items);

    if (capture === undefined) continue;

    const duration = captureSeconds(capture);
    const localDay = dayFormatter.format(parseInsightTimestamp(capture.occurredAt));
    const bucket = days.get(localDay) ?? { takes: 0, captured_seconds: 0 };

    bucket.takes += 1;
    bucket.captured_seconds += duration;
    days.set(localDay, bucket);
  }

  return days;
}

interface DayBucket {
  takes: number;
  captured_seconds: number;
}

export interface LocalWallTime {
  readonly hour: number;
  readonly minute: number;
  readonly second: number;
  readonly utc_offset_hours: number;
}

/**
 * (local hour, minute, second, UTC offset in hours) for DST-ambiguity proofs:
 * during a fallback the same wall time recurs under two offsets, and the
 * offset — not the wall clock — disambiguates them.
 */
export function localWallTime(occurredAt: string, timezoneName: string): LocalWallTime {
  const formatter = localWallFormatter(timezoneName);
  const parts = formatter.formatToParts(parseInsightTimestamp(occurredAt));
  const values = new Map<string, string>();

  for (const part of parts) {
    if (part.type !== "literal") values.set(part.type, part.value);
  }

  const hour = Number.parseInt(values.get("hour") ?? "", 10);
  const minute = Number.parseInt(values.get("minute") ?? "", 10);
  const second = Number.parseInt(values.get("second") ?? "", 10);

  if (![hour, minute, second].every((part) => Number.isInteger(part))) {
    throw new InsightContractError(`could not read local wall time in ${timezoneName}`);
  }

  return {
    hour,
    minute,
    second,
    utc_offset_hours: parseGmtOffset(values.get("timeZoneName") ?? ""),
  };
}

/** Parse a "GMT+02:00"/"GMT-1"/"GMT" long-offset name into signed hours. */
function parseGmtOffset(text: string): number {
  const match = /^GMT(?:([+-])(\d{1,2})(?::(\d{2}))?)?$/.exec(text);

  if (match === null) {
    throw new InsightContractError(`could not read a UTC offset from ${text}`);
  }

  if (match[1] === undefined) return 0;

  const hours = Number.parseInt(match[2] ?? "0", 10);
  const minutes = match[3] === undefined ? 0 : Number.parseInt(match[3], 10);
  const magnitude = hours + minutes / 60;

  return match[1] === "-" ? -magnitude : magnitude;
}

const INSIGHT_TIMESTAMP_PATTERN =
  /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$/;

/** Parse an occurred_at stamp; a naive timestamp without an offset refuses. */
export function parseInsightTimestamp(occurredAt: string): number {
  if (!INSIGHT_TIMESTAMP_PATTERN.test(occurredAt)) {
    throw new InsightContractError("occurred_at must carry an explicit UTC offset");
  }

  const ms = Date.parse(occurredAt);

  if (Number.isNaN(ms)) {
    throw new InsightContractError("occurred_at must carry an explicit UTC offset");
  }

  return ms;
}

const dayFormatters = new Map<string, Intl.DateTimeFormat>();

const wallFormatters = new Map<string, Intl.DateTimeFormat>();

function localDayFormatter(timezoneName: string): Intl.DateTimeFormat {
  let formatter = dayFormatters.get(timezoneName);

  if (formatter === undefined) {
    formatter = resolveTimezone(timezoneName, () => buildLocalDayFormatter(timezoneName));
    dayFormatters.set(timezoneName, formatter);
  }

  return formatter;
}

function localWallFormatter(timezoneName: string): Intl.DateTimeFormat {
  let formatter = wallFormatters.get(timezoneName);

  if (formatter === undefined) {
    formatter = resolveTimezone(timezoneName, () => buildLocalWallFormatter(timezoneName));
    wallFormatters.set(timezoneName, formatter);
  }

  return formatter;
}

function buildLocalDayFormatter(timezoneName: string): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: timezoneName,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

function buildLocalWallFormatter(timezoneName: string): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: timezoneName,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
    timeZoneName: "longOffset",
  });
}

function resolveTimezone(
  timezoneName: string,
  build: () => Intl.DateTimeFormat,
): Intl.DateTimeFormat {
  try {
    return build();
  } catch {
    throw new InsightContractError(`Unknown reporting timezone: ${timezoneName}`);
  }
}
