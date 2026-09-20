import { decodePcm16Wav } from "@starling/dictation";
import {
  InsightEventConflictError,
  InsightEventValidationError,
  insightEventProblems,
  isRecognitionSelected,
  isTransformationCompleted,
  sameEventPayload,
  type CaptureFinalizedEvent,
  type ChangeCounts,
  type DeliveryRecordedEvent,
  type DeliveryStatus,
  type InsightEvent,
  type InsightEventStore,
  type RecognitionSelectedEvent,
  type TransformationCompletedEvent,
} from "./insightEvents";

/**
 * The emitter half of Insights (E29): derive contract-shaped events from the
 * app's real lifecycles — session creation (capture_finalized), a transcript
 * settling (recognition_selected), a refinement landing (transformation_
 * completed), a copy or export leaving the app (delivery_recorded), and the
 * confirmed delete path (capture_deleted). Every value in an event is
 * computed, never copied from user content: word counts come from the
 * declared tokenizer below, and no event ever carries text, selections,
 * paths or secrets (the schema's closed branches enforce it structurally,
 * and nothing here even tries).
 */

/** Declared tokenizer: UAX#29 word segmentation via the host's ICU. */
export const TOKENIZER_ID = "uax29-intl-v1";

/**
 * The mode label for takes recorded by this app. Starling's stance is that
 * the raw transcript is never rewritten, so every take is "faithful"; the
 * fixtures use the same vocabulary.
 */
export const DEFAULT_MODE_ID = "faithful";

let wordSegmenter: Intl.Segmenter | undefined;

function segmenter(): Intl.Segmenter {
  wordSegmenter ??= new Intl.Segmenter("und", { granularity: "word" });

  return wordSegmenter;
}

export interface WordCounts {
  /**
   * Word-like segments (letters/digits). This is the "lexical words" count —
   * the speech-word population every speech metric is computed over.
   */
  readonly lexical: number;
  /** All non-whitespace segments, punctuation clusters included. */
  readonly raw: number;
}

/**
 * Count words under the declared tokenizer. `raw` counts every
 * non-whitespace UAX#29 segment; `lexical` counts only word-like segments, so
 * `lexical <= raw` holds by construction (the aggregate enforces the same
 * invariant on read).
 */
export function segmentWordCounts(text: string): WordCounts {
  let lexical = 0;
  let raw = 0;

  for (const segment of segmenter().segment(text)) {
    if (segment.segment.trim().length === 0) continue;

    raw += 1;

    if (segment.isWordLike === true) lexical += 1;
  }

  return { lexical, raw };
}

function lexicalTokens(text: string): readonly string[] {
  const tokens: string[] = [];

  for (const segment of segmenter().segment(text)) {
    if (segment.isWordLike === true) tokens.push(segment.segment);
  }

  return tokens;
}

/** Word-level multiset difference between two texts, under the declared tokenizer. */
export interface WordMultisetDelta {
  readonly additions: number;
  readonly removals: number;
}

/**
 * Word-level multiset difference between two texts. `additions` are words
 * present in the revised text but not the original (multiset semantics — two
 * inserted copies of a word that already appeared once still count as one
 * addition); `removals` are the converse.
 */
export function wordMultisetDifference(
  originalText: string,
  revisedText: string,
): WordMultisetDelta {
  const counts = new Map<string, number>();

  for (const token of lexicalTokens(originalText)) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  let additions = 0;

  for (const token of lexicalTokens(revisedText)) {
    const remaining = counts.get(token) ?? 0;

    if (remaining > 0) {
      counts.set(token, remaining - 1);
    } else {
      additions += 1;
    }
  }

  let removals = 0;

  for (const token of lexicalTokens(originalText)) {
    removals += counts.get(token) ?? 0;
    counts.set(token, 0);
  }

  return { additions, removals };
}

/** One authored revision's declared change counts and generated-word attribution. */
export interface TransformationDiff {
  readonly change_counts: ChangeCounts;
  readonly generated_words: number;
}

/**
 * Change counts for one authored revision, from the word-level diff. The
 * mapping is declared, not observed fact: paired word swaps (min of
 * additions/removals) read as `style` changes, net size changes read as
 * `structural`, and `user`/`dictionary`/`snippet` stay zero because no such
 * transformation exists in this app yet. Changes are never labeled corrected
 * errors — without a reference transcript, accuracy is unknown.
 */
export function transformationChanges(rawText: string, revisedText: string): TransformationDiff {
  const { additions, removals } = wordMultisetDifference(rawText, revisedText);
  const replacements = Math.min(additions, removals);

  return {
    change_counts: {
      structural: additions + removals - 2 * replacements,
      user: 0,
      dictionary: 0,
      snippet: 0,
      style: replacements,
    },
    // Words that exist in the revision because of the transformation.
    generated_words: additions,
  };
}

/** The reporting timezone this app declares on every capture. */
export function localReportingTimezone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}

/**
 * Sample frames and rate of a canonical stored WAV. The dictation store
 * keeps 16 kHz mono PCM16, but the frames are read from the file itself —
 * the honest sample_count the capture metric divides by the actual rate.
 */
export async function wavCaptureStats(
  wav: Blob,
): Promise<{ readonly sampleCount: number; readonly sampleRate: number }> {
  const decoded = decodePcm16Wav(new Uint8Array(await wav.arrayBuffer()));

  return {
    sampleCount: decoded.samples.length / (decoded.channels ?? 1),
    sampleRate: decoded.sampleRate,
  };
}

export interface InsightRecorderOptions {
  /** Injectable clock for deterministic tests. */
  readonly now?: () => Date;
  /** Injectable id source for delivery identities. */
  readonly uuid?: () => string;
}

export interface CaptureFinalizedInput {
  readonly captureId: string;
  readonly sampleCount: number;
  readonly sampleRate: number;
  readonly completeAudio: boolean;
  readonly modeId?: string;
  readonly reportingTimezone?: string;
}

export interface RecognitionSelectedInput {
  readonly captureId: string;
  /** The selected transcript text; only its word counts enter the event. */
  readonly transcriptText: string;
  /** Measured Stop-press-to-ready wait; null when this attempt has none. */
  readonly postStopReadyMs: number | null;
}

export interface TransformationCompletedInput {
  readonly captureId: string;
  readonly rawText: string;
  readonly revisedText: string;
}

export interface DeliveryRecordedInput {
  readonly captureId: string;
  readonly status: DeliveryStatus;
  /** The delivered text; only its word count enters the event. */
  readonly outputText: string;
  /**
   * The revision/baseline pair for generated-word attribution: additions of
   * the revision over the baseline are the words that exist because of
   * generation. Omit both when the delivery is pure recognized speech.
   */
  readonly revisedText?: string;
  readonly baselineText?: string;
}

/**
 * Records insight events against a durable store, keeping an in-memory
 * mirror for sequence derivation (selection_seq, revision numbering) and for
 * the UI's snapshot. Writes serialize on a chain so event order in the store
 * follows emission order, and the store's replay/conflict contract — not
 * this class — decides what a duplicate emission means.
 */
export class InsightRecorder {
  private readonly store: InsightEventStore;
  private readonly now: () => Date;
  private readonly uuid: () => string;
  private readonly eventsById = new Map<string, InsightEvent>();
  private writeChain: Promise<void> = Promise.resolve();
  private loadPromise: Promise<void> | undefined;

  constructor(store: InsightEventStore, options: InsightRecorderOptions = {}) {
    this.store = store;
    this.now = options.now ?? (() => new Date());
    this.uuid =
      options.uuid ??
      (() =>
        globalThis.crypto?.randomUUID?.() ??
        `id-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  }

  /**
   * Load the durable log into the mirror once. Replays are idempotent;
   * damaged stored records were already quarantined by the store's load().
   */
  async load(): Promise<void> {
    this.loadPromise ??= this.store.load().then((log) => {
      for (const event of log.events) this.eventsById.set(event.event_id, event);
    });

    return this.loadPromise;
  }

  /** The mirrored event log, insertion-ordered — the population to aggregate. */
  snapshot(): readonly InsightEvent[] {
    return [...this.eventsById.values()];
  }

  /** One retained take exists: its audio is durably owned by a session. */
  async captureFinalized(input: CaptureFinalizedInput): Promise<void> {
    const event: CaptureFinalizedEvent = {
      schema_version: 1,
      event_id: `cf-${input.captureId}`,
      capture_id: input.captureId,
      occurred_at: this.now().toISOString(),
      type: "capture_finalized",
      sample_count: input.sampleCount,
      sample_rate: input.sampleRate,
      complete_audio: input.completeAudio,
      mode_id: input.modeId ?? DEFAULT_MODE_ID,
      reporting_timezone: input.reportingTimezone ?? localReportingTimezone(),
    };

    await this.append(event);
  }

  /**
   * A recognition attempt settled as the capture's selected transcript.
   * Sequence numbers derive from the log, so a retry replaces the previous
   * selection (higher seq) and a replayed settle cannot double-count.
   */
  async recognitionSelected(input: RecognitionSelectedInput): Promise<void> {
    const seq = this.nextSequence(input.captureId, "recognition_selected", (event) =>
      isRecognitionSelected(event) ? event.selection_seq : undefined,
    );

    const { lexical, raw } = segmentWordCounts(input.transcriptText);

    const event: RecognitionSelectedEvent = {
      schema_version: 1,
      event_id: `rs-${input.captureId}-${seq}`,
      capture_id: input.captureId,
      occurred_at: this.now().toISOString(),
      type: "recognition_selected",
      attempt_id: `att-${input.captureId}-${seq}`,
      selection_seq: seq,
      lexical_words: lexical,
      raw_words: raw,
      tokenizer: TOKENIZER_ID,
      post_stop_ready_ms: input.postStopReadyMs,
    };

    await this.append(event);
  }

  /**
   * One explicit refinement pass completed. Each pass is its own revision
   * (distinct revision_id, seq 1): re-refining adds a pass, it does not
   * rewrite history, and the change counts describe the diff between the raw
   * transcript and the revised copy.
   */
  async transformationCompleted(input: TransformationCompletedInput): Promise<void> {
    const passCount = this.snapshot().filter(
      (event) => isTransformationCompleted(event) && event.capture_id === input.captureId,
    ).length;

    const { change_counts } = transformationChanges(input.rawText, input.revisedText);

    const event: TransformationCompletedEvent = {
      schema_version: 1,
      event_id: `tc-${input.captureId}-${passCount + 1}`,
      capture_id: input.captureId,
      occurred_at: this.now().toISOString(),
      type: "transformation_completed",
      revision_id: `rev-${input.captureId}-${passCount + 1}`,
      revision_seq: 1,
      transformation_kind: "model_authoring",
      formula_version: 1,
      change_counts,
    };

    await this.append(event);
  }

  /**
   * A delivered output: a clipboard copy that resolved is `confirmed`; a
   * file export is only `submitted_unconfirmed` because the download's
   * completion is not observable; a failed copy is `failed`. Generated words
   * are attributed via the revision/baseline pair when one is supplied.
   */
  async deliveryRecorded(input: DeliveryRecordedInput): Promise<void> {
    const outputWords = segmentWordCounts(input.outputText).lexical;

    const generatedWords =
      input.revisedText !== undefined && input.baselineText !== undefined
        ? wordMultisetDifference(input.baselineText, input.revisedText).additions
        : 0;

    const id = this.uuid();

    const event: DeliveryRecordedEvent = {
      schema_version: 1,
      event_id: `dl-${id}`,
      capture_id: input.captureId,
      occurred_at: this.now().toISOString(),
      type: "delivery_recorded",
      delivery_id: `d-${id}`,
      delivery_seq: 1,
      status: input.status,
      output_words: outputWords,
      generated_words: Math.min(generatedWords, outputWords),
    };

    await this.append(event);
  }

  /**
   * The confirmed-delete path's tombstone. Idempotent per capture: a second
   * deletion of the same capture is a no-op, and once a tombstone exists,
   * stale replays of the capture's events can never resurrect its totals.
   */
  async captureDeleted(captureId: string): Promise<void> {
    const existing = this.tombstoneFor(captureId);

    if (existing !== undefined) return;

    const event: InsightEvent = {
      schema_version: 1,
      event_id: `cd-${captureId}`,
      capture_id: captureId,
      occurred_at: this.now().toISOString(),
      type: "capture_deleted",
    };

    await this.append(event);
  }

  /** Reset: aggregation starts over from an empty event log. */
  async reset(): Promise<void> {
    await this.enqueue(async () => {
      await this.store.clear();
      this.eventsById.clear();
    });
  }

  private tombstoneFor(captureId: string): InsightEvent | undefined {
    return this.snapshot().find(
      (event) => event.type === "capture_deleted" && event.capture_id === captureId,
    );
  }

  private nextSequence(
    captureId: string,
    kind: InsightEvent["type"],
    sequenceOf: (event: InsightEvent) => number | undefined,
  ): number {
    let highest = 0;

    for (const event of this.eventsById.values()) {
      if (event.type !== kind || event.capture_id !== captureId) continue;

      const seq = sequenceOf(event);

      if (seq !== undefined && seq > highest) highest = seq;
    }

    return highest + 1;
  }

  private async append(event: InsightEvent): Promise<void> {
    const existing = this.eventsById.get(event.event_id);

    if (existing !== undefined) {
      if (!sameEventPayload(existing, event)) {
        throw new InsightEventConflictError(
          event.event_id,
          `Same event ID ${event.event_id} has conflicting payloads`,
        );
      }

      return;
    }

    // Nothing invalid may ever be persisted; the closed branches are the
    // privacy contract, so validation happens before the write is queued.
    if (insightEventProblems(event).length > 0) {
      throw new InsightEventValidationError(
        `event does not conform to the insight event schema: ${event.event_id}`,
      );
    }

    await this.enqueue(async () => {
      await this.store.append(event);
      this.eventsById.set(event.event_id, event);
    });
  }

  private enqueue(work: () => Promise<void>): Promise<void> {
    const run = this.writeChain.then(work, work);

    this.writeChain = run.catch(() => {
      /* the chain survives a failed write; the caller sees the rejection */
    });

    return run;
  }
}
