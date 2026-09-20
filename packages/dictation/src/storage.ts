import { Data, Option, Predicate, Schema } from "effect";

import { STARLING_SAMPLE_RATE, wav16kHeader } from "./audio.js";
import { TranscriptionResultSchema, type TranscriptionResult } from "./client.js";

export const DICTATION_SESSION_SCHEMA_VERSION = 1;

/**
 * IndexedDB schema version. 2 adds the streaming-capture journal stores; the
 * `sessions` records themselves are unchanged, so persisted v1 sessions load
 * without migration.
 */
const DICTATION_DATABASE_VERSION = 2;

export type DictationSessionStatus = "captured" | "transcribing" | "transcribed" | "failed";

const NonNegativeFinite = Schema.Finite.pipe(Schema.check(Schema.isGreaterThanOrEqualTo(0)));

/**
 * A refined transcript: the output of one explicit, opt-in refinement
 * request, stored beside the raw transcript — never in place of it. Text and
 * model are non-empty by contract (the refinement client refuses empty
 * content and requires a model), so an empty value here is damage, not data.
 */
export const RefinedTranscriptSchema = Schema.Struct({
  text: Schema.NonEmptyString,
  /** The chat model that produced this refinement, shown with the text. */
  model: Schema.NonEmptyString,
  createdAt: NonNegativeFinite,
  /**
   * Captured base identity (B11): the id of the thread member whose refined
   * text was the context this refinement built on. Absent for standalone
   * refinements and thread heads — anything refined without prior context —
   * so the base each saved edit used stays recorded and explainable instead
   * of being recomputed from whatever order the listing happens to read in.
   * Additive and optional like the fields around it: no schema version bump.
   */
  contextSourceId: Schema.optional(Schema.NonEmptyString),
});

export type RefinedTranscript = (typeof RefinedTranscriptSchema)["Type"];

/**
 * One recognition attempt's stored result: the wire `TranscriptionResult`
 * plus the provenance of the attempt that produced it (B04). The extra
 * fields are additive and optional — history entries persisted before they
 * existed decode unchanged, so no session schema or database version bump —
 * and both `transcript` and every `transcriptHistory` entry use this one
 * type: a successful retranscription moves the current attempt here with
 * its provenance still attached, so prior attempts are preserved as
 * separate, explainable entities instead of being erased.
 */
export const TranscriptAttemptSchema = Schema.Struct({
  ...TranscriptionResultSchema.fields,
  /** The transcription model this attempt was sent to, when the caller knows it. */
  model: Schema.optional(Schema.NonEmptyString),
  /** The wire protocol this attempt used ("starling" or "openai"). */
  protocol: Schema.optional(Schema.NonEmptyString),
  /** When the store settled this attempt's result, as an ISO timestamp. */
  savedAt: Schema.optional(Schema.NonEmptyString),
});

export type TranscriptAttempt = (typeof TranscriptAttemptSchema)["Type"];

export const DictationSessionSchema = Schema.Struct({
  id: Schema.NonEmptyString,
  createdAt: Schema.NonEmptyString,
  updatedAt: Schema.NonEmptyString,
  status: Schema.Literals(["captured", "transcribing", "transcribed", "failed"]),
  /** Canonical PCM16 16 kHz WAV. Retained until delete() is called. */
  wav: Schema.instanceOf(Blob),
  durationMs: Schema.optional(NonNegativeFinite),
  attemptCount: Schema.Natural,
  transcript: Schema.optional(TranscriptAttemptSchema),
  /**
   * Earlier recognition attempts, retained when a retry succeeds (B04):
   * each entry is its own result with its own provenance, never erased by a
   * later attempt.
   */
  transcriptHistory: Schema.optional(Schema.Array(TranscriptAttemptSchema)),
  lastError: Schema.optional(Schema.String),
  /** True when the final transcript arrived over the streaming connection. */
  streamed: Schema.optional(Schema.Boolean),
  /** Why live streaming failed here; kept for review even after a batch retry succeeds. */
  streamError: Schema.optional(Schema.String),
  /**
   * Optional LLM-refined copy of `transcript.text`, written only by an
   * explicit user action. Additive and optional like `streamed`, so records
   * persisted before this field existed decode unchanged — no session schema
   * or database version bump — and re-refining overwrites it in place.
   */
  refined: Schema.optional(RefinedTranscriptSchema),
  /**
   * Thread this take was explicitly assigned to for multi-turn refinement
   * (#117). Additive and optional like `refined`, so records persisted before
   * threads existed decode unchanged — no session schema or database version
   * bump. Purely a label for chaining refinement context: it never changes
   * which transcript is primary, and a take keeps its own raw transcript.
   */
  threadId: Schema.optional(Schema.NonEmptyString),
  /**
   * When this take was appended to its thread, in epoch ms (B11): the
   * explicit append sequence. createdAt stays the recording's own capture
   * time — accurate and untouched — while this stamp is the order turns
   * joined the conversation, so an older recording that joins later reads as
   * the thread's latest turn instead of rewinding the reading order before
   * the head. Additive and optional like `threadId`, so records persisted
   * before the sequence existed decode unchanged — they simply sort as the
   * oldest appends — and no session schema or database version bump is
   * needed.
   */
  threadJoinedAt: Schema.optional(NonNegativeFinite),
});

export type DictationSession = (typeof DictationSessionSchema)["Type"];

export interface CreateSessionInput {
  readonly id?: string | undefined;
  readonly wav: Blob;
  readonly durationMs?: number | undefined;
}

export interface SaveTranscriptOptions {
  /** Mark the transcript as delivered by the live streaming path. */
  readonly streamed?: boolean;
  /**
   * The model this attempt was sent to (B04). Recorded beside the stored
   * transcript and carried into `transcriptHistory` when a later attempt
   * supersedes it. Optional like the provenance itself: callers without a
   * known model omit it.
   */
  readonly model?: string;
  /** The wire protocol this attempt used (B04); recorded like `model`. */
  readonly protocol?: string;
}

export interface InvalidStoredSession {
  /** Best-effort id taken from the raw record; "(unknown id)" when absent. */
  readonly id: string;
  /**
   * The record's raw IndexedDB key, captured when the listing reads it. The
   * reported id is best-effort — an "(unknown id)" entry's key can be
   * non-string (a number) or empty under `keyPath: "id"` — so dismissal
   * deletes by this key, never by the id. Absent for listings that never
   * read keys (the memory store never quarantines, so it never sets one).
   */
  readonly key?: IDBValidKey | undefined;
  /** Why the record failed schema validation. */
  readonly cause: unknown;
  /**
   * The raw IndexedDB record, retained untouched in the database. Damaged
   * entries are quarantined out of the listing — never silently deleted — so
   * recoverable audio stays exportable until explicitly dismissed.
   */
  readonly record: unknown;
}

export interface DictationHistory {
  readonly sessions: readonly DictationSession[];
  readonly invalid: readonly InvalidStoredSession[];
}

export interface DictationSessionStore {
  create(input: CreateSessionInput): Promise<DictationSession>;
  get(id: string): Promise<DictationSession | undefined>;
  /**
   * Healthy sessions, newest first. Records that fail validation are
   * skipped — never deleted — and reported via `listReport()`.
   */
  list(): Promise<readonly DictationSession[]>;
  /**
   * `list()` plus per-record diagnostics for entries that failed
   * validation. The raw records stay in the database, so a single damaged
   * entry can neither hide healthy history nor block journal recovery.
   */
  listReport(): Promise<DictationHistory>;
  markAttempt(id: string): Promise<DictationSession>;
  saveTranscript(
    id: string,
    transcript: TranscriptionResult,
    options?: SaveTranscriptOptions,
  ): Promise<DictationSession>;
  /**
   * Attach (or overwrite) the refined transcript produced by one explicit
   * refinement request. Never touches `transcript`, `transcriptHistory`, or
   * `status`: the raw transcript stays the primary record and refinement is
   * a separate, labeled artifact on the same session.
   */
  saveRefinedTranscript(id: string, refined: RefinedTranscript): Promise<DictationSession>;
  /**
   * Assign the session to a refinement thread (#117). Like
   * `saveRefinedTranscript` this is written only by an explicit user action,
   * bumps `updatedAt`, and touches nothing else — the raw transcript, its
   * history, any refined copy, and the status all stay exactly as they were.
   * Membership is a label for chaining refinement context, never a mutation
   * of the take's own records, and a join also stamps `threadJoinedAt`
   * (B11): the append sequence that orders the thread's turns, so an older
   * recording joining an existing thread appends as the latest turn instead
   * of rewinding the reading order before the head. Stamping happens
   * exactly once per join: an idempotent repeat on the thread the take
   * already belongs to keeps the stamp it has — absent for members that
   * joined before the sequence existed, which stay sorted as the oldest
   * appends. Assignment is permanent for the session's lifetime: there is
   * deliberately no un-assign, and moving a take between threads can only
   * arrive as a future explicit feature — a re-assignment to a different id
   * supersedes the label and re-stamps the append, but nothing ever clears
   * it.
   */
  assignThread(id: string, threadId: string): Promise<DictationSession>;
  /**
   * Downgrade an in-flight session (`transcribing`/`captured`) to `failed`.
   * A session a live owner already settled keeps its state, so a sweep
   * racing a settling write cannot overwrite it (#162).
   */
  saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession>;
  /** Record why live streaming failed without changing the session status. */
  noteStreamError(id: string, message: string): Promise<DictationSession>;
  /**
   * True while a live owner (this window or another) is transcribing the
   * session right now. Startup sweeps consult this so a second window does
   * not interrupt another owner's in-flight attempt (#144).
   */
  transcriptionInFlight(id: string): Promise<boolean>;
  delete(id: string): Promise<void>;
  /**
   * Dismiss one quarantined history entry by its raw record key, as reported
   * on `InvalidStoredSession.key`. The record is re-read and re-validated in
   * the same transaction as the delete: an entry another window repaired
   * since the listing is kept, since the damaged record the user saw is no
   * longer what the key holds (#155). Resolves when the key holds nothing,
   * a healthy record, or a repaired entry — dismissal is best-effort, never
   * an error. Healthy sessions are outside this method's reach; use delete().
   */
  deleteInvalid(key: IDBValidKey): Promise<void>;
}

export interface DictationSessionManifest {
  readonly schemaVersion: typeof DICTATION_SESSION_SCHEMA_VERSION;
  readonly id: string;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly status: DictationSessionStatus;
  readonly audioFile: "recording.wav";
  readonly durationMs?: number;
  readonly attemptCount: number;
  readonly transcript?: TranscriptAttempt;
  readonly transcriptHistory?: readonly TranscriptAttempt[];
  readonly lastError?: string;
  readonly streamed?: boolean;
  readonly streamError?: string;
  readonly refined?: RefinedTranscript;
  readonly threadId?: string;
  readonly threadJoinedAt?: number;
}

export interface DictationSessionExport {
  readonly manifest: DictationSessionManifest;
  readonly wav: Blob;
}

interface ManifestDraft {
  schemaVersion: typeof DICTATION_SESSION_SCHEMA_VERSION;
  id: string;
  createdAt: string;
  updatedAt: string;
  status: DictationSessionStatus;
  audioFile: "recording.wav";
  durationMs?: number;
  attemptCount: number;
  transcript?: TranscriptAttempt;
  transcriptHistory?: readonly TranscriptAttempt[];
  lastError?: string;
  streamed?: boolean;
  streamError?: string;
  refined?: RefinedTranscript;
  threadId?: string;
  threadJoinedAt?: number;
}

export class DictationStorageError extends Data.TaggedError("DictationStorageError")<{
  readonly description: string;
  readonly cause: ErrorOptions["cause"];
}> {
  constructor(message: string, options?: ErrorOptions) {
    super({ description: message, cause: options?.cause });
  }

  override get message(): string {
    return this.description;
  }
}

export class DictationSessionNotFoundError extends DictationStorageError {
  override readonly name = "DictationSessionNotFoundError";

  constructor(readonly sessionId: string) {
    super(`dictation session ${sessionId} was not found`);
  }
}

function sessionId(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  );
}

const encodeDefect = Schema.encodeUnknownSync(Schema.Defect());

function errorText<Cause>(cause: Cause): string {
  if (Predicate.isError(cause)) return cause.message || cause.name;

  if (Predicate.isString(cause)) return cause;

  const serialized = JSON.stringify(encodeDefect(cause));

  return serialized ?? String(cause);
}

/**
 * The shared id rules: non-empty and single-line. Both the session id and
 * the refinement-thread id are checked through this one helper so their
 * contracts (and the error wording callers rely on) cannot drift apart.
 */
function assertNonEmptySingleLine(value: string, label: string): void {
  if (!value || /[\r\n]/.test(value)) {
    throw new TypeError(`${label} must be non-empty and contain no newlines`);
  }
}

function assertCreateInput(input: CreateSessionInput): void {
  if (!(input.wav instanceof Blob) || input.wav.size === 0) {
    throw new TypeError("wav must be a non-empty Blob");
  }

  if (input.id !== undefined) assertNonEmptySingleLine(input.id, "session id");

  if (
    input.durationMs !== undefined &&
    (!Number.isFinite(input.durationMs) || input.durationMs < 0)
  ) {
    throw new TypeError("durationMs must be a non-negative finite number");
  }
}

/**
 * Thread ids share the session-id rules through assertNonEmptySingleLine so
 * a value the schema's NonEmptyString would quarantine on its next read is
 * rejected at the write boundary in both stores, memory included.
 */
function assertThreadId(threadId: string): void {
  assertNonEmptySingleLine(threadId, "thread id");
}

interface TranscriptDraft {
  text: string;
  segments: TranscriptionResult["segments"];
  durationSeconds?: number;
  requestId?: string;
  model?: string;
  protocol?: string;
  savedAt?: string;
}

function freezeTranscript(value: TranscriptAttempt): TranscriptAttempt {
  const segments = Object.freeze(value.segments.map((segment) => Object.freeze({ ...segment })));

  const transcript: TranscriptDraft = {
    text: value.text,
    segments,
  };

  if (value.durationSeconds !== undefined) transcript.durationSeconds = value.durationSeconds;

  if (value.requestId !== undefined) transcript.requestId = value.requestId;

  if (value.model !== undefined) transcript.model = value.model;

  if (value.protocol !== undefined) transcript.protocol = value.protocol;

  if (value.savedAt !== undefined) transcript.savedAt = value.savedAt;

  return Object.freeze(transcript);
}

/**
 * Validate one attempt's optional provenance at the write boundary: a label
 * the schema's NonEmptyString would quarantine on its next read is rejected
 * before it can damage the session, in both stores, memory included.
 */
function assertAttemptProvenance(options?: SaveTranscriptOptions): void {
  if (options?.model !== undefined && options.model === "") {
    throw new TypeError("transcript attempt model must be a non-empty string when provided");
  }

  if (options?.protocol !== undefined && options.protocol === "") {
    throw new TypeError("transcript attempt protocol must be a non-empty string when provided");
  }
}

/**
 * The stored entry for one settling attempt (B04): the wire result plus the
 * store-stamped settle time and whatever provenance the caller supplied.
 * `savedAt` is always stamped — time provenance needs no caller opt-in —
 * while model and protocol ride along only when known.
 */
function attemptEntry(
  transcript: TranscriptionResult,
  options?: SaveTranscriptOptions,
): TranscriptAttempt {
  assertAttemptProvenance(options);

  const entry: TranscriptDraft = {
    text: transcript.text,
    segments: transcript.segments,
    savedAt: new Date().toISOString(),
  };

  if (transcript.durationSeconds !== undefined) {
    entry.durationSeconds = transcript.durationSeconds;
  }

  if (transcript.requestId !== undefined) entry.requestId = transcript.requestId;

  if (options?.model !== undefined) entry.model = options.model;

  if (options?.protocol !== undefined) entry.protocol = options.protocol;

  return Object.freeze(entry);
}

/**
 * The fields every settling `saveTranscript` writes, shared by both stores so
 * provenance stamping cannot drift between them.
 */
function transcriptSettling(
  transcript: TranscriptionResult,
  options?: SaveTranscriptOptions,
): SessionUpdate {
  const update: SessionUpdate = {
    status: "transcribed",
    transcript: attemptEntry(transcript, options),
    lastError: undefined,
  };

  if (options?.streamed === true) update.streamed = true;

  return update;
}

interface SessionDraft {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: DictationSessionStatus;
  wav: Blob;
  durationMs?: number | undefined;
  attemptCount: number;
  transcript?: TranscriptAttempt | undefined;
  transcriptHistory?: readonly TranscriptAttempt[] | undefined;
  lastError?: string | undefined;
  streamed?: boolean | undefined;
  streamError?: string | undefined;
  refined?: RefinedTranscript | undefined;
  threadId?: string | undefined;
  threadJoinedAt?: number | undefined;
}

/**
 * Mutable draft for one refined copy: the SessionDraft pattern, so the
 * optional captured-base identity (B11) is added only when the refinement
 * actually had prior context, never as a present-but-undefined field. The
 * one shared shape for callers drafting a refined transcript — the app's
 * refine path included — so it cannot drift from what freezeRefined writes.
 */
export interface RefinedDraft {
  text: string;
  model: string;
  createdAt: number;
  contextSourceId?: string;
}

function freezeRefined(value: RefinedTranscript): RefinedTranscript {
  const refined: RefinedDraft = {
    text: value.text,
    model: value.model,
    createdAt: value.createdAt,
  };

  if (value.contextSourceId !== undefined) refined.contextSourceId = value.contextSourceId;

  return Object.freeze(refined);
}

function freezeSession(value: DictationSession): DictationSession {
  const session: SessionDraft = {
    id: value.id,
    createdAt: value.createdAt,
    updatedAt: value.updatedAt,
    status: value.status,
    wav: value.wav,
    transcriptHistory: Object.freeze((value.transcriptHistory ?? []).map(freezeTranscript)),
    attemptCount: value.attemptCount,
  };

  if (value.durationMs !== undefined) session.durationMs = value.durationMs;

  if (value.transcript !== undefined) session.transcript = freezeTranscript(value.transcript);

  if (value.lastError !== undefined) session.lastError = value.lastError;

  if (value.streamed !== undefined) session.streamed = value.streamed;

  if (value.streamError !== undefined) session.streamError = value.streamError;

  if (value.refined !== undefined) session.refined = freezeRefined(value.refined);

  if (value.threadId !== undefined) session.threadId = value.threadId;

  if (value.threadJoinedAt !== undefined) session.threadJoinedAt = value.threadJoinedAt;

  return Object.freeze(session);
}

function initialSession(input: CreateSessionInput): DictationSession {
  assertCreateInput(input);

  const now = new Date().toISOString();

  const session: SessionDraft = {
    id: input.id ?? sessionId(),
    createdAt: now,
    updatedAt: now,
    status: "captured",
    wav: input.wav,
    attemptCount: 0,
  };

  if (input.durationMs !== undefined) session.durationMs = input.durationMs;

  return freezeSession(session);
}

/** Mutable draft of the fields `updatedSession` may change. */
type SessionUpdate = Partial<{
  status: DictationSessionStatus;
  attemptCount: number;
  transcript: TranscriptAttempt;
  lastError: string | undefined;
  streamed: boolean;
  streamError: string;
  refined: RefinedTranscript;
  threadId: string;
  threadJoinedAt: number;
}>;

/**
 * The fields one explicit thread assignment writes: the label plus the append
 * sequence stamp (B11). Appending to a thread stamps now — even when the
 * recording itself is older than the thread's head — so the reading order
 * follows the conversation, not the recording clock. Re-assigning to the
 * thread the take already belongs to writes nothing but the label: the join
 * already happened, so an idempotent repeat keeps the stamp exactly as it
 * is — a real stamp unchanged, and a member that joined before the sequence
 * existed stays unstamped instead of being silently moved to the thread's
 * end. Only a move to a different thread is a fresh append there and stamps
 * again.
 */
function threadAssignment(current: DictationSession, threadId: string): SessionUpdate {
  if (current.threadId === threadId) return { threadId };

  return { threadId, threadJoinedAt: Date.now() };
}

function updatedSession(current: DictationSession, update: SessionUpdate): DictationSession {
  return freezeSession({
    ...current,
    ...update,
    transcriptHistory:
      update.transcript !== undefined && current.transcript !== undefined
        ? [...(current.transcriptHistory ?? []), current.transcript]
        : current.transcriptHistory,
    updatedAt: new Date().toISOString(),
  });
}

export function exportDictationSession(session: DictationSession): DictationSessionExport {
  const manifest: ManifestDraft = {
    schemaVersion: DICTATION_SESSION_SCHEMA_VERSION,
    id: session.id,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    status: session.status,
    audioFile: "recording.wav",
    transcriptHistory: Object.freeze((session.transcriptHistory ?? []).map(freezeTranscript)),
    attemptCount: session.attemptCount,
  };

  if (session.durationMs !== undefined) manifest.durationMs = session.durationMs;

  if (session.transcript !== undefined) manifest.transcript = freezeTranscript(session.transcript);

  if (session.lastError !== undefined) manifest.lastError = session.lastError;

  if (session.streamed !== undefined) manifest.streamed = session.streamed;

  if (session.streamError !== undefined) manifest.streamError = session.streamError;

  if (session.refined !== undefined) manifest.refined = freezeRefined(session.refined);

  if (session.threadId !== undefined) manifest.threadId = session.threadId;

  if (session.threadJoinedAt !== undefined) manifest.threadJoinedAt = session.threadJoinedAt;

  return Object.freeze({ manifest: Object.freeze(manifest), wav: session.wav });
}

const decodeRescuedWavRecord = Schema.decodeUnknownOption(
  Schema.Struct({ wav: Schema.instanceOf(Blob) }),
);

/**
 * Best-effort audio rescue for a quarantined history entry: the raw record
 * is untouched by `listReport()`, so a damaged entry whose `wav` is still a
 * non-empty Blob can be downloaded before the user deletes it. Returns
 * undefined when no usable audio survives.
 */
export function invalidSessionWav(entry: InvalidStoredSession): Blob | undefined {
  const decoded = decodeRescuedWavRecord(entry.record);

  return Option.isSome(decoded) && decoded.value.wav.size > 0 ? decoded.value.wav : undefined;
}

export interface CreateStreamCaptureInput {
  readonly id?: string | undefined;
  readonly createdAt?: string | undefined;
}

export interface FinishedStreamCapture {
  readonly wav: Blob;
  readonly durationMs: number;
  /** Present unless durable persistence failed; then `wav` is the only copy. */
  readonly session?: DictationSession | undefined;
  readonly failure?: unknown;
}

/**
 * A recording being persisted chunk by chunk while it is captured. Chunks are
 * raw PCM16 16 kHz mono bytes; the canonical WAV header is stamped on
 * assembly, so the journal plus `wav16kHeader` reproduce `encodeWav16k`
 * output exactly.
 */
export interface DictationStreamCapture {
  readonly sessionId: string;
  /**
   * Journal one chunk. Rejects when durable persistence fails; the in-memory
   * mirror is still retained so `finish` can assemble the WAV.
   */
  append(pcm16: Uint8Array): Promise<void>;
  /** Frames journaled so far. */
  appendedFrames(): number;
  /**
   * Assemble the WAV and persist the session (source-of-truth save). The
   * cancellation probe runs after every await, before each durable write:
   * when it reports true the session stays unwritten and the provisional
   * journal is cleaned up, so a discarded take cannot finalize (#160).
   */
  finish(durationMs?: number, isCancelled?: () => boolean): Promise<FinishedStreamCapture>;
  /** Drop the journal without creating a session (recording discarded). */
  abandon(): Promise<void>;
}

export interface DictationStreamCaptureStore {
  beginStreamCapture(input?: CreateStreamCaptureInput): Promise<DictationStreamCapture>;
  /** Assemble journals orphaned by a crash mid-recording into retryable sessions. */
  recoverStreamCaptures(): Promise<readonly DictationSession[]>;
}

function assertPcm16Chunk(pcm16: Uint8Array): void {
  if (!(pcm16 instanceof Uint8Array) || pcm16.byteLength === 0 || pcm16.byteLength % 2 !== 0) {
    throw new TypeError("stream capture chunks must be non-empty PCM16 (an even number of bytes)");
  }
}

function assemblePcm16Wav(chunks: readonly Uint8Array[]) {
  const dataBytes = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const parts: BlobPart[] = [wav16kHeader(dataBytes)];

  for (const chunk of chunks) {
    // Copy into a fresh ArrayBuffer-backed view: IndexedDB can hand back
    // SharedArrayBuffer-backed buffers, which Blob rejects.
    parts.push(new Uint8Array(chunk));
  }

  return {
    wav: new Blob(parts, { type: "audio/wav" }),
    durationMs: (dataBytes / 2 / STARLING_SAMPLE_RATE) * 1_000,
  };
}

const RECOVERED_CAPTURE_MESSAGE =
  "Recovered after the app closed during recording. The audio is intact; retry when ready.";

/**
 * The retryable session an abandoned journal promotes into: already marked
 * failed with the recovery note, exactly as a create-then-fail sequence
 * would have left it — but written in one transaction with the journal
 * delete, so promotion is atomic.
 */
function recoveredSession(
  captureId: string,
  assembled: ReturnType<typeof assemblePcm16Wav>,
): DictationSession {
  const now = new Date().toISOString();

  return freezeSession({
    id: captureId,
    createdAt: now,
    updatedAt: now,
    status: "failed",
    wav: assembled.wav,
    durationMs: assembled.durationMs,
    attemptCount: 0,
    lastError: RECOVERED_CAPTURE_MESSAGE,
  });
}

function streamCaptureId(input: CreateStreamCaptureInput): string {
  const id = input.id ?? sessionId();

  if (!id || /[\r\n]/.test(id)) {
    throw new TypeError("stream capture id must be non-empty and contain no newlines");
  }

  return id;
}

class MemoryStreamCapture implements DictationStreamCapture {
  readonly sessionId: string;
  private readonly store: MemorySessionStore;
  private readonly chunks: Uint8Array[] = [];
  private closed = false;

  constructor(store: MemorySessionStore, sessionId: string) {
    this.store = store;
    this.sessionId = sessionId;
  }

  async append(pcm16: Uint8Array): Promise<void> {
    assertPcm16Chunk(pcm16);

    if (this.closed)
      throw new DictationStorageError(`stream capture ${this.sessionId} already closed`);
    this.chunks.push(pcm16);
  }

  appendedFrames(): number {
    return this.chunks.reduce((frames, chunk) => frames + chunk.byteLength / 2, 0);
  }

  async finish(durationMs?: number, isCancelled?: () => boolean): Promise<FinishedStreamCapture> {
    this.closed = true;
    const assembled = assemblePcm16Wav(this.chunks);
    const duration = durationMs ?? assembled.durationMs;

    // The session is the only durable write this path makes; a take
    // discarded while assembly awaited must not be persisted (#160).
    if (isCancelled?.()) {
      this.chunks.length = 0;

      return Object.freeze({ wav: assembled.wav, durationMs: duration, session: undefined });
    }

    try {
      const session = await this.store.create({
        id: this.sessionId,
        wav: assembled.wav,
        durationMs: duration,
      });

      return Object.freeze({ wav: assembled.wav, durationMs: duration, session });
    } catch (failure) {
      return Object.freeze({ wav: assembled.wav, durationMs: duration, failure });
    }
  }

  async abandon(): Promise<void> {
    this.closed = true;
    this.chunks.length = 0;
  }
}

export class MemorySessionStore implements DictationSessionStore, DictationStreamCaptureStore {
  private readonly sessions = new Map<string, DictationSession>();
  private readonly transcribing = new Set<string>();

  async create(input: CreateSessionInput): Promise<DictationSession> {
    const session = initialSession(input);

    if (this.sessions.has(session.id))
      throw new DictationStorageError(`dictation session ${session.id} already exists`);
    this.sessions.set(session.id, session);

    return session;
  }

  async get(id: string): Promise<DictationSession | undefined> {
    return this.sessions.get(id);
  }

  async list(): Promise<readonly DictationSession[]> {
    return Object.freeze(
      [...this.sessions.values()].sort((left, right) =>
        right.updatedAt.localeCompare(left.updatedAt),
      ),
    );
  }

  async listReport(): Promise<DictationHistory> {
    return Object.freeze({ sessions: await this.list(), invalid: Object.freeze([]) });
  }

  async markAttempt(id: string): Promise<DictationSession> {
    const marked = await this.update(id, (current) =>
      updatedSession(current, {
        status: "transcribing",
        attemptCount: current.attemptCount + 1,
        lastError: undefined,
      }),
    );

    this.transcribing.add(id);

    return marked;
  }

  async saveTranscript(
    id: string,
    transcript: TranscriptionResult,
    options?: SaveTranscriptOptions,
  ): Promise<DictationSession> {
    const update = transcriptSettling(transcript, options);

    try {
      return await this.update(id, (current) => updatedSession(current, update));
    } finally {
      this.transcribing.delete(id);
    }
  }

  async saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession> {
    try {
      return await this.update(id, (current) =>
        current.status === "transcribing" || current.status === "captured"
          ? updatedSession(current, {
              status: "failed",
              lastError: errorText(cause),
            })
          : current,
      );
    } finally {
      this.transcribing.delete(id);
    }
  }

  async noteStreamError(id: string, message: string): Promise<DictationSession> {
    return this.update(id, (current) => updatedSession(current, { streamError: message }));
  }

  async saveRefinedTranscript(id: string, refined: RefinedTranscript): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, { refined: freezeRefined(refined) }),
    );
  }

  async assignThread(id: string, threadId: string): Promise<DictationSession> {
    assertThreadId(threadId);

    return this.update(id, (current) =>
      updatedSession(current, threadAssignment(current, threadId)),
    );
  }

  async transcriptionInFlight(id: string): Promise<boolean> {
    return this.transcribing.has(id);
  }

  async delete(id: string): Promise<void> {
    this.sessions.delete(id);
    this.transcribing.delete(id);
  }

  /**
   * The memory store writes through the schema, so no record can be damaged:
   * there is nothing quarantined to dismiss. Kept on the interface so the
   * banner dismisses entries without knowing which store backs it.
   */
  async deleteInvalid(_key: IDBValidKey): Promise<void> {
    return undefined;
  }

  async beginStreamCapture(input: CreateStreamCaptureInput = {}): Promise<DictationStreamCapture> {
    return new MemoryStreamCapture(this, streamCaptureId(input));
  }

  async recoverStreamCaptures(): Promise<readonly DictationSession[]> {
    return [];
  }

  private update(
    id: string,
    transform: (current: DictationSession) => DictationSession,
  ): DictationSession {
    const current = this.sessions.get(id);

    if (!current) throw new DictationSessionNotFoundError(id);
    const next = transform(current);
    this.sessions.set(id, next);

    return next;
  }
}

/** Structural slice of a granted Web Lock handle. */
export interface GrantedWebLock {
  readonly name: string;
}

/** One entry of a lock-manager snapshot: the store only reads names. */
export interface HeldWebLock {
  /**
   * Optional like the DOM LockInfo it mirrors: a snapshot entry without a
   * name can never be an attempt signal, so the probe skips it.
   */
  readonly name?: string;
}

/** The slice of a lock-manager snapshot the store enumerates signals from. */
export interface WebLockSnapshot {
  /**
   * Optional like the DOM LockManagerSnapshot it mirrors: a snapshot
   * without a listing carries no observable signal, so the probe reads none.
   */
  readonly held?: readonly HeldWebLock[];
}

/** Acquisition options the store asks its lock manager for. */
export interface WebLockRequestOptions {
  readonly mode: "exclusive";
  readonly ifAvailable: boolean;
}

/**
 * The slice of the Web Locks API the store depends on: exclusive,
 * never-queueing acquisition plus snapshot enumeration. `navigator.locks`
 * satisfies this structurally; tests substitute an in-memory fake to drive
 * both ownership branches.
 */
export interface WebLocksLike {
  request<Result>(
    name: string,
    options: WebLockRequestOptions,
    granted: (lock: GrantedWebLock | null) => Promise<Result> | Result,
  ): Promise<Result>;
  /**
   * Currently held locks. The store enumerates per-attempt signals through
   * this instead of probing a single name, so every admitted attempt stays
   * visible even after a contender for the same session settles first (#162).
   */
  query(): Promise<WebLockSnapshot>;
}

/**
 * The host's Web Locks manager when it has one. Environments without
 * navigator.locks (Node, older embedders) run unlocked: capture ownership is
 * then tracked only within the environment, which two live stores in one
 * realm honor but two separate windows cannot.
 *
 * The DOM LockManager satisfies WebLocksLike structurally: request() takes
 * the same exclusive/ifAvailable shape, and query() reports held locks under
 * the same `held[].name` shape the store enumerates per-attempt signals
 * through.
 */
function navigatorLocks(): WebLocksLike | undefined {
  return globalThis.navigator?.locks ?? undefined;
}

export interface IndexedDbSessionStoreOptions {
  readonly databaseName?: string;
  readonly indexedDB?: IDBFactory | undefined;
  /**
   * Cross-tab ownership signal for streaming captures and transcription
   * attempts. Defaults to `navigator.locks` when the host provides it.
   * Passing the property with the value `undefined` (as opposed to omitting
   * it) forces the unlocked fallback even in a Web-Locks-capable host.
   */
  readonly webLocks?: WebLocksLike | undefined;
}

const OBJECT_STORE = "sessions";

const STREAM_META_STORE = "stream-captures";

const STREAM_CHUNK_STORE = "stream-chunks";

interface StreamMetaRecord {
  readonly id: string;
  readonly createdAt: string;
  /**
   * Owning store instance, written when the journal is created. Recovery
   * trusts locks and the environment registry for liveness; this records
   * which window's capture a journal was for whoever inspects the store.
   */
  readonly ownerId?: string | undefined;
}

interface StreamChunkRecord {
  readonly captureId: string;
  readonly index: number;
  readonly pcm: Uint8Array;
}

const decodeSession = Schema.decodeUnknownSync(DictationSessionSchema);

const decodeSessionId = Schema.decodeUnknownOption(Schema.Struct({ id: Schema.NonEmptyString }));

/**
 * Mutable draft of a quarantine entry: `key` is attached after the best-effort
 * id is known, only when the listing read the record's raw IndexedDB key.
 */
interface InvalidSessionDraft {
  id: string;
  key?: IDBValidKey | undefined;
  cause: DictationStorageError;
  record: unknown;
}

/**
 * Validate one stored record without touching the database: damaged entries
 * are reported with their raw record so recoverable audio stays exportable.
 * Callers that read the record's key pass it in, so dismissal can delete by
 * the raw key even when the reported id is best-effort ("(unknown id)").
 */
// A raw IndexedDB row is unknown by definition; this function is the boundary that parses it.
function isolateStoredSession(
  record: unknown, // oxlint-disable-line anti-slop/no-unknown-parameters -- see above
  key?: IDBValidKey,
): DictationSession | InvalidStoredSession {
  try {
    return freezeSession(decodeSession(record));
  } catch (cause) {
    const decodedId = decodeSessionId(record);

    const isolated: InvalidSessionDraft = {
      id: Option.isSome(decodedId) ? decodedId.value.id : "(unknown id)",
      cause: invalidStoredSession(cause),
      record,
    };

    if (key !== undefined) isolated.key = key;

    return Object.freeze(isolated);
  }
}

function invalidStoredSession<Cause>(cause: Cause): DictationStorageError {
  return new DictationStorageError("stored dictation session is invalid", { cause });
}

function runIdbRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(new DictationStorageError("an IndexedDB request failed", { cause: request.error }));
  });
}

function runIdbTransaction(
  database: IDBDatabase,
  stores: readonly string[],
  mode: IDBTransactionMode,
  work: (transaction: IDBTransaction) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    // SAFETY: the DOM signature asks for a mutable string[] but IndexedDB
    // never mutates the store-name list it reads.
    const transaction = database.transaction(stores as string[], mode);
    work(transaction);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () =>
      reject(
        new DictationStorageError("an IndexedDB transaction failed", {
          cause: transaction.error,
        }),
      );
    transaction.onabort = () =>
      reject(
        new DictationStorageError("an IndexedDB transaction was aborted", {
          cause: transaction.error,
        }),
      );
  });
}

function captureLockName(databaseName: string, captureId: string): string {
  return `starling:dictation:${databaseName}:capture:${captureId}`;
}

function attemptLockPrefix(databaseName: string): string {
  return `starling:dictation:${databaseName}:transcribe:`;
}

/**
 * One admitted attempt's cross-window signal. The trailing signal is unique
 * per markAttempt (never a separator), so every admitted attempt holds its
 * own lock: a contender that outlives the first settler still reads as
 * in-flight instead of looking abandoned (#162).
 */
function attemptLockName(databaseName: string, sessionId: string, signal: string): string {
  return `${attemptLockPrefix(databaseName)}${sessionId}:${signal}`;
}

/**
 * True when a held lock name is a per-attempt signal for this session. The
 * session id is matched exactly — split at the last separator, since only
 * the signal follows it — so an id containing separators can never collide
 * with another session's signal.
 */
function isAttemptSignalFor(lockName: string, lockPrefix: string, sessionId: string): boolean {
  if (!lockName.startsWith(lockPrefix)) return false;

  const rest = lockName.slice(lockPrefix.length);
  const separator = rest.lastIndexOf(":");

  return separator >= 0 && rest.slice(0, separator) === sessionId;
}

/**
 * Capture journals and transcription attempts currently held open by a live
 * store in this environment. Web Locks own cross-window detection; these
 * registries are the within-environment signal for hosts without
 * navigator.locks, and a second line of defense everywhere else.
 */
const liveCaptureRegistry = new Set<string>();

/**
 * Within-environment attempt signals. Every admitted attempt contributes its
 * signal to its session's set, so concurrent attempts for one session stay
 * visible until each settles; keyed by the exact session key with signals
 * as set members (never name suffixes), so attempts neither hide each other
 * nor collide across sessions.
 */
const liveAttemptSignals = new Map<string, Set<string>>();

function captureRegistryKey(databaseName: string, captureId: string): string {
  return `capture\u0000${databaseName}\u0000${captureId}`;
}

function attemptRegistryKey(databaseName: string, sessionId: string): string {
  return `attempt\u0000${databaseName}\u0000${sessionId}`;
}

/**
 * Fresh identity for one admitted attempt signal. Uniqueness is the whole
 * contract, plus one constraint: never a `:` or NUL separator, so the
 * liveness probe can split the lock name at its last separator and match
 * the session id exactly. randomUUID (and the fallback below) satisfy both.
 */
function attemptSignal(): string {
  return sessionId();
}

/** One owned signal's key in `attemptLocks`: split at the last separator. */
function ownedAttemptKey(id: string, signal: string): string {
  return `${id}\u0000${signal}`;
}

function trackAttemptSignal(databaseName: string, sessionId: string, signal: string): void {
  const key = attemptRegistryKey(databaseName, sessionId);
  let signals = liveAttemptSignals.get(key);

  if (!signals) {
    signals = new Set();
    liveAttemptSignals.set(key, signals);
  }

  signals.add(signal);
}

function discardAttemptSignal(databaseName: string, sessionId: string, signal: string): void {
  const key = attemptRegistryKey(databaseName, sessionId);
  const signals = liveAttemptSignals.get(key);

  if (!signals) return;

  signals.delete(signal);

  if (signals.size === 0) liveAttemptSignals.delete(key);
}

/** One lock handshake: the grant signal plus the held lock's release latch. */
class LockHandshake {
  private settleGranted: ((granted: boolean) => void) | undefined;
  private resolveHeld: (() => void) | undefined;

  readonly granted: Promise<boolean> = new Promise((resolve) => {
    this.settleGranted = resolve;
  });

  readonly held: Promise<void> = new Promise((resolve) => {
    this.resolveHeld = resolve;
  });

  settle(granted: boolean): void {
    this.settleGranted?.(granted);
  }

  release(): void {
    this.resolveHeld?.();
  }
}

/**
 * Take one named lock without queueing: resolves undefined while another
 * owner holds it, otherwise a function that releases the lock exactly once.
 */
async function acquireNamedLock(
  locks: WebLocksLike,
  name: string,
): Promise<(() => void) | undefined> {
  const handshake = new LockHandshake();

  const request = locks.request(name, { mode: "exclusive", ifAvailable: true }, (lock) => {
    if (lock === null) {
      handshake.settle(false);

      return null;
    }

    handshake.settle(true);

    // The lock is held for exactly as long as this promise stays pending.
    return handshake.held;
  });

  void request.catch(() => handshake.settle(false));

  if (!(await handshake.granted)) return undefined;

  return () => handshake.release();
}

class IndexedDbStreamCapture implements DictationStreamCapture {
  readonly sessionId: string;
  private readonly createSession: (input: CreateSessionInput) => Promise<DictationSession>;
  private readonly openDatabase: () => Promise<IDBDatabase>;
  private readonly releaseLock: (() => void) | undefined;
  private readonly retire: () => void;
  private readonly chunks: Uint8Array[] = [];
  private writeChain: Promise<void> = Promise.resolve();
  private nextIndex = 0;
  private closed = false;
  private durable = true;
  private ownershipReleased = false;

  constructor(
    sessionId: string,
    createSession: (input: CreateSessionInput) => Promise<DictationSession>,
    openDatabase: () => Promise<IDBDatabase>,
    releaseLock: (() => void) | undefined,
    retire: () => void,
  ) {
    this.sessionId = sessionId;
    this.createSession = createSession;
    this.openDatabase = openDatabase;
    this.releaseLock = releaseLock;
    this.retire = retire;
  }

  /**
   * Drop every ownership claim without touching the journal: the owning
   * store is going away, so from another window's point of view the owner
   * just terminated — the journal stays durable and recoverable.
   */
  orphan(): void {
    this.releaseOwnership();
  }

  append(pcm16: Uint8Array): Promise<void> {
    assertPcm16Chunk(pcm16);

    if (this.closed) {
      return Promise.reject(
        new DictationStorageError(`stream capture ${this.sessionId} already closed`),
      );
    }

    // The in-memory mirror is the assembly source and the crash fallback;
    // the journal makes journaled chunks survive a crash mid-recording.
    this.chunks.push(pcm16);

    if (!this.durable) return Promise.resolve();

    const index = this.nextIndex;
    this.nextIndex += 1;
    this.writeChain = this.writeChain.then(() => this.writeChunk(index, pcm16));

    return this.writeChain;
  }

  appendedFrames(): number {
    return this.chunks.reduce((frames, chunk) => frames + chunk.byteLength / 2, 0);
  }

  async finish(durationMs?: number, isCancelled?: () => boolean): Promise<FinishedStreamCapture> {
    this.closed = true;
    await this.writeChain.catch(() => {
      /* durability was already reported to the appender */
    });

    // The session insert is the only durable write this path makes; a take
    // discarded while the journal drained must not be persisted (#160).
    if (isCancelled?.()) {
      await this.discard().catch(() => {
        /* a journal that could not be deleted stays claimed by this window */
      });

      const assembled = assemblePcm16Wav(this.chunks);

      return Object.freeze({
        wav: assembled.wav,
        durationMs: durationMs ?? assembled.durationMs,
        session: undefined,
      });
    }

    const assembled = assemblePcm16Wav(this.chunks);
    const duration = durationMs ?? assembled.durationMs;

    try {
      const session = await this.createSession({
        id: this.sessionId,
        wav: assembled.wav,
        durationMs: duration,
      });

      await this.discard().catch(() => {
        /* the post-close sweep consumes a journal that could not be deleted */
      });

      return Object.freeze({ wav: assembled.wav, durationMs: duration, session });
    } catch (failure) {
      return Object.freeze({ wav: assembled.wav, durationMs: duration, failure });
    }
  }

  async abandon(): Promise<void> {
    this.closed = true;

    // Unconditional and best-effort, like finish(): a journal that stopped
    // being durable mid-take still has rows in the store, and a take the
    // user discarded must not resurrect via recovery.
    await this.discard().catch(() => {
      /* a journal that could not be deleted stays claimed by this window */
    });
  }

  private async writeChunk(index: number, pcm16: Uint8Array): Promise<void> {
    if (!this.durable || this.closed) return;

    try {
      const database = await this.openDatabase();
      const record: StreamChunkRecord = { captureId: this.sessionId, index, pcm: pcm16 };

      await runIdbTransaction(database, [STREAM_CHUNK_STORE], "readwrite", (transaction) => {
        transaction.objectStore(STREAM_CHUNK_STORE).put(record);
      });
    } catch (cause) {
      // One durable failure stops journaling (later chunks stay memory-only);
      // the rejection tells the caller to stop trusting the journal.
      this.durable = false;

      throw new DictationStorageError("failed to journal streaming audio", { cause });
    }
  }

  /** The journal is gone; the capture's ownership ends with it. */
  private releaseOwnership(): void {
    if (this.ownershipReleased) return;
    this.ownershipReleased = true;
    this.releaseLock?.();
    this.retire();
  }

  private async discard(): Promise<void> {
    const database = await this.openDatabase();

    await deleteStreamCapture(database, this.sessionId);

    // Ownership ends with the journal, and only then: a delete that failed
    // keeps the capture claimed, so no sweep can promote a journal its
    // owner discarded into a resurrected session while this window lives.
    this.releaseOwnership();
  }
}

const decodeStreamChunkRecord = Schema.decodeUnknownOption(
  Schema.Struct({
    captureId: Schema.String,
    index: Schema.Finite,
    pcm: Schema.instanceOf(Uint8Array),
  }),
);

const decodeStreamMetaRecord = Schema.decodeUnknownOption(
  Schema.Struct({ id: Schema.NonEmptyString }),
);

function isStreamChunkRecord(value: unknown): value is StreamChunkRecord {
  const decoded = decodeStreamChunkRecord(value);

  if (Option.isNone(decoded)) return false;
  const { index, pcm } = decoded.value;

  return Number.isInteger(index) && index >= 0 && pcm.byteLength > 0 && pcm.byteLength % 2 === 0;
}

function isStreamMetaRecord(value: unknown): value is StreamMetaRecord {
  return Option.isSome(decodeStreamMetaRecord(value));
}

async function readStreamChunks(database: IDBDatabase, captureId: string): Promise<Uint8Array[]> {
  const transaction = database.transaction([STREAM_CHUNK_STORE], "readonly");
  const request = transaction.objectStore(STREAM_CHUNK_STORE).getAll();
  const records = await runIdbRequest(request);

  return records
    .filter(
      (record): record is StreamChunkRecord =>
        isStreamChunkRecord(record) && record.captureId === captureId,
    )
    .sort((left, right) => left.index - right.index)
    .map((record) => record.pcm);
}

function deleteStreamCapture(database: IDBDatabase, captureId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction([STREAM_META_STORE, STREAM_CHUNK_STORE], "readwrite");
    const cursorRequest = transaction.objectStore(STREAM_CHUNK_STORE).openCursor();

    cursorRequest.onsuccess = () => {
      const cursor = cursorRequest.result;

      if (cursor) {
        if (isStreamChunkRecord(cursor.value) && cursor.value.captureId === captureId) {
          cursor.delete();
        }

        cursor.continue();
      }
    };

    transaction.objectStore(STREAM_META_STORE).delete(captureId);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () =>
      reject(
        new DictationStorageError("failed to delete a streaming capture journal", {
          cause: transaction.error,
        }),
      );
    transaction.onabort = () =>
      reject(
        new DictationStorageError("streaming capture journal delete was aborted", {
          cause: transaction.error,
        }),
      );
  });
}

export class IndexedDbSessionStore implements DictationSessionStore, DictationStreamCaptureStore {
  private readonly databaseName: string;
  private readonly factory: IDBFactory | undefined;
  private readonly locks: WebLocksLike | undefined;
  /** Identifies this instance in journal ownership metadata. */
  private readonly ownerId: string;
  private readonly openCaptures = new Map<string, IndexedDbStreamCapture>();
  /**
   * Every admitted transcription attempt's release latch keyed by an
   * attempt-local signal: more than one attempt for a session may be open
   * at once, and each must be visible to other windows until it settles.
   * Lockless and contended attempts hold no latch — the registry carries
   * their signal alone — but still get their entry so settlement clears it.
   */
  private readonly attemptLocks = new Map<string, (() => void) | undefined>();
  private databasePromise: Promise<IDBDatabase> | undefined;

  constructor(options: IndexedDbSessionStoreOptions = {}) {
    this.databaseName = options.databaseName ?? "starling-dictation";
    this.factory = options.indexedDB ?? globalThis.indexedDB;
    // Property presence, not nullish coalescing: an explicitly supplied
    // `webLocks: undefined` means "run unlocked" even where the host has
    // navigator.locks, while an omitted property takes the host default.
    this.locks = "webLocks" in options ? options.webLocks : navigatorLocks();
    this.ownerId = sessionId();
  }

  async create(input: CreateSessionInput): Promise<DictationSession> {
    const session = initialSession(input);
    await this.write("add", session);

    return session;
  }

  async get(id: string): Promise<DictationSession | undefined> {
    const database = await this.database();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readonly");
      const request = transaction.objectStore(OBJECT_STORE).get(id);

      request.onsuccess = () => {
        if (request.result === undefined) {
          resolve(undefined);

          return;
        }

        try {
          resolve(freezeSession(decodeSession(request.result)));
        } catch (cause) {
          reject(invalidStoredSession(cause));
        }
      };

      request.onerror = () =>
        reject(
          new DictationStorageError("failed to read dictation session", { cause: request.error }),
        );
      transaction.onabort = () =>
        reject(
          new DictationStorageError("dictation session read was aborted", {
            cause: transaction.error,
          }),
        );
    });
  }

  async list(): Promise<readonly DictationSession[]> {
    return (await this.listReport()).sessions;
  }

  /**
   * Decode records one at a time with a key cursor: a single damaged or
   * incompatible entry is quarantined into `invalid` instead of rejecting the
   * whole listing, so healthy history stays visible and recoverable audio is
   * never deleted. The cursor (not getAll) hands over each record's raw key,
   * which quarantine entries carry for later dismissal (#155).
   */
  async listReport(): Promise<DictationHistory> {
    const database = await this.database();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readonly");
      const sessions: DictationSession[] = [];
      const invalid: InvalidStoredSession[] = [];
      const cursorRequest = transaction.objectStore(OBJECT_STORE).openCursor();

      cursorRequest.onsuccess = () => {
        try {
          const cursor = cursorRequest.result;

          if (!cursor) {
            sessions.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
            resolve(
              Object.freeze({
                sessions: Object.freeze(sessions),
                invalid: Object.freeze(invalid),
              }),
            );

            return;
          }

          const isolated = isolateStoredSession(cursor.value, cursor.primaryKey);

          if ("record" in isolated) invalid.push(isolated);
          else sessions.push(isolated);

          cursor.continue();
        } catch (cause) {
          reject(new DictationStorageError("failed to list dictation sessions", { cause }));
        }
      };

      cursorRequest.onerror = () =>
        reject(
          new DictationStorageError("failed to list dictation sessions", {
            cause: cursorRequest.error,
          }),
        );
      transaction.onabort = () =>
        reject(
          new DictationStorageError("dictation session list was aborted", {
            cause: transaction.error,
          }),
        );
    });
  }

  async markAttempt(id: string): Promise<DictationSession> {
    const signal = attemptSignal();
    await this.holdAttemptSignal(id, signal);

    try {
      return await this.update(id, (current) =>
        updatedSession(current, {
          status: "transcribing",
          attemptCount: current.attemptCount + 1,
          lastError: undefined,
        }),
      );
    } catch (failure) {
      this.releaseAttemptSignal(id, signal);

      throw failure;
    }
  }

  async saveTranscript(
    id: string,
    transcript: TranscriptionResult,
    options?: SaveTranscriptOptions,
  ): Promise<DictationSession> {
    const update = transcriptSettling(transcript, options);

    try {
      return await this.update(id, (current) => updatedSession(current, update));
    } finally {
      // The settling write can reject (quota, abort, session deleted from
      // another window): the attempt signal must not outlive its attempt,
      // or every other window's liveness probe reports true forever.
      this.releaseAttemptSignals(id);
    }
  }

  /**
   * The confirmed-interruption half of the startup sweep: downgrade one
   * session from an in-flight state to failed, never touching a session a
   * live owner already settled or moved on. The sweep's transcriptionInFlight
   * probe stays a read-only hint; this single conditional write is the
   * enforcement, so a saveTranscript that commits between the probe and the
   * mark cannot last-writer-lose its transcript to a failure (#162).
   */
  async saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession> {
    try {
      return await this.update(id, (current) =>
        current.status === "transcribing" || current.status === "captured"
          ? updatedSession(current, {
              status: "failed",
              lastError: errorText(cause),
            })
          : current,
      );
    } finally {
      this.releaseAttemptSignals(id);
    }
  }

  async noteStreamError(id: string, message: string): Promise<DictationSession> {
    return this.update(id, (current) => updatedSession(current, { streamError: message }));
  }

  async saveRefinedTranscript(id: string, refined: RefinedTranscript): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, { refined: freezeRefined(refined) }),
    );
  }

  async assignThread(id: string, threadId: string): Promise<DictationSession> {
    assertThreadId(threadId);

    return this.update(id, (current) =>
      updatedSession(current, threadAssignment(current, threadId)),
    );
  }

  /**
   * True when ANY live owner (this window or another) holds an attempt
   * signal for the session. A contended second attempt carries its own
   * signal, so it stays visible even after the first settler's release —
   * the sweep must not mistake a live contender for a dead owner (#162).
   */
  async transcriptionInFlight(id: string): Promise<boolean> {
    if (liveAttemptSignals.has(attemptRegistryKey(this.databaseName, id))) return true;

    if (!this.locks) return false;

    const lockPrefix = attemptLockPrefix(this.databaseName);

    try {
      const snapshot = await this.locks.query();

      return (
        snapshot.held?.some(
          (held) => held.name !== undefined && isAttemptSignalFor(held.name, lockPrefix, id),
        ) ?? false
      );
    } catch {
      // A snapshot the manager cannot produce says nothing about liveness:
      // report in-flight, as the single-name probe did on rejection, so a
      // transient manager failure never manufactures an interruption.
      return true;
    }
  }

  async delete(id: string): Promise<void> {
    const database = await this.database();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readwrite");
      transaction.objectStore(OBJECT_STORE).delete(id);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new DictationStorageError("failed to delete dictation session", {
            cause: transaction.error,
          }),
        );
      transaction.onabort = () =>
        reject(
          new DictationStorageError("dictation session delete was aborted", {
            cause: transaction.error,
          }),
        );
    });
    this.releaseAttemptSignals(id);
  }

  /**
   * One quarantined-entry dismissal: re-read the key in a write transaction
   * and delete only when the record still fails validation, so the check and
   * the delete cannot race a concurrent repair (#155).
   */
  async deleteInvalid(key: IDBValidKey): Promise<void> {
    const database = await this.database();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readwrite");
      const store = transaction.objectStore(OBJECT_STORE);
      const request = store.get(key);

      request.onsuccess = () => {
        try {
          // Nothing there, or a repaired entry another window restored since
          // the listing: the damaged record the user saw is already gone, so
          // dismissal succeeds without deleting.
          if (request.result === undefined) return;

          if ("record" in isolateStoredSession(request.result)) store.delete(key);
        } catch (cause) {
          transaction.abort();
          reject(
            new DictationStorageError("failed to dismiss a damaged dictation session", { cause }),
          );
        }
      };

      request.onerror = () =>
        reject(
          new DictationStorageError("failed to dismiss a damaged dictation session", {
            cause: request.error,
          }),
        );

      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new DictationStorageError("failed to dismiss a damaged dictation session", {
            cause: transaction.error,
          }),
        );
      // Reject unconditionally: a double settle after the catch's own reject
      // is a harmless no-op, and a guarded reject here could leave the
      // promise pending forever when the transaction aborts after an empty
      // read — the sibling delete() relies on the same idiom.
      transaction.onabort = () =>
        reject(
          new DictationStorageError("damaged dictation session dismissal was aborted", {
            cause: transaction.error,
          }),
        );
    });
  }

  async beginStreamCapture(input: CreateStreamCaptureInput = {}): Promise<DictationStreamCapture> {
    const id = streamCaptureId(input);
    const createdAt = input.createdAt ?? new Date().toISOString();
    const database = await this.database();

    // The lock is the cross-window ownership claim for this capture id: held
    // from before the metadata row exists until the capture closes, so a
    // second window's recovery can never mistake the journal for abandoned.
    const release = this.locks
      ? await acquireNamedLock(this.locks, captureLockName(this.databaseName, id))
      : undefined;

    if (this.locks && !release) {
      throw new DictationStorageError(`stream capture ${id} is already owned by another window`);
    }

    const record: StreamMetaRecord = { id, createdAt, ownerId: this.ownerId };

    try {
      await runIdbTransaction(database, [STREAM_META_STORE], "readwrite", (transaction) => {
        transaction.objectStore(STREAM_META_STORE).add(record);
      });
    } catch (cause) {
      release?.();

      throw cause;
    }

    liveCaptureRegistry.add(captureRegistryKey(this.databaseName, id));

    const capture = new IndexedDbStreamCapture(
      id,
      (sessionInput) => this.create(sessionInput),
      () => this.database(),
      release,
      () => {
        liveCaptureRegistry.delete(captureRegistryKey(this.databaseName, id));
        this.openCaptures.delete(id);
      },
    );

    this.openCaptures.set(id, capture);

    return capture;
  }

  async recoverStreamCaptures(): Promise<readonly DictationSession[]> {
    const database = await this.database();

    const metas = await runIdbRequest(
      database.transaction(STREAM_META_STORE, "readonly").objectStore(STREAM_META_STORE).getAll(),
    );

    const recovered: DictationSession[] = [];

    for (const candidate of metas) {
      const meta = isStreamMetaRecord(candidate) ? candidate : undefined;

      if (!meta) {
        continue;
      }

      // A journal is consumed only when its owner is demonstrably gone: a
      // live store in this environment, this very instance's ownership
      // marker, or a capture lock another window still holds all mean
      // somebody is recording into it right now (#144).
      if (
        liveCaptureRegistry.has(captureRegistryKey(this.databaseName, meta.id)) ||
        meta.ownerId === this.ownerId
      ) {
        continue;
      }

      if (this.locks) {
        // The lock is held for the whole promotion, so concurrent sweeps
        // cannot double-consume a journal; `null` means a live owner.
        await this.locks.request(
          captureLockName(this.databaseName, meta.id),
          { mode: "exclusive", ifAvailable: true },
          async (lock) => {
            if (lock === null) return;

            await this.promoteJournal(database, meta.id, recovered);
          },
        );
      } else {
        await this.promoteJournal(database, meta.id, recovered);
      }
    }

    return Object.freeze(recovered);
  }

  close(): void {
    // From every other window's point of view this owner just terminated:
    // its claims end here, so abandoned journals become recoverable. Map
    // iteration tolerates the deletions these releases make.
    //
    // Precondition: captures have been finished or abandoned and attempts
    // have settled (saveTranscript/saveFailure/delete) before close() —
    // closing with work still in flight strands that work's journal and
    // leaves its chunks orphaned for every future sweep.
    for (const capture of this.openCaptures.values()) {
      capture.orphan();
    }

    for (const id of this.attemptSignalsOwned()) {
      this.releaseAttemptSignals(id);
    }

    // A failed open already rejected to its caller; closing afterwards must
    // not re-surface that failure as an unhandled rejection.
    void this.databasePromise?.then(
      (database) => database.close(),
      () => undefined,
    );
    this.databasePromise = undefined;
  }

  /**
   * Assemble one journal and promote it. The session insert and the journal
   * (metadata plus chunks) delete share a single transaction, so a crash
   * mid-promotion rolls back together: a journal can neither strand a
   * half-promoted session nor be consumed twice. Journals that cannot be
   * promoted now simply stay for the next attempt.
   */
  private async promoteJournal(
    database: IDBDatabase,
    captureId: string,
    recovered: DictationSession[],
  ): Promise<void> {
    try {
      const chunks = await readStreamChunks(database, captureId);
      const promoted = await this.promoteStreamCapture(database, captureId, chunks);

      if (promoted) recovered.push(promoted);
    } catch {
      // A journal that cannot be promoted now stays for the next attempt.
    }
  }

  /**
   * One-transaction promotion: insert the retryable session (unless it
   * already exists or the journal has no audio) and delete the journal.
   * Resolves the created session, or undefined when there was nothing to
   * insert — an empty journal or one whose session already survived.
   */
  private promoteStreamCapture(
    database: IDBDatabase,
    captureId: string,
    chunks: readonly Uint8Array[],
  ): Promise<DictationSession | undefined> {
    return new Promise((resolve, reject) => {
      const transaction = database.transaction(
        [OBJECT_STORE, STREAM_META_STORE, STREAM_CHUNK_STORE],
        "readwrite",
      );

      const existing = transaction.objectStore(OBJECT_STORE).get(captureId);

      existing.onsuccess = () => {
        let created: DictationSession | undefined;

        if (existing.result === undefined && chunks.length > 0) {
          created = recoveredSession(captureId, assemblePcm16Wav(chunks));
          transaction.objectStore(OBJECT_STORE).add(created);
        }

        transaction.objectStore(STREAM_META_STORE).delete(captureId);

        const cursorRequest = transaction.objectStore(STREAM_CHUNK_STORE).openCursor();

        cursorRequest.onsuccess = () => {
          const cursor = cursorRequest.result;

          if (!cursor) return;

          if (isStreamChunkRecord(cursor.value) && cursor.value.captureId === captureId) {
            cursor.delete();
          }

          cursor.continue();
        };

        transaction.oncomplete = () => resolve(created);
      };

      existing.onerror = () =>
        reject(
          new DictationStorageError("failed to read a session before journal promotion", {
            cause: existing.error,
          }),
        );
      transaction.onerror = () =>
        reject(
          new DictationStorageError("failed to promote a streaming capture journal", {
            cause: transaction.error,
          }),
        );
      transaction.onabort = () =>
        reject(
          new DictationStorageError("streaming capture journal promotion was aborted", {
            cause: transaction.error,
          }),
        );
    });
  }

  /**
   * Best-effort in-flight signal for one transcription attempt, so another
   * window's startup sweep can tell a live attempt from a dead owner's
   * leftover. The signal is advisory: contention never blocks a retry. Every
   * admitted attempt holds its own lock (`transcribe:<id>:<signal>`), so a
   * contended attempt stays visible even after the lock winner settles — a
   * sweep probing between the winner's release and the contender's settle
   * still reads in-flight (#162).
   */
  private async holdAttemptSignal(id: string, signal: string): Promise<void> {
    // The lock is held until this attempt settles. A contended or lockless
    // environment hands back no latch, but the registry still carries the
    // signal — so the latch map gets its entry either way and settlement
    // always finds and discards the registry signal (#162).
    const release = this.locks
      ? await acquireNamedLock(this.locks, attemptLockName(this.databaseName, id, signal))
      : undefined;

    this.attemptLocks.set(ownedAttemptKey(id, signal), release);
    trackAttemptSignal(this.databaseName, id, signal);
  }

  /**
   * End every attempt signal this store holds for a session. Settling writes
   * and deletion cannot know which local signal a foreign write belonged to,
   * and a signal that outlives its attempt would pin every other window's
   * liveness probe true forever — so releasing all local signals is the only
   * correct granularity here.
   */
  private releaseAttemptSignals(id: string): void {
    const prefix = `${id}\u0000`;
    const owned: string[] = [];

    for (const key of this.attemptLocks.keys()) {
      if (key.startsWith(prefix)) owned.push(key);
    }

    for (const key of owned) {
      const release = this.attemptLocks.get(key);

      this.attemptLocks.delete(key);
      discardAttemptSignal(this.databaseName, id, key.slice(prefix.length));
      release?.();
    }
  }

  /**
   * Session ids this store holds attempt signals for. Rebuilt from the
   * signal keys on each close() so concurrent attempts for one session are
   * all released, never just the first.
   */
  private attemptSignalsOwned(): string[] {
    const owned = new Set<string>();

    for (const key of this.attemptLocks.keys()) {
      const separator = key.indexOf("\u0000");

      if (separator >= 0) owned.add(key.slice(0, separator));
    }

    return [...owned];
  }

  /** End one attempt's signal after its markAttempt write rejects. */
  private releaseAttemptSignal(id: string, signal: string): void {
    const owned = ownedAttemptKey(id, signal);

    this.attemptLocks.get(owned)?.();
    this.attemptLocks.delete(owned);
    discardAttemptSignal(this.databaseName, id, signal);
  }

  private database(): Promise<IDBDatabase> {
    if (!this.factory) {
      return Promise.reject(
        new DictationStorageError("IndexedDB is unavailable in this environment"),
      );
    }

    if (!this.databasePromise) {
      this.databasePromise = new Promise((resolve, reject) => {
        const request = this.factory?.open(this.databaseName, DICTATION_DATABASE_VERSION);

        if (!request) {
          reject(new DictationStorageError("IndexedDB is unavailable in this environment"));

          return;
        }

        request.onupgradeneeded = () => {
          const database = request.result;

          if (!database.objectStoreNames.contains(OBJECT_STORE)) {
            const store = database.createObjectStore(OBJECT_STORE, { keyPath: "id" });
            store.createIndex("updatedAt", "updatedAt");
            store.createIndex("status", "status");
          }

          // Version 2: the streaming-capture journal. Session records are
          // untouched, so persisted v1 sessions load unchanged.
          if (!database.objectStoreNames.contains(STREAM_META_STORE)) {
            database.createObjectStore(STREAM_META_STORE, { keyPath: "id" });
          }

          if (!database.objectStoreNames.contains(STREAM_CHUNK_STORE)) {
            database
              .createObjectStore(STREAM_CHUNK_STORE, { keyPath: ["captureId", "index"] })
              .createIndex("captureId", "captureId", { unique: false });
          }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () =>
          reject(
            new DictationStorageError("failed to open dictation session database", {
              cause: request.error,
            }),
          );
        request.onblocked = () =>
          reject(new DictationStorageError("dictation session database upgrade is blocked"));
      });
    }

    return this.databasePromise;
  }

  private async write(mode: "add" | "put", session: DictationSession): Promise<void> {
    const database = await this.database();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readwrite");
      transaction.objectStore(OBJECT_STORE)[mode](session);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(
          new DictationStorageError("failed to write dictation session", {
            cause: transaction.error,
          }),
        );
      transaction.onabort = () =>
        reject(
          new DictationStorageError("dictation session write was aborted", {
            cause: transaction.error,
          }),
        );
    });
  }

  private async update(
    id: string,
    transform: (current: DictationSession) => DictationSession,
  ): Promise<DictationSession> {
    const database = await this.database();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readwrite");
      const store = transaction.objectStore(OBJECT_STORE);
      const request = store.get(id);
      let next: DictationSession | undefined;

      request.onsuccess = () => {
        if (request.result === undefined) {
          transaction.abort();
          reject(new DictationSessionNotFoundError(id));

          return;
        }

        try {
          next = transform(freezeSession(decodeSession(request.result)));
          store.put(next);
        } catch (cause) {
          transaction.abort();
          reject(invalidStoredSession(cause));
        }
      };

      request.onerror = () =>
        reject(
          new DictationStorageError("failed to read dictation session for update", {
            cause: request.error,
          }),
        );

      transaction.oncomplete = () => {
        if (next) resolve(next);
      };

      transaction.onerror = () =>
        reject(
          new DictationStorageError("failed to update dictation session", {
            cause: transaction.error,
          }),
        );

      transaction.onabort = () => {
        if (request.result !== undefined) {
          reject(
            new DictationStorageError("dictation session update was aborted", {
              cause: transaction.error,
            }),
          );
        }
      };
    });
  }
}
