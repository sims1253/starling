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

export const DictationSessionSchema = Schema.Struct({
  id: Schema.NonEmptyString,
  createdAt: Schema.NonEmptyString,
  updatedAt: Schema.NonEmptyString,
  status: Schema.Literals(["captured", "transcribing", "transcribed", "failed"]),
  /** Canonical PCM16 16 kHz WAV. Retained until delete() is called. */
  wav: Schema.instanceOf(Blob),
  durationMs: Schema.optional(NonNegativeFinite),
  attemptCount: Schema.Natural,
  transcript: Schema.optional(TranscriptionResultSchema),
  /** Earlier successful recognition results, retained when a retry succeeds. */
  transcriptHistory: Schema.optional(Schema.Array(TranscriptionResultSchema)),
  lastError: Schema.optional(Schema.String),
  /** True when the final transcript arrived over the streaming connection. */
  streamed: Schema.optional(Schema.Boolean),
  /** Why live streaming failed here; kept for review even after a batch retry succeeds. */
  streamError: Schema.optional(Schema.String),
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
}

export interface InvalidStoredSession {
  /** Best-effort id taken from the raw record; "(unknown id)" when absent. */
  readonly id: string;
  /** Why the record failed schema validation. */
  readonly cause: unknown;
  /**
   * The raw IndexedDB record, retained untouched in the database. Damaged
   * entries are quarantined out of the listing — never deleted — so
   * recoverable audio stays exportable.
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
  saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession>;
  /** Record why live streaming failed without changing the session status. */
  noteStreamError(id: string, message: string): Promise<DictationSession>;
  delete(id: string): Promise<void>;
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
  readonly transcript?: TranscriptionResult;
  readonly transcriptHistory?: readonly TranscriptionResult[];
  readonly lastError?: string;
  readonly streamed?: boolean;
  readonly streamError?: string;
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
  transcript?: TranscriptionResult;
  transcriptHistory?: readonly TranscriptionResult[];
  lastError?: string;
  streamed?: boolean;
  streamError?: string;
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

function assertCreateInput(input: CreateSessionInput): void {
  if (!(input.wav instanceof Blob) || input.wav.size === 0) {
    throw new TypeError("wav must be a non-empty Blob");
  }

  if (input.id !== undefined && (!input.id || /[\r\n]/.test(input.id))) {
    throw new TypeError("session id must be non-empty and contain no newlines");
  }

  if (
    input.durationMs !== undefined &&
    (!Number.isFinite(input.durationMs) || input.durationMs < 0)
  ) {
    throw new TypeError("durationMs must be a non-negative finite number");
  }
}

interface TranscriptDraft {
  text: string;
  segments: TranscriptionResult["segments"];
  durationSeconds?: number;
  requestId?: string;
}

function freezeTranscript(value: TranscriptionResult): TranscriptionResult {
  const segments = Object.freeze(value.segments.map((segment) => Object.freeze({ ...segment })));

  const transcript: TranscriptDraft = {
    text: value.text,
    segments,
  };

  if (value.durationSeconds !== undefined) transcript.durationSeconds = value.durationSeconds;

  if (value.requestId !== undefined) transcript.requestId = value.requestId;

  return Object.freeze(transcript);
}

interface SessionDraft {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: DictationSessionStatus;
  wav: Blob;
  durationMs?: number | undefined;
  attemptCount: number;
  transcript?: TranscriptionResult | undefined;
  transcriptHistory?: readonly TranscriptionResult[] | undefined;
  lastError?: string | undefined;
  streamed?: boolean | undefined;
  streamError?: string | undefined;
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
  transcript: TranscriptionResult;
  lastError: string | undefined;
  streamed: boolean;
  streamError: string;
}>;

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
  readonly session?: DictationSession;
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
  /** Assemble the WAV and persist the session (source-of-truth save). */
  finish(durationMs?: number): Promise<FinishedStreamCapture>;
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

  async finish(durationMs?: number): Promise<FinishedStreamCapture> {
    this.closed = true;
    const assembled = assemblePcm16Wav(this.chunks);
    const duration = durationMs ?? assembled.durationMs;

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
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "transcribing",
        attemptCount: current.attemptCount + 1,
        lastError: undefined,
      }),
    );
  }

  async saveTranscript(
    id: string,
    transcript: TranscriptionResult,
    options?: SaveTranscriptOptions,
  ): Promise<DictationSession> {
    const update: SessionUpdate = {
      status: "transcribed",
      transcript: freezeTranscript(transcript),
      lastError: undefined,
    };

    if (options?.streamed === true) update.streamed = true;

    return this.update(id, (current) => updatedSession(current, update));
  }

  async saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "failed",
        lastError: errorText(cause),
      }),
    );
  }

  async noteStreamError(id: string, message: string): Promise<DictationSession> {
    return this.update(id, (current) => updatedSession(current, { streamError: message }));
  }

  async delete(id: string): Promise<void> {
    this.sessions.delete(id);
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

export interface IndexedDbSessionStoreOptions {
  readonly databaseName?: string;
  readonly indexedDB?: IDBFactory | undefined;
}

const OBJECT_STORE = "sessions";

const STREAM_META_STORE = "stream-captures";

const STREAM_CHUNK_STORE = "stream-chunks";

interface StreamMetaRecord {
  readonly id: string;
  readonly createdAt: string;
}

interface StreamChunkRecord {
  readonly captureId: string;
  readonly index: number;
  readonly pcm: Uint8Array;
}

const decodeSession = Schema.decodeUnknownSync(DictationSessionSchema);

const decodeSessionId = Schema.decodeUnknownOption(Schema.Struct({ id: Schema.NonEmptyString }));

/**
 * Validate one stored record without touching the database: damaged entries
 * are reported with their raw record so recoverable audio stays exportable.
 */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- A raw IndexedDB row is unknown by definition; this function is the boundary that parses it.
function isolateStoredSession(record: unknown): DictationSession | InvalidStoredSession {
  try {
    return freezeSession(decodeSession(record));
  } catch (cause) {
    const decodedId = decodeSessionId(record);

    return Object.freeze({
      id: Option.isSome(decodedId) ? decodedId.value.id : "(unknown id)",
      cause: invalidStoredSession(cause),
      record,
    });
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

class IndexedDbStreamCapture implements DictationStreamCapture {
  readonly sessionId: string;
  private readonly createSession: (input: CreateSessionInput) => Promise<DictationSession>;
  private readonly openDatabase: () => Promise<IDBDatabase>;
  private readonly chunks: Uint8Array[] = [];
  private writeChain: Promise<void> = Promise.resolve();
  private nextIndex = 0;
  private closed = false;
  private durable = true;

  constructor(
    sessionId: string,
    createSession: (input: CreateSessionInput) => Promise<DictationSession>,
    openDatabase: () => Promise<IDBDatabase>,
  ) {
    this.sessionId = sessionId;
    this.createSession = createSession;
    this.openDatabase = openDatabase;
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

  async finish(durationMs?: number): Promise<FinishedStreamCapture> {
    this.closed = true;
    await this.writeChain.catch(() => {
      /* durability was already reported to the appender */
    });

    const assembled = assemblePcm16Wav(this.chunks);
    const duration = durationMs ?? assembled.durationMs;

    try {
      const session = await this.createSession({
        id: this.sessionId,
        wav: assembled.wav,
        durationMs: duration,
      });

      await this.discard().catch(() => {
        /* recovery sweeps journals whose session already exists */
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
      /* recovery sweeps journals that could not be deleted */
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

  private async discard(): Promise<void> {
    const database = await this.openDatabase();

    await deleteStreamCapture(database, this.sessionId);
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
  private databasePromise: Promise<IDBDatabase> | undefined;

  constructor(options: IndexedDbSessionStoreOptions = {}) {
    this.databaseName = options.databaseName ?? "starling-dictation";
    this.factory = options.indexedDB ?? globalThis.indexedDB;
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
   * Decode records one at a time: a single damaged or incompatible entry is
   * quarantined into `invalid` instead of rejecting the whole listing, so
   * healthy history stays visible and recoverable audio is never deleted.
   */
  async listReport(): Promise<DictationHistory> {
    const database = await this.database();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readonly");
      const request = transaction.objectStore(OBJECT_STORE).getAll();

      request.onsuccess = () => {
        try {
          const sessions: DictationSession[] = [];
          const invalid: InvalidStoredSession[] = [];

          for (const record of request.result) {
            const isolated = isolateStoredSession(record);

            if ("record" in isolated) invalid.push(isolated);
            else sessions.push(isolated);
          }

          sessions.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

          resolve(
            Object.freeze({ sessions: Object.freeze(sessions), invalid: Object.freeze(invalid) }),
          );
        } catch (cause) {
          reject(new DictationStorageError("failed to list dictation sessions", { cause }));
        }
      };

      request.onerror = () =>
        reject(
          new DictationStorageError("failed to list dictation sessions", { cause: request.error }),
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
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "transcribing",
        attemptCount: current.attemptCount + 1,
        lastError: undefined,
      }),
    );
  }

  async saveTranscript(
    id: string,
    transcript: TranscriptionResult,
    options?: SaveTranscriptOptions,
  ): Promise<DictationSession> {
    const update: SessionUpdate = {
      status: "transcribed",
      transcript: freezeTranscript(transcript),
      lastError: undefined,
    };

    if (options?.streamed === true) update.streamed = true;

    return this.update(id, (current) => updatedSession(current, update));
  }

  async saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "failed",
        lastError: errorText(cause),
      }),
    );
  }

  async noteStreamError(id: string, message: string): Promise<DictationSession> {
    return this.update(id, (current) => updatedSession(current, { streamError: message }));
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
  }

  async beginStreamCapture(input: CreateStreamCaptureInput = {}): Promise<DictationStreamCapture> {
    const id = streamCaptureId(input);
    const createdAt = input.createdAt ?? new Date().toISOString();
    const database = await this.database();
    const record: StreamMetaRecord = { id, createdAt };

    await runIdbTransaction(database, [STREAM_META_STORE], "readwrite", (transaction) => {
      transaction.objectStore(STREAM_META_STORE).add(record);
    });

    return new IndexedDbStreamCapture(
      id,
      (sessionInput) => this.create(sessionInput),
      () => this.database(),
    );
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

      try {
        const chunks = await readStreamChunks(database, meta.id);

        if (chunks.length === 0 || (await this.get(meta.id))) {
          await deleteStreamCapture(database, meta.id);

          continue;
        }

        const assembled = assemblePcm16Wav(chunks);

        const created = await this.create({
          id: meta.id,
          wav: assembled.wav,
          durationMs: assembled.durationMs,
        });

        const marked = await this.saveFailure(
          meta.id,
          "Recovered after the app closed during recording. The audio is intact; retry when ready.",
        );

        await deleteStreamCapture(database, meta.id);
        recovered.push(marked ?? created);
      } catch {
        // A journal that cannot be assembled now stays for the next attempt.
      }
    }

    return Object.freeze(recovered);
  }

  close(): void {
    // A failed open already rejected to its caller; closing afterwards must
    // not re-surface that failure as an unhandled rejection.
    void this.databasePromise?.then(
      (database) => database.close(),
      () => undefined,
    );
    this.databasePromise = undefined;
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
