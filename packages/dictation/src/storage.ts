import { Data, Predicate, Schema } from "effect";

import { TranscriptionResultSchema, type TranscriptionResult } from "./client.js";

export const DICTATION_SESSION_SCHEMA_VERSION = 1;

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
});

export type DictationSession = (typeof DictationSessionSchema)["Type"];

export interface CreateSessionInput {
  readonly id?: string | undefined;
  readonly wav: Blob;
  readonly durationMs?: number | undefined;
}

export interface DictationSessionStore {
  create(input: CreateSessionInput): Promise<DictationSession>;
  get(id: string): Promise<DictationSession | undefined>;
  list(): Promise<readonly DictationSession[]>;
  markAttempt(id: string): Promise<DictationSession>;
  saveTranscript(id: string, transcript: TranscriptionResult): Promise<DictationSession>;
  saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession>;
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

function updatedSession(
  current: DictationSession,
  update: Partial<Pick<DictationSession, "status" | "attemptCount" | "transcript" | "lastError">>,
): DictationSession {
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

  return Object.freeze({ manifest: Object.freeze(manifest), wav: session.wav });
}

export class MemorySessionStore implements DictationSessionStore {
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

  async markAttempt(id: string): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "transcribing",
        attemptCount: current.attemptCount + 1,
        lastError: undefined,
      }),
    );
  }

  async saveTranscript(id: string, transcript: TranscriptionResult): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "transcribed",
        transcript: freezeTranscript(transcript),
        lastError: undefined,
      }),
    );
  }

  async saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "failed",
        lastError: errorText(cause),
      }),
    );
  }

  async delete(id: string): Promise<void> {
    this.sessions.delete(id);
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

const decodeSession = Schema.decodeUnknownSync(DictationSessionSchema);

const decodeSessions = Schema.decodeUnknownSync(Schema.Array(DictationSessionSchema));

function invalidStoredSession<Cause>(cause: Cause): DictationStorageError {
  return new DictationStorageError("stored dictation session is invalid", { cause });
}

export class IndexedDbSessionStore implements DictationSessionStore {
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
    const database = await this.database();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(OBJECT_STORE, "readonly");
      const request = transaction.objectStore(OBJECT_STORE).getAll();

      request.onsuccess = () => {
        try {
          const sessions = decodeSessions(request.result)
            .map(freezeSession)
            .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

          resolve(Object.freeze(sessions));
        } catch (cause) {
          reject(invalidStoredSession(cause));
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

  async saveTranscript(id: string, transcript: TranscriptionResult): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "transcribed",
        transcript: freezeTranscript(transcript),
        lastError: undefined,
      }),
    );
  }

  async saveFailure<Cause>(id: string, cause: Cause): Promise<DictationSession> {
    return this.update(id, (current) =>
      updatedSession(current, {
        status: "failed",
        lastError: errorText(cause),
      }),
    );
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

  close(): void {
    void this.databasePromise?.then((database) => database.close());
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
        const request = this.factory?.open(this.databaseName, DICTATION_SESSION_SCHEMA_VERSION);

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
