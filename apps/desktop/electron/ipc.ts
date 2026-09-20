import { Schema } from "effect";
import type { StarlingStreamEvent } from "@starling/dictation";

export interface TranscriptionSegment {
  readonly text: string;
  readonly startSeconds: number;
  readonly endSeconds: number;
}

export interface TranscriptionResult {
  readonly text: string;
  readonly segments: ReadonlyArray<TranscriptionSegment>;
  readonly durationSeconds?: number;
  readonly requestId?: string;
}

export interface ServerHealth {
  readonly status: string;
  readonly phase?: string;
  readonly model?: string;
  readonly loaded?: boolean;
  readonly busy?: boolean;
  readonly queueDepth?: number;
}

export const ProtocolSchema = Schema.Literals(["starling", "openai"]);

export const HealthInputSchema = Schema.Struct({
  endpoint: Schema.String,
  protocol: ProtocolSchema,
  timeoutMs: Schema.optionalKey(Schema.Finite),
});

export const TranscribeInputSchema = Schema.Struct({
  endpoint: Schema.String,
  protocol: ProtocolSchema,
  model: Schema.String,
  requestId: Schema.String,
  audio: Schema.instanceOf(ArrayBuffer),
  timeoutMs: Schema.optionalKey(Schema.Finite),
});

/**
 * Plaintext refinement API key inbound to the main process, which returns
 * safeStorage ciphertext for at-rest storage. The plaintext never persists.
 * Non-empty by schema: clearing the key is the renderer's job (it removes
 * both stored entries without an IPC round trip), so an empty save is a
 * malformed call, not a "store nothing" request.
 */
export const RefinementKeySaveInputSchema = Schema.Struct({
  apiKey: Schema.NonEmptyString,
});

/**
 * Ciphertext previously returned by the save channel, inbound for decryption.
 * Non-empty by schema: an empty payload is rejected at the boundary instead
 * of relying on decryptString throwing on it.
 */
export const RefinementKeyLoadInputSchema = Schema.Struct({
  ciphertext: Schema.NonEmptyString,
});

export type HealthInput = typeof HealthInputSchema.Type;

export type TranscribeInput = typeof TranscribeInputSchema.Type;

export type RefinementKeySaveInput = typeof RefinementKeySaveInputSchema.Type;

export type RefinementKeyLoadInput = typeof RefinementKeyLoadInputSchema.Type;

/**
 * What the main process can honestly say about how (or whether) a key was
 * protected (B10): "encrypted" only for a real OS secret store; "unprotected"
 * for a backend that guards with a hardcoded password (Linux basic_text) or
 * cannot be identified; "unavailable" for no store; "failed" for a store
 * that answered available but refused the encryption call.
 */
export type KeyProtectionStatus = "encrypted" | "unprotected" | "unavailable" | "failed";

export interface RefinementKeySaveResult {
  /**
   * Base64 safeStorage ciphertext for the renderer to persist, or null when
   * this host cannot honestly encrypt (no OS secret store, an unprotected
   * basic_text backend, or a keychain that refused the call).
   */
  readonly ciphertext: string | null;

  /** Accurate per-save protection status; ciphertext is null unless "encrypted". */
  readonly protection: KeyProtectionStatus;

  /**
   * The safeStorage backend the host selected, when it reported one (Linux:
   * gnome_libsecret, kwallet5, basic_text…). Status reporting only.
   */
  readonly backend?: string;
}

export interface RefinementKeyLoadResult {
  /**
   * Decrypted key, or null when the host cannot decrypt this payload
   * (encryption unavailable, or the ciphertext is corrupt/foreign).
   */
  readonly apiKey: string | null;
}

export const PendingAudioStateSchema = Schema.Struct({
  recording: Schema.Boolean,
  finalizing: Schema.Boolean,
  unsavedCount: Schema.Finite,
  /**
   * The live recording is being journaled to storage chunk by chunk (the
   * streaming capture path). Older renderers omit it; the close guard then
   * assumes the stricter memory-only wording.
   */
  journaled: Schema.optionalKey(Schema.Boolean),
});

export type PendingAudioState = typeof PendingAudioStateSchema.Type;

/**
 * Live-streaming transport channels (B01): the renderer's static CSP cannot
 * enumerate user-configured LAN `ws://` or `wss://` endpoints, so the
 * packaged app carries its streaming sockets in the main process and speaks
 * over IPC. The endpoint arrives exactly as the batch channels take it —
 * `http(s)://host[:port]`, validated by the same rules — and the main process
 * derives the `ws(s)://…/stream` URL itself; a ws:// endpoint is refused at
 * the boundary instead of being trusted.
 */
export const StreamOpenInputSchema = Schema.Struct({
  endpoint: Schema.String,
  connectTimeoutMs: Schema.optionalKey(Schema.Finite),
  responseTimeoutMs: Schema.optionalKey(Schema.Finite),
});

export const StreamSendInputSchema = Schema.Struct({
  streamId: Schema.Number,
  audio: Schema.instanceOf(ArrayBuffer),
});

export const StreamCommandInputSchema = Schema.Struct({
  streamId: Schema.Number,
  command: Schema.Literals(["commit", "reset", "ping"]),
});

export const StreamCloseInputSchema = Schema.Struct({
  streamId: Schema.Number,
});

export type StreamOpenInput = typeof StreamOpenInputSchema.Type;
export type StreamSendInput = typeof StreamSendInputSchema.Type;
export type StreamCommandInput = typeof StreamCommandInputSchema.Type;
export type StreamCloseInput = typeof StreamCloseInputSchema.Type;

/** The awaited reply of `commit`; reset and ping resolve it without one. */
export interface StreamCommandResult {
  readonly transcript?: TranscriptionResult;
}

/** One validated stream event pushed to the renderer, tagged with its stream. */
export interface StreamEventMessage {
  readonly streamId: number;
  readonly event: StarlingStreamEvent;
}

export interface DesktopProcessMetric {
  readonly pid: number;
  readonly type: string;
  readonly workingSetKib: number;
}

export interface DesktopDiagnostics {
  readonly readyToShowMs?: number;
  readonly mainProcess: {
    readonly rssBytes: number;
    readonly heapUsedBytes: number;
  };
  readonly processes: ReadonlyArray<DesktopProcessMetric>;
}

export interface StarlingDesktopBridge {
  health(input: HealthInput): Promise<ServerHealth>;
  transcribe(input: TranscribeInput): Promise<TranscriptionResult>;
  diagnostics(): Promise<DesktopDiagnostics>;
  ready(): void;
  onToggleRecording(callback: () => void): () => void;
  setPendingAudio(state: PendingAudioState): void;
  /**
   * Optional so older preload builds (and the browser preview, which has no
   * bridge at all) degrade to session-only key retention instead of failing:
   * callers must guard on the method's presence and read the returned
   * protection status before persisting anything (B10).
   */
  storeRefinementKey?(input: RefinementKeySaveInput): Promise<RefinementKeySaveResult>;
  loadRefinementKey?(input: RefinementKeyLoadInput): Promise<RefinementKeyLoadResult>;
  /**
   * The main process chose an explicit Discard in the close guard; the
   * renderer drops its durable streaming journal, then reports back via
   * discardCleanedUp so the window is not destroyed mid-delete.
   */
  onDiscardPending(callback: () => void): () => void;
  discardCleanedUp(): void;
  /**
   * Live-streaming transport over IPC (B01). Optional so older preload builds
   * and the browser preview (no bridge at all) keep using the renderer's
   * direct WebSocket, which the static CSP permits for loopback endpoints.
   * Callers must guard on streamOpen's presence.
   */
  streamOpen?(input: StreamOpenInput): Promise<{ streamId: number }>;
  streamSend?(input: StreamSendInput): Promise<void>;
  streamCommand?(input: StreamCommandInput): Promise<StreamCommandResult>;
  streamClose?(input: StreamCloseInput): void;
  onStreamEvent?(listener: (message: StreamEventMessage) => void): () => void;
}
