import { Schema } from "effect";

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

export type HealthInput = typeof HealthInputSchema.Type;

export type TranscribeInput = typeof TranscribeInputSchema.Type;

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
   * The main process chose an explicit Discard in the close guard; the
   * renderer drops its durable streaming journal, then reports back via
   * discardCleanedUp so the window is not destroyed mid-delete.
   */
  onDiscardPending(callback: () => void): () => void;
  discardCleanedUp(): void;
}
