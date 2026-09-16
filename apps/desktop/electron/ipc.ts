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
}
