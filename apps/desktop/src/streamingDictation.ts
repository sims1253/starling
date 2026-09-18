import type {
  FinishedStreamCapture,
  StarlingStreamEvent,
  TranscriptionResult,
} from "@starling/dictation";

/**
 * Orchestrates one live-streamed recording: every captured PCM16 chunk is
 * journaled durably first, and only then offered to the streaming socket.
 * Any failure marks the take "not stream-committable"; the recording keeps
 * going and Stop falls back to the batch upload of the assembled WAV, so a
 * streaming failure can never lose or shorten the transcript source.
 */

export interface StreamingTransport {
  open(): Promise<void>;
  sendPcm(bytes: Uint8Array): Promise<void>;
  commit(): Promise<TranscriptionResult>;
  close(): void;
  readonly isOpen: boolean;
  onEvent(listener: (event: StarlingStreamEvent) => void): () => void;
}

export interface StreamingCapture {
  append(pcm16: Uint8Array): Promise<void>;
  finish(durationMs?: number): Promise<FinishedStreamCapture>;
  abandon(): Promise<void>;
}

export type StreamingState = "connecting" | "live" | "unavailable" | "interrupted";

export interface StreamingDictationEvents {
  onPartial?(text: string): void;
  onStateChange?(state: StreamingState, detail?: string): void;
}

export interface StreamingDictationResult extends FinishedStreamCapture {
  /** True only when the final transcript arrived over the stream. */
  readonly streamed: boolean;
  readonly transcript?: TranscriptionResult;
  /** Why the take fell back to the batch path, for session visibility. */
  readonly streamNote?: string;
}

function messageFrom(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

export class StreamingDictation {
  private readonly transport: StreamingTransport;
  private readonly capture: StreamingCapture;
  private readonly events: StreamingDictationEvents;
  private state: StreamingState = "connecting";
  private detail?: string;
  /** False the moment any audio might not have reached the server intact. */
  private complete = true;
  private queue: Uint8Array[] = [];
  /** Nonzero chunks journaled; zero means the microphone never arrived here. */
  private journaledChunks = 0;
  private pipeline: Promise<void> = Promise.resolve();
  private stopEvents: () => void;

  constructor(
    transport: StreamingTransport,
    capture: StreamingCapture,
    events: StreamingDictationEvents = {},
  ) {
    this.transport = transport;
    this.capture = capture;
    this.events = events;
    this.stopEvents = transport.onEvent((event) => {
      this.handleEvent(event);
    });
  }

  get streamingState(): StreamingState {
    return this.state;
  }

  /** Chunks durably journaled so far; zero means no audio ever arrived. */
  get journaledChunkCount(): number {
    return this.journaledChunks;
  }

  /**
   * Connect the socket. Never rejects: a connection that cannot be
   * established leaves the take in batch mode with a reason.
   */
  async connect(): Promise<void> {
    try {
      await this.transport.open();
    } catch (cause) {
      if (this.state === "connecting") {
        this.setState("unavailable", `Live transcription is unavailable (${messageFrom(cause)}).`);
        this.complete = false;
        this.queue = [];
      }

      return;
    }

    if (this.state === "connecting") {
      this.state = "live";
      this.setState("live");
      this.flushQueue();
    }
  }

  /** One captured PCM16 16 kHz mono chunk. Journaled first, then streamed. */
  onChunk(pcm16: Uint8Array): void {
    if (pcm16.byteLength === 0) return;

    this.journaledChunks += 1;
    this.pipeline = this.pipeline
      .then(() => this.capture.append(pcm16))
      .then(() => {
        this.deliver(pcm16);
      })
      .catch((cause) => {
        this.fail(`Live transcription stopped: ${messageFrom(cause)}`);
      });
  }

  /**
   * Mark the stream unusable from outside (for example a capture running at
   * an unexpected sample rate); Stop falls back to the batch path.
   */
  fail(reason: string): void {
    this.complete = false;
    this.queue = [];

    if (this.state === "live" || this.state === "connecting") {
      this.setState("interrupted", reason);
    }

    this.transport.close();
  }

  /** Drain every chunk, finalize the WAV (source of truth), then commit. */
  async finish(durationMs?: number): Promise<StreamingDictationResult> {
    await this.pipeline;

    // The durable save completes before anything is accepted as final, per
    // the fidelity contract.
    const finished = await this.capture.finish(durationMs);

    if (this.canCommit()) {
      try {
        const transcript = await this.transport.commit();

        this.close();

        return Object.freeze({ ...finished, streamed: true, transcript });
      } catch (cause) {
        this.close();

        return Object.freeze({
          ...finished,
          streamed: false,
          streamNote: `Live transcription could not finish: ${messageFrom(cause)}`,
        });
      }
    }

    this.close();

    return Object.freeze({
      ...finished,
      streamed: false,
      streamNote:
        this.journaledChunks === 0
          ? "No audio reached the live stream; the saved WAV carries the take."
          : (this.detail ??
            "Live streaming did not finish before the recording stopped; using the saved WAV."),
    });
  }

  /** Drop the journal without transcribing (take too short, user discard). */
  async abandon(): Promise<void> {
    await this.pipeline.catch(() => {});
    this.close();
    await this.capture.abandon();
  }

  private canCommit(): boolean {
    return (
      this.state === "live" &&
      this.complete &&
      this.transport.isOpen &&
      this.queue.length === 0 &&
      // An empty journal assembles a header-only WAV whose zero-duration
      // commit the server answers with an empty success; never accept that
      // as a final — the recorder's capture must carry the take (#143).
      this.journaledChunks > 0
    );
  }

  private deliver(pcm16: Uint8Array): void {
    if (this.state === "live" && this.transport.isOpen) {
      void this.transport.sendPcm(pcm16).catch((cause) => {
        this.fail(`Live transcription stopped: ${messageFrom(cause)}`);
      });
    } else if (this.state === "connecting") {
      // Hold frames until the socket opens so the stream never starts with a
      // hole; if it never opens, the take falls back to batch.
      this.queue.push(pcm16);
    }
  }

  private flushQueue(): void {
    const pending = this.queue;
    this.queue = [];

    for (const frame of pending) this.deliver(frame);
  }

  private handleEvent(event: StarlingStreamEvent): void {
    if (event.type === "partial") {
      this.events.onPartial?.(event.text);
    } else if (event.type === "error") {
      const detail = event.bufferLimit
        ? `Live transcription stopped: the server's live buffer filled up (${event.limitSeconds ?? "?"} s).`
        : `Live transcription stopped: ${event.message}`;

      this.fail(detail);
    } else if (event.type === "closed") {
      if (this.state === "live") {
        this.complete = false;
        this.queue = [];
        this.setState(
          "interrupted",
          "The live transcription connection closed. The recording continues and will upload when you stop.",
        );
      } else if (this.state === "connecting") {
        // The socket died before it ever opened; the take never went live.
        this.complete = false;
        this.queue = [];
        this.setState(
          "unavailable",
          "Live transcription could not connect. The recording continues and will upload when you stop.",
        );
      }
    }
  }

  private setState(state: StreamingState, detail?: string): void {
    this.state = state;
    this.detail = detail;
    this.events.onStateChange?.(state, detail);
  }

  private close(): void {
    this.stopEvents();
    this.transport.close();
  }
}
