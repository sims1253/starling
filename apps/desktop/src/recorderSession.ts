type StoppableTrack = {
  stop(): void;
};

/**
 * The minimal structural surface of the Web Audio objects one capture owns.
 * Narrow ports keep this module testable with plain fakes while still
 * accepting the real DOM nodes unchanged.
 */
export type RecorderHandles = {
  readonly stream: { getTracks(): Array<StoppableTrack> };
  readonly context: { readonly sampleRate: number; close(): Promise<void> };
  readonly processor: {
    onaudioprocess: ScriptProcessorNode["onaudioprocess"];
    disconnect(): void;
  };
  readonly source: { disconnect(): void };
  readonly analyser: {
    readonly frequencyBinCount: number;
    getByteFrequencyData(array: Uint8Array): void;
  };
};

/**
 * Stops every resource one capture acquired. Safe to call twice: tracks and
 * contexts tolerate being stopped or closed repeatedly.
 */
export async function discardRecorderHandles(handles: RecorderHandles): Promise<void> {
  handles.processor.onaudioprocess = null;

  try {
    handles.processor.disconnect();
  } catch {
    /* already disconnected */
  }

  try {
    handles.source.disconnect();
  } catch {
    /* already disconnected */
  }

  handles.stream.getTracks().forEach((track) => track.stop());

  try {
    await handles.context.close();
  } catch {
    /* already closed */
  }
}

/** The stopped-capture fields the keep/discard verdict depends on. */
export interface StoppedTakeCapture {
  readonly sampleCount: number;
  readonly durationMs: number;
}

/** What Stop should do with the capture a take produced. */
export type StoppedTakeVerdict =
  /** Real audio was captured; the take must stay reviewable and retryable. */
  | { readonly keep: true }
  /** No samples arrived; an accidental empty activation, safe to drop. */
  | { readonly keep: false; readonly reason: "empty" };

/**
 * Decide whether a stopped capture is a take worth keeping (B02): any capture
 * with samples is kept whatever its duration — a short answer such as a
 * letter or "no" is exactly the payload dictation exists for — and only a
 * capture with no samples at all is an accidental empty activation. Duration
 * is accepted so the rule stays visible, but it never discards audio alone;
 * explicit discard is a separate path that never consults this verdict.
 */
export function stoppedTakeVerdict(capture: StoppedTakeCapture | undefined): StoppedTakeVerdict {
  if (capture && capture.sampleCount > 0) return { keep: true };

  return { keep: false, reason: "empty" };
}

/**
 * Owns the Web Audio handles of the current microphone capture.
 *
 * `release()` detaches the handles synchronously, before awaiting
 * `AudioContext.close()`, and the cleanup after that await touches only the
 * detached locals. A `start()` issued while an older release is still awaiting
 * close() therefore installs its handles into a free slot, and the late
 * cleanup can never erase them (#119).
 */
export class RecorderSession {
  private handles: RecorderHandles | undefined;

  current(): RecorderHandles | undefined {
    return this.handles;
  }

  install(handles: RecorderHandles): void {
    if (this.handles) void discardRecorderHandles(this.handles);

    this.handles = handles;
  }

  async release(): Promise<void> {
    const owned = this.handles;

    this.handles = undefined;

    if (owned) await discardRecorderHandles(owned);
  }
}
