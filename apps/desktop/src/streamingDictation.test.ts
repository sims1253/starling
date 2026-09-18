import { describe, expect, it } from "vite-plus/test";
import type { DictationSession, StarlingStreamEvent } from "@starling/dictation";

import {
  StreamingDictation,
  type StreamingCapture,
  type StreamingTransport,
} from "./streamingDictation";

class FakeTransport implements StreamingTransport {
  opens = 0;
  sends: Uint8Array[] = [];
  commits = 0;
  closeCalls = 0;
  openFailure: Error | undefined;
  commitResult = { text: "stream final", segments: [] };
  commitFailure: Error | undefined;
  /** Keeps open() pending until releaseOpen(), like a real handshake. */
  blockOpen = false;
  private listeners = new Set<(event: StarlingStreamEvent) => void>();
  private open_ = false;
  private release?: () => void;

  get isOpen(): boolean {
    return this.open_;
  }

  async open(): Promise<void> {
    this.opens += 1;

    if (this.openFailure) throw this.openFailure;

    if (this.blockOpen) await new Promise<void>((resolve) => (this.release = resolve));

    this.open_ = true;
  }

  releaseOpen(): void {
    this.release?.();
    this.release = undefined;
  }

  async sendPcm(bytes: Uint8Array): Promise<void> {
    if (!this.open_) throw new Error("transport closed");
    this.sends.push(new Uint8Array(bytes));
  }

  async commit(): Promise<{ text: string; segments: never[] }> {
    this.commits += 1;

    if (this.commitFailure) throw this.commitFailure;

    return this.commitResult;
  }

  close(): void {
    this.closeCalls += 1;
    this.open_ = false;
    this.releaseOpen();
    this.emit({ type: "closed" });
  }

  onEvent(listener: (event: StarlingStreamEvent) => void): () => void {
    this.listeners.add(listener);

    return () => this.listeners.delete(listener);
  }

  emit(event: StarlingStreamEvent): void {
    for (const listener of Array.from(this.listeners)) listener(event);
  }
}

class FakeCapture implements StreamingCapture {
  appended: Uint8Array[] = [];
  finishes = 0;
  abandons = 0;
  appendFailure: Error | undefined;
  finishFailure: Error | undefined;

  async append(pcm16: Uint8Array): Promise<void> {
    if (this.appendFailure) throw this.appendFailure;
    this.appended.push(new Uint8Array(pcm16));
  }

  async finish(durationMs?: number) {
    this.finishes += 1;

    if (this.finishFailure) throw this.finishFailure;

    const wav = new Blob([new Uint8Array(44)]);

    const session: DraftSession = {
      id: "streamed-take",
      createdAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      status: "captured",
      wav,
      attemptCount: 0,
    };

    if (durationMs !== undefined) session.durationMs = durationMs;

    const stored: DictationSession = Object.freeze(session);

    return Object.freeze({ wav, durationMs: durationMs ?? 0, session: stored });
  }

  async abandon(): Promise<void> {
    this.abandons += 1;
  }
}

/** The subset of a stored session this fake needs to build. */
interface DraftSession {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: "captured";
  wav: Blob;
  attemptCount: number;
  durationMs?: number;
}

const frame = (bytes: number): Uint8Array => new Uint8Array(bytes).fill(1);

const drained = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

describe("StreamingDictation", () => {
  it("streams chunks only after they are journaled, then commits a final transcript", async () => {
    const transport = new FakeTransport();
    const capture = new FakeCapture();
    const partials: string[] = [];
    const states: string[] = [];

    const controller = new StreamingDictation(transport, capture, {
      onPartial: (text) => partials.push(text),
      onStateChange: (state) => states.push(state),
    });

    await controller.connect();
    controller.onChunk(frame(8));
    await drained();

    expect(capture.appended.map((chunk) => chunk.byteLength)).toEqual([8]);
    expect(transport.sends.map((chunk) => chunk.byteLength)).toEqual([8]);

    transport.emit({ type: "partial", text: "so far" });
    expect(partials).toEqual(["so far"]);

    const result = await controller.finish(500);

    expect(result.streamed).toBe(true);
    expect(result.transcript?.text).toBe("stream final");
    expect(capture.finishes).toBe(1);
    expect(transport.commits).toBe(1);
    expect(states).toEqual(["live"]);
    expect(result.session).toBeTruthy();
  });

  it("queues chunks captured before the socket opens and flushes them in order", async () => {
    const transport = new FakeTransport();
    transport.blockOpen = true;
    const capture = new FakeCapture();
    const controller = new StreamingDictation(transport, capture);

    const connecting = controller.connect();
    controller.onChunk(frame(2));
    controller.onChunk(frame(4));
    await drained();

    // Nothing streamed while connecting; both chunks journaled.
    expect(transport.sends).toEqual([]);
    expect(capture.appended.length).toBe(2);

    transport.releaseOpen();
    await connecting;
    await drained();

    expect(transport.sends.map((chunk) => chunk.byteLength)).toEqual([2, 4]);
  });

  it("falls back to batch when the connection cannot be established", async () => {
    const transport = new FakeTransport();
    transport.openFailure = new Error("connection refused");
    const capture = new FakeCapture();

    const states: Array<[string, string | undefined]> = [];

    const controller = new StreamingDictation(transport, capture, {
      onStateChange: (state, detail) => states.push([state, detail]),
    });

    await controller.connect();
    controller.onChunk(frame(6));
    await drained();

    expect(transport.sends).toEqual([]);
    expect(capture.appended.length).toBe(1);

    const result = await controller.finish();

    expect(result.streamed).toBe(false);
    expect(result.transcript).toBeUndefined();
    expect(transport.commits).toBe(0);
    expect(result.streamNote ?? "").toMatch(/unavailable|refused/);
    expect(states[0]?.[1] ?? "").toMatch(/connection refused/);
  });

  it("reports a socket that closes before opening as unavailable, not interrupted", async () => {
    const transport = new FakeTransport();
    transport.blockOpen = true;
    const capture = new FakeCapture();

    const states: string[] = [];

    const controller = new StreamingDictation(transport, capture, {
      onStateChange: (state) => states.push(state),
    });

    const connecting = controller.connect();
    controller.onChunk(frame(6));
    await drained();

    // The close lands while the connect promise is still pending, like a
    // refused socket whose error and close share a tick.
    transport.close();
    await connecting;

    expect(controller.streamingState).toBe("unavailable");
    expect(capture.appended.length).toBe(1);

    const result = await controller.finish();

    expect(result.streamed).toBe(false);
    expect(result.streamNote ?? "").toMatch(/could not connect/);
  });

  it("marks the buffer-cap error and keeps the take recoverable via batch", async () => {
    const transport = new FakeTransport();
    const capture = new FakeCapture();
    const states: Array<[string, string | undefined]> = [];

    const controller = new StreamingDictation(transport, capture, {
      onStateChange: (state, detail) => states.push([state, detail]),
    });

    await controller.connect();
    controller.onChunk(frame(6));
    await drained();

    transport.emit({
      type: "error",
      message: "stream buffer limit reached (60 s live buffer); audio ignored until reset",
      bufferLimit: true,
      limitSeconds: 60,
    });
    controller.onChunk(frame(6));
    await drained();

    // Recording continues into the journal, not the dead socket.
    expect(capture.appended.length).toBe(2);
    expect(transport.sends.length).toBe(1);
    expect(states.at(-1)?.[1] ?? "").toMatch(/live buffer filled up/);

    const result = await controller.finish();

    expect(result.streamed).toBe(false);
    expect(transport.commits).toBe(0);
  });

  it("falls back to batch when the journal fails mid-recording", async () => {
    const transport = new FakeTransport();
    const capture = new FakeCapture();
    const controller = new StreamingDictation(transport, capture);

    await controller.connect();
    controller.onChunk(frame(4));
    await drained();

    capture.appendFailure = new Error("quota exceeded");
    controller.onChunk(frame(4));
    await drained();

    expect(transport.sends.length).toBe(1);
    expect(controller.streamingState).toBe("interrupted");

    const result = await controller.finish();

    expect(result.streamed).toBe(false);
    expect(result.streamNote ?? "").toMatch(/quota exceeded/);
  });

  it("falls back to batch when commit fails after the durable save", async () => {
    const transport = new FakeTransport();
    transport.commitFailure = new Error("server busy");
    const capture = new FakeCapture();
    const controller = new StreamingDictation(transport, capture);

    await controller.connect();
    controller.onChunk(frame(4));
    await drained();

    const result = await controller.finish();

    // The session is finalized before the commit is attempted.
    expect(capture.finishes).toBe(1);
    expect(transport.commits).toBe(1);
    expect(result.streamed).toBe(false);
    expect(result.session).toBeTruthy();
    expect(result.streamNote ?? "").toMatch(/server busy/);
  });

  it("drops the journal on abandon and stops streaming", async () => {
    const transport = new FakeTransport();
    const capture = new FakeCapture();
    const controller = new StreamingDictation(transport, capture);

    await controller.connect();
    controller.onChunk(frame(4));
    await drained();

    await controller.abandon();

    expect(capture.abandons).toBe(1);
    expect(transport.closeCalls).toBe(1);
  });

  it("lets the caller fail the stream externally without touching the journal", async () => {
    const transport = new FakeTransport();
    const capture = new FakeCapture();
    const controller = new StreamingDictation(transport, capture);

    await controller.connect();
    controller.fail("microphone capture rate not supported");
    controller.onChunk(frame(4));
    await drained();

    expect(transport.sends.length).toBe(0);
    expect(capture.appended.length).toBe(1);

    const result = await controller.finish();

    expect(result.streamed).toBe(false);
    expect(result.streamNote ?? "").toMatch(/not supported/);
  });

  it("never commits a take whose journal stayed empty", async () => {
    // The #143 wrong-take shape: the microphone belonged to another
    // controller, so this one reaches Stop with zero journaled frames.
    const transport = new FakeTransport();
    const capture = new FakeCapture();
    const controller = new StreamingDictation(transport, capture);

    await controller.connect();
    await drained();

    expect(controller.journaledChunkCount).toBe(0);

    const result = await controller.finish(900);

    expect(result.streamed).toBe(false);
    expect(transport.commits).toBe(0);
    expect(result.streamNote ?? "").toMatch(/no audio reached the live stream/i);
    expect(transport.closeCalls).toBe(1);
  });

  it("counts only nonzero chunks toward the journaled chunk count", async () => {
    const transport = new FakeTransport();
    const capture = new FakeCapture();
    const controller = new StreamingDictation(transport, capture);

    controller.onChunk(new Uint8Array(0));
    expect(controller.journaledChunkCount).toBe(0);

    await controller.connect();
    controller.onChunk(frame(4));
    controller.onChunk(frame(4));
    await drained();

    expect(controller.journaledChunkCount).toBe(2);
  });
});
