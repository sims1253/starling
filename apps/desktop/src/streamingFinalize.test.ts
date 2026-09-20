import { describe, expect, it } from "vite-plus/test";
import type {
  DictationSession,
  FinishedStreamCapture,
  TranscriptionResult,
} from "@starling/dictation";

import { finishStreamingTake, type StreamingFinalizeDeps } from "./streamingFinalize";
import {
  StreamingDictation,
  type StreamingDictationResult,
  type StreamingTransport,
} from "./streamingDictation";

const wav = new Blob([new Uint8Array(44)]);

/** The subset of a stored session the finalize writes need to round-trip. */
interface FinalizeSessionDraft {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: "captured" | "transcribed";
  wav: Blob;
  attemptCount: number;
  transcript?: TranscriptionResult;
  streamed?: boolean;
}

function session(id = "take-one"): DictationSession {
  const draft: FinalizeSessionDraft = {
    id,
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
    status: "captured",
    wav,
    attemptCount: 0,
  };

  const stored: DictationSession = Object.freeze(draft);

  return stored;
}

const transcript: TranscriptionResult = { text: "stream final", segments: [] };

interface Deferred {
  readonly promise: Promise<StreamingDictationResult>;
  resolve: (value: StreamingDictationResult) => void;
}

function deferred(): Deferred {
  let resolve: (value: StreamingDictationResult) => void = () => {};

  const promise = new Promise<StreamingDictationResult>((inner) => {
    resolve = inner;
  });

  return { promise, resolve };
}

class FakeStream extends StreamingDictation {
  abandons = 0;
  private pending: Deferred | undefined;
  finishResult: StreamingDictationResult = Object.freeze({
    wav,
    durationMs: 500,
    session: session(),
    streamed: false,
  });

  constructor() {
    super(new SilentTransport(), new SilentCapture());
  }

  override get journaledChunkCount(): number {
    return this.journaled;
  }

  journaled = 0;

  override async finish(
    _durationMs?: number,
    isCancelled?: () => boolean,
    onDurableSave?: (session: DictationSession) => void,
  ): Promise<StreamingDictationResult> {
    if (this.pending) {
      const settled = await this.pending.promise;

      // Mirrors StreamingDictation.finish: a take cancelled while the
      // journal drained skips the commit and hands the provisional row
      // back for deletion.
      if (isCancelled?.()) return { ...settled, streamed: false };

      if (settled.session !== undefined) onDurableSave?.(settled.session);

      return settled;
    }

    // Mirrors StreamingDictation.finish's controller probe after the
    // capture layer's own, then the durable report on the live path.
    if (isCancelled?.()) return { ...this.finishResult, streamed: false };

    if (this.finishResult.session !== undefined) {
      onDurableSave?.(this.finishResult.session);
    }

    return this.finishResult;
  }

  override async abandon(): Promise<void> {
    this.abandons += 1;
  }

  blockFinish(): Deferred {
    const gate = deferred();

    this.pending = gate;

    return gate;
  }
}

class SilentTransport implements StreamingTransport {
  get isOpen(): boolean {
    return true;
  }

  async open(): Promise<void> {}

  async sendPcm(): Promise<void> {}

  async commit(): Promise<TranscriptionResult> {
    return transcript;
  }

  close(): void {}

  onEvent(): () => void {
    return () => {};
  }
}

class SilentCapture {
  async append(): Promise<void> {}

  async finish(durationMs?: number): Promise<FinishedStreamCapture> {
    return Object.freeze({ wav, durationMs: durationMs ?? 0, session: session() });
  }

  async abandon(): Promise<void> {}
}

class FakeStore {
  saved: Array<{ id: string; streamed?: boolean; protocol?: string }> = [];
  noted: Array<{ id: string; message: string }> = [];
  deleted: string[] = [];
  selected: string[] = [];
  sessions = new Map<string, DictationSession>([["take-one", session()]]);
  transcribed: string[] = [];

  async saveTranscript(
    id: string,
    value: TranscriptionResult,
    options?: { streamed?: boolean; protocol?: string },
  ): Promise<DictationSession> {
    this.saved.push({ id, streamed: options?.streamed, protocol: options?.protocol });

    const current = this.sessions.get(id) ?? session(id);

    const next: FinalizeSessionDraft = {
      ...current,
      status: "transcribed",
      transcript: value,
      streamed: options?.streamed,
    };

    const stored: DictationSession = Object.freeze(next);

    this.sessions.set(id, stored);

    return stored;
  }

  async noteStreamError(id: string, message: string): Promise<DictationSession> {
    this.noted.push({ id, message });
    const next = this.sessions.get(id) ?? session(id);

    this.sessions.set(id, next);

    return next;
  }

  async delete(id: string): Promise<void> {
    this.deleted.push(id);
    this.sessions.delete(id);
  }
}

function handlersFor(
  stream: FakeStream,
  store: FakeStore,
  isCurrentTake: () => boolean,
  park?: (wav: Blob) => void,
): StreamingFinalizeDeps {
  return {
    stream,
    durationMs: 500,
    isCurrentTake,
    parkUnsavedWav:
      park ??
      (() => {
        throw new Error("parked unexpectedly");
      }),
    refresh: () => Promise.resolve(),
    setSelectedId: (id) => store.selected.push(id),
    setConnectionReady: () => {},
    transcribe: (next: DictationSession) => {
      store.transcribed.push(next.id);

      return Promise.resolve();
    },
  };
}

/** Keeps a promise pending until open(), like a slow transcription backend. */
function gate(): { readonly promise: Promise<void>; open(): void } {
  let open: () => void = () => {};

  const promise = new Promise<void>((resolve) => {
    open = resolve;
  });

  return { promise, open };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("finishStreamingTake", () => {
  it("abandons an empty journal and falls back to the batch path", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 0;

    const result = await finishStreamingTake(
      handlersFor(stream, store, () => true),
      store,
    );

    expect(result).toEqual({ streamed: false, batchFallback: true });
    expect(stream.abandons).toBe(1);
    expect(store.saved).toEqual([]);
    expect(store.deleted).toEqual([]);
    expect(store.transcribed).toEqual([]);
  });

  it("saves the streamed transcript when the finalize is still current", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;
    stream.finishResult = Object.freeze({
      wav,
      durationMs: 500,
      session: session(),
      streamed: true,
      transcript,
    });

    const result = await finishStreamingTake(
      handlersFor(stream, store, () => true),
      store,
    );

    expect(result.streamed).toBe(true);
    expect(result.session?.id).toBe("take-one");
    expect(result.batchFallback).toBe(false);
    expect(store.saved).toEqual([{ id: "take-one", streamed: true, protocol: "starling" }]);
    expect(store.deleted).toEqual([]);
  });

  it("undoes the streamed save when Discard lands during the write", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;
    stream.finishResult = Object.freeze({
      wav,
      durationMs: 500,
      session: session(),
      streamed: true,
      transcript,
    });

    // Current through the controller probe and the pre-save probes (calls
    // 1–3), stale at the post-save probe (call 4): the completed row is
    // this finalize's to remove, and the dropped take must not be surfaced.
    let probes = 0;

    const result = await finishStreamingTake(
      handlersFor(stream, store, () => ++probes < 4),
      store,
    );

    expect(result.discarded).toBe(true);
    expect(store.saved).toEqual([{ id: "take-one", streamed: true, protocol: "starling" }]);
    expect(store.deleted).toEqual(["take-one"]);
    expect(store.selected).toEqual([]);
  });

  it("does not fall through to the batch path when Discard lands on an empty journal", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 0;

    const result = await finishStreamingTake(
      handlersFor(stream, store, () => false),
      store,
    );

    expect(result).toEqual({ streamed: false, discarded: true, batchFallback: false });
    expect(stream.abandons).toBe(1);
    expect(store.saved).toEqual([]);
    expect(store.deleted).toEqual([]);
    expect(store.transcribed).toEqual([]);
  });

  it("writes nothing when Discard lands while the finalize is in flight", async () => {
    // The #160 race: Stop's finalize is awaiting the journal drain when the
    // close-guard Discard bumps the generation; every durable write after
    // that point must be skipped.
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;

    let current = true;

    const gate = stream.blockFinish();

    const finalizing = finishStreamingTake(
      handlersFor(stream, store, () => current),
      store,
    );

    current = false;
    gate.resolve(
      Object.freeze({ wav, durationMs: 500, session: session(), streamed: true, transcript }),
    );

    const result = await finalizing;

    expect(result.discarded).toBe(true);
    expect(result.session).toBeUndefined();
    expect(store.saved).toEqual([]);
    expect(store.noted).toEqual([]);
    expect(store.transcribed).toEqual([]);
    // The provisional row this finalize's finish created is removed; nothing
    // else is touched.
    expect(store.deleted).toEqual(["take-one"]);
  });

  it("leaves the Discard's settled state untouched once the take is stale", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;
    stream.finishResult = Object.freeze({
      wav,
      durationMs: 500,
      session: session(),
      streamed: false,
      streamNote: "socket closed",
    });

    const result = await finishStreamingTake(
      handlersFor(stream, store, () => false),
      store,
    );

    expect(result.discarded).toBe(true);
    expect(store.saved).toEqual([]);
    expect(store.noted).toEqual([]);
    expect(store.transcribed).toEqual([]);
    expect(store.deleted).toEqual(["take-one"]);
  });

  it("parks the unsaved WAV and throws when the durable save itself fails", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;
    stream.finishResult = Object.freeze({
      wav,
      durationMs: 500,
      session: undefined,
      failure: new Error("idb write denied"),
      streamed: false,
    });

    const parked: Blob[] = [];
    let thrown: unknown;

    try {
      await finishStreamingTake(
        handlersFor(
          stream,
          store,
          () => true,
          (next) => parked.push(next),
        ),
        store,
      );
    } catch (cause) {
      thrown = cause;
    }

    expect(thrown).toBeInstanceOf(Error);
    expect(String(thrown)).toContain("Local storage failed");
    expect(parked).toEqual([wav]);
    expect(store.saved).toEqual([]);
    expect(store.noted).toEqual([]);
    expect(store.transcribed).toEqual([]);
    expect(store.deleted).toEqual([]);
  });

  it("deletes the batch-transcribed session when Discard lands during transcribe", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;
    stream.finishResult = Object.freeze({
      wav,
      durationMs: 500,
      session: session(),
      streamed: false,
      streamNote: "socket closed",
    });

    // Current through the controller probe and the pre-transcribe probes
    // (calls 1–4), stale at the post-transcribe probe (call 5): the
    // transcription completed, but the row is this finalize's to remove.
    let probes = 0;

    const result = await finishStreamingTake(
      handlersFor(stream, store, () => ++probes < 5),
      store,
    );

    expect(result.discarded).toBe(true);
    expect(store.transcribed).toEqual(["take-one"]);
    expect(store.deleted).toEqual(["take-one"]);
  });

  it("notes the stream error and transcribes via batch on the live path", async () => {
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;
    stream.finishResult = Object.freeze({
      wav,
      durationMs: 500,
      session: session(),
      streamed: false,
      streamNote: "socket closed",
    });

    const result = await finishStreamingTake(
      handlersFor(stream, store, () => true),
      store,
    );

    expect(result.streamed).toBe(false);
    expect(result.discarded).toBeUndefined();
    expect(store.noted).toEqual([{ id: "take-one", message: "socket closed" }]);
    expect(store.transcribed).toEqual(["take-one"]);
    expect(store.deleted).toEqual([]);
  });

  it("releases the capture lifecycle at the durable save, before the transcription runs (B03)", async () => {
    // The B03 acceptance shape: a deliberately delayed transcription
    // response. The release fires once the journal is durably owned by its
    // session, while the transcript work is still pending — a new take may
    // start here.
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;
    stream.finishResult = Object.freeze({
      wav,
      durationMs: 500,
      session: session(),
      streamed: false,
    });

    const transcribing = gate();
    const released: number[] = [];
    const handlers = handlersFor(stream, store, () => true);

    const finalizing = finishStreamingTake(
      {
        ...handlers,
        transcribe: () =>
          transcribing.promise.then(() => {
            store.transcribed.push("take-one");
          }),
        onDurableSave: () => released.push(released.length),
      },
      store,
    );

    await flush();

    expect(released.length).toBe(1);
    expect(store.transcribed).toEqual([]);

    let settled = false;

    void finalizing.then(() => {
      settled = true;
    });

    await flush();
    expect(settled).toBe(false);

    transcribing.open();
    const result = await finalizing;

    expect(result.batchFallback).toBe(false);
    expect(released.length).toBe(1);
    expect(store.transcribed).toEqual(["take-one"]);
  });

  it("does not release the capture lifecycle for a discarded take", async () => {
    // The #160 protection: a close-guard Discard that lands mid-finalize
    // settles the take without it; the release belongs to the durable save
    // that never happened (#160 + B03).
    const stream = new FakeStream();
    const store = new FakeStore();
    stream.journaled = 1;

    const gateKeeper = stream.blockFinish();
    const released: number[] = [];
    const finalizing = finishStreamingTake(
      {
        ...handlersFor(stream, store, () => false),
        onDurableSave: () => released.push(released.length),
      },
      store,
    );

    gateKeeper.resolve(
      Object.freeze({ wav, durationMs: 500, session: session(), streamed: false }),
    );

    const result = await finalizing;

    expect(result.discarded).toBe(true);
    expect(released).toEqual([]);
    expect(store.deleted).toEqual(["take-one"]);
  });

  it("attaches late transcripts only to their originating takes, in any order (B03)", async () => {
    // Two takes finalizing against one store while a slow backend answers
    // the second first: each transcript lands on the session that recorded
    // it, and each take's release fired exactly once, at its own save.
    const store = new FakeStore();

    const streamOne = new FakeStream();
    streamOne.journaled = 1;

    const streamTwo = new FakeStream();
    streamTwo.journaled = 1;

    // Both journals are still draining; neither take is durable yet.
    const gateOne = streamOne.blockFinish();
    const gateTwo = streamTwo.blockFinish();

    const released: string[] = [];

    const finalizeOne = finishStreamingTake(
      {
        ...handlersFor(streamOne, store, () => true),
        onDurableSave: () => released.push("take-one"),
      },
      store,
    );
    const finalizeTwo = finishStreamingTake(
      {
        ...handlersFor(streamTwo, store, () => true),
        onDurableSave: () => released.push("take-two"),
      },
      store,
    );

    await flush();

    expect(released).toEqual([]);

    // The second take's backend responds first: its journal commits, its
    // release fires, its transcript saves — while the first take is still
    // waiting for its own response.
    gateTwo.resolve(
      Object.freeze({
        wav,
        durationMs: 500,
        session: session("take-two"),
        streamed: true,
        transcript,
      }),
    );

    const resultTwo = await finalizeTwo;

    expect(released).toEqual(["take-two"]);

    let oneSettled = false;

    void finalizeOne.then(() => {
      oneSettled = true;
    });

    await flush();
    expect(oneSettled).toBe(false);

    gateOne.resolve(
      Object.freeze({
        wav,
        durationMs: 500,
        session: session("take-one"),
        streamed: true,
        transcript,
      }),
    );

    const resultOne = await finalizeOne;

    expect(resultOne.session?.id).toBe("take-one");
    expect(resultTwo.session?.id).toBe("take-two");
    expect(released).toEqual(["take-two", "take-one"]);
    expect(store.saved.map((saved) => saved.id)).toEqual(["take-two", "take-one"]);
    expect(store.deleted).toEqual([]);
  });
});
