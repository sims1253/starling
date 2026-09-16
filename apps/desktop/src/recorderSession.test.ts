import { describe, expect, it } from "vite-plus/test";
import { discardRecorderHandles, RecorderSession, type RecorderHandles } from "./recorderSession";

function fakeHandles() {
  const calls = { trackStops: 0, closes: 0, processorDisconnects: 0 };

  let resolveClose: () => void = () => {};

  const closed = new Promise<void>((resolve) => {
    resolveClose = resolve;
  });

  const handles: RecorderHandles = {
    stream: { getTracks: () => [{ stop: () => (calls.trackStops += 1) }] },
    context: {
      sampleRate: 48_000,

      close: () => {
        calls.closes += 1;

        return closed;
      },
    },
    processor: {
      onaudioprocess: () => {},
      disconnect: () => (calls.processorDisconnects += 1),
    },
    source: { disconnect: () => {} },
    analyser: { frequencyBinCount: 0, getByteFrequencyData: () => {} },
  };

  return { calls, handles, resolveClose };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("RecorderSession", () => {
  it("keeps a capture installed while an older release is still awaiting close()", async () => {
    // The #119 interleaving: stop A, start B before A's AudioContext.close()
    // resolves, then confirm A's late cleanup never erases B's handles.
    const a = fakeHandles();
    const b = fakeHandles();
    const session = new RecorderSession();

    session.install(a.handles);

    const releasingA = session.release();

    // Handles are detached synchronously, so B installs into a free slot
    // while A's close() is still pending.
    expect(session.current()).toBeUndefined();

    session.install(b.handles);

    expect(session.current()).toBe(b.handles);

    a.resolveClose();
    await releasingA;

    expect(session.current()).toBe(b.handles);

    expect(a.calls.trackStops).toBe(1);
    expect(a.calls.closes).toBe(1);
    expect(b.calls.trackStops).toBe(0);
    expect(b.calls.closes).toBe(0);

    // B must still be stoppable exactly like a first-class capture.
    const releasingB = session.release();

    b.resolveClose();
    await releasingB;

    expect(session.current()).toBeUndefined();

    expect(b.calls.trackStops).toBe(1);
    expect(b.calls.closes).toBe(1);
  });

  it("awaits AudioContext.close() before settling", async () => {
    const a = fakeHandles();
    const session = new RecorderSession();

    session.install(a.handles);

    let settled = false;

    const releasing = session.release().then(() => {
      settled = true;
    });

    await flush();

    expect(settled).toBe(false);

    a.resolveClose();
    await releasing;

    expect(settled).toBe(true);
  });

  it("stops each capture's track exactly once across overlapping releases", async () => {
    const a = fakeHandles();
    const session = new RecorderSession();

    session.install(a.handles);

    const releases = Promise.all([session.release(), session.release(), session.release()]);

    a.resolveClose();
    await releases;

    expect(a.calls.trackStops).toBe(1);
    expect(a.calls.closes).toBe(1);

    await session.release();

    expect(a.calls.trackStops).toBe(1);
  });

  it("discards stray handles defensively when install replaces them", async () => {
    const stray = fakeHandles();
    const next = fakeHandles();
    const session = new RecorderSession();

    session.install(stray.handles);
    session.install(next.handles);

    expect(session.current()).toBe(next.handles);

    stray.resolveClose();
    await flush();

    expect(stray.calls.trackStops).toBe(1);
    expect(next.calls.trackStops).toBe(0);
  });
});

describe("discardRecorderHandles", () => {
  it("stops every acquired resource and detaches the audio callback", async () => {
    const a = fakeHandles();

    a.resolveClose();
    await discardRecorderHandles(a.handles);

    expect(a.handles.processor.onaudioprocess).toBeNull();
    expect(a.calls.processorDisconnects).toBe(1);
    expect(a.calls.trackStops).toBe(1);
    expect(a.calls.closes).toBe(1);
  });

  it("survives nodes that are already torn down", async () => {
    let disconnects = 0;

    const handles: RecorderHandles = {
      stream: { getTracks: () => [{ stop: () => {} }] },
      context: { sampleRate: 48_000, close: () => Promise.reject(new Error("already closed")) },
      processor: {
        onaudioprocess: () => {},
        disconnect: () => {
          disconnects += 1;
          throw new Error("already disconnected");
        },
      },
      source: { disconnect: () => {} },
      analyser: { frequencyBinCount: 0, getByteFrequencyData: () => {} },
    };

    await discardRecorderHandles(handles);

    expect(disconnects).toBe(1);
  });
});
