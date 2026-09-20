import { describe, expect, it } from "vite-plus/test";
import { StarlingStream, type StarlingStreamEvent } from "@starling/dictation";

import {
  BridgeStreamTransport,
  createStreamingTransport,
  type DesktopStreamBridge,
} from "./streamTransport";

import type { StreamCommandResult } from "../electron/ipc.js";

const TRANSCRIPT: NonNullable<StreamCommandResult["transcript"]> = { text: "bridge final", segments: [] };

class FakeBridge implements DesktopStreamBridge {
  opens: Array<{ endpoint: string }> = [];
  sends: Array<{ streamId: number; audio: ArrayBuffer }> = [];
  commands: Array<{ streamId: number; command: "commit" | "reset" | "ping" }> = [];
  closes: number[] = [];
  openFailure: Error | undefined;
  commandResult: StreamCommandResult | Error | undefined;
  private nextId = 41;
  private listeners = new Set<(message: { streamId: number; event: StarlingStreamEvent }) => void>();

  streamOpen(input: { endpoint: string }): Promise<{ streamId: number }> {
    this.opens.push({ endpoint: input.endpoint });

    if (this.openFailure) return Promise.reject(this.openFailure);

    return Promise.resolve({ streamId: this.nextId++ });
  }

  streamSend(input: { streamId: number; audio: ArrayBuffer }): Promise<void> {
    this.sends.push({ streamId: input.streamId, audio: input.audio });

    return Promise.resolve();
  }

  streamCommand(input: { streamId: number; command: "commit" | "reset" | "ping" }): Promise<StreamCommandResult> {
    this.commands.push({ streamId: input.streamId, command: input.command });

    if (this.commandResult instanceof Error) return Promise.reject(this.commandResult);

    return Promise.resolve(this.commandResult ?? { transcript: TRANSCRIPT });
  }

  streamClose(input: { streamId: number }): void {
    this.closes.push(input.streamId);
  }

  onStreamEvent(
    listener: (message: { streamId: number; event: StarlingStreamEvent }) => void,
  ): () => void {
    this.listeners.add(listener);

    return () => this.listeners.delete(listener);
  }

  /** Test double for the main process pushing starling:stream:event. */
  emit(message: { streamId: number; event: StarlingStreamEvent }): void {
    for (const listener of Array.from(this.listeners)) listener(message);
  }
}

async function openedTransport(bridge: FakeBridge, endpoint = "http://127.0.0.1:8181") {
  const transport = new BridgeStreamTransport(bridge, endpoint);
  const opening = transport.open();

  return { transport, opening };
}

describe("BridgeStreamTransport", () => {
  it("opens through the bridge and reports open", async () => {
    const bridge = new FakeBridge();
    const { transport, opening } = await openedTransport(bridge);
    await opening;

    expect(bridge.opens).toEqual([{ endpoint: "http://127.0.0.1:8181" }]);
    expect(transport.isOpen).toBe(true);
    transport.close();
  });

  it("delivers only its own stream's events, and only until close", async () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    const events: StarlingStreamEvent[] = [];
    transport.onEvent((event) => events.push(event));
    await transport.open();

    bridge.emit({ streamId: 41, event: { type: "partial", text: "hello" } });
    bridge.emit({ streamId: 999, event: { type: "partial", text: "someone else" } });

    transport.close();
    bridge.emit({ streamId: 41, event: { type: "partial", text: "after close" } });

    expect(events).toEqual([{ type: "partial", text: "hello" }]);
  });

  it("holds events that arrive before the stream id is assigned", async () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    const events: StarlingStreamEvent[] = [];
    transport.onEvent((event) => events.push(event));

    // The subscription exists from the constructor, so a fast main process
    // can push before the open() reply lands; those events must not vanish.
    const opening = transport.open();
    bridge.emit({ streamId: 41, event: { type: "partial", text: "early" } });
    await opening;

    expect(events).toEqual([{ type: "partial", text: "early" }]);
    transport.close();
  });

  it("marks the transport closed when the server closes the stream", async () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    await transport.open();
    expect(transport.isOpen).toBe(true);

    bridge.emit({ streamId: 41, event: { type: "closed", code: 1006 } });

    expect(transport.isOpen).toBe(false);
    transport.close();
  });

  it("sends PCM frames as a private copy under its stream id", async () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    await transport.open();

    const bytes = Uint8Array.from([1, 2, 3, 4]);
    const sending = transport.sendPcm(bytes);
    // Mutating the caller's view after the call must not reach IPC.
    bytes.set([99, 99, 99, 99]);
    await sending;

    expect(bridge.sends).toHaveLength(1);
    expect(bridge.sends[0]?.streamId).toBe(41);
    expect(Array.from(new Uint8Array(bridge.sends[0]?.audio ?? new ArrayBuffer(0)))).toEqual([
      1, 2, 3, 4,
    ]);

    transport.close();
  });

  it("refuses frames before the take is open", () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");

    expect(transport.isOpen).toBe(false);
    expect(transport.sendPcm(Uint8Array.from([0, 0]))).rejects.toThrow(/not open/);
  });

  it("commits through the command channel and returns the transcript", async () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    await transport.open();

    await expect(transport.commit()).resolves.toEqual(TRANSCRIPT);
    expect(bridge.commands).toEqual([{ streamId: 41, command: "commit" }]);

    transport.close();
  });

  it("surfaces a missing transcript as a failure, not a silent success", async () => {
    const bridge = new FakeBridge();
    bridge.commandResult = {};
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    await transport.open();

    await expect(transport.commit()).rejects.toThrow(/no final transcript/);
    transport.close();
  });

  it("forwards command rejections", async () => {
    const bridge = new FakeBridge();
    bridge.commandResult = new Error("reply never came");
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    await transport.open();

    await expect(transport.commit()).rejects.toThrow(/reply never came/);
    transport.close();
  });

  it("closes by stream id exactly once, then refuses further work", async () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    await transport.open();

    transport.close();
    transport.close();

    expect(bridge.closes).toEqual([41]);
    expect(transport.sendPcm(Uint8Array.from([0, 0]))).rejects.toThrow(/not open/);
  });

  it("keeps a failed open from ever addressing the main process", async () => {
    const bridge = new FakeBridge();
    bridge.openFailure = new Error("could not connect");
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");

    await expect(transport.open()).rejects.toThrow(/could not connect/);
    expect(bridge.closes).toEqual([]);

    transport.close();
    expect(bridge.closes).toEqual([]);
  });

  it("refuses a second open of the same take", async () => {
    const bridge = new FakeBridge();
    const transport = new BridgeStreamTransport(bridge, "http://127.0.0.1:8181");
    await transport.open();

    await expect(transport.open()).rejects.toThrow(/already opened/);
    transport.close();
  });
});

describe("createStreamingTransport", () => {
  const originalWindow = (globalThis as { window?: unknown }).window;

  function stubWindow(bridge: unknown): void {
    Object.defineProperty(globalThis, "window", { value: { starlingDesktop: bridge }, configurable: true });
  }

  function restoreWindow(): void {
    Object.defineProperty(globalThis, "window", { value: originalWindow, configurable: true });
  }

  it("uses the renderer socket when no desktop bridge exists (browser preview)", () => {
    stubWindow(undefined);

    try {
      expect(createStreamingTransport("http://127.0.0.1:8181")).toBeInstanceOf(StarlingStream);
    } finally {
      restoreWindow();
    }
  });

  /** createStreamingTransport detaches the five channels from the bridge
   * object, so a fake using `this` must expose bound functions — exactly the
   * shape the real preload bridge (closures over invoke/ipcRenderer) has. */
  function boundBridge(fake: FakeBridge): DesktopStreamBridge {
    return {
      streamOpen: (input) => fake.streamOpen(input),
      streamSend: (input) => fake.streamSend(input),
      streamCommand: (input) => fake.streamCommand(input),
      streamClose: (input) => fake.streamClose(input),
      onStreamEvent: (listener) => fake.onStreamEvent(listener),
    };
  }

  it("uses the native bridge when every stream channel is present", () => {
    stubWindow(boundBridge(new FakeBridge()));

    try {
      const transport = createStreamingTransport("http://127.0.0.1:8181");

      expect(transport).toBeInstanceOf(BridgeStreamTransport);
      transport.close();
    } finally {
      restoreWindow();
    }
  });

  it("falls back to the renderer socket for a bridge missing any channel", () => {
    const channels = boundBridge(new FakeBridge()) as Partial<DesktopStreamBridge>;

    delete channels.onStreamEvent;
    stubWindow(channels);

    try {
      expect(createStreamingTransport("http://127.0.0.1:8181")).toBeInstanceOf(StarlingStream);
    } finally {
      restoreWindow();
    }
  });

  it("still throws on endpoints no transport can take (caller fallback parity)", () => {
    stubWindow(undefined);

    try {
      expect(() => createStreamingTransport("ftp://192.168.1.5:8181")).toThrow(TypeError);
    } finally {
      restoreWindow();
    }
  });
});
