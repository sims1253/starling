import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";

import {
  DictationInputError,
  DictationTimeoutError,
  DictationTransportError,
} from "../src/client.js";
import {
  DictationStreamServerError,
  StarlingStream,
  streamWebSocketUrl,
  type StarlingSocketEvent,
  type StarlingStreamEvent,
  type StarlingStreamSocket,
} from "../src/streaming.js";

class FakeSocket implements StarlingStreamSocket {
  readyState = 0;
  readonly sent: Array<string | Uint8Array | ArrayBuffer> = [];
  readonly clientCloses: Array<{ code?: number; reason?: string | undefined }> = [];
  private readonly handlers = new Map<string, Set<(event: StarlingSocketEvent) => void>>();

  addEventListener(type: string, listener: (event: StarlingSocketEvent) => void): void {
    const set = this.handlers.get(type) ?? new Set();
    set.add(listener);
    this.handlers.set(type, set);
  }

  removeEventListener(type: string, listener: (event: StarlingSocketEvent) => void): void {
    this.handlers.get(type)?.delete(listener);
  }

  send(data: string | Uint8Array | ArrayBuffer): void {
    if (this.readyState !== 1) throw new Error("fake socket is not open");
    this.sent.push(data);
  }

  close(code = 1000, reason?: string): void {
    if (this.readyState >= 2) return;
    this.readyState = 3;
    this.clientCloses.push({ code, reason });
    this.emit("close", reason !== undefined ? { code, reason } : { code });
  }

  /** Test doubles for the server side of the connection. */
  serverOpen(): void {
    this.readyState = 1;
    this.emit("open", {});
  }

  serverMessage(data: string): void {
    this.emit("message", { data });
  }

  serverClose(code?: number, reason?: string): void {
    this.readyState = 3;
    this.emit("close", code !== undefined && reason !== undefined ? { code, reason } : {});
  }

  private emit(type: string, event: StarlingSocketEvent): void {
    for (const handler of Array.from(this.handlers.get(type) ?? [])) handler(event);
  }
}

const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

function fakeFactory() {
  const socket = new FakeSocket();

  return { socket, factory: (): FakeSocket => socket };
}

async function openedStream(
  options: Partial<{ connectTimeoutMs: number; responseTimeoutMs: number }> = {},
) {
  const { socket, factory } = fakeFactory();

  const stream = new StarlingStream({
    baseUrl: "http://localhost:8181",
    socket: factory,
    ...options,
  });

  const opening = stream.open();
  await flush();
  socket.serverOpen();
  await opening;

  return { stream, socket };
}

const pcmFrame = (samples: number): Uint8Array => new Uint8Array(samples * 2).fill(1);

describe("streamWebSocketUrl", () => {
  it("maps http(s) bases to ws(s) stream URLs", () => {
    assert.equal(streamWebSocketUrl("http://127.0.0.1:8181"), "ws://127.0.0.1:8181/stream");
    assert.equal(streamWebSocketUrl("https://api.example.com/"), "wss://api.example.com/stream");
    assert.equal(streamWebSocketUrl("http://host:1/base/", "/custom"), "ws://host:1/base/custom");
  });

  it("resolves path-only bases against the browser location", () => {
    const original = globalThis.location;

    const setLocation = (value: { protocol: string; host: string }): void => {
      Object.defineProperty(globalThis, "location", { value, configurable: true });
    };

    setLocation({ protocol: "http:", host: "127.0.0.1:1420" });

    try {
      assert.equal(streamWebSocketUrl("/api"), "ws://127.0.0.1:1420/api/stream");
      setLocation({ protocol: "https:", host: "app.example.com" });
      assert.equal(streamWebSocketUrl("/api"), "wss://app.example.com/api/stream");
    } finally {
      // Node has no location; restore whatever the test started with.
      Object.defineProperty(globalThis, "location", { value: original, configurable: true });
    }
  });

  it("rejects unusable bases", () => {
    assert.throws(() => streamWebSocketUrl(""), TypeError);
    assert.throws(() => streamWebSocketUrl("localhost:8181"), TypeError);
  });
});

describe("StarlingStream", () => {
  it("opens the socket at the stream URL and surfaces validated partials", async () => {
    const { socket, factory } = fakeFactory();
    const stream = new StarlingStream({ baseUrl: "http://localhost:8181/", socket: factory });
    const events: StarlingStreamEvent[] = [];
    stream.onEvent((event) => events.push(event));

    assert.equal(stream.url, "ws://localhost:8181/stream");

    const opening = stream.open();
    await flush();
    socket.serverOpen();
    await opening;

    assert.equal(stream.isOpen, true);
    socket.serverMessage('{"type":"partial","text":"hello","start_s":0,"end_s":1.25}');
    assert.deepEqual(events, [
      { type: "partial", text: "hello", startSeconds: 0, endSeconds: 1.25 },
    ]);
  });

  it("sends PCM16 chunks as binary frames and refuses malformed ones", async () => {
    const { stream, socket } = await openedStream();
    const frame = pcmFrame(4);

    await stream.sendPcm(frame);

    assert.equal(socket.sent.length, 1);

    // The client may re-wrap the view; compare the payload, not the identity.
    const first = socket.sent[0];

    assert.ok(first instanceof Uint8Array);
    assert.deepEqual(first, frame);

    await assert.rejects(stream.sendPcm(new Uint8Array(3)), DictationInputError);
    await assert.rejects(stream.sendPcm(new Uint8Array(0)), DictationInputError);
  });

  it("commits and returns a normalized final transcript", async () => {
    const { stream, socket } = await openedStream();
    await stream.sendPcm(pcmFrame(16_000));

    const committing = stream.commit();
    socket.serverMessage(
      '{"type":"final","text":"hello world","segments":[{"text":"hello","start_s":0,"end_s":1}],"duration_s":1}',
    );
    const transcript = await committing;

    assert.deepEqual(socket.sent[1], '{"type":"commit"}');
    assert.equal(transcript.text, "hello world");
    assert.deepEqual(transcript.segments, [{ text: "hello", startSeconds: 0, endSeconds: 1 }]);
    assert.equal(transcript.durationSeconds, 1);
  });

  it("awaits reset_ack and pong for the reset and ping controls", async () => {
    const { stream, socket } = await openedStream();

    const resetting = stream.reset();
    socket.serverMessage('{"type":"reset_ack"}');

    const pinging = stream.ping();
    socket.serverMessage('{"type":"pong"}');

    await resetting;
    await pinging;
    assert.deepEqual(socket.sent, ['{"type":"reset"}', '{"type":"ping"}']);
  });

  it("marks the live-buffer cap error distinctly with its limit", async () => {
    const { stream, socket } = await openedStream();
    const events: StarlingStreamEvent[] = [];
    stream.onEvent((event) => events.push(event));

    socket.serverMessage(
      '{"type":"error","message":"stream buffer limit reached (60 s live buffer); audio ignored until reset"}',
    );

    assert.equal(events[0]?.type, "error");
    assert.equal(events[0]?.type === "error" && events[0].bufferLimit, true);
    assert.equal(events[0]?.type === "error" && events[0].limitSeconds, 60);

    socket.serverMessage('{"type":"error","message":"server busy"}');
    assert.equal(events[1]?.type === "error" && events[1].bufferLimit, false);
  });

  it("fails a pending commit when the server replies with an error frame", async () => {
    const { stream, socket } = await openedStream();

    const committing = stream.commit();
    socket.serverMessage('{"type":"error","message":"server busy"}');

    await assert.rejects(
      committing,
      (cause) => cause instanceof DictationStreamServerError && cause.message === "server busy",
    );
  });

  it("fails pending commands and reports protocol violations for undecodable messages", async () => {
    const { stream, socket } = await openedStream();
    const events: StarlingStreamEvent[] = [];
    stream.onEvent((event) => events.push(event));

    const committing = stream.commit();
    socket.serverMessage('{"type":"final","text":42}');

    await assert.rejects(committing, /did not match the Starling stream schema|not valid JSON/);
    assert.equal(events[0]?.type, "error");
    assert.match(events[0]?.type === "error" ? events[0].message : "", /protocol violation/);

    socket.serverMessage("not json at all");
    assert.match(events[1]?.type === "error" ? events[1].message : "", /not valid JSON/);
  });

  it("fails waiters with a transport error when the server closes mid-command", async () => {
    const { stream, socket } = await openedStream();
    const events: StarlingStreamEvent[] = [];
    stream.onEvent((event) => events.push(event));

    const committing = stream.commit();
    socket.serverClose(1011, "boom");

    await assert.rejects(
      committing,
      (cause) => cause instanceof DictationTransportError && cause.message.includes("code 1011"),
    );
    assert.deepEqual(events.at(-1), { type: "closed", code: 1011, reason: "boom" });
    assert.equal(stream.isOpen, false);
  });

  it("surfaces connect failures and timeouts without leaking the socket", async () => {
    const { socket, factory } = fakeFactory();
    const refused = new StarlingStream({ baseUrl: "http://localhost:8181", socket: factory });
    const refusing = refused.open();
    await flush();
    socket.serverClose();

    await assert.rejects(refusing, DictationTransportError);
    assert.equal(refused.isOpen, false);

    const { socket: silent, factory: silentFactory } = fakeFactory();

    const slow = new StarlingStream({
      baseUrl: "http://localhost:8181",
      connectTimeoutMs: 10,
      socket: silentFactory,
    });

    await assert.rejects(slow.open(), DictationTimeoutError);
    assert.equal(silent.readyState, 3);
    assert.deepEqual(silent.clientCloses, [{ code: 1000, reason: undefined }]);
  });

  it("times out commands that never get a reply", async () => {
    const { stream } = await openedStream({ responseTimeoutMs: 10 });

    await assert.rejects(stream.ping(), DictationTimeoutError);
  });

  it("refuses commands, sends, and reopening after close", async () => {
    const { stream } = await openedStream();
    stream.close();

    await assert.rejects(stream.commit(), DictationInputError);
    await assert.rejects(stream.sendPcm(pcmFrame(2)), DictationTransportError);
    await assert.rejects(stream.open(), DictationInputError);
  });

  it("treats close as idempotent and usable before the socket opens", async () => {
    const { socket, factory } = fakeFactory();
    const stream = new StarlingStream({ baseUrl: "http://localhost:8181", socket: factory });

    stream.close();
    stream.close();
    assert.deepEqual(socket.clientCloses, []);
  });
});
