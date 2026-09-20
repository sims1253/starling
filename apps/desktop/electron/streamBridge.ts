import {
  StarlingStream,
  type StarlingStreamSocket,
  type StarlingStreamSocketFactory,
} from "@starling/dictation";
import type { StreamCommandResult, StreamEventMessage, StreamOpenInput } from "./ipc.js";

/**
 * The native half of the streaming transport bridge (B01): the renderer's
 * static CSP cannot name user-configured LAN `ws://` or `wss://` endpoints,
 * so the packaged app opens its live-dictation sockets here in the main
 * process and relays frames and events over IPC. The protocol, validation,
 * and timeouts are the shared `StarlingStream` — this bridge adds only
 * multiplexing (a stream id per take) and the socket factory, so the
 * renderer-side and bridge-side transports behave identically.
 */

/** The structural slice of a Node (undici) WebSocket the adapter needs. */
export interface NodeWebSocket {
  readonly readyState: number;
  send(data: string | Uint8Array<ArrayBuffer> | ArrayBuffer): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: string, listener: EventListener): void;
  removeEventListener(type: string, listener: EventListener): void;
}

/** Adapt a Node WebSocket to the socket shape `StarlingStream` consumes. */
export function adaptNodeSocket(socket: NodeWebSocket): StarlingStreamSocket {
  const forward = (listener: unknown): EventListener => listener as EventListener;

  return {
    get readyState() {
      return socket.readyState;
    },
    send: (data) => socket.send(data),
    close: (code, reason) => socket.close(code, reason),
    addEventListener: (type, listener) => socket.addEventListener(type, forward(listener)),
    removeEventListener: (type, listener) => socket.removeEventListener(type, forward(listener)),
  };
}

/** Socket factory over the main process's global (undici) WebSocket. */
export function nodeSocketFactory(url: string): StarlingStreamSocket {
  return adaptNodeSocket(new WebSocket(url));
}

/** Where a stream's events go: the renderer WebContents that opened it. */
export interface StreamSink {
  send(message: StreamEventMessage): void;

  /** invoked once when the receiving side is gone, so the socket is closed */
  onceDestroyed(cleanup: () => void): void;
}

interface LiveStream {
  readonly stream: StarlingStream;
  readonly stopEvents: () => void;
}

/**
 * Multiplexes streaming takes over one IPC boundary. Every method rejects
 * with the underlying transport error; stream ids are handed out in open
 * order and never reused, so a stale renderer reply cannot address a newer
 * take's socket.
 */
export class StreamBridge {
  private readonly streams = new Map<number, LiveStream>();
  private readonly socketFactory: StarlingStreamSocketFactory;
  private nextStreamId = 1;

  constructor(socketFactory: StarlingStreamSocketFactory = nodeSocketFactory) {
    this.socketFactory = socketFactory;
  }

  /**
   * Validate the endpoint and open one streaming take. The endpoint follows
   * the same contract as the batch channels — an absolute `http(s)://` URL
   * without embedded credentials — and the ws/wss derivation happens inside
   * `StarlingStream`, never on the renderer's word.
   */
  async open(input: StreamOpenInput, sink: StreamSink): Promise<{ streamId: number }> {
    const endpoint = normalizeEndpoint(input.endpoint);
    const streamId = this.nextStreamId;
    this.nextStreamId += 1;

    const stream = new StarlingStream({
      baseUrl: endpoint,
      socket: this.socketFactory,
      connectTimeoutMs: input.connectTimeoutMs,
      responseTimeoutMs: input.responseTimeoutMs,
    });

    const stopEvents = stream.onEvent((event) => sink.send({ streamId, event }));
    const live: LiveStream = { stream, stopEvents };
    this.streams.set(streamId, live);

    sink.onceDestroyed(() => this.dispose(streamId));

    try {
      await stream.open();

      return { streamId };
    } catch (cause) {
      this.dispose(streamId);

      throw cause;
    }
  }

  /** One PCM16 frame from the journal-then-send pipeline. */
  send(input: { streamId: number; audio: ArrayBuffer }): Promise<void> {
    return this.live(input.streamId).sendPcm(new Uint8Array(input.audio));
  }

  /** Await a commit/reset/ping reply; only commit carries a transcript. */
  async command(input: { streamId: number; command: "commit" | "reset" | "ping" }): Promise<StreamCommandResult> {
    const live = this.live(input.streamId);

    if (input.command === "commit") return { transcript: await live.commit() };

    if (input.command === "reset") {
      await live.reset();

      return {};
    }

    await live.ping();

    return {};
  }

  /** Drop the socket and its event forwarding; idempotent per stream id. */
  close(streamId: number): void {
    this.dispose(streamId);
  }

  /** Close every open stream, for teardown of the whole bridge. */
  closeAll(): void {
    for (const streamId of [...this.streams.keys()]) this.dispose(streamId);
  }

  private live(streamId: number): StarlingStream {
    const live = this.streams.get(streamId);

    if (!live) throw new Error("That streaming take is no longer open.");

    return live.stream;
  }

  private dispose(streamId: number): void {
    const live = this.streams.get(streamId);

    if (!live) return;

    this.streams.delete(streamId);
    live.stopEvents();
    live.stream.close();
  }
}

/**
 * The endpoint contract shared with the batch channels in main.ts: an
 * absolute http(s) URL, no credentials, and one trailing slash dropped —
 * the shape `StarlingStream` derives its `ws(s)://…/stream` URL from.
 */
function normalizeEndpoint(endpoint: string): string {
  const url = new URL(endpoint);

  if (url.protocol !== "http:" && url.protocol !== "https:")
    throw new Error("Server endpoint must use http or https.");

  if (url.username || url.password)
    throw new Error("Put credentials in a trusted proxy, not the endpoint URL.");

  return url.href.replace(/\/$/, "");
}
