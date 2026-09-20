import {
  StarlingStream,
  type StarlingStreamEvent,
  type TranscriptionResult,
} from "@starling/dictation";
import type {
  StreamCloseInput,
  StreamCommandInput,
  StreamCommandResult,
  StreamEventMessage,
  StreamOpenInput,
  StreamSendInput,
} from "../electron/ipc.js";
import type { StreamingTransport } from "./streamingDictation";

/**
 * The live-dictation transport choice (B01).
 *
 * - Browser preview (or an older preload without the stream channels): the
 *   renderer opens `StarlingStream` directly, which the static CSP permits
 *   for loopback endpoints and the page-origin `/api` ws proxy.
 * - Packaged app: the transport runs over the main process's native bridge,
 *   so user-configured LAN `ws://` and `wss://` endpoints work without
 *   widening the renderer's CSP by a single token.
 *
 * Both implement the same `StreamingTransport`, so the journal-then-send
 * pipeline in `StreamingDictation` is untouched by the choice.
 */

/** The stream channels this transport uses — a structural slice of the bridge. */
export interface DesktopStreamBridge {
  streamOpen(input: StreamOpenInput): Promise<{ streamId: number }>;
  streamSend(input: StreamSendInput): Promise<void>;
  streamCommand(input: StreamCommandInput): Promise<StreamCommandResult>;
  streamClose(input: StreamCloseInput): void;
  onStreamEvent(listener: (message: StreamEventMessage) => void): () => void;
}

/**
 * One streamed take over the desktop bridge. Frames and commands cross as
 * IPC invokes (kept in order by the pipeline that awaits them); server
 * events flow back tagged with the stream id this take was assigned.
 */
export class BridgeStreamTransport implements StreamingTransport {
  private readonly listeners = new Set<(event: StarlingStreamEvent) => void>();
  private readonly incoming: StreamEventMessage[] = [];
  private readonly stopEvents: () => void;
  private streamId: number | undefined;
  private opened = false;

  constructor(
    private readonly bridge: DesktopStreamBridge,
    private readonly endpoint: string,
  ) {
    // Subscribed from the constructor so no event can slip between the
    // open() call and the subscription; until the id is assigned, messages
    // are queued and filtered when they drain.
    this.stopEvents = bridge.onStreamEvent((message) => {
      if (this.streamId === undefined) {
        this.incoming.push(message);

        return;
      }

      if (message.streamId === this.streamId) this.deliver(message.event);
    });
  }

  get isOpen(): boolean {
    return this.opened;
  }

  onEvent(listener: (event: StarlingStreamEvent) => void): () => void {
    this.listeners.add(listener);

    return () => this.listeners.delete(listener);
  }

  async open(): Promise<void> {
    if (this.streamId !== undefined) throw new Error("This streaming take was already opened.");

    const { streamId } = await this.bridge.streamOpen({ endpoint: this.endpoint });

    this.streamId = streamId;
    this.opened = true;

    const pending = this.incoming.splice(0, this.incoming.length);

    for (const message of pending) {
      if (message.streamId === streamId) this.deliver(message.event);
    }
  }

  async sendPcm(bytes: Uint8Array): Promise<void> {
    // A fresh copy, so IPC's structured clone never sees a view into a pooled
    // or shared buffer the recorder still writes to.
    await this.bridge.streamSend({
      streamId: this.requireOpen("send audio"),
      audio: bytes.slice().buffer,
    });
  }

  async commit(): Promise<TranscriptionResult> {
    const { transcript } = await this.bridge.streamCommand({
      streamId: this.requireOpen("commit"),
      command: "commit",
    });

    if (!transcript) throw new Error("The streaming bridge returned no final transcript.");

    return transcript;
  }

  close(): void {
    this.opened = false;

    const streamId = this.streamId;
    this.streamId = undefined;
    this.incoming.length = 0;
    this.stopEvents();

    if (streamId !== undefined) this.bridge.streamClose({ streamId });
  }

  private deliver(event: StarlingStreamEvent): void {
    if (event.type === "closed") this.opened = false;

    // Snapshot: a listener may unsubscribe — itself or another — while the
    // event is still dispatching, and the copy keeps the dispatch stable.
    for (const listener of Array.from(this.listeners)) listener(event);
  }

  private requireOpen(action: string): number {
    if (this.streamId === undefined)
      throw new Error(`Cannot ${action}: the streaming take is not open.`);

    return this.streamId;
  }
}

/**
 * Pick the transport for one take: the native bridge when its channels are
 * present, the renderer's direct WebSocket otherwise. Throws exactly where
 * `new StarlingStream` throws, so the caller's fallback is unchanged.
 */
export function createStreamingTransport(endpoint: string): StreamingTransport {
  const open = window.starlingDesktop?.streamOpen;
  const send = window.starlingDesktop?.streamSend;
  const command = window.starlingDesktop?.streamCommand;
  const closeStream = window.starlingDesktop?.streamClose;
  const onStreamEvent = window.starlingDesktop?.onStreamEvent;

  // All five channels, or none: a half-present bridge means an older preload,
  // and the renderer socket the CSP permits is the honest fallback.
  if (open && send && command && closeStream && onStreamEvent) {
    return new BridgeStreamTransport(
      {
        streamOpen: open,
        streamSend: send,
        streamCommand: command,
        streamClose: closeStream,
        onStreamEvent,
      },
      endpoint,
    );
  }

  return new StarlingStream({ baseUrl: endpoint });
}
