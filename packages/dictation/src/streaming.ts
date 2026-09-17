import { Effect, Option, Predicate, Schema } from "effect";

import {
  DictationInputError,
  DictationProtocolError,
  DictationTimeoutError,
  DictationTransportError,
  type TranscriptionResult,
  type TranscriptionSegment,
} from "./client.js";

/**
 * Streaming dictation over the Starling `WS /stream` protocol
 * (docs/native-serving.md). Binary frames carry raw PCM16 16 kHz mono audio;
 * JSON text frames carry partial/final/error/pong/reset_ack messages and the
 * commit/reset/ping controls.
 *
 * The browser WebSocket API cannot set request headers, so streaming assumes
 * the same loopback-or-trusted deployment the native server documents; the
 * batch client keeps its credential support for proxied deployments.
 */

const NonNegativeFinite = Schema.Finite.check(Schema.isGreaterThanOrEqualTo(0));

const WireStreamSegmentSchema = Schema.Struct({
  text: Schema.String,
  start_s: NonNegativeFinite,
  end_s: NonNegativeFinite,
});

const WirePartialSchema = Schema.Struct({
  type: Schema.Literal("partial"),
  text: Schema.String,
  start_s: Schema.optionalKey(NonNegativeFinite),
  end_s: Schema.optionalKey(NonNegativeFinite),
});

const WireFinalSchema = Schema.Struct({
  type: Schema.Literal("final"),
  text: Schema.String,
  segments: Schema.optionalKey(Schema.Array(WireStreamSegmentSchema)),
  duration_s: Schema.optionalKey(NonNegativeFinite),
});

const WireErrorSchema = Schema.Struct({
  type: Schema.Literal("error"),
  message: Schema.String,
});

const WirePongSchema = Schema.Struct({ type: Schema.Literal("pong") });

const WireResetAckSchema = Schema.Struct({ type: Schema.Literal("reset_ack") });

// Two boundary stages: JSON text first (binary frames and malformed text
// fail here), then the Starling stream schema. Splitting them lets the
// protocol error say which contract the frame broke.
const parseJsonFrame = Schema.decodeUnknownOption(Schema.fromJsonString(Schema.Unknown));

const WireServerMessageSchema = Schema.Union([
  WirePartialSchema,
  WireFinalSchema,
  WireErrorSchema,
  WirePongSchema,
  WireResetAckSchema,
]);

/** Named owner contract for the decoded server message union. */
type ServerStreamMessage = Schema.Schema.Type<typeof WireServerMessageSchema>;

const decodeServerMessage = Schema.decodeUnknownOption(WireServerMessageSchema);

/** The server's buffer-cap error text: `stream buffer limit reached (60 s live buffer); ...`. */
const BUFFER_LIMIT_PATTERN = /^stream buffer limit reached \((\d+(?:\.\d+)?) s live buffer\)/;

export interface StarlingStreamPartialEvent {
  readonly type: "partial";
  readonly text: string;
  readonly startSeconds?: number;
  readonly endSeconds?: number;
}

export interface StarlingStreamFinalEvent {
  readonly type: "final";
  readonly transcript: TranscriptionResult;
}

export interface StarlingStreamPongEvent {
  readonly type: "pong";
}

export interface StarlingStreamResetAckEvent {
  readonly type: "reset_ack";
}

export interface StarlingStreamServerErrorEvent {
  readonly type: "error";
  readonly message: string;
  /** True only for the live-buffer cap error, so UI can explain the reset rule. */
  readonly bufferLimit: boolean;
  readonly limitSeconds?: number;
}

/** Emitted locally when the socket closes, whichever side asked for it. */
export interface StarlingStreamClosedEvent {
  readonly type: "closed";
  readonly code?: number;
  readonly reason?: string;
}

export type StarlingStreamEvent =
  | StarlingStreamPartialEvent
  | StarlingStreamFinalEvent
  | StarlingStreamPongEvent
  | StarlingStreamResetAckEvent
  | StarlingStreamServerErrorEvent
  | StarlingStreamClosedEvent;

export class DictationStreamServerError extends Schema.TaggedError<DictationStreamServerError>()(
  "DictationStreamServerError",
  {
    message: Schema.String,
    bufferLimit: Schema.Boolean,
  },
) {
  constructor(message: string, bufferLimit = false) {
    super({ message, bufferLimit });
  }
}

export type DictationStreamError =
  | DictationInputError
  | DictationProtocolError
  | DictationStreamServerError
  | DictationTimeoutError
  | DictationTransportError;

/** The structural slice of `WebSocket` this client uses. Narrowing keeps the
 * client testable with plain fakes while still accepting the DOM object. */
export interface StarlingSocketEvent {
  readonly data?: unknown;
  readonly code?: number;
  readonly reason?: string;
}

export interface StarlingStreamSocket {
  readonly readyState: number;
  send(data: string | Uint8Array<ArrayBuffer> | ArrayBuffer): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: "open", listener: () => void): void;
  addEventListener(type: "message", listener: (event: StarlingSocketEvent) => void): void;
  addEventListener(type: "close", listener: (event: StarlingSocketEvent) => void): void;
  addEventListener(type: "error", listener: () => void): void;
  removeEventListener(type: "open", listener: () => void): void;
  removeEventListener(type: "message", listener: (event: StarlingSocketEvent) => void): void;
  removeEventListener(type: "close", listener: (event: StarlingSocketEvent) => void): void;
  removeEventListener(type: "error", listener: () => void): void;
}

export type StarlingStreamSocketFactory = (url: string) => StarlingStreamSocket;

export interface StarlingStreamOptions {
  readonly baseUrl: string;
  /** WebSocket path on the server; defaults to the native `/stream`. */
  readonly path?: string;
  /** Deadline for the socket to open; 0 disables. Defaults to 10 s. */
  readonly connectTimeoutMs?: number;
  /** Deadline for commit/reset/ping replies; 0 disables. Defaults to 120 s. */
  readonly responseTimeoutMs?: number;
  /** Injectable socket factory, mirroring the fetch injection in StarlingClient. */
  readonly socket?: StarlingStreamSocketFactory;
}

const CONNECT_TIMEOUT_MS = 10_000;

const RESPONSE_TIMEOUT_MS = 120_000;

function defaultSocketFactory(url: string): StarlingStreamSocket {
  return new WebSocket(url);
}

function describeCause(cause: unknown): string {
  if (Predicate.isError(cause)) return cause.message || cause.name;

  if (Predicate.isString(cause)) return cause;

  return String(cause);
}

function withTimeout<A, E>(
  effect: Effect.Effect<A, E>,
  timeoutMs: number,
): Effect.Effect<A, E | DictationTimeoutError> {
  if (timeoutMs === 0) return effect;

  return effect.pipe(
    Effect.timeoutOrElse({
      duration: timeoutMs,
      orElse: () => Effect.fail(new DictationTimeoutError(timeoutMs)),
    }),
  );
}

/**
 * Resolve the WebSocket URL for a streaming connection.
 *
 * Accepts the same base as `StarlingClient`: an `http(s)://host[:port][/prefix]`
 * endpoint (scheme swapped to `ws(s)`) or, in a browser, an absolute path such
 * as the `/api` dev proxy, resolved against the current page.
 */
export function streamWebSocketUrl(baseUrl: string, path = "/stream"): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");

  if (!trimmed) throw new TypeError("baseUrl must not be empty");

  if (/^https?:\/\//i.test(trimmed)) {
    const url = new URL(trimmed);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";

    return `${url.href.replace(/\/+$/, "")}${path}`;
  }

  if (!trimmed.startsWith("/")) {
    throw new TypeError("baseUrl must be an http(s) URL or an absolute path");
  }

  const location = globalThis.location;

  if (!location) {
    throw new TypeError("a path-only baseUrl requires a browser location");
  }

  const scheme = location.protocol === "https:" ? "wss" : "ws";

  return `${scheme}://${location.host}${trimmed}${path}`;
}

interface CommandWaiter {
  readonly accepts: (event: StarlingStreamEvent) => boolean;
  readonly settle: (effect: Effect.Effect<StarlingStreamEvent, DictationStreamError>) => void;
}

interface MutablePartialEvent {
  type: "partial";
  text: string;
  startSeconds?: number;
  endSeconds?: number;
}

interface MutableServerErrorEvent {
  type: "error";
  message: string;
  bufferLimit: boolean;
  limitSeconds?: number;
}

interface MutableClosedEvent {
  type: "closed";
  code?: number;
  reason?: string;
}

interface MutableTranscriptionDraft {
  text: string;
  segments: ReadonlyArray<TranscriptionSegment>;
  durationSeconds?: number;
}

function normalizeFinal(
  wire: Schema.Schema.Type<typeof WireFinalSchema>,
): StarlingStreamFinalEvent {
  const segments: Array<TranscriptionSegment> = [];

  for (const segment of wire.segments ?? []) {
    segments.push(
      Object.freeze({
        text: segment.text,
        startSeconds: segment.start_s,
        endSeconds: segment.end_s,
      }),
    );
  }

  const transcript: MutableTranscriptionDraft = {
    text: wire.text,
    segments: Object.freeze(segments),
  };

  if (wire.duration_s !== undefined) transcript.durationSeconds = wire.duration_s;

  return Object.freeze({ type: "final", transcript: Object.freeze(transcript) });
}

function normalizeError(
  wire: Schema.Schema.Type<typeof WireErrorSchema>,
): StarlingStreamServerErrorEvent {
  const match = BUFFER_LIMIT_PATTERN.exec(wire.message);

  if (!match) {
    return Object.freeze({ type: "error", message: wire.message, bufferLimit: false });
  }

  const limitSeconds = Number(match[1]);

  const event: MutableServerErrorEvent = {
    type: "error",
    message: wire.message,
    bufferLimit: true,
  };

  if (Number.isFinite(limitSeconds)) event.limitSeconds = limitSeconds;

  return Object.freeze(event);
}

/**
 * One streaming recording session against `WS /stream`.
 *
 * A single connection covers one dictation: connect with `open`, push PCM16
 * frames with `sendPcm`, then `commit` for the final transcript. Callers own
 * the failure policy: every method fails with a typed error instead of
 * retrying, so a UI can fall back to the batch upload of its saved WAV.
 */
export class StarlingStream {
  readonly url: string;
  private readonly connectTimeoutMs: number;
  private readonly responseTimeoutMs: number;
  private readonly factory: StarlingStreamSocketFactory;
  private readonly listeners = new Set<(event: StarlingStreamEvent) => void>();
  private readonly waiters: CommandWaiter[] = [];
  private socket: StarlingStreamSocket | undefined;
  private used = false;
  private closeEmitted = false;

  constructor(options: StarlingStreamOptions) {
    this.url = streamWebSocketUrl(options.baseUrl, options.path);
    this.connectTimeoutMs = options.connectTimeoutMs ?? CONNECT_TIMEOUT_MS;
    this.responseTimeoutMs = options.responseTimeoutMs ?? RESPONSE_TIMEOUT_MS;
    this.factory = options.socket ?? defaultSocketFactory;

    if (
      !Number.isFinite(this.connectTimeoutMs) ||
      this.connectTimeoutMs < 0 ||
      !Number.isFinite(this.responseTimeoutMs) ||
      this.responseTimeoutMs < 0
    ) {
      throw new TypeError("streaming timeouts must be finite non-negative numbers");
    }
  }

  get isOpen(): boolean {
    return this.socket?.readyState === 1;
  }

  /** Subscribe to validated server events plus the local `closed` event. */
  onEvent(listener: (event: StarlingStreamEvent) => void): () => void {
    this.listeners.add(listener);

    return () => this.listeners.delete(listener);
  }

  openEffect(): Effect.Effect<void, DictationStreamError> {
    return Effect.fnUntraced(function* (self: StarlingStream) {
      if (self.used) {
        return yield* new DictationInputError(
          "this streaming connection was already opened; create a new StarlingStream per recording",
        );
      }

      const socket = yield* Effect.try({
        try: () => self.attach(self.factory(self.url)),
        catch: (cause) =>
          new DictationTransportError(
            `could not create the streaming socket: ${describeCause(cause)}`,
          ),
      });

      return yield* withTimeout(
        Effect.callback<void, DictationStreamError>((resume) => {
          let connectSettled = false;

          const settleConnect = (effect: Effect.Effect<void, DictationStreamError>): void => {
            if (connectSettled) return;
            connectSettled = true;
            socket.removeEventListener("open", onOpen);
            socket.removeEventListener("error", onError);
            resume(effect);
          };

          const onOpen = (): void => settleConnect(Effect.succeed(undefined));

          const onMessage = (event: StarlingSocketEvent): void => {
            // Boundary parse, stage one: binary frames and non-JSON text.
            const text = parseJsonFrame(event.data);

            if (Option.isNone(text)) {
              self.protocolFailure("the streaming frame was not valid JSON");

              return;
            }

            // Stage two: a frame that parses but is off the Starling schema.
            const decoded = decodeServerMessage(text.value);

            if (Option.isNone(decoded)) {
              self.protocolFailure("the streaming frame did not match the Starling stream schema");

              return;
            }

            self.dispatch(decoded.value);
          };

          const onClose = (event: { readonly code?: number; readonly reason?: string }): void => {
            settleConnect(
              Effect.fail(
                new DictationTransportError(
                  `the streaming server refused the connection${event.code ? ` (code ${event.code})` : ""}`,
                ),
              ),
            );
            socket.removeEventListener("message", onMessage);
            socket.removeEventListener("close", onClose);
            self.handleClose(event.code, event.reason);
          };

          const onError = (): void => {
            // A WebSocket error is always followed by close; settling here
            // gives the connect path its message while onClose finishes teardown.
            settleConnect(
              Effect.fail(
                new DictationTransportError(
                  `could not open the streaming connection to ${self.url}`,
                ),
              ),
            );
          };

          socket.addEventListener("open", onOpen);
          socket.addEventListener("message", onMessage);
          socket.addEventListener("close", onClose);
          socket.addEventListener("error", onError);

          // Fiber interrupted (including by the timeout above): undo the
          // subscription and tear the socket down so nothing dangles. The
          // pending resume is a no-op on an interrupted fiber.
          return Effect.sync(() => {
            socket.removeEventListener("open", onOpen);
            socket.removeEventListener("message", onMessage);
            socket.removeEventListener("close", onClose);
            socket.removeEventListener("error", onError);
            self.close();
          });
        }),
        self.connectTimeoutMs,
      );
    })(this);
  }

  sendPcmEffect(bytes: Uint8Array): Effect.Effect<void, DictationStreamError> {
    return Effect.fnUntraced(function* (self: StarlingStream) {
      if (bytes.byteLength === 0 || bytes.byteLength % 2 !== 0) {
        return yield* new DictationInputError(
          "streaming audio frames must be non-empty PCM16 (an even number of bytes)",
        );
      }

      const socket = self.socket;

      if (!socket || socket.readyState !== 1) {
        return yield* new DictationTransportError("the streaming connection is not open");
      }

      // A view over a non-ArrayBuffer (e.g. SharedArrayBuffer) is not a valid
      // BufferSource; re-wrap so any Uint8Array can be streamed.
      const frame: Uint8Array<ArrayBuffer> =
        bytes.buffer instanceof ArrayBuffer
          ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
          : new Uint8Array(bytes);

      return yield* Effect.try({
        try: () => socket.send(frame),
        catch: (cause) =>
          new DictationTransportError(
            `failed to send a streaming audio frame: ${describeCause(cause)}`,
          ),
      });
    })(this);
  }

  /** Finalize all streamed audio; resolves with the server's final transcript. */
  commitEffect(): Effect.Effect<TranscriptionResult, DictationStreamError> {
    return this.commandEffect("commit", (event) => event.type === "final").pipe(
      Effect.flatMap((event) =>
        event.type === "final"
          ? Effect.succeed(event.transcript)
          : Effect.fail(new DictationProtocolError("unexpected streaming response to commit")),
      ),
    );
  }

  /** Discard the server's buffered audio and re-enable it after a buffer-cap error. */
  resetEffect(): Effect.Effect<void, DictationStreamError> {
    return this.commandEffect("reset", (event) => event.type === "reset_ack").pipe(
      Effect.map(() => undefined),
    );
  }

  pingEffect(): Effect.Effect<void, DictationStreamError> {
    return this.commandEffect("ping", (event) => event.type === "pong").pipe(
      Effect.map(() => undefined),
    );
  }

  /**
   * Close the socket and fail anything still awaiting a reply. Idempotent;
   * safe to call from cleanup paths regardless of connection state.
   */
  close(): void {
    const socket = this.socket;
    this.socket = undefined;

    if (socket) {
      try {
        if (socket.readyState < 2) socket.close(1000);
      } catch {
        /* already closing or closed */
      }
    }

    this.handleClose(1000, undefined);
  }

  open(): Promise<void> {
    return Effect.runPromise(this.openEffect());
  }

  sendPcm(bytes: Uint8Array): Promise<void> {
    return Effect.runPromise(this.sendPcmEffect(bytes));
  }

  commit(): Promise<TranscriptionResult> {
    return Effect.runPromise(this.commitEffect());
  }

  reset(): Promise<void> {
    return Effect.runPromise(this.resetEffect());
  }

  ping(): Promise<void> {
    return Effect.runPromise(this.pingEffect());
  }

  private commandEffect(
    control: "commit" | "reset" | "ping",
    accepts: (event: StarlingStreamEvent) => boolean,
  ): Effect.Effect<StarlingStreamEvent, DictationStreamError> {
    return Effect.fnUntraced(function* (self: StarlingStream) {
      const socket = self.socket;

      if (!socket || socket.readyState !== 1) {
        return yield* new DictationInputError(`${control} requires an open streaming connection`);
      }

      // Register the waiter before sending: the reply can race the send.
      const awaiting = self.awaitEvent(accepts);

      yield* Effect.try({
        try: () => socket.send(`{"type":"${control}"}`),
        catch: (cause) =>
          new DictationTransportError(`failed to send ${control}: ${describeCause(cause)}`),
      });

      return yield* awaiting;
    })(this).pipe((effect) => withTimeout(effect, this.responseTimeoutMs));
  }

  private awaitEvent(
    accepts: (event: StarlingStreamEvent) => boolean,
  ): Effect.Effect<StarlingStreamEvent, DictationStreamError> {
    return Effect.callback<StarlingStreamEvent, DictationStreamError>((resume) => {
      const waiter: CommandWaiter = { accepts, settle: (effect) => resume(effect) };

      this.waiters.push(waiter);

      return Effect.sync(() => {
        const index = this.waiters.indexOf(waiter);

        if (index >= 0) this.waiters.splice(index, 1);
      });
    });
  }

  private attach(socket: StarlingStreamSocket): StarlingStreamSocket {
    this.socket = socket;
    this.used = true;
    this.closeEmitted = false;

    return socket;
  }

  private dispatch(wire: ServerStreamMessage): void {
    if (!this.socket) return;

    if (wire.type === "final") {
      this.emit(normalizeFinal(wire));
    } else if (wire.type === "partial") {
      const partial: MutablePartialEvent = { type: "partial", text: wire.text };

      if (wire.start_s !== undefined) partial.startSeconds = wire.start_s;

      if (wire.end_s !== undefined) partial.endSeconds = wire.end_s;

      this.emit(Object.freeze(partial));
    } else if (wire.type === "error") {
      this.emit(normalizeError(wire));
    } else {
      this.emit(Object.freeze({ type: wire.type }));
    }
  }

  private protocolFailure(detail: string): void {
    const failure = new DictationProtocolError(detail);

    const error: StarlingStreamEvent = Object.freeze({
      type: "error",
      message: `dictation server protocol violation: ${detail}`,
      bufferLimit: false,
    });

    this.failWaiters(failure);

    for (const listener of this.listeners) listener(error);
  }

  private emit(event: StarlingStreamEvent): void {
    if (event.type === "error") {
      this.failWaiters(new DictationStreamServerError(event.message, event.bufferLimit));
    } else {
      for (let index = this.waiters.length - 1; index >= 0; index -= 1) {
        const waiter = this.waiters[index];

        if (waiter?.accepts(event)) {
          this.waiters.splice(index, 1);
          waiter.settle(Effect.succeed(event));
        }
      }
    }

    for (const listener of this.listeners) listener(event);
  }

  private failWaiters(cause: DictationStreamError): void {
    const pending = this.waiters.splice(0, this.waiters.length);

    for (const waiter of pending) waiter.settle(Effect.fail(cause));
  }

  private handleClose(code?: number, reason?: string): void {
    if (this.closeEmitted) return;

    this.closeEmitted = true;
    this.socket = undefined;
    this.failWaiters(
      new DictationTransportError(
        `the streaming connection closed before the server replied${code ? ` (code ${code})` : ""}`,
      ),
    );

    const event: MutableClosedEvent = { type: "closed" };

    if (code !== undefined) event.code = code;

    if (reason !== undefined) event.reason = reason;

    for (const listener of this.listeners) listener(Object.freeze(event));
  }
}
