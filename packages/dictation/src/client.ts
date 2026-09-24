import { Effect, Option, Predicate, Schema } from "effect";

import {
  AudioFormatError,
  PcmAudioSchema,
  prepareWav16k,
  type AudioSource,
  type PreparedWav,
} from "./audio.js";

export type StarlingBackend = "auto" | "python" | "native";

export interface BearerAuth {
  readonly token: string;
  readonly scheme?: string;
  readonly header?: string;
}

interface StringHeaders {
  readonly [name: string]: string;
}

export interface StarlingClientOptions {
  readonly baseUrl: string;
  readonly backend?: StarlingBackend;
  readonly endpoint?: string;
  readonly model?: string;
  readonly auth?: BearerAuth | (() => BearerAuth | Promise<BearerAuth>);
  readonly headers?: StringHeaders;
  readonly timeoutMs?: number;
  readonly maxResponseBytes?: number;
  readonly fetch?: typeof globalThis.fetch;
}

export interface TranscribeEffectOptions {
  readonly requestId?: string | undefined;
}

export interface TranscribeOptions extends TranscribeEffectOptions {
  readonly signal?: AbortSignal | undefined;
}

const NonNegativeFinite = Schema.Finite.check(Schema.isGreaterThanOrEqualTo(0));

// The response-cap limit is strictly positive: 0 bytes would refuse every
// response, so unlike NonNegativeFinite (which lets a timeout be 0 =
// disabled) there is no zero case to admit.
const PositiveFinite = Schema.Finite.check(Schema.isGreaterThan(0));

export const TranscriptionSegmentSchema = Schema.Struct({
  text: Schema.String,
  startSeconds: NonNegativeFinite,
  endSeconds: NonNegativeFinite,
});

export const TranscriptionResultSchema = Schema.Struct({
  text: Schema.String,
  segments: Schema.Array(TranscriptionSegmentSchema),
  durationSeconds: Schema.optionalKey(NonNegativeFinite),
  requestId: Schema.optionalKey(Schema.String),
});

export type TranscriptionSegment = Schema.Schema.Type<typeof TranscriptionSegmentSchema>;

export type TranscriptionResult = Schema.Schema.Type<typeof TranscriptionResultSchema>;

export const ServerHealthSchema = Schema.Struct({
  status: Schema.String,
  phase: Schema.optionalKey(Schema.String),
  model: Schema.optionalKey(Schema.String),
  loaded: Schema.optionalKey(Schema.Boolean),
  busy: Schema.optionalKey(Schema.Boolean),
  queueDepth: Schema.optionalKey(NonNegativeFinite),
});

export type ServerHealth = Schema.Schema.Type<typeof ServerHealthSchema>;

export const TranscriptionResponseSchema = Schema.Struct({
  text: Schema.String,
});

const OpenAiModelResponseSchema = Schema.Struct({ id: Schema.String });

export const OpenAiModelsResponseSchema = Schema.Struct({
  object: Schema.optionalKey(Schema.Literal("list")),
  data: Schema.Array(OpenAiModelResponseSchema),
});

const NestedServerErrorSchema = Schema.Struct({ message: Schema.String });

export const ServerErrorResponseSchema = Schema.Struct({
  detail: Schema.optionalKey(Schema.String),
  error: Schema.optionalKey(Schema.Union([Schema.String, NestedServerErrorSchema])),
  message: Schema.optionalKey(Schema.String),
});

export class DictationError extends Error {
  override readonly name: string = "DictationError";
}

export class DictationHttpError extends Schema.TaggedError<DictationHttpError>()(
  "DictationHttpError",
  {
    message: Schema.String,
    status: Schema.Number,
    statusText: Schema.String,
    responseBody: Schema.String,
  },
) {
  constructor(
    status: number,
    statusText: string,
    responseBody: string,
    message = `dictation server returned HTTP ${status}${statusText ? ` ${statusText}` : ""}`,
  ) {
    super({ message, status, statusText, responseBody });
  }
}

export class DictationProtocolError extends Schema.TaggedError<DictationProtocolError>()(
  "DictationProtocolError",
  { message: Schema.String },
) {
  constructor(message: string) {
    super({ message });
  }
}

/** Response bodies are refused past this size (issue #235). Health and
 * transcription payloads are small JSON documents (a transcript is text),
 * so 10 MiB is orders of magnitude above anything a real backend returns
 * while still bounding what a broken or hostile server can make the
 * client buffer. Override per client with `maxResponseBytes`.
 */
const DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024;

// The error keeps the standard fields-object signature so schema-driven
// instantiation (decode, Effect serialization) constructs it exactly like
// the other tagged errors; `limitBytes: PositiveFinite` is what refuses a
// non-positive cap at the type level. Runtime caps are validated once, at
// StarlingClient construction (`maxResponseBytes`).
export class DictationResponseTooLargeError extends Schema.TaggedError<DictationResponseTooLargeError>()(
  "DictationResponseTooLargeError",
  { message: Schema.String, limitBytes: PositiveFinite },
) {}

/** The cap refusal at every throw site: one message shape, one place. */
function responseTooLarge(limitBytes: number): DictationResponseTooLargeError {
  return new DictationResponseTooLargeError({
    message: `dictation response body exceeded the ${limitBytes} byte limit`,
    limitBytes,
  });
}

export class DictationTimeoutError extends Schema.TaggedError<DictationTimeoutError>()(
  "DictationTimeoutError",
  { message: Schema.String, timeoutMs: NonNegativeFinite },
) {
  constructor(timeoutMs: number) {
    super({ message: `dictation request timed out after ${timeoutMs} ms`, timeoutMs });
  }
}

export class DictationTransportError extends Schema.TaggedError<DictationTransportError>()(
  "DictationTransportError",
  { message: Schema.String },
) {
  constructor(message: string) {
    super({ message });
  }
}

export class DictationInputError extends Schema.TaggedError<DictationInputError>()(
  "DictationInputError",
  { message: Schema.String },
) {
  constructor(message: string) {
    super({ message });
  }
}

export type DictationClientError =
  | AudioFormatError
  | DictationHttpError
  | DictationInputError
  | DictationProtocolError
  | DictationResponseTooLargeError
  | DictationTimeoutError
  | DictationTransportError;

const PreparedWavSchema = Schema.Struct({
  wav: Schema.instanceOf(Uint8Array),
  bytes: Schema.instanceOf(Uint8Array),
  blob: Schema.instanceOf(Blob),
  sampleRate: Schema.Literal(16_000),
  channels: Schema.Literal(1),
  durationMs: NonNegativeFinite,
  durationSeconds: NonNegativeFinite,
});

const isPreparedWav = Schema.is(PreparedWavSchema);

const AudioSourceSchema = Schema.Union([
  PcmAudioSchema,
  Schema.instanceOf(Blob),
  Schema.instanceOf(ArrayBuffer),
  Schema.instanceOf(Uint8Array),
]);

interface BufferedResponse {
  readonly body: string;
  readonly ok: boolean;
  readonly opaqueRedirect: boolean;
  readonly requestId: string | null;
  readonly status: number;
  readonly statusText: string;
}

interface MutableTranscriptionResult {
  text: string;
  segments: ReadonlyArray<TranscriptionSegment>;
  durationSeconds?: number;
  requestId?: string;
}

interface MutableServerHealth {
  status: string;
  phase?: string;
  model?: string;
  loaded?: boolean;
  busy?: boolean;
  queueDepth?: number;
}

function cleanBaseUrl(value: string): string {
  const normalized = value.replace(/\/+$/, "");

  if (!normalized) throw new TypeError("baseUrl must not be empty");

  return normalized;
}

function validRequestId(value: string): boolean {
  return Boolean(value) && !value.startsWith("#") && !/[\r\n]/.test(value);
}

function generatedRequestId(): string {
  const randomUuid = globalThis.crypto?.randomUUID?.();

  if (randomUuid) return randomUuid;

  return `dictation-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function describeCause(cause: unknown): string {
  if (Predicate.isError(cause)) return cause.message || cause.name;

  if (Predicate.isString(cause)) return cause;

  return String(cause);
}

function serverErrorDetail(body: string): string | undefined {
  const decoded = Schema.decodeUnknownOption(Schema.fromJsonString(ServerErrorResponseSchema))(
    body,
  );

  if (Option.isNone(decoded)) return undefined;
  const payload = decoded.value;

  if (payload.detail) return payload.detail;

  if (Predicate.isString(payload.error)) return payload.error;

  if (payload.error) return payload.error.message;

  return payload.message;
}

function ensureOk(response: BufferedResponse): Effect.Effect<BufferedResponse, DictationHttpError> {
  if (response.ok) return Effect.succeed(response);

  return Effect.fail(
    new DictationHttpError(
      response.status,
      response.statusText,
      response.body,
      serverErrorDetail(response.body),
    ),
  );
}

function ensureNotRedirected(
  response: BufferedResponse,
): Effect.Effect<BufferedResponse, DictationHttpError> {
  // Node and Electron surface the real 3xx for redirect: "manual"; a Chromium
  // renderer hands back an opaque redirect instead. Both are refused with the
  // Electron bridge wording so audio and credentials are never re-sent to an
  // origin the user did not configure.
  if (response.opaqueRedirect || (response.status >= 300 && response.status < 400)) {
    const status = response.opaqueRedirect ? undefined : response.status;

    return Effect.fail(
      new DictationHttpError(
        response.status,
        response.statusText,
        response.body,
        `Server redirect blocked${status ? ` (${status})` : ""}. Set the final endpoint explicitly.`,
      ),
    );
  }

  return Effect.succeed(response);
}

function protocolError(label: string): DictationProtocolError {
  return new DictationProtocolError(`dictation server returned invalid ${label} JSON`);
}

/** Reads a response body under a hard cap (issue #235). A declared
 * content-length already past the cap is refused before a byte is read
 * (parity with the Rust client); otherwise the stream is consumed
 * incrementally and cancelled the moment the cap is crossed, so a broken
 * or hostile server cannot balloon memory: the accumulated chunks stay
 * under the limit, and the final join transiently adds one more copy —
 * the peak is bounded at roughly twice the limit. The failure is the
 * distinct `DictationResponseTooLargeError`, never a silent truncation.
 * The decoder runs in streaming mode, so multi-byte UTF-8 sequences
 * split across chunks survive.
 */
async function readBodyCapped(
  response: Response,
  limitBytes: number,
  signal: AbortSignal,
): Promise<string> {
  // The cap bounds what the client holds in memory, so it counts
  // DECOMPRESSED bytes: fetch transparently inflates gzip/br responses,
  // meaning a declared Content-Length (the compressed wire size) can sit
  // under the cap while the decoded body does not — the streaming check
  // below is what enforces the real bound; this pre-check is the fast
  // path for uncompressed bodies. (Deliberate asymmetry with the Rust
  // client, whose transport does not decompress: its pre-check refuses
  // an oversized declaration outright, while here the streaming counter
  // catches a compressed-undersized declaration — both clients bound the
  // decoded stream regardless.) The header parse is deliberately
  // lenient: a malformed value ("12abc") parses to NaN and simply skips
  // the fast path, and runtimes that expose Content-Length as the
  // decompressed size — or strip it — skip it the same way. The
  // streaming counter is the real bound either way.
  const declared = response.headers.get("Content-Length");
  const declaredBytes = declared === null ? Number.NaN : Number(declared);

  // Checked before anything else — including the null-body return — so
  // an oversized declaration is refused no matter what the body looks
  // like. The cancel is fire-and-forget: a rejecting cancel against an
  // already-dead connection must not mask the cap error (and flip
  // downstream retry classification) with a transport failure.
  if (Number.isFinite(declaredBytes) && declaredBytes > limitBytes) {
    void response.body?.cancel().catch(() => {});

    throw responseTooLarge(limitBytes);
  }

  const body = response.body;

  if (body === null) return "";

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let received = 0;
  const chunks: Array<string> = [];
  let completed = false;

  // When the deadline (or a fiber interrupt) aborts the request, a
  // pending read on an injected fetcher's stream may never settle on its
  // own — nothing else is wired to the signal. Cancel from the abort so
  // the connection is released promptly instead of at GC; the effect has
  // already failed by then, so whatever this loop resolves with is
  // discarded.
  signal.addEventListener(
    "abort",
    () => {
      if (!completed) void reader.cancel().catch(() => {});
    },
    { once: true },
  );

  try {
    for (;;) {
      const { done, value } = await reader.read();

      if (done) {
        completed = true;

        return chunks.join("") + decoder.decode();
      }

      // Decompressed bytes — see the pre-check note above.
      received += value.byteLength;

      if (received > limitBytes) throw responseTooLarge(limitBytes);

      // Joined once at the end: `+=` on a growing string is quadratic in
      // the body size, which matters near the 10 MiB default cap.
      chunks.push(decoder.decode(value, { stream: true }));
    }
  } finally {
    // Every non-clean exit — the cap refusal, a failed read, or the
    // abort above racing a slow chunk — releases the reader. The cancel
    // stays fire-and-forget: a rejecting cancel must not mask the cap
    // error with a transport failure. The clean return above is the only
    // path that leaves the stream alone.
    if (!completed) void reader.cancel().catch(() => {});
  }
}

const decodeTranscriptionResponse = Schema.decodeEffect(
  Schema.fromJsonString(TranscriptionResponseSchema),
);

const decodeModelsResponse = Schema.decodeEffect(Schema.fromJsonString(OpenAiModelsResponseSchema));

/**
 * The model sent when a caller configures none. One named, exported
 * constant so the browser client, the Electron bridge, and the settings
 * defaults cannot drift apart.
 */
export const DEFAULT_TRANSCRIPTION_MODEL = "parakeet";

function normalizeTranscription(
  wire: Schema.Schema.Type<typeof TranscriptionResponseSchema>,
  headerRequestId: string | null,
): Effect.Effect<TranscriptionResult, DictationProtocolError> {
  const requestId = headerRequestId ?? undefined;

  // Batch transcription returns text only — the OpenAI-shaped wire schema
  // above decodes no timing data — so segments stay empty here; timed
  // segments are populated exclusively by the streaming path.
  const normalized: MutableTranscriptionResult = {
    text: wire.text,
    segments: Object.freeze([]),
  };

  if (requestId !== undefined) normalized.requestId = requestId;

  return Effect.succeed(Object.freeze(normalized));
}

function normalizeModels(
  wire: Schema.Schema.Type<typeof OpenAiModelsResponseSchema>,
): ServerHealth {
  // busy/queueDepth stay absent rather than fabricated: the OpenAI models
  // route cannot observe them, and an absent optional is the honest
  // encoding of "unknown" for the consumers that check them.
  const health: MutableServerHealth = {
    status: "ok",
    phase: "ready",
  };

  const model = wire.data[0]?.id;

  if (model !== undefined) health.model = model;

  return Object.freeze(health);
}

function prepareSource(
  source: AudioSource | PreparedWav,
): Effect.Effect<PreparedWav, AudioFormatError | DictationInputError> {
  if (isPreparedWav(source)) return Effect.succeed(source);
  const decoded = Schema.decodeUnknownOption(AudioSourceSchema)(source);

  if (Option.isNone(decoded)) {
    return Effect.fail(new DictationInputError("unsupported audio source"));
  }

  return Effect.tryPromise({
    try: () => prepareWav16k(decoded.value),
    catch: (cause) =>
      cause instanceof AudioFormatError ? cause : new DictationInputError(describeCause(cause)),
  });
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

export class StarlingClient {
  readonly backend: StarlingBackend;
  private readonly auth: StarlingClientOptions["auth"] | undefined;
  private readonly baseUrl: string;
  private readonly endpoint: string;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly headers: StringHeaders;
  private readonly maxResponseBytes: number;
  private readonly model: string;
  private readonly timeoutMs: number;

  constructor(options: StarlingClientOptions) {
    this.baseUrl = cleanBaseUrl(options.baseUrl);
    this.backend = options.backend ?? "auto";
    this.endpoint = options.endpoint ?? "/v1/audio/transcriptions";
    this.model = (options.model ?? DEFAULT_TRANSCRIPTION_MODEL).trim();
    this.auth = options.auth;
    this.headers = options.headers ?? {};
    this.timeoutMs = options.timeoutMs ?? 120_000;
    this.maxResponseBytes = options.maxResponseBytes ?? DEFAULT_MAX_RESPONSE_BYTES;

    if (!Number.isFinite(this.timeoutMs) || this.timeoutMs < 0) {
      throw new TypeError("timeoutMs must be a finite non-negative number");
    }

    if (!Number.isFinite(this.maxResponseBytes) || this.maxResponseBytes <= 0) {
      throw new TypeError("maxResponseBytes must be a finite positive number");
    }

    const fetcher = options.fetch ?? globalThis.fetch;

    if (!fetcher) throw new TypeError("a Fetch API implementation is required");
    this.fetcher = fetcher.bind(globalThis);
  }

  transcribeEffect(
    source: AudioSource | PreparedWav,
    options: TranscribeEffectOptions = {},
  ): Effect.Effect<TranscriptionResult, DictationClientError> {
    const prepare = prepareSource(source);

    return Effect.fnUntraced(function* (client: StarlingClient) {
      if (!client.model) {
        return yield* new DictationInputError("Enter the model name served by the backend");
      }

      const prepared = yield* prepare;
      const requestId = options.requestId ?? generatedRequestId();

      if (!validRequestId(requestId)) {
        return yield* new DictationInputError(
          "requestId must be non-empty, contain no newlines, and not start with #",
        );
      }

      // The prepared Blob is appended by reference (issue #235): the
      // client never copies the WAV into a second buffer — the platform
      // serializes the multipart body incrementally while sending.
      const form = new FormData();
      form.append("file", prepared.blob, "recording.wav");

      form.append("model", client.model);
      form.append("response_format", "json");

      const headers = yield* client.requestHeadersEffect({ "X-Request-Id": requestId });
      headers.delete("Content-Type");

      const response = yield* client.requestEffect(`${client.baseUrl}${client.endpoint}`, {
        method: "POST",
        headers,
        body: form,
      });

      const wire = yield* decodeTranscriptionResponse(response.body).pipe(
        Effect.mapError(() => protocolError("transcription response")),
      );

      return yield* normalizeTranscription(wire, response.requestId);
    })(this);
  }

  transcribe(
    source: AudioSource | PreparedWav,
    options: TranscribeOptions = {},
  ): Promise<TranscriptionResult> {
    return Effect.runPromise(
      this.transcribeEffect(source, { requestId: options.requestId }),
      options.signal ? { signal: options.signal } : undefined,
    );
  }

  healthEffect(): Effect.Effect<ServerHealth, DictationClientError> {
    return Effect.fnUntraced(function* (client: StarlingClient) {
      const headers = yield* client.requestHeadersEffect();

      const response = yield* client.requestEffect(`${client.baseUrl}/v1/models`, {
        method: "GET",
        headers,
      });

      const wire = yield* decodeModelsResponse(response.body).pipe(
        Effect.mapError(() => protocolError("models response")),
      );

      return normalizeModels(wire);
    })(this);
  }

  health(signal?: AbortSignal): Promise<ServerHealth> {
    return Effect.runPromise(this.healthEffect(), signal ? { signal } : undefined);
  }

  cancelEffect(requestId: string): Effect.Effect<void, DictationClientError> {
    if (!validRequestId(requestId)) {
      return Effect.fail(
        new DictationInputError(
          "requestId must be non-empty, contain no newlines, and not start with #",
        ),
      );
    }

    return Effect.fnUntraced(function* (client: StarlingClient) {
      const headers = yield* client.requestHeadersEffect();
      // Derived from the same configurable endpoint transcribeEffect uses,
      // so a client mounted under a sub-path cancels against the right
      // route instead of silently missing the queued request.
      yield* client.requestEffect(
        `${client.baseUrl}${client.endpoint}/${encodeURIComponent(requestId)}`,
        {
          method: "DELETE",
          headers,
        },
      );
    })(this);
  }

  cancel(requestId: string, signal?: AbortSignal): Promise<void> {
    return Effect.runPromise(this.cancelEffect(requestId), signal ? { signal } : undefined);
  }

  private requestHeadersEffect(
    extra: StringHeaders = {},
  ): Effect.Effect<Headers, DictationTransportError> {
    return Effect.tryPromise({
      try: async () => {
        const headers = new Headers(this.headers);

        for (const [name, value] of Object.entries(extra)) headers.set(name, value);
        const configured = Predicate.isFunction(this.auth) ? await this.auth() : this.auth;

        if (configured) {
          const scheme = configured.scheme ?? "Bearer";
          const value = scheme ? `${scheme} ${configured.token}` : configured.token;
          headers.set(configured.header ?? "Authorization", value);
        }

        return headers;
      },
      catch: (cause) => new DictationTransportError(describeCause(cause)),
    });
  }

  private requestEffect(
    url: string,
    init: RequestInit,
  ): Effect.Effect<BufferedResponse, DictationClientError> {
    const request = Effect.tryPromise({
      try: async (signal) => {
        // Redirects are blocked with the same policy as the Electron bridge
        // and the iOS client: audio and credentials must never silently
        // follow a server redirect to an origin the user did not configure.
        const response = await this.fetcher(url, { ...init, signal, redirect: "manual" });
        const body = await readBodyCapped(response, this.maxResponseBytes, signal);

        return {
          body,
          ok: response.ok,
          opaqueRedirect: response.type === "opaqueredirect",
          requestId: response.headers.get("X-Request-Id"),
          status: response.status,
          statusText: response.statusText,
        } satisfies BufferedResponse;
      },
      catch: (cause) =>
        // The size cap is a first-class failure, not a transport fault:
        // keep it distinct on its way out of the promise boundary.
        cause instanceof DictationResponseTooLargeError
          ? cause
          : new DictationTransportError(describeCause(cause)),
    }).pipe(Effect.flatMap(ensureNotRedirected), Effect.flatMap(ensureOk));

    return withTimeout(request, this.timeoutMs);
  }
}
