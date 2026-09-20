import { Effect, Option, Predicate, Schema } from "effect";

import {
  AudioFormatError,
  PcmAudioSchema,
  prepareWav16k,
  type AudioSource,
  type PreparedWav,
} from "./audio.js";

export type StarlingBackend = "auto" | "python" | "native";

export type TranscriptionProtocol = "starling" | "openai";

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
  readonly protocol?: TranscriptionProtocol;
  readonly endpoint?: string;
  readonly model?: string;
  readonly auth?: BearerAuth | (() => BearerAuth | Promise<BearerAuth>);
  readonly headers?: StringHeaders;
  readonly timeoutMs?: number;
  readonly fetch?: typeof globalThis.fetch;
}

export interface TranscribeEffectOptions {
  readonly requestId?: string | undefined;
}

export interface TranscribeOptions extends TranscribeEffectOptions {
  readonly signal?: AbortSignal | undefined;
}

const NonNegativeFinite = Schema.Finite.check(Schema.isGreaterThanOrEqualTo(0));

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

const TranscriptionSegmentResponseSchema = Schema.Struct({
  text: Schema.String,
  start_s: Schema.optionalKey(NonNegativeFinite),
  end_s: Schema.optionalKey(NonNegativeFinite),
  start: Schema.optionalKey(NonNegativeFinite),
  end: Schema.optionalKey(NonNegativeFinite),
});

export const TranscriptionResponseSchema = Schema.Struct({
  text: Schema.String,
  segments: Schema.optionalKey(Schema.Array(TranscriptionSegmentResponseSchema)),
  duration_s: Schema.optionalKey(NonNegativeFinite),
  duration: Schema.optionalKey(NonNegativeFinite),
  request_id: Schema.optionalKey(Schema.String),
});

export const ServerHealthResponseSchema = Schema.Struct({
  status: Schema.String,
  phase: Schema.optionalKey(Schema.String),
  model: Schema.optionalKey(Schema.String),
  loaded: Schema.optionalKey(Schema.Boolean),
  busy: Schema.optionalKey(Schema.Boolean),
  queue_depth: Schema.optionalKey(NonNegativeFinite),
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

const decodeTranscriptionResponse = Schema.decodeEffect(
  Schema.fromJsonString(TranscriptionResponseSchema),
);

const decodeHealthResponse = Schema.decodeEffect(Schema.fromJsonString(ServerHealthResponseSchema));

const decodeModelsResponse = Schema.decodeEffect(Schema.fromJsonString(OpenAiModelsResponseSchema));

function normalizeTranscription(
  wire: Schema.Schema.Type<typeof TranscriptionResponseSchema>,
  headerRequestId: string | null,
): Effect.Effect<TranscriptionResult, DictationProtocolError> {
  const segments: Array<TranscriptionSegment> = [];

  for (const segment of wire.segments ?? []) {
    const start = segment.start_s ?? segment.start;
    const end = segment.end_s ?? segment.end;

    if (start === undefined || end === undefined || end < start) {
      return Effect.fail(
        new DictationProtocolError("transcription segment has missing or invalid timestamps"),
      );
    }

    segments.push(Object.freeze({ text: segment.text, startSeconds: start, endSeconds: end }));
  }

  const durationSeconds = wire.duration_s ?? wire.duration;
  const responseRequestId = wire.request_id || undefined;
  const requestId = responseRequestId ?? headerRequestId ?? undefined;

  const normalized: MutableTranscriptionResult = {
    text: wire.text,
    segments: Object.freeze(segments),
  };

  if (durationSeconds !== undefined) normalized.durationSeconds = durationSeconds;

  if (requestId !== undefined) normalized.requestId = requestId;

  return Effect.succeed(Object.freeze(normalized));
}

function normalizeHealth(
  wire: Schema.Schema.Type<typeof ServerHealthResponseSchema>,
): ServerHealth {
  const health: MutableServerHealth = {
    status: wire.status,
  };

  if (wire.phase !== undefined) health.phase = wire.phase;

  if (wire.model !== undefined) health.model = wire.model;

  if (wire.loaded !== undefined) health.loaded = wire.loaded;

  if (wire.busy !== undefined) health.busy = wire.busy;

  if (wire.queue_depth !== undefined) health.queueDepth = wire.queue_depth;

  return Object.freeze(health);
}

function normalizeModels(
  wire: Schema.Schema.Type<typeof OpenAiModelsResponseSchema>,
): ServerHealth {
  const health: MutableServerHealth = {
    status: "ok",
    phase: "ready",
    busy: false,
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
  readonly protocol: TranscriptionProtocol;
  private readonly auth: StarlingClientOptions["auth"] | undefined;
  private readonly baseUrl: string;
  private readonly endpoint: string;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly headers: StringHeaders;
  private readonly model: string;
  private readonly timeoutMs: number;

  constructor(options: StarlingClientOptions) {
    this.baseUrl = cleanBaseUrl(options.baseUrl);
    this.backend = options.backend ?? "auto";
    this.protocol = options.protocol ?? "starling";
    this.endpoint =
      options.endpoint ?? (this.protocol === "openai" ? "/v1/audio/transcriptions" : "/inference");
    this.model = options.model?.trim() ?? "";
    this.auth = options.auth;
    this.headers = options.headers ?? {};
    this.timeoutMs = options.timeoutMs ?? 120_000;

    if (!Number.isFinite(this.timeoutMs) || this.timeoutMs < 0) {
      throw new TypeError("timeoutMs must be a finite non-negative number");
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
      if (client.protocol === "openai" && !client.model) {
        return yield* new DictationInputError("Enter the model name served by the backend");
      }

      const prepared = yield* prepare;
      const requestId = options.requestId ?? generatedRequestId();

      if (!validRequestId(requestId)) {
        return yield* new DictationInputError(
          "requestId must be non-empty, contain no newlines, and not start with #",
        );
      }

      const form = new FormData();
      form.append("file", prepared.blob, "recording.wav");

      if (client.protocol === "openai") {
        form.append("model", client.model);
        form.append("response_format", "json");
      }

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
      const path = client.protocol === "openai" ? "/v1/models" : "/health";

      const response = yield* client.requestEffect(`${client.baseUrl}${path}`, {
        method: "GET",
        headers,
      });

      if (client.protocol === "openai") {
        const wire = yield* decodeModelsResponse(response.body).pipe(
          Effect.mapError(() => protocolError("models response")),
        );

        return normalizeModels(wire);
      }

      const wire = yield* decodeHealthResponse(response.body).pipe(
        Effect.mapError(() => protocolError("health response")),
      );

      return normalizeHealth(wire);
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
      yield* client.requestEffect(`${client.baseUrl}/inference/${encodeURIComponent(requestId)}`, {
        method: "DELETE",
        headers,
      });
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
        const body = await response.text();

        return {
          body,
          ok: response.ok,
          opaqueRedirect: response.type === "opaqueredirect",
          requestId: response.headers.get("X-Request-Id"),
          status: response.status,
          statusText: response.statusText,
        } satisfies BufferedResponse;
      },
      catch: (cause) => new DictationTransportError(describeCause(cause)),
    }).pipe(Effect.flatMap(ensureNotRedirected), Effect.flatMap(ensureOk));

    return withTimeout(request, this.timeoutMs);
  }
}
