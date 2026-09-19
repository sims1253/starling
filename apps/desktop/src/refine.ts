import { Effect, Option, Predicate, Schema } from "effect";
import { ServerErrorResponseSchema } from "@starling/dictation";

/**
 * Optional transcript refinement: an explicit, per-take request to an
 * OpenAI-compatible chat-completions endpoint (a local Ollama, llama.cpp, or
 * LM Studio server, or a hosted API). The result is stored beside the raw
 * transcript, never in place of it — the raw transcript stays the primary
 * record, so refinement failures leave the session exactly as it was.
 */

export const DEFAULT_REFINEMENT_TIMEOUT_MS = 180_000;

/**
 * The built-in instruction used when the user has not written one. It asks for
 * the lightest honest cleanup — punctuation, capitalization, obvious
 * misrecognitions — and pins everything that matters: wording, meaning,
 * language, and order.
 */
export const REFINEMENT_DEFAULT_INSTRUCTION = [
  "You are a transcription editor.",
  "Clean up the raw speech-to-text transcript the user sends:",
  "fix punctuation, capitalization, and obvious misrecognitions.",
  "Preserve the original wording, meaning, language, and order;",
  "never add, omit, summarize, or translate content.",
  "Reply with the refined transcript only — no preamble, explanation, or quotation marks.",
].join(" ");

/** One chat-completions turn as sent on the wire. */
export interface RefinementMessage {
  /** "assistant" is accepted so a future caller can inject prior-turn context. */
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

export interface RefinementSettings {
  /**
   * Base URL of an OpenAI-compatible endpoint, including any version path the
   * server needs — for example http://127.0.0.1:11434/v1 for Ollama or
   * https://api.openai.com/v1 for the hosted API. Trailing slashes are
   * stripped before `/chat/completions` is joined.
   */
  readonly baseUrl: string;
  readonly model: string;
  /** Sent as Bearer auth only when present; local servers need none. */
  readonly apiKey?: string;
  /** Empty or absent falls back to REFINEMENT_DEFAULT_INSTRUCTION. */
  readonly instruction?: string;
}

export interface RefineEffectOptions {
  readonly fetchImpl?: typeof globalThis.fetch | undefined;
  readonly signal?: AbortSignal | undefined;
  readonly timeoutMs?: number | undefined;
}

interface ChatCompletionRequest {
  readonly model: string;
  readonly messages: readonly RefinementMessage[];
  readonly stream: false;
}

interface BufferedCompletion {
  readonly body: string;
  readonly ok: boolean;
  readonly opaqueRedirect: boolean;
  readonly status: number;
  readonly statusText: string;
}

const NonNegativeFinite = Schema.Finite.pipe(Schema.check(Schema.isGreaterThanOrEqualTo(0)));

/**
 * Lenient about extra fields; only the refined text is load-bearing. The
 * non-empty guarantee is enforced after decoding so an empty completion is
 * reported as its own protocol failure, not a parse error.
 */
const ChatCompletionResponseSchema = Schema.Struct({
  choices: Schema.Array(Schema.Struct({ message: Schema.Struct({ content: Schema.String }) })),
});

export class RefinementHttpError extends Schema.TaggedError<RefinementHttpError>()(
  "RefinementHttpError",
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
    message = `The refinement server returned HTTP ${status}${statusText ? ` ${statusText}` : ""}.`,
  ) {
    super({ message, status, statusText, responseBody });
  }
}

export class RefinementProtocolError extends Schema.TaggedError<RefinementProtocolError>()(
  "RefinementProtocolError",
  {
    message: Schema.String,
  },
) {
  constructor(message: string) {
    super({ message });
  }
}

export class RefinementTimeoutError extends Schema.TaggedError<RefinementTimeoutError>()(
  "RefinementTimeoutError",
  {
    message: Schema.String,
    timeoutMs: NonNegativeFinite,
  },
) {
  constructor(timeoutMs: number) {
    super({ message: `The refinement request timed out after ${timeoutMs} ms.`, timeoutMs });
  }
}

export class RefinementTransportError extends Schema.TaggedError<RefinementTransportError>()(
  "RefinementTransportError",
  {
    message: Schema.String,
  },
) {
  constructor(message: string) {
    super({ message });
  }
}

export class RefinementInputError extends Schema.TaggedError<RefinementInputError>()(
  "RefinementInputError",
  {
    message: Schema.String,
  },
) {
  constructor(message: string) {
    super({ message });
  }
}

export type RefinementError =
  | RefinementHttpError
  | RefinementInputError
  | RefinementProtocolError
  | RefinementTimeoutError
  | RefinementTransportError;

/**
 * Chat-completions messages for one refinement request: the instruction as
 * the system turn, the raw transcript as the user turn. The pair is rebuilt
 * from the raw transcript every call; a future caller can extend the array
 * with prior assistant/user turns for iterative refinement.
 */
export function buildRefinementMessages(
  rawTranscript: string,
  settings: RefinementSettings,
): readonly RefinementMessage[] {
  const instruction = settings.instruction?.trim();

  return Object.freeze([
    Object.freeze({
      role: "system",
      content: instruction || REFINEMENT_DEFAULT_INSTRUCTION,
    }) satisfies RefinementMessage,
    Object.freeze({ role: "user", content: rawTranscript }) satisfies RefinementMessage,
  ]);
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

function ensureOk(
  response: BufferedCompletion,
): Effect.Effect<BufferedCompletion, RefinementHttpError> {
  if (response.ok) return Effect.succeed(response);

  return Effect.fail(
    new RefinementHttpError(
      response.status,
      response.statusText,
      response.body,
      serverErrorDetail(response.body),
    ),
  );
}

function ensureNotRedirected(
  response: BufferedCompletion,
): Effect.Effect<BufferedCompletion, RefinementHttpError> {
  // Node and Electron surface the real 3xx for redirect: "manual"; a Chromium
  // renderer hands back an opaque redirect instead. Both are refused with the
  // same wording as the dictation client so the transcript and any API key
  // are never re-sent to an origin the user did not configure.
  if (response.opaqueRedirect || (response.status >= 300 && response.status < 400)) {
    const status = response.opaqueRedirect ? undefined : response.status;

    return Effect.fail(
      new RefinementHttpError(
        response.status,
        response.statusText,
        response.body,
        `Server redirect blocked${status ? ` (${status})` : ""}. Set the final endpoint explicitly.`,
      ),
    );
  }

  return Effect.succeed(response);
}

function withTimeout<A, E>(
  effect: Effect.Effect<A, E>,
  timeoutMs: number,
): Effect.Effect<A, E | RefinementTimeoutError> {
  // 0 disables the deadline, matching the dictation client's convention.
  if (timeoutMs === 0) return effect;

  return effect.pipe(
    Effect.timeoutOrElse({
      duration: timeoutMs,
      orElse: () => Effect.fail(new RefinementTimeoutError(timeoutMs)),
    }),
  );
}

function withExternalAbort<A, E>(
  effect: Effect.Effect<A, E>,
  signal: AbortSignal | undefined,
): Effect.Effect<A, E | RefinementTransportError> {
  if (!signal) return effect;

  // The listener race makes an externally cancelled request fail like any
  // other: the losing side is interrupted, which aborts the in-flight fetch
  // through its fiber signal. A resume after the request already settled is
  // a no-op, so the once-listener never produces a late failure.
  const cancelled = Effect.callback<never, RefinementTransportError>((resume) => {
    const abort = () =>
      resume(Effect.fail(new RefinementTransportError("The refinement request was cancelled.")));

    if (signal.aborted) {
      abort();

      return;
    }

    signal.addEventListener("abort", abort, { once: true });
  });

  return Effect.raceFirst(effect, cancelled);
}

const decodeCompletionResponse = Schema.decodeEffect(
  Schema.fromJsonString(ChatCompletionResponseSchema),
);

/**
 * Send the raw transcript for refinement and resolve with the refined text,
 * exactly as the model returned it — the caller stores it beside the raw
 * transcript, never in place of it.
 */
export function refineEffect(
  rawTranscript: string,
  settings: RefinementSettings,
  options: RefineEffectOptions = {},
): Effect.Effect<string, RefinementError> {
  return Effect.fnUntraced(function* () {
    // Only trailing slashes are normalized; the version path is part of the
    // user's configured base (http://127.0.0.1:11434/v1, https://api.openai.com/v1).
    const baseUrl = settings.baseUrl.trim().replace(/\/+$/, "");

    if (!baseUrl) {
      return yield* new RefinementInputError(
        "Enter the base URL of an OpenAI-compatible refinement endpoint, for example http://127.0.0.1:11434/v1.",
      );
    }

    const model = settings.model.trim();

    if (!model) {
      return yield* new RefinementInputError(
        "Enter the model name used for transcript refinement.",
      );
    }

    if (rawTranscript.length === 0) {
      return yield* new RefinementInputError("There is no transcript text to refine yet.");
    }

    const fetcher = options.fetchImpl ?? globalThis.fetch;

    if (!fetcher) {
      return yield* new RefinementInputError(
        "A Fetch API implementation is required for transcript refinement.",
      );
    }

    const headers = new Headers({ "Content-Type": "application/json" });
    const apiKey = settings.apiKey?.trim();

    // Bearer auth is attached only when a key is configured, so purely local
    // endpoints (Ollama, llama.cpp, LM Studio) never receive the header.
    if (apiKey) headers.set("Authorization", `Bearer ${apiKey}`);

    const request = Effect.tryPromise({
      try: async (fiberSignal) => {
        const response = await fetcher(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model,
            messages: buildRefinementMessages(rawTranscript, settings),
            stream: false,
          } satisfies ChatCompletionRequest),
          signal: fiberSignal,
          redirect: "manual",
        });

        const body = await response.text();

        return {
          body,
          ok: response.ok,
          opaqueRedirect: response.type === "opaqueredirect",
          status: response.status,
          statusText: response.statusText,
        } satisfies BufferedCompletion;
      },
      catch: (cause) => new RefinementTransportError(describeCause(cause)),
    }).pipe(Effect.flatMap(ensureNotRedirected), Effect.flatMap(ensureOk));

    const response = yield* withExternalAbort(
      withTimeout(request, options.timeoutMs ?? DEFAULT_REFINEMENT_TIMEOUT_MS),
      options.signal,
    );

    const wire = yield* decodeCompletionResponse(response.body).pipe(
      Effect.mapError(
        () =>
          new RefinementProtocolError(
            "The refinement server returned a malformed chat completion.",
          ),
      ),
    );

    const choice = wire.choices[0];

    if (choice === undefined) {
      return yield* new RefinementProtocolError(
        "The refinement server returned no refined transcript.",
      );
    }

    if (choice.message.content.length === 0) {
      return yield* new RefinementProtocolError(
        "The refinement server returned an empty refined transcript.",
      );
    }

    return choice.message.content;
  })();
}
