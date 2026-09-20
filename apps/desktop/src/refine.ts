import { Duration, Effect, Option, Predicate, Schema } from "effect";
import { ServerErrorResponseSchema } from "@starling/dictation";
import { completionRejectionMessage, completionVerdict } from "./completionVerdict";

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

/**
 * The built-in instruction for a turn refined inside a thread (#117): the
 * assistant message holds the thread's current text and the user message is
 * a new dictated turn — either an edit instruction about that text or
 * additional dictation. The never-omit clause is scoped to the thread's
 * EXISTING content so a literal-minded model still appends additional
 * dictation instead of declining to "add content". The reply is the
 * complete updated text.
 */
export const REFINEMENT_THREAD_INSTRUCTION = [
  "You are a transcription editor working on a running dictation thread.",
  "The assistant message is the thread's current text.",
  "The user message is a new dictated turn: either an edit instruction about that text or additional dictation — apply it accordingly and return the complete updated text, and nothing else.",
  "Preserve the thread's language and the meaning and order of its existing content.",
  "Never omit, summarize, or translate the thread's existing content; apply the user's turn to it.",
  "Reply without preamble, explanation, or quotation marks.",
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
  /**
   * Multi-turn context (#117): the thread's current text, sent as the
   * assistant message before the new dictated turn. Whitespace-only is
   * treated as absent, so a thread whose earlier turns never refined sends
   * the same system+user pair as a standalone refinement.
   */
  readonly contextText?: string | undefined;
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
 * Lenient about extra fields; the refined text and the termination metadata
 * (finish_reason, message.refusal) are load-bearing (B09). The non-empty
 * guarantee is enforced by the verdict after decoding so an empty completion
 * is reported as its own protocol failure, not a parse error.
 */
const ChatCompletionResponseSchema = Schema.Struct({
  choices: Schema.Array(
    Schema.Struct({
      message: Schema.Struct({
        content: Schema.String,
        refusal: Schema.optionalKey(Schema.NullOr(Schema.String)),
      }),
      finish_reason: Schema.optionalKey(Schema.NullOr(Schema.String)),
    }),
  ),
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

/**
 * The server answered, but its own termination metadata says the content is
 * not a whole refinement: truncated on the output limit, filtered, refused,
 * or carrying a finish_reason this client cannot vouch for (B09). Failing
 * with this error keeps the take's previous refined text and raw transcript
 * exactly as they were.
 */
export class RefinementIncompleteError extends Schema.TaggedError<RefinementIncompleteError>()(
  "RefinementIncompleteError",
  {
    message: Schema.String,
    finishReason: Schema.optionalKey(Schema.String),
  },
) {
  constructor(message: string, finishReason?: string) {
    super(finishReason === undefined ? { message } : { message, finishReason });
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
  | RefinementIncompleteError
  | RefinementInputError
  | RefinementProtocolError
  | RefinementTimeoutError
  | RefinementTransportError;

/** Optional multi-turn context accepted by buildRefinementMessages. */
export interface BuildRefinementOptions {
  /**
   * The thread's current text, sent as the assistant message before the new
   * dictated turn. Whitespace-only is treated as absent, which keeps a
   * thread head or an unrefined thread on the standalone message pair.
   */
  readonly contextText?: string | undefined;
}

/**
 * The system turn for one request. A custom instruction wins verbatim in
 * both modes, exactly like the single-turn path of #191; otherwise thread
 * context selects the multi-turn contract and its absence the base
 * instruction. A named branch instead of a nested ternary, so the selection
 * reads as the contract it encodes.
 */
function resolveSystemInstruction(
  settings: RefinementSettings,
  contextText: string | undefined,
): string {
  const instruction = settings.instruction?.trim();

  if (instruction) return instruction;

  return contextText ? REFINEMENT_THREAD_INSTRUCTION : REFINEMENT_DEFAULT_INSTRUCTION;
}

/**
 * Chat-completions messages for one refinement request. Standalone, the pair
 * is the instruction as the system turn and the raw transcript as the user
 * turn. With `options.contextText` present, an assistant turn carrying the
 * thread's current text is inserted between them and the built-in
 * instruction becomes the multi-turn contract — a custom instruction still
 * wins verbatim in both modes, exactly like the single-turn path.
 */
export function buildRefinementMessages(
  rawTranscript: string,
  settings: RefinementSettings,
  options?: BuildRefinementOptions,
): readonly RefinementMessage[] {
  const contextText = options?.contextText?.trim();

  const messages: RefinementMessage[] = [
    Object.freeze({
      role: "system",
      content: resolveSystemInstruction(settings, contextText),
    }) satisfies RefinementMessage,
  ];

  // The thread's current text rides as the assistant turn: the model reads
  // its own prior answer, which is what a running document is to it.
  if (contextText) {
    messages.push(
      Object.freeze({ role: "assistant", content: contextText }) satisfies RefinementMessage,
    );
  }

  messages.push(
    Object.freeze({ role: "user", content: rawTranscript }) satisfies RefinementMessage,
  );

  return Object.freeze(messages);
}

function describeCause(cause: unknown): string {
  if (Predicate.isError(cause)) return cause.message || cause.name;

  if (Predicate.isString(cause)) return cause;

  return String(cause);
}

/**
 * Replace the API key's own value wherever a server echoed it (B10): error
 * details and bodies sometimes quote the Authorization header back, and the
 * surfaced message must never repeat a credential.
 */
function redactSecret(text: string, secret: string | undefined): string {
  return secret ? text.split(secret).join("[redacted]") : text;
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
  // Duration.millis is explicit on purpose: a bare number in Duration.Input
  // also means milliseconds in this Effect version (verified against the
  // vendored rc.115 — Duration.toMillis(1000) === 1000 — the same reading
  // client.ts relies on), and naming the unit keeps that from being
  // re-derived from the call site.
  if (timeoutMs === 0) return effect;

  return effect.pipe(
    Effect.timeoutOrElse({
      duration: Duration.millis(timeoutMs),
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
  // through its fiber signal.
  let detach: () => void = () => {};

  const cancelled = Effect.callback<never, RefinementTransportError>((resume) => {
    const onAbort = (): void => {
      detach();
      resume(Effect.fail(new RefinementTransportError("The refinement request was cancelled.")));
    };

    detach = () => signal.removeEventListener("abort", onAbort);

    if (signal.aborted) {
      onAbort();

      return;
    }

    signal.addEventListener("abort", onAbort, { once: true });
  });

  // The listener is detached the moment the race settles: onExit observes
  // every exit of the raced request — including the interruption the race
  // imposes on the loser — so it can never outlive the request it was
  // registered for.
  return Effect.raceFirst(
    Effect.onExit(effect, () => Effect.sync(detach)),
    cancelled,
  );
}

/**
 * Hosts for which plain http stays acceptable even with a Bearer key: the
 * loopback interfaces never leave the machine, so local servers (Ollama,
 * llama.cpp, LM Studio) keep working without TLS.
 */
function isLoopbackHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
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

    // Same rule as the dictation client's constructor (which throws a
    // TypeError for the same condition): 0 disables the deadline, anything
    // not a finite non-negative number is refused. Here it surfaces as a
    // typed failure instead of a throw because this is already an Effect.
    const timeoutMs = options.timeoutMs ?? DEFAULT_REFINEMENT_TIMEOUT_MS;

    if (!Number.isFinite(timeoutMs) || timeoutMs < 0) {
      return yield* new RefinementInputError(
        "timeoutMs must be a finite non-negative number, or 0 to disable the deadline.",
      );
    }

    const headers = new Headers({ "Content-Type": "application/json" });
    const apiKey = settings.apiKey?.trim();

    // A Bearer key must never cross the wire in cleartext: with a key set,
    // plain http is reserved for loopback hosts; every other endpoint needs
    // https. Without a key, http stays allowed everywhere (the transcript is
    // not a credential, and the user may genuinely be on an trusted LAN).
    if (apiKey) {
      let url: URL;

      try {
        url = new URL(baseUrl);
      } catch {
        return yield* new RefinementInputError(
          "Enter a valid http(s) base URL for the refinement endpoint, for example https://api.openai.com/v1.",
        );
      }

      if (url.protocol !== "https:" && !isLoopbackHost(url.hostname)) {
        return yield* new RefinementInputError(
          "The refinement API key would be sent in cleartext. Use an https base URL, or a loopback endpoint such as http://127.0.0.1:11434/v1.",
        );
      }
    }

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
            messages: buildRefinementMessages(rawTranscript, settings, {
              contextText: options.contextText,
            }),
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
    }).pipe(
      Effect.flatMap(ensureNotRedirected),
      Effect.flatMap(ensureOk),
      // A server that echoes the Authorization header in its error detail
      // must not get the key repeated back to the user (B10).
      Effect.mapError((error) =>
        error instanceof RefinementHttpError
          ? new RefinementHttpError(
              error.status,
              error.statusText,
              redactSecret(error.responseBody, apiKey),
              redactSecret(error.message, apiKey),
            )
          : error,
      ),
    );

    const response = yield* withExternalAbort(withTimeout(request, timeoutMs), options.signal);

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

    // Termination metadata decides before the content is trusted: a
    // finish_reason of "length" means the non-empty content is only a
    // prefix, and a stated refusal is not a refinement at all (B09).
    const verdict = completionVerdict({
      content: choice.message.content,
      finishReason: choice.finish_reason,
      refusal: choice.message.refusal,
    });

    if (!verdict.accepted) {
      const message = completionRejectionMessage(verdict.rejection);

      if (verdict.rejection.reason === "empty") {
        return yield* new RefinementProtocolError(message);
      }

      const finishReason =
        verdict.rejection.reason === "refusal" ? undefined : verdict.rejection.finishReason;

      return yield* new RefinementIncompleteError(message, finishReason);
    }

    return verdict.content;
  })();
}
