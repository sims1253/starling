import { describe, expect, it } from "vite-plus/test";
import { Effect, Schema } from "effect";

import {
  REFINEMENT_DEFAULT_INSTRUCTION,
  REFINEMENT_THREAD_INSTRUCTION,
  RefinementHttpError,
  RefinementIncompleteError,
  RefinementInputError,
  RefinementProtocolError,
  RefinementTimeoutError,
  RefinementTransportError,
  buildRefinementMessages,
  refineEffect,
  type RefinementSettings,
} from "./refine";

const settings: RefinementSettings = {
  baseUrl: "http://127.0.0.1:11434/v1",
  model: "llama3.1",
};

interface CapturedRequest {
  readonly url: string;
  readonly init: RequestInit;
}

interface RecordingFetcher {
  readonly captured: CapturedRequest[];
  readonly fetchImpl: typeof fetch;
}

function completion(content: string): Response {
  return new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content } }] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** Named wire contracts for terminatedCompletion's response body (B09). */
interface WireCompletionMessage {
  readonly role: "assistant";
  readonly content: string;
  readonly refusal?: string;
}

interface WireCompletionChoice {
  readonly message: WireCompletionMessage;
  readonly finish_reason?: string | null;
}

/**
 * A completion response carrying termination metadata (B09): finish_reason
 * and, for the refusal case, the message's own refusal field.
 */
function terminatedCompletion(fields: {
  content: string;
  finishReason?: string | null;
  refusal?: string;
}): Response {
  const message: WireCompletionMessage =
    fields.refusal === undefined
      ? { role: "assistant", content: fields.content }
      : { role: "assistant", content: fields.content, refusal: fields.refusal };

  const choice: WireCompletionChoice =
    fields.finishReason === undefined
      ? { message }
      : { message, finish_reason: fields.finishReason };

  return new Response(JSON.stringify({ choices: [choice] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** Recording stand-in for fetch; no module mocking, just injection. */
function recordingFetch(respond: (request: CapturedRequest) => Response): RecordingFetcher {
  const captured: CapturedRequest[] = [];

  return {
    captured,
    fetchImpl: async (input, init) => {
      const request: CapturedRequest = { url: String(input), init: init ?? {} };

      captured.push(request);

      return respond(request);
    },
  };
}

const WireBodySchema = Schema.Struct({
  model: Schema.String,
  messages: Schema.Array(Schema.Struct({ role: Schema.String, content: Schema.String })),
  stream: Schema.Boolean,
});

function wireBody(init: RequestInit) {
  return Schema.decodeUnknownSync(WireBodySchema)(JSON.parse(String(init.body)));
}

describe("buildRefinementMessages", () => {
  it("uses the built-in instruction unless a custom one is set", () => {
    expect(buildRefinementMessages("so i says", settings)).toEqual([
      { role: "system", content: REFINEMENT_DEFAULT_INSTRUCTION },
      { role: "user", content: "so i says" },
    ]);

    expect(buildRefinementMessages("so i says", { ...settings, instruction: "   \n\t  " })).toEqual(
      [
        { role: "system", content: REFINEMENT_DEFAULT_INSTRUCTION },
        { role: "user", content: "so i says" },
      ],
    );

    expect(
      buildRefinementMessages("so i says", { ...settings, instruction: "Make it polite." }),
    ).toEqual([
      { role: "system", content: "Make it polite." },
      { role: "user", content: "so i says" },
    ]);
  });

  it("returns a frozen system+user pair", () => {
    const messages = buildRefinementMessages("raw words", settings);

    expect(Object.isFrozen(messages)).toBe(true);

    for (const message of messages) expect(Object.isFrozen(message)).toBe(true);
    expect(messages.map((message) => message.role)).toEqual(["system", "user"]);
  });

  it("inserts thread context as an assistant turn under the multi-turn instruction", () => {
    expect(
      buildRefinementMessages("make that more formal", settings, {
        contextText: "so i says to her",
      }),
    ).toEqual([
      { role: "system", content: REFINEMENT_THREAD_INSTRUCTION },
      { role: "assistant", content: "so i says to her" },
      { role: "user", content: "make that more formal" },
    ]);
  });

  it("keeps a custom instruction verbatim even with thread context", () => {
    expect(
      buildRefinementMessages(
        "next turn",
        { ...settings, instruction: "Tighten it." },
        {
          contextText: "current text",
        },
      ),
    ).toEqual([
      { role: "system", content: "Tighten it." },
      { role: "assistant", content: "current text" },
      { role: "user", content: "next turn" },
    ]);
  });

  it("treats absent or whitespace-only context as standalone", () => {
    for (const contextText of [undefined, "", "   \n\t  "]) {
      expect(buildRefinementMessages("raw", settings, { contextText })).toEqual([
        { role: "system", content: REFINEMENT_DEFAULT_INSTRUCTION },
        { role: "user", content: "raw" },
      ]);
    }
  });

  it("returns a frozen system+assistant+user triple with context", () => {
    const messages = buildRefinementMessages("raw words", settings, { contextText: "current" });

    expect(Object.isFrozen(messages)).toBe(true);

    for (const message of messages) expect(Object.isFrozen(message)).toBe(true);
    expect(messages.map((message) => message.role)).toEqual(["system", "assistant", "user"]);
  });
});

describe("refineEffect", () => {
  it("posts a non-streaming chat completion and resolves with the refined text", async () => {
    const recorder = recordingFetch(() => completion("So I says."));

    const text = await Effect.runPromise(
      refineEffect("so i says", settings, { fetchImpl: recorder.fetchImpl }),
    );

    expect(text).toBe("So I says.");
    expect(recorder.captured.length).toBe(1);

    const [request] = recorder.captured;

    if (!request) throw new Error("no request was captured");
    expect(request.url).toBe("http://127.0.0.1:11434/v1/chat/completions");
    expect(request.init.method).toBe("POST");
    expect(request.init.redirect).toBe("manual");

    const headers = new Headers(request.init.headers);

    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("authorization")).toBe(null);

    const body = wireBody(request.init);

    expect(body.model).toBe("llama3.1");
    expect(body.stream).toBe(false);
    expect(body.messages).toEqual([
      { role: "system", content: REFINEMENT_DEFAULT_INSTRUCTION },
      { role: "user", content: "so i says" },
    ]);
  });

  it("normalizes trailing slashes off the configured base URL", async () => {
    const recorder = recordingFetch(() => completion("Kept."));

    await Effect.runPromise(
      refineEffect(
        "kept",
        { baseUrl: "http://127.0.0.1:11434/v1///", model: "qwen" },
        {
          fetchImpl: recorder.fetchImpl,
        },
      ),
    );

    expect(recorder.captured[0]?.url).toBe("http://127.0.0.1:11434/v1/chat/completions");
  });

  it("sends the thread context through to the wire as an assistant turn", async () => {
    const recorder = recordingFetch(() => completion("Updated."));

    const text = await Effect.runPromise(
      refineEffect("make that more formal", settings, {
        fetchImpl: recorder.fetchImpl,
        contextText: "so i says to her",
      }),
    );

    expect(text).toBe("Updated.");

    const body = wireBody(recorder.captured[0]?.init ?? {});

    expect(body.messages).toEqual([
      { role: "system", content: REFINEMENT_THREAD_INSTRUCTION },
      { role: "assistant", content: "so i says to her" },
      { role: "user", content: "make that more formal" },
    ]);
  });

  it("attaches bearer auth only when an API key is set", async () => {
    const withKey = recordingFetch(() => completion("A."));

    await Effect.runPromise(
      refineEffect("a", { ...settings, apiKey: "sk-test" }, { fetchImpl: withKey.fetchImpl }),
    );

    const withoutKey = recordingFetch(() => completion("B."));

    await Effect.runPromise(refineEffect("b", settings, { fetchImpl: withoutKey.fetchImpl }));

    expect(new Headers(withKey.captured[0]?.init.headers).get("authorization")).toBe(
      "Bearer sk-test",
    );
    expect(new Headers(withoutKey.captured[0]?.init.headers).get("authorization")).toBe(null);
  });

  it("refuses a bearer key over non-loopback cleartext http", async () => {
    // A Bearer key must never cross the wire in cleartext: remote http is
    // refused before any request is sent, while loopback http keeps working
    // for local servers and https keeps working everywhere.
    const unreachable: typeof fetch = async () => {
      throw new Error("a cleartext remote request must never be sent");
    };

    const cleartext = await Effect.runPromise(
      Effect.flip(
        refineEffect(
          "hello",
          { baseUrl: "http://example.com:8080/v1", model: "m", apiKey: "sk" },
          {
            fetchImpl: unreachable,
          },
        ),
      ),
    );

    expect(cleartext).toBeInstanceOf(RefinementInputError);
    expect(cleartext.message).toContain("cleartext");

    const invalidUrl = await Effect.runPromise(
      Effect.flip(
        refineEffect(
          "hello",
          { baseUrl: "not a url", model: "m", apiKey: "sk" },
          {
            fetchImpl: unreachable,
          },
        ),
      ),
    );

    expect(invalidUrl).toBeInstanceOf(RefinementInputError);

    const loopback = recordingFetch(() => completion("Local."));

    await Effect.runPromise(
      refineEffect("hello", { ...settings, apiKey: "sk" }, { fetchImpl: loopback.fetchImpl }),
    );

    expect(loopback.captured.length).toBe(1);

    const https = recordingFetch(() => completion("Remote."));

    await Effect.runPromise(
      refineEffect(
        "hello",
        { baseUrl: "https://api.openai.com/v1", model: "gpt-4o-mini", apiKey: "sk" },
        {
          fetchImpl: https.fetchImpl,
        },
      ),
    );

    expect(https.captured.length).toBe(1);

    // Without a key, plain remote http stays allowed: the transcript is not
    // a credential and the endpoint may be an intentionally trusted LAN box.
    const plainHttp = recordingFetch(() => completion("LAN."));

    await Effect.runPromise(
      refineEffect(
        "hello",
        { baseUrl: "http://192.168.1.10:8080/v1", model: "m" },
        {
          fetchImpl: plainHttp.fetchImpl,
        },
      ),
    );

    expect(plainHttp.captured.length).toBe(1);
  });

  it("surfaces HTTP status and the server's own error detail", async () => {
    const recorder = recordingFetch(
      () =>
        new Response(JSON.stringify({ error: { message: "model not loaded" } }), {
          status: 500,
          statusText: "Internal Server Error",
        }),
    );

    const failure = await Effect.runPromise(
      Effect.flip(refineEffect("hello", settings, { fetchImpl: recorder.fetchImpl })),
    );

    expect(failure).toBeInstanceOf(RefinementHttpError);

    if (failure instanceof RefinementHttpError) {
      expect(failure.status).toBe(500);
      expect(failure.message).toBe("model not loaded");
      expect(failure.responseBody).toContain("model not loaded");
    }
  });

  it("blocks redirects with the same wording as the dictation client", async () => {
    const recorder = recordingFetch(
      () =>
        new Response("", {
          status: 302,
          statusText: "Found",
          headers: { Location: "https://elsewhere.example/v1/chat/completions" },
        }),
    );

    const failure = await Effect.runPromise(
      Effect.flip(refineEffect("hello", settings, { fetchImpl: recorder.fetchImpl })),
    );

    expect(failure).toBeInstanceOf(RefinementHttpError);
    expect(failure.message).toBe(
      "Server redirect blocked (302). Set the final endpoint explicitly.",
    );
    expect(recorder.captured[0]?.init.redirect).toBe("manual");
  });

  it("rejects opaque renderer redirects", async () => {
    // A Chromium renderer resolves redirect: "manual" to an opaque response;
    // only its type says "this was a redirect".
    const opaque = new Response("", { status: 302 });

    Object.defineProperty(opaque, "type", { value: "opaqueredirect" });

    const failure = await Effect.runPromise(
      Effect.flip(refineEffect("hello", settings, { fetchImpl: async () => opaque })),
    );

    expect(failure).toBeInstanceOf(RefinementHttpError);
    expect(failure.message).toBe("Server redirect blocked. Set the final endpoint explicitly.");
  });

  it("rejects malformed completion bodies", async () => {
    for (const body of ["not json", "{}", JSON.stringify({ choices: "no" })]) {
      const failure = await Effect.runPromise(
        Effect.flip(
          refineEffect("hello", settings, {
            fetchImpl: async () => new Response(body, { status: 200 }),
          }),
        ),
      );

      expect(failure).toBeInstanceOf(RefinementProtocolError);
      expect(failure.message).toContain("malformed");
    }
  });

  it("rejects completions with no choices or empty content", async () => {
    const noChoices = await Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, {
          fetchImpl: async () => new Response(JSON.stringify({ choices: [] }), { status: 200 }),
        }),
      ),
    );

    expect(noChoices).toBeInstanceOf(RefinementProtocolError);
    expect(noChoices.message).toContain("no refined transcript");

    const emptyContent = await Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, {
          fetchImpl: async () =>
            new Response(JSON.stringify({ choices: [{ message: { content: "" } }] }), {
              status: 200,
            }),
        }),
      ),
    );

    expect(emptyContent).toBeInstanceOf(RefinementProtocolError);
    expect(emptyContent.message).toContain("empty refined transcript");
  });

  it("rejects whitespace-only content as an empty transcript", async () => {
    const failure = await Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, {
          fetchImpl: async () =>
            new Response(
              JSON.stringify({
                choices: [{ message: { role: "assistant", content: "   \n\t  " } }],
              }),
              { status: 200 },
            ),
        }),
      ),
    );

    expect(failure).toBeInstanceOf(RefinementProtocolError);
    expect(failure.message).toContain("empty refined transcript");
  });

  it("rejects a non-empty but truncated completion instead of refining with it", async () => {
    // finish_reason "length" means the content is only a prefix: accepting it
    // would store a silent omission as a complete refinement (B09).
    const failure = await Effect.runPromise(
      Effect.flip(
        refineEffect("a long dictated take", settings, {
          fetchImpl: async () =>
            terminatedCompletion({ content: "Only the first half of", finishReason: "length" }),
        }),
      ),
    );

    expect(failure).toBeInstanceOf(RefinementIncompleteError);
    expect(failure.message).toContain('finish_reason "length"');
    expect(failure.message).toContain("output limit");

    if (failure instanceof RefinementIncompleteError) expect(failure.finishReason).toBe("length");
  });

  it("rejects a content-filtered completion", async () => {
    const failure = await Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, {
          fetchImpl: async () =>
            terminatedCompletion({ content: "", finishReason: "content_filter" }),
        }),
      ),
    );

    expect(failure).toBeInstanceOf(RefinementIncompleteError);
    expect(failure.message).toContain("content_filter");
  });

  it("rejects a model refusal carried in the message", async () => {
    const failure = await Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, {
          fetchImpl: async () =>
            terminatedCompletion({
              content: "",
              finishReason: "stop",
              refusal: "I cannot edit this transcript.",
            }),
        }),
      ),
    );

    expect(failure).toBeInstanceOf(RefinementIncompleteError);
    expect(failure.message).toContain("refused");
    expect(failure.message).toContain("I cannot edit this transcript.");
  });

  it("rejects an unrecognized finish_reason instead of treating it as stop", async () => {
    const failure = await Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, {
          fetchImpl: async () =>
            terminatedCompletion({ content: "Something.", finishReason: "eos" }),
        }),
      ),
    );

    expect(failure).toBeInstanceOf(RefinementIncompleteError);
    expect(failure.message).toContain('"eos"');
  });

  it("accepts an explicit stop and a compatible local response without the field", async () => {
    const stopped = await Effect.runPromise(
      refineEffect("hello", settings, {
        fetchImpl: async () => terminatedCompletion({ content: "Stopped.", finishReason: "stop" }),
      }),
    );

    expect(stopped).toBe("Stopped.");

    const nullReason = await Effect.runPromise(
      refineEffect("hello", settings, {
        fetchImpl: async () =>
          terminatedCompletion({ content: "Null reason.", finishReason: null }),
      }),
    );

    expect(nullReason).toBe("Null reason.");

    const omitted = await Effect.runPromise(
      refineEffect("hello", settings, { fetchImpl: async () => completion("Omitted.") }),
    );

    expect(omitted).toBe("Omitted.");
  });

  it("shows an actionable recovery path when a long transcript hits the output limit", async () => {
    // A transcript long enough to strain a local model's output budget, sent
    // whole: the recovery path must be about limits and shorter takes, not a
    // protocol complaint.
    const longTranscript = `${"so i says to her the model keeps going and ".repeat(400)}`;

    const truncated = await Effect.runPromise(
      Effect.flip(
        refineEffect(longTranscript, settings, {
          fetchImpl: async () =>
            terminatedCompletion({
              content: "So I says to her the model keeps going and ".repeat(40),
              finishReason: "length",
            }),
        }),
      ),
    );

    expect(truncated).toBeInstanceOf(RefinementIncompleteError);
    expect(truncated.message).toContain("num_predict");
    expect(truncated.message).toContain("shorter take");

    // The same long transcript with a whole answer still refines: limits are
    // the server's verdict, not an assumption about transcript size.
    const whole = await Effect.runPromise(
      refineEffect(longTranscript, settings, {
        fetchImpl: async () =>
          terminatedCompletion({
            content: `${longTranscript.trim()}.`,
            finishReason: "stop",
          }),
      }),
    );

    expect(whole.endsWith(".")).toBe(true);
  });

  it("redacts an echoed API key from surfaced error text and bodies (B10)", async () => {
    // A misbehaving server can echo the Authorization header inside its own
    // error detail; the surfaced error must not repeat the credential.
    const failure = await Effect.runPromise(
      Effect.flip(
        refineEffect(
          "hello",
          { ...settings, apiKey: "sk-secret-value" },
          {
            fetchImpl: async () =>
              new Response(JSON.stringify({ detail: "Invalid API key: Bearer sk-secret-value" }), {
                status: 401,
                statusText: "Unauthorized",
              }),
          },
        ),
      ),
    );

    expect(failure).toBeInstanceOf(RefinementHttpError);
    expect(failure.message).not.toContain("sk-secret-value");
    expect(failure.message).toContain("[redacted]");

    if (failure instanceof RefinementHttpError) {
      expect(failure.responseBody).not.toContain("sk-secret-value");
      expect(failure.responseBody).toContain("[redacted]");
    }
  });

  it("turns its deadline into a timeout error and aborts the request", async () => {
    let requestAborted = false;

    const never: typeof fetch = (_input, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => {
            requestAborted = true;
            reject(init.signal?.reason);
          },
          { once: true },
        );
      });

    const startedAt = Date.now();

    const failure = await Effect.runPromise(
      Effect.flip(refineEffect("hello", settings, { fetchImpl: never, timeoutMs: 5 })),
    );

    // Units guard, not just outcome: Duration.Input reads bare numbers and
    // Duration.millis identically in this Effect version, so 5 means 5 ms and
    // the deadline must land far inside a second — a seconds interpretation
    // of the same option would take ~5_000 ms.
    expect(Date.now() - startedAt).toBeLessThan(1_000);
    expect(failure).toBeInstanceOf(RefinementTimeoutError);
    expect(requestAborted).toBe(true);
  });

  it("treats zero as no deadline and rejects invalid deadlines as input errors", async () => {
    // 0 disables the deadline: a fetch that resolves on a later tick succeeds.
    const later: typeof fetch = async () => {
      await new Promise<void>((resolve) => setTimeout(resolve, 20));

      return completion("Unhurried.");
    };

    const unhurried = await Effect.runPromise(
      refineEffect("hello", settings, { fetchImpl: later, timeoutMs: 0 }),
    );

    expect(unhurried).toBe("Unhurried.");

    for (const timeoutMs of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const failure = await Effect.runPromise(
        Effect.flip(refineEffect("hello", settings, { fetchImpl: later, timeoutMs })),
      );

      expect(failure).toBeInstanceOf(RefinementInputError);
      expect(failure.message).toContain("timeoutMs");
    }
  });

  it("fails as cancelled when the caller aborts mid-request", async () => {
    let requestSignal: AbortSignal | undefined;

    const stalled: typeof fetch = (_input, init) =>
      new Promise((_resolve, reject) => {
        requestSignal = init?.signal ?? undefined;
        requestSignal?.addEventListener("abort", () => reject(new Error("aborted")), {
          once: true,
        });
      });

    const controller = new AbortController();

    const outcome = Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, { fetchImpl: stalled, signal: controller.signal }),
      ),
    );

    while (requestSignal === undefined) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }

    controller.abort();

    const failure = await outcome;

    expect(failure).toBeInstanceOf(RefinementTransportError);
    expect(failure.message).toBe("The refinement request was cancelled.");
    expect(requestSignal.aborted).toBe(true);
  });

  it("detaches the abort listener once the request settles", async () => {
    // After a successful refinement, the caller's signal firing must be a
    // complete no-op: the listener was removed when the race settled, so no
    // late resume — and no unhandled rejection — can follow teardown.
    const controller = new AbortController();
    const recorder = recordingFetch(() => completion("Done."));

    const text = await Effect.runPromise(
      refineEffect("hello", settings, { fetchImpl: recorder.fetchImpl, signal: controller.signal }),
    );

    expect(text).toBe("Done.");

    controller.abort();
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    expect(recorder.captured.length).toBe(1);
  });

  it("fails as cancelled when the signal is already aborted", async () => {
    const controller = new AbortController();

    controller.abort();

    const failure = await Effect.runPromise(
      Effect.flip(
        refineEffect("hello", settings, {
          fetchImpl: async () => completion("never read"),
          signal: controller.signal,
        }),
      ),
    );

    expect(failure).toBeInstanceOf(RefinementTransportError);
    expect(failure.message).toBe("The refinement request was cancelled.");
  });

  it("fails on input before any request is sent", async () => {
    const unreachable: typeof fetch = async () => {
      throw new Error("refinement must fail before any request is sent");
    };

    const cases: Array<Parameters<typeof refineEffect>> = [
      ["hello", { baseUrl: "  ", model: "llama3.1" }],
      ["hello", { baseUrl: "http://127.0.0.1:11434/v1", model: "  " }],
      ["", settings],
    ];

    for (const [transcript, invalid] of cases) {
      const failure = await Effect.runPromise(
        Effect.flip(refineEffect(transcript, invalid, { fetchImpl: unreachable })),
      );

      expect(failure).toBeInstanceOf(RefinementInputError);
    }
  });
});
