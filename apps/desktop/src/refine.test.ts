import { describe, expect, it } from "vite-plus/test";
import { Effect, Schema } from "effect";

import {
  REFINEMENT_DEFAULT_INSTRUCTION,
  RefinementHttpError,
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

    const failure = await Effect.runPromise(
      Effect.flip(refineEffect("hello", settings, { fetchImpl: never, timeoutMs: 5 })),
    );

    expect(failure).toBeInstanceOf(RefinementTimeoutError);
    expect(requestAborted).toBe(true);
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
