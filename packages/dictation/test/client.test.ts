import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";
import { Effect, Fiber } from "effect";

import {
  DictationHttpError,
  DictationInputError,
  DictationProtocolError,
  DictationTimeoutError,
  StarlingClient,
} from "../src/client.js";
import { prepareWav16k } from "../src/audio.js";

const prepared = await prepareWav16k({
  samples: new Float32Array(1_600),
  sampleRate: 16_000,
});

describe("StarlingClient protocol compatibility", () => {
  it("uploads multipart WAV to the legacy endpoint with proxy auth", async () => {
    let capturedUrl = "";
    let capturedHeaders: Headers | undefined;
    let capturedBody: FormData | undefined;

    const fetcher: typeof fetch = async (input, init) => {
      capturedUrl = String(input);
      capturedHeaders = new Headers(init?.headers);
      const body = init?.body;
      assert.ok(body instanceof FormData);
      capturedBody = body;

      return new Response(
        JSON.stringify({
          text: "  never remove like  ",
          segments: [{ text: "never", start_s: 0, end_s: 0.1 }],
          duration_s: 0.1,
          request_id: "server-id",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    };

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181/",
      auth: { token: "secret" },
      fetch: fetcher,
    });

    const result = await client.transcribe(prepared, { requestId: "client-id" });
    assert.equal(capturedUrl, "http://localhost:8181/inference");
    assert.equal(capturedHeaders?.get("authorization"), "Bearer secret");
    assert.equal(capturedHeaders?.get("x-request-id"), "client-id");
    const uploadedFile = capturedBody?.get("file");
    assert.ok(uploadedFile instanceof Blob);
    assert.equal(uploadedFile.type, "audio/wav");
    assert.equal(capturedBody?.has("model"), false);
    assert.equal(result.text, "  never remove like  ");
    assert.equal(result.durationSeconds, 0.1);
    assert.equal(result.requestId, "server-id");
    assert.deepEqual(result.segments, [{ text: "never", startSeconds: 0, endSeconds: 0.1 }]);
  });

  it("requires an explicit model before uploading through OpenAI", async () => {
    for (const options of [{}, { model: "" }, { model: "   " }]) {
      const client = new StarlingClient({
        baseUrl: "http://localhost:8181",
        protocol: "openai",
        ...options,
        fetch: async () => assert.fail("Missing models must fail before sending audio"),
      });

      await assert.rejects(client.transcribe(prepared), DictationInputError);
    }
  });

  it("uses the standard OpenAI fields and accepts text-only JSON", async () => {
    let capturedUrl = "";
    let capturedBody: FormData | undefined;

    const fetcher: typeof fetch = async (input, init) => {
      capturedUrl = String(input);
      const body = init?.body;
      assert.ok(body instanceof FormData);
      capturedBody = body;

      return new Response(JSON.stringify({ text: "auth" }), {
        status: 200,
        headers: { "Content-Type": "application/json", "X-Request-Id": "response-id" },
      });
    };

    const result = await new StarlingClient({
      baseUrl: "http://localhost:8181",
      protocol: "openai",
      model: "parakeet",
      fetch: fetcher,
    }).transcribe(prepared);

    assert.equal(capturedUrl, "http://localhost:8181/v1/audio/transcriptions");
    assert.equal(capturedBody?.get("model"), "parakeet");
    assert.equal(capturedBody?.get("response_format"), "json");
    assert.equal(result.text, "auth");
    assert.deepEqual(result.segments, []);
    assert.equal(result.durationSeconds, undefined);
    assert.equal(result.requestId, "response-id");
  });

  it("uses the OpenAI models endpoint for health and validates its response", async () => {
    let capturedUrl = "";

    const fetcher: typeof fetch = async (input) => {
      capturedUrl = String(input);

      return new Response(
        JSON.stringify({
          object: "list",
          data: [{ id: "parakeet", object: "model" }],
        }),
        { status: 200 },
      );
    };

    const health = await new StarlingClient({
      baseUrl: "http://localhost:8181",
      protocol: "openai",
      fetch: fetcher,
    }).health();

    assert.equal(capturedUrl, "http://localhost:8181/v1/models");
    assert.deepEqual(health, { status: "ok", phase: "ready", busy: false, model: "parakeet" });

    const malformed = new StarlingClient({
      baseUrl: "http://localhost:8181",
      protocol: "openai",
      fetch: async () =>
        new Response(JSON.stringify({ object: "list", data: [{ name: "missing-id" }] }), {
          status: 200,
        }),
    });

    await assert.rejects(malformed.health(), DictationProtocolError);
  });

  it("surfaces status, server detail, and raw response on HTTP errors", async () => {
    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: async () =>
        new Response(JSON.stringify({ error: "server busy" }), {
          status: 503,
          statusText: "Unavailable",
        }),
    });

    await assert.rejects(
      client.transcribe(prepared),
      (cause) =>
        cause instanceof DictationHttpError &&
        cause.status === 503 &&
        cause.message === "server busy" &&
        cause.responseBody.includes("server busy"),
    );
  });

  it("blocks redirects with the Electron bridge wording instead of following them", async () => {
    let capturedRedirect: string | undefined;

    const fetcher: typeof fetch = async (_input, init) => {
      capturedRedirect = init?.redirect;

      return new Response("", {
        status: 302,
        statusText: "Found",
        headers: { Location: "https://elsewhere.example/inference" },
      });
    };

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: fetcher,
    });

    await assert.rejects(
      client.transcribe(prepared),
      (cause) =>
        cause instanceof DictationHttpError &&
        cause.status === 302 &&
        cause.message === "Server redirect blocked (302). Set the final endpoint explicitly.",
    );
    assert.equal(capturedRedirect, "manual");
  });

  it("sends every request with redirect manual and still accepts ordinary successes", async () => {
    const capturedRedirects: Array<string | undefined> = [];

    const fetcher: typeof fetch = async (_input, init) => {
      capturedRedirects.push(init?.redirect);

      return new Response(JSON.stringify({ status: "ready" }), { status: 200 });
    };

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: fetcher,
    });

    assert.equal((await client.health()).status, "ready");
    await client.cancel("request-id");
    assert.deepEqual(capturedRedirects, ["manual", "manual"]);
  });

  it("surfaces nested OpenAI error messages", async () => {
    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      protocol: "openai",
      model: "missing-model",
      fetch: async () =>
        new Response(JSON.stringify({ error: { message: "unknown model" } }), {
          status: 404,
          statusText: "Not Found",
        }),
    });

    await assert.rejects(
      client.transcribe(prepared),
      (cause) =>
        cause instanceof DictationHttpError &&
        cause.status === 404 &&
        cause.message === "unknown model",
    );
  });

  it("rejects malformed successful responses", async () => {
    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: async () => new Response(JSON.stringify({ text: 42 }), { status: 200 }),
    });

    await assert.rejects(client.transcribe(prepared), DictationProtocolError);
  });

  it("turns transport aborts caused by its deadline into timeout errors", async () => {
    const fetcher: typeof fetch = async (_input, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
      });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      timeoutMs: 5,
      fetch: fetcher,
    });

    await assert.rejects(client.transcribe(prepared), DictationTimeoutError);
  });

  it("keeps protocol failures in the Effect error channel", async () => {
    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: async () => new Response(JSON.stringify({ status: 42 }), { status: 200 }),
    });

    const failure = await Effect.runPromise(Effect.flip(client.healthEffect()));
    assert.ok(failure instanceof DictationProtocolError);
  });

  it("times out while a response body is stalled and aborts its reader", async () => {
    let bodyWasAborted = false;

    const fetcher: typeof fetch = async (_input, init) => {
      const signal = init?.signal;
      assert.ok(signal);

      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          signal.addEventListener(
            "abort",
            () => {
              bodyWasAborted = true;
              controller.error(new Error("body read aborted"));
            },
            { once: true },
          );
        },
      });

      return new Response(body, { status: 200 });
    };

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      timeoutMs: 5,
      fetch: fetcher,
    });

    await assert.rejects(client.health(), DictationTimeoutError);
    assert.equal(bodyWasAborted, true);
  });

  it("aborts fetch when an Effect fiber is interrupted", async () => {
    let requestSignal: AbortSignal | undefined;

    const fetcher: typeof fetch = async (_input, init) =>
      new Promise((_resolve, reject) => {
        requestSignal = init?.signal ?? undefined;
        requestSignal?.addEventListener("abort", () => reject(new Error("interrupted")), {
          once: true,
        });
      });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: fetcher,
    });

    const fiber = Effect.runFork(client.healthEffect());

    while (requestSignal === undefined) {
      await new Promise<void>((resolve) => setImmediate(resolve));
    }

    await Effect.runPromise(Fiber.interrupt(fiber));
    assert.equal(requestSignal.aborted, true);
  });
});
