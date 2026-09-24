import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";
import { Effect, Fiber } from "effect";

import {
  DictationHttpError,
  DictationInputError,
  DictationProtocolError,
  DictationResponseTooLargeError,
  DictationTimeoutError,
  DictationTransportError,
  StarlingClient,
} from "../src/client.js";
import { prepareWav16k } from "../src/audio.js";

const prepared = await prepareWav16k({
  samples: new Float32Array(1_600),
  sampleRate: 16_000,
});

describe("StarlingClient transcription API", () => {
  it("uploads multipart WAV to the transcription endpoint with proxy auth", async () => {
    let capturedUrl = "";
    let capturedHeaders: Headers | undefined;
    let capturedBody: FormData | undefined;

    const fetcher: typeof fetch = async (input, init) => {
      capturedUrl = String(input);
      capturedHeaders = new Headers(init?.headers);
      const body = init?.body;
      assert.ok(body instanceof FormData);
      capturedBody = body;

      return new Response(JSON.stringify({ text: "  never remove like  " }), {
        status: 200,
        headers: { "Content-Type": "application/json", "X-Request-Id": "server-id" },
      });
    };

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181/",
      auth: { token: "secret" },
      fetch: fetcher,
    });

    const result = await client.transcribe(prepared, { requestId: "client-id" });
    assert.equal(capturedUrl, "http://localhost:8181/v1/audio/transcriptions");
    assert.equal(capturedHeaders?.get("authorization"), "Bearer secret");
    assert.equal(capturedHeaders?.get("x-request-id"), "client-id");
    const uploadedFile = capturedBody?.get("file");
    assert.ok(uploadedFile instanceof Blob);
    assert.equal(uploadedFile.type, "audio/wav");
    assert.equal(capturedBody?.get("model"), "parakeet");
    assert.equal(result.text, "  never remove like  ");
    assert.equal(result.durationSeconds, undefined);
    assert.equal(result.requestId, "server-id");
    assert.deepEqual(result.segments, []);
  });

  it("requires a nonempty model before uploading", async () => {
    for (const options of [{ model: "" }, { model: "   " }]) {
      const client = new StarlingClient({
        baseUrl: "http://localhost:8181",
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
      fetch: fetcher,
    }).health();

    assert.equal(capturedUrl, "http://localhost:8181/v1/models");
    // busy stays absent: the OpenAI models route cannot observe it.
    assert.deepEqual(health, { status: "ok", phase: "ready", model: "parakeet" });

    const malformed = new StarlingClient({
      baseUrl: "http://localhost:8181",
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
        headers: { Location: "https://elsewhere.example/v1/audio/transcriptions" },
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

  it("rejects browser-opaque redirects with the redirect-blocked message", async () => {
    // A Chromium renderer resolves redirect: "manual" to an opaque response
    // with no readable status; only its type says "this was a redirect".
    const opaque = new Response("", { status: 302 });
    Object.defineProperty(opaque, "type", { value: "opaqueredirect" });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: async () => opaque,
    });

    await assert.rejects(
      client.transcribe(prepared),
      (cause) =>
        cause instanceof DictationHttpError &&
        cause.message === "Server redirect blocked. Set the final endpoint explicitly.",
    );
  });

  it("sends every request with redirect manual and still accepts ordinary successes", async () => {
    const capturedRedirects: Array<string | undefined> = [];

    const fetcher: typeof fetch = async (_input, init) => {
      capturedRedirects.push(init?.redirect);

      return new Response(JSON.stringify({ object: "list", data: [] }), { status: 200 });
    };

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: fetcher,
    });

    assert.equal((await client.health()).status, "ok");
    await client.cancel("request-id");
    assert.deepEqual(capturedRedirects, ["manual", "manual"]);
  });

  it("surfaces nested OpenAI error messages", async () => {
    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
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

  it("cancels a stalled body reader when the deadline fires", async () => {
    let cancelled = false;

    // A passive stream: it never enqueues and reacts to nothing — unlike
    // the test above, nothing errors the controller on abort, so only
    // the client's own deadline machinery can release the connection.
    const body = new ReadableStream<Uint8Array>({
      cancel() {
        cancelled = true;
      },
    });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      timeoutMs: 10,
      fetch: async () => new Response(body, { status: 200 }),
    });

    await assert.rejects(client.health(), DictationTimeoutError);
    assert.equal(cancelled, true, "the stalled reader must be cancelled at the deadline");
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

  it("uploads the prepared WAV by reference instead of re-buffering it", async () => {
    const blobReads: string[] = [];

    // A Blob whose read surface is instrumented: any client-side
    // re-buffering (arrayBuffer/text/stream/slice) would have to go
    // through one of these. Only the platform's own serialization,
    // inside a real fetch, may touch them — this fetch double never does.
    const watchedBlob = new Blob([prepared.wav.slice()], { type: "audio/wav" });

    for (const method of ["arrayBuffer", "slice", "stream", "text"] as const) {
      Object.defineProperty(watchedBlob, method, {
        value: () => {
          blobReads.push(method);
          throw new Error(`client must not call Blob.${method}() while building the request`);
        },
      });
    }

    const watched: typeof prepared = { ...prepared, blob: watchedBlob };
    let capturedBody: FormData | undefined;

    const fetcher: typeof fetch = async (_input, init) => {
      const body = init?.body;
      assert.ok(body instanceof FormData);
      capturedBody = body;

      return new Response(JSON.stringify({ text: "streamed" }), { status: 200 });
    };

    const result = await new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: fetcher,
    }).transcribe(watched);

    // The file part carries the whole recording (issue #235) without the
    // client ever reading or copying the WAV into a second buffer.
    const uploaded = capturedBody?.get("file");
    assert.ok(uploaded instanceof Blob);
    assert.equal(uploaded.type, "audio/wav");
    assert.equal(uploaded.size, prepared.wav.byteLength);
    assert.deepEqual(blobReads, []);
    assert.equal(result.text, "streamed");
  });

  it("rejects an oversized response body with a distinct error and cancels its reader", async () => {
    let pulls = 0;
    let cancelled = false;

    // More chunks than the cap could ever allow: only the client's limit
    // can end the read.
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (pulls === 5) {
          controller.close();

          return;
        }

        pulls += 1;
        controller.enqueue(new Uint8Array(64).fill(0x61));
      },
      cancel() {
        cancelled = true;
      },
    });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      maxResponseBytes: 100,
      fetch: async () => new Response(body, { status: 200 }),
    });

    await assert.rejects(
      client.health(),
      (cause) =>
        cause instanceof DictationResponseTooLargeError &&
        cause.limitBytes === 100 &&
        cause.message === "dictation response body exceeded the 100 byte limit",
    );
    assert.equal(cancelled, true, "the reader must be cancelled at the cap, not drained");
    // 64 + 64 crosses 100 bytes, so the read stops after the second
    // chunk reaches the accumulator — several chunks before the stream
    // ends. (The reader may prefetch one chunk ahead, hence the bound.)
    assert.ok(pulls < 5, `the client kept reading past the cap (${pulls} pulls)`);
  });

  it("enforces the default response cap when none is configured", async () => {
    const twoMebibytes = 2 * 1024 * 1024;
    let sent = 0;

    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (sent === 6) {
          controller.close();

          return;
        }

        // 6 x 2 MiB = 12 MiB crosses the 10 MiB default on chunk six.
        sent += 1;
        controller.enqueue(new Uint8Array(twoMebibytes));
      },
    });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: async () => new Response(body, { status: 200 }),
    });

    await assert.rejects(
      client.health(),
      (cause) =>
        cause instanceof DictationResponseTooLargeError && cause.limitBytes === 10 * 1024 * 1024,
    );
    assert.equal(sent, 6);
  });

  it("reassembles chunked bodies under the cap across multibyte boundaries", async () => {
    const encoded = new TextEncoder().encode('{"data":[{"id":"réady"}]}');
    // Split inside the two-byte é (0xc3 0xa9): a naive per-chunk decode
    // would corrupt it, the streaming decoder must not.
    const split = encoded.indexOf(0xc3) + 1;
    const chunks = [encoded.slice(0, split), encoded.slice(split)];
    let index = 0;

    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        const chunk = chunks[index];
        index += 1;

        if (chunk) controller.enqueue(chunk);
        else controller.close();
      },
    });

    const health = await new StarlingClient({
      baseUrl: "http://localhost:8181",
      fetch: async () => new Response(body, { status: 200 }),
    }).health();

    assert.equal(health.model, "réady");
  });

  it("refuses a response whose declared content-length already exceeds the cap", async () => {
    let cancelled = false;

    // 64 bytes queued — at, not over, the 64-byte cap — while the header
    // declares 65: the pre-check must refuse before a single chunk is
    // read (parity with the Rust client's content-length pre-check).
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(64).fill(0x61));
      },
      cancel() {
        cancelled = true;
      },
    });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      maxResponseBytes: 64,
      fetch: async () => new Response(body, { status: 200, headers: { "Content-Length": "65" } }),
    });

    await assert.rejects(
      client.health(),
      (cause) => cause instanceof DictationResponseTooLargeError && cause.limitBytes === 64,
    );
    assert.equal(cancelled, true, "the reader must be cancelled without reading");
  });

  it("refuses an oversized declared length even when the body is null", async () => {
    // The declaration alone is grounds for refusal: a null body must not
    // slip past the pre-check through the empty-body early return.
    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      maxResponseBytes: 64,
      fetch: async () => new Response(null, { status: 200, headers: { "Content-Length": "65" } }),
    });

    await assert.rejects(
      client.health(),
      (cause) => cause instanceof DictationResponseTooLargeError && cause.limitBytes === 64,
    );
  });

  it("accepts a body exactly at the cap and refuses one byte past it", async () => {
    // The cap is inclusive: `received === limitBytes` must succeed, only
    // `>` throws (mirrors the Rust client's at-the-limit test).
    const clientWith = (fetcher: typeof fetch) =>
      new StarlingClient({
        baseUrl: "http://localhost:8181",
        maxResponseBytes: 64,
        fetch: fetcher,
      });

    const atCap = JSON.stringify({ data: [] }).padEnd(64);

    assert.equal(atCap.length, 64);

    const health = await clientWith(async () => new Response(atCap, { status: 200 })).health();

    assert.equal(health.status, "ok");

    await assert.rejects(
      clientWith(async () => new Response("a".repeat(65), { status: 200 })).health(),
      (cause) => cause instanceof DictationResponseTooLargeError && cause.limitBytes === 64,
    );
  });

  it("keeps the cap refusal out of the transport error channel", async () => {
    // readBodyCapped throws inside Effect.tryPromise: the catch must pass
    // the tagged error through unchanged, never rewrap it as a transport
    // failure (which would flip retry classification downstream).
    const failure = await Effect.runPromise(
      Effect.flip(
        new StarlingClient({
          baseUrl: "http://localhost:8181",
          maxResponseBytes: 64,
          fetch: async () => new Response("a".repeat(65), { status: 200 }),
        }).healthEffect(),
      ),
    );

    assert.ok(failure instanceof DictationResponseTooLargeError);
    assert.equal(failure.limitBytes, 64);
    assert.equal(failure instanceof DictationTransportError, false);
  });

  it("still reports the streaming refusal when cancelling the reader rejects", async () => {
    // The cancel is fire-and-forget: a transport failure while releasing
    // an already-dead connection must not mask the cap error. Under an
    // awaited `reader.cancel()` shape this reject would win and surface
    // as a DictationTransportError — that is what this test pins.
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(65).fill(0x61));
      },
      cancel() {
        return Promise.reject(new TypeError("cancel failed"));
      },
    });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      maxResponseBytes: 64,
      fetch: async () => new Response(body, { status: 200 }),
    });

    await assert.rejects(
      client.health(),
      (cause) => cause instanceof DictationResponseTooLargeError && cause.limitBytes === 64,
    );
  });

  it("still reports the declared-length refusal when cancelling the body rejects", async () => {
    // The pre-check's fire-and-forget cancel, same pin: the rejecting
    // cancel of the underlying source must not displace the cap error.
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(64).fill(0x61));
      },
      cancel() {
        return Promise.reject(new TypeError("cancel failed"));
      },
    });

    const client = new StarlingClient({
      baseUrl: "http://localhost:8181",
      maxResponseBytes: 64,
      fetch: async () => new Response(body, { status: 200, headers: { "Content-Length": "65" } }),
    });

    await assert.rejects(
      client.health(),
      (cause) => cause instanceof DictationResponseTooLargeError && cause.limitBytes === 64,
    );
  });

  it("rejects a non-positive or non-finite maxResponseBytes", () => {
    for (const maxResponseBytes of [0, -1, Number.POSITIVE_INFINITY, Number.NaN]) {
      assert.throws(
        () =>
          new StarlingClient({
            baseUrl: "http://localhost:8181",
            maxResponseBytes,
            fetch: async () => assert.fail("invalid options must fail before any request"),
          }),
        TypeError,
      );
    }
  });

  it("rejects a non-positive limit at the schema level", () => {
    // The error keeps the standard fields-object signature (schema-driven
    // instantiation constructs it like every other tagged error); the
    // strictly positive `limitBytes` field is the enforcement, so the
    // schema machinery itself refuses a non-positive cap. Runtime caps
    // are validated once, at StarlingClient construction.
    const good = new DictationResponseTooLargeError({
      message: "dictation response body exceeded the 64 byte limit",
      limitBytes: 64,
    });

    assert.ok(good instanceof DictationResponseTooLargeError);
    assert.equal(good.limitBytes, 64);

    for (const limitBytes of [0, -1, Number.POSITIVE_INFINITY, Number.NaN]) {
      assert.throws(() => new DictationResponseTooLargeError({ message: "x", limitBytes }));
    }
  });
});
