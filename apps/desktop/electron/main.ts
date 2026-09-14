import {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  session,
  type IpcMainInvokeEvent,
} from "electron";
import { Data, Effect, Option, Schema } from "effect";
import { performance } from "node:perf_hooks";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import {
  HealthInputSchema,
  TranscribeInputSchema,
  type DesktopDiagnostics,
  type HealthInput,
  type TranscribeInput,
  type TranscriptionResult,
  type ServerHealth,
} from "./ipc.js";

const processStartedAt = performance.now();

const directory = path.dirname(fileURLToPath(import.meta.url));

const packagedRenderer = path.join(directory, "..", "dist", "index.html");

const rendererUrl = process.env.STARLING_RENDERER_URL;

const maximumAudioBytes = 256 * 1024 * 1024;

const defaultTimeoutMs = 180_000;

type Mutable<T> = { -readonly [Key in keyof T]: T[Key] };

let readyToShowMs: number | undefined;

let rendererReady = false;

let pendingToggle = false;

class RequestInputError extends Data.TaggedError("RequestInputError")<{
  readonly message: string;
}> {}

class RequestTransportError extends Data.TaggedError("RequestTransportError")<{
  readonly message: string;
  readonly cause: unknown;
}> {}

class RequestTimeoutError extends Data.TaggedError("RequestTimeoutError")<{
  readonly message: string;
}> {}

class RequestHttpError extends Data.TaggedError("RequestHttpError")<{
  readonly message: string;
  readonly status: number;
}> {}

const ErrorDetailSchema = Schema.Struct({ detail: Schema.String });

const ErrorStringSchema = Schema.Struct({ error: Schema.String });

const ErrorObjectSchema = Schema.Struct({ error: Schema.Struct({ message: Schema.String }) });

const ErrorMessageSchema = Schema.Struct({ message: Schema.String });

const WireSegmentSchema = Schema.Struct({
  text: Schema.String,
  start_s: Schema.optionalKey(Schema.Finite),
  end_s: Schema.optionalKey(Schema.Finite),
  start: Schema.optionalKey(Schema.Finite),
  end: Schema.optionalKey(Schema.Finite),
});

const WireTranscriptionSchema = Schema.Struct({
  text: Schema.String,
  segments: Schema.optionalKey(Schema.Array(WireSegmentSchema)),
  duration_s: Schema.optionalKey(Schema.Finite),
  duration: Schema.optionalKey(Schema.Finite),
  request_id: Schema.optionalKey(Schema.String),
});

const WireHealthSchema = Schema.Struct({
  status: Schema.String,
  phase: Schema.optionalKey(Schema.String),
  model: Schema.optionalKey(Schema.String),
  loaded: Schema.optionalKey(Schema.Boolean),
  busy: Schema.optionalKey(Schema.Boolean),
  queue_depth: Schema.optionalKey(Schema.Finite),
});

const OpenAiModelsSchema = Schema.Struct({
  object: Schema.optionalKey(Schema.Literal("list")),
  data: Schema.Array(Schema.Struct({ id: Schema.String })),
});

const ErrorDetailJsonSchema = Schema.fromJsonString(ErrorDetailSchema);

const ErrorStringJsonSchema = Schema.fromJsonString(ErrorStringSchema);

const ErrorObjectJsonSchema = Schema.fromJsonString(ErrorObjectSchema);

const ErrorMessageJsonSchema = Schema.fromJsonString(ErrorMessageSchema);

const WireTranscriptionJsonSchema = Schema.fromJsonString(WireTranscriptionSchema);

const WireHealthJsonSchema = Schema.fromJsonString(WireHealthSchema);

const OpenAiModelsJsonSchema = Schema.fromJsonString(OpenAiModelsSchema);

function cleanEndpoint(value: string): Effect.Effect<string, RequestInputError> {
  return Effect.try({
    try: () => {
      const url = new URL(value);

      if (url.protocol !== "http:" && url.protocol !== "https:")
        throw new Error("Server endpoint must use http or https.");

      if (url.username || url.password)
        throw new Error("Put credentials in a trusted proxy, not the endpoint URL.");

      return url.href.replace(/\/$/, "");
    },
    catch: (cause) =>
      new RequestInputError({
        message: cause instanceof Error ? cause.message : "Invalid server endpoint.",
      }),
  });
}

function requestTimeout(value: number | undefined): Effect.Effect<number, RequestInputError> {
  const milliseconds = value ?? defaultTimeoutMs;

  return Number.isFinite(milliseconds) && milliseconds >= 1 && milliseconds <= 600_000
    ? Effect.succeed(milliseconds)
    : Effect.fail(
        new RequestInputError({ message: "Request timeout must be between 1 ms and 10 minutes." }),
      );
}

function validateRequestId(value: string): Effect.Effect<string, RequestInputError> {
  return value && !value.startsWith("#") && !/[\r\n]/.test(value)
    ? Effect.succeed(value)
    : Effect.fail(new RequestInputError({ message: "Invalid transcription request id." }));
}

function errorMessage(body: string): string | undefined {
  const detail = Schema.decodeUnknownOption(ErrorDetailJsonSchema)(body);

  if (Option.isSome(detail)) return detail.value.detail;
  const direct = Schema.decodeUnknownOption(ErrorStringJsonSchema)(body);

  if (Option.isSome(direct)) return direct.value.error;
  const nested = Schema.decodeUnknownOption(ErrorObjectJsonSchema)(body);

  if (Option.isSome(nested)) return nested.value.error.message;

  return Option.getOrUndefined(Schema.decodeUnknownOption(ErrorMessageJsonSchema)(body))?.message;
}

function fetchResponse(url: string, init: RequestInit, timeoutMs: number) {
  return Effect.tryPromise({
    try: async (signal) => {
      const response = await fetch(url, { ...init, signal, redirect: "manual" });
      const body = await response.text();

      return { response, body };
    },
    catch: (cause) =>
      new RequestTransportError({
        message: cause instanceof Error ? cause.message : "Network request failed.",
        cause,
      }),
  }).pipe(
    Effect.timeoutOrElse({
      duration: timeoutMs,
      orElse: () =>
        Effect.fail(
          new RequestTimeoutError({ message: `Request timed out after ${timeoutMs} ms.` }),
        ),
    }),
  );
}

function requestBody(url: string, init: RequestInit, timeoutMs: number) {
  return Effect.gen(function* () {
    const { response, body } = yield* fetchResponse(url, init, timeoutMs);

    if (response.status >= 300 && response.status < 400) {
      return yield* Effect.fail(
        new RequestHttpError({
          status: response.status,
          message: `Server redirect blocked (${response.status}). Set the final endpoint explicitly.`,
        }),
      );
    }

    if (!response.ok) {
      return yield* Effect.fail(
        new RequestHttpError({
          status: response.status,
          message:
            errorMessage(body) ?? `Server returned ${response.status}: ${body.slice(0, 500)}`,
        }),
      );
    }

    return { body, responseRequestId: response.headers.get("x-request-id") };
  });
}

function normalizeTranscription(
  value: typeof WireTranscriptionSchema.Type,
  headerRequestId: string | null,
  fallbackRequestId: string,
) {
  return Effect.gen(function* () {
    const segments = yield* Effect.forEach(value.segments ?? [], (segment) => {
      const start = segment.start_s ?? segment.start;
      const end = segment.end_s ?? segment.end;

      if (start === undefined || end === undefined || start < 0 || end < start) {
        return Effect.fail(
          new RequestInputError({ message: "The server returned invalid transcript timestamps." }),
        );
      }

      return Effect.succeed({ text: segment.text, startSeconds: start, endSeconds: end });
    });

    const duration = value.duration_s ?? value.duration;

    if (duration !== undefined && duration < 0) {
      return yield* Effect.fail(
        new RequestInputError({ message: "The server returned an invalid recording duration." }),
      );
    }

    if (duration === undefined)
      return {
        text: value.text,
        segments,
        requestId: value.request_id ?? headerRequestId ?? fallbackRequestId,
      } satisfies TranscriptionResult;

    return {
      text: value.text,
      segments,
      durationSeconds: duration,
      requestId: value.request_id ?? headerRequestId ?? fallbackRequestId,
    } satisfies TranscriptionResult;
  });
}

function healthProgram(input: HealthInput) {
  return Effect.gen(function* () {
    const base = yield* cleanEndpoint(input.endpoint);
    const timeoutMs = yield* requestTimeout(input.timeoutMs);
    const route = input.protocol === "openai" ? "/v1/models" : "/health";
    const response = yield* requestBody(`${base}${route}`, { method: "GET" }, timeoutMs);

    if (input.protocol === "openai") {
      const models = yield* Schema.decodeUnknownEffect(OpenAiModelsJsonSchema)(response.body);

      const result: Mutable<ServerHealth> = { status: "ok", phase: "ready", busy: false };

      if (models.data[0]) result.model = models.data[0].id;

      return result satisfies ServerHealth;
    }

    const health = yield* Schema.decodeUnknownEffect(WireHealthJsonSchema)(response.body);

    const result: Mutable<ServerHealth> = { status: health.status };

    if (health.phase !== undefined) result.phase = health.phase;

    if (health.model !== undefined) result.model = health.model;

    if (health.loaded !== undefined) result.loaded = health.loaded;

    if (health.busy !== undefined) result.busy = health.busy;

    if (health.queue_depth !== undefined) result.queueDepth = health.queue_depth;

    return result satisfies ServerHealth;
  });
}

function transcribeProgram(input: TranscribeInput) {
  return Effect.gen(function* () {
    const base = yield* cleanEndpoint(input.endpoint);
    const timeoutMs = yield* requestTimeout(input.timeoutMs);
    const id = yield* validateRequestId(input.requestId);

    if (input.audio.byteLength < 44 || input.audio.byteLength > maximumAudioBytes) {
      return yield* Effect.fail(
        new RequestInputError({ message: "Audio payload is empty or too large." }),
      );
    }

    let route = "/transcribe";
    let body: BodyInit;
    let headers: HeadersInit = { "x-request-id": id };

    if (input.protocol === "openai") {
      route = "/v1/audio/transcriptions";
      const form = new FormData();
      form.append("file", new Blob([input.audio], { type: "audio/wav" }), "recording.wav");
      form.append("model", input.model || "parakeet");
      form.append("response_format", "json");
      body = form;
    } else {
      body = Buffer.from(input.audio);
      headers = { ...headers, "content-type": "audio/wav" };
    }

    const response = yield* requestBody(
      `${base}${route}`,
      { method: "POST", headers, body },
      timeoutMs,
    );

    const transcription = yield* Schema.decodeUnknownEffect(WireTranscriptionJsonSchema)(
      response.body,
    );

    return yield* normalizeTranscription(transcription, response.responseRequestId, id);
  });
}

function trustedRenderer(raw: string): boolean {
  try {
    const url = new URL(raw);

    if (!rendererUrl) return url.href === pathToFileURL(packagedRenderer).href;

    return url.origin === new URL(rendererUrl).origin;
  } catch {
    return false;
  }
}

function validateSender(event: IpcMainInvokeEvent): void {
  if (!event.senderFrame || !trustedRenderer(event.senderFrame.url))
    throw new RequestInputError({ message: "Request rejected from an untrusted window." });
}

function runForSender<A, E>(event: IpcMainInvokeEvent, effect: Effect.Effect<A, E>): Promise<A> {
  validateSender(event);
  const controller = new AbortController();
  const interrupt = (): void => controller.abort();
  event.sender.once("destroyed", interrupt);

  return Effect.runPromise(effect, { signal: controller.signal }).finally(() =>
    event.sender.removeListener("destroyed", interrupt),
  );
}

function diagnostics(): DesktopDiagnostics {
  const memory = process.memoryUsage();

  const common = {
    mainProcess: { rssBytes: memory.rss, heapUsedBytes: memory.heapUsed },
    processes: app.getAppMetrics().map((metric) => ({
      pid: metric.pid,
      type: metric.type,
      workingSetKib: metric.memory.workingSetSize,
    })),
  };

  if (readyToShowMs === undefined) return common;

  return { ...common, readyToShowMs };
}

ipcMain.handle("starling:health", (event, input: HealthInput) =>
  runForSender(
    event,
    Schema.decodeUnknownEffect(HealthInputSchema)(input).pipe(Effect.flatMap(healthProgram)),
  ),
);

ipcMain.handle("starling:transcribe", (event, input: TranscribeInput) =>
  runForSender(
    event,
    Schema.decodeUnknownEffect(TranscribeInputSchema)(input).pipe(
      Effect.flatMap(transcribeProgram),
    ),
  ),
);

ipcMain.handle("starling:diagnostics", (event) => {
  validateSender(event);

  if (app.isPackaged)
    throw new RequestInputError({
      message: "Diagnostics are available in development builds only.",
    });

  return diagnostics();
});

ipcMain.on("starling:renderer-ready", (event) => {
  validateSender(event);
  rendererReady = true;

  if (pendingToggle) {
    pendingToggle = false;
    event.sender.send("starling:toggle-recording");
  }
});

function createWindow(): BrowserWindow {
  rendererReady = false;

  const window = new BrowserWindow({
    title: "Starling",
    width: 1180,
    height: 760,
    minWidth: 920,
    minHeight: 620,
    backgroundColor: "#171813",
    show: false,
    webPreferences: {
      preload: path.join(directory, "preload.cjs"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webSecurity: true,
    },
  });

  window.once("ready-to-show", () => {
    readyToShowMs ??= Math.round(performance.now() - processStartedAt);

    if (!app.isPackaged && process.env.STARLING_DIAGNOSTICS === "1")
      console.log(`STARLING_DIAGNOSTICS ${JSON.stringify(diagnostics())}`);
    window.show();
  });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event, url) => {
    if (!trustedRenderer(url)) event.preventDefault();
  });

  if (rendererUrl) void window.loadURL(rendererUrl);
  else void window.loadFile(packagedRenderer);

  return window;
}

void app.whenReady().then(() => {
  session.defaultSession.setPermissionCheckHandler(
    (contents, permission) =>
      permission === "media" && contents !== null && trustedRenderer(contents.getURL()),
  );
  session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    callback(
      permission === "media" &&
        "mediaTypes" in details &&
        details.mediaTypes?.includes("audio") === true &&
        trustedRenderer(contents.getURL()),
    );
  });
  let window = createWindow();
  const accelerator = "CommandOrControl+Shift+Space";

  if (
    !globalShortcut.register(accelerator, () => {
      if (window.isDestroyed()) window = createWindow();

      if (window.isMinimized()) window.restore();
      window.show();
      window.focus();

      if (rendererReady) window.webContents.send("starling:toggle-recording");
      else pendingToggle = true;
    })
  )
    console.warn(`Global shortcut unavailable: ${accelerator}`);
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) window = createWindow();
  });
});

app.on("will-quit", () => globalShortcut.unregisterAll());

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
