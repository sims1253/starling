import {
  app,
  BrowserWindow,
  dialog,
  globalShortcut,
  ipcMain,
  safeStorage,
  session,
  type IpcMainEvent,
  type IpcMainInvokeEvent,
} from "electron";
import { Data, Effect, Option, Schema } from "effect";
import { performance } from "node:perf_hooks";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import { parsePendingAudio, pendingAudioReloadWarning, pendingAudioWarning } from "./closeGuard.js";
import { storeRefinementKeySafe } from "./keyProtection.js";
import { StreamBridge, type StreamSink } from "./streamBridge.js";
import {
  HealthInputSchema,
  RefinementKeyLoadInputSchema,
  RefinementKeySaveInputSchema,
  StreamCloseInputSchema,
  StreamCommandInputSchema,
  StreamOpenInputSchema,
  StreamSendInputSchema,
  TranscribeInputSchema,
  type DesktopDiagnostics,
  type HealthInput,
  type PendingAudioState,
  type RefinementKeyLoadInput,
  type RefinementKeySaveInput,
  type RefinementKeySaveResult,
  type RefinementKeyLoadResult,
  type StreamOpenInput,
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

// The renderer's mirror of audio that exists only in its memory (#121). The
// close guard reads it synchronously when a close or quit must be gated.
let pendingAudio: PendingAudioState = { recording: false, finalizing: false, unsavedCount: 0 };

let quitting = false;

// How long an explicit Discard waits for the renderer to delete a durable
// streaming journal before the window is destroyed anyway.
const DISCARD_CLEANUP_BUDGET_MS = 400;

/**
 * After an explicit Discard, ask the renderer to drop its durable streaming
 * journal, then wait (bounded) for the confirmation so the window is not
 * destroyed mid-delete. A renderer that never replies — hung, or an older
 * build without the channel — times out and the journal survives, which
 * recovery turns into a retryable session on next start: fail-safe.
 */
function discardPendingAudio(window: BrowserWindow): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;

    const settle = (): void => {
      if (settled) return;

      settled = true;
      clearTimeout(timer);
      ipcMain.removeListener("starling:discard-cleaned", onCleaned);
      resolve();
    };

    const timer = setTimeout(settle, DISCARD_CLEANUP_BUDGET_MS);

    const onCleaned = (event: IpcMainEvent): void => {
      // Same trust rule as every renderer→main channel: an unconfirmed
      // sender simply never confirms, and the budget expires instead.
      if (!event.senderFrame || !trustedRenderer(event.senderFrame.url)) return;

      settle();
    };

    ipcMain.on("starling:discard-cleaned", onCleaned);

    window.webContents.send("starling:discard-pending");
  });
}

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

/**
 * Store the refinement API key under the host's secret store (B10). The
 * decision lives in keyProtection.ts: ciphertext is produced only by a real
 * OS secret store — never by Linux's basic_text fallback, whose hardcoded
 * password is not protection — and every other outcome reports its status
 * instead of throwing, so the renderer can demand an explicit choice before
 * any plaintext persists. The result never carries the key itself.
 */
function storeRefinementKeyProgram(
  input: RefinementKeySaveInput,
): Effect.Effect<RefinementKeySaveResult, never> {
  return Effect.sync(() => storeRefinementKeySafe(safeStorage, process.platform, input.apiKey));
}

/**
 * Decrypt a previously stored ciphertext. Undecryptable input (corrupt or
 * produced by another app/origin) resolves apiKey null instead of failing:
 * losing the key only disables hosted refinement, it must never block use.
 */
function loadRefinementKeyProgram(
  input: RefinementKeyLoadInput,
): Effect.Effect<RefinementKeyLoadResult, never> {
  return Effect.sync(() => {
    if (!safeStorage.isEncryptionAvailable()) return { apiKey: null };

    try {
      return { apiKey: safeStorage.decryptString(Buffer.from(input.ciphertext, "base64")) };
    } catch {
      return { apiKey: null };
    }
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

/**
 * The packaged app's live-streaming sockets live in the main process (B01):
 * the renderer's static CSP cannot enumerate user-configured LAN ws:// or
 * wss:// endpoints, so the renderer drives its takes over the IPC channels
 * below instead of opening a WebSocket itself. The endpoint arrives under the
 * same validation as the batch channels, and the ws/wss derivation happens
 * only here.
 */
const streamBridge = new StreamBridge();

function streamTransportError(cause: unknown): RequestTransportError {
  return new RequestTransportError({
    message: cause instanceof Error ? cause.message : "The streaming connection failed.",
    cause,
  });
}

/** Route one stream's events to the WebContents that opened it, until it dies. */
function streamSink(event: IpcMainInvokeEvent): StreamSink {
  const contents = event.sender;

  return {
    send: (message) => {
      if (!contents.isDestroyed()) contents.send("starling:stream:event", message);
    },
    onceDestroyed: (cleanup) => contents.once("destroyed", cleanup),
  };
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

// Same trust rule and decode-at-the-boundary recipe as the channels above:
// only the trusted renderer may hand keys in for encryption or ciphertexts
// in for decryption.
ipcMain.handle("starling:refine-key:save", (event, input: RefinementKeySaveInput) =>
  runForSender(
    event,
    Schema.decodeUnknownEffect(RefinementKeySaveInputSchema)(input).pipe(
      Effect.flatMap(storeRefinementKeyProgram),
    ),
  ),
);

ipcMain.handle("starling:refine-key:load", (event, input: RefinementKeyLoadInput) =>
  runForSender(
    event,
    Schema.decodeUnknownEffect(RefinementKeyLoadInputSchema)(input).pipe(
      Effect.flatMap(loadRefinementKeyProgram),
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

// Streaming transport channels (B01): decode at the boundary, then hand the
// validated payload to the bridge. Failures reject with the transport wording
// the preload's readableRejection already knows how to surface.
ipcMain.handle("starling:stream:open", (event, input: StreamOpenInput) =>
  runForSender(
    event,
    Schema.decodeUnknownEffect(StreamOpenInputSchema)(input).pipe(
      Effect.flatMap((decoded) =>
        Effect.gen(function* () {
          // Input-error wording parity with the batch channels; the bridge
          // applies the same rule when it derives the ws(s):// URL.
          yield* cleanEndpoint(decoded.endpoint);

          return yield* Effect.tryPromise({
            try: () => streamBridge.open(decoded, streamSink(event)),
            catch: (cause) => streamTransportError(cause),
          });
        }),
      ),
    ),
  ),
);

ipcMain.handle("starling:stream:send", (event, input: unknown) =>
  runForSender(
    event,
    Schema.decodeUnknownEffect(StreamSendInputSchema)(input).pipe(
      Effect.flatMap((decoded) =>
        Effect.tryPromise({
          try: () => streamBridge.send(decoded),
          catch: (cause) => streamTransportError(cause),
        }),
      ),
    ),
  ),
);

ipcMain.handle("starling:stream:command", (event, input: unknown) =>
  runForSender(
    event,
    Schema.decodeUnknownEffect(StreamCommandInputSchema)(input).pipe(
      Effect.flatMap((decoded) =>
        Effect.tryPromise({
          try: () => streamBridge.command(decoded),
          catch: (cause) => streamTransportError(cause),
        }),
      ),
    ),
  ),
);

// Fire-and-forget teardown from the renderer's close(): no reply is needed,
// and a dropped packet only strands a socket the destroyed cleanup reaps.
ipcMain.on("starling:stream:close", (event, input: unknown) => {
  if (!event.senderFrame || !trustedRenderer(event.senderFrame.url)) return;

  const decoded = Schema.decodeUnknownOption(StreamCloseInputSchema)(input);

  if (Option.isSome(decoded)) streamBridge.close(decoded.value.streamId);
});

ipcMain.on("starling:renderer-ready", (event) => {
  if (!event.senderFrame || !trustedRenderer(event.senderFrame.url)) return;
  rendererReady = true;

  if (pendingToggle) {
    pendingToggle = false;
    event.sender.send("starling:toggle-recording");
  }
});

ipcMain.on("starling:pending-audio", (event, state: PendingAudioState) => {
  if (!event.senderFrame || !trustedRenderer(event.senderFrame.url)) return;

  const parsed = parsePendingAudio(state);

  if (parsed) pendingAudio = parsed;
});

function createWindow(): BrowserWindow {
  rendererReady = false;
  pendingAudio = { recording: false, finalizing: false, unsavedCount: 0 };

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

  window.webContents.on("will-frame-navigate", (event) => {
    if (!trustedRenderer(event.url)) event.preventDefault();
  });

  // Closing the window, quitting the app, or reloading the renderer destroys
  // audio that exists only in renderer memory (#121). Only an explicit
  // Discard in a native dialog may proceed past this gate.
  const confirmDiscard = (detail: string): boolean =>
    dialog.showMessageBoxSync(window, {
      type: "warning",
      title: "Discard unsaved audio?",
      message: "Discard unsaved audio?",
      detail,
      buttons: ["Discard", "Cancel"],
      defaultId: 1,
      cancelId: 1,
      noLink: true,
    }) === 0;

  // Bumped by every close request; a discard winding down checks it before
  // destroying so a second dialog the user cancelled is never overridden.
  let closeRequests = 0;

  window.on("close", (event) => {
    closeRequests += 1;

    const detail = pendingAudioWarning(pendingAudio);

    if (!detail) return;

    event.preventDefault();

    if (!confirmDiscard(detail)) {
      // cancelled: keep the window open and abort any quit in progress
      quitting = false;

      return;
    }

    // Discard: give a responsive renderer a beat to drop its durable
    // streaming journal, so the take cannot resurrect on next start; a hung
    // renderer times out and keeps the journal (fail-safe toward recovery).
    // destroy() skips this handler. A quit in progress still completes once
    // the last window is gone; the explicit restart is a cross-version
    // safety net, not a requirement on current Electron. If another close
    // request arrived while this budget ran — the nested dialog lets the
    // user reconsider — this flow stands down for that decision instead.
    const request = closeRequests;

    void discardPendingAudio(window).then(() => {
      if (request !== closeRequests) return;

      window.destroy();

      if (quitting) app.quit();
    });
  });

  // The renderer's beforeunload handler blocks reloads while audio is at
  // risk; preventDefault here ignores that handler and lets the reload
  // through — only after an explicit Discard. A reload cannot wait out the
  // journal delete, so a journaled take is told it will be recovered, not
  // deleted.
  window.webContents.on("will-prevent-unload", (event) => {
    const detail = pendingAudioReloadWarning(pendingAudio) ?? "Reload discards unsaved audio.";

    if (confirmDiscard(detail)) event.preventDefault();
  });

  if (rendererUrl) void window.loadURL(rendererUrl);
  else void window.loadFile(packagedRenderer);

  return window;
}

void app.whenReady().then(() => {
  // clipboard-sanitized-write backs navigator.clipboard.writeText; clipboard reads stay denied.
  session.defaultSession.setPermissionCheckHandler(
    (contents, permission, _requestingOrigin, details) =>
      (permission === "clipboard-sanitized-write" ||
        (permission === "media" && details.mediaType === "audio")) &&
      contents !== null &&
      details.isMainFrame &&
      trustedRenderer(details.requestingUrl ?? ""),
  );
  session.defaultSession.setPermissionRequestHandler((_contents, permission, callback, details) => {
    callback(
      (permission === "clipboard-sanitized-write" ||
        (permission === "media" &&
          "mediaTypes" in details &&
          details.mediaTypes?.length === 1 &&
          details.mediaTypes.every((type) => type === "audio"))) &&
        details.isMainFrame &&
        trustedRenderer(details.requestingUrl),
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

app.on("before-quit", () => {
  quitting = true;
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();

  // Live takes cannot survive the process; close their sockets before exit.
  streamBridge.closeAll();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
