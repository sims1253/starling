import { Effect, Match } from "effect";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeTranscript,
  encodePcm16kMono,
  IndexedDbSessionStore,
  invalidSessionWav,
  prepareWav16k,
  StarlingClient,
  StarlingStream,
  type DictationSession,
  type InvalidStoredSession,
  type TranscriptionProtocol,
  type TranscriptionResult,
} from "@starling/dictation";
import {
  Check,
  ChevronRight,
  CircleAlert,
  Clipboard,
  Clock3,
  Download,
  FileAudio,
  LoaderCircle,
  Mic,
  Radio,
  RefreshCw,
  Settings2,
  Trash2,
  X,
} from "lucide-react";
import { useRecorder } from "./useRecorder";
import { StreamingDictation, type StreamingState } from "./streamingDictation";
import { TakeLifecycle, type TakePhase } from "./takeLifecycle";
import type { PendingAudioState } from "../electron/ipc.js";

type Connection = "checking" | "ready" | "busy" | "offline";

const DEFAULT_ENDPOINT = window.starlingDesktop ? "http://127.0.0.1:8181" : "/api";

const SETTINGS_FOCUSABLE = "button:not([disabled]), input:not([disabled]), select:not([disabled])";

const store = new IndexedDbSessionStore();

function formatDuration(ms?: number) {
  if (!ms) return "0:00";
  const total = Math.round(ms / 1000);

  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function formatWhen(iso: string) {
  const date = new Date(iso);
  const today = new Date().toDateString() === date.toDateString();

  return today
    ? date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
    : date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function takeTitle(session: DictationSession) {
  return (
    session.transcript?.text ||
    (session.status === "failed" ? "Saved. Retry available" : "Transcribing…")
  );
}

function historyRowLabel(session: DictationSession) {
  const title = takeTitle(session);
  const codePoints = Array.from(title);
  const brief = codePoints.length > 60 ? `${codePoints.slice(0, 60).join("")}…` : title;

  return `${brief}, ${formatWhen(session.createdAt)}`;
}

function messageFrom(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
}

/** Stable-enough id for WAVs parked in memory when storage fails. */
function unsavedWavId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `unsaved-${Date.now()}-${Math.random()}`;
}

async function nativeTranscribe(
  endpoint: string,
  protocol: TranscriptionProtocol,
  model: string,
  id: string,
  wav: Blob,
): Promise<TranscriptionResult> {
  const bridge = window.starlingDesktop;

  if (!bridge) throw new Error("The desktop bridge is unavailable.");

  return bridge.transcribe({
    endpoint,
    requestId: id,
    audio: await wav.arrayBuffer(),
    protocol,
    model,
  });
}

export default function App() {
  const [endpoint, setEndpoint] = useState(
    () => localStorage.getItem("starling:endpoint") ?? DEFAULT_ENDPOINT,
  );

  const [protocol, setProtocol] = useState<TranscriptionProtocol>(() =>
    localStorage.getItem("starling:protocol") === "openai" ? "openai" : "starling",
  );

  const [model, setModel] = useState(() => localStorage.getItem("starling:model") ?? "parakeet");

  const [streamLive, setStreamLive] = useState(
    () => localStorage.getItem("starling:streaming") !== "0",
  );

  const [draftEndpoint, setDraftEndpoint] = useState(endpoint);
  const [connection, setConnection] = useState<Connection>("checking");
  const [serverModel, setServerModel] = useState("server");
  const [sessions, setSessions] = useState<DictationSession[]>([]);
  const [damaged, setDamaged] = useState<readonly InvalidStoredSession[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [activeIds, setActiveIds] = useState<ReadonlySet<string>>(() => new Set());
  const activeUploadsRef = useRef(new Set<string>());
  const [error, setError] = useState<string>();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [copied, setCopied] = useState(false);

  const [expectedTerms, setExpectedTerms] = useState(
    () => localStorage.getItem("starling:terms") ?? "",
  );

  const fileRef = useRef<HTMLInputElement>(null);
  const settingsGearRef = useRef<HTMLButtonElement>(null);
  const settingsDialogRef = useRef<HTMLElement>(null);
  const endpointInputRef = useRef<HTMLInputElement>(null);

  const closeSettings = useCallback(() => setSettingsOpen(false), []);

  const { recording, elapsedMs, levels, start, stop } = useRecorder();

  // Live transcription state for the take in progress.
  const [partialText, setPartialText] = useState<string>();
  const [streamStatus, setStreamStatus] = useState<{ state: StreamingState; detail?: string }>();
  const streamRef = useRef<StreamingDictation | undefined>(undefined);
  // Set when the capture rate cannot be streamed; Stop then uses the batch path.
  const streamBailedRef = useRef(false);
  // True from Stop until the streamed take is finalized (persisted and
  // transcribed or handed to batch): its audio stays durable throughout, so
  // the close guard must keep reporting it as journaled. State, not a ref:
  // the pending-audio mirror must re-run the moment durability ends.
  const [streamingFinalize, setStreamingFinalize] = useState(false);

  // One take transitions at a time: the record button and the global shortcut
  // both claim the lifecycle before awaiting anything, so a start or stop
  // issued mid-transition cannot interleave with it (#143).
  const [lifecycle] = useState(() => new TakeLifecycle());
  const [takePhase, setTakePhase] = useState<TakePhase>("idle");
  // Identity of the take being started, captured before any await; bumped
  // whenever the current take is invalidated so late work disposes itself.
  const takeSeqRef = useRef(0);

  const selected = sessions.find((session) => session.id === selectedId);
  const [audioUrl, setAudioUrl] = useState<string>();

  const [unsavedWavs, setUnsavedWavs] = useState<
    Array<{ id: string; blob: Blob; createdAt: string }>
  >([]);

  // True from Stop until the capture is either durably stored or parked in
  // unsavedWavs; that window holds the only copy in memory.
  const [finalizing, setFinalizing] = useState(false);

  const pendingAudioRef = useRef<PendingAudioState>({
    recording: false,
    finalizing: false,
    unsavedCount: 0,
  });

  // Startup recovery must settle before a new take can begin (#144): a Start
  // that raced the sweep would journal into a database still being recovered.
  const startupRecoveryDoneRef = useRef(false);

  const busy = activeIds.size > 0;

  const fidelity = useMemo(
    () =>
      selected?.transcript
        ? analyzeTranscript(selected.transcript.text, {
            expectedTerms: expectedTerms
              .split(",")
              .map((word) => word.trim())
              .filter(Boolean),
          })
        : undefined,
    [expectedTerms, selected],
  );

  const refresh = useCallback(async () => {
    const report = await store.listReport();
    const next = [...report.sessions];
    next.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    setSessions(next);
    setDamaged(report.invalid);
    setSelectedId((current) =>
      current && next.some((item) => item.id === current) ? current : next[0]?.id,
    );
  }, []);

  const checkHealth = useCallback(
    (target = endpoint) => {
      const bridge = window.starlingDesktop;

      const request = bridge
        ? Effect.tryPromise({
            try: () => bridge.health({ endpoint: target, protocol }),
            catch: (cause) => new Error(messageFrom(cause)),
          })
        : new StarlingClient({ baseUrl: target, protocol }).healthEffect();

      return Effect.runPromise(
        request.pipe(
          Effect.match({
            onSuccess: (health) => {
              setServerModel(health.model ?? "server");

              if (
                protocol === "openai" &&
                health.model !== undefined &&
                !localStorage.getItem("starling:model")
              )
                setModel(health.model);
              setConnection(health.busy || (health.queueDepth ?? 0) > 0 ? "busy" : "ready");
              setError(undefined);
            },
            onFailure: (failure) => {
              setConnection("offline");
              // The bridge keeps validation reasons (bad scheme, embedded
              // credentials) verbatim; only transport failures arrive
              // pre-wrapped, and those are the ones worth naming the target.
              setError(
                failure.message.startsWith("Could not reach the transcription server")
                  ? `Could not reach the transcription server at ${target}.`
                  : failure.message,
              );
            },
          }),
        ),
      );
    },
    [endpoint, protocol],
  );

  useEffect(() => {
    void (async () => {
      // Only the interrupted-session sweep needs the listing, and a bad
      // history entry is quarantined by listReport() rather than rejecting —
      // so journal recovery below runs even when history is unreadable.
      try {
        const saved = await store.listReport();

        const interrupted: Array<Promise<DictationSession>> = [];

        for (const session of saved.sessions) {
          // Another window may still own this attempt (#144); only an
          // owner whose signal is gone is treated as interrupted.
          if (session.status !== "transcribing") continue;

          if (await store.transcriptionInFlight(session.id)) continue;

          interrupted.push(
            store.saveFailure(
              session.id,
              "Interrupted before the server returned a transcript. Your audio is ready to retry.",
            ),
          );
        }

        await Promise.all(interrupted);
      } catch (caught) {
        setError(`Could not open saved recordings: ${messageFrom(caught)}`);
      }

      // Streaming captures journal audio as it arrives; anything the app
      // left behind becomes a retryable session instead of orphaned bytes.
      await store
        .recoverStreamCaptures()
        .catch((caught) =>
          setError(`Could not recover streaming captures: ${messageFrom(caught)}`),
        );
      await refresh();
    })()
      .catch((caught) => setError(`Could not open saved recordings: ${messageFrom(caught)}`))
      .finally(() => {
        // However recovery ended, the sweep has run: takes may begin.
        startupRecoveryDoneRef.current = true;
      });
  }, [refresh]);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  // Mirror audio that exists only in this window's memory (#121) to the main
  // process, so closing Starling can warn before the only copy is destroyed.
  // A streaming take additionally journals chunks durably; the guard needs
  // to know so its warning does not claim no audio exists yet.
  useEffect(() => {
    const state: PendingAudioState = {
      recording,
      finalizing,
      unsavedCount: unsavedWavs.length,
      journaled:
        (recording && streamRef.current !== undefined) || streamingFinalize ? true : undefined,
    };

    pendingAudioRef.current = state;
    window.starlingDesktop?.setPendingAudio(state);
  }, [recording, finalizing, streamingFinalize, unsavedWavs.length]);

  // Second line of defense for reloads: the main process asks before honoring
  // a blocked unload, but only this handler knows the live state.
  useEffect(() => {
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      const state = pendingAudioRef.current;

      if (!state.recording && !state.finalizing && state.unsavedCount === 0) return;
      event.preventDefault();
      event.returnValue = true;
    };

    window.addEventListener("beforeunload", onBeforeUnload);

    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, []);

  useEffect(() => {
    if (!selected) return;

    const url = URL.createObjectURL(selected.wav);
    // oxlint-disable-next-line react/set-state-in-effect -- A Blob URL is an external browser resource. Create and revoke it in the effect so abandoned renders cannot leak URLs.
    setAudioUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [selected]);

  useEffect(() => {
    if (!settingsOpen) return;

    endpointInputRef.current?.focus();
    const gear = settingsGearRef.current;

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closeSettings();

        return;
      }

      if (event.key !== "Tab") return;

      const items = Array.from(
        settingsDialogRef.current?.querySelectorAll<HTMLElement>(SETTINGS_FOCUSABLE) ?? [],
      );

      const active = items.findIndex((item) => item === document.activeElement);
      const last = items.length - 1;

      const next = event.shiftKey
        ? items[active <= 0 ? last : active - 1]
        : items[active === -1 || active === last ? 0 : active + 1];

      if (!next) return;
      event.preventDefault();
      next.focus();
    }

    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      gear?.focus();
    };
  }, [closeSettings, settingsOpen]);

  const transcribe = useCallback(
    async (session: DictationSession) => {
      if (activeUploadsRef.current.has(session.id)) return;
      activeUploadsRef.current.add(session.id);
      setActiveIds(new Set(activeUploadsRef.current));
      setSelectedId(session.id);
      setError(undefined);

      try {
        await store.markAttempt(session.id);
        await refresh();

        const request = window.starlingDesktop
          ? Effect.tryPromise({
              try: () => nativeTranscribe(endpoint, protocol, model, session.id, session.wav),
              catch: (cause) => new Error(messageFrom(cause)),
            })
          : new StarlingClient({ baseUrl: endpoint, protocol, model }).transcribeEffect(
              session.wav,
              { requestId: session.id },
            );

        const result = await Effect.runPromise(request);

        await store.saveTranscript(session.id, result);
        setConnection("ready");
      } catch (caught) {
        let failure = messageFrom(caught);

        try {
          if (await store.get(session.id)) await store.saveFailure(session.id, caught);
        } catch (storageError) {
          failure = `${failure} Local history update also failed: ${messageFrom(storageError)}`;
        }

        setConnection("offline");
        setError(failure);
      } finally {
        activeUploadsRef.current.delete(session.id);
        setActiveIds(new Set(activeUploadsRef.current));

        try {
          await refresh();
        } catch (caught) {
          setError(messageFrom(caught));
        }
      }
    },
    [endpoint, model, protocol, refresh],
  );

  const saveAndTranscribe = useCallback(
    async (wav: Blob, durationMs?: number) => {
      let created: DictationSession;

      try {
        created = await store.create({ wav, durationMs });
      } catch (caught) {
        setUnsavedWavs((current) => [
          ...current,
          { id: unsavedWavId(), blob: wav, createdAt: new Date().toISOString() },
        ]);
        throw new Error(
          `Local storage failed: ${messageFrom(caught)} Keep this window open and download the unsaved WAV to recover it.`,
        );
      }

      await refresh();
      await transcribe(created);
    },
    [refresh, transcribe],
  );

  const parkUnsavedWav = useCallback((wav: Blob): void => {
    setUnsavedWavs((current) => [
      ...current,
      { id: unsavedWavId(), blob: wav, createdAt: new Date().toISOString() },
    ]);
  }, []);

  /** Wire live streaming for the next take; undefined means "use batch mode". */
  const beginStreamingTake = useCallback(
    async (
      takeId: number,
    ): Promise<((chunk: Float32Array, sampleRate: number) => void) | undefined> => {
      setPartialText(undefined);
      setStreamStatus({ state: "connecting" });
      streamBailedRef.current = false;

      let transport: StarlingStream;

      try {
        transport = new StarlingStream({ baseUrl: endpoint });
      } catch {
        setStreamStatus(undefined);

        return undefined;
      }

      let capture;

      try {
        capture = await store.beginStreamCapture();
      } catch {
        setStreamStatus(undefined);

        return undefined;
      }

      if (takeSeqRef.current !== takeId) {
        // This take was invalidated (for example discarded) while the journal
        // was being created: dispose the provisional journal and socket so
        // neither can be mistaken for a live take later (#143).
        transport.close();
        await capture.abandon().catch(() => {});
        setStreamStatus(undefined);

        return undefined;
      }

      const controller = new StreamingDictation(transport, capture, {
        onPartial: (text) => setPartialText(text),
        onStateChange: (state, detail) => setStreamStatus({ state, detail }),
      });

      streamRef.current = controller;

      void controller.connect();

      return (chunk, sampleRate) => {
        if (sampleRate === 16_000) {
          controller.onChunk(encodePcm16kMono(chunk));

          return;
        }

        // Unexpected capture rate: never stream or journal resampled guesses.
        if (!streamBailedRef.current) {
          streamBailedRef.current = true;
          controller.fail("The microphone capture rate is not supported for live streaming.");
        }
      };
    },
    [endpoint],
  );

  const discardStreamingTake = useCallback(async (): Promise<void> => {
    // Invalidate the in-flight take identity: a beginStreamingTake that is
    // still awaiting storage must not install its controller after this.
    takeSeqRef.current += 1;

    const stream = streamRef.current;
    streamRef.current = undefined;
    streamBailedRef.current = false;
    setPartialText(undefined);
    setStreamStatus(undefined);
    await stream?.abandon().catch(() => {});
  }, []);

  // After an explicit Discard in the close guard, drop the durable journal
  // of the in-flight streaming take so it cannot resurrect on next start,
  // then tell the main process it is safe to destroy the window.
  useEffect(() => {
    const bridge = window.starlingDesktop;

    if (!bridge?.onDiscardPending) return;

    return bridge.onDiscardPending(() => {
      void discardStreamingTake().finally(() => bridge.discardCleanedUp());
    });
  }, [discardStreamingTake]);

  /**
   * Finalize the streamed take this Stop was issued against. Returns false
   * when the stream was never usable and the caller should save the
   * recorder's own capture via the batch path.
   */
  const finishStreamingTake = useCallback(
    async (stream: StreamingDictation, durationMs: number): Promise<boolean> => {
      streamRef.current = undefined;
      const bailed = streamBailedRef.current;
      streamBailedRef.current = false;

      if (bailed) {
        setPartialText(undefined);
        setStreamStatus(undefined);
        await stream.abandon();
        // The journal is gone; the recorder's memory-only capture carries
        // this take through the batch path, so it is no longer journaled.
        setStreamingFinalize(false);

        return false;
      }

      if (stream.journaledChunkCount === 0) {
        // The microphone never reached this controller's journal — its
        // session would be a header-only WAV: drop the empty journal and
        // keep the recorder's capture through the batch path (#143).
        setPartialText(undefined);
        setStreamStatus(undefined);
        await stream.abandon().catch(() => {});
        setStreamingFinalize(false);

        return false;
      }

      const result = await stream.finish(durationMs);
      setPartialText(undefined);
      setStreamStatus(undefined);

      if (!result.session) {
        // The durable save failed; the assembled WAV is the only copy.
        setStreamingFinalize(false);
        parkUnsavedWav(result.wav);
        throw new Error(
          `Local storage failed: ${messageFrom(result.failure ?? "unknown storage failure")} Keep this window open and download the unsaved WAV to recover it.`,
        );
      }

      await refresh();

      if (result.streamed && result.transcript) {
        await store.saveTranscript(result.session.id, result.transcript, { streamed: true });
        setSelectedId(result.session.id);
        setConnection("ready");
        await refresh();
      } else {
        if (result.streamNote) {
          await store.noteStreamError(result.session.id, result.streamNote).catch(() => {});
        }

        await transcribe(result.session);
      }

      return true;
    },
    [parkUnsavedWav, refresh, transcribe],
  );

  const toggleRecording = useCallback(async () => {
    setError(undefined);

    // The lifecycle guard is claimed synchronously, before any await, and is
    // shared by the record button and the global shortcut: a toggle issued
    // while a begin/start/stop/finalize transition is in flight is ignored
    // instead of interleaving with it (#143).
    const shouldStop = lifecycle.current() === "recording";

    if (shouldStop) {
      if (!lifecycle.beginStop()) return;

      // The controller bound to the take being stopped, read before any
      // await: Stop must finalize the take that was recorded (#143).
      const stream = streamRef.current;

      setTakePhase("stopping");

      // From Stop until the capture is durably stored (or parked in
      // unsavedWavs), the only copy lives in this window's memory (#121) —
      // except a streamed take, whose journal/session keeps it durable, so
      // the close guard reports it as journaled for the whole finalize.
      setStreamingFinalize(stream !== undefined);
      setFinalizing(true);

      try {
        const capture = await stop();

        // Test scripts stop after 400 ms; keep this cutoff at or below 250 ms.
        if (!capture || capture.audio.samples.length === 0) {
          await discardStreamingTake();
          throw new Error("No microphone audio was captured.");
        }

        if (capture.durationMs < 250) {
          await discardStreamingTake();
          throw new Error("Recording was too short to keep.");
        }

        if (stream && (await finishStreamingTake(stream, capture.durationMs))) {
          return;
        }

        const prepared = await prepareWav16k(capture.audio);
        await saveAndTranscribe(prepared.blob, capture.durationMs);
      } catch (caught) {
        setError(messageFrom(caught));
      } finally {
        lifecycle.endStop();
        setTakePhase("idle");
        setStreamingFinalize(false);
        setFinalizing(false);
      }

      return;
    }

    // The startup sweep may still be deciding which journals and attempts
    // are abandoned; a take started now could race it (#144).
    if (!startupRecoveryDoneRef.current) {
      setError(
        "Still restoring recordings from a previous session. Try again in a moment.",
      );

      return;
    }

    if (!lifecycle.beginStart()) return;

    setTakePhase("starting");

    try {
      // Everything this start allocates belongs to this take alone; the
      // identity is captured before any await (#143).
      const takeId = ++takeSeqRef.current;
      let onChunk: ((chunk: Float32Array, sampleRate: number) => void) | undefined;

      if (streamLive && protocol === "starling") {
        onChunk = await beginStreamingTake(takeId);
      } else {
        setPartialText(undefined);
        setStreamStatus(undefined);
      }

      // 16 kHz lets each chunk stream as-is; any other actual rate is
      // reported per chunk and degrades that take to batch mode.
      if (onChunk) {
        let started = false;

        try {
          started = await start({ sampleRate: 16_000, onChunk });
        } catch (cause) {
          // Capture setup failed after the stream was wired: drop its
          // journal and socket so neither lingers until the next start.
          await discardStreamingTake();
          throw cause;
        }

        if (!started) {
          // The recorder rejected this start — a preceding stop is still
          // releasing its audio context: dispose the provisional journal
          // and socket of the take that never recorded (#143).
          await discardStreamingTake();

          return;
        }
      } else if (!(await start())) {
        return;
      }

      lifecycle.endStart(true);
      setTakePhase("recording");
    } catch (caught) {
      setError(messageFrom(caught));
    } finally {
      if (lifecycle.current() === "starting") {
        lifecycle.endStart(false);
        setTakePhase("idle");
      }
    }
  }, [
    beginStreamingTake,
    discardStreamingTake,
    finishStreamingTake,
    lifecycle,
    protocol,
    saveAndTranscribe,
    start,
    stop,
    streamLive,
  ]);

  useEffect(() => {
    const bridge = window.starlingDesktop;

    if (!bridge) return;

    const dispose = bridge.onToggleRecording(() => {
      void toggleRecording();
    });

    bridge.ready();

    return dispose;
  }, [toggleRecording]);

  async function importAudio(file?: File) {
    if (!file) return;
    setError(undefined);

    try {
      const prepared = await prepareWav16k(file);
      await saveAndTranscribe(prepared.blob, prepared.durationMs);
    } catch (caught) {
      setError(messageFrom(caught));
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function copyTranscript() {
    if (!selected?.transcript) return;

    try {
      await navigator.clipboard.writeText(selected.transcript.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch (caught) {
      setError(`Could not copy the transcript: ${messageFrom(caught)}`);
    }
  }

  function exportTranscript() {
    if (!selected?.transcript) return;

    const url = URL.createObjectURL(
      new Blob([selected.transcript.text], { type: "text/plain;charset=utf-8" }),
    );

    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `starling-${selected.createdAt.replace(/[:.]/g, "-")}.txt`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
  }

  function exportAudio() {
    if (!selected || !audioUrl) return;
    const anchor = document.createElement("a");
    anchor.href = audioUrl;
    anchor.download = `starling-${selected.createdAt.replace(/[:.]/g, "-")}.wav`;
    anchor.click();
  }

  function exportUnsavedAudio(id: string) {
    const capture = unsavedWavs.find((item) => item.id === id);

    if (capture) {
      const url = URL.createObjectURL(capture.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `starling-unsaved-${capture.createdAt.replace(/[:.]/g, "-")}-${capture.id.slice(0, 8)}.wav`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
    }
    // A download can be cancelled. Keep the in-memory recording until the
    // user explicitly discards it or closes the window.
  }

  function discardUnsavedAudio() {
    if (
      window.confirm(
        "Discard the unsaved recordings in this window? Download any recordings you want to keep first.",
      )
    ) {
      setUnsavedWavs([]);
      setError(undefined);
    }
  }

  // Quarantined history entries stay in the database untouched; this only
  // rescues audio that survived the damage so it can be downloaded.
  function exportDamagedAudio(id: string) {
    const entry = damaged.find((item) => item.id === id);
    const wav = entry && invalidSessionWav(entry);

    if (wav) {
      const url = URL.createObjectURL(wav);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `starling-damaged-${id.replace(/[^a-z0-9-]/gi, "").slice(0, 8)}.wav`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
    }
  }

  async function removeSession(id: string) {
    if (activeUploadsRef.current.has(id)) return;

    try {
      await store.delete(id);
      await refresh();
    } catch (caught) {
      setError(`Could not delete the recording: ${messageFrom(caught)}`);
    }
  }

  function openSettings() {
    setDraftEndpoint(endpoint);
    setSettingsOpen(true);
  }

  async function testConnection() {
    setTestingConnection(true);

    try {
      await checkHealth(draftEndpoint);
    } finally {
      setTestingConnection(false);
    }
  }

  function saveSettings() {
    const clean = draftEndpoint.trim().replace(/\/$/, "");

    if (!clean) return;
    setEndpoint(clean);
    localStorage.setItem("starling:endpoint", clean);
    localStorage.setItem("starling:protocol", protocol);
    localStorage.setItem("starling:model", model);
    localStorage.setItem("starling:streaming", streamLive ? "1" : "0");
    localStorage.setItem("starling:terms", expectedTerms);
    closeSettings();
    void checkHealth(clean);
  }

  return (
    <div className={`app-shell ${selected ? "has-transcript" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <span />
          </span>
          <span>starling</span>
        </div>
        <div className={`connection ${connection}`}>
          <span className="status-dot" />
          {Match.value(connection).pipe(
            Match.when("ready", () => `${serverModel} ready`),
            Match.when("busy", () => `${serverModel} working`),
            Match.orElse((status) => status),
          )}
        </div>
        <button
          ref={settingsGearRef}
          className="icon-button"
          onClick={openSettings}
          aria-label="Open server settings"
        >
          <Settings2 size={19} />
        </button>
      </header>

      <main className="workspace">
        <section className="capture-pane">
          <div className="capture-copy">
            <p className="eyebrow">LOCAL DICTATION</p>
            <h1>{recording ? "Listening closely." : "Say it as you mean it."}</h1>
            <p className="lede">
              Your recording is saved locally, sent only to your selected server, and shown exactly
              as the model returned it.
            </p>
          </div>

          <div className={`recorder ${recording ? "is-recording" : ""}`}>
            <div className="waveform" aria-hidden="true">
              {levels.map((level, index) => (
                <i key={index} style={{ height: `${Math.max(5, level * 106)}px` }} />
              ))}
            </div>
            <button
              className="record-button"
              onClick={() => void toggleRecording()}
              disabled={takePhase === "starting" || takePhase === "stopping"}
              aria-label={recording ? "Stop recording" : "Start recording"}
            >
              <span className="record-button-inner">
                {recording ? <span className="stop-glyph" /> : <Mic size={34} strokeWidth={1.7} />}
              </span>
            </button>
            <div className="record-meta">
              <span>
                {recording ? formatDuration(elapsedMs) : busy ? "Transcribing…" : "Click to record"}
              </span>
              <kbd>{navigator.platform.includes("Mac") ? "⌘" : "Ctrl"} Shift Space</kbd>
            </div>
          </div>

          {recording && streamStatus && (
            <div className={`live-stream ${streamStatus.state}`}>
              {streamStatus.state === "live" && (
                <p className="live-text" aria-live="polite">
                  {partialText || <em>Listening for the first words…</em>}
                </p>
              )}
              {streamStatus.state === "connecting" && (
                <p className="live-note">
                  <Radio size={15} /> Connecting live transcription…
                </p>
              )}
              {(streamStatus.state === "unavailable" || streamStatus.state === "interrupted") && (
                <p className="live-note" role="status">
                  <CircleAlert size={15} />
                  {streamStatus.detail ?? "Live transcription is unavailable."} The recording is
                  still saved and will be transcribed when you stop.
                </p>
              )}
            </div>
          )}

          <button
            className="import-button"
            aria-describedby="import-format"
            onClick={() => fileRef.current?.click()}
          >
            <FileAudio size={17} /> Import an audio file <small aria-hidden="true">.wav</small>
          </button>
          <span id="import-format" className="visually-hidden">
            WAV files only
          </span>
          <input
            ref={fileRef}
            type="file"
            accept=".wav,audio/wav,audio/x-wav"
            hidden
            onChange={(event) => void importAudio(event.target.files?.[0])}
          />

          {(error || unsavedWavs.length > 0) && (
            <div className="error-banner" role="alert">
              <CircleAlert size={18} />
              <div>
                <strong>{error ? "Action failed" : "Recording not saved"}</strong>
                {error && <span>{error}</span>}
                <small>
                  {unsavedWavs.length > 0
                    ? `${unsavedWavs.length} recording${unsavedWavs.length === 1 ? "" : "s"} could not be saved. Download ${unsavedWavs.length === 1 ? "it" : "them"} before you close Starling.`
                    : "Starling keeps audio in history after a successful local save."}
                </small>
              </div>
              {unsavedWavs.length > 0 && (
                <div className="recovery-actions">
                  {unsavedWavs.map((capture, index) => (
                    <button
                      key={capture.id}
                      className="recover-audio"
                      onClick={() => exportUnsavedAudio(capture.id)}
                    >
                      Download WAV {index + 1}
                    </button>
                  ))}
                  <button className="recover-audio" onClick={discardUnsavedAudio}>
                    Discard unsaved
                  </button>
                </div>
              )}
              {unsavedWavs.length === 0 && (
                <button onClick={() => setError(undefined)} aria-label="Dismiss error">
                  <X size={16} />
                </button>
              )}
            </div>
          )}
        </section>

        <aside className="history-pane">
          <div className="history-head">
            <div>
              <p className="eyebrow">ARCHIVE</p>
              <h2>Recent takes</h2>
            </div>
            <Clock3 size={18} />
          </div>
          <div className="history-list">
            {damaged.length > 0 && (
              <div className="history-warning" role="status">
                <CircleAlert size={16} />
                <div>
                  <strong>
                    {damaged.length === 1
                      ? "1 saved recording could not be read"
                      : `${damaged.length} saved recordings could not be read`}
                  </strong>
                  <small>
                    Damaged entries are kept untouched; the list below shows every recording that is
                    still readable.
                  </small>
                  {damaged.some((entry) => invalidSessionWav(entry) !== undefined) && (
                    <div className="recovery-actions">
                      {damaged.map((entry, index) => {
                        const wav = invalidSessionWav(entry);

                        return wav ? (
                          <button
                            key={`${entry.id}-${index}`}
                            className="recover-audio"
                            onClick={() => exportDamagedAudio(entry.id)}
                          >
                            Download WAV {index + 1}
                          </button>
                        ) : null;
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
            {sessions.length === 0 && (
              <div className="history-empty">
                Your recordings will collect here, ready to retry or export.
              </div>
            )}
            {sessions.map((session) => (
              <button
                key={session.id}
                className={`history-row ${session.id === selectedId ? "active" : ""}`}
                onClick={() => setSelectedId(session.id)}
                aria-label={historyRowLabel(session)}
              >
                <span className={`take-state ${session.status}`}>
                  {session.status === "transcribing" ? <LoaderCircle size={14} /> : <span />}
                </span>
                <span className="take-copy">
                  <strong>{takeTitle(session)}</strong>
                  <small>
                    {formatWhen(session.createdAt)} · {formatDuration(session.durationMs)}
                    {session.attemptCount > 1 ? ` · ${session.attemptCount} attempts` : ""}
                  </small>
                </span>
                <ChevronRight size={16} />
              </button>
            ))}
          </div>
        </aside>
      </main>

      {selected && (
        <section className="transcript-drawer">
          <div className="drawer-grip" />
          <div className="transcript-head">
            <div>
              <p className="eyebrow">RAW TRANSCRIPT</p>
              <span>No cleanup or silent rewriting</span>
            </div>
            <div className="transcript-actions">
              {selected.status !== "transcribed" && !activeIds.has(selected.id) && (
                <button onClick={() => void transcribe(selected)}>
                  <RefreshCw size={16} /> Retry
                </button>
              )}
              <button onClick={exportAudio}>
                <FileAudio size={16} /> WAV
              </button>
              <button disabled={!selected.transcript} onClick={() => void copyTranscript()}>
                {copied ? <Check size={16} /> : <Clipboard size={16} />}
                {copied ? "Copied" : "Copy"}
              </button>
              <button disabled={!selected.transcript} onClick={exportTranscript}>
                <Download size={16} /> Export
              </button>
              <button
                className="danger"
                disabled={activeIds.has(selected.id)}
                onClick={() => void removeSession(selected.id)}
                aria-label="Delete saved recording"
              >
                <Trash2 size={16} />
              </button>
            </div>
          </div>
          <div className={`transcript-body ${selected.status}`}>
            {audioUrl && (
              <audio className="audio-review" controls preload="metadata" src={audioUrl}>
                Review saved recording
              </audio>
            )}
            {selected.status === "transcribing" && (
              <p className="processing">
                <LoaderCircle size={20} /> The server is transcribing this recording…
              </p>
            )}
            {selected.status === "failed" && (
              <p className="failed-copy">
                {selected.lastError ||
                  "The server could not transcribe this take. The original audio is safe here."}
              </p>
            )}
            {selected.transcript && (
              <p>{selected.transcript.text || <em>The model returned an empty transcript.</em>}</p>
            )}
          </div>
          {selected.streamError && selected.transcript ? (
            <div className="fidelity-note">
              <CircleAlert size={15} />
              <span>
                {selected.streamError} This transcript came from a full upload of the saved WAV.
              </span>
            </div>
          ) : null}
          {fidelity?.warnings.length ? (
            <div className="fidelity-note">
              <CircleAlert size={15} />
              <span>{fidelity.warnings.map((warning) => warning.message).join(" ")}</span>
            </div>
          ) : null}
        </section>
      )}

      {settingsOpen && (
        <div
          className="modal-layer"
          onMouseDown={(event) => event.target === event.currentTarget && closeSettings()}
        >
          <section
            ref={settingsDialogRef}
            className="settings-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
          >
            <button className="settings-close" onClick={closeSettings} aria-label="Close settings">
              <X size={18} />
            </button>
            <p className="eyebrow">CONNECTION</p>
            <h2 id="settings-title">Transcription server</h2>
            <p>
              Choose a <code>starling-serve</code> endpoint or an OpenAI-compatible transcription
              endpoint.
            </p>
            <label>
              Server endpoint
              <input
                ref={endpointInputRef}
                value={draftEndpoint}
                onChange={(event) => setDraftEndpoint(event.target.value)}
                placeholder="http://127.0.0.1:8181"
              />
            </label>
            <div className="settings-grid">
              <label>
                API format
                <select
                  value={protocol}
                  onChange={(event) =>
                    setProtocol(event.target.value === "openai" ? "openai" : "starling")
                  }
                >
                  <option value="starling">Starling native</option>
                  <option value="openai">OpenAI compatible</option>
                </select>
              </label>
              <label>
                Model
                <input
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  placeholder="parakeet"
                />
              </label>
            </div>
            <label className="settings-check">
              <input
                type="checkbox"
                checked={streamLive}
                disabled={protocol === "openai"}
                onChange={(event) => setStreamLive(event.target.checked)}
              />
              <span>
                Live streaming transcript
                <small>
                  Streams audio to a Starling native server while you speak. Requires the Starling
                  API format; recordings always save locally first and fall back to a full upload if
                  the stream fails.
                </small>
              </span>
            </label>
            <label>
              Words to watch
              <input
                value={expectedTerms}
                onChange={(event) => setExpectedTerms(event.target.value)}
                placeholder="auth, Starling, GGUF"
              />
              <small>
                Comma-separated terms are checked after transcription. They are never inserted or
                substituted.
              </small>
            </label>
            <div className="settings-callout">
              <span className={`status-dot ${connection}`} />
              <div>
                <strong>
                  {connection === "ready" ? "Server connected" : "Server needs attention"}
                </strong>
                <span>{endpoint}</span>
              </div>
            </div>
            <div className="settings-footer">
              <button
                className="secondary"
                disabled={testingConnection}
                onClick={() => void testConnection()}
              >
                {testingConnection ? "Testing…" : "Test connection"}
              </button>
              <button className="primary" onClick={saveSettings}>
                Save settings
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
