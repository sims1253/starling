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
  type RefinedDraft,
  type RefinedTranscript,
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
  MessagesSquare,
  Mic,
  Radio,
  RefreshCw,
  Settings2,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useRecorder } from "./useRecorder";
import { stoppedTakeVerdict } from "./recorderSession";
import { REFINEMENT_DEFAULT_INSTRUCTION, refineEffect, type RefinementSettings } from "./refine";
import { StreamingDictation, type StreamingState } from "./streamingDictation";
import { finishStreamingTake as finalizeStreamingTake } from "./streamingFinalize";
import { TakeLifecycle, type TakePhase } from "./takeLifecycle";
import { activeThreadId, newThreadId, threadContextBase, threadTurns } from "./threads";
import type { PendingAudioState } from "../electron/ipc.js";

type Connection = "checking" | "ready" | "busy" | "offline";

const DEFAULT_ENDPOINT = window.starlingDesktop ? "http://127.0.0.1:8181" : "/api";

const SETTINGS_FOCUSABLE = "button:not([disabled]), input:not([disabled]), select:not([disabled])";

/**
 * localStorage key for the active-thread hint, the one thread-related value
 * that lives outside the sessions themselves: a stored id pins the thread
 * the next "Refine in thread" joins, a stored empty string is the
 * "start new thread" veto, and an absent key follows the history. Every
 * read and write goes through this constant so the encoding cannot drift
 * between call sites.
 */
const THREAD_ACTIVE_HINT_KEY = "starling:thread:activeHint";

/**
 * Shown when a threaded refine is refused because another turn in the same
 * thread is already refining: the second walk would use the same context and
 * its later save would clobber the thread's text with stale-thread output.
 */
const THREAD_REFINE_BUSY_MESSAGE =
  "Another turn in this thread is already refining. Wait for it to finish, then refine this turn again.";

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

  return `${brief}, ${formatWhen(session.createdAt)}${
    session.threadId ? `, thread ${session.threadId.slice(0, 8)}` : ""
  }`;
}

function messageFrom(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
}

/** The "model, when" attribution stamped on every refined copy. */
function refinedTranscriptStamp(refined: RefinedTranscript, when: string) {
  return `${refined.model}, ${when}`;
}

/** The separator line that appends a refined copy to a text export. */
function refinedTranscriptSeparator(refined: RefinedTranscript, iso: string) {
  return `--- REFINED TRANSCRIPT — ${refinedTranscriptStamp(refined, iso)} ---`;
}

/**
 * The refine action's caption: refining in flight, a stored copy to replace,
 * or the first run. A named branch instead of a nested ternary at the call
 * site, so the drawer JSX stays flat.
 */
function refineActionLabel(refining: boolean, hasRefined: boolean) {
  if (refining) return "Refining…";

  return hasRefined ? "Refine again" : "Refine";
}

/**
 * The threaded refine action's caption: refining in flight, a re-run against
 * the thread's current text, or the first join. Same shape as
 * refineActionLabel so the second drawer button stays as flat as the first.
 */
function refineThreadActionLabel(refining: boolean, threaded: boolean) {
  if (refining) return "Refining…";

  return threaded ? "Refine with thread context" : "Refine in thread";
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

  // Transcript refinement settings: optional, and inert until both a base URL
  // and a model are configured. The raw transcript is never rewritten.
  const [refineBaseUrl, setRefineBaseUrl] = useState(
    () => localStorage.getItem("starling:refine:baseUrl") ?? "",
  );

  const [refineModel, setRefineModel] = useState(
    () => localStorage.getItem("starling:refine:model") ?? "",
  );

  const [refineApiKey, setRefineApiKey] = useState(
    () => localStorage.getItem("starling:refine:apiKey") ?? "",
  );

  const [refineInstruction, setRefineInstruction] = useState(
    () => localStorage.getItem("starling:refine:instruction") ?? "",
  );

  // Multi-turn threads (#117): the active-thread hint is pure UI state, never
  // a session mutation. A stored id pins the thread the next "Refine in
  // thread" joins; a stored empty string is the "start new thread" veto that
  // makes the next press mint a fresh thread; an absent key follows the
  // history — the most recently updated threaded session.
  const [activeThreadHint, setActiveThreadHint] = useState<string | undefined>(() => {
    const stored = localStorage.getItem(THREAD_ACTIVE_HINT_KEY);

    return stored === null ? undefined : stored;
  });

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
  // Takes invalidated by an explicit Discard (B03): a newer take starting no
  // longer cancels an older finalize — only a Discard does — so invalidation
  // is recorded per generation instead of being implied by the sequence head.
  const discardedTakesRef = useRef(new Set<number>());

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

  // Transcript refinement in flight, keyed by session id like activeIds:
  // each take shows its own spinner, and one take refining never disables or
  // animates another. The error and the copied flash are keyed the same way:
  // both belong to the take they came from and never bleed across selections.
  const [refiningIds, setRefiningIds] = useState<ReadonlySet<string>>(() => new Set());
  const refiningIdsRef = useRef(new Set<string>());
  const refineAbortRef = useRef(new Map<string, AbortController>());
  const [refineError, setRefineError] = useState<{ id: string; message: string }>();
  const [copiedRefinedId, setCopiedRefinedId] = useState<string>();
  const copiedRefinedTimerRef = useRef<number | undefined>(undefined);
  // True once the user edits the refinement key field this session. The
  // mount-time decrypt below must never clobber what they typed while it was
  // in flight, so an edited field is off limits to the async load.
  const refineApiKeyEditedRef = useRef(false);
  // Which takes are refining WITH thread context, keyed by session id like
  // refiningIds: each of the drawer's two refine buttons reflects its own
  // take and mode, so a standalone refine on one take never animates the
  // thread button on another.
  const [refiningThreadIds, setRefiningThreadIds] = useState<ReadonlySet<string>>(() => new Set());

  // Threads with a threaded refine in flight. Per-session concurrency is
  // master's design and standalone refinement keeps it; a thread, though,
  // is one running document — two takes refining into it simultaneously
  // would both walk the same context and the later save would clobber the
  // thread's text. The claim set is a ref so a press arriving mid-flight
  // reads the truth without waiting for a re-render.
  const refiningThreadsRef = useRef(new Set<string>());

  // Refinement stays inert until both a base URL and a model are configured.
  const refineConfigured = refineBaseUrl.trim() !== "" && refineModel.trim() !== "";

  // A pin whose thread has no members left is stale residue, not a fallback
  // candidate: it means "no active thread" instead of silently jumping to
  // the derived thread. Only the derived answer lives here; the stored copy
  // is cleaned up by the effect below, because a render-phase removeItem
  // would be a side effect — StrictMode's double render could fire it, and
  // a discarded concurrent render could delete a pin that is valid against
  // fresher data than the render saw.
  const pinnedThreadDangling =
    activeThreadHint !== undefined &&
    activeThreadHint !== "" &&
    !sessions.some((session) => session.threadId === activeThreadHint);

  // The thread the next "Refine in thread" press joins: the pinned hint when
  // its thread still has members, else the most recently updated thread in
  // history. The "start new thread" veto ("") selects neither.
  const activeThread = useMemo(() => {
    if (activeThreadHint === "") return undefined;

    if (pinnedThreadDangling) return undefined;

    if (activeThreadHint !== undefined) return activeThreadHint;

    return activeThreadId(sessions);
  }, [activeThreadHint, pinnedThreadDangling, sessions]);

  // The selected take's position inside its thread, for the drawer's context
  // line. Undefined when the take is unthreaded or not (yet) listed.
  const threadPosition = useMemo(() => {
    if (!selected?.threadId) return undefined;

    const turns = threadTurns(sessions, selected.threadId);
    const index = turns.findIndex((turn) => turn.id === selected.id);

    return index === -1 ? undefined : { turn: index + 1, total: turns.length };
  }, [selected, sessions]);

  // Which of the selected take's two refine actions is running: the busy id
  // set drives both buttons' disabled state, the thread-mode set decides
  // which one spins and says "Refining…".
  const selectedRefining = selected !== undefined && refiningIds.has(selected.id);
  const selectedRefiningThread = selected !== undefined && refiningThreadIds.has(selected.id);

  // Refinements still in flight when the app unmounts must settle without
  // corrupting state: abort each one so its promise rejects and late setState
  // calls become no-ops on the unmounted component — the same guarantee
  // transcribe() relies on through its ref-guarded finally blocks. The copy
  // timer is cleared for the same reason. The ref containers (not their
  // current values) are captured so the cleanup reads the latest entries.
  useEffect(() => {
    const controllers = refineAbortRef;
    const timer = copiedRefinedTimerRef;

    return () => {
      for (const controller of controllers.current.values()) controller.abort();
      window.clearTimeout(timer.current);
    };
  }, []);

  // The residue cleanup for a dangling pin, committed after render: the
  // stored value is removed so a reload follows the history instead of the
  // dead pin, while the memo above already reports "no active thread" so
  // the strip disappears immediately. removeItem is idempotent, so re-runs
  // while the pin stays stale are free; the state value is left alone
  // because nothing react-owned needs to change for the cleanup to hold.
  useEffect(() => {
    if (pinnedThreadDangling) localStorage.removeItem(THREAD_ACTIVE_HINT_KEY);
  }, [pinnedThreadDangling]);

  // Refinement key at rest: prefer the safeStorage ciphertext the main
  // process can decrypt over any plaintext copy. The loaded key only fills
  // a field the user has not edited meanwhile (the decrypt is async and can
  // resolve after they start typing); once it applied, the plaintext copy
  // that predates the encrypted form is residue and is removed.
  useEffect(() => {
    const bridge = window.starlingDesktop;
    const encrypted = localStorage.getItem("starling:refine:apiKeyEnc");

    if (!bridge?.loadRefinementKey || !encrypted) return;

    void bridge
      .loadRefinementKey({ ciphertext: encrypted })
      .then((loaded) => {
        if (!loaded.apiKey) return;

        if (refineApiKeyEditedRef.current) return;
        setRefineApiKey(loaded.apiKey);
        localStorage.removeItem("starling:refine:apiKey");
      })
      .catch(() => {
        /* best-effort: the plaintext fallback key (if any) stays in place */
      });
  }, []);

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

  /**
   * Durably save one finished capture as its own session (B03): once this
   * resolves, the store owns the WAV and the recording lifecycle can be
   * released while transcription runs separately. A storage failure parks
   * the only in-memory copy in unsavedWavs and rejects.
   */
  const saveTake = useCallback(async (wav: Blob, durationMs?: number): Promise<DictationSession> => {
    try {
      return await store.create({ wav, durationMs });
    } catch (caught) {
      setUnsavedWavs((current) => [
        ...current,
        { id: unsavedWavId(), blob: wav, createdAt: new Date().toISOString() },
      ]);
      throw new Error(
        `Local storage failed: ${messageFrom(caught)} Keep this window open and download the unsaved WAV to recover it.`,
      );
    }
  }, []);

  const saveAndTranscribe = useCallback(
    async (wav: Blob, durationMs?: number) => {
      const created = await saveTake(wav, durationMs);

      await refresh();
      await transcribe(created);
    },
    [refresh, saveTake, transcribe],
  );

  /**
   * Apply the active-thread hint everywhere it lives at once: component
   * state for this render, localStorage for the next launch. A thread id is
   * the pin, "" the start-new-thread veto; every writer goes through here so
   * the two stores can never disagree.
   */
  const applyThreadHint = useCallback((value: string) => {
    setActiveThreadHint(value);
    localStorage.setItem(THREAD_ACTIVE_HINT_KEY, value);
  }, []);

  /**
   * Refine one take's raw transcript through the configured OpenAI-compatible
   * endpoint. Explicit and opt-in per press: the result is saved to the
   * session's separate `refined` field, and the raw transcript — the copy the
   * drawer leads with — is never touched, whether refinement succeeds, fails,
   * or is cancelled. Guarded per session id like transcribe(): a second press
   * on the same take while its refinement runs is ignored, while other takes
   * stay free to refine concurrently.
   *
   * With `options.inThread` the press also carries the multi-turn context
   * (#117): thread state is read fresh from the store, an unthreaded take is
   * then assigned — by this explicit action only — to the active thread (or
   * a freshly minted one) and refined against the thread's current text; a
   * take already in a thread skips the assignment and uses the same context.
   * A thread with no refined predecessor sends the standalone pair, so the
   * thread head behaves like a normal refinement while still opening the
   * thread. One threaded refine per thread at a time: a second press on the
   * same thread exits with a message instead of racing the first save.
   */
  const refineTranscript = useCallback(
    async (session: DictationSession, options?: { inThread?: boolean }) => {
      const transcript = session.transcript;
      const inThread = options?.inThread === true;

      if (!transcript || refiningIdsRef.current.has(session.id)) return;

      // Fast exit for the common conflict: the render-time target thread
      // already has a threaded refine in flight. The authoritative claim is
      // made after the fresh listing read below; standalone refinement never
      // touches the claim set and stays fully concurrent.
      if (inThread) {
        const target = session.threadId ?? activeThread;

        if (target !== undefined && refiningThreadsRef.current.has(target)) {
          setRefineError({ id: session.id, message: THREAD_REFINE_BUSY_MESSAGE });

          return;
        }
      }

      refiningIdsRef.current.add(session.id);
      setRefiningIds(new Set(refiningIdsRef.current));
      setRefineError((current) => (current?.id === session.id ? undefined : current));

      if (inThread) {
        setRefiningThreadIds((current) => new Set([...current, session.id]));
      }

      const controller = new AbortController();

      refineAbortRef.current.set(session.id, controller);

      // The thread this press claimed, released in the finally below.
      let claimedThread: string | undefined;
      // Set when THIS press assigned the take to its thread, so a refine
      // failure afterwards can say the join itself succeeded.
      let joinedThread: string | undefined;

      const settings: RefinementSettings = {
        baseUrl: refineBaseUrl,
        model: refineModel,
        apiKey: refineApiKey || undefined,
        instruction: refineInstruction || undefined,
      };

      try {
        let contextText: string | undefined;
        // Captured base identity (B11): the member whose refined text the
        // context came from, recorded on the saved refinement so the base a
        // thread edit used stays explainable instead of being recomputed
        // from whatever order a later listing reads in.
        let contextSourceId: string | undefined;

        if (inThread) {
          // Thread state is read fresh from the store — the same read
          // refresh() uses — never from the render-time closure, which
          // predates every await in this function: cross-window updates and
          // just-written assignments are visible here. The read happens
          // BEFORE this take's own assignment, so a joining take still
          // counts every current member as earlier, whatever its age — the
          // join semantics the context walk documents.
          const fresh = await store.listReport();
          const stored = fresh.sessions.find((candidate) => candidate.id === session.id);
          const threadId = stored?.threadId ?? activeThread ?? newThreadId();

          if (refiningThreadsRef.current.has(threadId)) {
            setRefineError({ id: session.id, message: THREAD_REFINE_BUSY_MESSAGE });

            return;
          }

          refiningThreadsRef.current.add(threadId);
          claimedThread = threadId;

          if (stored?.threadId === undefined) {
            // The explicit assignment: this take joins the thread now, and
            // membership stays visible even when the refinement below fails.
            // Its failure is its own sentence — never a generic refinement
            // error — and nothing else has been touched yet.
            try {
              await store.assignThread(session.id, threadId);
            } catch (caught) {
              setRefineError({
                id: session.id,
                message: `Could not add this take to thread ${threadId.slice(0, 8)}: ${messageFrom(caught)}`,
              });

              return;
            }

            joinedThread = threadId;
            await refresh();
          }

          // Both the join and the already-threaded path pin the hint: the
          // thread this press refined in is the one the next join targets,
          // so a stale pin on another thread can never silently win.
          applyThreadHint(threadId);

          // The context itself comes from the same fresh snapshot: the
          // assignment only labeled THIS take, so the thread's other members
          // — the earlier turns this walks — are exactly what the store
          // holds right now. Because the assignment stamped this take as the
          // thread's latest append, the walk finds the same base here as the
          // pre-assignment read did: a joining take — older or not — refines
          // against the thread's current document, and a retry computes the
          // same answer as the press that joined (B11).
          const base = threadContextBase(fresh.sessions, threadId, session.id);

          contextText = base?.refined?.text;
          contextSourceId = base?.id;
        }

        const text = await Effect.runPromise(
          refineEffect(transcript.text, settings, {
            signal: controller.signal,
            contextText,
          }),
        );

        // The refined copy is drafted mutable so the captured base identity
        // is added only when this refinement actually had one.
        const refined: RefinedDraft = { text, model: refineModel.trim(), createdAt: Date.now() };

        if (contextSourceId !== undefined) refined.contextSourceId = contextSourceId;

        await store.saveRefinedTranscript(session.id, refined);
        await refresh();
      } catch (caught) {
        // A refine failure after a fresh join must not read as though the
        // join failed: the take IS threaded now, and only the refinement
        // needs repeating.
        setRefineError({
          id: session.id,
          message:
            joinedThread !== undefined
              ? `This take joined thread ${joinedThread.slice(0, 8)}, but its refinement failed: ${messageFrom(caught)}`
              : messageFrom(caught),
        });
      } finally {
        refiningIdsRef.current.delete(session.id);
        setRefiningIds(new Set(refiningIdsRef.current));
        refineAbortRef.current.delete(session.id);

        if (claimedThread !== undefined) {
          refiningThreadsRef.current.delete(claimedThread);
        }

        if (inThread) {
          setRefiningThreadIds((current) => {
            const next = new Set(current);

            next.delete(session.id);

            return next;
          });
        }
      }
    },
    [
      activeThread,
      applyThreadHint,
      refresh,
      refineApiKey,
      refineBaseUrl,
      refineInstruction,
      refineModel,
    ],
  );

  /**
   * Clear the active-thread hint so the next "Refine in thread" starts a
   * fresh thread. Pure UI state: no session is deleted, mutated, or
   * reassigned — existing threads and their takes stay exactly as they are.
   */
  function startNewThread() {
    applyThreadHint("");
  }

  async function copyRefinedText() {
    if (!selected?.refined) return;

    try {
      await navigator.clipboard.writeText(selected.refined.text);
      setCopiedRefinedId(selected.id);
      window.clearTimeout(copiedRefinedTimerRef.current);
      copiedRefinedTimerRef.current = window.setTimeout(() => setCopiedRefinedId(undefined), 1400);
    } catch (caught) {
      setRefineError({
        id: selected.id,
        message: `Could not copy the refined transcript: ${messageFrom(caught)}`,
      });
    }
  }

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
    // still awaiting storage must not install its controller after this,
    // and a finalize still running must write nothing further. A newer take
    // starting does not invalidate an older finalize (B03); only this
    // explicit discard does.
    discardedTakesRef.current.add(takeSeqRef.current);
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
   * Finalize the streamed take this Stop was issued against. The caller
   * captures the take generation before any await — a close-guard Discard
   * offered while the stop itself is still running already invalidated it —
   * and the finalize re-validates it around each durable write, so a Discard
   * racing the finalize wins: the stale take writes nothing and leaves the
   * Discard's state untouched (#160). Returns false when the stream was
   * never usable and the caller should save the recorder's own capture via
   * the batch path.
   *
   * `releaseCapture` is invoked the moment the journal is durably owned by
   * its session, before the transcript work: the stop transition ends there
   * so the next take can start while transcription is in flight (B03).
   */
  const finishStreamingTake = useCallback(
    async (
      stream: StreamingDictation,
      durationMs: number,
      generation: number,
      releaseCapture: () => void,
    ): Promise<boolean> => {
      streamRef.current = undefined;
      const bailed = streamBailedRef.current;
      streamBailedRef.current = false;

      // The take identity this Stop was issued against. Only an explicit
      // Discard invalidates it (B03): a newer take starting while this
      // finalize still runs no longer cancels its writes. The close-guard
      // Discard still wins its races (#160).
      const isCurrentTake = () => !discardedTakesRef.current.has(generation);

      if (bailed) {
        setPartialText(undefined);
        setStreamStatus(undefined);
        await stream.abandon();
        // The journal is gone; the recorder's memory-only capture carries
        // this take through the batch path, so it is no longer journaled.
        setStreamingFinalize(false);

        return false;
      }

      setStreamingFinalize(true);

      try {
        const finalized = await finalizeStreamingTake(
          {
            stream,
            durationMs,
            isCurrentTake,
            parkUnsavedWav,
            refresh,
            setSelectedId: (id) => setSelectedId(id),
            setConnectionReady: () => setConnection("ready"),
            transcribe,
            onDurableSave: releaseCapture,
          },
          store,
        );

        if (finalized.discarded) return true;

        return !finalized.batchFallback;
      } finally {
        // A newer take may own the live-stream state by now (B03): only the
        // take this finalize belonged to may clear it.
        if (takeSeqRef.current === generation) {
          setPartialText(undefined);
          setStreamStatus(undefined);
          setStreamingFinalize(false);
        }
      }
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

      // The take identity this Stop was issued against, captured before any
      // await: a close-guard Discard offered while the stop itself is still
      // running invalidates exactly this generation (#160, B03).
      const generation = takeSeqRef.current;

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

      // Ends the capture transition the moment this take's audio is durably
      // owned by its session (B03): the streamed journal's commit or the
      // batch store.create. Transcription keeps running in the background,
      // so a new take may start while it is in flight. Released at most
      // once: a stop that already handed off must never end a newer take's
      // transition from its finally.
      let released = false;

      const releaseCapture = () => {
        if (released) return;

        released = true;
        lifecycle.endStop();
        setTakePhase("idle");
        setFinalizing(false);
        // The session row (or the parked unsaved WAV) owns durability now,
        // so the close-guard mirror no longer needs the journaled flag.
        setStreamingFinalize(false);
      };

      try {
        const capture = await stop();

        // Any capture with samples is a real take, however short (B02): a
        // one-letter answer stays reviewable and retryable, and only an
        // empty capture — no samples at all — is an accidental activation.
        const verdict = stoppedTakeVerdict(capture && { sampleCount: capture.audio.samples.length });

        if (!capture || !verdict.keep) {
          await discardStreamingTake();
          throw new Error("No microphone audio was captured.");
        }

        if (
          stream &&
          (await finishStreamingTake(stream, capture.durationMs, generation, releaseCapture))
        ) {
          return;
        }

        const prepared = await prepareWav16k(capture.audio);
        const created = await saveTake(prepared.blob, capture.durationMs);
        await refresh();

        // Durable: the session store owns the WAV. Release the capture
        // lifecycle and let the transcription finish in the background
        // (B03); the upload indicator tracks this session alone.
        releaseCapture();
        void transcribe(created);
      } catch (caught) {
        setError(messageFrom(caught));
      } finally {
        // Covers every path that never reached a durable save (failed
        // starts of the finalize, storage failures, empty captures).
        releaseCapture();
      }

      return;
    }

    // The startup sweep may still be deciding which journals and attempts
    // are abandoned; a take started now could race it (#144).
    if (!startupRecoveryDoneRef.current) {
      setError("Still restoring recordings from a previous session. Try again in a moment.");

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
    refresh,
    saveTake,
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

    // The raw transcript stays first and intact; a refined copy, when one
    // exists, is appended under a clear separator instead of replacing it.
    const refined = selected.refined;

    const text = refined
      ? `${selected.transcript.text}\n\n${refinedTranscriptSeparator(refined, new Date(refined.createdAt).toISOString())}\n\n${refined.text}\n`
      : selected.transcript.text;

    const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));

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
  // rescues audio that survived the damage so it can be downloaded. Entries
  // are matched by position because best-effort ids are not unique — several
  // "(unknown id)" entries can share one label with different raw keys.
  function exportDamagedAudio(index: number) {
    const entry = damaged[index];

    if (!entry) return;
    const wav = invalidSessionWav(entry);

    if (wav) {
      const url = URL.createObjectURL(wav);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `starling-damaged-${entry.id.replace(/[^a-z0-9-]/gi, "").slice(0, 8)}.wav`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
    }
  }

  // Dismiss one quarantined entry by its raw record key, then refresh so the
  // banner drops it without a reload. Confirm first: the damaged record may
  // hold the only copy of audio that survived (#155).
  async function dismissDamagedEntry(index: number) {
    const entry = damaged[index];

    if (!entry || entry.key === undefined) return;

    if (
      !window.confirm(
        "Delete this damaged recording? It cannot be read, and deleting it is permanent. Download its WAV first if you want to keep the audio.",
      )
    )
      return;

    try {
      await store.deleteInvalid(entry.key);
      await refresh();
    } catch (caught) {
      setError(`Could not delete the damaged recording: ${messageFrom(caught)}`);
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

  /**
   * Persist the refinement API key. With the desktop bridge and OS-backed
   * encryption available, only safeStorage ciphertext is written and any
   * plaintext copy is removed. Without either (browser preview, or a host
   * with no keychain backend), the key falls back to plaintext localStorage —
   * a documented, deliberate fallback, never a silent one.
   *
   * The lifecycle is symmetric on every path: whenever a plaintext key is
   * written, the ciphertext entry is removed too — otherwise the next mount
   * would decrypt the stale ciphertext over the newer plaintext. Saving an
   * empty key clears both entries: there is nothing to encrypt and no
   * residue to keep.
   */
  async function persistRefinementKey(apiKey: string): Promise<void> {
    if (apiKey === "") {
      localStorage.removeItem("starling:refine:apiKey");
      localStorage.removeItem("starling:refine:apiKeyEnc");

      return;
    }

    const bridge = window.starlingDesktop;

    if (!bridge?.storeRefinementKey) {
      localStorage.setItem("starling:refine:apiKey", apiKey);
      localStorage.removeItem("starling:refine:apiKeyEnc");

      return;
    }

    try {
      const stored = await bridge.storeRefinementKey({ apiKey });

      if (stored.ciphertext === null) {
        localStorage.setItem("starling:refine:apiKey", apiKey);
        localStorage.removeItem("starling:refine:apiKeyEnc");

        return;
      }

      localStorage.setItem("starling:refine:apiKeyEnc", stored.ciphertext);
      localStorage.removeItem("starling:refine:apiKey");
    } catch {
      localStorage.setItem("starling:refine:apiKey", apiKey);
      localStorage.removeItem("starling:refine:apiKeyEnc");
    }
  }

  async function saveSettings() {
    const clean = draftEndpoint.trim().replace(/\/$/, "");

    if (!clean) return;
    setEndpoint(clean);
    localStorage.setItem("starling:endpoint", clean);
    localStorage.setItem("starling:protocol", protocol);
    localStorage.setItem("starling:model", model);
    localStorage.setItem("starling:streaming", streamLive ? "1" : "0");
    localStorage.setItem("starling:terms", expectedTerms);
    localStorage.setItem("starling:refine:baseUrl", refineBaseUrl.trim());
    localStorage.setItem("starling:refine:model", refineModel.trim());
    localStorage.setItem("starling:refine:instruction", refineInstruction);

    try {
      await persistRefinementKey(refineApiKey);
    } catch (caught) {
      // The fire-and-forget click must not swallow this: persistence can
      // fail without the bridge too (quota, private mode).
      setError(`Could not save the refinement API key: ${messageFrom(caught)}`);
    }

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
          {activeThread && (
            <div className="thread-strip">
              <span>Active thread: {activeThread.slice(0, 8)}</span>
              <button onClick={startNewThread}>start new thread</button>
            </div>
          )}
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
                    Damaged entries are kept until you delete them; the list below shows every
                    recording that is still readable.
                  </small>
                  <div className="recovery-actions">
                    {damaged.map((entry, index) => {
                      const wav = invalidSessionWav(entry);

                      return (
                        <span key={`${entry.id}-${index}`} className="damaged-entry">
                          {wav ? (
                            <button
                              className="recover-audio"
                              onClick={() => exportDamagedAudio(index)}
                            >
                              Download WAV {index + 1}
                            </button>
                          ) : null}
                          {entry.key !== undefined ? (
                            <button
                              className="recover-audio dismiss-damaged"
                              onClick={() => void dismissDamagedEntry(index)}
                              aria-label={`Delete damaged recording ${index + 1}`}
                            >
                              Delete entry {index + 1}
                            </button>
                          ) : null}
                        </span>
                      );
                    })}
                  </div>
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
                  {session.threadId && (
                    <span className="thread-badge">thread {session.threadId.slice(0, 8)}</span>
                  )}
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
          {selected.transcript && (
            <section className="refined-block" aria-label="Refined transcript">
              <div className="refined-head">
                <div>
                  <p className="eyebrow">REFINED TRANSCRIPT</p>
                  <span>
                    {selected.refined
                      ? `${refinedTranscriptStamp(selected.refined, formatWhen(new Date(selected.refined.createdAt).toISOString()))} — kept beside the raw transcript above`
                      : "Optional LLM cleanup, run on demand. The raw transcript above never changes."}
                  </span>
                </div>
                <div className="transcript-actions">
                  <button
                    onClick={() => void refineTranscript(selected)}
                    disabled={selectedRefining || !refineConfigured}
                  >
                    {selectedRefining && !selectedRefiningThread ? (
                      <LoaderCircle className="spinning" size={16} />
                    ) : (
                      <Sparkles size={16} />
                    )}
                    {refineActionLabel(
                      selectedRefining && !selectedRefiningThread,
                      selected.refined !== undefined,
                    )}
                  </button>
                  <button
                    onClick={() => void refineTranscript(selected, { inThread: true })}
                    disabled={selectedRefining || !refineConfigured}
                  >
                    {selectedRefiningThread ? (
                      <LoaderCircle className="spinning" size={16} />
                    ) : (
                      <MessagesSquare size={16} />
                    )}
                    {refineThreadActionLabel(
                      selectedRefiningThread,
                      selected.threadId !== undefined,
                    )}
                  </button>
                  {selected.refined && (
                    <button
                      disabled={refiningIds.has(selected.id)}
                      onClick={() => void copyRefinedText()}
                    >
                      {copiedRefinedId === selected.id ? (
                        <Check size={16} />
                      ) : (
                        <Clipboard size={16} />
                      )}
                      {copiedRefinedId === selected.id ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>
              </div>
              {selected.threadId && threadPosition && (
                <p className="thread-line">
                  Thread {selected.threadId.slice(0, 8)} — turn {threadPosition.turn} of{" "}
                  {threadPosition.total}. Each turn keeps its own raw transcript.
                </p>
              )}
              {refineError?.id === selected.id && (
                <p className="refined-error" role="alert">
                  <CircleAlert size={15} /> {refineError.message}
                </p>
              )}
              {!refineConfigured && (
                <p className="refined-hint">
                  Add a refinement base URL and model in settings to enable this.
                </p>
              )}
              {selectedRefining && !selected.refined && (
                <p className="refined-status">
                  <LoaderCircle className="spinning" size={15} />{" "}
                  {selectedRefiningThread
                    ? "Sending the thread's current text and this new turn to the refinement model…"
                    : "Sending the raw transcript to the refinement model…"}
                </p>
              )}
              {selected.refined && <p className="refined-text">{selected.refined.text}</p>}
            </section>
          )}
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
            {/* The head stays fixed above the scrolling body: the close
                button lives in its zone, so scrolled settings never run
                under it, and the body region scrolls while the card stays
                viewport-bounded with the footer always reachable. */}
            <div className="settings-head">
              <p className="eyebrow">CONNECTION</p>
              <h2 id="settings-title">Transcription server</h2>
            </div>
            <div className="settings-body">
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
                    API format; recordings always save locally first and fall back to a full upload
                    if the stream fails.
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
              <div className="settings-section">
                <p className="eyebrow">TRANSCRIPT REFINEMENT</p>
                <p className="settings-section-lede">
                  Optional: press Refine on a finished take to send its raw transcript to an
                  OpenAI-compatible chat endpoint. The refined copy is stored and labeled
                  separately; the raw transcript is never rewritten.
                </p>
                <label>
                  Base URL
                  <input
                    value={refineBaseUrl}
                    onChange={(event) => setRefineBaseUrl(event.target.value)}
                    placeholder="http://127.0.0.1:11434/v1"
                  />
                  <small>
                    Include the version path the server needs, for example /v1 for Ollama or
                    https://api.openai.com/v1.
                  </small>
                </label>
                <div className="settings-grid">
                  <label>
                    Refinement model
                    <input
                      value={refineModel}
                      onChange={(event) => setRefineModel(event.target.value)}
                      placeholder="llama3.1"
                    />
                  </label>
                  <label>
                    API key (optional)
                    <input
                      type="password"
                      value={refineApiKey}
                      onChange={(event) => {
                        setRefineApiKey(event.target.value);
                        // Typed input outranks the async decrypt from mount.
                        refineApiKeyEditedRef.current = true;
                      }}
                      placeholder="Local servers need none"
                    />
                    <small>
                      Stored encrypted via your OS keychain in the desktop app when available; the
                      browser preview keeps it in local storage as plaintext.
                    </small>
                  </label>
                </div>
                <label>
                  Instruction
                  <textarea
                    rows={4}
                    value={refineInstruction}
                    onChange={(event) => setRefineInstruction(event.target.value)}
                    placeholder={REFINEMENT_DEFAULT_INSTRUCTION}
                  />
                  <small>
                    Leave empty to use the default shown here: light cleanup that keeps the wording,
                    meaning, language, and order.
                  </small>
                </label>
              </div>
              <div className="settings-callout">
                <span className={`status-dot ${connection}`} />
                <div>
                  <strong>
                    {connection === "ready" ? "Server connected" : "Server needs attention"}
                  </strong>
                  <span>{endpoint}</span>
                </div>
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
              <button className="primary" onClick={() => void saveSettings()}>
                Save settings
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
