import { Effect, Match } from "effect";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeTranscript,
  encodePcm16kMono,
  IndexedDbSessionStore,
  invalidSessionWav,
  prepareWav16k,
  StarlingClient,
  type DictationSession,
  type InvalidStoredSession,
  type RefinedDraft,
  type ServerHealth,
  type TranscriptionProtocol,
  type TranscriptionResult,
} from "@starling/dictation";
import {
  BarChart3,
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
import {
  StreamingDictation,
  type StreamingState,
  type StreamingTransport,
} from "./streamingDictation";
import { createStreamingTransport } from "./streamTransport";
import { finishStreamingTake as finalizeStreamingTake } from "./streamingFinalize";
import { TakeLifecycle, type TakePhase } from "./takeLifecycle";
import { SessionDeleteDialog, deletionWarning } from "./sessionDeletion";
import {
  attemptProvenanceLabel,
  canTranscribeAgain,
  refinedTranscriptStamp,
  sessionTitle,
  transcriptExportText,
  transcribeAgainLabel,
} from "./transcriptAttempts";
import { activeThreadId, newThreadId, threadContextBase, threadTurns } from "./threads";
import {
  CheckSequencer,
  connectionFailureMessage,
  probeOutcomeFromFailure,
  probeOutcomeFromHealth,
  settingsCalloutView,
  type ConnectionProbe,
} from "./connectionProbe";
import {
  normalizeSettings,
  persistSettings,
  readCommittedSettings,
  refinementKeyPlan,
  type SecureKeyStorage,
  type SettingsSnapshot,
} from "./settingsTransaction";
import type { PendingAudioState } from "../electron/ipc.js";
import { InsightsView } from "./insights/InsightsView";
import { InsightRecorder, wavCaptureStats } from "./insights/insightEmitter";
import { IndexedDbInsightEventStore, type InsightEvent } from "./insights/insightEvents";

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

/**
 * Local insight events (E29): a dedicated IndexedDB log, separate from
 * session audio and transcripts. Events carry counts and constrained tokens
 * only — never text, selections, paths or secrets — and recording one must
 * never break the action it describes: every emit below is fire-and-forget
 * with failures surfaced as an Insights notice instead of a take error.
 */
const insights = new InsightRecorder(new IndexedDbInsightEventStore());

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

function historyRowLabel(session: DictationSession) {
  const title = sessionTitle(session);
  const codePoints = Array.from(title);
  const brief = codePoints.length > 60 ? `${codePoints.slice(0, 60).join("")}…` : title;

  return `${brief}, ${formatWhen(session.createdAt)}${
    session.threadId ? `, thread ${session.threadId.slice(0, 8)}` : ""
  }`;
}

function messageFrom(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
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

/**
 * Ask the desktop bridge how the host's secret store received the key (B10).
 * Resolves null when there is nothing to ask — an empty key, no bridge (the
 * browser preview), or an older preload without the channel — and never
 * throws: an IPC failure reads as a failed store, so the caller's decision
 * always runs on classified information.
 */
async function resolveSecureKeyStorage(apiKey: string): Promise<SecureKeyStorage | null> {
  const bridge = window.starlingDesktop;

  if (apiKey === "" || !bridge?.storeRefinementKey) return null;

  try {
    const saved = await bridge.storeRefinementKey({ apiKey });

    // A null ciphertext with a claimed "encrypted" status would violate the
    // bridge contract; reading it as a failed store keeps the decision honest.
    return saved.ciphertext === null
      ? {
          kind: saved.protection === "encrypted" ? "failed" : saved.protection,
          backend: saved.backend,
        }
      : { kind: "encrypted", ciphertext: saved.ciphertext };
  } catch {
    return { kind: "failed" };
  }
}

export default function App() {
  // Committed settings (B06): one decoder reads every key, and the values
  // only ever change as a whole configuration when a settings save lands.
  const [initialSettings] = useState(() => readCommittedSettings(localStorage, DEFAULT_ENDPOINT));

  const [endpoint, setEndpoint] = useState(initialSettings.endpoint);

  const [protocol, setProtocol] = useState<TranscriptionProtocol>(initialSettings.protocol);

  const [model, setModel] = useState(initialSettings.model);

  const [streamLive, setStreamLive] = useState(initialSettings.streamLive);

  const [connection, setConnection] = useState<Connection>("checking");
  const [serverModel, setServerModel] = useState("server");
  const [sessions, setSessions] = useState<DictationSession[]>([]);
  const [damaged, setDamaged] = useState<readonly InvalidStoredSession[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [activeIds, setActiveIds] = useState<ReadonlySet<string>>(() => new Set());
  const activeUploadsRef = useRef(new Set<string>());
  const [error, setError] = useState<string>();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  // The Insights surface (E29): a view toggle over the same app — no router,
  // the events the surface aggregates, and the one storage/recording notice
  // it can show. Events refresh after each emit, so opening the view later
  // always reads the current population.
  const [view, setView] = useState<"capture" | "insights">("capture");

  const [insightEvents, setInsightEvents] = useState<readonly InsightEvent[]>(() => []);

  const [insightsIssue, setInsightsIssue] = useState<string>();

  // Post-Stop timestamps for takes whose first transcript has not settled
  // yet, keyed by session id and consumed by that first recognition: the
  // measured Stop-press-to-ready wait. A "Transcribe again" press finds no
  // entry, and its wait is honestly unknown (null) instead of guessed.
  const stopWaitStartsRef = useRef(new Map<string, number>());
  // A streamed take's session id exists only once the durable save hands it
  // over; the Stop timestamp waits here until onDurableSave names the session.
  const pendingStreamStopAtRef = useRef<number | null>(null);

  const [expectedTerms, setExpectedTerms] = useState(initialSettings.expectedTerms);

  // Transcript refinement settings: optional, and inert until both a base URL
  // and a model are configured. The raw transcript is never rewritten.
  const [refineBaseUrl, setRefineBaseUrl] = useState(initialSettings.refineBaseUrl);

  const [refineModel, setRefineModel] = useState(initialSettings.refineModel);

  const [refineApiKey, setRefineApiKey] = useState(initialSettings.refineApiKey);

  const [refineInstruction, setRefineInstruction] = useState(initialSettings.refineInstruction);

  // The explicit choice to store the refinement key as plaintext when no OS
  // secret store can encrypt it (B10). Off by default: without it, a key
  // that cannot be encrypted is kept for the session only.
  const [refineKeyPlaintextOptIn, setRefineKeyPlaintextOptIn] = useState(
    initialSettings.refineKeyPlaintextOptIn,
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

  // The settings dialog's own state (B06), all of it discarded with the
  // dialog: the complete draft configuration (defined only while the dialog
  // is open), the isolated Test Connection probe, and the inline validation
  // or save error. None of it can outlive Cancel, Escape, or an outside
  // click, and none of it is committed state.
  const [draft, setDraft] = useState<SettingsSnapshot>();

  const [probe, setProbe] = useState<ConnectionProbe>();

  const [settingsIssue, setSettingsIssue] = useState<string>();

  // Post-save key-protection status (B10): set when a save succeeded but the
  // key ended up session-only, so the dialog stays open and the accurate
  // per-save protection status is read in context. Cleared with the dialog.
  const [settingsNotice, setSettingsNotice] = useState<string>();

  // Live connection checks and dialog probes are sequenced separately: a
  // probe may never be silenced by a background live check or vice versa,
  // but within each family only the newest may report (B06).
  const [healthSequencer] = useState(() => new CheckSequencer());

  const [probeSequencer] = useState(() => new CheckSequencer());

  /**
   * Close the settings dialog and discard the draft (B06). Cancel, Escape,
   * and an outside click all land here: every field edit lived in the draft
   * alone, so committed settings and their storage entries were never
   * touched and there is nothing to undo. In-flight probes are retired with
   * the dialog so a late result cannot land in the next dialog session.
   */
  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    setDraft(undefined);
    setProbe(undefined);
    setSettingsIssue(undefined);
    setSettingsNotice(undefined);
    probeSequencer.cancelAll();
  }, [probeSequencer]);

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

  // Deletion confirmation state (B05): the dialog is the only path to
  // store.delete() for a saved recording. The guard instance sequences
  // request/confirm/cancel; deletePendingId is its render mirror, so the
  // modal appears and disappears with the guard's own truth.
  const [deleteDialog] = useState(() => new SessionDeleteDialog());
  const [deletePendingId, setDeletePendingId] = useState<string | undefined>(undefined);
  const deleteCancelRef = useRef<HTMLButtonElement>(null);
  const deleteDialogRef = useRef<HTMLElement>(null);
  const trashButtonRef = useRef<HTMLButtonElement>(null);

  /** Close the confirmation without deleting anything (Cancel or Escape). */
  const cancelDeleteSession = useCallback(() => {
    deleteDialog.cancel();
    setDeletePendingId(undefined);
  }, [deleteDialog]);

  // Identity of the take being started, captured before any await; bumped
  // whenever the current take is invalidated so late work disposes itself.
  const takeSeqRef = useRef(0);
  // Takes invalidated by an explicit Discard (B03): a newer take starting no
  // longer cancels an older finalize — only a Discard does — so invalidation
  // is recorded per generation instead of being implied by the sequence head.
  const discardedTakesRef = useRef(new Set<number>());

  const selected = sessions.find((session) => session.id === selectedId);

  // The take the deletion confirmation targets, while its dialog is open. A
  // take removed from another window mid-dialog makes this undefined and the
  // dialog simply closes: nothing is left to confirm (B05).
  const deleteTarget =
    deletePendingId === undefined
      ? undefined
      : sessions.find((session) => session.id === deletePendingId);

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

  // The refinement key this app knows is durably at rest, and in which form
  // (B10): seeded from the plaintext entry a legacy version may have left,
  // replaced when the mount-time decrypt applies the encrypted copy, and
  // updated by every save that actually persisted a key. A key typed into
  // the dialog is compared against this: unchanged means the stored copy
  // already speaks for it, changed means a failed secure save must block
  // rather than strand a mismatched endpoint/key pair.
  const retainedKeyRef = useRef<{ key: string; form: "encrypted" | "plaintext" } | undefined>(
    initialSettings.refineApiKey === ""
      ? undefined
      : { key: initialSettings.refineApiKey, form: "plaintext" },
  );

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
  // resolve after they start typing); a settings draft already open gets the
  // same value patched in, so saving the dialog cannot clobber the just
  // decrypted key with the stale empty field it was copied from. Once it
  // applied, the plaintext copy that predates the encrypted form is residue
  // and is removed.
  useEffect(() => {
    const bridge = window.starlingDesktop;
    const encrypted = localStorage.getItem("starling:refine:apiKeyEnc");

    if (!bridge?.loadRefinementKey || !encrypted) return;

    void bridge
      .loadRefinementKey({ ciphertext: encrypted })
      .then((loaded) => {
        const apiKey = loaded.apiKey;

        if (!apiKey) return;

        // The encrypted copy is now the durable record of this key (B10);
        // the decrypt also patches any open draft so a save cannot clobber
        // the just-decrypted key with the stale field it was copied from.
        retainedKeyRef.current = { key: apiKey, form: "encrypted" };

        if (refineApiKeyEditedRef.current) return;
        setRefineApiKey(apiKey);
        setDraft((current) =>
          current === undefined ? current : { ...current, refineApiKey: apiKey },
        );
        localStorage.removeItem("starling:refine:apiKey");
      })
      .catch(() => {
        /* best-effort: the plaintext fallback key (if any) stays in place */
      });
  }, []);

  const busy = activeIds.size > 0;

  // The insight event log loads once, best-effort: a damaged log becomes an
  // Insights notice, never a startup failure (dictation works without it).
  useEffect(() => {
    void insights
      .load()
      .then(() => setInsightEvents(insights.snapshot()))
      .catch((caught) =>
        setInsightsIssue(`Insights could not open the local event log: ${messageFrom(caught)}`),
      );
  }, []);

  /**
   * Record one insight event without ever blocking the action it describes
   * (E29): the emit is fire-and-forget, load() inside is idempotent so the
   * mirror exists before sequence numbers are derived, a failure lands in
   * the Insights notice instead of the take's flow, and the mirror refreshes
   * on success so the Insights view reads the new event without a reopen.
   */
  const recordInsight = useCallback((action: () => Promise<void>) => {
    void insights
      .load()
      .then(action)
      .then(() => setInsightEvents(insights.snapshot()))
      .catch((caught) =>
        setInsightsIssue(`Insights could not record an event: ${messageFrom(caught)}`),
      );
  }, []);

  /**
   * Emit recognition_selected for a settled transcript (E29): word counts
   * only, with the measured post-Stop wait when this is the take's first
   * transcript and null when it is not (a retranscription happens at
   * leisure; its wait is unknown, which suppresses the typing-time proxy
   * rather than fabricating one).
   */
  const recordRecognitionSelected = useCallback(
    (sessionId: string, transcriptText: string) => {
      const startedAt = stopWaitStartsRef.current.get(sessionId);

      stopWaitStartsRef.current.delete(sessionId);
      recordInsight(() =>
        insights.recognitionSelected({
          captureId: sessionId,
          transcriptText,
          postStopReadyMs: startedAt === undefined ? null : Date.now() - startedAt,
        }),
      );
    },
    [recordInsight],
  );

  /**
   * Emit capture_finalized once a take's audio is durably owned by a session
   * (E29): sample frames are read from the canonical WAV itself, and
   * complete_audio is true on every path that reaches here — a take with
   * incomplete audio is parked or discarded before a session owns it.
   */
  const recordCaptureFinalized = useCallback(
    (session: DictationSession) => {
      recordInsight(async () => {
        const stats = await wavCaptureStats(session.wav);

        await insights.captureFinalized({
          captureId: session.id,
          sampleCount: stats.sampleCount,
          sampleRate: stats.sampleRate,
          completeAudio: true,
        });
      });
    },
    [recordInsight],
  );

  /** Reset the insight event log; the surface starts over empty. */
  const resetInsights = useCallback(() => {
    void insights
      .reset()
      .then(() => setInsightEvents(insights.snapshot()))
      .catch((caught) =>
        setInsightsIssue(`Insights could not reset the event log: ${messageFrom(caught)}`),
      );
  }, []);

  // The dialog's status line (B06): the probe's own outcome while one is
  // running or has settled, else the committed endpoint's live status.
  const settingsCallout = settingsCalloutView(probe, connection, endpoint);

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

  /** Shared health-check transport: the desktop bridge, else the browser client. */
  const runHealthCheck = useCallback(
    (target: string, targetProtocol: TranscriptionProtocol): Promise<ServerHealth> => {
      const bridge = window.starlingDesktop;

      return bridge
        ? bridge.health({ endpoint: target, protocol: targetProtocol })
        : Effect.runPromise(
            new StarlingClient({ baseUrl: target, protocol: targetProtocol }).healthEffect(),
          );
    },
    [],
  );

  /**
   * Check the health of the COMMITTED endpoint and protocol (B06). The probe
   * behind Test Connection never routes through here: it owns its own
   * outcome, so a draft endpoint's failure cannot paint the live connection
   * offline. Each check claims a sequence token before awaiting anything, so
   * a slower check against an endpoint that has since been replaced is
   * dropped instead of overwriting the newer status.
   */
  const checkHealth = useCallback(() => {
    const token = healthSequencer.begin();

    return runHealthCheck(endpoint, protocol).then(
      (health) => {
        if (!healthSequencer.isCurrent(token)) return;

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
      (failure) => {
        if (!healthSequencer.isCurrent(token)) return;

        setConnection("offline");
        // Validation reasons (bad scheme, embedded credentials) stay
        // verbatim; transport failures name the endpoint actually checked.
        setError(connectionFailureMessage(endpoint, failure));
      },
    );
  }, [endpoint, healthSequencer, protocol, runHealthCheck]);

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

  // The deletion confirmation (B05): focus lands on Cancel, never on the
  // destructive button, so keyboard activation of the trash button cannot
  // roll straight into a confirmed delete — Enter alone opens and then
  // cancels. Escape closes, Tab stays inside the dialog, and focus returns
  // to the trash button that opened it.
  useEffect(() => {
    if (deletePendingId === undefined) return;

    deleteCancelRef.current?.focus();
    const trash = trashButtonRef.current;

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        cancelDeleteSession();

        return;
      }

      if (event.key !== "Tab") return;

      const items = Array.from(
        deleteDialogRef.current?.querySelectorAll<HTMLElement>(SETTINGS_FOCUSABLE) ?? [],
      );

      if (items.length === 0) return;

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
      trash?.focus();
    };
  }, [cancelDeleteSession, deletePendingId]);

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

        // The settling save records this attempt's provenance (B04): the
        // model and protocol are stamped beside the transcript, so when a
        // later "Transcribe again" supersedes it, the history entry that
        // keeps it stays explainable.
        await store.saveTranscript(session.id, result, {
          model: model.trim() || undefined,
          protocol,
        });
        recordRecognitionSelected(session.id, result.text);
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
    [endpoint, model, protocol, recordRecognitionSelected, refresh],
  );

  /**
   * Durably save one finished capture as its own session (B03): once this
   * resolves, the store owns the WAV and the recording lifecycle can be
   * released while transcription runs separately. A storage failure parks
   * the only in-memory copy in unsavedWavs and rejects.
   */
  const saveTake = useCallback(
    async (wav: Blob, durationMs?: number): Promise<DictationSession> => {
      try {
        const created = await store.create({ wav, durationMs });

        // A durable take now exists for Insights (E29); the emit is
        // fire-and-forget so insights can never break the capture path.
        recordCaptureFinalized(created);

        return created;
      } catch (caught) {
        setUnsavedWavs((current) => [
          ...current,
          { id: unsavedWavId(), blob: wav, createdAt: new Date().toISOString() },
        ]);
        throw new Error(
          `Local storage failed: ${messageFrom(caught)} Keep this window open and download the unsaved WAV to recover it.`,
        );
      }
    },
    [recordCaptureFinalized],
  );

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
        // One authored revision exists now (E29): change counts come from
        // the word-level diff between raw and refined text, and are never
        // labeled corrected errors.
        recordInsight(() =>
          insights.transformationCompleted({
            captureId: session.id,
            rawText: transcript.text,
            revisedText: text,
          }),
        );
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
      recordInsight,
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

      let transport: StreamingTransport;

      try {
        // The packaged app streams over the main process's native bridge;
        // the browser preview opens the renderer socket its CSP permits
        // (loopback or the page-origin ws proxy) — same interface (B01).
        transport = createStreamingTransport(endpoint);
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
            onStreamedSettled: (session, transcript) =>
              recordRecognitionSelected(session.id, transcript.text),
          },
          store,
        );

        // A streamed take's journal became its session (E29): the take now
        // exists for Insights. A discarded or batch-fallback take never
        // reaches here — the batch path's own save emits instead.
        if (finalized.session !== undefined) {
          recordCaptureFinalized(finalized.session);
        }

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
    [parkUnsavedWav, recordCaptureFinalized, recordRecognitionSelected, refresh, transcribe],
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

      // When this Stop was pressed (E29): the origin of the post-Stop wait
      // the first recognition for this take measures. A streamed take knows
      // its journal's session id up front; the batch path registers the id
      // once the session exists.
      const stopPressedAt = Date.now();

      if (stream !== undefined) pendingStreamStopAtRef.current = stopPressedAt;

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

      const releaseCapture = (session?: DictationSession) => {
        if (released) return;

        if (session && pendingStreamStopAtRef.current !== null) {
          stopWaitStartsRef.current.set(session.id, pendingStreamStopAtRef.current);
          pendingStreamStopAtRef.current = null;
        }

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
        const verdict = stoppedTakeVerdict(
          capture && { sampleCount: capture.audio.samples.length },
        );

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
        stopWaitStartsRef.current.set(created.id, stopPressedAt);
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
    transcribe,
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

    // The raw transcript stays first and intact; earlier recognition
    // attempts and any refined copy are appended under their own separators
    // instead of replacing it, so every transcript version leaves with the
    // same session (B04).
    const text = transcriptExportText(selected);

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

  /**
   * Open the deletion confirmation (B05) — the trash button never deletes
   * straight away. A take whose transcription or refinement is still in
   * flight is refused with the reason; anything else waits for the dialog's
   * explicit confirm.
   */
  function requestDeleteSession(session: DictationSession) {
    const intent = deleteDialog.request(session.id, {
      transcribing: activeUploadsRef.current.has(session.id),
      refining: refiningIdsRef.current.has(session.id),
    });

    if (intent.kind === "blocked") {
      setError(intent.message);

      return;
    }

    setDeletePendingId(intent.id);
  }

  /**
   * The one path from the confirmation to the destructive write: the id
   * handoff is exactly-once, so a double activation cannot re-delete, and
   * the dialog is dismissed before the first await.
   */
  async function confirmDeleteSession() {
    const id = deleteDialog.confirm();

    setDeletePendingId(undefined);

    if (id !== undefined) await removeSession(id);
  }

  /**
   * Delete one saved recording. Reached only through the confirmation
   * dialog's explicit confirm (B05); a failed delete leaves the recording
   * and its versions intact and surfaces the error instead.
   */
  async function removeSession(id: string) {
    if (activeUploadsRef.current.has(id)) return;

    try {
      await store.delete(id);
      await refresh();
    } catch (caught) {
      setError(`Could not delete the recording: ${messageFrom(caught)}`);
    }
  }

  /**
   * Open the settings dialog on a complete draft of the committed
   * configuration (B06). Every field the dialog shows is drafted — endpoint,
   * protocol, model, streaming, terms, and every refinement field — and the
   * draft exists only while the dialog is open.
   */
  function openSettings() {
    probeSequencer.cancelAll();
    setProbe(undefined);
    setSettingsIssue(undefined);
    setDraft({
      endpoint,
      protocol,
      model,
      streamLive,
      expectedTerms,
      refineBaseUrl,
      refineModel,
      refineApiKey,
      refineInstruction,
      refineKeyPlaintextOptIn,
    });
    setSettingsIssue(undefined);
    setSettingsNotice(undefined);
    setSettingsOpen(true);
  }

  /**
   * Test the DRAFT configuration without touching committed state (B06). The
   * probe runs against the draft endpoint AND the draft protocol — the
   * combination about to be saved — and its outcome lands in the dialog's
   * own probe state: the live connection status, the server model, the
   * committed model, and the global error banner are never written from
   * here. Only the newest probe may report, so a slow earlier probe cannot
   * overwrite a newer result.
   */
  async function testConnection() {
    if (!draft) return;

    const normalized = normalizeSettings(draft);

    if (!normalized.ok) {
      setProbe({
        state: "done",
        outcome: { state: "failed", endpoint: draft.endpoint, message: normalized.reason },
      });

      return;
    }

    const target = normalized.settings.endpoint;
    const token = probeSequencer.begin();

    setProbe({ state: "testing", endpoint: target });

    try {
      const health = await runHealthCheck(target, draft.protocol);

      if (!probeSequencer.isCurrent(token)) return;

      setProbe({ state: "done", outcome: probeOutcomeFromHealth(target, health) });
    } catch (failure) {
      if (!probeSequencer.isCurrent(token)) return;

      setProbe({ state: "done", outcome: probeOutcomeFromFailure(target, failure) });
    }
  }

  /**
   * Apply the whole draft as one transaction (B06): validate the complete
   * configuration, ask the secret store how it could hold the refinement key
   * (B10), decide that key's destination, persist every settings key as an
   * all-or-nothing storage transition — with rollback on failure — and only
   * then swap the committed React state to the same normalized snapshot. A
   * failed or blocked save leaves the dialog open with a clear error and
   * nothing partially applied; a save whose key stayed session-only keeps
   * the dialog open with the accurate protection status; the health-check
   * effect re-probes on its own when the committed endpoint or protocol
   * actually changed.
   */
  async function saveSettings() {
    if (!draft) return;

    const normalized = normalizeSettings(draft);

    if (!normalized.ok) {
      setSettingsIssue(normalized.reason);

      return;
    }

    setSettingsIssue(undefined);

    // The secret store runs before any storage write: by the time the
    // transaction starts, the key's destination is already decided, so the
    // transaction is the only writer and a failure cannot strand a
    // half-persisted key. Plaintext is written only on the explicit opt-in
    // (B10); without it an unencryptable key stays session-only or blocks
    // the save outright.
    const secure = await resolveSecureKeyStorage(normalized.settings.refineApiKey);

    const plan = refinementKeyPlan({
      apiKey: normalized.settings.refineApiKey,
      secure,
      plaintextOptIn: normalized.settings.refineKeyPlaintextOptIn,
      retained: retainedKeyRef.current,
    });

    if (plan.outcome === "blocked") {
      setSettingsIssue(plan.message);

      return;
    }

    const persisted = persistSettings(normalized.settings, plan, localStorage);

    if (!persisted.ok) {
      setSettingsIssue(
        `Could not save settings: ${persisted.message} Nothing was changed — try again, or cancel to keep the current settings.`,
      );

      return;
    }

    const committed = normalized.settings;

    // Committed state lands only after storage did: every setter receives
    // the same normalized snapshot the transaction wrote, so state and
    // storage activate this configuration together or not at all.
    setEndpoint(committed.endpoint);
    setProtocol(committed.protocol);
    setModel(committed.model);
    setStreamLive(committed.streamLive);
    setExpectedTerms(committed.expectedTerms);
    setRefineBaseUrl(committed.refineBaseUrl);
    setRefineModel(committed.refineModel);
    setRefineApiKey(committed.refineApiKey);
    setRefineInstruction(committed.refineInstruction);
    setRefineKeyPlaintextOptIn(committed.refineKeyPlaintextOptIn);

    // The durable record of the key follows what the save actually wrote:
    // encrypted and opt-in plaintext install a new durable copy, clearing
    // removes it, and a session-only save stored nothing — the plan already
    // guaranteed the untouched entries still speak for this same key.
    if (plan.outcome === "encrypted")
      retainedKeyRef.current = { key: committed.refineApiKey, form: "encrypted" };
    else if (plan.outcome === "plaintext")
      retainedKeyRef.current = { key: committed.refineApiKey, form: "plaintext" };
    else if (plan.outcome === "cleared") retainedKeyRef.current = undefined;

    if (plan.outcome === "session-only") {
      // Saved, but the key is not protected at rest: keep the dialog open so
      // the status is read where the choice can be changed (B10).
      setSettingsNotice(plan.status);
    } else {
      closeSettings();
    }

    // Re-check health from the saved configuration, as saves always did. The
    // call's closure may predate the commit above: when the endpoint or
    // protocol changed it probes the older pair, and the effect that fires
    // on the new checkHealth identity supersedes it via the sequence token —
    // only the check against the committed values survives.
    void checkHealth();
  }

  /** Patch fields of the open draft; the committed configuration is untouched. */
  function updateDraft(patch: Partial<SettingsSnapshot>) {
    setDraft((current) => (current === undefined ? current : { ...current, ...patch }));
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
                  <strong>{sessionTitle(session)}</strong>
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
              {canTranscribeAgain(selected.status, activeIds.has(selected.id)) && (
                <button onClick={() => void transcribe(selected)}>
                  <RefreshCw size={16} /> {transcribeAgainLabel(selected.status)}
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
                ref={trashButtonRef}
                className="danger"
                disabled={activeIds.has(selected.id)}
                onClick={() => requestDeleteSession(selected)}
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
            {(selected.transcriptHistory?.length ?? 0) > 0 && (
              <section className="attempt-history" aria-label="Earlier transcription attempts">
                <p className="attempt-history-head">
                  {selected.transcriptHistory?.length === 1
                    ? "1 earlier attempt — kept when a new one succeeds"
                    : `${selected.transcriptHistory?.length} earlier attempts — kept when a new one succeeds`}
                </p>
                <ul>
                  {(selected.transcriptHistory ?? []).map((attempt, index) => (
                    <li key={index}>
                      <small>{attemptProvenanceLabel(attempt, formatWhen)}</small>
                      <span>{attempt.text || <em>empty transcript</em>}</span>
                    </li>
                  ))}
                </ul>
              </section>
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

      {deleteTarget && (
        <div
          className="modal-layer"
          onMouseDown={(event) => event.target === event.currentTarget && cancelDeleteSession()}
        >
          <section
            ref={deleteDialogRef}
            className="settings-card confirm-card"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="delete-confirm-title"
            aria-describedby="delete-confirm-warning"
          >
            <div className="settings-head">
              <p className="eyebrow">DELETE RECORDING</p>
              <h2 id="delete-confirm-title">Delete this saved recording?</h2>
            </div>
            <div className="settings-body">
              <p id="delete-confirm-warning" className="confirm-warning">
                {deletionWarning(deleteTarget)}
              </p>
            </div>
            <div className="settings-footer">
              {/* Focus starts on Cancel (the effect above), so the
                  destructive button is never the default action. */}
              <button className="secondary" ref={deleteCancelRef} onClick={cancelDeleteSession}>
                Cancel
              </button>
              <button className="danger" onClick={() => void confirmDeleteSession()}>
                Delete permanently
              </button>
            </div>
          </section>
        </div>
      )}

      {settingsOpen && draft !== undefined && (
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
                  value={draft.endpoint}
                  onChange={(event) => updateDraft({ endpoint: event.target.value })}
                  placeholder="http://127.0.0.1:8181"
                />
              </label>
              <div className="settings-grid">
                <label>
                  API format
                  <select
                    value={draft.protocol}
                    onChange={(event) =>
                      updateDraft({
                        protocol: event.target.value === "openai" ? "openai" : "starling",
                      })
                    }
                  >
                    <option value="starling">Starling native</option>
                    <option value="openai">OpenAI compatible</option>
                  </select>
                </label>
                <label>
                  Model
                  <input
                    value={draft.model}
                    onChange={(event) => updateDraft({ model: event.target.value })}
                    placeholder="parakeet"
                  />
                </label>
              </div>
              <label className="settings-check">
                <input
                  type="checkbox"
                  checked={draft.streamLive}
                  disabled={draft.protocol === "openai"}
                  onChange={(event) => updateDraft({ streamLive: event.target.checked })}
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
                  value={draft.expectedTerms}
                  onChange={(event) => updateDraft({ expectedTerms: event.target.value })}
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
                    value={draft.refineBaseUrl}
                    onChange={(event) => updateDraft({ refineBaseUrl: event.target.value })}
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
                      value={draft.refineModel}
                      onChange={(event) => updateDraft({ refineModel: event.target.value })}
                      placeholder="llama3.1"
                    />
                  </label>
                  <label>
                    API key (optional)
                    <input
                      type="password"
                      value={draft.refineApiKey}
                      onChange={(event) => {
                        updateDraft({ refineApiKey: event.target.value });
                        // Typed input outranks the async decrypt from mount.
                        refineApiKeyEditedRef.current = true;
                      }}
                      placeholder="Local servers need none"
                    />
                    <small>
                      Stored encrypted via your OS keychain when the desktop app has one. Without
                      one — the browser preview, or Linux without a secret store — it is kept for
                      the current session only unless you explicitly choose plaintext storage below.
                    </small>
                  </label>
                </div>
                <label className="settings-check">
                  <input
                    type="checkbox"
                    checked={draft.refineKeyPlaintextOptIn}
                    onChange={(event) =>
                      updateDraft({ refineKeyPlaintextOptIn: event.target.checked })
                    }
                  />
                  <span>
                    Store API key unencrypted
                    <small>
                      Not recommended. Tick this to keep the key in plain local storage when no OS
                      keychain is available. Leave it off and an unencryptable key is used for this
                      session only, never written to disk.
                    </small>
                  </span>
                </label>
                <label>
                  Instruction
                  <textarea
                    rows={4}
                    value={draft.refineInstruction}
                    onChange={(event) => updateDraft({ refineInstruction: event.target.value })}
                    placeholder={REFINEMENT_DEFAULT_INSTRUCTION}
                  />
                  <small>
                    Leave empty to use the default shown here: light cleanup that keeps the wording,
                    meaning, language, and order.
                  </small>
                </label>
              </div>
              {/* The status line belongs to the probe once one ran (B06): a
                  draft endpoint's test result never paints the committed
                  connection, and without a probe the committed status shows. */}
              <div className="settings-callout" role="status">
                <span className={`status-dot ${settingsCallout.dot}`} />
                <div>
                  <strong>{settingsCallout.title}</strong>
                  <span>{settingsCallout.detail}</span>
                </div>
              </div>
            </div>
            {settingsIssue && (
              <p className="settings-issue" role="alert">
                <CircleAlert size={15} /> {settingsIssue}
              </p>
            )}
            {settingsNotice && (
              <p className="settings-notice" role="status">
                <Check size={15} /> {settingsNotice}
              </p>
            )}
            <div className="settings-footer">
              <button className="secondary" onClick={closeSettings}>
                Cancel
              </button>
              <button
                className="secondary"
                disabled={probe?.state === "testing"}
                onClick={() => void testConnection()}
              >
                {probe?.state === "testing" ? "Testing…" : "Test connection"}
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
