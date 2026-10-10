//! The StarlingApp entity: state, behaviors, and async flows ported from
//! `apps/desktop/src/App.tsx`.

use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use gpui::{
    AppContext, ClipboardItem, Context, Entity, FocusHandle, Focusable, Pixels, Render,
    Subscription, Task, Timer, Window, div, prelude::*,
};
use starling_dictation::{
    client::{self, StarlingClient},
    engine::{Backend, EngineConfig, EngineManager},
    fft,
    fidelity::{self, TranscriptAnalysisOptions},
    playback::{platform_backend, PlaybackAttenuation, PlaybackLease, PlaybackNotice},
    player::Player,
    recorder::RecorderHandle,
    settings::{
        ActivationMode, DictationSettings, EngineMode, EngineSettings, FeedbackSettings,
        InsertionSettings, LivePreviewSettings, MicrophoneSettings, OverlayMode, PlaybackMode,
        PlaybackSettings, ProcessingSettings, Settings, DEFAULT_SHORTCUT,
    },
    storage::{DamagedRecord, ListedRecord, SessionSummary},
};

use crate::{input::TextField, store::Store, theme, upload::refresh_sessions, views};
use crate::processing::{self, Providers, TakeProcessing};
use starling_processing::staging::Draft;
use starling_processing::CancelToken;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Connection {
    Checking,
    Ready,
    Busy,
    Offline,
}

/// Why a draft cannot be probed or saved at all: there is no endpoint to
/// connect to (`EMPTY_ENDPOINT_REASON` in `settingsTransaction.ts`).
pub(crate) const EMPTY_ENDPOINT_REASON: &str = "Enter a server endpoint before saving.";

/// Latest-wins sequencing for asynchronous connection checks (#207,
/// ported from `connectionProbe.ts`'s `CheckSequencer`): every check
/// claims a token before awaiting anything and may report its outcome
/// only while its token is still the newest. A slower, older check —
/// against an endpoint that has since been replaced —
/// resolves after a newer one and is dropped instead of overwriting its
/// result. `cancel_all` retires every in-flight token at once, so
/// closing the settings dialog leaves a stray probe with nothing to land
/// in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CheckSequencer {
    latest: u64,
}

impl CheckSequencer {
    pub(crate) fn new() -> Self {
        Self { latest: 0 }
    }

    /// Claim the right to report; the returned token is current until the
    /// next `begin`.
    pub(crate) fn begin(&mut self) -> u64 {
        self.latest = self.latest.wrapping_add(1);
        self.latest
    }

    pub(crate) fn is_current(&self, token: u64) -> bool {
        token == self.latest
    }

    /// Retire every in-flight token.
    pub(crate) fn cancel_all(&mut self) {
        self.latest = self.latest.wrapping_add(1);
    }
}

impl Default for CheckSequencer {
    fn default() -> Self {
        Self::new()
    }
}

/// The isolated outcome of one Test Connection press (#207, B06): it
/// belongs to the settings dialog alone — the live connection status, the
/// server model, and the global error banner are never written from a
/// probe. Transport-failure messages carry the URL that was actually
/// tried, so a failed probe names the draft endpoint, not the committed
/// one (the port's `connectionFailureMessage` is already built in).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProbeOutcome {
    Ok {
        model: String,
        busy: bool,
    },
    Failed {
        message: String,
    },
}

/// One Test Connection press: in flight against `endpoint`, or settled
/// with its own outcome for that endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConnectionProbe {
    Testing {
        endpoint: String,
    },
    Done {
        endpoint: String,
        outcome: ProbeOutcome,
    },
}

/// Why a live health check is running, and therefore what it may write
/// (#207). The probe behind Test Connection never routes through here:
/// it owns its own outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HealthCheckPurpose {
    /// Startup and settings-save checks against the committed endpoint: they
    /// own the badge, the server model, the model auto-sync (R02), and
    /// the global error banner.
    Live,
    /// The R13 re-check after a transport-class job failure: it owns the
    /// badge and the server model only — never the banner, which still
    /// carries the job's explanation for the missing transcript.
    Diagnostic,
}

/// The probe outcome for a health snapshot (`probeOutcomeFromHealth`).
pub(crate) fn probe_outcome_from_health(health: &client::ServerHealth) -> ProbeOutcome {
    ProbeOutcome::Ok {
        model: health.model.as_deref().unwrap_or("server").to_string(),
        busy: health_is_busy(health),
    }
}

/// Whether a health snapshot describes a busy server (#207): the busy
/// flag or a queued request. Shared by the live-check writes and the
/// probe outcome so the two derivations cannot drift.
pub(crate) fn health_is_busy(health: &client::ServerHealth) -> bool {
    health.busy.unwrap_or(false) || health.queue_depth.unwrap_or(0.0) > 0.0
}

/// What the settings dialog's status line shows (#207, B06): a probe that
/// ran owns the line — testing, its own outcome, its own endpoint — and
/// only without one does the line fall back to the live status of the
/// committed endpoint. A failed draft probe therefore never paints the
/// committed connection as offline.
pub(crate) struct SettingsCalloutView {
    /// The status dot's connection state: the probe's when one ran, the
    /// live one otherwise.
    pub dot: Connection,
    pub title: String,
    /// The endpoint probed, or the probe's failure message — never the
    /// live endpoint mixed in.
    pub detail: String,
}

pub(crate) fn settings_callout_view(
    probe: Option<&ConnectionProbe>,
    live: Connection,
    live_endpoint: &str,
) -> SettingsCalloutView {
    match probe {
        None => SettingsCalloutView {
            dot: live,
            title: match live {
                Connection::Ready => "Server connected".to_string(),
                // A live check is still in flight (startup, or the
                // re-check a save just triggered): pending, not failing —
                // the callout must not read "needs attention" the moment
                // the dialog opens on a fresh launch (#207 review).
                Connection::Checking => "Checking server…".to_string(),
                Connection::Busy | Connection::Offline => {
                    "Server needs attention".to_string()
                }
            },
            detail: live_endpoint.to_string(),
        },
        Some(ConnectionProbe::Testing { endpoint }) => SettingsCalloutView {
            dot: Connection::Checking,
            title: "Testing connection…".to_string(),
            detail: endpoint.clone(),
        },
        Some(ConnectionProbe::Done {
            outcome: ProbeOutcome::Failed { message },
            ..
        }) => SettingsCalloutView {
            dot: Connection::Offline,
            title: "Probe failed".to_string(),
            detail: message.clone(),
        },
        Some(ConnectionProbe::Done {
            endpoint,
            outcome: ProbeOutcome::Ok { model, busy },
        }) => SettingsCalloutView {
            dot: if *busy {
                Connection::Busy
            } else {
                Connection::Ready
            },
            title: if *busy {
                format!("{model} is working")
            } else {
                format!("{model} responded")
            },
            detail: endpoint.clone(),
        },
    }
}

/// What a settled live health check may write back to app state (#207):
/// pure, so the banner/ownership contract stays testable without a
/// window. Every field except `connection` is optional — `None` leaves
/// the current value untouched.
pub(crate) struct LiveCheckWrites {
    /// The badge always follows the check's outcome.
    pub connection: Connection,
    /// `Some(model)` on success (with the `server` fallback), `None` on
    /// failure: a failed check leaves the previous model label alone.
    pub server_model: Option<String>,
    /// The R02 model auto-sync: `Some(model)` only for a Live OpenAI
    /// check that reported a non-empty model while the model is not
    /// user-set.
    pub model_sync: Option<String>,
    /// The global banner slot: `Some(None)` clears it (a Live success),
    /// `Some(Some(message))` replaces it (a Live failure), `None` leaves
    /// whatever is already there — Diagnostic checks, whose R13 probe
    /// must not wipe the job's failure explanation.
    pub error_slot: Option<Option<String>>,
}

pub(crate) fn live_check_writes(
    purpose: HealthCheckPurpose,
    user_set_model: bool,
    result: &Result<client::ServerHealth, String>,
) -> LiveCheckWrites {
    match result {
        Ok(health) => LiveCheckWrites {
            connection: if health_is_busy(health) {
                Connection::Busy
            } else {
                Connection::Ready
            },
            server_model: Some(health.model.as_deref().unwrap_or("server").to_string()),
            // The guard and the value are one expression: the model is
            // resolved once, with no dead fallback for the case the
            // filter already excluded (#207 review).
            model_sync: if purpose == HealthCheckPurpose::Live
                && !user_set_model
            {
                health
                    .model
                    .as_deref()
                    .filter(|model| !model.is_empty())
                    .map(str::to_string)
            } else {
                None
            },
            error_slot: (purpose == HealthCheckPurpose::Live).then_some(None),
        },
        Err(message) => LiveCheckWrites {
            connection: Connection::Offline,
            server_model: None,
            model_sync: None,
            error_slot: (purpose == HealthCheckPurpose::Live)
                .then(|| Some(message.clone())),
        },
    }
}

/// The endpoint a draft would commit (#207): trimmed with trailing
/// slashes removed — the same normalization `save_settings` applies, so
/// the probe targets what saving this draft would commit.
pub(crate) fn normalize_draft_endpoint(draft: &str) -> String {
    draft.trim().trim_end_matches('/').to_string()
}

/// Validate the endpoint a save would commit (#213): takes the
/// already-normalized draft (see [`normalize_draft_endpoint`]) and refuses
/// it when empty — the Electron reference's `normalizeSettings` refusal,
/// same wording — or when the client's own `cleanEndpoint` rules reject
/// it (http/https only, no embedded credentials). A value the client
/// cannot use must never reach disk: it would re-offline the app on
/// every restart with the "Invalid server endpoint." banner until the
/// user revisited settings.
pub(crate) fn validated_draft_endpoint(clean: &str) -> Result<(), String> {
    if clean.is_empty() {
        return Err(EMPTY_ENDPOINT_REASON.to_string());
    }
    client::clean_endpoint(clean)
        .map_err(|err| err.to_string())
        .map(|_| ())
}

/// Decide one Test Connection press (#207 review) before any request
/// fires: dropped while the newest probe is still Testing (the disabled
/// button, enforced at the state layer as well — and a dropped press
/// retires nothing); otherwise it runs, claiming the sequencer token —
/// which retires any in-flight probe — and normalizing the draft
/// endpoint exactly like a save. An empty endpoint comes back as-is for
/// the caller to settle as the probe's own failure under the claimed
/// token, so the probe this press interrupted still has nowhere to land.
pub(crate) fn probe_press(
    sequencer: &mut CheckSequencer,
    current: Option<&ConnectionProbe>,
    draft_endpoint: &str,
) -> Option<(u64, String)> {
    if matches!(current, Some(ConnectionProbe::Testing { .. })) {
        return None;
    }
    let token = sequencer.begin();
    Some((token, normalize_draft_endpoint(draft_endpoint)))
}

pub struct UnsavedWav {
    pub id: String,
    pub wav: Arc<Vec<u8>>,
    pub created_at: String,
}

pub struct StarlingApp {
    /// The recording store: storage v2, unconditionally (D14 — there is
    /// no backend choice and no fallback). `None` only when the store
    /// could not be opened at startup; the error is surfaced honestly and
    /// new takes are kept in memory until it is fixed and the app is
    /// restarted.
    pub store: Option<Store>,
    pub store_error: Option<String>,
    pub player: Option<Player>,
    pub root_focus: FocusHandle,

    /// The bundled-engine supervisor (#362, #363): present exactly while
    /// the engine mode is builtin and a manager could be constructed.
    /// Everything it does is observable through `snapshot()`; the app
    /// owns starting/stopping it with the mode and shutting it down at
    /// quit.
    pub engine: Option<EngineManager>,
    /// Why no manager exists at all — the one failure `EngineManager`'s
    /// snapshot cannot express (`default_paths` failed before a manager
    /// could start). Surfaced like any engine failure.
    pub(crate) engine_startup_error: Option<String>,
    /// The committed engine settings: mode, the persisted active model,
    /// and the backend override. `active_model` follows the engine
    /// snapshot (the engine is the source of truth) and is persisted
    /// through [`StarlingApp::persist_committed_settings`].
    pub(crate) engine_settings: EngineSettings,
    /// The backend override the running manager was started or toggled
    /// with, so Save only restarts the engine when the override actually
    /// changed — never on every save.
    pub(crate) applied_backend_override: Option<String>,
    /// Bumped on every manager start/stop so the notifier loop can tell
    /// its manager from a replacement (a mode switch) and retire itself.
    pub(crate) engine_instance: u64,
    /// The settings dialog's engine drafts (#362): mode and backend
    /// override are Save-saved like the other fields; every other engine
    /// control (activate/download/...) is an immediate action on the
    /// manager and never lives here.
    pub(crate) draft_engine_mode: EngineMode,
    pub(crate) draft_backend_override: Option<String>,
    /// The committed microphone choice, and the dialog's draft of it
    /// (the microphone check runs on the draft, so a choice can be tried
    /// before it is saved).
    pub(crate) microphone_settings: MicrophoneSettings,
    pub(crate) draft_microphone: Option<String>,
    pub(crate) mic: crate::mic::MicState,
    /// Keeps the quit hook (engine shutdown) registered for the entity's
    /// lifetime — a dropped `Subscription` unsubscribes.
    quit_hook: Option<Subscription>,
    /// The current recording's endpoint/model binding (#363): resolved
    /// when recording starts, taken when it stops, and carried with the
    /// take so a model switch mid-take cannot move it.
    pub(crate) active_take: Option<crate::upload::TakeTarget>,

    pub endpoint: String,
    pub model: String,
    pub expected_terms_input: String,
    pub user_set_model: bool,

    pub settings_open: bool,
    pub draft_endpoint: Entity<TextField>,
    pub draft_model: Entity<TextField>,
    pub draft_terms: Entity<TextField>,

    /// Text processing after transcription (#295): the committed
    /// settings, the providers built from them, each take's processing
    /// view and staged draft, the running job per take, and when each
    /// recent take stopped (for stop-to-processed latency).
    pub processing_settings: ProcessingSettings,
    pub(crate) providers: Providers,
    pub(crate) processing: HashMap<String, TakeProcessing>,
    pub(crate) drafts: HashMap<String, Draft>,
    pub(crate) processing_jobs: HashMap<String, (String, CancelToken)>,
    pub(crate) stop_instants: HashMap<String, Instant>,
    /// Takes whose processing state is being loaded.
    pub(crate) processing_loading: HashSet<String>,
    /// The settings dialog's processing drafts (committed on save).
    pub draft_mode: String,
    pub draft_s1_endpoint: Entity<TextField>,
    pub draft_api_endpoint: Entity<TextField>,
    pub draft_api_model: Entity<TextField>,
    pub draft_api_key_env: Entity<TextField>,

    pub connection: Connection,
    pub server_model: String,
    /// The settings dialog's own Test Connection probe (#207, B06): the
    /// newest press's isolated state, or `None` when no probe has run —
    /// the callout then falls back to the live status of the committed
    /// endpoint. Live connection state is never written from here.
    pub(crate) probe: Option<ConnectionProbe>,
    /// Latest-wins sequencing for live health checks (#207): startup,
    /// settings-save, and R13 diagnostic re-checks compete here; only the
    /// newest may write connection state.
    pub(crate) health_sequencer: CheckSequencer,
    /// Latest-wins sequencing for the dialog's Test Connection probes
    /// (#207): a probe may never be silenced by a live check or vice
    /// versa, but within the family only the newest may report.
    pub(crate) probe_sequencer: CheckSequencer,

    /// History as metadata-only summaries (G02): the listing never loads
    /// audio; play/export/retry fetch one recording's WAV on demand.
    pub sessions: Vec<SessionSummary>,
    /// Records the store flagged as damaged (G02), shown in history with
    /// their reason and never deleted.
    pub damaged: Vec<DamagedRecord>,
    pub selected_id: Option<String>,
    pub active_ids: HashSet<String>,
    /// The take whose "Retry with" choices the drawer shows (#356).
    pub(crate) retry_menu: Option<String>,
    /// A retry waiting for the engine to switch to its model (#356).
    pub(crate) pending_retry: Option<crate::upload::PendingRetry>,
    /// Bumped by every retry request: only the newest acts once its
    /// audio has loaded.
    pub(crate) retry_seq: u64,
    pub error: Option<String>,
    pub capture_warning: Option<String>,
    /// Ephemeral one-off export notice (G05: a renamed export is surfaced,
    /// never silently written next to the file it dodged). Owns its own
    /// slot so a later capture warning cannot overwrite it mid-read, and
    /// vice versa; auto-clears on the same timer pattern as the other
    /// transient flags.
    pub export_notice: Option<String>,
    pub unsaved: Vec<UnsavedWav>,
    /// The armed discard confirmation (#214.4): the batch token (see
    /// [`unsaved_batch_token`]) of the unsaved batch the user armed
    /// discarding. Scoped to the exact batch on purpose — see
    /// [`discard_intent`].
    pub confirm_discard_for: Option<u64>,
    /// Ephemeral drawer flags, scoped to the take they were earned on
    /// (#214.2): the id of the session whose transcript was copied / whose
    /// WAV was saved, so the flag can never bleed onto another selection.
    pub copied: Option<String>,
    pub wav_saved: Option<String>,
    /// The armed delete confirmation (B05, #208): the id of the take whose
    /// trash button was clicked once. Only a second click on that same
    /// take's button deletes — see [`delete_intent`].
    pub confirm_delete_id: Option<String>,
    /// Sessions with a delete job in flight (#214.3): a delete click while
    /// the store operation is still running is ignored instead of spawning
    /// a second job that would race the first into a spurious `NotFound`.
    /// An id is released only by a listing that no longer contains it (see
    /// [`surviving_deletes`]) — or by the job's own failure, so a failed
    /// delete stays retryable.
    pub deleting_ids: HashSet<String>,
    /// The recording shortcut's activation machine (#221): presses,
    /// releases, the record button and Escape all go through it, and it
    /// owns the one-take-at-a-time and readiness rules.
    pub(crate) activation: crate::activation::Activation,
    /// The take the running recorder belongs to (set only once its start
    /// succeeded), so effects for any other take are ignored.
    pub(crate) recording_take: Option<crate::activation::TakeId>,
    /// The committed dictation settings and the shortcut they name.
    pub(crate) dictation_settings: DictationSettings,
    pub(crate) shortcut: crate::shortcut::Shortcut,
    /// A shortcut saved while a take was running: it takes over when the
    /// take ends, so the held key's release still finishes that take.
    pub(crate) pending_shortcut: Option<crate::shortcut::Shortcut>,
    /// The window's focus changes, oldest first, and the observer that
    /// records them (see `activation::focused_at`).
    pub(crate) window_focus: Vec<(Instant, bool)>,
    pub(crate) focus_observer: Option<Subscription>,
    /// The system-wide registrations (absent in tests and where the
    /// platform offers none), and how registering the shortcut went.
    pub(crate) global_shortcuts: Option<crate::shortcut::GlobalShortcuts>,
    pub(crate) shortcut_registration: Result<(), String>,
    /// Native Wayland's system-wide source: the desktop's GlobalShortcuts
    /// portal (absent outside a Wayland session), and the Linux setup
    /// check shown under Settings → Dictation.
    pub(crate) portal_shortcuts: Option<crate::portal::PortalShortcuts>,
    pub(crate) system_check: crate::system_check::SystemCheck,
    /// Keeps the in-window shortcut interceptor registered.
    pub(crate) key_interceptor: Option<Subscription>,
    /// What happened to a take that was cancelled (Escape, a microphone
    /// that never delivered audio): shown until dismissed.
    pub(crate) take_notice: Option<String>,
    /// Takes startup recovery brought back (#356): shown until dismissed.
    pub(crate) recovery_notice: Option<String>,
    /// The latest playback-attenuation notice. Its own slot: the take
    /// lifecycle clears `error` and `take_notice` on every start/stop.
    pub(crate) playback_notice: Option<PlaybackNotice>,
    /// The latest correction-decision write; each write awaits the one
    /// before it so decisions land in the order they were made.
    pub(crate) correction_chain: Option<Task<()>>,
    /// The settings dialog's dictation drafts (committed on save).
    pub(crate) draft_shortcut: Entity<TextField>,
    pub(crate) draft_activation: ActivationMode,
    pub(crate) draft_double_tap: bool,
    pub(crate) dictation_draft_error: Option<String>,
    pub(crate) draft_playback_mode: PlaybackMode,
    pub(crate) draft_lower_level: Entity<crate::slider::LevelSlider>,
    /// Playback attenuation during takes (#361); `playback_lease` is the
    /// live take's.
    pub(crate) playback_settings: PlaybackSettings,
    pub(crate) playback: PlaybackAttenuation,
    pub(crate) playback_lease: Option<PlaybackLease>,
    /// The overlay and start/stop cues (#221): the committed settings,
    /// the overlay window, which cues may still play, and the dialog's
    /// drafts.
    pub(crate) feedback: FeedbackSettings,
    pub(crate) overlay: crate::overlay::Overlay,
    pub(crate) cue_gate: crate::cues::CueGate,
    pub(crate) draft_overlay_mode: OverlayMode,
    pub(crate) draft_cues: bool,
    pub(crate) draft_cue_volume: Entity<crate::slider::LevelSlider>,
    /// History audio compression and retention (#342).
    pub(crate) audio_upkeep: crate::upkeep::AudioUpkeep,
    /// Typing finished takes into the window they were dictated into
    /// (#221), and the settings dialog's draft of its settings.
    pub(crate) delivery: crate::delivery::DeliveryState,
    pub(crate) draft_insertion: InsertionSettings,
    /// Whether this session types where the target cannot be verified
    /// (Wayland's virtual keyboard), asked in the background whenever the
    /// settings dialog opens.
    pub(crate) insertion_unverifiable: bool,
    pub playing_id: Option<String>,
    /// Identifies the current playback so poll-watchers can detect that they
    /// are stale (G04). Bumped whenever playback starts, stops, or is
    /// replaced; watchers capture the value at spawn time.
    pub playback_generation: u64,

    pub recorder: Option<RecorderHandle>,
    /// The take's live stream worker (#357) and its generation, which
    /// previews still arriving from an earlier take's worker do not match.
    pub(crate) stream_pump: Option<crate::stream_pump::StreamPump<crate::live_stream::LiveStream>>,
    pub(crate) stream_generation: u64,
    pub(crate) stream_trace: Option<std::sync::Arc<crate::stream_pump::StreamTrace>>,
    /// The preview cadence takes ask `/stream` for (#357).
    pub(crate) live_preview: LivePreviewSettings,
    pub(crate) draft_live_preview: LivePreviewSettings,
    /// The live line a direct-mode take shows while recording.
    pub(crate) live_partial: String,
    /// The staging panel (#297) and stagings whose transcript is still
    /// pending after the panel moved on.
    pub(crate) staging: Option<crate::staging::Staging>,
    pub(crate) background_stagings: Vec<crate::staging::Staging>,
    pub(crate) next_staging_token: u64,
    /// A focus change asked for outside a render (which has the window).
    /// The latest request wins: retiring a panel asks for the root, and
    /// the panel that replaces it in the same frame asks for its editor.
    pub(crate) pending_focus: Option<PendingFocus>,
    /// Set when live streaming is unavailable or died mid-recording: the
    /// partial view going quiet must be explainable, so it shows while the
    /// take records (the worker reports a mid-take death as it happens)
    /// and the stop path folds it into the capture warning shown beside
    /// the saved take.
    pub(crate) stream_degradation: Option<String>,
    pub levels: Vec<f32>,
    pub elapsed_ms: f64,

    pub diagnostics: Option<(Instant, bool)>,
}

/// Where the next render moves keyboard focus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PendingFocus {
    /// The visible staging panel's editor (the root when there is none).
    Staging,
    Root,
}

/// What a playback poll-watcher does after one tick (G04).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaybackWatch {
    /// The watched playback is still live: sleep and poll again.
    Poll,
    /// The watched playback drained naturally: the watcher clears
    /// `playing_id`, notifies, and exits.
    Finished,
    /// The watcher is stale or playback already ended by another path
    /// (stop, replacement, selection change, session deletion): exit
    /// without touching state.
    Cancelled,
}

/// Pure tick decision for a watcher that captured `watched` when its
/// playback started, against the app's current playback state.
///
/// `playing` is `playing_id.is_some()`; `player_is_playing` is `None` when
/// no player exists. Staleness is decided first, so a superseded watcher can
/// never report [`PlaybackWatch::Finished`] — and therefore never clear
/// `playing_id` — for a newer playback, even one that has already drained.
pub(crate) fn playback_watch(
    watched: u64,
    current: u64,
    playing: bool,
    player_is_playing: Option<bool>,
) -> PlaybackWatch {
    if watched != current {
        return PlaybackWatch::Cancelled;
    }

    if !playing {
        return PlaybackWatch::Cancelled;
    }

    match player_is_playing {
        Some(true) => PlaybackWatch::Poll,
        // Natural drain, or the player vanished mid-playback: this watcher
        // owns the still-recorded playback and releases it.
        Some(false) | None => PlaybackWatch::Finished,
    }
}

fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let rest = line.strip_prefix("VmRSS:")?;
                rest.split_whitespace()
                    .next()
                    .and_then(|kb| kb.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
        })
        .unwrap_or(0)
}

/// Manager identities for the notifier loop (#362): every start or stop
/// of a manager bumps this, so a loop polling a replaced manager sees a
/// mismatch and retires itself instead of notifying for a dead engine.
fn next_engine_instance() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The settings-save coordinator (#366): the Save button and the engine
/// active-model persistence both write the same settings file from
/// background tasks. Each save claims a sequence number when its document
/// is built; a writer holding the save mutex consults
/// [`settings_save_should_write`] — the newest document always wins, no
/// matter which background task happens to run last.
static SETTINGS_SAVE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static SETTINGS_SAVE_WRITTEN: AtomicU64 = AtomicU64::new(0);
static SETTINGS_SAVE_MUTEX: Mutex<()> = Mutex::new(());

/// Whether a queued settings writer still needs to write (#366 review):
/// skip only when a NEWER (or equal) sequence has already landed
/// successfully. A newer write that FAILED keeps older documents eligible
/// — the newest one did not reach the disk, so the older one must — which
/// makes the final on-disk document the newest document that saved
/// successfully, while a successfully-written newer document retires every
/// older one (they may never land after it).
pub(crate) fn settings_save_should_write(sequence: u64, newest_written: u64) -> bool {
    sequence > newest_written
}

/// The persisted backend override ("cpu"/"vulkan") as the engine's own
/// backend family (#362). An unknown string is `None` (automatic), not a
/// guess: settings this build no longer understands degrade to the safe
/// default rather than pinning the wrong engine.
pub(crate) fn backend_override_from_settings(value: &str) -> Option<Backend> {
    Backend::parse(value)
}

/// Start the bundled-engine manager for the committed engine settings
/// (#362): only in builtin mode, on the default data paths, with the
/// persisted model and backend override. `EngineManager::start` itself
/// never fails (problems surface in the snapshot); the one early error
/// is `default_paths` (an unresolvable data dir), which the app reports
/// through the engine status instead of pretending the engine exists.
fn start_engine(settings: &EngineSettings) -> (Option<EngineManager>, Option<String>) {
    if settings.mode != EngineMode::Builtin {
        return (None, None);
    }
    let mut config = match EngineConfig::default_paths() {
        Ok(config) => config,
        Err(err) => return (None, Some(err.to_string())),
    };
    config.backend_override = settings
        .backend_override
        .as_deref()
        .and_then(backend_override_from_settings);
    (Some(EngineManager::start(config, settings.active_model.clone())), None)
}

/// Split a metadata-only listing (G02): readable summaries in listing
/// order, damaged records flagged alongside them.
pub(crate) fn split_listing(
    list: Vec<ListedRecord>,
) -> (Vec<SessionSummary>, Vec<DamagedRecord>) {
    let mut sessions = Vec::new();
    let mut damaged = Vec::new();
    for record in list {
        match record {
            ListedRecord::Session(summary) => sessions.push(summary),
            ListedRecord::Damaged(record) => damaged.push(record),
        }
    }
    (sessions, damaged)
}

/// The selection after applying a listing (G02): kept while the selected
/// record is still readable, otherwise reset to the newest readable take —
/// never left pointing at a record the store just flagged as damaged.
pub(crate) fn next_selection(
    current: Option<&str>,
    sessions: &[SessionSummary],
) -> Option<String> {
    if current.is_some_and(|id| sessions.iter().any(|session| session.id == id)) {
        return current.map(str::to_string);
    }
    sessions.first().map(|session| session.id.clone())
}

/// The persisted `user_set_model` flag after an explicit settings save (R02).
///
/// The flag is set when this save actually changed the model, and sticky once
/// set: a later save that leaves the model untouched (the user only edited
/// the endpoint or terms) must not re-enable syncing the model from the
/// server's health response.
pub(crate) fn user_set_model_after_save(
    previous: bool,
    model_before: &str,
    model_saved: &str,
) -> bool {
    previous || model_before != model_saved
}

/// Which ephemeral notice the quality banner shows (G05): the capture
/// warning outranks the export notice — a take's clipping evidence stays
/// relevant for the session it describes, while a rename notice is a
/// one-off that clears itself. The two live in separate fields precisely
/// so neither can overwrite the other's content.
#[cfg(test)]
pub(crate) fn banner_notice<'a>(
    capture_warning: Option<&'a str>,
    export_notice: Option<&'a str>,
) -> Option<&'a str> {
    capture_warning.or(export_notice)
}

/// What a click on a take's trash button does (B05, #208).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeleteIntent {
    /// First click (or a click on a different take): arm the confirmation,
    /// delete nothing.
    Arm,
    /// Second click on the same take's armed button: delete it.
    Delete,
}

/// The delete confirmation state machine (B05, #208): one click arms
/// exactly the take it was pressed on; only a second click on that same
/// take deletes. A click on another take's button re-arms for it, so an
/// armed confirmation can never delete something it was not armed for.
pub(crate) fn delete_intent(armed_for: Option<&str>, clicked: &str) -> DeleteIntent {
    if armed_for == Some(clicked) {
        DeleteIntent::Delete
    } else {
        DeleteIntent::Arm
    }
}

/// Whether the armed delete confirmation still targets the selected take
/// (#208, with #214.2's lesson applied to it): an arm is inert the moment
/// the selection moves — including selection changes that bypass
/// `select_session`, like a new take jumping the drawer after it is
/// recorded — so a stale arm can never resurface as "Confirm delete?" on a
/// take the user never armed.
pub(crate) fn effective_delete_arm(armed_for: Option<&str>, selected: Option<&str>) -> bool {
    armed_for.is_some() && armed_for == selected
}

/// Whether applying a listing moves the selection off its current take
/// (review finding 1): the move must run the same reset as an explicit
/// click in `select_session` — including stopping playback of a take that
/// just vanished from the listing.
pub(crate) fn listing_moves_selection(
    current: Option<&str>,
    sessions: &[SessionSummary],
) -> bool {
    next_selection(current, sessions).as_deref() != current
}

/// A batch identity for the unsaved list (review finding 3): a digest over
/// the stashed ids in order, so ANY change to the batch — another stash
/// today, any insert, removal, or reorder a future change might add —
/// invalidates an armed discard confirmation. Derived on demand, so the
/// stash path itself (`stash_unsaved`) needs no hook.
///
/// `DefaultHasher::new()` hashes with fixed keys, making the token stable
/// for an unchanged batch across calls.
pub(crate) fn unsaved_batch_token(unsaved: &[UnsavedWav]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for capture in unsaved {
        capture.id.hash(&mut hasher);
    }
    hasher.finish()
}

/// What a click on the discard button does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiscardIntent {
    /// Arm the confirmation for the current batch of unsaved takes.
    Arm,
    /// Confirmed for exactly this batch: discard them.
    Execute,
}

/// The discard confirmation state machine, scoped to the batch it was
/// armed for (#214.4): the confirmation only executes while the unsaved
/// list is still the exact batch it was armed against, identified by
/// [`unsaved_batch_token`]. A take stashed later (a new failed save)
/// changes the token, so the carried-over arm is dead — the next click
/// re-arms for the new batch instead of single-click-discarding audio the
/// user never armed for. This is the derived form of resetting the arm in
/// the stash path itself.
pub(crate) fn discard_intent(armed_for: Option<u64>, batch_token: u64) -> DiscardIntent {
    if armed_for == Some(batch_token) {
        DiscardIntent::Execute
    } else {
        DiscardIntent::Arm
    }
}

/// Which in-flight delete ids survive a landed listing (review finding 2):
/// an id is released only when the listing itself confirms the row is
/// gone — never by the delete job's own completion, whose refresh may
/// fail or land late while the row is still on screen. Until a listing
/// without the row arrives, the guard stays up (the safe direction).
pub(crate) fn surviving_deletes(
    deleting_ids: &HashSet<String>,
    sessions: &[SessionSummary],
) -> HashSet<String> {
    deleting_ids
        .iter()
        .filter(|id| sessions.iter().any(|session| &session.id == *id))
        .cloned()
        .collect()
}

/// Which ephemeral drawer flag a scheduled reset owns, and the take it was
/// earned for (#214.2): a timer only clears its flag while the flag still
/// names that take, so a copy earned on take B is not wiped by take A's
/// still-running reset timer.
#[derive(Clone, Debug, PartialEq, Eq)]
enum EphemeralFlag {
    Copied(String),
    WavSaved(String),
}

/// Highest `-N` suffix attempted when dodging an existing download name.
const MAX_DOWNLOAD_NAME_ATTEMPTS: u32 = 1_000;

/// The `attempt`-th candidate name for a download: the first attempt is the
/// requested name itself, later ones insert `-N` before the extension
/// (`"starling-a.wav"` → `"starling-a-2.wav"`). A leading dot belongs to the
/// stem (`.zshrc`), and extension-less names take the suffix at the end.
fn download_name_candidate(name: &str, attempt: u32) -> String {
    if attempt <= 1 {
        return name.to_string();
    }

    let suffix = format!("-{attempt}");
    match name.rfind('.') {
        None | Some(0) => format!("{name}{suffix}"),
        Some(dot) => format!("{}{suffix}{}", &name[..dot], &name[dot..]),
    }
}

/// Write `bytes` into `dir` without ever overwriting an existing file (G05).
///
/// Every candidate name is opened with `create_new`, so an existing file —
/// or a symlink planted at that name — fails the open with `AlreadyExists`
/// instead of being replaced or followed, and the next `-N` candidate is
/// tried. A failed body write removes the partial file, so a full disk or
/// permission error never leaves a truncated export behind. Exclusive
/// creation is what makes this collision-safe: there is deliberately no
/// overwrite path to race with.
///
/// `sync` fsyncs the landed file. It is kept for WAV exports — an unsaved
/// take's export can be the only copy of the recording, and WAV durability
/// is not weakened here — while transcript `.txt` exports skip it: a text
/// file lost to a crash is byte-for-byte re-exportable from history.
fn write_download_exclusive(
    dir: &Path,
    name: &str,
    bytes: &[u8],
    sync: bool,
) -> std::io::Result<PathBuf> {
    for attempt in 1..=MAX_DOWNLOAD_NAME_ATTEMPTS {
        let candidate = download_name_candidate(name, attempt);
        let path = dir.join(&candidate);

        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                let written = file.write_all(bytes).and_then(|()| {
                    if sync {
                        file.sync_all()
                    } else {
                        Ok(())
                    }
                });
                if let Err(err) = written {
                    // Never leave a truncated export behind on disk.
                    let _ = std::fs::remove_file(&path);
                    return Err(err);
                }
                return Ok(path);
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!("no free filename for {name:?} in the downloads directory"),
    ))
}

impl StarlingApp {
    pub fn new(started: Instant, diagnostics: bool, cx: &mut Context<Self>) -> Self {
        let settings = Settings::load_or_default();
        // D14: storage v2 is THE store, opened unconditionally. An open
        // failure is a hard, honest startup error — there is no other
        // backend to fall back to and no flag to clear; the cause must be
        // fixed and the app restarted.
        let (store, store_error) = match Store::open() {
            Ok(store) => (Some(store), None),
            Err(err) => (
                None,
                Some(format!(
                    "Could not open the recording store (storage v2): {err}. There is no \
                     fallback store — fix the cause and restart. Until then, new recordings are \
                     kept in memory only and can be downloaded from the unsaved list."
                )),
            ),
        };

        Self::with_dependencies(
            started,
            diagnostics,
            settings,
            store,
            store_error,
            Player::new().ok(),
            cx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_dependencies(
        started: Instant,
        diagnostics: bool,
        settings: Settings,
        store: Option<Store>,
        store_error: Option<String>,
        player: Option<Player>,
        cx: &mut Context<Self>,
    ) -> Self {
        let endpoint = settings.endpoint.clone();
        let model = settings.model.clone();
        let terms_input = settings.expected_terms_input();
        let draft_endpoint = cx.new(|cx| TextField::new("http://127.0.0.1:8181", &endpoint, cx));
        let draft_model = cx.new(|cx| TextField::new("parakeet", &model, cx));
        let draft_terms = cx.new(|cx| TextField::new("auth, Starling, GGUF", &terms_input, cx));
        let processing_settings = settings.processing.clone();
        // A saved mode this build no longer ships falls back to the
        // default mode; say so rather than change behavior silently.
        let mode_note = crate::processing::unknown_mode_note(&processing_settings.mode);
        let draft_s1_endpoint = cx.new(|cx| {
            TextField::new("http://127.0.0.1:8182", &processing_settings.s1_endpoint, cx)
        });
        let draft_api_endpoint = cx.new(|cx| {
            TextField::new("https://api.openai.com/v1", &processing_settings.api_endpoint, cx)
        });
        let draft_api_model =
            cx.new(|cx| TextField::new("gpt-4.1-mini", &processing_settings.api_model, cx));
        let draft_api_key_env = cx.new(|cx| {
            TextField::new("OPENAI_API_KEY", &processing_settings.api_key_env, cx)
        });

        // #362: the engine manager starts with the app in builtin mode —
        // before the first render, off the UI-critical path (everything
        // slow runs on its supervisor thread). A start failure is honest
        // state, never a crash: the app still works in manual mode.
        let engine_settings = settings.engine.clone();
        let (engine, engine_startup_error) = start_engine(&engine_settings);
        let engine_instance = engine.as_ref().map(|_| next_engine_instance()).unwrap_or(0);
        let applied_backend_override = engine_settings.backend_override.clone();
        let draft_engine_mode = engine_settings.mode;
        let draft_backend_override = engine_settings.backend_override.clone();

        // #221: a stored shortcut this build cannot use falls back to the
        // default, and says so.
        let mut dictation_settings = settings.dictation.clone();
        let (shortcut, shortcut_note) = match crate::shortcut::Shortcut::parse(&dictation_settings.shortcut) {
            Ok(shortcut) => (shortcut, None),
            Err(reason) => {
                dictation_settings.shortcut = DEFAULT_SHORTCUT.to_string();
                let shortcut = crate::shortcut::Shortcut::parse(DEFAULT_SHORTCUT)
                    .expect("the default shortcut parses");
                let note = format!(
                    "The saved recording shortcut could not be used ({reason}); {} is active \
                     instead.",
                    shortcut.label()
                );
                (shortcut, Some(note))
            }
        };
        let draft_shortcut = cx.new(|cx| {
            TextField::new(DEFAULT_SHORTCUT, &dictation_settings.shortcut, cx)
        });
        let mode_note = match (mode_note, shortcut_note) {
            (Some(mode), Some(shortcut)) => Some(format!("{mode}\n{shortcut}")),
            (mode, shortcut) => mode.or(shortcut),
        };

        let draft_lower_level =
            cx.new(|_| crate::slider::LevelSlider::new(settings.playback.lower_level_percent));
        // The level readout lives in the settings view.
        cx.observe(&draft_lower_level, |_, _, cx| cx.notify()).detach();
        let draft_cue_volume =
            cx.new(|_| crate::slider::LevelSlider::new(settings.feedback.cue_volume_percent));
        cx.observe(&draft_cue_volume, |_, _, cx| cx.notify()).detach();

        Self {
            error: match (store_error.clone(), mode_note) {
                (Some(store), Some(mode)) => Some(format!("{store}\n{mode}")),
                (store, mode) => store.or(mode),
            },
            capture_warning: None,
            export_notice: None,
            store,
            store_error,
            player,
            root_focus: cx.focus_handle(),
            engine,
            engine_startup_error,
            engine_settings,
            applied_backend_override,
            engine_instance,
            draft_engine_mode,
            draft_backend_override,
            draft_microphone: settings.microphone.preferred_device.clone(),
            microphone_settings: settings.microphone.clone(),
            mic: crate::mic::MicState::default(),
            quit_hook: None,
            active_take: None,
            endpoint,
            model,
            expected_terms_input: terms_input,
            user_set_model: settings.user_set_model,
            settings_open: false,
            draft_endpoint,
            draft_model,
            draft_terms,
            providers: processing::build_providers(&processing_settings),
            // What actually runs: an unknown saved mode resolves to the
            // default, and the dialog must show that one selected.
            draft_mode: crate::processing::mode(&processing_settings.mode).id.clone(),
            processing_settings,
            processing: HashMap::new(),
            drafts: HashMap::new(),
            processing_jobs: HashMap::new(),
            stop_instants: HashMap::new(),
            processing_loading: HashSet::new(),
            draft_s1_endpoint,
            draft_api_endpoint,
            draft_api_model,
            draft_api_key_env,
            connection: Connection::Checking,
            server_model: "server".to_string(),
            probe: None,
            health_sequencer: CheckSequencer::new(),
            probe_sequencer: CheckSequencer::new(),
            sessions: Vec::new(),
            damaged: Vec::new(),
            selected_id: None,
            active_ids: HashSet::new(),
            retry_menu: None,
            pending_retry: None,
            retry_seq: 0,
            unsaved: Vec::new(),
            confirm_discard_for: None,
            copied: None,
            wav_saved: None,
            confirm_delete_id: None,
            deleting_ids: HashSet::new(),
            activation: crate::activation::Activation::new(
                crate::activation::ActivationConfig::from_settings(&dictation_settings),
            ),
            recording_take: None,
            draft_shortcut,
            draft_activation: dictation_settings.activation,
            draft_double_tap: dictation_settings.double_tap_hands_free,
            dictation_draft_error: None,
            draft_playback_mode: settings.playback.during_recording,
            draft_lower_level,
            playback_settings: settings.playback,
            playback: PlaybackAttenuation::start(platform_backend()),
            playback_lease: None,
            feedback: settings.feedback,
            overlay: crate::overlay::Overlay::new(Instant::now()),
            cue_gate: crate::cues::CueGate::default(),
            draft_overlay_mode: settings.feedback.overlay,
            draft_cues: settings.feedback.cues,
            draft_cue_volume,
            audio_upkeep: crate::upkeep::AudioUpkeep::new(settings.storage),
            delivery: crate::delivery::DeliveryState::new(
                Arc::new(starling_insertion::Inserter::for_this_session()),
                settings.insertion,
            ),
            draft_insertion: settings.insertion,
            insertion_unverifiable: false,
            shortcut,
            pending_shortcut: None,
            window_focus: Vec::new(),
            focus_observer: None,
            dictation_settings,
            global_shortcuts: None,
            shortcut_registration: Ok(()),
            portal_shortcuts: None,
            system_check: Default::default(),
            key_interceptor: None,
            take_notice: None,
            recovery_notice: None,
            playback_notice: None,
            correction_chain: None,
            playing_id: None,
            playback_generation: 0,
            recorder: None,
            stream_pump: None,
            stream_generation: 0,
            stream_trace: None,
            live_preview: settings.live_preview,
            draft_live_preview: settings.live_preview,
            live_partial: String::new(),
            staging: None,
            background_stagings: Vec::new(),
            next_staging_token: 0,
            pending_focus: None,
            stream_degradation: None,
            levels: vec![0.06; 52],
            elapsed_ms: 0.0,
            diagnostics: diagnostics.then_some((started, false)),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(store: Option<Store>, cx: &mut Context<Self>) -> Self {
        // Manual engine mode: tests exercise the app, not the bundled
        // engine — starting a manager here would probe the developer's
        // real engine/data directories from every test.
        let mut settings = Settings::default_settings();
        settings.engine.mode = EngineMode::Manual;
        Self::with_dependencies(
            Instant::now(),
            false,
            settings,
            store,
            None,
            None,
            cx,
        )
    }

    pub fn init(&mut self, cx: &mut Context<Self>) {
        if let Some(store) = self.store.clone() {
            cx.spawn(async move |this, cx| {
                // Startup recovery, v2 (§4): reconcile journals against the
                // metadata rows and fail recognition attempts a previous
                // run left "started" (the "stuck in Transcribing" fix —
                // same note, same outcome). Findings surface through the
                // error banner; recovery is never a reason to abort
                // startup.
                let recovered = {
                    let store = store.clone();
                    cx.background_spawn(async move { store.startup_recovery() }).await
                };
                let recheck = crate::upload::show_startup_recovery(&this, recovered, cx);

                refresh_sessions(&this, &store, cx).await;
                // #342: compression and retention after recovery settled.
                this.update(cx, |app, cx| app.start_audio_upkeep(cx)).ok();
                if !recheck.is_empty() {
                    crate::upload::recheck_capture_journals(&this, &store, recheck, cx).await;
                }
            })
            .detach();
        }
        self.watch_playback_notices(cx);
        match self.engine_settings.mode {
            // #362: in builtin mode the connection indicator derives from
            // the engine snapshot (the notifier loop refreshes it); probing
            // `self.endpoint` would report a server takes never use.
            EngineMode::Builtin => {
                if let Some(engine) = self.engine.clone() {
                    let instance = self.engine_instance;
                    self.watch_engine(engine, instance, cx);
                }
            }
            // Manual keeps today's startup health probe (#207).
            EngineMode::Manual => {
                self.check_health(
                    HealthCheckPurpose::Live,
                    self.endpoint.clone(),
                    cx,
                );
            }
        }
        self.register_quit_hook(cx);
    }

    /// The engine notifier loop (#362): polls the manager's generation
    /// every 200 ms (the same shape as the global-hotkey loop in
    /// `main`) and, on change, refreshes the connection indicator and
    /// notifies. A changed active model is persisted immediately through
    /// the same background save path as `save_settings` — the engine is
    /// the source of truth, the file only restores it at launch. The
    /// loop retires itself when its manager is replaced (a mode switch
    /// bumped `engine_instance`) or the entity is released.
    fn watch_engine(&mut self, engine: EngineManager, instance: u64, cx: &mut Context<Self>) {
        let mut last_generation = engine.generation();
        cx.spawn(async move |this, cx| {
            loop {
                Timer::after(Duration::from_millis(200)).await;
                let keep_going = this
                    .update(cx, |app, cx| {
                        if app.engine_instance != instance {
                            return false;
                        }
                        let generation = engine.generation();
                        if generation == last_generation {
                            return true;
                        }
                        last_generation = generation;
                        let snapshot = engine.snapshot();
                        app.connection = views::engine_status_view(&snapshot).connection;
                        if let Some(active) = &snapshot.active {
                            if app.engine_settings.active_model.as_deref()
                                != Some(active.model_id.as_str())
                            {
                                app.engine_settings.active_model =
                                    Some(active.model_id.clone());
                                app.persist_committed_settings(cx);
                            }
                        }
                        cx.notify();
                        true
                    })
                    .unwrap_or(false);
                if !keep_going {
                    break;
                }
            }
        })
        .detach();
    }

    /// Stop the engine when the app quits (#362): the sidecar is ours,
    /// so it must not outlive the window. This entity hook is the single
    /// owner of engine shutdown — quit runs it (main.rs only quits, it
    /// does not shut the engine down itself). Blocking here is correct —
    /// quit waits for the stop — and the server's `--parent-pid`
    /// watchdog is the backstop if the app crashes first.
    fn register_quit_hook(&mut self, cx: &mut Context<Self>) {
        if self.quit_hook.is_some() {
            return;
        }
        // `Context::on_app_quit` hands the entity itself; the returned
        // Subscription is kept so the hook lives as long as the app.
        // Bumping the instance first retires the notifier loop (#366):
        // while quit waits for the stop, the loop must not repaint
        // `connection` from a dying engine or persist engine settings.
        self.quit_hook = Some(cx.on_app_quit(|app, _cx| {
            app.engine_instance = next_engine_instance();
            if let Some(engine) = app.engine.take() {
                engine.shutdown();
            }
            // Restores playback if a take is still live.
            app.playback.shutdown();
            async {}
        }));
    }

    /// Surfaces the playback service's notices (#361), polled like the
    /// engine notifier.
    fn watch_playback_notices(&mut self, cx: &mut Context<Self>) {
        let handle = self.playback.handle();
        cx.spawn(async move |this, cx| {
            loop {
                Timer::after(Duration::from_millis(250)).await;
                let notices = handle.take_notices();
                let alive = this
                    .update(cx, |app, cx| {
                        if let Some(notice) = notices.into_iter().last() {
                            app.playback_notice = Some(notice);
                            cx.notify();
                        }
                    })
                    .is_ok();
                if !alive {
                    break;
                }
            }
        })
        .detach();
    }

    /// The settings file's contents as of the app's current state: every
    /// save path (the Save button, engine active-model persistence)
    /// builds the same document, so the two writers cannot disagree
    /// about fields the other one never touched.
    fn committed_settings(&self) -> Settings {
        let mut settings = Settings {
            endpoint: self.endpoint.clone(),
            model: self.model.clone(),
            expected_terms: Vec::new(),
            user_set_model: self.user_set_model,
            processing: self.processing_settings.clone(),
            engine: self.engine_settings.clone(),
            dictation: self.dictation_settings.clone(),
            microphone: self.microphone_settings.clone(),
            playback: self.playback_settings,
            feedback: self.feedback,
            storage: self.audio_upkeep.settings,
            insertion: self.delivery.settings,
            live_preview: self.live_preview,
        };
        settings.set_expected_terms_input(&self.expected_terms_input);
        settings
    }

    /// Persist the committed settings in the background (the engine
    /// active-model path of #363: the manager switched models, the file
    /// follows; #221: a refused mid-take shortcut swap writes the shortcut
    /// that stayed active back). A failure surfaces through the error
    /// banner like any other save.
    pub(crate) fn persist_committed_settings(&self, cx: &mut Context<Self>) {
        let Ok(path) = Settings::default_path() else {
            return;
        };
        self.spawn_settings_save(path, cx);
    }

    /// The shared background settings save: the Save button and the engine
    /// active-model persistence both write this file from background
    /// tasks, so the writers are serialized newest-wins — without this, an
    /// older document (spawned first, written last) would overwrite a newer
    /// one. Each save claims a sequence number when its document is built;
    /// the writer, holding the process-wide save mutex, skips the write
    /// when a newer sequence has already been written. A failure surfaces
    /// through the error banner like any other save.
    fn spawn_settings_save(&self, path: PathBuf, cx: &mut Context<Self>) {
        let settings = self.committed_settings();
        let storage = settings.storage;
        let sequence = SETTINGS_SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
        cx.spawn(async move |this, cx| {
            let saved = cx
                .background_spawn(async move {
                    let _writer = SETTINGS_SAVE_MUTEX
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if !settings_save_should_write(
                        sequence,
                        SETTINGS_SAVE_WRITTEN.load(Ordering::Relaxed),
                    ) {
                        // A newer document has already been written; this
                        // older one must not land after it.
                        return Ok(false);
                    }
                    let saved = settings.save(&path);
                    if saved.is_ok() {
                        SETTINGS_SAVE_WRITTEN.store(sequence, Ordering::Relaxed);
                    }
                    saved.map(|()| true)
                })
                .await;
            this.update(cx, |app, cx| match saved {
                // #342: retention limits take effect only once the file
                // that holds them is written.
                Ok(true) => app.storage_settings_saved(sequence, storage, cx),
                Ok(false) => {}
                Err(err) => {
                    app.error = Some(format!("Could not save settings: {err}"));
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    /// Apply a committed engine mode change (#362): builtin→manual stops
    /// the manager off the UI thread (shutdown can wait out a sidecar
    /// stop), manual→builtin starts one and begins watching it. The
    /// persisted model and backend override ride along on start.
    fn apply_engine_mode_change(&mut self, cx: &mut Context<Self>) {
        match self.engine_settings.mode {
            EngineMode::Builtin => {
                // Retire any in-flight manual health probe (the same
                // latest-wins rule a saved endpoint uses, #207): switching
                // to builtin hands the indicator to the engine snapshot,
                // and a slow manual probe landing afterwards would
                // overwrite it with Offline — and an error banner — for a
                // server no longer in use.
                let _ = self.health_sequencer.begin();
                if self.engine.is_none() {
                    let (engine, startup_error) = start_engine(&self.engine_settings);
                    self.engine_startup_error = startup_error;
                    if let Some(engine) = engine {
                        self.applied_backend_override =
                            self.engine_settings.backend_override.clone();
                        self.engine_instance = next_engine_instance();
                        let instance = self.engine_instance;
                        self.connection = views::engine_status_view(&engine.snapshot()).connection;
                        self.watch_engine(engine.clone(), instance, cx);
                        self.engine = Some(engine);
                    }
                    cx.notify();
                }
            }
            EngineMode::Manual => {
                if let Some(engine) = self.engine.take() {
                    self.engine_startup_error = None;
                    self.engine_instance = next_engine_instance();
                    // The notifier loop retires on the instance bump; the
                    // connection indicator is the manual health probe's
                    // to write again (the save triggers one).
                    cx.background_spawn(async move { engine.shutdown(); }).detach();
                    cx.notify();
                }
            }
        }
    }

    /// The live engine snapshot for the views (None when no manager
    /// runs — manual mode, or the startup failure in
    /// `engine_startup_error`).
    pub fn engine_snapshot(&self) -> Option<starling_dictation::engine::EngineSnapshot> {
        self.engine.as_ref().map(|engine| engine.snapshot())
    }

    // ---- engine actions (#362, #363) ---------------------------------
    // All of these are immediate: they command the manager and the
    // notifier loop paints the result. None of them is part of Save.

    /// Download (background) a model without activating it.
    pub fn engine_download(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(engine) = &self.engine {
            engine.download(id);
        }
        cx.notify();
    }

    /// Cancel a model's running download.
    pub fn engine_cancel_download(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(engine) = &self.engine {
            engine.cancel_download(id);
        }
        cx.notify();
    }

    /// Download-if-needed then switch to a model (#363).
    pub fn engine_activate(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(engine) = &self.engine {
            engine.activate(id);
        }
        cx.notify();
    }

    /// Delete a model's files; the manager refuses active/switching/
    /// downloading models and the refusal is surfaced, not swallowed.
    pub fn engine_delete_model(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(engine) = &self.engine {
            if let Err(err) = engine.delete_model(id) {
                self.error = Some(err.to_string());
            }
        }
        cx.notify();
    }

    /// Clear a Failed/crash-loop state and retry the last model. With no
    /// manager at all (the startup failure in `engine_startup_error`),
    /// retry starting one — the failure UI offers Retry for that too.
    pub fn engine_retry(&mut self, cx: &mut Context<Self>) {
        match &self.engine {
            Some(engine) => engine.retry(),
            None if self.engine_settings.mode == EngineMode::Builtin => {
                self.apply_engine_mode_change(cx);
            }
            None => {}
        }
        cx.notify();
    }

    /// Answer a pending NeedsDrain decision: switch after the take.
    pub fn engine_confirm_drain_swap(&mut self, cx: &mut Context<Self>) {
        if let Some(engine) = &self.engine {
            engine.confirm_drain_swap();
        }
        cx.notify();
    }

    /// Cancel a running switch; the current model keeps serving.
    pub fn engine_cancel_switch(&mut self, cx: &mut Context<Self>) {
        if let Some(engine) = &self.engine {
            engine.cancel_switch();
        }
        cx.notify();
    }

    /// The "Use CPU engine" / "Use automatic engine" toggle (#362):
    /// applies immediately (the manager re-selects the backend and moves
    /// the engine, draining in-flight takes) AND persists immediately,
    /// like `engine_switch_to_manual` does for the mode — an immediate
    /// action must not wait behind Save, or a later Cancel would leave
    /// the running engine diverged from the saved settings. The dialog
    /// draft stays in sync, so an unchanged Save is a no-op and Cancel
    /// keeps what was applied (the draft resets from the committed
    /// value when the dialog reopens).
    pub fn engine_toggle_cpu(&mut self, cx: &mut Context<Self>) {
        let pinned = self.draft_backend_override.as_deref() == Some("cpu");
        let next = if pinned { None } else { Some("cpu".to_string()) };
        if let Some(engine) = &self.engine {
            engine.set_backend_override(
                next.as_deref().and_then(backend_override_from_settings),
            );
            self.applied_backend_override = next.clone();
        }
        self.engine_settings.backend_override = next.clone();
        self.draft_backend_override = next;
        self.persist_committed_settings(cx);
        cx.notify();
    }

    /// The failure action "Switch to my own server" (#362): an explicit
    /// user decision, so unlike the radio it applies and persists
    /// immediately rather than waiting for Save.
    pub fn engine_switch_to_manual(&mut self, cx: &mut Context<Self>) {
        self.draft_engine_mode = EngineMode::Manual;
        self.engine_settings.mode = EngineMode::Manual;
        self.apply_engine_mode_change(cx);
        self.persist_committed_settings(cx);
        // The manual indicator starts as a probe in flight, exactly like
        // a startup in manual mode.
        self.connection = Connection::Checking;
        self.check_health(HealthCheckPurpose::Live, self.endpoint.clone(), cx);
    }

    /// The dialog's engine mode radio (#362): a draft like the other
    /// fields — it commits (and starts/stops the manager) on Save.
    pub(crate) fn pick_draft_engine_mode(&mut self, mode: EngineMode, cx: &mut Context<Self>) {
        self.draft_engine_mode = mode;
        cx.notify();
    }

    pub fn busy(&self) -> bool {
        !self.active_ids.is_empty()
    }

    pub fn selected(&self) -> Option<&SessionSummary> {
        self.selected_id
            .as_ref()
            .and_then(|id| self.sessions.iter().find(|session| &session.id == id))
    }

    pub fn is_active(&self, id: &str) -> bool {
        self.active_ids.contains(id)
    }

    /// Split a metadata-only listing (G02) into readable summaries and
    /// damaged records. Selection survives when the selected record is
    /// still readable; a selection that became damaged (or vanished) falls
    /// back to the newest readable take rather than leaving the drawer
    /// pointed at a record that can no longer be read.
    pub fn apply_sessions(&mut self, list: Vec<ListedRecord>) {
        let (sessions, damaged) = split_listing(list);
        let next = next_selection(self.selected_id.as_deref(), &sessions);
        if next != self.selected_id {
            // Review finding 1: a listing-driven selection move runs the
            // same reset as an explicit click — playback of the take that
            // just vanished stops instead of running on under the drawer.
            self.selection_moved();
        }
        self.selected_id = next;
        self.sessions = sessions;
        self.damaged = damaged;
        // Review finding 2: this is the only place a landed listing may
        // release an in-flight delete — the row's absence from the listing
        // is the proof the delete landed.
        self.deleting_ids = surviving_deletes(&self.deleting_ids, &self.sessions);
    }

    /// G02: interacting with a damaged history row surfaces the recorded
    /// reason — the quarantine is visible and explained, never a silent
    /// gap, and nothing behind it was deleted.
    pub fn surface_damage(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(damaged) = self.damaged.iter().find(|record| record.id == id) {
            self.error = Some(format!(
                "This recording could not be read and was kept as-is: {}. Nothing was deleted; \
                 the files are untouched for manual recovery.",
                damaged.reason
            ));
            cx.notify();
        }
    }

    /// Check the health of the committed endpoint (#207,
    /// B06). The endpoint is passed explicitly — never captured from
    /// drafts — so a caller cannot hand the live-state writer a
    /// configuration the user has not saved. The probe behind Test
    /// Connection never routes through here: it owns its own outcome, so
    /// a draft endpoint's failure cannot paint the live connection
    /// offline. Each check claims a sequence token before awaiting
    /// anything, so a slower check against an endpoint that has since
    /// been replaced is dropped instead of overwriting the newer status.
    pub fn check_health(
        &mut self,
        purpose: HealthCheckPurpose,
        endpoint: String,
        cx: &mut Context<Self>,
    ) {
        let model = self.model.clone();
        let token = self.health_sequencer.begin();
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let client = StarlingClient::new(&endpoint, &model)
                        .and_then(|client| client.with_timeout_ms(5_000))
                        .map_err(|err| err.to_string())?;
                    client.health().map_err(|err| err.to_string())
                })
                .await;
            this.update(cx, |app, cx| {
                if !app.health_sequencer.is_current(token) {
                    // A newer live check already landed (a saved endpoint
                    // replaced this one): this result is against an endpoint
                    // that is no longer current and must not overwrite it.
                    return;
                }
                let writes = live_check_writes(purpose, app.user_set_model, &result);
                app.connection = writes.connection;
                if let Some(server_model) = writes.server_model {
                    app.server_model = server_model;
                }
                if let Some(model) = writes.model_sync {
                    app.model = model;
                }
                if let Some(error) = writes.error_slot {
                    app.error = error;
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn open_settings(&mut self, cx: &mut Context<Self>) {
        // #221: shortcut presses that happened before the dialog opened
        // are judged with the dialog closed.
        self.flush_system_events(cx);
        self.settings_open = true;
        // A fresh dialog starts with no probe (#207): retire anything
        // still in flight from a previous dialog and clear its settled
        // outcome, so the callout shows the committed status until the
        // user tests again.
        self.probe_sequencer.cancel_all();
        self.probe = None;
        let endpoint = self.endpoint.clone();
        let model = self.model.clone();
        let terms = self.expected_terms_input.clone();
        // The engine drafts start from the committed values (#362):
        // mode and backend override are Save-saved like these fields;
        // everything else in the Engine section is live manager state.
        self.draft_engine_mode = self.engine_settings.mode;
        self.draft_backend_override = self.engine_settings.backend_override.clone();
        // The device list and the shortcut evidence are fresh for every
        // dialog: the shortcut heard last time may have changed since.
        self.draft_microphone = self.microphone_settings.preferred_device.clone();
        self.cancel_mic_check();
        self.mic.settings_launch_error = None;
        self.mic.settings_launch_generation += 1;
        self.mic.shortcut_heard = None;
        self.refresh_input_devices(cx);
        self.draft_endpoint.update(cx, |field, cx| {
            field.set_value(&endpoint, cx);
        });
        self.draft_model.update(cx, |field, cx| {
            field.set_value(&model, cx);
        });
        self.draft_terms.update(cx, |field, cx| {
            field.set_value(&terms, cx);
        });
        let processing = self.processing_settings.clone();
        // The mode that actually runs (an unknown saved id is the default).
        self.draft_mode = crate::processing::mode(&processing.mode).id.clone();
        self.draft_s1_endpoint.update(cx, |field, cx| {
            field.set_value(&processing.s1_endpoint, cx);
        });
        self.draft_api_endpoint.update(cx, |field, cx| {
            field.set_value(&processing.api_endpoint, cx);
        });
        self.draft_api_model.update(cx, |field, cx| {
            field.set_value(&processing.api_model, cx);
        });
        self.draft_api_key_env.update(cx, |field, cx| {
            field.set_value(&processing.api_key_env, cx);
        });
        // #221: the dictation drafts start from the committed values.
        let shortcut = self.dictation_settings.shortcut.clone();
        self.draft_shortcut.update(cx, |field, cx| {
            field.set_value(&shortcut, cx);
        });
        self.draft_activation = self.dictation_settings.activation;
        self.draft_double_tap = self.dictation_settings.double_tap_hands_free;
        self.dictation_draft_error = None;
        self.draft_playback_mode = self.playback_settings.during_recording;
        self.draft_live_preview = self.live_preview;
        self.draft_lower_level.update(cx, |slider, cx| {
            slider.set_value(self.playback_settings.lower_level_percent, cx);
        });
        self.draft_overlay_mode = self.feedback.overlay;
        self.draft_cues = self.feedback.cues;
        self.draft_cue_volume.update(cx, |slider, cx| {
            slider.set_value(self.feedback.cue_volume_percent, cx);
        });
        self.audio_upkeep.draft = self.audio_upkeep.settings;
        self.draft_insertion = self.delivery.settings;
        // A display round trip: off the UI thread, so a compositor that
        // stopped answering cannot freeze the dialog.
        self.check_session_verifies(cx);
        cx.notify();
    }

    /// The processing settings the dialog currently shows (for the
    /// destination disclosure, before anything is saved).
    pub(crate) fn draft_processing_settings(&self, cx: &gpui::App) -> ProcessingSettings {
        ProcessingSettings {
            mode: self.draft_mode.clone(),
            s1_endpoint: self.draft_s1_endpoint.read(cx).value().trim().to_string(),
            api_endpoint: self.draft_api_endpoint.read(cx).value().trim().to_string(),
            api_model: self.draft_api_model.read(cx).value().trim().to_string(),
            api_key_env: self.draft_api_key_env.read(cx).value().trim().to_string(),
        }
    }

    pub(crate) fn pick_draft_mode(&mut self, mode: &str, cx: &mut Context<Self>) {
        self.draft_mode = mode.to_string();
        cx.notify();
    }

    pub fn close_settings(&mut self, cx: &mut Context<Self>) {
        // #221: presses made while the dialog was open never start a take.
        self.flush_system_events(cx);
        self.settings_open = false;
        // A running microphone check releases its device with the dialog;
        // a pending check transcription has nowhere to land.
        self.cancel_mic_check();
        self.mic.settings_launch_error = None;
        self.mic.settings_launch_generation += 1;
        // B06 (#207): in-flight probes are retired with the dialog — a
        // stray probe has nothing to land in, and the settled outcome
        // dies with the draft it tested. Live state was never theirs to
        // change, so there is nothing to re-probe on close.
        self.probe_sequencer.cancel_all();
        self.probe = None;
        cx.notify();
    }

    /// Test the DRAFT configuration without touching committed state
    /// (#207, B06). The probe runs against the draft endpoint. Its outcome
    /// lands in the dialog's own probe state: the live connection
    /// status, the server model, the committed model, and the global
    /// error banner are never written from here. Only the newest probe
    /// may report, so a slow earlier probe cannot overwrite a newer
    /// result.
    pub fn test_connection(&mut self, cx: &mut Context<Self>) {
        let draft = self.draft_endpoint.read(cx).value();
        let Some((token, clean)) = probe_press(&mut self.probe_sequencer, self.probe.as_ref(), &draft)
        else {
            // The newest press is still in flight: drop this one, the
            // state-layer twin of the disabled button (#207 review).
            return;
        };
        let model = self.draft_model.read(cx).value();

        if clean.is_empty() {
            // The draft has no endpoint to connect to: settle the save's
            // reason as the probe's own failure — under `token`, which
            // has already retired the in-flight probe, so its late
            // landing cannot overwrite this newer outcome (#207 review).
            self.probe = Some(ConnectionProbe::Done {
                endpoint: clean,
                outcome: ProbeOutcome::Failed {
                    message: EMPTY_ENDPOINT_REASON.to_string(),
                },
            });
            cx.notify();
            return;
        }

        self.probe = Some(ConnectionProbe::Testing {
            endpoint: clean.clone(),
        });
        cx.notify();
        cx.spawn(async move |this, cx| {
            // The request consumes its own copy; `clean` stays for the
            // landing, which records the endpoint it probed.
            let request_endpoint = clean.clone();
            let result = cx
                .background_spawn(async move {
                    let client = StarlingClient::new(&request_endpoint, &model)
                        .and_then(|client| client.with_timeout_ms(5_000))
                        .map_err(|err| err.to_string())?;
                    client.health().map_err(|err| err.to_string())
                })
                .await;
            this.update(cx, |app, cx| {
                if !app.probe_sequencer.is_current(token) {
                    // The dialog closed or a newer probe began: this
                    // outcome has nowhere to land.
                    return;
                }
                app.probe = Some(ConnectionProbe::Done {
                    endpoint: clean,
                    outcome: match result {
                        Ok(health) => probe_outcome_from_health(&health),
                        Err(message) => ProbeOutcome::Failed { message },
                    },
                });
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn save_settings(&mut self, cx: &mut Context<Self>) {
        // #221: a shortcut the app cannot use is refused before anything
        // is applied or persisted; the dialog stays open with the reason.
        let shortcut = match crate::shortcut::Shortcut::parse(&self.draft_shortcut.read(cx).value()) {
            Ok(shortcut) => shortcut,
            Err(reason) => {
                self.dictation_draft_error = Some(reason);
                cx.notify();
                return;
            }
        };
        self.dictation_draft_error = None;
        // #362: the engine mode commits with this save. The manual
        // endpoint rules (#213) apply only in manual mode — a builtin
        // install has no server to name, and refusing the save over an
        // empty endpoint field the dialog does not even show would lock
        // the user out of saving anything else.
        let engine_mode = self.draft_engine_mode;
        let manual_endpoint = if engine_mode == EngineMode::Manual {
            let draft = self.draft_endpoint.read(cx).value();
            let clean = normalize_draft_endpoint(&draft);
            if let Err(reason) = validated_draft_endpoint(&clean) {
                // #213: a draft the client cannot use is refused before
                // anything is applied or persisted — the dialog stays open
                // with the reason in its callout, the in-modal twin of the
                // Electron reference's `settingsIssue` refusal. The refusal
                // settles under a freshly claimed token, so an in-flight
                // probe cannot land afterwards and overwrite the reason
                // (the same ownership rule as the empty-draft probe
                // failure in `test_connection`).
                let _ = self.probe_sequencer.begin();
                self.probe = Some(ConnectionProbe::Done {
                    endpoint: clean,
                    outcome: ProbeOutcome::Failed { message: reason },
                });
                cx.notify();
                return;
            }
            Some(clean)
        } else {
            None
        };
        // #221: the shortcut swap is the last check that can refuse the
        // save, so it runs before any field is applied: a platform refusal
        // keeps the previous shortcut registered, nothing else changes,
        // and the dialog stays open with the reason. Mid-take the swap
        // waits for the take to end; a failure then keeps the old shortcut
        // running and says why.
        if self.activation.is_active() {
            self.pending_shortcut = Some(shortcut.clone());
        } else {
            self.pending_shortcut = None;
            if let Err(reason) = self.apply_shortcut(shortcut.clone()) {
                self.dictation_draft_error = Some(reason);
                cx.notify();
                return;
            }
        }
        if let Some(clean) = manual_endpoint {
            self.endpoint = clean;
            // R02: only a save that changes the model marks it user-set, so an
            // endpoint-only edit keeps the server's health auto-sync alive.
            let draft_model = self.draft_model.read(cx).value();
            self.user_set_model =
                user_set_model_after_save(self.user_set_model, &self.model, &draft_model);
            self.model = draft_model;
        }
        self.expected_terms_input = self.draft_terms.read(cx).value();
        // A new processing configuration takes effect for the next job;
        // jobs already running keep the provider they started with.
        self.processing_settings = self.draft_processing_settings(cx);
        self.providers = processing::build_providers(&self.processing_settings);
        // The next take resolves against the saved choice; a take already
        // recording keeps the device it opened.
        self.microphone_settings.preferred_device = self.draft_microphone.clone();

        // #362: mode and backend override commit here, like the other
        // fields — and are applied now (the manager starts or stops with
        // the mode; a changed override moves the engine once, never on
        // an unchanged re-save).
        let previous_mode = self.engine_settings.mode;
        self.engine_settings.mode = engine_mode;
        self.engine_settings.backend_override = self.draft_backend_override.clone();
        if previous_mode != engine_mode {
            self.apply_engine_mode_change(cx);
        } else if engine_mode == EngineMode::Builtin
            && self.engine_settings.backend_override != self.applied_backend_override
        {
            if let Some(engine) = &self.engine {
                engine.set_backend_override(
                    self.engine_settings
                        .backend_override
                        .as_deref()
                        .and_then(backend_override_from_settings),
                );
                self.applied_backend_override = self.engine_settings.backend_override.clone();
            }
        }

        self.dictation_settings = DictationSettings {
            shortcut: shortcut.text().to_string(),
            activation: self.draft_activation,
            double_tap_hands_free: self.draft_double_tap,
        };
        self.activation
            .set_config(crate::activation::ActivationConfig::from_settings(&self.dictation_settings));

        // Applies from the next take on.
        self.live_preview = self.draft_live_preview;
        self.playback_settings = PlaybackSettings {
            during_recording: self.draft_playback_mode,
            lower_level_percent: self.draft_lower_level.read(cx).value(),
        };
        // The overlay follows at once; cues from the next cue on.
        self.feedback = FeedbackSettings {
            overlay: self.draft_overlay_mode,
            cues: self.draft_cues,
            cue_volume_percent: self.draft_cue_volume.read(cx).value(),
        };
        self.sync_overlay(cx);
        self.commit_storage_draft();
        // From the next delivery on; a take already running keeps its
        // capture and follows the new setting when it finishes.
        self.set_insertion_settings(self.draft_insertion);

        // R11: an unresolvable config directory is surfaced, not swallowed —
        // settings must not silently land in the current working directory.
        let path = match Settings::default_path() {
            Ok(path) => path,
            Err(err) => {
                self.error = Some(format!("Could not save settings: {err}"));
                cx.notify();
                return;
            }
        };
        self.spawn_settings_save(path, cx);

        // Close through the same path as Cancel and the scrim: retire any
        // in-flight probe with the dialog it belonged to (#207).
        self.close_settings(cx);
        // Re-check health from the saved configuration, as saves always
        // did — explicitly against the committed endpoint, and sequenced
        // so a check against an endpoint an earlier save committed is
        // dropped when this one supersedes it (#207). Builtin mode has
        // no manual endpoint to probe: its indicator is the engine.
        if self.engine_settings.mode == EngineMode::Manual {
            self.check_health(
                HealthCheckPurpose::Live,
                self.endpoint.clone(),
                cx,
            );
        }
    }

    pub fn select_session(&mut self, id: String, cx: &mut Context<Self>) {
        if self.selected_id.as_deref() != Some(id.as_str()) {
            self.selection_moved();
        }
        self.selected_id = Some(id.clone());
        self.restore_unsaved_staging(&id, cx);
        self.load_processing(id, cx);
        cx.notify();
    }

    /// Everything a move off the current selection must reset, in one place
    /// so an explicit click (`select_session`) and a listing-driven change
    /// (`apply_sessions`) cannot drift apart (review finding 1):
    /// playback of the old take stops, and the per-selection ephemeral
    /// state (#214.2's Copied/Saved flags, #208's armed delete) is
    /// dropped.
    fn selection_moved(&mut self) {
        self.stop_playback();
        self.selection_changed();
    }

    /// Ephemeral state that describes the *selected* take, so a selection
    /// change must drop it (#214.2): the Copied/Saved flips were earned on
    /// the previous take and would otherwise bleed onto the new one, and an
    /// armed delete confirmation (#208) must never carry over to a take the
    /// user never armed.
    fn selection_changed(&mut self) {
        self.copied = None;
        self.wav_saved = None;
        self.confirm_delete_id = None;
    }

    /// Whether the drawer's delete button is in its armed "Confirm delete?"
    /// state for the selected take (B05, #208). Derived, so an arm left
    /// behind by any selection change bypassing `select_session` is inert.
    pub fn delete_armed(&self) -> bool {
        effective_delete_arm(self.confirm_delete_id.as_deref(), self.selected_id.as_deref())
    }

    /// Whether a delete job for `id` is still in flight (#214.3).
    pub fn is_deleting(&self, id: &str) -> bool {
        self.deleting_ids.contains(id)
    }

    /// B05, #208: the trash button never deletes straight away. The first
    /// click arms a "Confirm delete?" state for exactly that take; only a
    /// second click on the same take's armed button reaches
    /// [`remove_session`]. The arm is cleared on confirm, and dropped by
    /// every selection change ([`selection_moved`]).
    pub fn request_delete_session(&mut self, id: String, cx: &mut Context<Self>) {
        match delete_intent(self.confirm_delete_id.as_deref(), &id) {
            DeleteIntent::Delete => {
                self.confirm_delete_id = None;
                // Review finding 4: the confirmed delete can still be
                // refused (transcription started, a delete already in
                // flight, or the store is gone). Surface that instead of
                // silently doing nothing — nothing was deleted.
                if !self.remove_session(id, cx) {
                    self.error = Some(
                        "This take cannot be deleted right now — it is still transcribing, a \
                         delete is already in flight, or the session store is unavailable. \
                         Nothing was deleted."
                            .to_string(),
                    );
                    cx.notify();
                }
            }
            DeleteIntent::Arm => {
                // Review finding 5: arm only the take the drawer is
                // showing, so the stored arm always equals the selection —
                // correctness must not depend on the drawer's rendering
                // reach.
                if self.selected_id.as_deref() == Some(id.as_str()) {
                    self.confirm_delete_id = Some(id);
                }
                cx.notify();
            }
        }
    }

    /// Deletes one recording. Returns whether a delete job actually
    /// started; `false` means the request was refused (transcription in
    /// flight, a delete already in flight, or no store).
    pub fn remove_session(&mut self, id: String, cx: &mut Context<Self>) -> bool {
        if self.active_ids.contains(&id) || self.deleting_ids.contains(&id) {
            // #214.3: a delete already in flight for this take ignores the
            // click — the second job would only race the first into a
            // `NotFound` and overwrite an already-vanished row with a
            // spurious "Could not delete the recording" error.
            return false;
        }
        if self.playing_id.as_deref() == Some(id.as_str()) {
            self.stop_playback();
        }
        let Some(store) = self.store.clone() else {
            return false;
        };
        // Delete during processing: the job is cancelled and its draft
        // deleted, so a late result lands nowhere.
        self.drop_processing(&id);
        self.deleting_ids.insert(id.clone());
        cx.spawn(async move |this, cx| {
            let deleted = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move {
                    // R21: the confirmed delete (B05's "permanently removes
                    // the audio" warning) takes the capture journal with it
                    // — v1 quarantines the manifest-linked journal before
                    // the row removal; v2's delete_capture quarantines the
                    // audio journal and tombstones the row — so startup
                    // recovery can never resurrect the take. The R05 stash
                    // path never comes through here: only this entry point
                    // tombstones.
                    store.delete(&id)
                })
                .await
            };
            match deleted {
                Ok(()) => {
                    // Review finding 2: the in-flight slot is deliberately
                    // NOT released here. The refresh may fail or land late
                    // while the row is still on screen, and the
                    // double-click guard must hold until a listing that
                    // actually omits the row arrives — that release lives
                    // in `apply_sessions`.
                    refresh_sessions(&this, &store, cx).await;
                }
                Err(err) => {
                    this.update(cx, |app, cx| {
                        // A failed delete leaves the recording in place, so
                        // release the slot here and let the user retry.
                        app.deleting_ids.remove(&id);
                        app.error = Some(format!("Could not delete the recording: {err}"));
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
        true
    }

    pub fn copy_transcript(&mut self, cx: &mut Context<Self>) {
        // Review finding 6: the clipboard write and its Copied flag are
        // one operation on one selected take — either both happen or
        // neither does.
        let Some(id) = self.selected_id.clone() else {
            return;
        };
        // The take's head: the used processed text, or the raw transcript.
        let Some(text) = self.head_text(&id) else {
            self.note_head_loading(&id, cx);
            return;
        };
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        // #214.2: the flag names the take it was earned on, so it can never
        // show "Copied" on a different selection.
        self.copied = Some(id.clone());
        self.schedule_flag_reset(EphemeralFlag::Copied(id), cx);
        cx.notify();
    }

    pub fn export_transcript(&mut self, cx: &mut Context<Self>) {
        let Some(session) = self.selected() else {
            return;
        };
        let name = format!(
            "starling-{}.txt",
            session.created_at.replace([':', '.'], "-")
        );
        let Some(text) = self.head_text(&session.id) else {
            let id = session.id.clone();
            self.note_head_loading(&id, cx);
            return;
        };
        // Re-exportable from history, so no fsync (see write_download_exclusive);
        // a transcript export flips no Saved flag.
        self.write_download(
            name,
            Arc::new(text.into_bytes()),
            None,
            false,
            cx,
        );
    }

    pub fn export_audio(&mut self, cx: &mut Context<Self>) {
        self.export_audio_as(false, cx);
    }

    /// Export the selected take's audio as WAV, or as lossless FLAC
    /// (#356) — the same samples either way.
    pub fn export_audio_as(&mut self, flac: bool, cx: &mut Context<Self>) {
        let Some(session) = self.selected() else {
            return;
        };
        let name = format!(
            "starling-{}.{}",
            session.created_at.replace([':', '.'], "-"),
            if flac { "flac" } else { "wav" }
        );
        let id = session.id.clone();
        // G02: history holds metadata only — fetch this one recording's
        // audio on demand (a damaged record surfaces its reason here).
        let Some(store) = self.store.clone() else {
            return;
        };
        cx.spawn(async move |this, cx| {
            let loaded = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move {
                    if flac {
                        store.audio_flac(&id)
                    } else {
                        store.audio_wav(&id)
                    }
                })
                .await
            };
            this.update(cx, |app, cx| match loaded {
                Ok(Some(wav)) => {
                    // The drawer's "Saved" flag belongs to its WAV button.
                    let saved_for = (!flac).then(|| id.clone());
                    app.write_download(name, wav, saved_for, true, cx);
                }
                Ok(None) => {
                    app.error = Some(format!("Recording {id} was not found."));
                    cx.notify();
                }
                Err(err) => {
                    app.error = Some(err.to_string());
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    pub fn export_unsaved_audio(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(capture) = self.unsaved.iter().find(|capture| capture.id == id) else {
            return;
        };
        let name = format!(
            "starling-unsaved-{}-{}.wav",
            capture.created_at.replace([':', '.'], "-"),
            &capture.id[capture.id.len().saturating_sub(8)..]
        );
        let wav = capture.wav.clone();
        // The only copy of the recording: fsync it. Flips no Saved flag —
        // the unsaved banner is not the drawer's WAV button.
        self.write_download(name, wav, None, true, cx);
    }

    /// Whether the discard button is in its armed "Confirm discard" state
    /// for the current batch of unsaved takes (#214.4).
    pub fn discard_armed(&self) -> bool {
        discard_intent(self.confirm_discard_for, unsaved_batch_token(&self.unsaved))
            == DiscardIntent::Execute
    }

    pub fn discard_unsaved(&mut self, cx: &mut Context<Self>) {
        match discard_intent(self.confirm_discard_for, unsaved_batch_token(&self.unsaved)) {
            DiscardIntent::Execute => {
                self.unsaved.clear();
                self.error = None;
                self.confirm_discard_for = None;
            }
            // Also the re-arm path: an arm carried over from an earlier
            // batch (a take stashed since) is dead and becomes a fresh
            // arm for the batch now in the banner.
            DiscardIntent::Arm => {
                self.confirm_discard_for = Some(unsaved_batch_token(&self.unsaved));
            }
        }
        cx.notify();
    }

    fn write_download(
        &mut self,
        name: String,
        bytes: Arc<Vec<u8>>,
        saved_for: Option<String>,
        sync: bool,
        cx: &mut Context<Self>,
    ) {
        cx.spawn(async move |this, cx| {
            let requested = name.clone();
            let result = cx
                .background_spawn(async move {
                    let dir = dirs::download_dir().unwrap_or_else(|| PathBuf::from("."));
                    write_download_exclusive(&dir, &requested, bytes.as_slice(), sync)
                        .map(|path| (dir, path))
                })
                .await;
            this.update(cx, |app, cx| match result {
                Ok((dir, path)) => {
                    // G05: an export that had to change names is surfaced,
                    // never silently written next to the file it dodged.
                    // Its own notice slot, so a capture warning (or another
                    // export notice) can no longer overwrite it mid-read.
                    if path.file_name().and_then(|file| file.to_str()) != Some(name.as_str()) {
                        let landed = path
                            .file_name()
                            .and_then(|file| file.to_str())
                            .unwrap_or_default();
                        app.export_notice = Some(format!(
                            "Exported as {landed} — {name} already existed in {} and was left \
                             untouched.",
                            dir.display()
                        ));
                        app.schedule_export_notice_reset(cx);
                    }
                    // Review finding 7: move the owner in, clone it exactly
                    // once — the second use moves it on into the timer.
                    if let Some(owner) = saved_for {
                        // #214.2: the Saved flag names the take it was
                        // earned on, so it can never show on a different
                        // selection.
                        app.wav_saved = Some(owner.clone());
                        app.schedule_flag_reset(EphemeralFlag::WavSaved(owner), cx);
                    }
                    cx.notify();
                }
                Err(err) => {
                    app.error = Some(format!("Could not save the file: {err}"));
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    /// Clears the export notice after a read-through window, on the same
    /// ephemeral-flag pattern as `copied`/`wav_saved`. Longer than those
    /// flips because the notice names two files the user may need to tell
    /// apart.
    fn schedule_export_notice_reset(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            Timer::after(Duration::from_millis(6_000)).await;
            this.update(cx, |app, cx| {
                app.export_notice = None;
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Clears one ephemeral drawer flag after its read-through window.
    /// The timer only clears the flag while it still names the take it was
    /// earned for (#214.2): a reset scheduled for take A's copy must not
    /// wipe the "Copied" flip take B earned while A's timer was running.
    fn schedule_flag_reset(&mut self, flag: EphemeralFlag, cx: &mut Context<Self>) {
        let delay = match &flag {
            EphemeralFlag::Copied(_) => 1_400,
            EphemeralFlag::WavSaved(_) => 2_000,
        };
        cx.spawn(async move |this, cx| {
            Timer::after(Duration::from_millis(delay)).await;
            this.update(cx, |app, cx| {
                match &flag {
                    EphemeralFlag::Copied(owner) => {
                        if app.copied.as_deref() == Some(owner.as_str()) {
                            app.copied = None;
                        }
                    }
                    EphemeralFlag::WavSaved(owner) => {
                        if app.wav_saved.as_deref() == Some(owner.as_str()) {
                            app.wav_saved = None;
                        }
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn toggle_play(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(player) = self.player.as_ref() else {
            return;
        };
        let known = self.sessions.iter().any(|session| session.id == id);
        if !known {
            return;
        }
        if self.playing_id.as_deref() == Some(id) {
            player.stop();
            self.playing_id = None;
            self.retire_playback_generation();
            cx.notify();
            return;
        }

        // Stop whatever is playing now; the new clip starts once its audio
        // has been fetched (G02: history holds metadata only, so playback
        // loads one recording's WAV on demand — a damaged record surfaces
        // its reason through the same path).
        player.stop();
        self.playing_id = None;
        self.retire_playback_generation();
        cx.notify();

        let Some(store) = self.store.clone() else {
            return;
        };
        let id = id.to_string();
        cx.spawn(async move |this, cx| {
            let loaded = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move { store.audio_wav(&id) }).await
            };
            this.update(cx, |app, cx| {
                match loaded {
                    Ok(Some(wav)) => {
                        let Some(player) = app.player.as_ref() else {
                            return;
                        };
                        match player.play(wav.as_slice()) {
                            Ok(()) => {
                                app.playing_id = Some(id.clone());
                                // New playback, new generation: any watcher
                                // still polling for the previous playback is
                                // stale from here on.
                                app.retire_playback_generation();
                                app.watch_playback(cx);
                            }
                            Err(err) => {
                                app.playing_id = None;
                                app.error = Some(err.to_string());
                            }
                        }
                    }
                    Ok(None) => {
                        app.error = Some(format!("Recording {id} was not found."));
                    }
                    Err(err) => {
                        app.error = Some(err.to_string());
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn stop_playback(&mut self) {
        if let Some(player) = self.player.as_ref() {
            player.stop();
        }
        self.playing_id = None;
        self.retire_playback_generation();
    }

    /// Invalidate every playback watcher spawned so far (G04).
    ///
    /// Watchers capture the generation when their playback starts; bumping it
    /// makes their next tick [`PlaybackWatch::Cancelled`], so stop,
    /// replacement, selection change, and session deletion each end the
    /// previous polling task instead of leaving it polling forever.
    fn retire_playback_generation(&mut self) {
        self.playback_generation = self.playback_generation.wrapping_add(1);
    }

    fn watch_playback(&mut self, cx: &mut Context<Self>) {
        let generation = self.playback_generation;
        cx.spawn(async move |this, cx| {
            loop {
                Timer::after(Duration::from_millis(250)).await;
                // A failed update means the entity is destroyed: stop polling.
                let watch = this
                    .update(cx, |app, cx| {
                        let watch = playback_watch(
                            generation,
                            app.playback_generation,
                            app.playing_id.is_some(),
                            app.player.as_ref().map(|player| player.is_playing()),
                        );
                        if watch == PlaybackWatch::Finished {
                            // Only the watcher of the current generation ever
                            // lands here, so a stale watcher cannot clear
                            // `playing_id` for a newer playback.
                            app.playing_id = None;
                            cx.notify();
                        }
                        watch
                    })
                    .unwrap_or(PlaybackWatch::Cancelled);
                if watch != PlaybackWatch::Poll {
                    break;
                }
            }
        })
        .detach();
    }

    pub fn fidelity_warnings(&self) -> Vec<String> {
        let Some(transcript) = self
            .selected()
            .and_then(|session| session.transcript.as_ref())
        else {
            return Vec::new();
        };
        let mut settings = Settings::default_settings();
        settings.set_expected_terms_input(&self.expected_terms_input);
        let analysis = fidelity::analyze_transcript(
            &transcript.text,
            &TranscriptAnalysisOptions {
                expected_terms: settings.expected_terms,
                recording_duration_seconds: None,
                covered_duration_seconds: None,
            },
        );
        analysis
            .warnings
            .into_iter()
            .map(|warning| warning.message)
            .collect()
    }

    pub fn transcript_scale(&self, viewport: Pixels) -> Pixels {
        theme::clamp_px(viewport * 0.02, 18., 27.)
    }
}

impl Render for StarlingApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some((started, printed)) = self.diagnostics.as_mut() {
            if !*printed {
                *printed = true;
                println!(
                    "STARLING_DIAGNOSTICS {{\"readyToShowMs\":{},\"rssBytes\":{}}}",
                    started.elapsed().as_millis(),
                    rss_bytes()
                );
            }
        }

        // The overlay is placed in this window's scale (#221).
        self.overlay.scale = window.scale_factor();
        if let Some(handle) = self.recorder.as_mut() {
            let window_samples = handle.latest_window(1024);
            let magnitudes = fft::magnitude_spectrum(&window_samples);
            self.levels = fft::waveform_levels(&magnitudes, 52);
            self.elapsed_ms = handle.elapsed().as_secs_f64() * 1000.0;
            window.request_animation_frame();
        }
        self.tick_mic_check(window);

        match self.pending_focus.take() {
            Some(PendingFocus::Staging) => match self.staging.as_ref() {
                Some(staging) => window.focus(&staging.editor.focus_handle(cx)),
                None => window.focus(&self.root_focus),
            },
            Some(PendingFocus::Root) => window.focus(&self.root_focus),
            None => {}
        }

        let has_transcript = self.selected().is_some();
        let root_focus = self.root_focus.clone();

        div()
            .id("starling-root")
            .track_focus(&root_focus)
            .key_context("Starling")
            // #221: shortcut presses arrive through the keystroke
            // interceptor (`activation.rs`), which sees them before any
            // binding; releases come here, in the capture phase, so no
            // child can swallow the end of a hold.
            .capture_key_up(cx.listener(|this, event: &gpui::KeyUpEvent, _window, cx| {
                if this.shortcut_key_up(&event.keystroke, cx) {
                    // The press was swallowed by the interceptor; its
                    // release must not reach the focused editor either.
                    cx.stop_propagation();
                }
            }))
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(theme::BG)
            .text_color(theme::INK)
            .child(views::render_topbar(self, window, cx))
            .child(views::render_workspace(self, window, cx))
            .when(has_transcript, |root| {
                root.child(views::render_drawer(self, window, cx))
            })
            .when(self.settings_open, |root| {
                root.child(views::render_settings_modal(self, window, cx))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starling_dictation::storage::SessionStatus;

    /// A fresh scratch directory under the system temp dir, removed first so
    /// reruns start clean. Each test uses its own tag.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("starling-g05-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn the_backend_override_parses_the_settings_keys_only() {
        // #362: "cpu"/"vulkan" map to the engine families; anything else
        // (a settings file from a future build) degrades to automatic
        // instead of pinning the wrong engine family.
        assert_eq!(backend_override_from_settings("cpu"), Some(Backend::Cpu));
        assert_eq!(backend_override_from_settings("vulkan"), Some(Backend::Vulkan));
        assert_eq!(backend_override_from_settings("cuda"), None);
        assert_eq!(backend_override_from_settings(""), None);
    }

    #[test]
    fn an_older_settings_save_is_skipped_only_behind_a_successful_newer_one() {
        // #366: the newest document that SAVED successfully must be the one
        // on disk. A newer write that failed lands nothing, so an older
        // writer that runs after it must still write; a newer successful
        // write retires every older one.
        assert!(settings_save_should_write(2, 0), "nothing written yet");
        assert!(settings_save_should_write(2, 1),
            "the newer write failed (written stayed at 1): the older one must still land");
        assert!(!settings_save_should_write(1, 2),
            "a newer document was written successfully: the older one must not land after it");
        assert!(!settings_save_should_write(2, 2),
            "this very sequence already landed");
    }

    #[test]
    fn candidate_names_suffix_before_the_extension() {
        assert_eq!(download_name_candidate("starling-a.txt", 1), "starling-a.txt");
        assert_eq!(
            download_name_candidate("starling-2026-a.wav", 2),
            "starling-2026-a-2.wav"
        );
        assert_eq!(download_name_candidate("starling-a.wav", 10), "starling-a-10.wav");
        // Extension-less names take the suffix at the end.
        assert_eq!(download_name_candidate("readme", 3), "readme-3");
        // A leading dot is the stem, not an extension.
        assert_eq!(download_name_candidate(".zshrc", 2), ".zshrc-2");
    }

    #[test]
    fn watcher_polls_only_while_the_current_generation_is_audibly_playing() {
        assert_eq!(
            playback_watch(7, 7, true, Some(true)),
            PlaybackWatch::Poll
        );
    }

    #[test]
    fn natural_drain_or_a_missing_player_finishes_the_current_watcher() {
        // Player drained: the current watcher clears playing_id and exits.
        assert_eq!(
            playback_watch(7, 7, true, Some(false)),
            PlaybackWatch::Finished
        );
        // Player vanished mid-playback: release the recorded playback.
        assert_eq!(playback_watch(7, 7, true, None), PlaybackWatch::Finished);
    }

    #[test]
    fn a_stale_watcher_cancels_instead_of_clearing_newer_playback() {
        // Superseded generation: must never report Finished — not even when
        // nothing is audibly playing — or it would clear the newer
        // playback's playing_id.
        assert_eq!(
            playback_watch(6, 7, true, Some(false)),
            PlaybackWatch::Cancelled
        );
        assert_eq!(
            playback_watch(6, 7, true, None),
            PlaybackWatch::Cancelled
        );
        assert_eq!(
            playback_watch(6, 7, true, Some(true)),
            PlaybackWatch::Cancelled
        );
        // Generation wrap-around still compares unequal.
        assert_eq!(
            playback_watch(u64::MAX, 0, true, Some(true)),
            PlaybackWatch::Cancelled
        );
    }

    #[test]
    fn playback_ended_by_any_other_path_cancels_the_current_watcher() {
        // playing_id cleared with the generation otherwise unchanged (stop,
        // selection change, session deletion): the watcher must exit rather
        // than poll forever waiting for playing_id to return.
        assert_eq!(
            playback_watch(7, 7, false, Some(true)),
            PlaybackWatch::Cancelled
        );
        assert_eq!(
            playback_watch(7, 7, false, Some(false)),
            PlaybackWatch::Cancelled
        );
        assert_eq!(playback_watch(7, 7, false, None), PlaybackWatch::Cancelled);
    }

    #[test]
    fn a_save_that_leaves_the_model_untouched_keeps_auto_sync() {
        // R02: editing only the endpoint (or terms) must not mark the model
        // user-set, or one unrelated save would permanently disable the
        // health auto-sync.
        assert!(!user_set_model_after_save(
            false,
            "whisper-large-v3",
            "whisper-large-v3"
        ));
    }

    #[test]
    fn a_save_that_changes_the_model_marks_it_user_set() {
        assert!(user_set_model_after_save(
            false,
            "parakeet",
            "whisper-large-v3"
        ));
    }

    #[test]
    fn the_user_set_model_flag_is_sticky_across_later_saves() {
        // Once the user chose a model, an endpoint-only save must not
        // silently hand the choice back to the server.
        assert!(user_set_model_after_save(
            true,
            "whisper-large-v3",
            "whisper-large-v3"
        ));
    }

    #[test]
    fn the_quality_banner_prefers_the_capture_warning_over_an_export_notice() {
        // G05: the two notices live in separate fields, so neither can
        // overwrite the other — but the banner shows at most one, and a
        // take's clipping evidence outranks a one-off rename notice.
        assert_eq!(
            banner_notice(Some("heavily clipped"), Some("exported as -2")),
            Some("heavily clipped")
        );
        assert_eq!(
            banner_notice(None, Some("exported as -2")),
            Some("exported as -2")
        );
        assert_eq!(
            banner_notice(Some("heavily clipped"), None),
            Some("heavily clipped")
        );
        assert_eq!(banner_notice(None, None), None);
    }

    #[test]
    fn exclusive_write_errors_without_overwriting_when_all_names_are_taken() {
        let dir = scratch_dir("exhausted");
        // Seed every candidate the policy would try: the name plus -2..-1000.
        for attempt in 1..=MAX_DOWNLOAD_NAME_ATTEMPTS {
            let candidate = download_name_candidate("starling-t.txt", attempt);
            std::fs::write(dir.join(&candidate), format!("seed {attempt}"))
                .expect("seed candidate");
        }

        let result = write_download_exclusive(&dir, "starling-t.txt", b"NEW", false);
        assert!(result.is_err(), "exhausted candidates error out");

        // The very first file is still the original seed, byte for byte.
        assert_eq!(
            std::fs::read(dir.join("starling-t.txt")).expect("read original"),
            b"seed 1"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exclusive_write_lands_on_a_free_name() {
        let dir = scratch_dir("free");
        let path = write_download_exclusive(&dir, "starling-t.wav", b"NEW", true)
            .expect("write succeeds");
        assert_eq!(path, dir.join("starling-t.wav"));
        assert_eq!(std::fs::read(&path).expect("read back"), b"NEW");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exclusive_write_never_overwrites_existing_content() {
        let dir = scratch_dir("keep-old");
        std::fs::write(dir.join("starling-t.txt"), b"OLD TRANSCRIPT")
            .expect("seed the existing export");

        let path = write_download_exclusive(&dir, "starling-t.txt", b"NEW TRANSCRIPT", false)
            .expect("dodges instead of failing");

        // The user's existing file is byte-for-byte untouched.
        assert_eq!(
            std::fs::read(dir.join("starling-t.txt")).expect("read original"),
            b"OLD TRANSCRIPT"
        );
        assert_eq!(path, dir.join("starling-t-2.txt"));
        assert_eq!(
            std::fs::read(&path).expect("read new export"),
            b"NEW TRANSCRIPT"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exclusive_write_walks_past_a_chain_of_collisions() {
        let dir = scratch_dir("chain");
        std::fs::write(dir.join("starling-t.txt"), b"1").expect("seed");
        std::fs::write(dir.join("starling-t-2.txt"), b"2").expect("seed");

        let path = write_download_exclusive(&dir, "starling-t.txt", b"3", true).expect("third name");
        assert_eq!(path, dir.join("starling-t-3.txt"));
        assert_eq!(std::fs::read(dir.join("starling-t.txt")).expect("original"), b"1");
        assert_eq!(std::fs::read(dir.join("starling-t-2.txt")).expect("second"), b"2");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exclusive_write_surfaces_unwritable_targets_as_errors() {
        let missing = std::env::temp_dir().join("starling-g05-no-such-dir");
        let _ = std::fs::remove_dir_all(&missing);
        assert!(
            write_download_exclusive(&missing, "starling-t.txt", b"x", false).is_err(),
            "a missing directory must surface an error, not create files elsewhere"
        );
    }

    #[cfg(unix)]
    #[test]
    fn exclusive_write_does_not_follow_a_symlink_planted_at_the_name() {
        let dir = scratch_dir("symlink");
        let target = dir.join("target.txt");
        std::fs::write(&target, b"PRECIOUS").expect("seed symlink target");
        std::os::unix::fs::symlink(&target, dir.join("starling-t.txt")).expect("plant symlink");

        let path =
            write_download_exclusive(&dir, "starling-t.txt", b"EXPORT", true).expect("dodges symlink");

        // The symlink and its target are untouched; the export landed beside it.
        assert_eq!(std::fs::read(&target).expect("target content"), b"PRECIOUS");
        assert_eq!(path, dir.join("starling-t-2.txt"));
        assert_eq!(std::fs::read(&path).expect("export content"), b"EXPORT");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn summary(id: &str) -> SessionSummary {
        SessionSummary {
            id: id.to_string(),
            created_at: "2026-09-20T00:00:00.000Z".to_string(),
            updated_at: "2026-09-20T00:00:00.000Z".to_string(),
            status: SessionStatus::Captured,
            duration_ms: None,
            attempt_count: 0,
            transcript: None,
            last_error: None,
            model_label: None,
            journal_id: None,
            archival: false,
            interrupted: false,
            confirmed_ms: None,
            results: Vec::new(),
        }
    }

    fn damaged_record(id: &str, reason: &str) -> DamagedRecord {
        DamagedRecord {
            id: id.to_string(),
            reason: reason.to_string(),
        }
    }

    #[test]
    fn a_listing_splits_into_summaries_and_damaged_records() {
        // G02: both kinds arrive in one listing; the app keeps the readable
        // takes in order and the damaged flags beside them.
        let (sessions, damaged) = split_listing(vec![
            ListedRecord::Session(summary("good")),
            ListedRecord::Damaged(damaged_record("torn", "recording.wav: not found")),
            ListedRecord::Session(summary("recovered")),
        ]);

        assert_eq!(
            sessions.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            ["good", "recovered"]
        );
        assert_eq!(damaged.len(), 1);
        assert_eq!(damaged[0].id, "torn");
        assert_eq!(damaged[0].reason, "recording.wav: not found");
    }

    #[test]
    fn the_selection_survives_while_the_record_stays_readable() {
        let sessions = vec![summary("newest"), summary("selected")];
        assert_eq!(
            next_selection(Some("selected"), &sessions),
            Some("selected".to_string())
        );
    }

    #[test]
    fn a_selection_that_became_damaged_falls_back_to_the_newest_take() {
        // G02: the store flagged the selected record between refreshes —
        // the drawer must not stay pointed at a record it can no longer
        // read, and must not show nothing either.
        let sessions = vec![summary("newest"), summary("next")];
        assert_eq!(
            next_selection(Some("now-damaged"), &sessions),
            Some("newest".to_string())
        );
    }

    #[test]
    fn no_sessions_leaves_no_selection() {
        assert_eq!(next_selection(Some("gone"), &[]), None);
        assert_eq!(next_selection(None, &[]), None);
    }

    #[test]
    fn a_first_trash_click_arms_and_a_second_on_the_same_take_confirms() {
        // B05, #208: the first click on a take's trash button may never
        // delete; only the second click on that same button does.
        assert_eq!(delete_intent(None, "take-a"), DeleteIntent::Arm);
        assert_eq!(delete_intent(Some("take-a"), "take-a"), DeleteIntent::Delete);
    }

    #[test]
    fn an_armed_delete_never_confirms_for_a_different_take() {
        // #208 with #214.4's lesson applied: an arm belongs to the take it
        // was pressed on. A click on another take's trash while the first
        // is armed re-arms for the new take instead of deleting anything.
        assert_eq!(delete_intent(Some("take-a"), "take-b"), DeleteIntent::Arm);
    }

    #[test]
    fn an_armed_delete_is_inert_once_the_selection_moves_on() {
        // #208: the arm dies with the selection it was armed under — also
        // when the selection changed without passing through
        // `select_session` (a fresh take jumping the drawer).
        assert!(effective_delete_arm(Some("take-a"), Some("take-a")));
        assert!(!effective_delete_arm(Some("take-a"), Some("take-b")));
        assert!(!effective_delete_arm(Some("take-a"), None));
        assert!(!effective_delete_arm(None, Some("take-a")));
    }

    fn unsaved(id: &str) -> UnsavedWav {
        UnsavedWav {
            id: id.to_string(),
            wav: Arc::new(Vec::new()),
            created_at: "2026-09-21T00:00:00.000Z".to_string(),
        }
    }

    #[test]
    fn an_armed_discard_executes_only_for_the_batch_it_was_armed_for() {
        // Two clicks on the unchanged batch discard it.
        let batch = vec![unsaved("unsaved-1")];
        let token = unsaved_batch_token(&batch);
        // Stable for an unchanged batch (review finding 3).
        assert_eq!(unsaved_batch_token(&batch), token);
        assert_eq!(discard_intent(None, token), DiscardIntent::Arm);
        assert_eq!(discard_intent(Some(token), token), DiscardIntent::Execute);
    }

    #[test]
    fn a_take_stashed_after_arming_disarms_the_discard_confirmation() {
        // #214.4: arm "Confirm discard", change your mind; a new failed
        // save stashes another take. The carried-over arm must be dead:
        // one click re-arms for the new batch, never single-click-
        // discards audio the user never armed for.
        let armed = unsaved_batch_token(&[unsaved("unsaved-1")]);
        let after_stash = unsaved_batch_token(&[unsaved("unsaved-1"), unsaved("unsaved-2")]);
        assert_eq!(discard_intent(Some(armed), after_stash), DiscardIntent::Arm);
        // Re-arming the grown batch executes on it.
        assert_eq!(
            discard_intent(Some(after_stash), after_stash),
            DiscardIntent::Execute
        );
        // A third stash kills that arm again.
        let after_third =
            unsaved_batch_token(&[unsaved("unsaved-1"), unsaved("unsaved-2"), unsaved("unsaved-3")]);
        assert_eq!(
            discard_intent(Some(after_stash), after_third),
            DiscardIntent::Arm
        );
    }

    #[test]
    fn the_discard_batch_token_covers_any_batch_mutation_not_just_growth() {
        // Review finding 3: the token identifies the batch by content and
        // order, so a future insert, removal, or reorder — not only a size
        // change — invalidates an armed confirmation.
        let armed = unsaved_batch_token(&[unsaved("unsaved-1"), unsaved("unsaved-2")]);
        let reordered = unsaved_batch_token(&[unsaved("unsaved-2"), unsaved("unsaved-1")]);
        assert_ne!(armed, reordered);
        assert_eq!(discard_intent(Some(armed), reordered), DiscardIntent::Arm);
    }

    #[test]
    fn a_listing_that_loses_the_selected_take_moves_the_selection() {
        // Review finding 1: when a refresh drops the selected take
        // (deleted from another window, flagged damaged), the drawer moves
        // — and the move must run the same reset as an explicit click,
        // including stopping playback of the vanished take.
        let sessions = vec![summary("next")];
        assert!(listing_moves_selection(Some("gone"), &sessions));
        assert!(listing_moves_selection(Some("gone"), &[]));
        // A listing that keeps the selection is not a move.
        assert!(!listing_moves_selection(Some("next"), &sessions));
        assert!(!listing_moves_selection(
            Some("selected"),
            &[summary("newest"), summary("selected")]
        ));
        // Gaining a first selection runs the same reset as clicking that
        // row with nothing selected — exactly what `select_session` does.
        assert!(listing_moves_selection(None, &sessions));
    }

    #[test]
    fn an_in_flight_delete_is_released_only_when_a_listing_confirms_it_gone() {
        // Review finding 2: the delete job's own completion releases
        // nothing — the refresh may fail or land late while the row is
        // still on screen. Only a listing that omits the row releases the
        // id; a row still listed keeps its guard up.
        let sessions = vec![summary("still-listed")];
        let deleting: HashSet<String> = ["still-listed", "deleted-elsewhere"]
            .iter()
            .map(|id| id.to_string())
            .collect();
        let surviving = surviving_deletes(&deleting, &sessions);
        assert!(surviving.contains("still-listed"));
        assert!(!surviving.contains("deleted-elsewhere"));
        // A listing where everything landed removes every guard.
        assert!(surviving_deletes(&deleting, &[]).is_empty());
    }

    // --- #207: the health-probe subsystem ---

    fn health(
        model: Option<&str>,
        busy: Option<bool>,
        queue_depth: Option<f64>,
    ) -> client::ServerHealth {
        client::ServerHealth {
            status: "ok".to_string(),
            phase: None,
            model: model.map(str::to_string),
            loaded: None,
            busy,
            queue_depth,
        }
    }

    #[test]

    fn only_the_newest_sequencer_token_may_report() {
        // #207 defect 3: save endpoint A (slow probe), then B (fast) — B
        // lands first, and A's late result must be dropped, not layered on
        // top. The token a check claims on begin is retired by the next
        // begin, so the older landing compares unequal.
        let mut sequencer = CheckSequencer::new();
        let slow = sequencer.begin();
        let fast = sequencer.begin();
        assert!(sequencer.is_current(fast));
        assert!(
            !sequencer.is_current(slow),
            "an older check must be retired by a newer one"
        );
    }

    #[test]

    fn cancel_all_retires_every_in_flight_token() {
        let mut sequencer = CheckSequencer::new();
        let first = sequencer.begin();
        let second = sequencer.begin();
        sequencer.cancel_all();
        assert!(!sequencer.is_current(first));
        assert!(
            !sequencer.is_current(second),
            "closing the dialog retires even the newest probe"
        );
        // A later check still works and is current.
        let third = sequencer.begin();
        assert!(sequencer.is_current(third));
    }

    #[test]
    fn live_checks_and_probes_sequence_independent_families() {
        // B06: a live check and a dialog probe may be in flight at once;
        // beginning one must never silence the other, but each family's
        // own begin retires that family's older tokens.
        let mut health = CheckSequencer::new();
        let mut probe = CheckSequencer::new();
        let live_token = health.begin();
        let probe_token = probe.begin();
        assert!(health.is_current(live_token));
        assert!(probe.is_current(probe_token));
        let newer_live = health.begin();
        assert!(!health.is_current(live_token));
        assert!(probe.is_current(probe_token), "a live check never silences a probe");
        assert!(health.is_current(newer_live));
    }

    #[test]
    fn a_press_while_the_newest_probe_is_testing_is_dropped() {
        // #207 review: the disabled button, enforced at the state layer —
        // and a dropped press must not disturb the in-flight probe's
        // token, or it would retire the very probe it is deferring to.
        let mut sequencer = CheckSequencer::new();
        let in_flight = sequencer.begin();
        let testing = ConnectionProbe::Testing {
            endpoint: "http://draft:9000".to_string(),
        };
        assert_eq!(
            probe_press(&mut sequencer, Some(&testing), "http://other:1"),
            None,
            "a press during Testing is dropped, not run"
        );
        assert!(
            sequencer.is_current(in_flight),
            "a dropped press retires nothing"
        );
    }

    #[test]

    fn a_settled_or_absent_probe_lets_the_press_claim_the_next_token() {
        let mut sequencer = CheckSequencer::new();
        let previous = sequencer.begin();
        let settled = ConnectionProbe::Done {
            endpoint: "http://draft:9000".to_string(),
            outcome: ProbeOutcome::Ok {
                model: "parakeet".to_string(),
                busy: false,
            },
        };
        match probe_press(&mut sequencer, Some(&settled), "  http://next:9000/ ") {
            Some((token, endpoint)) => {
                assert!(sequencer.is_current(token));
                assert_eq!(
                    endpoint, "http://next:9000",
                    "the press normalizes the draft like a save"
                );
            }
            other => panic!("a settled probe must not block a new press, got {other:?}"),
        }
        assert!(!sequencer.is_current(previous));
    }

    #[test]
    fn an_empty_draft_press_claims_the_token_that_retires_the_in_flight_probe() {
        // #207 review: the immediate failure an empty draft settles is
        // owned by a freshly claimed token, so the slower probe it
        // interrupted cannot land afterwards and overwrite the newer
        // outcome — the empty path goes through the sequencer like every
        // other outcome.
        let mut sequencer = CheckSequencer::new();
        let in_flight = sequencer.begin();
        match probe_press(&mut sequencer, None, "   ") {
            Some((token, endpoint)) => {
                assert_eq!(endpoint, "");
                assert!(sequencer.is_current(token));
                assert!(
                    !sequencer.is_current(in_flight),
                    "the stale Testing probe has nowhere to land"
                );
            }
            other => panic!("an empty draft still settles a probe outcome, got {other:?}"),
        }
    }

    #[test]
    fn without_a_probe_the_callout_shows_the_live_committed_status() {
        let ready =
            settings_callout_view(None, Connection::Ready, "http://committed:8181");
        assert_eq!(ready.dot, Connection::Ready);
        assert_eq!(ready.title, "Server connected");
        assert_eq!(ready.detail, "http://committed:8181");

        let offline =
            settings_callout_view(None, Connection::Offline, "http://committed:8181");
        assert_eq!(offline.dot, Connection::Offline);
        assert_eq!(offline.title, "Server needs attention");
    }

    #[test]
    fn a_pending_live_check_shows_a_neutral_callout_not_a_failure() {
        // #207 review: right after launch, or while the re-check a save
        // just triggered is in flight, the live connection is Checking —
        // the no-probe fallback must read as pending, not as a server
        // that "needs attention".
        let view = settings_callout_view(None, Connection::Checking, "http://committed:8181");
        assert_eq!(view.dot, Connection::Checking);
        assert_eq!(view.title, "Checking server…");
        assert_eq!(view.detail, "http://committed:8181");
    }

    #[test]
    fn a_testing_probe_owns_the_callout_with_its_own_endpoint() {
        let view = settings_callout_view(
            Some(&ConnectionProbe::Testing {
                endpoint: "http://draft:9000".to_string(),
            }),
            Connection::Offline,
            "http://committed:8181",
        );
        assert_eq!(view.dot, Connection::Checking);
        assert_eq!(view.title, "Testing connection…");
        assert_eq!(view.detail, "http://draft:9000");
    }

    #[test]
    fn a_failed_probe_reports_its_own_failure_not_the_live_status() {
        // #207 defect 2, at the pure boundary: the callout shows the
        // probe's own outcome and endpoint. The live state handed in is
        // only the no-probe fallback — a mistyped draft can never paint
        // the committed connection offline.
        let view = settings_callout_view(
            Some(&ConnectionProbe::Done {
                endpoint: "http://draft:9000".to_string(),
                outcome: ProbeOutcome::Failed {
                    message:
                        "error sending request for url (http://draft:9000/health)".to_string(),
                },
            }),
            Connection::Ready,
            "http://committed:8181",
        );
        assert_eq!(view.dot, Connection::Offline);
        assert_eq!(view.title, "Probe failed");
        assert_eq!(
            view.detail,
            "error sending request for url (http://draft:9000/health)"
        );
    }

    #[test]
    fn a_settled_probe_reports_the_probed_model_and_busyness() {
        let idle = settings_callout_view(
            Some(&ConnectionProbe::Done {
                endpoint: "http://draft:9000".to_string(),
                outcome: ProbeOutcome::Ok {
                    model: "whisper-large-v3".to_string(),
                    busy: false,
                },
            }),
            Connection::Offline,
            "http://committed:8181",
        );
        assert_eq!(idle.dot, Connection::Ready);
        assert_eq!(idle.title, "whisper-large-v3 responded");
        assert_eq!(idle.detail, "http://draft:9000");

        let busy = settings_callout_view(
            Some(&ConnectionProbe::Done {
                endpoint: "http://draft:9000".to_string(),
                outcome: ProbeOutcome::Ok {
                    model: "whisper-large-v3".to_string(),
                    busy: true,
                },
            }),
            Connection::Ready,
            "http://committed:8181",
        );
        assert_eq!(busy.dot, Connection::Busy);
        assert_eq!(busy.title, "whisper-large-v3 is working");
        assert_eq!(busy.detail, "http://draft:9000");
    }

    #[test]
    fn a_probe_outcome_defaults_the_model_and_derives_busyness() {
        // A health snapshot without a model still names the server
        // "server"; a queued job means busy even when the flag is absent.
        assert_eq!(
            probe_outcome_from_health(&health(None, Some(false), Some(1.0))),
            ProbeOutcome::Ok {
                model: "server".to_string(),
                busy: true
            }
        );
        assert_eq!(
            probe_outcome_from_health(&health(Some("parakeet"), None, None)),
            ProbeOutcome::Ok {
                model: "parakeet".to_string(),
                busy: false
            }
        );
    }

    #[test]
    fn a_live_check_success_clears_the_banner_and_syncs_an_openai_model() {
        let writes = live_check_writes(
            HealthCheckPurpose::Live,
            false,
            &Ok(health(Some("parakeet"), Some(false), None)),
        );
        assert_eq!(writes.connection, Connection::Ready);
        assert_eq!(writes.server_model.as_deref(), Some("parakeet"));
        assert_eq!(writes.model_sync.as_deref(), Some("parakeet"));
        assert_eq!(
            writes.error_slot,
            Some(None),
            "a live success clears the banner"
        );
    }

    #[test]
    fn a_live_check_failure_owns_the_banner_with_the_failure() {
        let writes = live_check_writes(
            HealthCheckPurpose::Live,
            false,
            &Err("connection refused".to_string()),
        );
        assert_eq!(writes.connection, Connection::Offline);
        assert_eq!(
            writes.error_slot,
            Some(Some("connection refused".to_string()))
        );
        assert_eq!(writes.model_sync, None);
        assert_eq!(
            writes.server_model, None,
            "a failed check leaves the model label alone"
        );
    }

    #[test]
    fn a_diagnostic_check_never_touches_the_banner() {
        // #207 defect 4: the R13 probe after a transport-class job failure
        // may correct the badge, but the job's explanation for the missing
        // transcript must survive both a successful and a failed probe.
        let success = live_check_writes(
            HealthCheckPurpose::Diagnostic,
            false,
            &Ok(health(None, Some(false), None)),
        );
        assert_eq!(success.connection, Connection::Ready);
        assert_eq!(success.error_slot, None);
        assert_eq!(
            success.model_sync, None,
            "only Live checks auto-sync the model"
        );

        let failure = live_check_writes(
            HealthCheckPurpose::Diagnostic,
            false,
            &Err("connection refused".to_string()),
        );
        assert_eq!(failure.connection, Connection::Offline);
        assert_eq!(failure.error_slot, None);
    }

    #[test]
    fn the_model_auto_sync_needs_a_reported_model_and_no_user_choice() {
        // A user-set model or a response without a model leaves the
        // committed model alone.
        let cases = [(true, Some("parakeet")), (false, None)];
        for (user_set, model) in cases {
            let writes = live_check_writes(
                HealthCheckPurpose::Live,
                user_set,
                &Ok(health(model, None, None)),
            );
            assert_eq!(
                writes.model_sync,
                None,
                "no auto-sync for user_set={user_set} model={model:?}"
            );
        }
    }

    #[test]
    fn busyness_comes_from_the_flag_or_a_queue_depth() {
        let flagged = live_check_writes(
            HealthCheckPurpose::Live,
            false,
            &Ok(health(None, Some(true), None)),
        );
        assert_eq!(flagged.connection, Connection::Busy);
        let queued = live_check_writes(
            HealthCheckPurpose::Live,
            false,
            &Ok(health(None, None, Some(2.0))),
        );
        assert_eq!(queued.connection, Connection::Busy);
    }

    #[test]
    fn a_draft_endpoint_normalizes_like_a_save() {
        // #207 defect 1: the probe must target exactly what saving the
        // draft would commit — same trim, same trailing-slash removal.
        assert_eq!(
            normalize_draft_endpoint("  http://127.0.0.1:8181/ "),
            "http://127.0.0.1:8181"
        );
        assert_eq!(
            normalize_draft_endpoint("http://host:8181///"),
            "http://host:8181"
        );
        assert_eq!(normalize_draft_endpoint(""), "");
    }

    /// The save's exact two steps (#213): normalize, then validate — one
    /// place, so the tests exercise the same contract the save does.
    fn validated(draft: &str) -> Result<(), String> {
        validated_draft_endpoint(&normalize_draft_endpoint(draft))
    }

    #[test]
    fn a_usable_draft_endpoint_passes_the_save_validation_normalized() {
        // #213: a save commits the trimmed, slash-dropped shape — the same
        // values the Electron reference's `normalizeSettings` returns for
        // a usable draft.
        assert_eq!(validated("  http://127.0.0.1:8181/ "), Ok(()));
        assert_eq!(validated("https://example.net/api"), Ok(()));
    }

    #[test]
    fn an_empty_draft_endpoint_is_refused_with_the_reference_wording() {
        // The `normalizeSettings` refusal, verbatim — the save no longer
        // returns silently on an empty draft (#213).
        assert_eq!(validated("   "), Err(EMPTY_ENDPOINT_REASON.to_string()));
        assert_eq!(validated(""), Err(EMPTY_ENDPOINT_REASON.to_string()));
    }

    #[test]
    fn a_draft_the_client_cannot_use_is_refused_with_its_reason() {
        // #213: the same rules the client applies to a committed endpoint,
        // with the same messages — "localhost 8181" has no URL base, a
        // wrong scheme and embedded credentials each name their own fix.
        assert_eq!(
            validated("localhost 8181"),
            Err("Invalid server endpoint.".to_string())
        );
        assert_eq!(
            validated("ftp://x"),
            Err("Server endpoint must use http or https.".to_string())
        );
        assert_eq!(
            validated("http://user:pass@host:8181"),
            Err("Put credentials in a trusted proxy, not the endpoint URL.".to_string())
        );
    }
}
