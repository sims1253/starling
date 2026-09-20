//! The StarlingApp entity: state, behaviors, and async flows ported from
//! `apps/desktop/src/App.tsx`.

use std::{
    collections::HashSet,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use gpui::{
    AppContext, ClipboardItem, Context, Entity, FocusHandle, Pixels, Render, Timer, Window,
    actions, div, prelude::*,
};
use starling_dictation::{
    client::{self, StarlingClient},
    fft,
    fidelity::{self, TranscriptAnalysisOptions},
    player::Player,
    recorder::RecorderHandle,
    settings::{self, Settings},
    storage::{DictationSession, FileSessionStore, SessionStatus},
};

use crate::{input::TextField, theme, upload::refresh_sessions, views};

actions!(starling, [ToggleRecording]);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Connection {
    Checking,
    Ready,
    Busy,
    Offline,
}

pub struct UnsavedWav {
    pub id: String,
    pub wav: Arc<Vec<u8>>,
    pub created_at: String,
}

pub struct StarlingApp {
    pub store: Option<Arc<FileSessionStore>>,
    pub store_error: Option<String>,
    pub player: Option<Player>,
    pub root_focus: FocusHandle,

    pub endpoint: String,
    pub protocol: settings::Protocol,
    pub model: String,
    pub expected_terms_input: String,
    pub user_set_model: bool,

    pub settings_open: bool,
    pub settings_protocol: settings::Protocol,
    pub draft_endpoint: Entity<TextField>,
    pub draft_model: Entity<TextField>,
    pub draft_terms: Entity<TextField>,

    pub connection: Connection,
    pub server_model: String,

    pub sessions: Vec<DictationSession>,
    pub selected_id: Option<String>,
    pub active_ids: HashSet<String>,
    pub error: Option<String>,
    pub capture_warning: Option<String>,
    pub unsaved: Vec<UnsavedWav>,
    pub confirm_discard: bool,
    pub copied: bool,
    pub wav_saved: bool,
    pub playing_id: Option<String>,
    /// Identifies the current playback so poll-watchers can detect that they
    /// are stale (G04). Bumped whenever playback starts, stops, or is
    /// replaced; watchers capture the value at spawn time.
    pub playback_generation: u64,

    pub recorder: Option<RecorderHandle>,
    pub levels: Vec<f32>,
    pub elapsed_ms: f64,

    pub diagnostics: Option<(Instant, bool)>,
}

pub(crate) fn client_protocol(protocol: settings::Protocol) -> client::Protocol {
    match protocol {
        settings::Protocol::Starling => client::Protocol::Starling,
        settings::Protocol::OpenAI => client::Protocol::OpenAi,
    }
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
                rest.trim()
                    .split_whitespace()
                    .next()
                    .and_then(|kb| kb.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
        })
        .unwrap_or(0)
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
fn write_download_exclusive(dir: &Path, name: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
    for attempt in 1..=MAX_DOWNLOAD_NAME_ATTEMPTS {
        let candidate = download_name_candidate(name, attempt);
        let path = dir.join(&candidate);

        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                if let Err(err) = file.write_all(bytes).and_then(|()| file.sync_all()) {
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
        let endpoint = settings.endpoint.clone();
        let protocol = settings.protocol;
        let model = settings.model.clone();
        let terms_input = settings.expected_terms_input();
        let (store, store_error) = match FileSessionStore::open(FileSessionStore::default_root()) {
            Ok(store) => (Some(Arc::new(store)), None),
            Err(err) => (
                None,
                Some(format!("Could not open saved recordings: {err}")),
            ),
        };
        let player = Player::new().ok();
        let draft_endpoint = cx.new(|cx| TextField::new("http://127.0.0.1:8181", &endpoint, cx));
        let draft_model = cx.new(|cx| TextField::new("parakeet", &model, cx));
        let draft_terms = cx.new(|cx| TextField::new("auth, Starling, GGUF", &terms_input, cx));

        Self {
            error: store_error.clone(),
            capture_warning: None,
            store,
            store_error,
            player,
            root_focus: cx.focus_handle(),
            endpoint,
            protocol,
            model,
            expected_terms_input: terms_input,
            user_set_model: settings.user_set_model,
            settings_open: false,
            settings_protocol: settings.protocol,
            draft_endpoint,
            draft_model,
            draft_terms,
            connection: Connection::Checking,
            server_model: "server".to_string(),
            sessions: Vec::new(),
            selected_id: None,
            active_ids: HashSet::new(),
            unsaved: Vec::new(),
            confirm_discard: false,
            copied: false,
            wav_saved: false,
            playing_id: None,
            playback_generation: 0,
            recorder: None,
            levels: vec![0.06; 52],
            elapsed_ms: 0.0,
            diagnostics: diagnostics.then_some((started, false)),
        }
    }

    pub fn init(&mut self, cx: &mut Context<Self>) {
        if let Some(store) = self.store.clone() {
            let list_store = store.clone();
            let fix_store = store.clone();
            cx.spawn(async move |this, cx| {
                let listed = cx
                    .background_spawn(async move { list_store.list() })
                    .await;
                match listed {
                    Ok(sessions) => {
                        let interrupted: Vec<String> = sessions
                            .iter()
                            .filter(|session| session.status == SessionStatus::Transcribing)
                            .map(|session| session.id.clone())
                            .collect();
                        if !interrupted.is_empty() {
                            let fix = cx.background_spawn(async move {
                                for id in interrupted {
                                    let _ = fix_store.save_failure(
                                        &id,
                                        "Interrupted before the server returned a transcript. Your audio is ready to retry.",
                                    );
                                }
                            });
                            fix.await;
                        }
                        refresh_sessions(&this, &store, cx).await;
                    }
                    Err(err) => {
                        this.update(cx, |app, cx| {
                            app.error = Some(format!("Could not open saved recordings: {err}"));
                            cx.notify();
                        })
                        .ok();
                    }
                }
            })
            .detach();
        }
        self.check_health(self.endpoint.clone(), cx);
    }

    pub fn busy(&self) -> bool {
        !self.active_ids.is_empty()
    }

    pub fn selected(&self) -> Option<&DictationSession> {
        self.selected_id
            .as_ref()
            .and_then(|id| self.sessions.iter().find(|session| &session.id == id))
    }

    pub fn is_active(&self, id: &str) -> bool {
        self.active_ids.contains(id)
    }

    pub fn apply_sessions(&mut self, list: Vec<DictationSession>) {
        let keep = self
            .selected_id
            .as_ref()
            .is_some_and(|current| list.iter().any(|session| &session.id == current));
        if !keep {
            self.selected_id = list.first().map(|session| session.id.clone());
        }
        self.sessions = list;
    }

    pub fn check_health(&mut self, endpoint: String, cx: &mut Context<Self>) {
        let protocol = client_protocol(self.protocol);
        let model = self.model.clone();
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let client = StarlingClient::new(&endpoint, protocol, &model)
                        .and_then(|client| client.with_timeout_ms(5_000))
                        .map_err(|err| err.to_string())?;
                    client.health().map_err(|err| err.to_string())
                })
                .await;
            this.update(cx, |app, cx| {
                match result {
                    Ok(health) => {
                        app.server_model =
                            health.model.clone().unwrap_or_else(|| "server".to_string());
                        if protocol == client::Protocol::OpenAi
                            && let Some(server_model) = &health.model
                            && !app.user_set_model
                            && !server_model.is_empty()
                        {
                            app.model = server_model.clone();
                        }
                        app.connection = if health.busy.unwrap_or(false)
                            || health.queue_depth.unwrap_or(0.0) > 0.0
                        {
                            Connection::Busy
                        } else {
                            Connection::Ready
                        };
                        app.error = None;
                    }
                    Err(message) => {
                        app.connection = Connection::Offline;
                        app.error = Some(message);
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn open_settings(&mut self, cx: &mut Context<Self>) {
        self.settings_open = true;
        self.settings_protocol = self.protocol;
        let endpoint = self.endpoint.clone();
        let model = self.model.clone();
        let terms = self.expected_terms_input.clone();
        self.draft_endpoint.update(cx, |field, cx| {
            field.set_value(&endpoint, cx);
        });
        self.draft_model.update(cx, |field, cx| {
            field.set_value(&model, cx);
        });
        self.draft_terms.update(cx, |field, cx| {
            field.set_value(&terms, cx);
        });
        cx.notify();
    }

    pub fn close_settings(&mut self, cx: &mut Context<Self>) {
        self.settings_open = false;
        cx.notify();
    }

    pub fn test_connection(&mut self, cx: &mut Context<Self>) {
        let draft = self.draft_endpoint.read(cx).value();
        self.check_health(draft, cx);
    }

    pub fn save_settings(&mut self, cx: &mut Context<Self>) {
        let draft = self.draft_endpoint.read(cx).value();
        let clean = draft.trim().trim_end_matches('/').to_string();
        if clean.is_empty() {
            return;
        }
        self.endpoint = clean.clone();
        self.protocol = self.settings_protocol;
        // R02: only a save that changes the model marks it user-set, so an
        // endpoint-only edit keeps the server's health auto-sync alive.
        let draft_model = self.draft_model.read(cx).value();
        self.user_set_model =
            user_set_model_after_save(self.user_set_model, &self.model, &draft_model);
        self.model = draft_model;
        self.expected_terms_input = self.draft_terms.read(cx).value();

        let mut settings = Settings {
            endpoint: self.endpoint.clone(),
            protocol: self.protocol,
            model: self.model.clone(),
            expected_terms: Vec::new(),
            user_set_model: self.user_set_model,
        };
        settings.set_expected_terms_input(&self.expected_terms_input);

        cx.spawn(async move |this, cx| {
            let path = Settings::default_path();
            let saved = cx
                .background_spawn(async move { settings.save(&path) })
                .await;
            if let Err(err) = saved {
                this.update(cx, |app, cx| {
                    app.error = Some(format!("Could not save settings: {err}"));
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();

        self.settings_open = false;
        self.check_health(clean, cx);
    }

    pub fn select_session(&mut self, id: String, cx: &mut Context<Self>) {
        if self.selected_id.as_deref() != Some(id.as_str()) {
            self.stop_playback();
        }
        self.selected_id = Some(id);
        cx.notify();
    }

    pub fn remove_session(&mut self, id: String, cx: &mut Context<Self>) {
        if self.active_ids.contains(&id) {
            return;
        }
        if self.playing_id.as_deref() == Some(id.as_str()) {
            self.stop_playback();
        }
        let Some(store) = self.store.clone() else {
            return;
        };
        cx.spawn(async move |this, cx| {
            let deleted = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move { store.delete(&id) }).await
            };
            match deleted {
                Ok(()) => refresh_sessions(&this, &store, cx).await,
                Err(err) => {
                    this.update(cx, |app, cx| {
                        app.error = Some(format!("Could not delete the recording: {err}"));
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    pub fn copy_transcript(&mut self, cx: &mut Context<Self>) {
        let Some(transcript) = self
            .selected()
            .and_then(|session| session.transcript.clone())
        else {
            return;
        };
        cx.write_to_clipboard(ClipboardItem::new_string(transcript.text));
        self.copied = true;
        self.schedule_flag_reset(true, false, cx);
        cx.notify();
    }

    pub fn export_transcript(&mut self, cx: &mut Context<Self>) {
        let Some(session) = self.selected() else {
            return;
        };
        let Some(transcript) = session.transcript.clone() else {
            return;
        };
        let name = format!(
            "starling-{}.txt",
            session.created_at.replace([':', '.'], "-")
        );
        self.write_download(name, Arc::new(transcript.text.into_bytes()), false, cx);
    }

    pub fn export_audio(&mut self, cx: &mut Context<Self>) {
        let Some(session) = self.selected() else {
            return;
        };
        let name = format!(
            "starling-{}.wav",
            session.created_at.replace([':', '.'], "-")
        );
        let wav = session.wav.clone();
        self.write_download(name, wav, true, cx);
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
        self.write_download(name, wav, false, cx);
    }

    pub fn discard_unsaved(&mut self, cx: &mut Context<Self>) {
        if self.confirm_discard {
            self.unsaved.clear();
            self.error = None;
            self.confirm_discard = false;
        } else {
            self.confirm_discard = true;
        }
        cx.notify();
    }

    fn write_download(
        &mut self,
        name: String,
        bytes: Arc<Vec<u8>>,
        mark_saved: bool,
        cx: &mut Context<Self>,
    ) {
        cx.spawn(async move |this, cx| {
            let requested = name.clone();
            let result = cx
                .background_spawn(async move {
                    let dir = dirs::download_dir().unwrap_or_else(|| PathBuf::from("."));
                    write_download_exclusive(&dir, &requested, bytes.as_slice())
                        .map(|path| (dir, path))
                })
                .await;
            this.update(cx, |app, cx| match result {
                Ok((dir, path)) => {
                    // G05: an export that had to change names is surfaced,
                    // never silently written next to the file it dodged.
                    if path.file_name().and_then(|file| file.to_str()) != Some(name.as_str()) {
                        let landed = path
                            .file_name()
                            .and_then(|file| file.to_str())
                            .unwrap_or_default();
                        app.capture_warning = Some(format!(
                            "Exported as {landed} — {name} already existed in {} and was left \
                             untouched.",
                            dir.display()
                        ));
                    }
                    if mark_saved {
                        app.wav_saved = true;
                        app.schedule_flag_reset(false, true, cx);
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

    fn schedule_flag_reset(&mut self, copied: bool, wav_saved: bool, cx: &mut Context<Self>) {
        let delay = if copied { 1400 } else { 2000 };
        cx.spawn(async move |this, cx| {
            Timer::after(Duration::from_millis(delay)).await;
            this.update(cx, |app, cx| {
                if copied {
                    app.copied = false;
                }
                if wav_saved {
                    app.wav_saved = false;
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
        let Some(session) = self.sessions.iter().find(|session| session.id == id) else {
            return;
        };
        if self.playing_id.as_deref() == Some(id) {
            player.stop();
            self.playing_id = None;
            self.retire_playback_generation();
        } else {
            player.stop();
            match player.play(session.wav.as_slice()) {
                Ok(()) => {
                    self.playing_id = Some(id.to_string());
                    // New playback, new generation: any watcher still polling
                    // for the previous playback is stale from here on.
                    self.retire_playback_generation();
                    self.watch_playback(cx);
                }
                Err(err) => {
                    self.playing_id = None;
                    self.error = Some(err.to_string());
                }
            }
        }
        cx.notify();
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

        if let Some(handle) = self.recorder.as_mut() {
            let window_samples = handle.latest_window(1024);
            let magnitudes = fft::magnitude_spectrum(&window_samples);
            self.levels = fft::waveform_levels(&magnitudes, 52);
            self.elapsed_ms = handle.elapsed().as_secs_f64() * 1000.0;
            window.request_animation_frame();
        }

        let has_transcript = self.selected().is_some();
        let root_focus = self.root_focus.clone();

        div()
            .id("starling-root")
            .track_focus(&root_focus)
            .key_context("Starling")
            .on_action(cx.listener(|this, _: &ToggleRecording, _window, cx| {
                this.toggle_recording(cx);
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

    /// A fresh scratch directory under the system temp dir, removed first so
    /// reruns start clean. Each test uses its own tag.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("starling-g05-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
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
    fn exclusive_write_errors_without_overwriting_when_all_names_are_taken() {
        let dir = scratch_dir("exhausted");
        // Seed every candidate the policy would try: the name plus -2..-1000.
        for attempt in 1..=MAX_DOWNLOAD_NAME_ATTEMPTS {
            let candidate = download_name_candidate("starling-t.txt", attempt);
            std::fs::write(dir.join(&candidate), format!("seed {attempt}"))
                .expect("seed candidate");
        }

        let result = write_download_exclusive(&dir, "starling-t.txt", b"NEW");
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
        let path = write_download_exclusive(&dir, "starling-t.wav", b"NEW")
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

        let path = write_download_exclusive(&dir, "starling-t.txt", b"NEW TRANSCRIPT")
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

        let path = write_download_exclusive(&dir, "starling-t.txt", b"3").expect("third name");
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
            write_download_exclusive(&missing, "starling-t.txt", b"x").is_err(),
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
            write_download_exclusive(&dir, "starling-t.txt", b"EXPORT").expect("dodges symlink");

        // The symlink and its target are untouched; the export landed beside it.
        assert_eq!(std::fs::read(&target).expect("target content"), b"PRECIOUS");
        assert_eq!(path, dir.join("starling-t-2.txt"));
        assert_eq!(std::fs::read(&path).expect("export content"), b"EXPORT");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
