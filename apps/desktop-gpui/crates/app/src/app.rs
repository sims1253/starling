//! The StarlingApp entity: state, behaviors, and async flows ported from
//! `apps/desktop/src/App.tsx`.

use std::{
    collections::HashSet,
    path::PathBuf,
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

fn user_set_model_flag() -> bool {
    std::fs::read_to_string(Settings::default_path())
        .map(|text| text.contains("\"model\""))
        .unwrap_or(false)
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
            user_set_model: user_set_model_flag(),
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
        self.model = self.draft_model.read(cx).value();
        self.expected_terms_input = self.draft_terms.read(cx).value();
        self.user_set_model = true;

        let mut settings = Settings {
            endpoint: self.endpoint.clone(),
            protocol: self.protocol,
            model: self.model.clone(),
            expected_terms: Vec::new(),
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
            let result = cx
                .background_spawn(async move {
                    let dir = dirs::download_dir().unwrap_or_else(|| PathBuf::from("."));
                    std::fs::write(dir.join(&name), bytes.as_slice())
                })
                .await;
            this.update(cx, |app, cx| match result {
                Ok(()) => {
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
        } else {
            player.stop();
            match player.play(session.wav.as_slice()) {
                Ok(()) => {
                    self.playing_id = Some(id.to_string());
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
    }

    fn watch_playback(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            loop {
                Timer::after(Duration::from_millis(250)).await;
                let finished = this
                    .update(cx, |app, cx| {
                        let finished = app.playing_id.is_some()
                            && app
                                .player
                                .as_ref()
                                .is_none_or(|player| !player.is_playing());
                        if finished {
                            app.playing_id = None;
                            cx.notify();
                        }
                        finished
                    })
                    .unwrap_or(true);
                if finished {
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
