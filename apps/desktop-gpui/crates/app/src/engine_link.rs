//! The app's side of the runtime host's engine (#220).
//!
//! The host owns the bundled-engine manager (#362, #363): this window
//! runs no engine. Settings → Engine's actions and engine settings are
//! requests to the host ([`EngineRequest`]), answered on a background
//! task; what they set in motion — a download's progress, a switch's
//! stages, a refusal for lack of memory — arrives as the [`EngineStatus`]
//! the host pushes to every watching window, which the views render.
//!
//! The window keeps its committed engine settings in step with the host:
//! another window may have changed the engine, and a later Save here must
//! not undo that. While this window's own `Configure` is unanswered, a
//! pushed status may predate it, so its settings are not adopted then.

use std::sync::Arc;

use gpui::{AppContext, Context};
use starling_dictation::engine::{EnginePhase, EngineSnapshot};
use starling_dictation::settings::EngineMode;
use starling_runtime_host::client::HostClient;
use starling_runtime_host::engine::{EngineIntent, EngineReply, EngineRequest, EngineStatus};

use crate::app::{backend_override_from_settings, Connection, StarlingApp};
use crate::views;

/// What a request that could not be sent says.
const NOT_CONNECTED: &str = "Starling's recording service is not connected, so the engine \
                             cannot be changed right now.";

impl StarlingApp {
    /// The engine settings this window has committed, as the host takes
    /// them.
    pub(crate) fn engine_intent(&self) -> EngineIntent {
        EngineIntent {
            mode: self.engine_settings.mode,
            active_model: self.engine_settings.active_model.clone(),
            backend_override: self
                .engine_settings
                .backend_override
                .as_deref()
                .and_then(backend_override_from_settings),
            endpoint: self.endpoint.clone(),
            model: self.model.clone(),
        }
    }

    /// Has the host's engine follow the committed engine settings now
    /// (mode, backend, the user's server). The host only acts on what
    /// changed. Without a connection the settings file — which the
    /// caller persists — carries them to the host.
    pub(crate) fn configure_engine(&mut self, cx: &mut Context<Self>) {
        let request = EngineRequest::Configure {
            intent: self.engine_intent(),
        };
        if self.host.client.is_none() {
            return;
        }
        self.engine_configuring += 1;
        self.send_engine_request(request, cx, |app, reply, cx| {
            app.engine_configuring = app.engine_configuring.saturating_sub(1);
            match reply {
                Ok(EngineReply::Done { revision } | EngineReply::Activating { revision, .. }) => {
                    app.engine_revision = app.engine_revision.max(revision);
                }
                Ok(EngineReply::Refused { message }) => app.error = Some(message),
                Err(message) => app.error = Some(message),
            }
            cx.notify();
        });
    }

    /// Sends `request` to the host's engine; a refusal (or no answer) is
    /// shown in the error banner.
    pub(crate) fn engine_request(&mut self, request: EngineRequest, cx: &mut Context<Self>) {
        self.send_engine_request(request, cx, |app, reply, cx| {
            match reply {
                Ok(EngineReply::Done { .. } | EngineReply::Activating { .. }) => {}
                Ok(EngineReply::Refused { message }) => app.error = Some(message),
                Err(message) => app.error = Some(message),
            }
            cx.notify();
        });
    }

    /// Sends `request` on a background task and hands its answer (or why
    /// there is none, as a sentence) to `then`.
    pub(crate) fn send_engine_request(
        &mut self,
        request: EngineRequest,
        cx: &mut Context<Self>,
        then: impl FnOnce(&mut StarlingApp, Result<EngineReply, String>, &mut Context<Self>) + 'static,
    ) {
        let client: Option<Arc<HostClient>> = self.host.client.clone();
        cx.spawn(async move |this, cx| {
            let reply = match client {
                Some(client) => cx
                    .background_spawn(async move { client.engine(request) })
                    .await
                    .map_err(|err| {
                        format!("Starling's recording service did not answer about the engine ({err}).")
                    }),
                None => Err(NOT_CONNECTED.to_string()),
            };
            this.update(cx, |app, cx| then(app, reply, cx)).ok();
        })
        .detach();
    }

    /// The host reported its engine (on connect, and on every change).
    pub(crate) fn engine_status_update(&mut self, status: EngineStatus, cx: &mut Context<Self>) {
        // The settings another window (or a hand edit) gave the engine
        // become this window's committed ones — unless this window's own
        // change may not have reached the host yet.
        // In manual mode the indicator is this window's own probe of the
        // server: a server (or mode) taken over from the host is probed.
        let mut probe = false;
        if self.engine_configuring == 0 && status.revision >= self.engine_revision {
            if self.engine_settings.mode != status.mode {
                if self.draft_engine_mode == self.engine_settings.mode {
                    self.draft_engine_mode = status.mode;
                }
                if status.mode == EngineMode::Builtin {
                    self.retire_manual_probe();
                } else {
                    probe = true;
                }
                self.engine_settings.mode = status.mode;
            }
            if let Some((endpoint, model)) = &status.server {
                if (&self.endpoint, &self.model) != (endpoint, model) {
                    self.endpoint = endpoint.clone();
                    self.model = model.clone();
                    probe = true;
                }
            }
            if status.mode == EngineMode::Builtin {
                let backend = status.backend_override.map(|backend| backend.as_str().to_string());
                if self.engine_settings.backend_override != backend {
                    if self.draft_backend_override == self.engine_settings.backend_override {
                        self.draft_backend_override = backend.clone();
                    }
                    self.engine_settings.backend_override = backend;
                }
            }
        }
        // The engine is the source of truth for the active model; the
        // window that asked for it writes it to the settings file (the
        // file only restores it at launch).
        if let Some(active) = status
            .snapshot
            .as_ref()
            .filter(|snapshot| snapshot.phase == EnginePhase::Ready)
            .and_then(|snapshot| snapshot.active.as_ref())
        {
            if self.engine_settings.active_model.as_deref() != Some(active.model_id.as_str()) {
                self.engine_settings.active_model = Some(active.model_id.clone());
                if self.engine_activating.as_deref() == Some(active.model_id.as_str()) {
                    self.persist_committed_settings(cx);
                }
            }
            if self.engine_activating.as_deref() == Some(active.model_id.as_str()) {
                self.engine_activating = None;
            }
        }
        self.engine_status = Some(status);
        if probe {
            self.connection = Connection::Checking;
            self.check_health(
                crate::app::HealthCheckPurpose::Live,
                self.endpoint.clone(),
                cx,
            );
        }
        if self.engine_settings.mode == EngineMode::Builtin {
            self.connection = match self.engine_snapshot() {
                Some(snapshot) => views::engine_status_view(&snapshot).connection,
                None => Connection::Offline,
            };
        }
        cx.notify();
    }

    /// The built-in engine's state as the host last reported it (`None`
    /// when it runs none — manual mode, an engine that cannot run, or no
    /// report yet: see [`Self::engine_unavailable`]).
    pub fn engine_snapshot(&self) -> Option<EngineSnapshot> {
        self.engine_status
            .as_ref()
            .filter(|status| status.mode == EngineMode::Builtin)
            .and_then(|status| status.snapshot.clone())
    }

    /// Why no built-in engine state can be shown: it cannot run, or the
    /// recording service has not reported it.
    pub(crate) fn engine_unavailable(&self) -> String {
        match &self.engine_status {
            Some(status) => status
                .unavailable
                .clone()
                .unwrap_or_else(|| "The built-in engine is not available.".to_string()),
            None => "The built-in engine runs in Starling's recording service, which has not \
                     reported it yet."
                .to_string(),
        }
    }

    /// Retires any in-flight manual health probe (the latest-wins rule a
    /// saved endpoint uses, #207): the built-in engine's status owns the
    /// indicator now, and a slow probe landing afterwards would overwrite
    /// it with Offline — and an error banner — for a server no longer in
    /// use.
    pub(crate) fn retire_manual_probe(&mut self) {
        let _ = self.health_sequencer.begin();
    }

    // ---- engine actions (#362, #363) ---------------------------------
    // All of these are immediate requests to the host's engine; the
    // status pushes paint the result. None of them is part of Save.

    /// Download (background) a model without activating it.
    pub fn engine_download(&mut self, id: &str, cx: &mut Context<Self>) {
        self.engine_request(
            EngineRequest::Download {
                model_id: id.to_string(),
            },
            cx,
        );
    }

    /// Cancel a model's running download.
    pub fn engine_cancel_download(&mut self, id: &str, cx: &mut Context<Self>) {
        self.engine_request(
            EngineRequest::CancelDownload {
                model_id: id.to_string(),
            },
            cx,
        );
    }

    /// Download-if-needed then switch to a model (#363).
    pub fn engine_activate(&mut self, id: &str, cx: &mut Context<Self>) {
        self.engine_activating = Some(id.to_string());
        self.engine_request(
            EngineRequest::Activate {
                model_id: id.to_string(),
            },
            cx,
        );
    }

    /// Delete a model's files; the engine refuses active/switching/
    /// downloading models and the refusal is surfaced, not swallowed.
    pub fn engine_delete_model(&mut self, id: &str, cx: &mut Context<Self>) {
        self.engine_request(
            EngineRequest::Delete {
                model_id: id.to_string(),
            },
            cx,
        );
    }

    /// Clear a Failed/crash-loop state and retry the last model — or,
    /// when the built-in engine could not run at all, try starting it
    /// again.
    pub fn engine_retry(&mut self, cx: &mut Context<Self>) {
        self.engine_request(EngineRequest::Retry, cx);
    }

    /// Answer a pending NeedsDrain decision: switch after the take.
    pub fn engine_confirm_drain_swap(&mut self, cx: &mut Context<Self>) {
        self.engine_request(EngineRequest::ConfirmDrainSwap, cx);
    }

    /// Cancel a running switch; the current model keeps serving.
    pub fn engine_cancel_switch(&mut self, cx: &mut Context<Self>) {
        self.engine_request(EngineRequest::CancelSwitch, cx);
    }

    /// The "Use CPU engine" / "Use automatic engine" toggle (#362):
    /// applies immediately (the host's engine reloads on the new backend,
    /// draining in-flight takes) AND persists immediately, like
    /// `engine_switch_to_manual` does for the mode — an immediate action
    /// must not wait behind Save, or a later Cancel would leave the
    /// running engine diverged from the saved settings. The dialog draft
    /// stays in sync, so an unchanged Save is a no-op and Cancel keeps
    /// what was applied (the draft resets from the committed value when
    /// the dialog reopens).
    pub fn engine_toggle_cpu(&mut self, cx: &mut Context<Self>) {
        let pinned = self.draft_backend_override.as_deref() == Some("cpu");
        let next = if pinned { None } else { Some("cpu".to_string()) };
        self.engine_settings.backend_override = next.clone();
        self.draft_backend_override = next;
        self.persist_committed_settings(cx);
        self.configure_engine(cx);
        cx.notify();
    }

    /// The failure action "Switch to my own server" (#362): an explicit
    /// user decision, so unlike the radio it applies and persists
    /// immediately rather than waiting for Save.
    pub fn engine_switch_to_manual(&mut self, cx: &mut Context<Self>) {
        self.draft_engine_mode = EngineMode::Manual;
        self.engine_settings.mode = EngineMode::Manual;
        self.persist_committed_settings(cx);
        self.configure_engine(cx);
        // The manual indicator starts as a probe in flight, exactly like
        // a startup in manual mode.
        self.connection = Connection::Checking;
        self.check_health(
            crate::app::HealthCheckPurpose::Live,
            self.endpoint.clone(),
            cx,
        );
    }
}
