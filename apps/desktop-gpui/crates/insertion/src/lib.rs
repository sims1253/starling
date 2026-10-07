//! `starling-insertion` — focus-safe text insertion (issue #221,
//! slice 2 phase A).
//!
//! This crate owns the *delivery end* of dictation: turning a finished
//! transcript into keystrokes in whatever external window the user was
//! focused on when the take started — without ever typing into the
//! wrong window. It deliberately knows nothing about audio, the
//! recorder (issue #222 owns that) or any UI: it is a small set of
//! platform backends behind one trait, plus the [`Inserter`] that
//! orders them and the `runtime::InsertionDeliveryAdapter` bridge
//! (feature `runtime`) that plugs them into the runtime's
//! `delivery.apply` path.
//!
//! # The one rule: never a blind write
//!
//! The runtime contract (packages/contracts/runtime-protocol) pins the
//! semantics: the target is revalidated *immediately before* apply, a
//! conflict becomes `delivery.conflict` instead of a write, and **no
//! Enter injection** ever happens. This crate enforces the same rules
//! one layer down, at the key sender itself, so no caller can bypass
//! them:
//!
//! - Every [`InsertionBackend::insert`] revalidates the snapshot as its
//!   first act and returns [`InsertError::TargetChanged`] /
//!   [`InsertError::TargetGone`] instead of typing into anything else.
//!   Focus may have moved anywhere between capture and insert (the take
//!   ran for minutes); the check happens last. And because focus can
//!   *keep* moving while a long text types, typing runs in chunks with
//!   the same revalidation repeated before every chunk — a change
//!   part-way stops the typing immediately and reports how much may
//!   already have landed ([`InsertError::PartialDelivery`]).
//! - Text containing any control character — `\n`, `\r`, `\t`, or any
//!   other Cc character — is refused with
//!   [`InsertError::MultilineUnsupported`] before a single key moves.
//!   Transcripts are single-line paragraphs in v1; sending Enter (or
//!   Tab) into an unknown window can submit forms, send messages, or
//!   change focus, which is exactly the class of accident this slice
//!   exists to prevent.
//! - Typing never rides on held modifiers: if the user still physically
//!   holds Ctrl/Alt/Shift/Super (typically from the dictation shortcut
//!   itself), the backend waits up to [`MODIFIER_RELEASE_WAIT`] for
//!   release and then refuses with [`InsertError::ModifiersHeld`] —
//!   characters typed with modifiers held become commands, and no
//!   modifier release is ever synthesized to fake readiness.
//! - A target owned by the inserting process itself is refused with
//!   [`InsertError::TargetIsStarling`]: Starling never types into
//!   Starling (a dictation take landing in its own editor would both
//!   confuse the user and could re-trigger the shortcut). The check
//!   consults a process-wide *exclusion set* (see
//!   [`Inserter::with_excluded_pids`]) against the **live** target's
//!   pid, not the frozen snapshot field.
//! - Evidence is honest: a typing backend can only ever return
//!   [`EVIDENCE_SYNTHETIC_KEYS`] — synthetic key *acceptance* is not
//!   proof the text landed (the target may swallow, drop or transform
//!   keys). Only a real IME commit path (phase C) may claim
//!   [`EVIDENCE_IME_COMMIT`].
//!
//! # Target identity is a string, not a pointer
//!
//! A [`TargetSnapshot`] freezes *what is focused now*: on X11 the
//! active + focus window ids, on Windows the foreground + focus
//! hwnds, plus the owning pid. The frozen identity travels as
//! `target_ref` — a self-describing, process-independent string
//! (`x11:<active-hex>:<focus-hex>:<pid>`, `win:<hwnd-hex>:<focus-hex>:<pid>`,
//! `fake:<id-hex>:<focus-hex>:<pid>` for tests) so the runtime host
//! process (mode B: the app captures, the host delivers) can revalidate
//! a ref the app captured. Refs are safeToken-compatible wherever the
//! contract needs one (`delivery.prepare{targetRef}`,
//! `delivery.conflict{expectedTarget, actualTarget}`).
//!
//! # What exists in phase A
//!
//! - X11 (WSLg/XWayland and plain X): [`x11::X11Backend`] — XTest
//!   typing through the keyboard mapping, with a spare-keycode remap
//!   for characters no key produces.
//! - Windows: `windows::WindowsBackend` — `SendInput` with
//!   `KEYEVENTF_UNICODE` per UTF-16 unit.
//! - Fake: `testing::FakeBackend` (feature `test-doubles`) —
//!   scripted focus and identity for the tests, including this
//!   crate's own.
//!
//! Wayland portal and IBus backends are phase C (slice 2b); their
//! [`BackendKind`]s exist now so refs and capability tables are stable
//! when they land.

use std::fmt;

#[cfg(feature = "runtime")]
pub mod runtime;
#[cfg(any(test, feature = "test-doubles"))]
pub mod testing;
#[cfg(windows)]
pub mod windows;
#[cfg(target_os = "linux")]
pub mod x11;

/// How long an insert waits for physically held modifier keys
/// (Ctrl/Alt/Shift/Super — the dictation shortcut's own keys) to be
/// released, polling about every [`MODIFIER_POLL_INTERVAL`], before
/// refusing with [`InsertError::ModifiersHeld`]. Bounded so a stuck key
/// delays one insert by a moment instead of hanging the delivery actor;
/// the copy fallback remains the escape hatch when the user really is
/// holding a key down. Modifiers are never released synthetically —
/// that would be Starling editing the user's physical keyboard state.
pub const MODIFIER_RELEASE_WAIT: std::time::Duration = std::time::Duration::from_millis(1500);
/// The polling granularity of the held-modifier wait.
pub const MODIFIER_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

/// Which backend produced a [`TargetSnapshot`] / serves an insert.
/// Every variant exists now so `target_ref` schemes and capability
/// tables are stable before the phase C backends land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// X11 / XWayland: identity is the active + focus window; typing is
    /// XTest synthetic keys (phase A).
    X11,
    /// Wayland RemoteDesktop portal (phase C, slice 2b).
    WaylandPortal,
    /// IBus engine (phase C, slice 2b) — the only backend allowed to
    /// return `ime_commit` evidence, and the one that can report
    /// surrounding text on free-desktops.
    IBus,
    /// Windows: identity is the foreground + focus hwnd; typing is
    /// `SendInput` `KEYEVENTF_UNICODE` (phase A).
    Windows,
    /// The scripted test double (feature `test-doubles`).
    Fake,
}

impl BackendKind {
    /// The `target_ref` scheme prefix for this backend kind. Parsing is
    /// the inverse (`parse_ref`); unknown schemes simply fail to
    /// parse, which is how a host rejects a ref from a newer Starling.
    pub fn scheme(self) -> &'static str {
        match self {
            BackendKind::X11 => "x11",
            // Phase C schemes are reserved now so phase B/C refs never
            // need to change shape.
            BackendKind::WaylandPortal => "rdp",
            BackendKind::IBus => "ibus",
            BackendKind::Windows => "win",
            BackendKind::Fake => "fake",
        }
    }

    fn from_scheme(scheme: &str) -> Option<Self> {
        match scheme {
            "x11" => Some(BackendKind::X11),
            "rdp" => Some(BackendKind::WaylandPortal),
            "ibus" => Some(BackendKind::IBus),
            "win" => Some(BackendKind::Windows),
            "fake" => Some(BackendKind::Fake),
            _ => None,
        }
    }

    /// The capability table phase A can honestly state for this kind.
    /// A parsed ref carries no live probe, so these are the *stable*
    /// per-backend facts, not a health check.
    fn capabilities(self) -> Capabilities {
        match self {
            BackendKind::X11 | BackendKind::Windows => Capabilities {
                insert: true,
                revalidate_identity: true,
                // X11 core has no portable surrounding-text protocol and
                // Windows needs UIA (later); absence is `Ok(None)`.
                surrounding_text: false,
                confirms_commit: false,
            },
            BackendKind::WaylandPortal => Capabilities {
                insert: true,
                revalidate_identity: true,
                surrounding_text: false,
                confirms_commit: false,
            },
            BackendKind::IBus => Capabilities {
                insert: true,
                revalidate_identity: true,
                // #341 depends on IBus reporting before/after cursor.
                surrounding_text: true,
                confirms_commit: false,
            },
            BackendKind::Fake => Capabilities {
                insert: true,
                revalidate_identity: true,
                surrounding_text: true,
                confirms_commit: false,
            },
        }
    }
}

/// What a backend can honestly do. Phase A typing backends report
/// `insert` + `revalidate_identity` only; the values feed the Linux
/// setup check (#221 phase B) and the recovery panel's promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// The backend can type into the captured target at all.
    pub insert: bool,
    /// The backend can re-check the frozen identity immediately before
    /// typing (every phase A backend can; a backend that cannot must
    /// refuse to insert rather than type unchecked).
    pub revalidate_identity: bool,
    /// The backend can report text around the target's cursor.
    pub surrounding_text: bool,
    /// The backend observes the target committing the text itself (an
    /// IME commit). Only such a backend may return
    /// [`EVIDENCE_IME_COMMIT`]; phase A backends never do.
    pub confirms_commit: bool,
}

/// The focused target *as captured*: the identity the rest of the
/// system compares against, plus what could be learned about the app
/// for display. [`Self::target_ref`] is the whole identity; the other
/// fields are advisory (an app may set none of them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetSnapshot {
    /// Which backend captured (and must serve) this target.
    pub backend: BackendKind,
    /// The self-describing identity string — the only field later
    /// revalidation relies on.
    pub target_ref: String,
    /// A human-facing app name if the platform offered one (`WM_CLASS`
    /// on X11, the exe name on Windows).
    pub app: Option<String>,
    /// The window title if the platform offered one.
    pub title: Option<String>,
    /// The owning process id, when the platform offered one. Also the
    /// Starling-owns-it guard: a pid equal to the inserting process's
    /// own means [`InsertError::TargetIsStarling`].
    pub pid: Option<u32>,
    /// What the backend can do with this target.
    pub capabilities: Capabilities,
}

impl TargetSnapshot {
    /// The `(active, focus)` window ids (and owning pid) carried by
    /// this snapshot's ref — the numeric identity underneath the
    /// string. Public for tests and diagnostics that must correlate a
    /// ref with a live window (the integration test does exactly
    /// that); production paths compare refs, never ids.
    pub fn ids(&self) -> Option<(u64, u64, Option<u32>)> {
        parse_ref(&self.target_ref).map(|(_kind, active, focus, pid)| (active, focus, pid))
    }
}

/// Text around the target's cursor, in **char offsets** for the
/// selection (byte offsets would make a JS/host layer re-count to do
/// anything useful with them). Phase C's IBus backend is the first to
/// actually produce one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurroundingText {
    pub before: String,
    pub after: String,
    /// The selected range within `before + after`, if the target
    /// reports one; `None` when there is no selection (a plain caret).
    pub selection: Option<std::ops::Range<usize>>,
}

/// What a revalidation of a frozen snapshot found. The distinction
/// matters to the user: `Changed` can name what took the focus,
/// `Gone` means the app closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetCheck {
    /// Still the same window with the same focus: safe to type.
    Same,
    /// Focus moved (or the active window changed): `{expected, actual}`
    /// refs. Never type; the caller decides whether to re-capture.
    Changed { expected: String, actual: String },
    /// The captured window no longer exists (the app closed or
    /// crashed).
    Gone,
}

/// Whether a backend can run in this session at all. This is the
/// backend *talking about itself*, not about any target: it feeds the
/// Linux setup check (#221 phase B) and the `Unavailable` errors from
/// [`Inserter::capture`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// The backend is usable (a target may still be unfocusable —
    /// that surfaces from `capture`, not here).
    Ready,
    /// The backend cannot run; `setup_hint` is the actionable part for
    /// a settings/setup surface (`reason` may name internals).
    Unavailable {
        reason: String,
        setup_hint: Option<String>,
    },
}

/// Strong evidence: the target's own IME committed the text. Only an
/// IME-commit backend (phase C) may claim this.
pub const EVIDENCE_IME_COMMIT: &str = "ime_commit";
/// Weak evidence: synthetic keys were *sent* (and only sent). XTest /
/// `SendInput` acceptance is not proof the text landed — the target
/// might swallow or transform keys — so `delivery.confirmed` carries
/// this level and the UI words it honestly.
pub const EVIDENCE_SYNTHETIC_KEYS: &str = "synthetic_keys_sent";

/// Proof-of-delivery label an insert returns. See the constants for
/// what each level honestly means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertReceipt {
    pub evidence: &'static str,
}

/// Every failure mode of capture/revalidate/insert. `code()` is the
/// machine-facing safeToken (it becomes `delivery.failed{reason}` and
/// must satisfy the contract's safeToken pattern
/// `^[A-Za-z0-9_.:+-]{1,128}$`); `message()` is the human sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// The backend itself cannot run here (no display, a Wayland
    /// session with only the phase C escape hatches, ...). Distinct
    /// from target problems: nothing about the *target* failed.
    Unavailable {
        reason: String,
        setup_hint: Option<String>,
    },
    /// The OS refused the input (UIPI on Windows: the target runs
    /// elevated above Starling). `settings_hint` is the actionable fix.
    PermissionDenied {
        reason: String,
        settings_hint: String,
    },
    /// Focus/identity moved since capture. Never a blind write.
    TargetChanged { expected: String, actual: String },
    /// The captured window no longer exists.
    TargetGone,
    /// The target belongs to this Starling process. Starling never
    /// types into itself.
    TargetIsStarling,
    /// Physically held modifier keys (Ctrl/Alt/Shift/Super, typically
    /// left over from the dictation shortcut) that did not clear within
    /// [`MODIFIER_RELEASE_WAIT`]. Typing with them held would turn
    /// characters into commands, so nothing is typed; `held` names the
    /// keys for the recovery panel. The check runs before every chunk,
    /// so as a *cause* of [`InsertError::PartialDelivery`] it means the
    /// next chunk never started, not that typing rode the keys.
    ModifiersHeld { held: Vec<String> },
    /// Typing stopped part-way through the text: `delivered_chars`
    /// characters were fully typed before `cause` stopped the rest —
    /// the target may hold that much, and only that much. `cause` is
    /// boxed so the variant stays flat in size. Zero characters
    /// delivered never produces this variant: with nothing typed, the
    /// cause is returned bare (there is nothing "partial" to report).
    PartialDelivery {
        delivered_chars: usize,
        total_chars: usize,
        cause: Box<InsertError>,
    },
    /// The keyboard's live state (active group, Caps/Shift Lock, latched
    /// modifiers — read through XKB on X11) is something synthetic
    /// typing cannot reproduce faithfully, so the backend refuses
    /// rather than guess and type the wrong characters. The IME path
    /// (phase C) is the real fix for such layouts.
    KeyboardStateUnsupported { reason: String },
    /// A keyboard mapping temporarily borrowed for characters no key
    /// produces could not be given back. The text was typed, but the
    /// user's keyboard may produce wrong characters until the layout is
    /// reloaded — a visible, actionable failure, not a silent success.
    KeyboardRestoreFailed { detail: String },
    /// The text contains a control character (`\n`, `\r`, `\t`, ...);
    /// v1 inserts single-line text only, and the copy fallback is the
    /// escape hatch.
    MultilineUnsupported,
    /// The target side refused or had nothing to type into (no focused
    /// window, no spare keycode to remap, an empty payload, ...). The
    /// catch-all for target-level refusals that are not one of the
    /// named situations above.
    Rejected { reason: String },
}

impl InsertError {
    /// The safeToken code for `delivery.failed{reason}`. Keep in sync
    /// with the tests asserting the safeToken pattern.
    pub fn code(&self) -> &'static str {
        match self {
            InsertError::Unavailable { .. } => "insertion_unavailable",
            InsertError::PermissionDenied { .. } => "insertion_permission_denied",
            InsertError::TargetChanged { .. } => "target_changed",
            InsertError::TargetGone => "target_gone",
            InsertError::TargetIsStarling => "target_is_starling",
            InsertError::ModifiersHeld { .. } => "modifiers_held",
            InsertError::PartialDelivery { .. } => "partial_delivery",
            InsertError::KeyboardStateUnsupported { .. } => "keyboard_state_unsupported",
            InsertError::KeyboardRestoreFailed { .. } => "keyboard_restore_failed",
            InsertError::MultilineUnsupported => "multiline_unsupported",
            InsertError::Rejected { .. } => "insertion_rejected",
        }
    }

    /// Whether the recovery panel should offer the copy fallback.
    /// Every failure means *the transcript did not land in the target*,
    /// so the retained text is the rescue in all cases — this stays a
    /// method (not a constant) because a future retryable variant
    /// (say, a momentarily busy target) should not send the user
    /// through the fallback flow.
    pub fn fallback_suggested(&self) -> bool {
        true
    }

    /// The human sentence, including the platform's detail. Stable
    /// wording is *not* promised (it may improve); machines use
    /// [`Self::code`].
    pub fn message(&self) -> String {
        match self {
            InsertError::Unavailable { reason, setup_hint } => match setup_hint {
                Some(hint) => format!("insertion is unavailable here: {reason} ({hint})"),
                None => format!("insertion is unavailable here: {reason}"),
            },
            InsertError::PermissionDenied {
                reason,
                settings_hint,
            } => format!("the system refused the input: {reason} ({settings_hint})"),
            InsertError::TargetChanged { expected, actual } => format!(
                "the focused target changed since the take started (was {expected}, is now \
                 {actual})",
            ),
            InsertError::TargetGone => {
                "the target window closed while the text was being delivered".to_string()
            }
            InsertError::TargetIsStarling => {
                "the focused window belongs to Starling itself; Starling never types into \
                 itself"
                    .to_string()
            }
            InsertError::ModifiersHeld { held } => format!(
                "modifier keys are still held down ({}); release them so the typed text is \
                 not turned into commands",
                held.join(", ")
            ),
            InsertError::PartialDelivery {
                delivered_chars,
                total_chars,
                cause,
            } => format!(
                "typing stopped part-way: up to {delivered_chars} of {total_chars} \
                 characters may have been typed before the failure ({cause})"
            ),
            InsertError::KeyboardStateUnsupported { reason } => format!(
                "the keyboard is in a state synthetic typing cannot reproduce safely \
                 ({reason}); use the copy fallback"
            ),
            InsertError::KeyboardRestoreFailed { detail } => format!(
                "the keyboard mapping borrowed for unmapped characters could not be \
                 restored ({detail}); the keyboard may type wrong characters until the \
                 layout is reloaded"
            ),
            InsertError::MultilineUnsupported => {
                "the text contains a line break or other control character; only single-line \
                 text can be typed"
                    .to_string()
            }
            InsertError::Rejected { reason } => {
                format!("the target refused the insertion: {reason}")
            }
        }
    }
}

impl fmt::Display for InsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for InsertError {}

/// One platform's capture/revalidate/type implementation. The trait is
/// deliberately synchronous: each method is a short X/Win32 round
/// trip, and the runtime host calls them from its delivery actor where
/// blocking briefly is already the model (the actor has no async
/// runtime). Backends must follow the crate rules documented on the
/// module: `insert` revalidates first itself and never emits a control
/// character.
pub trait InsertionBackend: Send + Sync {
    /// Which kind this is (also the `target_ref` scheme it serves).
    fn kind(&self) -> BackendKind;
    /// Whether this backend can run in the current session. Cheap
    /// enough to call per capture; never touches a target.
    fn availability(&self) -> Availability;
    /// Capture the focused target *now*. Fails when nothing useful is
    /// focused or the platform refuses — never guesses a target.
    fn capture(&self) -> Result<TargetSnapshot, InsertError>;
    /// Compare the frozen snapshot against the live focus state.
    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError>;
    /// Text around the target's cursor; `Ok(None)` when the backend
    /// has no such capability (both phase A backends).
    fn surrounding_text(
        &self,
        target: &TargetSnapshot,
    ) -> Result<Option<SurroundingText>, InsertError>;
    /// Type `text` into `target`. Must itself revalidate immediately
    /// before the first key, refuse control characters, and refuse a
    /// target owned by this process.
    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError>;
}

/// The ordered backend set for a session, and the front door callers
/// actually use. Ordering is the platform policy: the first available
/// backend captures, and a captured ref is served by *its* backend
/// (never re-interpreted by another), so a ref captured via X11 stays
/// X11 even if another backend becomes available later.
pub struct Inserter {
    backends: Vec<Box<dyn InsertionBackend>>,
}

impl Inserter {
    /// The production backend order for this platform. Phase A has one
    /// backend per platform; phase C inserts the Wayland portal and
    /// IBus *before* X11 on Linux (a portal that works is strictly
    /// more honest than X11 guessing under Wayland). Self-target
    /// protection covers only this process's own pid; the mode-B host
    /// (or any embedding that splits capture from delivery) wants
    /// [`Self::with_excluded_pids`].
    pub fn for_this_session() -> Inserter {
        Self::with_excluded_pids(Vec::new())
    }

    /// The session backends with an *ownership policy*: every pid in
    /// `excluded` — plus always `std::process::id()` — marks a target
    /// Starling refuses to type into ([`InsertError::TargetIsStarling`]),
    /// checked against the **live** target's pid at insert time, not
    /// the frozen snapshot field (the snapshot's pid is advisory and
    /// may be stale or absent).
    ///
    /// This is the mode-B constructor: the runtime host (#220) runs in
    /// its own process and passes the desktop app's pid, so a ref the
    /// app captured cannot be typed back into the app by the host.
    /// On platforms with no phase A backend the list is still stored by
    /// the backends that will exist; nothing can be inserted either
    /// way. Note the exclusion is a *refusal* policy, not identity: it
    /// widens the self-target guard, never narrows it.
    pub fn with_excluded_pids(excluded: Vec<u32>) -> Inserter {
        let excluded = merge_excluded_pids(excluded);
        #[cfg(target_os = "linux")]
        let backends: Vec<Box<dyn InsertionBackend>> =
            vec![Box::new(x11::X11Backend::with_excluded_pids(excluded))];
        #[cfg(windows)]
        let backends: Vec<Box<dyn InsertionBackend>> = vec![Box::new(
            windows::WindowsBackend::with_excluded_pids(excluded),
        )];
        #[cfg(not(any(target_os = "linux", windows)))]
        let backends: Vec<Box<dyn InsertionBackend>> = Vec::new();
        Inserter::with_backends(backends)
    }

    /// An explicit backend list (tests, probes, future platform
    /// policies). The first available backend wins at
    /// [`Self::capture`].
    pub fn with_backends(backends: Vec<Box<dyn InsertionBackend>>) -> Inserter {
        Inserter { backends }
    }

    /// Capture the focused target through the first available backend.
    /// When every backend is unavailable the error carries the
    /// *first*'s reason (the ordered policy's best guess at what the
    /// user should fix); with no backends at all it says so plainly.
    pub fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        let mut first_blocker: Option<InsertError> = None;
        for backend in &self.backends {
            match backend.availability() {
                Availability::Ready => return backend.capture(),
                Availability::Unavailable { reason, setup_hint } => {
                    if first_blocker.is_none() {
                        first_blocker = Some(InsertError::Unavailable { reason, setup_hint });
                    }
                }
            }
        }
        Err(first_blocker.unwrap_or(InsertError::Unavailable {
            reason: "no insertion backend exists for this platform".to_string(),
            setup_hint: None,
        }))
    }

    /// The backend that owns this target's scheme, if present in this
    /// inserter. A ref captured by an app is only insertable where the
    /// matching backend runs.
    pub fn backend_for(&self, target: &TargetSnapshot) -> Option<&dyn InsertionBackend> {
        self.backends
            .iter()
            .map(|b| b.as_ref())
            .find(|b| b.kind() == target.backend)
    }

    /// Parse a `target_ref` back into a snapshot. Refs are
    /// self-describing (scheme + ids + optional pid), so this needs no
    /// backend and no server round trip — the runtime host parses refs
    /// the app captured. Advisory fields (`app`/`title`) are not in the
    /// ref and come back `None`; identity is complete. Returns `None`
    /// for anything malformed or of an unknown scheme (a host rejects
    /// such a ref at `delivery.prepare`).
    pub fn parse_target_ref(&self, target_ref: &str) -> Option<TargetSnapshot> {
        let (backend, _active, _focus, pid) = parse_ref(target_ref)?;
        Some(TargetSnapshot {
            backend,
            target_ref: target_ref.to_string(),
            app: None,
            title: None,
            pid,
            capabilities: backend.capabilities(),
        })
    }

    /// A one-line honest description for adapter `describe()`
    /// snapshots: which backends this inserter holds, in order.
    pub fn describe(&self) -> String {
        if self.backends.is_empty() {
            "no insertion backend on this platform".to_string()
        } else {
            format!(
                "insertion backends: {}",
                self.backends
                    .iter()
                    .map(|backend| backend.kind().scheme())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// Format a target ref. Window ids are lowercase hex (no `0x`), the
/// pid is decimal and omitted when the platform could not learn one —
/// both forms parse, and both are safeToken-clean.
pub(crate) fn format_ref(kind: BackendKind, active: u64, focus: u64, pid: Option<u32>) -> String {
    match pid {
        Some(pid) => format!("{}:{:x}:{:x}:{}", kind.scheme(), active, focus, pid),
        None => format!("{}:{:x}:{:x}", kind.scheme(), active, focus),
    }
}

/// Parse a target ref into `(kind, active, focus, pid)`. Strict on
/// shape — hex ids must be valid u32s, the pid a decimal u32 — because
/// a malformed ref must fail `delivery.prepare`, not linger as a
/// string that compares unequal to itself. Hex is case-insensitive on
/// the way in (identity is the numeric id, not the spelling).
pub(crate) fn parse_ref(target_ref: &str) -> Option<(BackendKind, u64, u64, Option<u32>)> {
    // safeToken's own bound; refs ride that field on the wire.
    if target_ref.len() > 128 {
        return None;
    }
    let (scheme, rest) = target_ref.split_once(':')?;
    let backend = BackendKind::from_scheme(scheme)?;
    let mut parts = rest.split(':');
    let active = u32::from_str_radix(parts.next()?, 16).ok()? as u64;
    let focus = u32::from_str_radix(parts.next()?, 16).ok()? as u64;
    let pid = match parts.next() {
        Some(pid) => Some(pid.parse::<u32>().ok()?),
        None => None,
    };
    if parts.next().is_some() {
        return None;
    }
    if active == 0 {
        // A zero "active" is never a real window; a capture would have
        // refused to build such a ref in the first place.
        return None;
    }
    Some((backend, active, focus, pid))
}

/// Share one backend instance between an [`Inserter`] and whoever
/// else needs it (a host constructing the `runtime` adapter while
/// keeping a scripting handle for its tests) — delegation keeps the
/// boxed type a single backend, not a wrapper with its own identity.
impl<T: InsertionBackend + ?Sized> InsertionBackend for std::sync::Arc<T> {
    fn kind(&self) -> BackendKind {
        (**self).kind()
    }
    fn availability(&self) -> Availability {
        (**self).availability()
    }
    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        (**self).capture()
    }
    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        (**self).revalidate(target)
    }
    fn surrounding_text(
        &self,
        target: &TargetSnapshot,
    ) -> Result<Option<SurroundingText>, InsertError> {
        (**self).surrounding_text(target)
    }
    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        (**self).insert(target, text)
    }
}

/// A 64-bit FNV-1a digest. The compare token is an integrity check of
/// *which ref a delivery was prepared against* — not a security
/// boundary against an attacker who can forge refs (they can forge
/// digests too) — so a small dependency-free digest beats pulling a
/// crypto crate in for theater. Only the `runtime` bridge mints
/// tokens, so the digest exists only there.
#[cfg(feature = "runtime")]
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// The pids Starling refuses to type into, from any caller-supplied
/// list plus always this process's own: sorted and deduped so equality
/// and containment behave. Public only as the shared helper behind
/// [`Inserter::with_excluded_pids`] and the backend constructors that
/// mirror it.
pub(crate) fn merge_excluded_pids(excluded: Vec<u32>) -> Vec<u32> {
    let mut pids = excluded;
    pids.push(std::process::id());
    pids.sort_unstable();
    pids.dedup();
    pids
}

/// The cheap, target-independent pre-insert guards: nothing typed,
/// nothing learned about the live target. Real backends run these
/// first, then do their platform preparation, then their *live*
/// revalidation last (see [`deliver_in_chunks`]); the fake has no
/// preparation phase and uses the combined [`insertion_guards`].
pub(crate) fn cheap_insertion_guards(
    text: &str,
    pid: Option<u32>,
    excluded_pids: &[u32],
) -> Result<(), InsertError> {
    if text.is_empty() {
        // An empty insert types nothing and "succeeded" would claim
        // evidence it cannot have; phase B must not offer an empty
        // take for delivery at all.
        return Err(InsertError::Rejected {
            reason: "empty text: nothing to insert".to_string(),
        });
    }
    if text.chars().any(char::is_control) {
        // No Enter, no Tab, no C0/C1 surprise — checked before any
        // server round trip so refusal is guaranteed even if the
        // platform layer were forgetful.
        return Err(InsertError::MultilineUnsupported);
    }
    if pid.is_some_and(|pid| excluded_pids.contains(&pid)) {
        // The frozen snapshot's own pid — an early refusal for a ref
        // that was Starling-owned at capture time. The authoritative
        // check is against the *live* pid in each backend's
        // revalidation; this one only catches what the ref itself
        // already admits.
        return Err(InsertError::TargetIsStarling);
    }
    Ok(())
}

/// The shared pre-insert guards, in the order every backend must apply
/// them. Centralized so a new backend cannot get the order wrong:
/// cheap content checks first (nothing typed, nothing learned), the
/// self-ownership refusal next, then the backend's own live
/// revalidation — as a closure because each backend revalidates over
/// *its* connection, immediately before typing, so the check and the
/// keys ride the same server view. Only the test-double fake uses
/// this combined form (the platform backends run the cheap guards,
/// their preparation, then their own chunked checks); hence the cfg.
#[cfg(any(test, feature = "test-doubles"))]
pub(crate) fn insertion_guards(
    text: &str,
    pid: Option<u32>,
    excluded_pids: &[u32],
    live_revalidate: impl FnOnce() -> Result<TargetCheck, InsertError>,
) -> Result<(), InsertError> {
    cheap_insertion_guards(text, pid, excluded_pids)?;
    match live_revalidate()? {
        TargetCheck::Same => Ok(()),
        check @ (TargetCheck::Changed { .. } | TargetCheck::Gone) => Err(check_to_error(check)),
    }
}

/// Map a completed [`TargetCheck`] to the corresponding insert refusal.
/// `TargetCheck::Changed` already carries the snapshot's own
/// expected/actual pair, so nothing is reformatted here; `Same` is not
/// a refusal at all and maps to a clearly-internal error (a caller
/// that lands there has a bug, and an honest error beats a silent
/// `Ok`).
#[cfg(any(test, feature = "test-doubles"))]
pub(crate) fn check_to_error(check: TargetCheck) -> InsertError {
    match check {
        TargetCheck::Same => InsertError::Rejected {
            reason: "internal error: target check passed but insert still refused".to_string(),
        },
        TargetCheck::Changed { expected, actual } => {
            InsertError::TargetChanged { expected, actual }
        }
        TargetCheck::Gone => InsertError::TargetGone,
    }
}

/// Split `text` into segments of at most `max_weight` "weight" per
/// segment (`weight` is 1 per char for X11's char cap, the UTF-16 unit
/// count for Windows' SendInput cap), never cutting a character in
/// half — surrogate pairs always travel inside one segment. Text is
/// non-empty by the time this runs (the guards refused the empty
/// case), so the result is never empty.
pub(crate) fn weighed_segments<'a, F: Fn(char) -> usize>(
    text: &'a str,
    max_weight: usize,
    weight: F,
) -> Vec<&'a str> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut weight_sum = 0usize;
    for (index, character) in text.char_indices() {
        let char_weight = weight(character);
        if index > start && weight_sum + char_weight > max_weight {
            segments.push(&text[start..index]);
            start = index;
            weight_sum = 0;
        }
        weight_sum += char_weight;
    }
    segments.push(&text[start..]);
    segments
}

/// How many leading characters of `text` are *completely* contained in
/// the first `units` UTF-16 units — a character whose surrogate pair
/// was cut in half was not delivered. Windows reports SendInput's
/// progress in events (down+up pairs per unit), and half a pair is
/// worse than nothing, so only whole characters count.
#[cfg(any(windows, test))]
pub(crate) fn chars_complete_within_units(text: &str, units: usize) -> usize {
    let mut remaining = units;
    let mut characters = 0usize;
    for character in text.chars() {
        let length = character.len_utf16();
        if length > remaining {
            break;
        }
        remaining -= length;
        characters += 1;
    }
    characters
}

/// What a segment typer reports when typing failed part-way through
/// the segment: how many characters were delivered *in total* (the
/// chunks before this one plus whatever of this one landed) and the
/// underlying cause.
pub(crate) struct ChunkFailure {
    pub delivered: usize,
    pub cause: InsertError,
}

/// The chunked typing loop every platform backend drives: `check` runs
/// before *every* chunk (held modifiers waited out, focus/identity
/// revalidated — the same final check as before the first chunk), then
/// `type_segment` types one chunk and reports exactly how many
/// characters it delivered. Any failure stops typing immediately:
/// with nothing delivered the cause is returned bare, otherwise it is
/// wrapped as [`InsertError::PartialDelivery`] so the user learns how
/// much may have landed. Success is only the weak synthetic-keys
/// evidence — chunking does not make delivery more provable.
pub(crate) fn deliver_in_chunks(
    total_chars: usize,
    segments: &[&str],
    mut check: impl FnMut() -> Result<(), InsertError>,
    mut type_segment: impl FnMut(&str, usize) -> Result<usize, ChunkFailure>,
) -> Result<InsertReceipt, InsertError> {
    let mut delivered = 0usize;
    for segment in segments {
        if let Err(cause) = check() {
            return Err(partialize(delivered, total_chars, cause));
        }
        match type_segment(segment, delivered) {
            Ok(typed) => {
                debug_assert_eq!(
                    typed,
                    segment.chars().count(),
                    "a segment typer either delivers the whole segment or fails"
                );
                delivered += typed;
            }
            Err(failure) => return Err(partialize(failure.delivered, total_chars, failure.cause)),
        }
    }
    Ok(InsertReceipt {
        evidence: EVIDENCE_SYNTHETIC_KEYS,
    })
}

/// Wrap a mid-typing failure: nothing delivered means the cause tells
/// the whole story on its own; anything delivered turns it into a
/// partial-delivery report that says how much may have landed.
pub(crate) fn partialize(delivered: usize, total_chars: usize, cause: InsertError) -> InsertError {
    if delivered == 0 {
        cause
    } else {
        InsertError::PartialDelivery {
            delivered_chars: delivered,
            total_chars,
            cause: Box::new(cause),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract's safeToken pattern, applied to every error code
    /// and to the ref round-trips: these strings ride
    /// `delivery.failed{reason}` and `delivery.prepare{targetRef}`.
    fn is_safe_token(token: &str) -> bool {
        !token.is_empty()
            && token.len() <= 128
            && token
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '+' | '-'))
    }

    #[test]
    fn error_codes_are_safe_tokens() {
        let samples = [
            InsertError::Unavailable {
                reason: "no display".into(),
                setup_hint: None,
            },
            InsertError::PermissionDenied {
                reason: "UIPI blocked SendInput".into(),
                settings_hint: "run at the same integrity level".into(),
            },
            InsertError::TargetChanged {
                expected: "x11:1:1".into(),
                actual: "x11:2:2".into(),
            },
            InsertError::TargetGone,
            InsertError::TargetIsStarling,
            InsertError::ModifiersHeld {
                held: vec!["Control".into(), "Shift".into()],
            },
            InsertError::PartialDelivery {
                delivered_chars: 16,
                total_chars: 64,
                cause: Box::new(InsertError::TargetChanged {
                    expected: "x11:1:1".into(),
                    actual: "x11:2:2".into(),
                }),
            },
            InsertError::KeyboardStateUnsupported {
                reason: "Caps Lock is engaged".into(),
            },
            InsertError::KeyboardRestoreFailed {
                detail: "the X server refused the restore".into(),
            },
            InsertError::MultilineUnsupported,
            InsertError::Rejected {
                reason: "no focused window".into(),
            },
        ];
        for error in &samples {
            assert!(
                is_safe_token(error.code()),
                "{} is a safeToken",
                error.code()
            );
            assert!(error.fallback_suggested());
            assert!(!error.message().is_empty());
        }
        // And every distinct variant has a distinct code, so events
        // stay distinguishable without string matching.
        let mut codes: Vec<_> = samples.iter().map(|e| e.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), samples.len());
    }

    #[test]
    fn partial_delivery_message_reports_how_much_may_have_landed() {
        let error = InsertError::PartialDelivery {
            delivered_chars: 16,
            total_chars: 64,
            cause: Box::new(InsertError::TargetChanged {
                expected: "x11:1:1".into(),
                actual: "x11:2:2".into(),
            }),
        };
        let message = error.message();
        assert!(
            message.contains("16"),
            "names the delivered count: {message}"
        );
        assert!(message.contains("64"), "names the total: {message}");
        assert!(
            message.contains("may have been typed"),
            "hedges honestly: {message}"
        );
        assert_eq!(error.code(), "partial_delivery");
        assert!(error.fallback_suggested());
        // The cause is inspectable for recovery logic ("keep the tail,
        // not the head").
        let cause = match error {
            InsertError::PartialDelivery { cause, .. } => *cause,
            other => panic!("expected a partial delivery, got {other:?}"),
        };
        assert!(matches!(cause, InsertError::TargetChanged { .. }));
    }

    #[test]
    fn held_modifier_and_keyboard_state_errors_carry_their_detail() {
        let held = InsertError::ModifiersHeld {
            held: vec!["Control".into(), "Mod4".into()],
        };
        let message = held.message();
        assert!(
            message.contains("Control"),
            "names the held keys: {message}"
        );
        assert!(message.contains("Mod4"), "names the held keys: {message}");
        assert!(
            message.contains("still held"),
            "says what happened: {message}"
        );
        assert_eq!(held.code(), "modifiers_held");
        assert!(held.fallback_suggested());

        let state = InsertError::KeyboardStateUnsupported {
            reason: "group 1 is active".into(),
        };
        assert!(state.message().contains("group 1"));
        assert_eq!(state.code(), "keyboard_state_unsupported");
        assert!(state.fallback_suggested());

        let restore = InsertError::KeyboardRestoreFailed {
            detail: "connection lost".into(),
        };
        assert!(restore.message().contains("connection lost"));
        assert_eq!(restore.code(), "keyboard_restore_failed");
        assert!(restore.fallback_suggested());
    }

    #[test]
    fn target_refs_round_trip_through_format_and_parse() {
        for (kind, active, focus, pid) in [
            (BackendKind::X11, 0x600001, 0x600002, Some(4213)),
            (BackendKind::Windows, 0x00060418, 0x00090c2e, Some(8)),
            (BackendKind::Fake, 1, 1, None),
            (BackendKind::X11, u32::MAX as u64, 1, None),
        ] {
            let reference = format_ref(kind, active, focus, pid);
            assert!(is_safe_token(&reference), "{reference} must be a safeToken");
            let parsed = parse_ref(&reference).expect("round trip parses");
            assert_eq!(parsed, (kind, active, focus, pid), "{reference}");
        }
    }

    #[test]
    fn parse_ref_rejects_malformed_and_foreign_refs() {
        // Each of these must fail prepare loudly, not silently mean
        // something else.
        for bad in [
            "",                             // nothing
            "x11",                          // scheme only
            "x11:",                         // empty ids
            "x11:0:0",                      // zero window id never happens
            "x11:zz:1",                     // non-hex
            "x11:1:2:notanumber",           // pid not decimal
            "x11:1:2:3:4",                  // trailing component
            "x11:1:2:18446744073709551616", // pid overflow
            "macos:1:2",                    // scheme from a future/other port
            "fake:1:2:3:4:5",               // too many parts for the fake scheme too
            &"x:".repeat(100),              // over the safeToken bound
        ] {
            assert!(parse_ref(bad).is_none(), "{bad:?} must not parse");
        }
        // Hex case-insensitivity: the identity is the number.
        assert_eq!(parse_ref("x11:ABC:Def:7"), parse_ref("x11:abc:def:7"));
    }

    #[test]
    #[cfg(feature = "runtime")]
    fn fnv1a64_is_deterministic_and_input_sensitive() {
        assert_eq!(fnv1a64(b"x11:1:2:3"), fnv1a64(b"x11:1:2:3"));
        assert_ne!(fnv1a64(b"x11:1:2:3"), fnv1a64(b"x11:1:2:4"));
        assert_ne!(fnv1a64(b""), fnv1a64(b"x"));
    }

    #[test]
    fn inserter_reports_first_blocker_when_no_backend_is_available() {
        // Two unavailable backends: the ordered policy's first reason
        // is the one a user can act on.
        struct Blocked(&'static str);
        impl InsertionBackend for Blocked {
            fn kind(&self) -> BackendKind {
                BackendKind::Fake
            }
            fn availability(&self) -> Availability {
                Availability::Unavailable {
                    reason: self.0.to_string(),
                    setup_hint: None,
                }
            }
            fn capture(&self) -> Result<TargetSnapshot, InsertError> {
                Err(InsertError::Rejected {
                    reason: "never called".into(),
                })
            }
            fn revalidate(&self, _: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
                Ok(TargetCheck::Same)
            }
            fn surrounding_text(
                &self,
                _: &TargetSnapshot,
            ) -> Result<Option<SurroundingText>, InsertError> {
                Ok(None)
            }
            fn insert(&self, _: &TargetSnapshot, _: &str) -> Result<InsertReceipt, InsertError> {
                Err(InsertError::Rejected {
                    reason: "never called".into(),
                })
            }
        }
        let inserter = Inserter::with_backends(vec![
            Box::new(Blocked("first blocker")),
            Box::new(Blocked("second blocker")),
        ]);
        match inserter.capture() {
            Err(InsertError::Unavailable { reason, .. }) => {
                assert_eq!(reason, "first blocker");
            }
            other => panic!("expected the first blocker, got {other:?}"),
        }
        // And with no backends at all the error says exactly that.
        let empty = Inserter::with_backends(Vec::new());
        assert!(matches!(
            empty.capture(),
            Err(InsertError::Unavailable { reason, .. })
                if reason.contains("no insertion backend")
        ));
    }

    #[test]
    fn segments_split_by_weight_without_cutting_characters() {
        // X11: 16 characters per chunk, weight 1 per char.
        let text = "a".repeat(40);
        let segments = weighed_segments(&text, 16, |_| 1);
        assert_eq!(
            segments
                .iter()
                .map(|s| s.chars().count())
                .collect::<Vec<_>>(),
            [16, 16, 8],
            "40 chars pack into 16-char chunks"
        );
        assert_eq!(segments.concat(), text);

        // Windows: 32 UTF-16 units per chunk; a surrogate pair never
        // splits, and one pair more than the cap starts a new segment.
        let emoji: String = "😀".repeat(17); // 34 units
        let segments = weighed_segments(&emoji, 32, char::len_utf16);
        assert_eq!(
            segments
                .iter()
                .map(|s| s.chars().count())
                .collect::<Vec<_>>(),
            [16, 1],
            "16 pairs (32 units) then the 17th alone"
        );
        assert_eq!(segments.concat(), emoji);

        // Mixed BMP and supplementary characters stay in order.
        let mixed = "a😀b";
        assert_eq!(weighed_segments(mixed, 32, char::len_utf16), vec![mixed]);
        // A single character heavier than the cap still ships — the
        // alternative (dropping it) would silently eat text.
        assert_eq!(weighed_segments("😀", 1, char::len_utf16), vec!["😀"]);
    }

    #[test]
    fn complete_chars_never_count_a_severed_surrogate_half() {
        let text = "a😀b";
        assert_eq!(chars_complete_within_units(text, 0), 0);
        assert_eq!(chars_complete_within_units(text, 1), 1, "just 'a'");
        assert_eq!(
            chars_complete_within_units(text, 2),
            1,
            "the pair's first half does not count as a delivered character"
        );
        assert_eq!(chars_complete_within_units(text, 3), 2, "'a😀'");
        assert_eq!(chars_complete_within_units(text, 4), 3, "everything");
        assert_eq!(chars_complete_within_units(text, 99), 3);
    }

    /// The chunk loop, scripted like a backend: a queue of check and
    /// typing outcomes the test drives step by step.
    #[test]
    fn the_chunk_loop_types_every_segment_after_rechecking() {
        let text = "0123456789abcdefghij"; // 20 chars
        let segments = weighed_segments(text, 16, |_| 1);
        let mut checks = 0;
        let mut typed = Vec::new();
        let receipt = deliver_in_chunks(
            text.chars().count(),
            &segments,
            || {
                checks += 1;
                Ok(())
            },
            |segment, _| {
                typed.push(segment.to_string());
                Ok(segment.chars().count())
            },
        )
        .expect("a clean run types everything");
        assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
        assert_eq!(checks, 2, "one check per chunk, including the first");
        assert_eq!(typed.len(), 2);
        assert_eq!(typed.concat(), text);
    }

    #[test]
    fn the_chunk_loop_wraps_mid_way_failures_but_not_first_chunk_ones() {
        let text = "0123456789abcdefghij";
        let segments = weighed_segments(text, 16, |_| 1);
        let changed = InsertError::TargetChanged {
            expected: "x11:1:1".into(),
            actual: "x11:2:2".into(),
        };

        // The check before the *first* chunk fails: nothing was typed,
        // so the cause returns bare (there is nothing "partial").
        let error = deliver_in_chunks(
            text.chars().count(),
            &segments,
            || Err(changed.clone()),
            |segment, _| Ok(segment.chars().count()),
        )
        .unwrap_err();
        assert_eq!(error, changed);

        // The check before the *second* chunk fails after 16 chars
        // landed: a partial-delivery report naming the counts.
        let mut checks = 0;
        let error = deliver_in_chunks(
            text.chars().count(),
            &segments,
            || {
                checks += 1;
                if checks == 1 {
                    Ok(())
                } else {
                    Err(changed.clone())
                }
            },
            |segment, _| Ok(segment.chars().count()),
        )
        .unwrap_err();
        match error {
            InsertError::PartialDelivery {
                delivered_chars,
                total_chars,
                cause,
            } => {
                assert_eq!(delivered_chars, 16);
                assert_eq!(total_chars, 20);
                assert_eq!(*cause, changed);
            }
            other => panic!("a mid-typing change is partial delivery: {other:?}"),
        }

        // Typing itself fails mid-segment (say the 5th char of chunk
        // two): the report counts exactly what landed, chunk plus
        // partial chunk.
        let error = deliver_in_chunks(
            text.chars().count(),
            &segments,
            || Ok(()),
            |segment, delivered| {
                if delivered == 0 {
                    Ok(segment.chars().count())
                } else {
                    Err(ChunkFailure {
                        delivered: delivered + 5,
                        cause: InsertError::PermissionDenied {
                            reason: "blocked".into(),
                            settings_hint: "same integrity level".into(),
                        },
                    })
                }
            },
        )
        .unwrap_err();
        match error {
            InsertError::PartialDelivery {
                delivered_chars, ..
            } => {
                assert_eq!(delivered_chars, 21, "16 from chunk one + 5 of chunk two")
            }
            other => panic!("a mid-segment typing failure is partial delivery: {other:?}"),
        }

        // Typing fails with nothing landed in the failing chunk's run
        // at all (first chunk, zero chars): bare cause again.
        let error = deliver_in_chunks(
            text.chars().count(),
            &segments,
            || Ok(()),
            |_, _| {
                Err(ChunkFailure {
                    delivered: 0,
                    cause: changed.clone(),
                })
            },
        )
        .unwrap_err();
        assert_eq!(error, changed);
    }

    #[test]
    fn excluded_pids_always_include_this_process() {
        assert_eq!(
            merge_excluded_pids(Vec::new()),
            vec![std::process::id()],
            "the plain self-target guard survives"
        );
        let merged = merge_excluded_pids(vec![7, std::process::id(), 7, 900]);
        assert_eq!(merged, {
            let mut expected = vec![std::process::id(), 7, 900];
            expected.sort_unstable();
            expected
        });
        // The production constructor routes the same merge into its
        // platform backend (X11 here; the Windows twin compiles there
        // and is exercised by its interactive test).
        #[cfg(target_os = "linux")]
        {
            let backend = crate::x11::X11Backend::with_excluded_pids(vec![4213]);
            let excluded = backend.excluded_pids();
            assert!(excluded.contains(&4213));
            assert!(excluded.contains(&std::process::id()));
        }
    }
}
