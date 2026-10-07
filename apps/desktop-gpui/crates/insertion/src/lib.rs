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
//!   ran for minutes); the check happens last.
//! - Text containing any control character — `\n`, `\r`, `\t`, or any
//!   other Cc character — is refused with
//!   [`InsertError::MultilineUnsupported`] before a single key moves.
//!   Transcripts are single-line paragraphs in v1; sending Enter (or
//!   Tab) into an unknown window can submit forms, send messages, or
//!   change focus, which is exactly the class of accident this slice
//!   exists to prevent.
//! - A target owned by the inserting process itself is refused with
//!   [`InsertError::TargetIsStarling`]: Starling never types into
//!   Starling (a dictation take landing in its own editor would both
//!   confuse the user and could re-trigger the shortcut).
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

#[cfg(target_os = "linux")]
pub mod x11;
#[cfg(feature = "runtime")]
pub mod runtime;
#[cfg(any(test, feature = "test-doubles"))]
pub mod testing;
#[cfg(windows)]
pub mod windows;

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
    PermissionDenied { reason: String, settings_hint: String },
    /// Focus/identity moved since capture. Never a blind write.
    TargetChanged { expected: String, actual: String },
    /// The captured window no longer exists.
    TargetGone,
    /// The target belongs to this Starling process. Starling never
    /// types into itself.
    TargetIsStarling,
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
                 {actual}); nothing was typed",
            ),
            InsertError::TargetGone => {
                "the target window closed before the text could be delivered; nothing was \
                 typed"
                    .to_string()
            }
            InsertError::TargetIsStarling => {
                "the focused window belongs to Starling itself; Starling never types into \
                 itself"
                    .to_string()
            }
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
    fn surrounding_text(&self, target: &TargetSnapshot)
        -> Result<Option<SurroundingText>, InsertError>;
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
    /// more honest than X11 guessing under Wayland).
    pub fn for_this_session() -> Inserter {
        #[cfg(target_os = "linux")]
        let backends: Vec<Box<dyn InsertionBackend>> =
            vec![Box::new(x11::X11Backend::new())];
        #[cfg(windows)]
        let backends: Vec<Box<dyn InsertionBackend>> =
            vec![Box::new(windows::WindowsBackend::new())];
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
                        first_blocker = Some(InsertError::Unavailable {
                            reason,
                            setup_hint,
                        });
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
pub(crate) fn format_ref(
    kind: BackendKind,
    active: u64,
    focus: u64,
    pid: Option<u32>,
) -> String {
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

/// The shared pre-insert guards, in the order every backend must apply
/// them. Centralized so a new backend cannot get the order wrong:
/// cheap content checks first (nothing typed, nothing learned), the
/// self-ownership refusal next, then the backend's own live
/// revalidation — as a closure because each backend revalidates over
/// *its* connection, immediately before typing, so the check and the
/// keys ride the same server view.
pub(crate) fn insertion_guards(
    text: &str,
    pid: Option<u32>,
    live_revalidate: impl FnOnce() -> Result<TargetCheck, InsertError>,
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
    if pid == Some(std::process::id()) {
        return Err(InsertError::TargetIsStarling);
    }
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
pub(crate) fn check_to_error(check: TargetCheck) -> InsertError {
    match check {
        TargetCheck::Same => InsertError::Rejected {
            reason: "internal error: target check passed but insert still refused".to_string(),
        },
        TargetCheck::Changed { expected, actual } => InsertError::TargetChanged {
            expected,
            actual,
        },
        TargetCheck::Gone => InsertError::TargetGone,
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
            InsertError::MultilineUnsupported,
            InsertError::Rejected {
                reason: "no focused window".into(),
            },
        ];
        for error in &samples {
            assert!(is_safe_token(error.code()), "{} is a safeToken", error.code());
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
            "",                       // nothing
            "x11",                    // scheme only
            "x11:",                   // empty ids
            "x11:0:0",                // zero window id never happens
            "x11:zz:1",               // non-hex
            "x11:1:2:notanumber",     // pid not decimal
            "x11:1:2:3:4",            // trailing component
            "x11:1:2:18446744073709551616", // pid overflow
            "macos:1:2",              // scheme from a future/other port
            "fake:1:2:3:4:5",         // too many parts for the fake scheme too
            &"x:".repeat(100),        // over the safeToken bound
        ] {
            assert!(parse_ref(bad).is_none(), "{bad:?} must not parse");
        }
        // Hex case-insensitivity: the identity is the number.
        assert_eq!(
            parse_ref("x11:ABC:Def:7"),
            parse_ref("x11:abc:def:7")
        );
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
}
