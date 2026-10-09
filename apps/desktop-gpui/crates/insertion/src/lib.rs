//! `starling-insertion`: focus-safe text insertion into the focused
//! external window. Platform backends sit behind [`InsertionBackend`];
//! the [`Inserter`] picks one per session, and `runtime::InsertionDeliveryAdapter`
//! (feature `runtime`) plugs it into the runtime's `delivery.*` path.
//!
//! Rules every backend enforces at the key sender itself, so no caller
//! can bypass them:
//!
//! - The target is revalidated immediately before typing and again
//!   before every chunk of a long text. A change before the first chunk
//!   refuses ([`InsertError::TargetChanged`] / [`InsertError::TargetGone`]);
//!   a change part-way stops and reports [`InsertError::PartialDelivery`].
//! - Text with any control character is refused
//!   ([`InsertError::MultilineUnsupported`]): Enter or Tab into an unknown
//!   window can submit forms or move focus.
//! - Held modifiers (typically the dictation shortcut's own keys) are
//!   waited out for up to [`MODIFIER_RELEASE_WAIT`], then the insert
//!   refuses ([`InsertError::ModifiersHeld`]). A release is never
//!   synthesized.
//! - A target owned by an excluded pid (always including this process)
//!   is refused ([`InsertError::TargetIsStarling`]), checked against the
//!   live target, not the snapshot.
//! - Typing backends only claim [`EVIDENCE_SYNTHETIC_KEYS`]: accepted
//!   synthetic keys are not proof the target kept the text.
//!
//! A [`TargetSnapshot`]'s identity is its `target_ref`, a
//! process-independent safeToken string (`x11:<active>:<focus>[:<pid>]`,
//! `win:<hwnd>:<focus>:<pid>`, `fake:...`; ids in hex), so the runtime
//! host can revalidate a ref the app captured.

use std::fmt;
use std::time::{Duration, Instant};

#[cfg(feature = "runtime")]
pub mod runtime;
#[cfg(any(test, feature = "test-doubles"))]
pub mod testing;
#[cfg(windows)]
pub mod windows;
#[cfg(target_os = "linux")]
pub mod x11;

/// How long an insert waits for held modifiers to be released before
/// refusing with [`InsertError::ModifiersHeld`]. Bounded so a stuck key
/// delays one insert instead of hanging the delivery actor.
pub const MODIFIER_RELEASE_WAIT: Duration = Duration::from_millis(1500);
const MODIFIER_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Which backend captured a [`TargetSnapshot`] and must serve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    X11,
    Windows,
    /// The scripted test double (feature `test-doubles`).
    Fake,
}

impl BackendKind {
    /// The `target_ref` scheme prefix.
    pub fn scheme(self) -> &'static str {
        match self {
            BackendKind::X11 => "x11",
            BackendKind::Windows => "win",
            BackendKind::Fake => "fake",
        }
    }

    fn from_scheme(scheme: &str) -> Option<Self> {
        [BackendKind::X11, BackendKind::Windows, BackendKind::Fake]
            .into_iter()
            .find(|kind| kind.scheme() == scheme)
    }
}

/// The focused target as captured. `target_ref` is the whole identity;
/// `app`, `title` and `pid` are advisory display data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetSnapshot {
    pub backend: BackendKind,
    pub target_ref: String,
    /// `WM_CLASS` on X11, the exe name on Windows.
    pub app: Option<String>,
    pub title: Option<String>,
    pub pid: Option<u32>,
}

impl TargetSnapshot {
    /// The `(active, focus, pid)` carried by the ref, for diagnostics and
    /// tests that correlate a ref with a live window.
    pub fn ids(&self) -> Option<(u32, u32, Option<u32>)> {
        parse_ref(&self.target_ref).map(|(_, active, focus, pid)| (active, focus, pid))
    }
}

/// Text around the target's cursor; `selection` is a char range within
/// `before + after`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurroundingText {
    pub before: String,
    pub after: String,
    pub selection: Option<std::ops::Range<usize>>,
}

/// What revalidating a snapshot found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetCheck {
    Same,
    /// Focus or the active window moved; `actual` is the live ref.
    Changed {
        expected: String,
        actual: String,
    },
    /// The captured window no longer exists.
    Gone,
}

/// The only evidence a typing backend can give: synthetic keys were
/// sent, which is not proof the target kept the text.
pub const EVIDENCE_SYNTHETIC_KEYS: &str = "synthetic_keys_sent";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertReceipt {
    pub evidence: &'static str,
}

/// Every failure of capture/revalidate/insert. [`Self::code`] is the
/// safeToken that becomes `delivery.failed{reason}`; [`Self::message`] is
/// for humans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// The backend cannot run in this session (no display, a Wayland
    /// session, a dead connection).
    Unavailable {
        reason: String,
    },
    TargetChanged {
        expected: String,
        actual: String,
    },
    TargetGone,
    TargetIsStarling,
    /// Modifiers still held after [`MODIFIER_RELEASE_WAIT`]; `held` names
    /// them.
    ModifiersHeld {
        held: Vec<String>,
    },
    /// Typing stopped part-way. `delivered_chars` is an upper bound: a
    /// character counts once its key-down was accepted, even if the rest
    /// of its keystroke failed. Never produced with zero delivered; the
    /// bare cause is returned instead.
    PartialDelivery {
        delivered_chars: usize,
        total_chars: usize,
        cause: Box<InsertError>,
    },
    /// A keyboard state synthetic typing cannot reproduce faithfully
    /// (X11: a non-first group, a lock or latch other than Num_Lock).
    KeyboardStateUnsupported {
        reason: String,
    },
    /// A keycode borrowed for unmapped characters could not be given
    /// back; the keyboard may type wrong characters until the layout is
    /// reloaded.
    KeyboardRestoreFailed {
        detail: String,
    },
    MultilineUnsupported,
    /// Any other refusal: nothing focused, empty text, no spare keycode,
    /// a keyboard mapping changed by another client, blocked input.
    Rejected {
        reason: String,
    },
}

impl InsertError {
    pub fn code(&self) -> &'static str {
        match self {
            InsertError::Unavailable { .. } => "insertion_unavailable",
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

    pub fn message(&self) -> String {
        match self {
            InsertError::Unavailable { reason } => {
                format!("insertion is unavailable here: {reason}")
            }
            InsertError::TargetChanged { expected, actual } => format!(
                "the focused target changed since the take started (was {expected}, is now \
                 {actual})",
            ),
            InsertError::TargetGone => "the target window closed".to_string(),
            InsertError::TargetIsStarling => {
                "the focused window belongs to Starling itself".to_string()
            }
            InsertError::ModifiersHeld { held } => format!(
                "modifier keys are still held down ({}); release them so the text is not \
                 typed as commands",
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
            InsertError::KeyboardStateUnsupported { reason } => {
                format!("the keyboard is in a state synthetic typing cannot reproduce ({reason})")
            }
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

/// One platform's capture/revalidate/type implementation. Synchronous:
/// each call is a short X/Win32 round trip, made from the runtime's
/// delivery actor, which has no async runtime.
pub trait InsertionBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    /// `Err(InsertError::Unavailable)` when this backend cannot run in the
    /// current session. Never touches a target.
    fn availability(&self) -> Result<(), InsertError>;
    /// Capture the focused target now. Fails rather than guess when
    /// nothing usable is focused.
    fn capture(&self) -> Result<TargetSnapshot, InsertError>;
    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError>;
    /// Text around the target's cursor; `Ok(None)` when the backend
    /// cannot report it.
    fn surrounding_text(
        &self,
        _target: &TargetSnapshot,
    ) -> Result<Option<SurroundingText>, InsertError> {
        Ok(None)
    }
    /// Type `text` into `target`, following the crate rules above.
    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError>;
}

/// The ordered backends of a session. The first available backend
/// captures; a captured ref is always served by the backend of its
/// scheme.
pub struct Inserter {
    backends: Vec<Box<dyn InsertionBackend>>,
}

impl Inserter {
    /// The platform backends, refusing only this process as a target.
    pub fn for_this_session() -> Inserter {
        Self::with_excluded_pids(Vec::new())
    }

    /// The platform backends, refusing targets owned by any of
    /// `excluded` or this process. A host in its own process passes the
    /// app's pid so a ref the app captured is never typed back into the
    /// app.
    pub fn with_excluded_pids(excluded: Vec<u32>) -> Inserter {
        #[cfg(target_os = "linux")]
        let backends: Vec<Box<dyn InsertionBackend>> =
            vec![Box::new(x11::X11Backend::with_excluded_pids(excluded))];
        #[cfg(windows)]
        let backends: Vec<Box<dyn InsertionBackend>> = vec![Box::new(
            windows::WindowsBackend::with_excluded_pids(excluded),
        )];
        #[cfg(not(any(target_os = "linux", windows)))]
        let backends: Vec<Box<dyn InsertionBackend>> = {
            let _ = excluded;
            Vec::new()
        };
        Inserter::with_backends(backends)
    }

    pub fn with_backends(backends: Vec<Box<dyn InsertionBackend>>) -> Inserter {
        Inserter { backends }
    }

    /// Capture through the first available backend. When none is
    /// available, the first backend's reason is returned.
    pub fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        let mut first_blocker = None;
        for backend in &self.backends {
            match backend.availability() {
                Ok(()) => return backend.capture(),
                Err(error) => {
                    first_blocker.get_or_insert(error);
                }
            }
        }
        Err(first_blocker.unwrap_or(InsertError::Unavailable {
            reason: "no insertion backend exists for this platform".to_string(),
        }))
    }

    pub fn backend_for(&self, target: &TargetSnapshot) -> Option<&dyn InsertionBackend> {
        self.backends
            .iter()
            .map(|b| b.as_ref())
            .find(|b| b.kind() == target.backend)
    }

    /// Rebuild a snapshot from a ref (no server round trip). `app` and
    /// `title` are not part of the ref and come back `None`. `None` for a
    /// malformed ref or an unknown scheme.
    pub fn parse_target_ref(&self, target_ref: &str) -> Option<TargetSnapshot> {
        let (backend, _, _, pid) = parse_ref(target_ref)?;
        Some(TargetSnapshot {
            backend,
            target_ref: target_ref.to_string(),
            app: None,
            title: None,
            pid,
        })
    }

    pub fn describe(&self) -> String {
        if self.backends.is_empty() {
            "no insertion backend on this platform".to_string()
        } else {
            let schemes: Vec<_> = self.backends.iter().map(|b| b.kind().scheme()).collect();
            format!("insertion backends: {}", schemes.join(", "))
        }
    }
}

/// Lets one backend instance be shared between an [`Inserter`] and a
/// caller that keeps a handle to it (tests scripting a fake).
impl<T: InsertionBackend + ?Sized> InsertionBackend for std::sync::Arc<T> {
    fn kind(&self) -> BackendKind {
        (**self).kind()
    }
    fn availability(&self) -> Result<(), InsertError> {
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

/// `<scheme>:<active-hex>:<focus-hex>[:<pid>]`.
pub(crate) fn format_ref(kind: BackendKind, active: u32, focus: u32, pid: Option<u32>) -> String {
    match pid {
        Some(pid) => format!("{}:{active:x}:{focus:x}:{pid}", kind.scheme()),
        None => format!("{}:{active:x}:{focus:x}", kind.scheme()),
    }
}

/// The inverse of [`format_ref`]; strict, so a malformed ref fails
/// `delivery.prepare` instead of comparing unequal later.
pub(crate) fn parse_ref(target_ref: &str) -> Option<(BackendKind, u32, u32, Option<u32>)> {
    // The safeToken length bound; refs ride that field on the wire.
    if target_ref.len() > 128 {
        return None;
    }
    let (scheme, rest) = target_ref.split_once(':')?;
    let backend = BackendKind::from_scheme(scheme)?;
    let mut parts = rest.split(':');
    let active = u32::from_str_radix(parts.next()?, 16).ok()?;
    let focus = u32::from_str_radix(parts.next()?, 16).ok()?;
    let pid = match parts.next() {
        Some(pid) => Some(pid.parse::<u32>().ok()?),
        None => None,
    };
    if parts.next().is_some() || active == 0 {
        return None;
    }
    Some((backend, active, focus, pid))
}

/// `excluded` plus this process, sorted and deduplicated.
pub(crate) fn merge_excluded_pids(mut excluded: Vec<u32>) -> Vec<u32> {
    excluded.push(std::process::id());
    excluded.sort_unstable();
    excluded.dedup();
    excluded
}

/// The checks every insert runs before touching the platform. The
/// snapshot's pid only allows an early refusal; backends re-check the
/// live pid before every chunk.
pub(crate) fn insertion_guards(
    text: &str,
    pid: Option<u32>,
    excluded_pids: &[u32],
) -> Result<(), InsertError> {
    if text.is_empty() {
        return Err(InsertError::Rejected {
            reason: "empty text: nothing to insert".to_string(),
        });
    }
    if text.chars().any(char::is_control) {
        return Err(InsertError::MultilineUnsupported);
    }
    if pid.is_some_and(|pid| excluded_pids.contains(&pid)) {
        return Err(InsertError::TargetIsStarling);
    }
    Ok(())
}

/// Poll `held` until no modifier is held, refusing after
/// [`MODIFIER_RELEASE_WAIT`].
pub(crate) fn wait_modifiers_released(
    mut held: impl FnMut() -> Result<Option<Vec<String>>, InsertError>,
) -> Result<(), InsertError> {
    let deadline = Instant::now() + MODIFIER_RELEASE_WAIT;
    while let Some(names) = held()? {
        if Instant::now() >= deadline {
            return Err(InsertError::ModifiersHeld { held: names });
        }
        std::thread::sleep(MODIFIER_POLL_INTERVAL);
    }
    Ok(())
}

/// Split `text` into segments of at most `max_weight`, never splitting a
/// character (so a surrogate pair stays in one segment). A single
/// character heavier than the cap gets a segment of its own.
pub(crate) fn weighed_segments(
    text: &str,
    max_weight: usize,
    weight: impl Fn(char) -> usize,
) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut weight_sum = 0;
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

/// How many leading characters of `text` had a key-down accepted when
/// the first `accepted_events` of its `SendInput` stream (down/up per
/// UTF-16 unit) went in. A surrogate pair counts once its first unit's
/// key-down was accepted: half a pair may still have landed as garbage.
#[cfg(any(windows, test))]
pub(crate) fn chars_keyed_down_within_events(text: &str, accepted_events: usize) -> usize {
    let mut unit_index = 0;
    let mut characters = 0;
    for character in text.chars() {
        if 2 * unit_index >= accepted_events {
            break;
        }
        characters += 1;
        unit_index += character.len_utf16();
    }
    characters
}

/// A segment that failed part-way: how many of *its* characters may
/// have landed, and why it stopped.
pub(crate) struct ChunkFailure {
    pub delivered: usize,
    pub cause: InsertError,
}

/// The chunk loop of every platform backend: `check` before each
/// segment, then `type_segment` with what the check found, stopping at the
/// first failure.
pub(crate) fn deliver_in_chunks<C>(
    total_chars: usize,
    segments: &[&str],
    mut check: impl FnMut() -> Result<C, InsertError>,
    mut type_segment: impl FnMut(&str, C) -> Result<(), ChunkFailure>,
) -> Result<InsertReceipt, InsertError> {
    let mut delivered = 0;
    for segment in segments {
        let checked = check().map_err(|cause| partialize(delivered, total_chars, cause))?;
        type_segment(segment, checked).map_err(|failure| {
            partialize(delivered + failure.delivered, total_chars, failure.cause)
        })?;
        delivered += segment.chars().count();
    }
    Ok(InsertReceipt {
        evidence: EVIDENCE_SYNTHETIC_KEYS,
    })
}

fn partialize(delivered: usize, total_chars: usize, cause: InsertError) -> InsertError {
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
    use crate::testing::FakeBackend;

    /// The contract's safeToken pattern `^[A-Za-z0-9_.:+-]{1,128}$`.
    fn is_safe_token(token: &str) -> bool {
        !token.is_empty()
            && token.len() <= 128
            && token
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '+' | '-'))
    }

    #[test]
    fn error_codes_are_distinct_safe_tokens() {
        let samples = [
            InsertError::Unavailable {
                reason: "no display".into(),
            },
            InsertError::TargetChanged {
                expected: "x11:1:1".into(),
                actual: "x11:2:2".into(),
            },
            InsertError::TargetGone,
            InsertError::TargetIsStarling,
            InsertError::ModifiersHeld {
                held: vec!["Control".into()],
            },
            InsertError::PartialDelivery {
                delivered_chars: 16,
                total_chars: 64,
                cause: Box::new(InsertError::TargetGone),
            },
            InsertError::KeyboardStateUnsupported {
                reason: "Caps Lock is engaged".into(),
            },
            InsertError::KeyboardRestoreFailed {
                detail: "connection lost".into(),
            },
            InsertError::MultilineUnsupported,
            InsertError::Rejected {
                reason: "no focused window".into(),
            },
        ];
        for error in &samples {
            assert!(is_safe_token(error.code()), "{}", error.code());
        }
        let mut codes: Vec<_> = samples.iter().map(|e| e.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), samples.len());
    }

    #[test]
    fn partial_delivery_message_names_the_counts_and_cause() {
        let error = InsertError::PartialDelivery {
            delivered_chars: 16,
            total_chars: 64,
            cause: Box::new(InsertError::TargetGone),
        };
        assert_eq!(
            error.message(),
            "typing stopped part-way: up to 16 of 64 characters may have been typed before \
             the failure (the target window closed)"
        );
    }

    #[test]
    fn target_refs_round_trip_through_format_and_parse() {
        for (kind, active, focus, pid) in [
            (BackendKind::X11, 0x600001, 0x600002, Some(4213)),
            (BackendKind::Windows, 0x00060418, 0x00090c2e, Some(8)),
            (BackendKind::Fake, 1, 1, None),
            (BackendKind::X11, u32::MAX, 1, None),
        ] {
            let reference = format_ref(kind, active, focus, pid);
            assert!(is_safe_token(&reference), "{reference}");
            assert_eq!(parse_ref(&reference), Some((kind, active, focus, pid)));
        }
    }

    #[test]
    fn parse_ref_rejects_malformed_and_foreign_refs() {
        for bad in [
            "",
            "x11",
            "x11:",
            "x11:0:0",
            "x11:zz:1",
            "x11:1:2:notanumber",
            "x11:1:2:3:4",
            "x11:1:2:18446744073709551616",
            "x11:100000000:1",
            "macos:1:2",
            &"x:".repeat(100),
        ] {
            assert!(parse_ref(bad).is_none(), "{bad:?} must not parse");
        }
        assert_eq!(parse_ref("x11:ABC:Def:7"), parse_ref("x11:abc:def:7"));
    }

    #[test]
    fn inserter_reports_the_first_blocker_when_no_backend_is_available() {
        let blocked = |reason: &str| {
            let fake = FakeBackend::new();
            fake.set_availability(Err(InsertError::Unavailable {
                reason: reason.to_string(),
            }));
            Box::new(fake) as Box<dyn InsertionBackend>
        };
        let inserter = Inserter::with_backends(vec![blocked("first"), blocked("second")]);
        assert_eq!(
            inserter.capture(),
            Err(InsertError::Unavailable {
                reason: "first".into()
            })
        );
        assert!(matches!(
            Inserter::with_backends(Vec::new()).capture(),
            Err(InsertError::Unavailable { reason }) if reason.contains("no insertion backend")
        ));
    }

    #[test]
    fn segments_split_by_weight_without_cutting_characters() {
        let counts = |segments: Vec<&str>| -> Vec<usize> {
            segments.iter().map(|s| s.chars().count()).collect()
        };
        let text = "a".repeat(40);
        assert_eq!(counts(weighed_segments(&text, 16, |_| 1)), [16, 16, 8]);

        // 17 surrogate pairs = 34 UTF-16 units: 16 pairs, then the 17th.
        let emoji = "😀".repeat(17);
        let segments = weighed_segments(&emoji, 32, char::len_utf16);
        assert_eq!(segments.concat(), emoji);
        assert_eq!(counts(segments), [16, 1]);

        assert_eq!(weighed_segments("😀", 1, char::len_utf16), vec!["😀"]);
    }

    #[test]
    fn a_character_counts_once_its_first_keydown_was_accepted() {
        // "a😀b" sends down/up for a, hi, lo, b: events 0..8.
        let text = "a😀b";
        for (accepted, chars) in [(0, 0), (1, 1), (2, 1), (3, 2), (5, 2), (7, 3), (99, 3)] {
            assert_eq!(
                chars_keyed_down_within_events(text, accepted),
                chars,
                "{accepted} events accepted"
            );
        }
    }

    #[test]
    fn the_chunk_loop_checks_before_every_segment() {
        let text = "0123456789abcdefghij";
        let segments = weighed_segments(text, 16, |_| 1);
        let mut checks = 0;
        let mut typed = String::new();
        let receipt = deliver_in_chunks(
            20,
            &segments,
            || {
                checks += 1;
                Ok(())
            },
            |segment, ()| {
                typed.push_str(segment);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
        assert_eq!(checks, 2);
        assert_eq!(typed, text);
    }

    #[test]
    fn the_chunk_loop_reports_partial_delivery_only_after_something_was_typed() {
        let segments = weighed_segments("0123456789abcdefghij", 16, |_| 1);
        let partial = |delivered_chars| InsertError::PartialDelivery {
            delivered_chars,
            total_chars: 20,
            cause: Box::new(InsertError::TargetGone),
        };

        // A failing first check: nothing typed, the cause comes back bare.
        let error = deliver_in_chunks(
            20,
            &segments,
            || Err(InsertError::TargetGone),
            |_, ()| Ok(()),
        );
        assert_eq!(error, Err(InsertError::TargetGone));

        // A failing second check after 16 characters.
        let mut checks = 0;
        let error = deliver_in_chunks(
            20,
            &segments,
            || {
                checks += 1;
                if checks == 1 {
                    Ok(())
                } else {
                    Err(InsertError::TargetGone)
                }
            },
            |_, ()| Ok(()),
        );
        assert_eq!(error, Err(partial(16)));

        // The second segment fails after two of its characters.
        let error = deliver_in_chunks(
            20,
            &segments,
            || Ok(()),
            |segment, ()| {
                if segment.starts_with('0') {
                    Ok(())
                } else {
                    Err(ChunkFailure {
                        delivered: 2,
                        cause: InsertError::TargetGone,
                    })
                }
            },
        );
        assert_eq!(error, Err(partial(18)));

        // The first segment fails after its first key-down.
        let error = deliver_in_chunks(
            20,
            &segments,
            || Ok(()),
            |_, ()| {
                Err(ChunkFailure {
                    delivered: 1,
                    cause: InsertError::TargetGone,
                })
            },
        );
        assert_eq!(error, Err(partial(1)));
    }

    #[test]
    fn excluded_pids_always_include_this_process() {
        let own = std::process::id();
        assert_eq!(merge_excluded_pids(Vec::new()), vec![own]);
        let mut expected = vec![own, 7, 900];
        expected.sort_unstable();
        assert_eq!(merge_excluded_pids(vec![7, own, 7, 900]), expected);
    }
}
