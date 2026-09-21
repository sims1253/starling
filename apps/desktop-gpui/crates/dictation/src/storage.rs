//! Storage-facing types and shared storage helpers. The v1
//! `FileSessionStore` this module once held is deleted (D14: no
//! backwards compatibility of any kind) — [`crate::store_v2`] is THE
//! store; what remains here are the UI-shaped types the app's facade maps
//! v2 rows onto (`SessionStatus`, `SessionSummary`, `ListedRecord`,
//! `TranscriptionResult`), the error type shared across the storage
//! surface, and the small path/time utilities the store and journal
//! layers build on.

use std::io;
use std::path::Path;

use time::format_description::FormatItem;
use time::macros::format_description;
use time::OffsetDateTime;

const RFC3339_MILLIS: &[FormatItem] =
    format_description!("[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z");

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Captured,
    Transcribing,
    Transcribed,
    Failed,
    /// The take ended without a clean stop — an app crash found by journal
    /// recovery, or a stop-handshake quiesce timeout whose salvaged audio
    /// was persisted. The audio is kept and usable; the note in `last_error`
    /// states exactly what survived (e17 §2.1 `Interrupted`).
    Interrupted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("dictation session {0} was not found")]
    NotFound(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("stored dictation session is invalid: {0}")]
    Invalid(String),
    /// The platform data directory could not be resolved (on Linux,
    /// `$XDG_DATA_HOME` and `$HOME` are both unset). Returned by
    /// [`crate::store_v2::StoreV2::default_root`] instead of silently
    /// storing data in whatever directory the app happened to start in
    /// (R11).
    #[error("could not resolve the user data directory; set XDG_DATA_HOME or HOME")]
    DataDirUnavailable,
}

/// Metadata-only view of one session (G02): everything the history list and
/// drawer need — never the WAV bytes. Audio loads lazily per record on
/// demand (open/play/transcribe), so a listing never pulls the full history
/// into memory.
#[derive(Clone, Debug)]
pub struct SessionSummary {
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub status: SessionStatus,
    pub duration_ms: Option<f64>,
    pub attempt_count: u32,
    pub transcript: Option<TranscriptionResult>,
    pub last_error: Option<String>,
    /// Always `None` on v2 rows (the capture id *is* the journal linkage);
    /// kept in the shape the UI consumes.
    pub journal_id: Option<String>,
}

/// One damaged record as the listing reports it (G02): flagged with the
/// reason it failed, never deleted — the underlying evidence stays exactly
/// where it is until the user decides, and every read of that record
/// surfaces `reason` instead of a generic error.
#[derive(Clone, Debug)]
pub struct DamagedRecord {
    pub id: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub enum ListedRecord {
    Session(SessionSummary),
    Damaged(DamagedRecord),
}

/// RFC3339 UTC with exactly 3 subsecond digits (JS `Date.toISOString()`):
/// `2026-09-15T12:34:56.789Z`.
pub fn now_iso() -> String {
    OffsetDateTime::now_utc()
        .format(RFC3339_MILLIS)
        .expect("current time formats as RFC3339 with milliseconds")
}

/// Whether `id` is safe to join onto a filesystem path as one component:
/// non-empty, no separators or newlines, not `.` or `..`. Shared by capture
/// ids, journal ids, and any other identifier that becomes a path (R21).
pub(crate) fn is_safe_path_component(id: &str) -> bool {
    !id.is_empty()
        && !id.contains('/')
        && !id.contains('\\')
        && !id.contains('\r')
        && !id.contains('\n')
        && id != "."
        && id != ".."
}

/// fsync a directory's own entry (POSIX; best-effort no-op elsewhere) so
/// renames and creations inside it survive a power loss (#205). Shared by
/// the journal's tombstone path and the storage-v2 staging→audio
/// promotion.
///
/// Platform caveat (#247 review): on non-Unix (Windows) this is a no-op,
/// so the strict ordering guarantees documented on durable writes —
/// "once this returns, the new contents are durable" — hold as written on
/// Unix only. Every desktop target this port builds today is Unix; a
/// Windows build would need FlushFileBuffers on the directory handle to
/// make the same claim.
pub(crate) fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_iso_has_millisecond_precision_and_utc_suffix() {
        let stamp = now_iso();

        assert_eq!(
            stamp.len(),
            24,
            "expected YYYY-MM-DDTHH:MM:SS.mmmZ: {stamp}"
        );
        let bytes = stamp.as_bytes();
        assert_eq!(bytes[4], b'-');
        assert_eq!(bytes[7], b'-');
        assert_eq!(bytes[10], b'T');
        assert_eq!(bytes[13], b':');
        assert_eq!(bytes[16], b':');
        assert_eq!(bytes[19], b'.');
        assert_eq!(bytes[23], b'Z');
        assert!(stamp[..23].bytes().all(|byte| byte.is_ascii_digit()
            || byte == b'-'
            || byte == b'T'
            || byte == b':'
            || byte == b'.'));
        assert!(stamp[20..23].bytes().all(|byte| byte.is_ascii_digit()));
    }

    #[test]
    fn path_component_safety_rejects_traversal_and_emptiness() {
        assert!(is_safe_path_component("c_0123abcd"));
        assert!(is_safe_path_component("j_plain"));
        assert!(!is_safe_path_component(""), "empty");
        assert!(!is_safe_path_component("."), "current directory");
        assert!(!is_safe_path_component(".."), "parent traversal");
        assert!(!is_safe_path_component("a/b"), "posix separator");
        assert!(!is_safe_path_component("a\\b"), "windows separator");
        assert!(!is_safe_path_component("a\nb"), "newline");
        assert!(!is_safe_path_component("a\rb"), "carriage return");
    }
}
