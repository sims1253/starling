//! File-backed dictation session storage, ported from
//! `packages/dictation/src/storage.ts` (IndexedDB replaced by one directory per
//! session under a store root). See `apps/desktop-gpui/PORT.md`.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use time::format_description::FormatItem;
use time::macros::format_description;
use time::OffsetDateTime;

pub const DICTATION_SESSION_SCHEMA_VERSION: u32 = 1;

const AUDIO_FILE: &str = "recording.wav";
const MANIFEST_FILE: &str = "manifest.json";

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

#[derive(Clone, Debug)]
pub struct DictationSession {
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub status: SessionStatus,
    pub wav: Arc<Vec<u8>>,
    pub duration_ms: Option<f64>,
    pub attempt_count: u32,
    pub transcript: Option<TranscriptionResult>,
    pub transcript_history: Vec<TranscriptionResult>,
    pub last_error: Option<String>,
    /// Additive (I1 phase 2): the capture journal this session's audio was
    /// mirrored to, linking the session to its `journals/<journal_id>.sj`
    /// source evidence. Recovery skips journals already linked here, so a
    /// take can never be recovered twice. Absent on sessions created before
    /// journals or without one.
    pub journal_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("dictation session {0} was not found")]
    NotFound(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("stored dictation session is invalid: {0}")]
    Invalid(String),
}

/// On-disk mirror of `DictationSessionManifest` from storage.ts (camelCase
/// JSON): `{ schemaVersion, id, createdAt, updatedAt, status, audioFile,
/// durationMs?, attemptCount, transcript?, transcriptHistory, lastError? }`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionManifest {
    schema_version: u32,
    id: String,
    created_at: String,
    updated_at: String,
    status: SessionStatus,
    audio_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<f64>,
    attempt_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    transcript: Option<TranscriptionResult>,
    /// `exportDictationSession` always writes this key, even when empty.
    #[serde(default)]
    transcript_history: Vec<TranscriptionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    /// Additive I1 phase 2 linkage; absent in older manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    journal_id: Option<String>,
}

impl SessionManifest {
    fn from_session(session: &DictationSession) -> Self {
        Self {
            schema_version: DICTATION_SESSION_SCHEMA_VERSION,
            id: session.id.clone(),
            created_at: session.created_at.clone(),
            updated_at: session.updated_at.clone(),
            status: session.status,
            audio_file: AUDIO_FILE.to_string(),
            duration_ms: session.duration_ms,
            attempt_count: session.attempt_count,
            transcript: session.transcript.clone(),
            transcript_history: session.transcript_history.clone(),
            last_error: session.last_error.clone(),
            journal_id: session.journal_id.clone(),
        }
    }

    fn into_session(self, wav: Vec<u8>) -> DictationSession {
        DictationSession {
            id: self.id,
            created_at: self.created_at,
            updated_at: self.updated_at,
            status: self.status,
            wav: Arc::new(wav),
            duration_ms: self.duration_ms,
            attempt_count: self.attempt_count,
            transcript: self.transcript,
            transcript_history: self.transcript_history,
            last_error: self.last_error,
            journal_id: self.journal_id,
        }
    }
}

pub struct FileSessionStore {
    root: PathBuf,
}

impl FileSessionStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn default_root() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("starling-gpui")
            .join("sessions")
    }

    pub fn create(
        &self,
        wav: Vec<u8>,
        duration_ms: Option<f64>,
    ) -> Result<DictationSession, StorageError> {
        self.create_with_journal(wav, duration_ms, None)
    }

    /// [`Self::create`] with the additive journal linkage: `journal_id` is
    /// the id of the `journals/<journal_id>.sj` capture journal this
    /// session's audio was mirrored to (I1 phase 2). Journal recovery skips
    /// ids that appear here.
    pub fn create_with_journal(
        &self,
        wav: Vec<u8>,
        duration_ms: Option<f64>,
        journal_id: Option<&str>,
    ) -> Result<DictationSession, StorageError> {
        validate_wav(&wav)?;
        validate_duration_ms(duration_ms)?;

        let now = now_iso();
        let session = DictationSession {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: now.clone(),
            updated_at: now,
            status: SessionStatus::Captured,
            wav: Arc::new(wav),
            duration_ms,
            attempt_count: 0,
            transcript: None,
            transcript_history: Vec::new(),
            last_error: None,
            journal_id: journal_id.map(str::to_string),
        };

        let dir = self.session_dir(&session.id);
        std::fs::create_dir_all(&dir)?;
        // Write the retained audio first so the manifest only ever describes a
        // recording that is already on disk.
        std::fs::write(dir.join(AUDIO_FILE), session.wav.as_slice())?;
        self.write_manifest(&session)?;

        Ok(session)
    }

    pub fn get(&self, id: &str) -> Result<Option<DictationSession>, StorageError> {
        validate_session_id(id)?;
        self.read_session(id)
    }

    pub fn list(&self) -> Result<Vec<DictationSession>, StorageError> {
        let mut sessions = Vec::new();

        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            let id = entry.file_name().to_string_lossy().into_owned();

            if let Some(session) = self.read_session(&id)? {
                sessions.push(session);
            }
        }

        sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));

        Ok(sessions)
    }

    pub fn mark_attempt(&self, id: &str) -> Result<DictationSession, StorageError> {
        self.update(id, |mut session| {
            session.status = SessionStatus::Transcribing;
            session.attempt_count += 1;
            session.last_error = None;
            session
        })
    }

    pub fn save_transcript(
        &self,
        id: &str,
        transcript: TranscriptionResult,
    ) -> Result<DictationSession, StorageError> {
        self.update(id, |mut session| {
            // A successful retry keeps the earlier recognition in history.
            if let Some(previous) = session.transcript.take() {
                session.transcript_history.push(previous);
            }

            session.status = SessionStatus::Transcribed;
            session.transcript = Some(transcript);
            session.last_error = None;
            session
        })
    }

    pub fn save_failure(
        &self,
        id: &str,
        message: impl Into<String>,
    ) -> Result<DictationSession, StorageError> {
        let message = message.into();
        self.update(id, |mut session| {
            session.status = SessionStatus::Failed;
            session.last_error = Some(message);
            session
        })
    }

    /// Marks a session `interrupted` with a note stating exactly what
    /// survived (I1 phase 2): journal-recovered takes and quiesce-timeout
    /// salvages. Unlike [`Self::save_failure`] this is not a server attempt
    /// that failed — the audio itself is the recovered evidence, so the
    /// status says interrupted rather than failed.
    pub fn mark_interrupted(
        &self,
        id: &str,
        note: impl Into<String>,
    ) -> Result<DictationSession, StorageError> {
        let note = note.into();
        self.update(id, |mut session| {
            session.status = SessionStatus::Interrupted;
            session.last_error = Some(note);
            session
        })
    }

    /// Metadata-only scan of every session manifest's `journal_id` (I1
    /// phase 2). Reads manifests, never loads audio — the linkage check
    /// journal recovery runs at startup must not eagerly read every WAV
    /// (the G02 direction). A directory whose manifest cannot be parsed is
    /// skipped, not fatal: it cannot be linked, and quarantining damaged
    /// records is I2's job.
    pub fn journal_ids(&self) -> Result<HashSet<String>, StorageError> {
        let mut linked = HashSet::new();

        for entry in std::fs::read_dir(&self.root)? {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Ok(bytes) = std::fs::read(path.join(MANIFEST_FILE)) else {
                continue;
            };
            if let Ok(manifest) = serde_json::from_slice::<SessionManifest>(&bytes) {
                if let Some(journal_id) = manifest.journal_id {
                    linked.insert(journal_id);
                }
            }
        }

        Ok(linked)
    }

    pub fn delete(&self, id: &str) -> Result<(), StorageError> {
        validate_session_id(id)?;

        match std::fs::remove_dir_all(self.session_dir(id)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    fn session_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    fn read_session(&self, id: &str) -> Result<Option<DictationSession>, StorageError> {
        let manifest_path = self.session_dir(id).join(MANIFEST_FILE);
        let manifest_bytes = match std::fs::read(&manifest_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };

        let manifest: SessionManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| StorageError::Invalid(format!("{MANIFEST_FILE}: {error}")))?;

        if manifest.schema_version != DICTATION_SESSION_SCHEMA_VERSION {
            return Err(StorageError::Invalid(format!(
                "unsupported manifest schemaVersion {} (expected {DICTATION_SESSION_SCHEMA_VERSION})",
                manifest.schema_version
            )));
        }

        let wav = std::fs::read(self.session_dir(id).join(AUDIO_FILE))
            .map_err(|error| StorageError::Invalid(format!("{AUDIO_FILE}: {error}")))?;

        Ok(Some(manifest.into_session(wav)))
    }

    fn write_manifest(&self, session: &DictationSession) -> Result<(), StorageError> {
        let manifest = SessionManifest::from_session(session);
        let json = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| StorageError::Invalid(format!("{MANIFEST_FILE}: {error}")))?;
        write_atomic(&self.session_dir(&session.id).join(MANIFEST_FILE), &json)?;

        Ok(())
    }

    fn update(
        &self,
        id: &str,
        transform: impl FnOnce(DictationSession) -> DictationSession,
    ) -> Result<DictationSession, StorageError> {
        validate_session_id(id)?;

        let current = self
            .read_session(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;

        let mut next = transform(current);
        next.updated_at = now_iso();
        self.write_manifest(&next)?;

        Ok(next)
    }
}

/// In-memory twin of `MemorySessionStore` from storage.ts, used by tests.
#[derive(Default)]
pub struct MemorySessionStore {
    sessions: Mutex<HashMap<String, DictationSession>>,
}

impl MemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(
        &self,
        wav: Vec<u8>,
        duration_ms: Option<f64>,
    ) -> Result<DictationSession, StorageError> {
        self.create_with_journal(wav, duration_ms, None)
    }

    /// In-memory twin of [`FileSessionStore::create_with_journal`].
    pub fn create_with_journal(
        &self,
        wav: Vec<u8>,
        duration_ms: Option<f64>,
        journal_id: Option<&str>,
    ) -> Result<DictationSession, StorageError> {
        validate_wav(&wav)?;
        validate_duration_ms(duration_ms)?;

        let now = now_iso();
        let session = DictationSession {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: now.clone(),
            updated_at: now,
            status: SessionStatus::Captured,
            wav: Arc::new(wav),
            duration_ms,
            attempt_count: 0,
            transcript: None,
            transcript_history: Vec::new(),
            last_error: None,
            journal_id: journal_id.map(str::to_string),
        };

        let mut sessions = Self::lock(&self.sessions);
        sessions.insert(session.id.clone(), session.clone());

        Ok(session)
    }

    pub fn get(&self, id: &str) -> Result<Option<DictationSession>, StorageError> {
        Ok(Self::lock(&self.sessions).get(id).cloned())
    }

    pub fn list(&self) -> Result<Vec<DictationSession>, StorageError> {
        let mut sessions = Self::lock(&self.sessions)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));

        Ok(sessions)
    }

    pub fn mark_attempt(&self, id: &str) -> Result<DictationSession, StorageError> {
        self.update(id, |mut session| {
            session.status = SessionStatus::Transcribing;
            session.attempt_count += 1;
            session.last_error = None;
            session
        })
    }

    pub fn save_transcript(
        &self,
        id: &str,
        transcript: TranscriptionResult,
    ) -> Result<DictationSession, StorageError> {
        self.update(id, |mut session| {
            if let Some(previous) = session.transcript.take() {
                session.transcript_history.push(previous);
            }

            session.status = SessionStatus::Transcribed;
            session.transcript = Some(transcript);
            session.last_error = None;
            session
        })
    }

    pub fn save_failure(
        &self,
        id: &str,
        message: impl Into<String>,
    ) -> Result<DictationSession, StorageError> {
        let message = message.into();
        self.update(id, |mut session| {
            session.status = SessionStatus::Failed;
            session.last_error = Some(message);
            session
        })
    }

    /// In-memory twin of [`FileSessionStore::mark_interrupted`].
    pub fn mark_interrupted(
        &self,
        id: &str,
        note: impl Into<String>,
    ) -> Result<DictationSession, StorageError> {
        let note = note.into();
        self.update(id, |mut session| {
            session.status = SessionStatus::Interrupted;
            session.last_error = Some(note);
            session
        })
    }

    pub fn delete(&self, id: &str) -> Result<(), StorageError> {
        Self::lock(&self.sessions).remove(id);

        Ok(())
    }

    fn update(
        &self,
        id: &str,
        transform: impl FnOnce(DictationSession) -> DictationSession,
    ) -> Result<DictationSession, StorageError> {
        let mut sessions = Self::lock(&self.sessions);
        let next = sessions
            .get(id)
            .cloned()
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;

        let mut next = next;
        next.updated_at = now_iso();
        let next = transform(next);
        sessions.insert(id.to_string(), next.clone());

        Ok(next)
    }

    fn lock(
        sessions: &Mutex<HashMap<String, DictationSession>>,
    ) -> std::sync::MutexGuard<'_, HashMap<String, DictationSession>> {
        sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// RFC3339 UTC with exactly 3 subsecond digits (JS `Date.toISOString()`):
/// `2026-09-15T12:34:56.789Z`.
pub fn now_iso() -> String {
    OffsetDateTime::now_utc()
        .format(RFC3339_MILLIS)
        .expect("current time formats as RFC3339 with milliseconds")
}

fn validate_session_id(id: &str) -> Result<(), StorageError> {
    if id.is_empty()
        || id.contains('/')
        || id.contains('\\')
        || id.contains('\r')
        || id.contains('\n')
        || id == "."
        || id == ".."
    {
        return Err(StorageError::Invalid(format!(
            "session id {id:?} must be non-empty and contain no path separators"
        )));
    }

    Ok(())
}

fn validate_wav(wav: &[u8]) -> Result<(), StorageError> {
    if wav.is_empty() {
        return Err(StorageError::Invalid("wav must be non-empty".to_string()));
    }

    Ok(())
}

fn validate_duration_ms(duration_ms: Option<f64>) -> Result<(), StorageError> {
    if let Some(duration) = duration_ms {
        if !duration.is_finite() || duration < 0.0 {
            return Err(StorageError::Invalid(
                "durationMs must be a non-negative finite number".to_string(),
            ));
        }
    }

    Ok(())
}

/// Write through a sibling `<file>.tmp`, then rename over the target.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file_name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    file_name.push(".tmp");
    let tmp = path.with_file_name(file_name);

    let result = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }

    result
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use tempfile::TempDir;

    fn sample_wav() -> Vec<u8> {
        vec![82, 73, 70, 70, 1, 2, 3, 4]
    }

    fn transcript(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            text: text.to_string(),
            segments: Vec::new(),
            duration_seconds: None,
            request_id: None,
        }
    }

    /// Shared method surface so one suite can cover both stores.
    trait TestSessionStore {
        fn create(
            &self,
            wav: Vec<u8>,
            duration_ms: Option<f64>,
        ) -> Result<DictationSession, StorageError>;
        fn get(&self, id: &str) -> Result<Option<DictationSession>, StorageError>;
        fn list(&self) -> Result<Vec<DictationSession>, StorageError>;
        fn mark_attempt(&self, id: &str) -> Result<DictationSession, StorageError>;
        fn save_transcript(
            &self,
            id: &str,
            transcript: TranscriptionResult,
        ) -> Result<DictationSession, StorageError>;
        fn save_failure(&self, id: &str, message: &str) -> Result<DictationSession, StorageError>;
        fn delete(&self, id: &str) -> Result<(), StorageError>;
    }

    impl TestSessionStore for FileSessionStore {
        fn create(
            &self,
            wav: Vec<u8>,
            duration_ms: Option<f64>,
        ) -> Result<DictationSession, StorageError> {
            FileSessionStore::create(self, wav, duration_ms)
        }

        fn get(&self, id: &str) -> Result<Option<DictationSession>, StorageError> {
            FileSessionStore::get(self, id)
        }

        fn list(&self) -> Result<Vec<DictationSession>, StorageError> {
            FileSessionStore::list(self)
        }

        fn mark_attempt(&self, id: &str) -> Result<DictationSession, StorageError> {
            FileSessionStore::mark_attempt(self, id)
        }

        fn save_transcript(
            &self,
            id: &str,
            transcript: TranscriptionResult,
        ) -> Result<DictationSession, StorageError> {
            FileSessionStore::save_transcript(self, id, transcript)
        }

        fn save_failure(&self, id: &str, message: &str) -> Result<DictationSession, StorageError> {
            FileSessionStore::save_failure(self, id, message)
        }

        fn delete(&self, id: &str) -> Result<(), StorageError> {
            FileSessionStore::delete(self, id)
        }
    }

    impl TestSessionStore for MemorySessionStore {
        fn create(
            &self,
            wav: Vec<u8>,
            duration_ms: Option<f64>,
        ) -> Result<DictationSession, StorageError> {
            MemorySessionStore::create(self, wav, duration_ms)
        }

        fn get(&self, id: &str) -> Result<Option<DictationSession>, StorageError> {
            MemorySessionStore::get(self, id)
        }

        fn list(&self) -> Result<Vec<DictationSession>, StorageError> {
            MemorySessionStore::list(self)
        }

        fn mark_attempt(&self, id: &str) -> Result<DictationSession, StorageError> {
            MemorySessionStore::mark_attempt(self, id)
        }

        fn save_transcript(
            &self,
            id: &str,
            transcript: TranscriptionResult,
        ) -> Result<DictationSession, StorageError> {
            MemorySessionStore::save_transcript(self, id, transcript)
        }

        fn save_failure(&self, id: &str, message: &str) -> Result<DictationSession, StorageError> {
            MemorySessionStore::save_failure(self, id, message)
        }

        fn delete(&self, id: &str) -> Result<(), StorageError> {
            MemorySessionStore::delete(self, id)
        }
    }

    /// storage.test.ts: "retains audio through attempts, failures, and
    /// successful retry".
    fn retains_audio_through_attempts_failures_and_retry(
        store: &dyn TestSessionStore,
        wav_retained_by_reference: bool,
    ) {
        let wav = sample_wav();

        let captured = store.create(wav.clone(), Some(100.0)).expect("create");
        assert_eq!(captured.status, SessionStatus::Captured);
        assert_eq!(captured.attempt_count, 0);
        assert_eq!(captured.duration_ms, Some(100.0));
        assert!(captured.transcript.is_none());
        assert!(captured.transcript_history.is_empty());
        assert!(captured.last_error.is_none());

        let attempting = store.mark_attempt(&captured.id).expect("mark_attempt");
        assert_eq!(attempting.status, SessionStatus::Transcribing);
        assert_eq!(attempting.attempt_count, 1);
        assert_eq!(attempting.wav.as_slice(), wav.as_slice());
        if wav_retained_by_reference {
            assert!(Arc::ptr_eq(&attempting.wav, &captured.wav));
        }

        let failed = store
            .save_failure(&captured.id, "offline")
            .expect("save_failure");
        assert_eq!(failed.status, SessionStatus::Failed);
        assert_eq!(failed.last_error.as_deref(), Some("offline"));
        assert_eq!(failed.attempt_count, 1);
        assert_eq!(failed.wav.as_slice(), wav.as_slice());

        let retrying = store.mark_attempt(&captured.id).expect("mark_attempt 2");
        assert_eq!(retrying.status, SessionStatus::Transcribing);
        assert_eq!(retrying.attempt_count, 2);
        assert_eq!(retrying.last_error, None);

        let complete = store
            .save_transcript(&captured.id, transcript("agreed"))
            .expect("save_transcript");
        assert_eq!(complete.status, SessionStatus::Transcribed);
        assert_eq!(complete.attempt_count, 2);
        assert_eq!(
            complete.transcript.as_ref().expect("transcript").text,
            "agreed"
        );
        assert!(complete.transcript_history.is_empty());
        assert_eq!(complete.wav.as_slice(), wav.as_slice());
        assert!(complete.last_error.is_none());

        let stored = store
            .get(&captured.id)
            .expect("get")
            .expect("session still present");
        assert_eq!(stored.wav.as_slice(), wav.as_slice());
        assert_eq!(stored.status, SessionStatus::Transcribed);

        store.delete(&captured.id).expect("delete");
        assert!(store.get(&captured.id).expect("get after delete").is_none());
    }

    #[test]
    fn memory_store_retains_audio_through_attempts_failures_and_retry() {
        retains_audio_through_attempts_failures_and_retry(&MemorySessionStore::new(), true);
    }

    #[test]
    fn file_store_retains_audio_through_attempts_failures_and_retry() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        retains_audio_through_attempts_failures_and_retry(&store, false);
    }

    fn list_orders_by_updated_at_desc(store: &dyn TestSessionStore) {
        let first = store.create(sample_wav(), None).expect("create first");
        std::thread::sleep(Duration::from_millis(15));
        let second = store.create(sample_wav(), None).expect("create second");
        std::thread::sleep(Duration::from_millis(15));
        let third = store.create(sample_wav(), None).expect("create third");

        let listed = store.list().expect("list");
        let ids = listed
            .iter()
            .map(|session| session.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [third.id.as_str(), second.id.as_str(), first.id.as_str()]
        );

        // Touching the oldest session bumps it to the front.
        std::thread::sleep(Duration::from_millis(15));
        store.mark_attempt(&first.id).expect("mark_attempt");

        let listed = store.list().expect("list after touch");
        let ids = listed
            .iter()
            .map(|session| session.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [first.id.as_str(), third.id.as_str(), second.id.as_str()]
        );
    }

    #[test]
    fn memory_store_lists_by_updated_at_desc() {
        list_orders_by_updated_at_desc(&MemorySessionStore::new());
    }

    #[test]
    fn file_store_lists_by_updated_at_desc() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        list_orders_by_updated_at_desc(&store);
    }

    /// storage.test.ts: "retains the original recognition when a later retry
    /// changes meaning".
    fn save_transcript_moves_prior_transcript_into_history(store: &dyn TestSessionStore) {
        let session = store.create(sample_wav(), None).expect("create");

        store.mark_attempt(&session.id).expect("mark_attempt");
        store
            .save_transcript(&session.id, transcript("never merge this"))
            .expect("first transcript");
        store.mark_attempt(&session.id).expect("mark_attempt 2");

        let updated = store
            .save_transcript(&session.id, transcript("merge this"))
            .expect("second transcript");
        assert_eq!(
            updated.transcript.as_ref().expect("transcript").text,
            "merge this"
        );
        assert_eq!(updated.transcript_history.len(), 1);
        assert_eq!(updated.transcript_history[0].text, "never merge this");

        let stored = store
            .get(&session.id)
            .expect("get")
            .expect("session present");
        assert_eq!(
            stored.transcript.as_ref().expect("transcript").text,
            "merge this"
        );
        assert_eq!(stored.transcript_history[0].text, "never merge this");
    }

    #[test]
    fn memory_store_moves_prior_transcript_into_history() {
        save_transcript_moves_prior_transcript_into_history(&MemorySessionStore::new());
    }

    #[test]
    fn file_store_moves_prior_transcript_into_history() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        save_transcript_moves_prior_transcript_into_history(&store);
    }

    fn missing_sessions_are_not_found(store: &dyn TestSessionStore) {
        assert!(store.get("missing").expect("get").is_none());

        match store.mark_attempt("missing") {
            Err(StorageError::NotFound(id)) => assert_eq!(id, "missing"),
            other => panic!("expected NotFound, got {other:?}"),
        }

        match store.save_transcript("missing", transcript("nope")) {
            Err(StorageError::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }

        match store.save_failure("missing", "nope") {
            Err(StorageError::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn memory_store_reports_missing_sessions_as_not_found() {
        missing_sessions_are_not_found(&MemorySessionStore::new());
    }

    #[test]
    fn file_store_reports_missing_sessions_as_not_found() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        missing_sessions_are_not_found(&store);
    }

    fn create_rejects_invalid_input(store: &dyn TestSessionStore) {
        match store.create(Vec::new(), None) {
            Err(StorageError::Invalid(message)) => assert!(message.contains("non-empty")),
            other => panic!("expected Invalid for empty wav, got {other:?}"),
        }

        match store.create(sample_wav(), Some(-1.0)) {
            Err(StorageError::Invalid(message)) => assert!(message.contains("durationMs")),
            other => panic!("expected Invalid for negative duration, got {other:?}"),
        }

        match store.create(sample_wav(), Some(f64::NAN)) {
            Err(StorageError::Invalid(message)) => assert!(message.contains("durationMs")),
            other => panic!("expected Invalid for NaN duration, got {other:?}"),
        }

        // Zero is a valid duration.
        store
            .create(sample_wav(), Some(0.0))
            .expect("zero duration is accepted");
    }

    #[test]
    fn memory_store_rejects_invalid_create_input() {
        create_rejects_invalid_input(&MemorySessionStore::new());
    }

    #[test]
    fn file_store_rejects_invalid_create_input() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        create_rejects_invalid_input(&store);
    }

    fn delete_is_idempotent(store: &dyn TestSessionStore) {
        store.delete("never-existed").expect("delete missing id");
    }

    #[test]
    fn memory_store_delete_is_idempotent() {
        delete_is_idempotent(&MemorySessionStore::new());
    }

    #[test]
    fn file_store_delete_is_idempotent() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        delete_is_idempotent(&store);
    }

    #[test]
    fn file_store_persists_sessions_across_instances() {
        let temp = TempDir::new().expect("tempdir");
        let wav = sample_wav();

        let first = FileSessionStore::open(temp.path()).expect("open");
        let created = first.create(wav.clone(), Some(25.0)).expect("create");
        first.mark_attempt(&created.id).expect("mark_attempt");
        first
            .save_failure(&created.id, "network unavailable")
            .expect("save_failure");

        // Both store instances observe the same on-disk state (mirrors the TS
        // test using two IndexedDbSessionStore instances over one database).
        let reopened = FileSessionStore::open(temp.path()).expect("reopen");
        let restored = reopened
            .get(&created.id)
            .expect("get")
            .expect("session survives reopening the store");

        assert_eq!(restored.id, created.id);
        assert_eq!(restored.status, SessionStatus::Failed);
        assert_eq!(restored.attempt_count, 1);
        assert_eq!(restored.last_error.as_deref(), Some("network unavailable"));
        assert_eq!(restored.duration_ms, Some(25.0));
        assert_eq!(restored.wav.as_slice(), wav.as_slice());

        reopened.delete(&created.id).expect("delete");
        assert!(first.get(&created.id).expect("get after delete").is_none());
    }

    #[test]
    fn file_store_rejects_unsafe_session_ids() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        for bad in ["", "a/b", "a\\b", "a\rb", "a\nb", ".", ".."] {
            match store.get(bad) {
                Err(StorageError::Invalid(_)) => {}
                other => panic!("expected Invalid for id {bad:?}, got {other:?}"),
            }

            match store.mark_attempt(bad) {
                Err(StorageError::Invalid(_)) => {}
                other => panic!("expected Invalid for id {bad:?}, got {other:?}"),
            }

            match store.delete(bad) {
                Err(StorageError::Invalid(_)) => {}
                other => panic!("expected Invalid for id {bad:?}, got {other:?}"),
            }
        }

        // Generated ids are safe path components.
        let session = store.create(sample_wav(), None).expect("create");
        assert!(!session.id.contains('/') && !session.id.contains('\\'));
    }

    /// storage.test.ts: "rejects corrupted IndexedDB records at the schema
    /// boundary" — same idea for a corrupted manifest.json.
    #[test]
    fn file_store_rejects_corrupted_manifests() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");
        let session = store.create(sample_wav(), None).expect("create");
        let manifest_path = temp.path().join(&session.id).join(MANIFEST_FILE);

        // Mirror the TS test: attemptCount holding a string instead of a number.
        let wrong_type = format!(
            r#"{{"schemaVersion":1,"id":"{id}","createdAt":"2026-01-01T00:00:00.000Z","updatedAt":"2026-01-01T00:00:00.000Z","status":"captured","audioFile":"recording.wav","attemptCount":"not-a-number"}}"#,
            id = session.id
        );
        std::fs::write(&manifest_path, wrong_type).expect("write corrupted manifest");
        match store.get(&session.id) {
            Err(StorageError::Invalid(_)) => {}
            other => panic!("expected Invalid for wrong-typed attemptCount, got {other:?}"),
        }
        match store.list() {
            Err(StorageError::Invalid(_)) => {}
            other => panic!("expected Invalid listing a corrupted session, got {other:?}"),
        }

        // Truncated JSON.
        std::fs::write(&manifest_path, "{\"schemaVersion\":1,").expect("write truncated manifest");
        match store.get(&session.id) {
            Err(StorageError::Invalid(_)) => {}
            other => panic!("expected Invalid for truncated manifest, got {other:?}"),
        }

        // Unsupported schema version.
        std::fs::write(&manifest_path, r#"{"schemaVersion":999,"id":"x"}"#)
            .expect("write future manifest");
        match store.get(&session.id) {
            Err(StorageError::Invalid(_)) => {}
            other => panic!("expected Invalid for future schemaVersion, got {other:?}"),
        }
    }

    /// The manifest on disk mirrors `DictationSessionManifest` (camelCase,
    /// schemaVersion 1, audioFile "recording.wav").
    #[test]
    fn file_store_writes_a_versioned_manifest_beside_retained_audio() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        let session = store.create(sample_wav(), Some(25.0)).expect("create");

        let raw = std::fs::read_to_string(temp.path().join(&session.id).join(MANIFEST_FILE))
            .expect("read manifest");
        let manifest: serde_json::Value = serde_json::from_str(&raw).expect("parse manifest");

        assert_eq!(manifest["schemaVersion"], 1);
        assert_eq!(manifest["id"], session.id.as_str());
        assert_eq!(manifest["audioFile"], "recording.wav");
        assert_eq!(manifest["status"], "captured");
        assert_eq!(manifest["durationMs"], 25.0);
        assert_eq!(manifest["attemptCount"], 0);
        assert_eq!(manifest["transcriptHistory"], serde_json::json!([]));
        assert!(manifest.get("transcript").is_none());
        assert!(manifest.get("lastError").is_none());
        assert_eq!(manifest["createdAt"], session.created_at.as_str());
        assert_eq!(manifest["updatedAt"], session.updated_at.as_str());

        let stored_wav = std::fs::read(temp.path().join(&session.id).join(AUDIO_FILE))
            .expect("read retained wav");
        assert_eq!(stored_wav, sample_wav());

        // Transcript JSON is camelCase, matching the TS stored shape.
        let result = TranscriptionResult {
            text: "hello world".to_string(),
            segments: vec![TranscriptionSegment {
                text: "hello world".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.5,
            }],
            duration_seconds: Some(1.5),
            request_id: Some("request-one".to_string()),
        };
        store
            .save_transcript(&session.id, result)
            .expect("save transcript");

        let raw = std::fs::read_to_string(temp.path().join(&session.id).join(MANIFEST_FILE))
            .expect("read manifest");
        let manifest: serde_json::Value = serde_json::from_str(&raw).expect("parse manifest");
        assert_eq!(manifest["status"], "transcribed");
        assert_eq!(manifest["transcript"]["text"], "hello world");
        assert_eq!(manifest["transcript"]["segments"][0]["startSeconds"], 0.0);
        assert_eq!(manifest["transcript"]["segments"][0]["endSeconds"], 1.5);
        assert_eq!(manifest["transcript"]["durationSeconds"], 1.5);
        assert_eq!(manifest["transcript"]["requestId"], "request-one");
    }

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

    /// I1 phase 2: the additive `journalId` linkage round-trips through the
    /// manifest, is skipped when absent (older manifests), and never
    /// appears in manifests created without a journal.
    #[test]
    fn file_store_round_trips_the_additive_journal_linkage() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        // Without a journal: the key is absent from the JSON and None in
        // the session (schema stays backward-compatible for old readers).
        let plain = store.create(sample_wav(), None).expect("create plain");
        assert_eq!(plain.journal_id, None);
        let raw = std::fs::read_to_string(temp.path().join(&plain.id).join(MANIFEST_FILE))
            .expect("read manifest");
        let manifest: serde_json::Value = serde_json::from_str(&raw).expect("parse manifest");
        assert!(manifest.get("journalId").is_none(), "{raw}");

        // With a journal: present in the JSON and on the session.
        let linked = store
            .create_with_journal(sample_wav(), Some(10.0), Some("j_linkage"))
            .expect("create linked");
        assert_eq!(linked.journal_id.as_deref(), Some("j_linkage"));
        assert_eq!(linked.status, SessionStatus::Captured);
        let stored = store.get(&linked.id).expect("get").expect("present");
        assert_eq!(stored.journal_id.as_deref(), Some("j_linkage"));

        // An older manifest without the field still parses (serde default).
        let legacy = store.create(sample_wav(), None).expect("create legacy");
        let manifest_path = temp.path().join(&legacy.id).join(MANIFEST_FILE);
        let raw = std::fs::read_to_string(&manifest_path).expect("read");
        let mut manifest: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        manifest
            .as_object_mut()
            .expect("object")
            .remove("journalId");
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap())
            .expect("rewrite without journalId");
        let stored = store.get(&legacy.id).expect("get").expect("present");
        assert_eq!(stored.journal_id, None);
    }

    /// I1 phase 2: `mark_interrupted` is distinct from `save_failure` —
    /// the audio is recovered evidence, not a failed server attempt.
    #[test]
    fn interrupted_sessions_carry_their_note_and_survive_retry() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");
        let session = store
            .create_with_journal(sample_wav(), Some(80.0), Some("j_interrupted"))
            .expect("create");
        let interrupted = store
            .mark_interrupted(&session.id, "recovered with a torn tail of 12 bytes")
            .expect("mark interrupted");
        assert_eq!(interrupted.status, SessionStatus::Interrupted);
        assert_eq!(
            interrupted.last_error.as_deref(),
            Some("recovered with a torn tail of 12 bytes")
        );
        assert_eq!(interrupted.journal_id.as_deref(), Some("j_interrupted"));

        // The interrupted session is retryable like any other.
        let retrying = store.mark_attempt(&session.id).expect("mark attempt");
        assert_eq!(retrying.status, SessionStatus::Transcribing);
        assert_eq!(retrying.last_error, None);
        // …and the interruption is not forgotten: the linkage survives.
        assert_eq!(retrying.journal_id.as_deref(), Some("j_interrupted"));
    }

    /// I1 phase 2: the metadata-only linkage scan that journal recovery
    /// deduplicates against.
    #[test]
    fn journal_ids_scans_manifests_without_loading_audio() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        let one = store
            .create_with_journal(sample_wav(), None, Some("j_one"))
            .expect("create one");
        store
            .create_with_journal(sample_wav(), None, Some("j_two"))
            .expect("create two");
        store.create(sample_wav(), None).expect("create unlinked");

        assert_eq!(
            store.journal_ids().expect("journal ids"),
            HashSet::from(["j_one".to_string(), "j_two".to_string()])
        );

        // Deleting the session removes its linkage (a later recovery of the
        // same journal id would then legitimately recover it again).
        store.delete(&one.id).expect("delete one");
        assert_eq!(
            store.journal_ids().expect("journal ids after delete"),
            HashSet::from(["j_two".to_string()])
        );

        // A directory with a corrupt manifest cannot be linked and must not
        // abort the scan (quarantine is I2's job).
        std::fs::create_dir_all(temp.path().join("broken")).expect("broken dir");
        std::fs::write(temp.path().join("broken").join(MANIFEST_FILE), b"{nope")
            .expect("corrupt manifest");
        assert!(store.journal_ids().is_ok());
    }
}
