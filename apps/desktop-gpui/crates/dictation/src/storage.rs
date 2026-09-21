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

/// Trash tree under the store root (#206): the destination of
/// [`FileSessionStore::delete`]'s commit rename, mirroring the journal
/// root's `deleted/` (R21). A session mid-deletion sits here between the
/// rename and the removal; no records-root scan ever reads this tree, and
/// [`FileSessionStore::open`] sweeps whatever a crash left behind. Because
/// session ids *are* directory names, the name is reserved
/// ([`is_valid_session_dir_name`]) so a take can never collide with it.
pub(crate) const DELETED_DIR: &str = "deleted";

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
    /// The platform data directory could not be resolved (on Linux,
    /// `$XDG_DATA_HOME` and `$HOME` are both unset). Returned by
    /// [`FileSessionStore::default_root`] instead of silently storing
    /// sessions in whatever directory the app happened to start in (R11).
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
    pub journal_id: Option<String>,
}

impl From<&DictationSession> for SessionSummary {
    fn from(session: &DictationSession) -> Self {
        Self {
            id: session.id.clone(),
            created_at: session.created_at.clone(),
            updated_at: session.updated_at.clone(),
            status: session.status,
            duration_ms: session.duration_ms,
            attempt_count: session.attempt_count,
            transcript: session.transcript.clone(),
            last_error: session.last_error.clone(),
            journal_id: session.journal_id.clone(),
        }
    }
}

/// One record as [`FileSessionStore::list_records`] reports it (G02): a
/// readable session — good, or recovered from a WAV-only orphan directory —
/// or a damaged record quarantined in place with the reason it failed.
///
/// Damaged records are flagged, never deleted: the audio and manifest stay
/// exactly where they are until the user decides (the data may still be
/// recoverable by hand), and every read of that record (`get`, `update`,
/// play, transcribe) surfaces `reason` instead of a generic error.
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

impl ListedRecord {
    /// Sort key shared with the listing order: `updated_at` for sessions,
    /// empty for damaged records, so quarantined entries sort after every
    /// readable session instead of at a made-up timestamp.
    fn sort_key(&self) -> &str {
        match self {
            ListedRecord::Session(summary) => &summary.updated_at,
            ListedRecord::Damaged(_) => "",
        }
    }
}

/// One page of a metadata-only listing (G02): the records in
/// `[offset, offset + limit)` of the full ordering plus the un-paged
/// `total`, so callers can page without re-deriving counts.
#[derive(Clone, Debug)]
pub struct SessionPage {
    pub records: Vec<ListedRecord>,
    pub total: usize,
    pub offset: usize,
}

/// What [`FileSessionStore::save_journal_recovery`] did with one
/// journal-recovered take (#215).
#[derive(Clone, Debug)]
pub enum RecoverySave {
    /// No prior trace of the take: a fresh session was created, born
    /// `interrupted` and linked to its journal.
    Created(DictationSession),
    /// A prior trace of the take named after the journal was healed in
    /// place: audio rebuilt from the journal's verified bytes, manifest
    /// published with the linkage. No second session exists for the take.
    /// The trace is either a WAV-only orphan directory (the crash window
    /// between the WAV write and the manifest publish) or a directory
    /// damaged past classification (the power loss tore the WAV as well —
    /// nothing a verified journal cannot rebuild).
    Healed(DictationSession),
    /// A directory of that name already holds a valid manifest (a save
    /// raced the scan, or hand-crafted data); nothing was written and
    /// `String` says why. Recovery reports the journal and leaves it in
    /// place rather than duplicating the take.
    Occupied(String),
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

/// Serializes the journal-named session write paths within this process
/// (#247 review): `create_with_journal` (journal-linked branch) and
/// `save_journal_recovery` both classify-then-persist into a directory
/// whose name is fixed by the journal id, so a startup recovery racing an
/// interactive re-run could both observe "no prior trace" and interleave
/// writes. One lock turns that race into an ordered pair (the loser sees
/// the winner's record and reports `Occupied`). Cross-process exclusion is
/// not attempted here: two app instances on one data dir is outside the
/// store's contract (the same residual #213 documents for its sweep).
static RECOVERY_SERIALIZER: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct FileSessionStore {
    root: PathBuf,
}

impl FileSessionStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        // Reap trash a crashed delete left behind (#206): a session sits
        // under `deleted/` between the commit rename and the removal, and
        // nothing ever reads that tree — sweeping here is what keeps the
        // crash window from leaking the bytes forever. Best-effort: a
        // failed sweep leaves only trash, which the next open retries
        // (two app instances on one data dir are outside the store's
        // contract, the #213 stance the sweep shares).
        let _ = std::fs::remove_dir_all(root.join(DELETED_DIR));
        Ok(Self { root })
    }

    /// `dirs::data_dir()/starling-gpui/sessions`. A `None` from `dirs` is a
    /// typed error (R11): recorded audio must never silently land in an
    /// arbitrary working directory.
    pub fn default_root() -> Result<PathBuf, StorageError> {
        Ok(dirs::data_dir()
            .ok_or(StorageError::DataDirUnavailable)?
            .join("starling-gpui")
            .join("sessions"))
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
    ///
    /// A journal-linked take fixes its session id to the journal id (#215):
    /// the session directory is then named after the journal that mirrors
    /// it, so a crash between the WAV write and the manifest publish leaves
    /// a WAV-only orphan directory whose name alone identifies the journal
    /// it came from — startup recovery heals that orphan in place instead
    /// of creating a second session for the same take. The journal id
    /// becomes a path component, so an unsafe one is rejected rather than
    /// sanitized.
    pub fn create_with_journal(
        &self,
        wav: Vec<u8>,
        duration_ms: Option<f64>,
        journal_id: Option<&str>,
    ) -> Result<DictationSession, StorageError> {
        validate_wav(&wav)?;
        validate_duration_ms(duration_ms)?;

        let id = match journal_id {
            Some(journal_id) => {
                if !is_valid_session_dir_name(journal_id) {
                    return Err(StorageError::Invalid(format!(
                        "journal id {journal_id:?} must be non-empty, contain no path \
                         separators, and not be the reserved trash name {DELETED_DIR:?}"
                    )));
                }
                journal_id.to_string()
            }
            None => uuid::Uuid::new_v4().to_string(),
        };

        let now = now_iso();
        let session = DictationSession {
            id,
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

        // A journal-linked take writes into a directory whose name another
        // writer (journal recovery healing the same id) may target
        // concurrently — serialize those two writers in-process.
        let _guard = if journal_id.is_some() {
            Some(RECOVERY_SERIALIZER.lock().map_err(|_| {
                StorageError::Io(std::io::Error::other("recovery serializer poisoned"))
            })?)
        } else {
            None
        };
        if journal_id.is_some() {
            // The id names a fresh directory by contract (journal ids are
            // minted per take): anything already sitting under that name —
            // a valid record, an orphan, or damaged data — is a collision
            // (a raced heal, or a reused/hand-supplied id), and persisting
            // would clobber it. Reject the way save_journal_recovery does
            // (#247 review round 2).
            if !matches!(self.classify_record(&session.id), RecordState::Missing) {
                return Err(StorageError::Invalid(format!(
                    "session directory {:?} already exists; journal ids must be unique",
                    session.id
                )));
            }
        }
        self.persist_session_files(&session)?;

        Ok(session)
    }

    /// Durably lay down one session's audio and manifest (#205). The
    /// directory entry, the WAV bytes and the directory again are fsynced
    /// *before* the manifest is published, and the manifest itself is
    /// written through the fsyncing [`write_atomic`] — so once the manifest
    /// is durable, everything it describes already is. Without this, the
    /// write-first ordering only held for process crashes: under power
    /// loss, delayed allocation could make the manifest rename durable
    /// while the WAV's data blocks were lost or torn, leaving a session
    /// whose manifest parses but whose audio fails verification.
    fn persist_session_files(&self, session: &DictationSession) -> Result<(), StorageError> {
        let dir = self.session_dir(&session.id);
        std::fs::create_dir_all(&dir)?;
        // The new directory's own entry, then the audio, then the entry of
        // the audio — each durable before the next step observes it.
        sync_dir(&self.root)?;
        use std::io::Write as _;
        let mut audio = std::fs::File::create(dir.join(AUDIO_FILE))?;
        audio.write_all(session.wav.as_slice())?;
        audio.sync_all()?;
        sync_dir(&dir)?;
        // The manifest is published only after the recording it describes
        // is on disk — durably so.
        self.write_manifest(session)
    }

    pub fn get(&self, id: &str) -> Result<Option<DictationSession>, StorageError> {
        validate_session_id(id)?;
        self.read_session(id)
    }

    /// Metadata-only listing (G02): reads manifests and WAV headers, never
    /// loads audio into memory. One damaged record — corrupt manifest,
    /// missing/invalid audio, unsupported schema version, identity or path
    /// mismatch — quarantines only itself: it is returned as
    /// [`ListedRecord::Damaged`] with the reason while every other record
    /// stays visible. WAV-only directories (crash before the manifest write)
    /// come back as interrupted-but-usable sessions with the duration
    /// computed from the verified WAV bytes. Nothing is deleted or moved.
    pub fn list_records(&self) -> Result<Vec<ListedRecord>, StorageError> {
        let mut records = Vec::new();

        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            let id = entry.file_name().to_string_lossy().into_owned();

            // The trash tree (#206): a session mid-deletion lives here
            // between the commit rename and the removal. Never a record,
            // never classified, never listed.
            if id == DELETED_DIR {
                continue;
            }

            let record = match self.classify_record(&id) {
                RecordState::Manifest(manifest) => {
                    ListedRecord::Session(SessionSummary::from_manifest(manifest))
                }
                RecordState::Orphan(orphan) => {
                    ListedRecord::Session(orphan.into_summary(&id))
                }
                RecordState::Damaged(reason) => {
                    ListedRecord::Damaged(DamagedRecord { id, reason })
                }
                // Vanished between `read_dir` and the classification read:
                // nothing to report.
                RecordState::Missing => continue,
            };
            records.push(record);
        }

        // Newest first; damaged records (empty key) after every readable
        // session, ties broken by id so pages are deterministic.
        records.sort_by(|left, right| {
            right
                .sort_key()
                .cmp(left.sort_key())
                .then_with(|| record_id(left).cmp(&record_id(right)))
        });

        Ok(records)
    }

    /// [`Self::list_records`] paged: records `[offset, offset + limit)` of
    /// the full ordering plus the un-paged `total`.
    pub fn list_page(&self, offset: usize, limit: usize) -> Result<SessionPage, StorageError> {
        let records = self.list_records()?;
        let total = records.len();
        let records = records
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect::<Vec<_>>();

        Ok(SessionPage {
            records,
            total,
            offset,
        })
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

    /// Whether a session directory exists but carries no valid manifest of
    /// its own — the crash-window shapes (WAV-only orphan, or damaged past
    /// classification). The confirmed-delete fallback in
    /// [`crate::journal::delete_session_and_journal`] uses this to scope
    /// its journal-name linkage to takes that actually lack a manifest,
    /// so a healthy uuid-named session colliding with a journal id never
    /// gets its unrelated journal tombstoned (#247 review).
    pub(crate) fn is_unmanifested_session_dir(&self, id: &str) -> bool {
        matches!(
            self.classify_record(id),
            RecordState::Orphan(_) | RecordState::Damaged(_)
        )
    }

    /// Metadata-only scan of every session manifest's `journal_id` (I1
    /// phase 2). Reads manifests, never loads audio — the linkage check
    /// journal recovery runs at startup must not eagerly read every WAV
    /// (the G02 direction). A directory whose manifest cannot be parsed is
    /// skipped, not fatal: it cannot be linked, and quarantining damaged
    /// records is I2's job; G02's `list_records` flags them in the listing.
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
            // The trash tree (#206) is not a record: never scanned for
            // linkage, only ever swept.
            if entry.file_name().to_str() == Some(DELETED_DIR) {
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

    /// Metadata-only lookup of one session's journal linkage (R21): reads
    /// the manifest, never the audio, so the confirmed-delete flow can find
    /// the journal to quarantine without loading the WAV. A missing session
    /// (already deleted) or an unparseable manifest has no known linkage
    /// (`None`) — deletion then proceeds without a tombstone, matching
    /// [`Self::journal_ids`]' tolerance for damaged records (quarantining
    /// those is I2's job; G02's `list_records` flags them). A linkage that
    /// is not a safe path component is
    /// likewise dropped: it can never equal a scanned journal stem (file
    /// stems contain no separators), so dropping it cannot resurrect
    /// anything, while honoring it would brick the deletion.
    pub fn journal_id_of(&self, id: &str) -> Result<Option<String>, StorageError> {
        validate_session_id(id)?;

        let bytes = match std::fs::read(self.session_dir(id).join(MANIFEST_FILE)) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };

        Ok(serde_json::from_slice::<SessionManifest>(&bytes)
            .ok()
            .and_then(|manifest| manifest.journal_id)
            .filter(|journal_id| is_safe_path_component(journal_id)))
    }

    /// Sessions whose manifest parses and passes every manifest-level
    /// check, links a journal, but whose audio fails WAV verification
    /// (#205): `journal id -> session ids`. This is the power-loss shape —
    /// the manifest rename outlived the WAV's data blocks — and journal
    /// recovery consults it to rebuild those WAVs from the linked journal's
    /// checksum-verified samples instead of declaring the audio gone.
    ///
    /// Metadata-only apart from the bounded WAV header read (the same
    /// checks classification runs); a manifest that cannot be parsed has
    /// no readable linkage and is skipped — `list_records` already flags
    /// that record.
    pub fn damaged_audio_journal_links(
        &self,
    ) -> Result<HashMap<String, Vec<String>>, StorageError> {
        let mut links: HashMap<String, Vec<String>> = HashMap::new();

        for entry in std::fs::read_dir(&self.root)? {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            // The trash tree (#206) is not a record: never scanned for
            // damaged-audio linkage, only ever swept.
            if id == DELETED_DIR {
                continue;
            }
            let Ok(bytes) = read_manifest_bounded(&path.join(MANIFEST_FILE)) else {
                continue;
            };
            let Ok(manifest) = serde_json::from_slice::<SessionManifest>(&bytes) else {
                continue;
            };
            // Only the exact #205 shape: the manifest is fully valid for
            // its directory and carries a usable linkage; just the audio
            // is gone or torn.
            if manifest.schema_version != DICTATION_SESSION_SCHEMA_VERSION
                || manifest.id != id
                || manifest.audio_file != AUDIO_FILE
            {
                continue;
            }
            let Some(journal_id) = manifest.journal_id else {
                continue;
            };
            if !is_safe_path_component(&journal_id) {
                continue;
            }
            if verify_wav_header(&path.join(AUDIO_FILE)).is_ok() {
                continue;
            }
            links.entry(journal_id).or_default().push(id);
        }

        for session_ids in links.values_mut() {
            session_ids.sort();
        }

        Ok(links)
    }

    /// Rebuild one session's audio in place from journal-verified bytes
    /// (#205): a durable write of `wav` over `recording.wav`, fsynced with
    /// its directory. Repairing is not a second recovery: the session
    /// already exists, and its linkage is exactly what keeps the journal
    /// from being recovered into a new one.
    ///
    /// A **torn** rebuild (`note` present) restored only a verified prefix
    /// of the take the manifest still describes, so the manifest is updated
    /// in the same repair — status `interrupted`, the gap note, and the
    /// rebuilt audio's `duration_ms` — published as ONE atomic manifest
    /// rewrite, in a crash-safe order (#247 review):
    ///
    /// 1. the repaired WAV lands durable under a staging name;
    /// 2. the manifest is atomically rewritten to describe the rebuilt
    ///    audio (built from the manifest bytes directly — the live WAV is
    ///    still the torn one, so the verifying read paths cannot run);
    /// 3. the staging WAV is renamed into place and the directory fsynced.
    ///
    /// Every crash point re-enters this repair on the next startup: before
    /// step 2, or between 2 and 3, the live WAV still fails verification so
    /// `damaged_audio_journal_links` flags the session again and the repair
    /// re-runs (idempotently); after step 3 the audio and manifest agree.
    /// A finalized journal rebuilds byte-equivalent audio (`note` absent):
    /// the manifest already describes exactly this audio and is untouched.
    pub fn repair_session_audio(
        &self,
        id: &str,
        wav: &[u8],
        duration_ms: Option<f64>,
        note: Option<&str>,
    ) -> Result<(), StorageError> {
        validate_session_id(id)?;
        validate_wav(wav)?;

        let dir = self.session_dir(id);
        if !dir.is_dir() {
            return Err(StorageError::NotFound(id.to_string()));
        }

        use std::io::Write as _;
        if note.is_none() {
            // Byte-equivalent rebuild: write through, nothing else changes.
            let mut audio = std::fs::File::create(dir.join(AUDIO_FILE))?;
            audio.write_all(wav)?;
            audio.sync_all()?;
            sync_dir(&dir)?;
            return Ok(());
        }

        let staging = dir.join(format!("{AUDIO_FILE}.repair"));
        // Clear any staging file a prior crashed repair left behind
        // (#247 review round 2): a later re-entry would truncate it anyway,
        // but an errored repair or a crash before re-entry must not leave
        // the orphan behind forever.
        let _ = std::fs::remove_file(&staging);
        {
            let mut audio = std::fs::File::create(&staging)?;
            audio.write_all(wav)?;
            audio.sync_all()?;
        }

        // The manifest update is built from the manifest bytes directly:
        // `update()`/`read_session` verify the live audio, which is still
        // the torn recording at this point by construction.
        let manifest_path = dir.join(MANIFEST_FILE);
        let bytes = read_manifest_bounded(&manifest_path)
            .map_err(|reason| StorageError::Invalid(format!("{MANIFEST_FILE}: {reason}")))?;
        let mut manifest = serde_json::from_slice::<SessionManifest>(&bytes)
            .map_err(|error| StorageError::Invalid(format!("{MANIFEST_FILE}: {error}")))?;
        manifest.status = SessionStatus::Interrupted;
        manifest.last_error = Some(note.expect("torn repair always carries a note").to_string());
        if let Some(duration) = duration_ms {
            manifest.duration_ms = Some(duration);
        }
        manifest.updated_at = now_iso();
        let json = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| StorageError::Invalid(format!("{MANIFEST_FILE}: {error}")))?;
        write_atomic(&manifest_path, &json)?;

        std::fs::rename(&staging, dir.join(AUDIO_FILE))?;
        sync_dir(&dir)?;
        Ok(())
    }

    /// Persist one journal-recovered take under the session id equal to
    /// its journal id ([#215]): create the session, or — when a prior
    /// trace of the take already sits under that name — heal it in place
    /// instead of creating a second session beside it. The trace is the
    /// crash window inside [`Self::create_with_journal`] between the WAV
    /// write and the manifest publish: either a WAV-only orphan directory,
    /// or a directory the same power loss damaged past classification (a
    /// WAV too torn to verify). The directory's name (the journal id) is
    /// the only linkage that window leaves behind; healing it is what
    /// keeps one take from appearing twice in history. The session is born
    /// `interrupted` with `note` stating what survived, its manifest
    /// linking the journal.
    ///
    /// A directory of that name holding a *valid* manifest (a save raced
    /// the scan, or hand-crafted data) yields [`RecoverySave::Occupied`]
    /// — never a second directory for the take, and never a write over a
    /// record that already describes itself.
    ///
    /// [#215]: https://github.com/sims1253/starling/issues/215
    pub fn save_journal_recovery(
        &self,
        journal_id: &str,
        wav: Vec<u8>,
        duration_ms: Option<f64>,
        note: &str,
    ) -> Result<RecoverySave, StorageError> {
        validate_wav(&wav)?;
        validate_duration_ms(duration_ms)?;
        if !is_valid_session_dir_name(journal_id) {
            return Err(StorageError::Invalid(format!(
                "journal id {journal_id:?} must be non-empty, contain no path separators, \
                 and not be the reserved trash name {DELETED_DIR:?}"
            )));
        }

        // classify-then-persist is not atomic: serialize against the other
        // journal-named writer (a create racing this heal) in-process.
        let _guard = RECOVERY_SERIALIZER.lock().map_err(|_| {
            StorageError::Io(std::io::Error::other("recovery serializer poisoned"))
        })?;

        // The take's own stamp when a verified orphan carries one, else
        // now: healing rewrites the directory's contents but should not
        // pretend the take happened at healing time.
        let (prior_stamp, healed) = match self.classify_record(journal_id) {
            // No prior trace of the take: a fresh, linked session.
            RecordState::Missing => (None, false),
            // The #215 crash window with a verifiable WAV: audio exists,
            // the manifest does not. Heal the orphan — its audio is
            // replaced with the journal-verified bytes (a torn WAV tail is
            // rebuilt to the full verified prefix) and the manifest is
            // published with the linkage, under the take's own mtime
            // stamp.
            RecordState::Orphan(orphan) => (Some(orphan.stamp), true),
            // The #215 crash window with the WAV torn past verification
            // (the power loss struck before the audio fsync completed, so
            // the manifest was never attempted). Same heal: the journal's
            // verified bytes rebuild both the audio and the manifest the
            // crash took. There is no usable stamp — the orphan scan never
            // ran — so the session is stamped now. A manifest file that
            // EXISTS in a Damaged directory is not this window: it is an
            // unrelated (possibly still recoverable) record whose manifest
            // merely fails to parse or validate — healing over it would
            // clobber evidence and publish under an id it never carried
            // (#247 review round 2). Report the collision instead.
            RecordState::Damaged(reason) => {
                if dir_has_manifest(journal_id, &self.root) {
                    return Ok(RecoverySave::Occupied(format!(
                        "session directory {journal_id:?} holds damaged data ({reason}); the \
                         journal was left in place rather than healed over it"
                    )));
                }
                (None, true)
            }
            // The name is occupied by a record that already describes
            // itself: a manifest session (a save raced the scan, or
            // hand-crafted data). Never create a second directory for the
            // take beside it, and never write over a valid record; surface
            // the collision so the journal is reported and kept.
            RecordState::Manifest(_) => {
                return Ok(RecoverySave::Occupied(format!(
                    "session directory {journal_id:?} already exists; the journal was left \
                     in place rather than recovered into a second session"
                )));
            }
        };

        let stamp = prior_stamp.unwrap_or_else(now_iso);
        let session = DictationSession {
            id: journal_id.to_string(),
            created_at: stamp.clone(),
            updated_at: stamp,
            status: SessionStatus::Interrupted,
            wav: Arc::new(wav),
            duration_ms,
            attempt_count: 0,
            transcript: None,
            transcript_history: Vec::new(),
            last_error: Some(note.to_string()),
            journal_id: Some(journal_id.to_string()),
        };
        self.persist_session_files(&session)?;
        Ok(if healed {
            RecoverySave::Healed(session)
        } else {
            RecoverySave::Created(session)
        })
    }

    /// Confirmed deletion, made crash-safe (#206). The old
    /// `remove_dir_all` unlinked `manifest.json` and `recording.wav` as
    /// separate syscalls, so a crash between them left a WAV-only
    /// directory that the next startup classified as a recoverable orphan
    /// — the take the user just deleted reappeared in history as
    /// `interrupted`, the exact resurrection R21's tombstone exists to
    /// prevent, via a path it does not cover (it protects the linked
    /// journal, not the session's own audio). A plain delete has no
    /// tombstone at all, so any half-done removal resurrected the same
    /// way.
    ///
    /// Instead, mirroring [`crate::journal::quarantine_journal`]: one
    /// atomic rename moves the directory out of the records root into
    /// `<root>/deleted/` — the commit point, after which no scan
    /// (`list_records`, `journal_ids`, `damaged_audio_journal_links`,
    /// classification) can ever see the session again, whatever happens
    /// next — both directories are fsynced, and only then is the renamed
    /// directory removed. The crash matrix collapses to: before the
    /// rename, the session is fully intact (the delete simply did not
    /// happen); after it, only trash remains, which no code reads and the
    /// next [`Self::open`] sweeps. There is no interleaving that leaves a
    /// WAV-only directory in the records root.
    ///
    /// Idempotent, matching the old contract: deleting a missing session
    /// is `Ok(())` — an earlier delete, or a retry after a crash past the
    /// commit point. Failures before the rename propagate with the
    /// session still in place; the post-rename removal is best-effort for
    /// the same reason the open sweep is (its failure modes can neither
    /// resurrect anything nor surface as a delete that did not happen).
    pub fn delete(&self, id: &str) -> Result<(), StorageError> {
        validate_session_id(id)?;

        let dir = self.session_dir(id);
        match std::fs::symlink_metadata(&dir) {
            // Already gone — a retried delete after any crash point, or
            // the plain "never existed" case.
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }

        let trash = self.root.join(DELETED_DIR);
        std::fs::create_dir_all(&trash)?;
        let destination = trash.join(id);
        // Trash left under the same id by a delete that crashed between
        // its rename and its removal (and a backup then restored the
        // session) is cleared best-effort — a failure here must not fail
        // the delete; the rename retry below covers a destination that
        // still exists (it is already deleted data either way).
        let _ = std::fs::remove_dir_all(&destination);
        if std::fs::rename(&dir, &destination).is_err() {
            // A stale or raced destination survived the clear, or the
            // rename genuinely failed. Clear once more and retry; the
            // retry's error is the one that propagates (review on #250).
            let _ = std::fs::remove_dir_all(&destination);
            std::fs::rename(&dir, &destination)?;
        }
        sync_dir(&self.root)?;
        sync_dir(&trash)?;

        // Past the commit point: the session is gone from every scan. A
        // failure removing the trashed bytes cannot resurrect or hide
        // anything, so it is left to the open sweep rather than surfaced
        // as a failed delete.
        let _ = std::fs::remove_dir_all(&destination);
        Ok(())
    }

    fn session_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    /// Load one session in full (manifest plus audio bytes) for `get` and
    /// `update` — the only paths that may read a WAV (G02: audio loads
    /// lazily per record, never for a listing). A damaged record surfaces
    /// its exact classification reason; a WAV-only orphan directory is
    /// synthesized into a usable interrupted session (the first write — a
    /// retry, a status update — persists a manifest and heals it).
    fn read_session(&self, id: &str) -> Result<Option<DictationSession>, StorageError> {
        match self.classify_record(id) {
            RecordState::Missing => Ok(None),
            RecordState::Damaged(reason) => Err(StorageError::Invalid(reason)),
            RecordState::Manifest(manifest) => {
                let wav = std::fs::read(self.session_dir(id).join(AUDIO_FILE))
                    .map_err(|error| StorageError::Invalid(format!("{AUDIO_FILE}: {error}")))?;
                Ok(Some(manifest.into_session(wav)))
            }
            RecordState::Orphan(orphan) => {
                let wav = std::fs::read(self.session_dir(id).join(AUDIO_FILE))?;
                Ok(Some(orphan.into_session(id, Arc::new(wav))))
            }
        }
    }

    /// Classify one session directory without loading its audio (G02):
    /// usable manifest, recoverable orphan, damaged with a reason, or not
    /// present. Every manifest read is bounded; the WAV is verified by a
    /// bounded header scan only.
    fn classify_record(&self, id: &str) -> RecordState {
        let dir = self.session_dir(id);
        if !dir.is_dir() {
            return RecordState::Missing;
        }

        let manifest_path = dir.join(MANIFEST_FILE);
        match std::fs::metadata(&manifest_path) {
            // No manifest: the classic orphan (crash before the manifest
            // write — `create` publishes audio first). Recoverable when the
            // WAV verifies.
            Err(error) if error.kind() == io::ErrorKind::NotFound => match scan_orphan(&dir) {
                Ok(orphan) => RecordState::Orphan(orphan),
                Err(reason) => {
                    RecordState::Damaged(format!("no {MANIFEST_FILE}; {reason}"))
                }
            },
            Err(error) => RecordState::Damaged(format!("{MANIFEST_FILE}: {error}")),
            Ok(_) => {
                let manifest_bytes = match read_manifest_bounded(&manifest_path) {
                    Ok(bytes) => bytes,
                    Err(reason) => return RecordState::Damaged(reason),
                };

                match serde_json::from_slice::<SessionManifest>(&manifest_bytes) {
                    Ok(manifest) => validate_manifest(id, manifest, &dir),
                    // Unusable manifest with the audio intact is still an
                    // orphan: recovering the recording matters more than the
                    // file that failed to describe it.
                    Err(parse_error) => match scan_orphan(&dir) {
                        Ok(orphan) => RecordState::Orphan(orphan),
                        Err(wav_reason) => RecordState::Damaged(format!(
                            "{MANIFEST_FILE}: {parse_error}; {wav_reason}"
                        )),
                    },
                }
            }
        }
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

    /// In-memory twin of [`FileSessionStore::create_with_journal`],
    /// including its #215 rule that a journal-linked take's session id is
    /// the journal id (and that an unsafe journal id is rejected).
    pub fn create_with_journal(
        &self,
        wav: Vec<u8>,
        duration_ms: Option<f64>,
        journal_id: Option<&str>,
    ) -> Result<DictationSession, StorageError> {
        validate_wav(&wav)?;
        validate_duration_ms(duration_ms)?;

        let id = match journal_id {
            Some(journal_id) => {
                if !is_valid_session_dir_name(journal_id) {
                    return Err(StorageError::Invalid(format!(
                        "journal id {journal_id:?} must be non-empty, contain no path \
                         separators, and not be the reserved trash name {DELETED_DIR:?}"
                    )));
                }
                journal_id.to_string()
            }
            None => uuid::Uuid::new_v4().to_string(),
        };

        let now = now_iso();
        let session = DictationSession {
            id,
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

/// Whether `id` is safe to join onto a filesystem path as one component:
/// non-empty, no separators or newlines, not `.` or `..`. Shared by session
/// ids and manifest-provided journal ids (R21).
pub(crate) fn is_safe_path_component(id: &str) -> bool {
    !id.is_empty()
        && !id.contains('/')
        && !id.contains('\\')
        && !id.contains('\r')
        && !id.contains('\n')
        && id != "."
        && id != ".."
}

fn validate_session_id(id: &str) -> Result<(), StorageError> {
    if !is_safe_path_component(id) {
        return Err(StorageError::Invalid(format!(
            "session id {id:?} must be non-empty and contain no path separators"
        )));
    }
    if id == DELETED_DIR {
        return Err(StorageError::Invalid(format!(
            "session id {id:?} is reserved for the deletion trash directory"
        )));
    }

    Ok(())
}

/// Whether `id` may name a session directory under the store root: a safe
/// path component that is not [`DELETED_DIR`] (#206). Session ids *are*
/// directory names, so the trash tree's own name must stay unassignable —
/// a take named `deleted` would collide with the deletion commit point.
/// Backward compatibility (review on #250): an id literally equal to
/// `"deleted"` was theoretically addressable before the trash existed —
/// in practice minted ids are uuids or `j_*` journal ids, so no real
/// record can collide; if one ever did (hand-crafted data), its directory
/// is now quarantined-by-design: unreadable through the store's paths and
/// swept by `open` alongside other trash.
fn is_valid_session_dir_name(id: &str) -> bool {
    is_safe_path_component(id) && id != DELETED_DIR
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

/// fsync a directory's own entry (POSIX; best-effort no-op elsewhere) so
/// renames and creations inside it survive a power loss (#205). Shared by
/// the session store's durable writes, the journal's tombstone path, and
/// the storage-v2 staging→audio promotion.
///
/// Platform caveat (#247 review): on non-Unix (Windows) this is a no-op,
/// so the strict ordering guarantees documented on
/// [`FileSessionStore::persist_session_files`]-style durable writes —
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

/// Whether a session directory under `root` still holds a manifest FILE
/// (parseable or not) — used by [`FileSessionStore::save_journal_recovery`]
/// to keep the #215 heal scoped to the true crash window, where the
/// manifest was never attempted (#247 review round 2).
fn dir_has_manifest(id: &str, root: &Path) -> bool {
    root.join(id).join(MANIFEST_FILE).is_file()
}

/// Write through a sibling `<file>.tmp`, then rename over the target. The
/// temporary file is fsynced before the rename and the containing
/// directory after it (#205), so once this returns the target's new
/// contents are durable — a rename made durable while its data is not is
/// exactly the torn-file window the session store's ordering depends on.
/// (Unix only: the directory fsync is [`sync_dir`], a no-op elsewhere —
/// see its platform caveat.)
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;

    let mut file_name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    file_name.push(".tmp");
    let tmp = path.with_file_name(file_name);

    let result = (|| {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)?;
        if let Some(parent) = path.parent() {
            sync_dir(parent)?;
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }

    result
}

// ---------------------------------------------------------------------------
// G02: per-record damage classification and orphan recovery.
// ---------------------------------------------------------------------------

/// Upper bound on a manifest read ("bound file reads"): real manifests are a
/// few KiB of JSON; anything past this is damaged, not loaded.
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

/// Upper bound on the WAV header chunk walk, so a corrupt file with a
/// zero-size chunk loop cannot spin the scanner.
const MAX_WAV_SCAN_CHUNKS: usize = 4096;

/// The note an orphan-recovered session carries in `last_error`: it states
/// exactly what survived and what to do with it, in the voice of the other
/// `Interrupted` notes (e17 §2.1).
const ORPHAN_RECOVERY_NOTE: &str = "Recovered from a recording that was left without its \
     session data — the app most likely stopped before the session was saved. The audio was \
     kept untouched; you can play it, export it, or retry transcription.";

/// What one session directory's classification found (G02).
enum RecordState {
    /// Manifest parsed and validated; the audio file is present with a
    /// verifiable WAV header.
    Manifest(SessionManifest),
    /// No usable manifest, but the WAV verified: a recoverable take with the
    /// duration computed from the verified bytes and a timestamp taken from
    /// the audio file.
    Orphan(OrphanRecord),
    /// The record is damaged; the reason is surfaced by listings and reads.
    Damaged(String),
    /// The directory does not exist (deleted, or never created).
    Missing,
}

/// A WAV-only session directory promoted to a recoverable take (G02).
struct OrphanRecord {
    /// Duration from the verified WAV bytes (data length / byte rate).
    duration_ms: Option<f64>,
    /// RFC3339 millis timestamp from the audio file's mtime — the only clock
    /// an orphan has.
    stamp: String,
}

impl OrphanRecord {
    fn into_summary(self, id: &str) -> SessionSummary {
        SessionSummary {
            id: id.to_string(),
            created_at: self.stamp.clone(),
            updated_at: self.stamp,
            status: SessionStatus::Interrupted,
            duration_ms: self.duration_ms,
            attempt_count: 0,
            transcript: None,
            last_error: Some(ORPHAN_RECOVERY_NOTE.to_string()),
            journal_id: None,
        }
    }

    fn into_session(self, id: &str, wav: Arc<Vec<u8>>) -> DictationSession {
        DictationSession {
            id: id.to_string(),
            created_at: self.stamp.clone(),
            updated_at: self.stamp,
            status: SessionStatus::Interrupted,
            wav,
            duration_ms: self.duration_ms,
            attempt_count: 0,
            transcript: None,
            transcript_history: Vec::new(),
            last_error: Some(ORPHAN_RECOVERY_NOTE.to_string()),
            journal_id: None,
        }
    }
}

impl SessionSummary {
    fn from_manifest(manifest: SessionManifest) -> Self {
        Self {
            id: manifest.id,
            created_at: manifest.created_at,
            updated_at: manifest.updated_at,
            status: manifest.status,
            duration_ms: manifest.duration_ms,
            attempt_count: manifest.attempt_count,
            transcript: manifest.transcript,
            last_error: manifest.last_error,
            journal_id: manifest.journal_id,
        }
    }
}

fn record_id(record: &ListedRecord) -> &str {
    match record {
        ListedRecord::Session(summary) => &summary.id,
        ListedRecord::Damaged(damaged) => &damaged.id,
    }
}

/// Validate a parsed manifest against its directory (G02): schema version,
/// identity (manifest id == directory name) and audio path must agree, and
/// the audio file must exist with a WAV header. The audio itself is never
/// loaded here — the bounded header read is the check.
fn validate_manifest(id: &str, manifest: SessionManifest, dir: &Path) -> RecordState {
    if manifest.schema_version != DICTATION_SESSION_SCHEMA_VERSION {
        return RecordState::Damaged(format!(
            "unsupported manifest schemaVersion {} (expected \
             {DICTATION_SESSION_SCHEMA_VERSION})",
            manifest.schema_version
        ));
    }

    if manifest.id != id {
        return RecordState::Damaged(format!(
            "manifest id {:?} does not match its session directory {:?}",
            manifest.id, id
        ));
    }

    if manifest.audio_file != AUDIO_FILE {
        return RecordState::Damaged(format!(
            "manifest audioFile {:?} does not match the stored recording \
             {AUDIO_FILE:?}",
            manifest.audio_file
        ));
    }

    match verify_wav_header(&dir.join(AUDIO_FILE)) {
        Ok(()) => RecordState::Manifest(manifest),
        Err(reason) => RecordState::Damaged(reason),
    }
}

/// Read a manifest with a hard byte bound. Returns the parse-ready bytes or
/// the damaged-record reason (including a NotFound mapped by the caller's
/// metadata pre-check, so this only sees real read failures).
fn read_manifest_bounded(path: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;

    let file =
        std::fs::File::open(path).map_err(|error| format!("{MANIFEST_FILE}: {error}"))?;
    let mut bytes = Vec::new();
    let mut bounded = file.take(MAX_MANIFEST_BYTES + 1);
    bounded
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{MANIFEST_FILE}: {error}"))?;

    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(format!(
            "{MANIFEST_FILE}: larger than {MAX_MANIFEST_BYTES} bytes and not a session \
             manifest"
        ));
    }

    Ok(bytes)
}

/// Verify a session's audio with one bounded read: the 12-byte
/// RIFF/WAVE magic. Full validation happens when the audio is actually
/// loaded (play/transcribe decode it anyway); a listing must never read
/// the body.
fn verify_wav_header(path: &Path) -> Result<(), String> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).map_err(|error| format!("{AUDIO_FILE}: {error}"))?;
    let mut magic = [0u8; 12];
    file.read_exact(&mut magic)
        .map_err(|error| format!("{AUDIO_FILE}: {error}"))?;

    if &magic[0..4] != b"RIFF" || &magic[8..12] != b"WAVE" {
        return Err(format!("{AUDIO_FILE}: not a RIFF/WAVE recording"));
    }

    Ok(())
}

/// Scan a WAV-only directory for a recoverable take (G02): verify the WAV
/// header structure with bounded reads (never load the samples) and compute
/// the duration from the data chunk length over the fmt byte rate, clamped
/// to the bytes actually present — a crash mid-write leaves the data size
/// field promising more than the file holds, and the honest duration is the
/// verified one.
fn scan_orphan(dir: &Path) -> Result<OrphanRecord, String> {
    use std::io::{Read, Seek, SeekFrom};

    let path = dir.join(AUDIO_FILE);
    let mut file =
        std::fs::File::open(&path).map_err(|error| format!("{AUDIO_FILE}: {error}"))?;
    let file_len = file
        .metadata()
        .map_err(|error| format!("{AUDIO_FILE}: {error}"))?
        .len();

    let mut magic = [0u8; 12];
    file.read_exact(&mut magic)
        .map_err(|error| format!("{AUDIO_FILE}: {error}"))?;
    if &magic[0..4] != b"RIFF" || &magic[8..12] != b"WAVE" {
        return Err(format!("{AUDIO_FILE}: not a RIFF/WAVE recording"));
    }

    let mut byte_rate: Option<u32> = None;
    let mut data_len: Option<u64> = None;
    let mut offset: u64 = 12;

    for _ in 0..MAX_WAV_SCAN_CHUNKS {
        if offset + 8 > file_len {
            break;
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("{AUDIO_FILE}: {error}"))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)
            .map_err(|error| format!("{AUDIO_FILE}: {error}"))?;
        let chunk_id = [header[0], header[1], header[2], header[3]];
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as u64;
        let body = offset + 8;
        let available = file_len.saturating_sub(body);

        if &chunk_id == b"fmt " {
            if size < 16 {
                return Err(format!("{AUDIO_FILE}: fmt chunk is truncated"));
            }
            let mut fmt = [0u8; 16];
            file.read_exact(&mut fmt)
                .map_err(|error| format!("{AUDIO_FILE}: {error}"))?;
            let rate = u32::from_le_bytes([fmt[8], fmt[9], fmt[10], fmt[11]]);
            if rate == 0 {
                return Err(format!("{AUDIO_FILE}: fmt chunk has a zero byte rate"));
            }
            byte_rate = Some(rate);
        } else if &chunk_id == b"data" {
            // Clamp to the bytes actually on disk: a torn write promises
            // more data than the file holds.
            data_len = Some(size.min(available));
            if byte_rate.is_some() {
                break;
            }
        }

        offset = body + size + (size % 2);
        if offset >= file_len {
            break;
        }
    }

    let duration_ms = match (byte_rate, data_len) {
        (Some(rate), Some(len)) => len as f64 * 1000.0 / f64::from(rate),
        _ => return Err(format!("{AUDIO_FILE}: missing fmt or data chunk")),
    };

    let stamp = std::fs::metadata(&path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| {
            OffsetDateTime::from(modified)
                .format(RFC3339_MILLIS)
                .ok()
        })
        .unwrap_or_else(now_iso);

    Ok(OrphanRecord {
        duration_ms: Some(duration_ms),
        stamp,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use tempfile::TempDir;

    /// A minimal valid WAV (canonical header, no samples). Since G02 the
    /// store verifies the RIFF/WAVE magic even on metadata-only reads, so
    /// test audio has to be a plausible recording.
    fn sample_wav() -> Vec<u8> {
        wav_bytes(0.0)
    }

    fn transcript(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            text: text.to_string(),
            segments: Vec::new(),
            duration_seconds: None,
            request_id: None,
        }
    }

    /// Shared method surface so one suite can cover both stores. Listing is
    /// deliberately not part of it: `FileSessionStore` lists metadata-only
    /// (G02) while the in-memory twin hands out its sessions directly.
    trait TestSessionStore {
        fn create(
            &self,
            wav: Vec<u8>,
            duration_ms: Option<f64>,
        ) -> Result<DictationSession, StorageError>;
        fn get(&self, id: &str) -> Result<Option<DictationSession>, StorageError>;
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

    /// `list_ids` re-reads the store's listing (metadata-only for the file
    /// store, G02) and returns the readable session ids in order.
    fn list_orders_by_updated_at_desc(
        store: &dyn TestSessionStore,
        list_ids: impl Fn() -> Vec<String>,
    ) {
        let first = store.create(sample_wav(), None).expect("create first");
        std::thread::sleep(Duration::from_millis(15));
        let second = store.create(sample_wav(), None).expect("create second");
        std::thread::sleep(Duration::from_millis(15));
        let third = store.create(sample_wav(), None).expect("create third");

        assert_eq!(
            list_ids(),
            [third.id.as_str(), second.id.as_str(), first.id.as_str()]
        );

        // Touching the oldest session bumps it to the front.
        std::thread::sleep(Duration::from_millis(15));
        store.mark_attempt(&first.id).expect("mark_attempt");

        assert_eq!(
            list_ids(),
            [first.id.as_str(), third.id.as_str(), second.id.as_str()]
        );
    }

    #[test]
    fn memory_store_lists_by_updated_at_desc() {
        let store = MemorySessionStore::new();
        let listed = || {
            store
                .list()
                .expect("list")
                .into_iter()
                .map(|session| session.id)
                .collect::<Vec<_>>()
        };

        list_orders_by_updated_at_desc(&store, listed);
    }

    #[test]
    fn file_store_lists_by_updated_at_desc() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");
        let listed = || {
            store
                .list_records()
                .expect("list")
                .into_iter()
                .filter_map(|record| match record {
                    ListedRecord::Session(summary) => Some(summary.id),
                    ListedRecord::Damaged(_) => None,
                })
                .collect::<Vec<_>>()
        };

        list_orders_by_updated_at_desc(&store, listed);
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

    /// #206: a completing delete removes the session's bytes and the trash
    /// it created, and stays idempotent on a real session — the retried
    /// delete after any crash point is `Ok`. Stale trash under the same id
    /// (a delete that crashed past its rename, then a backup restored the
    /// session) is replaced by the commit rename, not fatal.
    #[test]
    fn file_store_delete_removes_bytes_and_stays_idempotent() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");
        let created = store.create(sample_wav(), Some(25.0)).expect("create");
        let session_dir = temp.path().join(&created.id);

        let trash = temp.path().join(DELETED_DIR);
        std::fs::create_dir_all(trash.join(&created.id)).expect("stale trash under the id");

        store.delete(&created.id).expect("delete");
        assert!(!session_dir.exists(), "session directory removed");
        assert!(store.get(&created.id).expect("get").is_none());
        assert!(
            !trash.join(&created.id).exists(),
            "no bytes linger in the trash, stale or fresh"
        );
        assert!(
            store.list_records().expect("list").is_empty(),
            "the trash directory itself must not be listed as a record"
        );

        store.delete(&created.id).expect("delete again is Ok");
    }

    /// #206, crash point (a): the delete committed its rename into the
    /// trash tree but died before the removal. Even the still-open instance
    /// — which never runs the open sweep — must not see the take anywhere:
    /// the trash tree is outside every scan.
    #[test]
    fn crash_after_trash_rename_leaves_no_session_or_orphan() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");
        let created = store.create(sample_wav(), Some(25.0)).expect("create");

        // The crash midpoint: the directory is in the trash, the removal
        // never ran.
        let trash = temp.path().join(DELETED_DIR);
        std::fs::create_dir_all(&trash).expect("trash dir");
        std::fs::rename(temp.path().join(&created.id), trash.join(&created.id))
            .expect("rename to trash");

        assert!(store.get(&created.id).expect("get").is_none());
        assert!(
            store.list_records().expect("list").is_empty(),
            "a session sitting in the trash must not be listed"
        );
        assert!(
            store.journal_ids().expect("journal ids").is_empty(),
            "the trash tree holds no linkage either"
        );

        // The next startup sweeps the leftover bytes entirely.
        let reopened = FileSessionStore::open(temp.path()).expect("reopen");
        assert!(reopened.get(&created.id).expect("get").is_none());
        assert!(reopened.list_records().expect("list").is_empty());
        assert!(!trash.exists(), "the open sweep reaped the trash");
    }

    /// #206, crash point (b) — the resurrection scenario itself: a plain
    /// delete (no journal, no tombstone) interrupted mid-removal must not
    /// leave a WAV-only directory for the next startup to classify as a
    /// recoverable orphan. The control half first proves that shape really
    /// does come back as `interrupted` today.
    #[test]
    fn crash_mid_delete_of_unlinked_session_does_not_resurrect() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");
        let created = store.create(sample_wav(), Some(25.0)).expect("create");

        // Control: the old failure shape — manifest unlinked, WAV left —
        // is exactly what a half-done `remove_dir_all` produced, and
        // classification recovers it.
        std::fs::remove_file(temp.path().join(&created.id).join(MANIFEST_FILE))
            .expect("unlink manifest");
        let records = store.list_records().expect("list");
        let summary = match records.as_slice() {
            [ListedRecord::Session(summary)] => summary,
            other => panic!("expected one recovered orphan session, got {other:?}"),
        };
        assert_eq!(summary.id, created.id);
        assert_eq!(summary.status, SessionStatus::Interrupted);
        assert!(
            summary
                .last_error
                .as_deref()
                .is_some_and(|note| note.contains("Recovered")),
            "the orphan carries the recovery note: {:?}",
            summary.last_error
        );

        // The new delete's crash midpoint instead: renamed into the trash,
        // removal not yet done. The take must stay deleted.
        let trash = temp.path().join(DELETED_DIR);
        std::fs::create_dir_all(&trash).expect("trash dir");
        std::fs::rename(temp.path().join(&created.id), trash.join(&created.id))
            .expect("rename to trash");

        let reopened = FileSessionStore::open(temp.path()).expect("reopen");
        assert!(reopened.get(&created.id).expect("get").is_none());
        assert!(
            reopened.list_records().expect("list").is_empty(),
            "the deleted take must not come back as a recovered orphan"
        );
    }

    /// #206: the trash tree's name cannot be assigned as a session id —
    /// the deletion commit point must never collide with a take's
    /// directory.
    #[test]
    fn trash_directory_name_is_reserved() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        assert!(matches!(
            store.create_with_journal(sample_wav(), Some(1.0), Some(DELETED_DIR)),
            Err(StorageError::Invalid(_))
        ));
        assert!(matches!(
            store.save_journal_recovery(DELETED_DIR, sample_wav(), Some(1.0), "note"),
            Err(StorageError::Invalid(_))
        ));
        assert!(matches!(
            store.get(DELETED_DIR),
            Err(StorageError::Invalid(_))
        ));
        assert!(matches!(
            store.delete(DELETED_DIR),
            Err(StorageError::Invalid(_))
        ));
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
    /// boundary" — same idea for a corrupted manifest.json. The audio is
    /// removed alongside, so the record cannot fall back to orphan recovery
    /// and must surface as damaged (the intact-audio variants live in the
    /// G02 mixed-store suite below).
    #[test]
    fn file_store_rejects_corrupted_manifests() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");
        let session = store.create(sample_wav(), None).expect("create");
        let session_dir = temp.path().join(&session.id);
        let manifest_path = session_dir.join(MANIFEST_FILE);

        // Mirror the TS test: attemptCount holding a string instead of a number.
        let wrong_type = format!(
            r#"{{"schemaVersion":1,"id":"{id}","createdAt":"2026-01-01T00:00:00.000Z","updatedAt":"2026-01-01T00:00:00.000Z","status":"captured","audioFile":"recording.wav","attemptCount":"not-a-number"}}"#,
            id = session.id
        );
        std::fs::write(&manifest_path, wrong_type).expect("write corrupted manifest");
        std::fs::remove_file(session_dir.join(AUDIO_FILE)).expect("remove wav");
        match store.get(&session.id) {
            Err(StorageError::Invalid(reason)) => {
                assert!(reason.contains("manifest.json"), "{reason}")
            }
            other => panic!("expected Invalid for wrong-typed attemptCount, got {other:?}"),
        }
        // G02: one corrupt record no longer fails the listing — it is the
        // only record, flagged with the reason.
        let listed = store.list_records().expect("listing survives damage");
        assert_eq!(listed.len(), 1);
        match &listed[0] {
            ListedRecord::Damaged(damaged) => {
                assert_eq!(damaged.id, session.id);
                assert!(damaged.reason.contains("manifest.json"), "{}", damaged.reason);
            }
            other => panic!("expected a damaged record, got {other:?}"),
        }

        // Truncated JSON.
        std::fs::write(&manifest_path, "{\"schemaVersion\":1,").expect("write truncated manifest");
        match store.get(&session.id) {
            Err(StorageError::Invalid(_)) => {}
            other => panic!("expected Invalid for truncated manifest, got {other:?}"),
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
        // abort the scan (the record itself is flagged by `list_records`).
        std::fs::create_dir_all(temp.path().join("broken")).expect("broken dir");
        std::fs::write(temp.path().join("broken").join(MANIFEST_FILE), b"{nope")
            .expect("corrupt manifest");
        assert!(store.journal_ids().is_ok());
    }

    // -----------------------------------------------------------------
    // G02: per-record isolation and WAV-only orphan recovery.
    // -----------------------------------------------------------------

    /// Canonical 44-byte-header WAV: PCM16 mono 16 kHz silence (the shape
    /// `crate::audio::encode_wav_16k` writes), built by hand so the storage
    /// tests do not depend on the audio module.
    fn wav_bytes(duration_seconds: f64) -> Vec<u8> {
        const SAMPLE_RATE: u32 = 16_000;
        let data_len = (SAMPLE_RATE as f64 * duration_seconds).round() as u32 * 2;

        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.resize(44 + data_len as usize, 0);
        wav
    }

    fn session_summary(record: &ListedRecord) -> &SessionSummary {
        match record {
            ListedRecord::Session(summary) => summary,
            ListedRecord::Damaged(_) => panic!("expected a readable session, got {record:?}"),
        }
    }

    fn damaged(record: &ListedRecord) -> &DamagedRecord {
        match record {
            ListedRecord::Damaged(damaged) => damaged,
            ListedRecord::Session(_) => panic!("expected a damaged record, got {record:?}"),
        }
    }

    /// The G02 fixture: a store holding one good record plus every damaged
    /// shape and both orphan shapes at once. Returns the ids in a fixed
    /// order for the assertions.
    struct MixedStore {
        temp: TempDir,
        good: String,
        missing_wav: String,
        corrupt_manifest: String,
        corrupt_manifest_intact_wav: String,
        orphan: String,
        future_schema: String,
        id_mismatch: String,
        garbage_wav: String,
    }

    fn mixed_store() -> MixedStore {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        let good = store
            .create(wav_bytes(0.3), Some(300.0))
            .expect("create good");

        // Manifest intact, audio gone.
        let missing_wav = store.create(wav_bytes(0.2), None).expect("create missing-wav");
        std::fs::remove_file(temp.path().join(&missing_wav.id).join(AUDIO_FILE))
            .expect("remove wav");

        // Manifest corrupt and nothing recoverable left.
        let corrupt_manifest = store
            .create(wav_bytes(0.2), None)
            .expect("create corrupt-manifest");
        std::fs::write(temp.path().join(&corrupt_manifest.id).join(MANIFEST_FILE), b"{nope")
            .expect("corrupt manifest");
        std::fs::remove_file(temp.path().join(&corrupt_manifest.id).join(AUDIO_FILE))
            .expect("remove wav too");

        // Manifest corrupt but the audio intact: recoverable as an orphan.
        let corrupt_manifest_intact_wav = store
            .create(wav_bytes(0.4), None)
            .expect("create corrupt-manifest-intact-wav");
        std::fs::write(
            temp.path()
                .join(&corrupt_manifest_intact_wav.id)
                .join(MANIFEST_FILE),
            "{\"schemaVersion\":1,",
        )
        .expect("truncate manifest");

        // WAV-only orphan: the crash-before-manifest-write shape.
        let orphan_dir = temp.path().join("orphan-take");
        std::fs::create_dir_all(&orphan_dir).expect("orphan dir");
        let orphan_wav = wav_bytes(0.75);
        std::fs::write(orphan_dir.join(AUDIO_FILE), &orphan_wav).expect("orphan wav");

        // A complete, well-formed manifest from a future schema version.
        let future_dir = temp.path().join("future-take");
        std::fs::create_dir_all(&future_dir).expect("future dir");
        std::fs::write(
            future_dir.join(MANIFEST_FILE),
            r#"{"schemaVersion":999,"id":"future-take","createdAt":"2026-01-01T00:00:00.000Z","updatedAt":"2026-01-01T00:00:00.000Z","status":"captured","audioFile":"recording.wav","attemptCount":0,"transcriptHistory":[]}"#,
        )
        .expect("future manifest");
        std::fs::write(future_dir.join(AUDIO_FILE), wav_bytes(0.1)).expect("future wav");

        // Identity mismatch: the manifest describes a different id.
        let mismatch_dir = temp.path().join("moved-take");
        std::fs::create_dir_all(&mismatch_dir).expect("mismatch dir");
        std::fs::write(
            mismatch_dir.join(MANIFEST_FILE),
            r#"{"schemaVersion":1,"id":"someone-else","createdAt":"2026-01-01T00:00:00.000Z","updatedAt":"2026-01-01T00:00:00.000Z","status":"captured","audioFile":"recording.wav","attemptCount":0,"transcriptHistory":[]}"#,
        )
        .expect("mismatch manifest");
        std::fs::write(mismatch_dir.join(AUDIO_FILE), wav_bytes(0.1)).expect("mismatch wav");

        // WAV-only directory whose audio is not a WAV at all.
        let garbage_dir = temp.path().join("garbage-wav");
        std::fs::create_dir_all(&garbage_dir).expect("garbage dir");
        std::fs::write(garbage_dir.join(AUDIO_FILE), b"definitely not audio")
            .expect("garbage wav");

        MixedStore {
            temp,
            good: good.id,
            missing_wav: missing_wav.id,
            corrupt_manifest: corrupt_manifest.id,
            corrupt_manifest_intact_wav: corrupt_manifest_intact_wav.id,
            orphan: "orphan-take".to_string(),
            future_schema: "future-take".to_string(),
            id_mismatch: "moved-take".to_string(),
            garbage_wav: "garbage-wav".to_string(),
        }
    }

    #[test]
    fn mixed_store_listing_isolates_damage_and_recovers_orphans() {
        let fixture = mixed_store();
        let store = FileSessionStore::open(fixture.temp.path()).expect("open");

        let listed = store.list_records().expect("listing never aborts on damage");

        let find = |id: &str| {
            listed
                .iter()
                .find(|record| record_id(record) == id)
                .unwrap_or_else(|| panic!("record {id} missing from listing"))
        };

        // Good records stay visible.
        let good = session_summary(find(&fixture.good));
        assert_eq!(good.status, SessionStatus::Captured);
        assert_eq!(good.duration_ms, Some(300.0));

        // The WAV-only orphan becomes a recoverable interrupted take with
        // the duration computed from the verified bytes.
        let orphan = session_summary(find(&fixture.orphan));
        assert_eq!(orphan.status, SessionStatus::Interrupted);
        assert_eq!(orphan.duration_ms, Some(750.0));
        assert_eq!(orphan.attempt_count, 0);
        assert!(orphan.transcript.is_none());
        assert_eq!(orphan.created_at.len(), 24, "RFC3339 millis stamp");
        let note = orphan.last_error.as_deref().expect("recovery note");
        assert!(note.contains("Recovered"), "{note}");
        assert!(note.contains("kept untouched"), "{note}");

        // A corrupt manifest with the audio intact recovers the same way —
        // the recording outranks the file that failed to describe it.
        let intact = session_summary(find(&fixture.corrupt_manifest_intact_wav));
        assert_eq!(intact.status, SessionStatus::Interrupted);
        assert_eq!(intact.duration_ms, Some(400.0));

        // Damaged records are flagged with their reason, not hidden.
        assert_eq!(damaged(find(&fixture.missing_wav)).id, fixture.missing_wav);
        let reason = &damaged(find(&fixture.missing_wav)).reason;
        assert!(reason.contains("recording.wav"), "{reason}");
        let reason = &damaged(find(&fixture.corrupt_manifest)).reason;
        assert!(reason.contains("manifest.json"), "{reason}");
        let reason = &damaged(find(&fixture.future_schema)).reason;
        assert!(
            reason.contains("unsupported manifest schemaVersion 999"),
            "{reason}"
        );
        let reason = &damaged(find(&fixture.id_mismatch)).reason;
        assert!(reason.contains("does not match"), "{reason}");
        let reason = &damaged(find(&fixture.garbage_wav)).reason;
        assert!(reason.contains("recording.wav"), "{reason}");

        // Every record is present: 3 readable sessions (good, orphan,
        // corrupt-manifest-with-intact-audio) + 5 damaged ones.
        assert_eq!(listed.len(), 8);

        // Damaged records sort after every readable session.
        let first_damaged = listed
            .iter()
            .position(|record| matches!(record, ListedRecord::Damaged(_)))
            .expect("damaged records present");
        assert!(
            listed[..first_damaged]
                .iter()
                .all(|record| matches!(record, ListedRecord::Session(_))),
            "readable sessions sort before damaged records"
        );

        // Nothing was deleted or moved: every directory still exists, and
        // the orphan's audio is byte-for-byte untouched.
        for id in [
            &fixture.good,
            &fixture.missing_wav,
            &fixture.corrupt_manifest,
            &fixture.corrupt_manifest_intact_wav,
            &fixture.orphan,
            &fixture.future_schema,
            &fixture.id_mismatch,
            &fixture.garbage_wav,
        ] {
            assert!(
                fixture.temp.path().join(id).is_dir(),
                "directory {id} must survive the listing"
            );
        }
        assert_eq!(
            std::fs::read(fixture.temp.path().join(&fixture.orphan).join(AUDIO_FILE))
                .expect("orphan wav"),
            wav_bytes(0.75)
        );

        // The orphan loads in full on demand — audio and all.
        let loaded = store
            .get(&fixture.orphan)
            .expect("get orphan")
            .expect("orphan is a session");
        assert_eq!(loaded.status, SessionStatus::Interrupted);
        assert_eq!(loaded.wav.as_slice(), wav_bytes(0.75).as_slice());
    }

    #[test]
    fn damaged_reads_surface_the_recorded_reason() {
        let fixture = mixed_store();
        let store = FileSessionStore::open(fixture.temp.path()).expect("open");

        for (id, expected) in [
            (&fixture.missing_wav, "recording.wav"),
            (&fixture.future_schema, "unsupported manifest schemaVersion 999"),
            (&fixture.id_mismatch, "does not match"),
            (&fixture.garbage_wav, "recording.wav"),
        ] {
            match store.get(id) {
                Err(StorageError::Invalid(reason)) => {
                    assert!(reason.contains(expected), "{id}: {reason}")
                }
                other => panic!("{id}: expected Invalid, got {other:?}"),
            }
            // The transcribe path (`mark_attempt`) surfaces the same reason.
            match store.mark_attempt(id) {
                Err(StorageError::Invalid(reason)) => {
                    assert!(reason.contains(expected), "{id}: {reason}")
                }
                other => panic!("{id}: expected Invalid from mark_attempt, got {other:?}"),
            }
        }
    }

    #[test]
    fn orphan_first_write_heals_the_record() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        let orphan_dir = temp.path().join("crash-before-manifest");
        std::fs::create_dir_all(&orphan_dir).expect("orphan dir");
        let wav = wav_bytes(0.5);
        std::fs::write(orphan_dir.join(AUDIO_FILE), &wav).expect("orphan wav");

        // Reads are side-effect free: the manifest appears only once a
        // write goes through, and the audio never moves.
        assert!(
            !orphan_dir.join(MANIFEST_FILE).exists(),
            "reading an orphan must not write anything"
        );
        let recovered = store
            .get("crash-before-manifest")
            .expect("get")
            .expect("orphan session");
        assert_eq!(recovered.status, SessionStatus::Interrupted);
        assert!(
            !orphan_dir.join(MANIFEST_FILE).exists(),
            "get must stay read-only"
        );

        // The first write persists a manifest: the take is healed into a
        // first-class session from here on.
        let attempted = store
            .mark_attempt("crash-before-manifest")
            .expect("retry on the recovered take");
        assert_eq!(attempted.status, SessionStatus::Transcribing);
        assert_eq!(attempted.attempt_count, 1);

        let manifest_path = orphan_dir.join(MANIFEST_FILE);
        let raw = std::fs::read_to_string(&manifest_path).expect("manifest was written");
        let manifest: serde_json::Value = serde_json::from_str(&raw).expect("parse manifest");
        assert_eq!(manifest["id"], "crash-before-manifest");
        assert_eq!(manifest["status"], "transcribing");
        assert_eq!(manifest["audioFile"], "recording.wav");

        // It now lists as a regular record, and the audio is untouched.
        let listed = store.list_records().expect("list");
        let healed = session_summary(&listed[0]);
        assert_eq!(healed.id, "crash-before-manifest");
        assert_eq!(healed.status, SessionStatus::Transcribing);
        assert_eq!(
            std::fs::read(orphan_dir.join(AUDIO_FILE)).expect("wav"),
            wav
        );
    }

    #[test]
    fn orphan_duration_is_clamped_to_the_bytes_actually_present() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        // A torn write: the data header promises 4 seconds (128_000 bytes)
        // but the file holds only 1 second of samples (32_000 bytes).
        let mut wav = wav_bytes(1.0);
        wav.truncate(44 + 32_000);
        wav[40..44].copy_from_slice(&128_000u32.to_le_bytes());
        let torn_dir = temp.path().join("torn-take");
        std::fs::create_dir_all(&torn_dir).expect("torn dir");
        std::fs::write(torn_dir.join(AUDIO_FILE), &wav).expect("torn wav");

        let listed = store.list_records().expect("list");
        let recovered = session_summary(&listed[0]);
        assert_eq!(recovered.duration_ms, Some(1000.0), "verified bytes only");
    }

    #[test]
    fn list_page_slices_the_full_ordering() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        let first = store.create(wav_bytes(0.1), None).expect("first");
        std::thread::sleep(Duration::from_millis(15));
        let second = store.create(wav_bytes(0.1), None).expect("second");
        std::thread::sleep(Duration::from_millis(15));
        let third = store.create(wav_bytes(0.1), None).expect("third");

        let page = store.list_page(0, 2).expect("first page");
        assert_eq!(page.total, 3);
        assert_eq!(page.offset, 0);
        let ids = page
            .records
            .iter()
            .map(record_id)
            .collect::<Vec<_>>();
        assert_eq!(ids, [third.id.as_str(), second.id.as_str()]);

        let page = store.list_page(2, 2).expect("tail page");
        assert_eq!(page.total, 3);
        assert_eq!(
            page.records.iter().map(record_id).collect::<Vec<_>>(),
            [first.id.as_str()]
        );

        // Off the end: empty, not an error, with the total still true.
        let page = store.list_page(9, 2).expect("past the end");
        assert_eq!(page.total, 3);
        assert!(page.records.is_empty());
    }

    // -----------------------------------------------------------------
    // #205 / #215: durable create ordering, journal-named sessions,
    // damaged-audio repair, orphan healing.
    // -----------------------------------------------------------------

    /// #215: a journal-linked take's session id is its journal id, so the
    /// crash window between the WAV write and the manifest publish leaves
    /// an orphan directory whose name identifies its journal — and an
    /// unsafe journal id is rejected rather than becoming a path.
    #[test]
    fn journal_linked_takes_are_named_after_their_journal() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        let linked = store
            .create_with_journal(sample_wav(), Some(10.0), Some("j_named"))
            .expect("create linked");
        assert_eq!(linked.id, "j_named");
        assert_eq!(linked.journal_id.as_deref(), Some("j_named"));
        assert!(temp.path().join("j_named").join(AUDIO_FILE).is_file());
        assert!(
            temp.path()
                .join("j_named")
                .join(MANIFEST_FILE)
                .is_file(),
            "the manifest is published after the audio"
        );

        // Unlinked takes keep minting fresh uuids.
        let plain = store.create(sample_wav(), None).expect("create plain");
        assert_ne!(plain.id, "j_named");
        assert!(plain.id.len() > "j_named".len());

        // The journal id becomes a path component: unsafe ones are refused.
        for bad in ["", "a/b", "a\\b", ".", ".."] {
            match store.create_with_journal(sample_wav(), None, Some(bad)) {
                Err(StorageError::Invalid(message)) => {
                    assert!(message.contains("journal id"), "{message}")
                }
                other => panic!("expected Invalid for journal id {bad:?}, got {other:?}"),
            }
        }
    }

    /// #205: the metadata-only scan that finds the power-loss shape — the
    /// manifest parses and links a journal, but the WAV fails verification
    /// — and nothing else.
    #[test]
    fn damaged_audio_journal_links_flag_only_the_power_loss_shape() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        // Healthy linked session: not flagged.
        store
            .create_with_journal(wav_bytes(0.1), None, Some("j_healthy"))
            .expect("healthy");

        // The #205 shape: manifest intact and linked, WAV gone.
        let damaged = store
            .create_with_journal(wav_bytes(0.2), None, Some("j_lost"))
            .expect("create");
        std::fs::remove_file(temp.path().join(&damaged.id).join(AUDIO_FILE))
            .expect("remove wav");

        // A torn WAV (magic destroyed) is the same shape.
        let torn = store
            .create_with_journal(wav_bytes(0.2), None, Some("j_torn"))
            .expect("create torn");
        std::fs::write(
            temp.path().join(&torn.id).join(AUDIO_FILE),
            b"RIFF but not really",
        )
        .expect("tear the wav");

        // A damaged WAV with no journal linkage: damaged, but nothing for
        // journal repair to consult.
        let unlinked = store.create(wav_bytes(0.2), None).expect("create unlinked");
        std::fs::remove_file(temp.path().join(&unlinked.id).join(AUDIO_FILE))
            .expect("remove unlinked wav");

        // An unparseable manifest cannot be linked at all.
        let corrupt = store
            .create_with_journal(wav_bytes(0.2), None, Some("j_corrupt"))
            .expect("create corrupt");
        std::fs::write(temp.path().join(&corrupt.id).join(MANIFEST_FILE), b"{nope")
            .expect("corrupt manifest");
        std::fs::remove_file(temp.path().join(&corrupt.id).join(AUDIO_FILE))
            .expect("remove wav too");

        let links = store.damaged_audio_journal_links().expect("links");
        assert_eq!(
            links,
            HashMap::from([
                ("j_lost".to_string(), vec![damaged.id.clone()]),
                ("j_torn".to_string(), vec![torn.id.clone()]),
            ]),
            "only manifest-intact, linked, wav-damaged records are flagged"
        );
    }

    /// #205: the repair primitive — one durable write of journal-verified
    /// bytes over the damaged WAV, manifest untouched.
    #[test]
    fn repair_session_audio_rebuilds_the_wav_in_place() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        let session = store
            .create_with_journal(wav_bytes(0.3), None, Some("j_repair"))
            .expect("create");
        store
            .save_transcript(
                &session.id,
                transcript("survives the power loss with the manifest"),
            )
            .expect("save transcript");

        // The power loss takes the WAV's data blocks.
        std::fs::write(
            temp.path().join(&session.id).join(AUDIO_FILE),
            b"zeroed by the disk",
        )
        .expect("destroy wav");
        match store.get(&session.id) {
            Err(StorageError::Invalid(reason)) => {
                assert!(reason.contains(AUDIO_FILE), "{reason}")
            }
            other => panic!("expected Invalid for the torn wav, got {other:?}"),
        }

        let rebuilt = wav_bytes(0.3);
        store
            .repair_session_audio(&session.id, &rebuilt, None, None)
            .expect("repair");

        // The record reads back whole: audio restored, manifest (and its
        // transcript) untouched by the byte-equivalent repair.
        let restored = store.get(&session.id).expect("get").expect("present");
        assert_eq!(restored.wav.as_slice(), rebuilt.as_slice());
        assert_eq!(restored.status, SessionStatus::Transcribed);
        assert_eq!(
            restored.transcript.as_ref().expect("transcript").text,
            "survives the power loss with the manifest"
        );
        assert_eq!(restored.journal_id.as_deref(), Some("j_repair"));

        // A torn repair (note present) folds the manifest update into the
        // same crash-safe sequence: interrupted status, gap note, and the
        // rebuilt audio's own duration (#247 review).
        let shorter = wav_bytes(0.1);
        store
            .repair_session_audio(
                &session.id,
                &shorter,
                Some(100.0),
                Some("rebuilt to the verified prefix"),
            )
            .expect("torn repair");
        let torn = store.get(&session.id).expect("get").expect("present");
        assert_eq!(torn.wav.as_slice(), shorter.as_slice());
        assert_eq!(torn.status, SessionStatus::Interrupted);
        assert_eq!(torn.duration_ms, Some(100.0));
        assert_eq!(
            torn.last_error.as_deref(),
            Some("rebuilt to the verified prefix")
        );

        // Unknown sessions are NotFound; unsafe ids are Invalid.
        match store.repair_session_audio("missing", &rebuilt, None, None) {
            Err(StorageError::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
        match store.repair_session_audio("a/b", &rebuilt, None, None) {
            Err(StorageError::Invalid(_)) => {}
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    /// #247 review round 2: the journal-named create refuses an occupied
    /// name (it must be fresh by contract), and the recovery heal refuses
    /// a Damaged directory that still holds manifest data — that is an
    /// unrelated record, not the crash window.
    #[test]
    fn journal_named_writes_refuse_occupied_or_evidenced_directories() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        // create_with_journal: a journal id naming an existing session is a
        // collision, never a clobber.
        let wav = wav_bytes(0.25);
        store.create(wav.clone(), Some(250.0)).expect("uuid session");
        let uuid_session = std::fs::read_dir(temp.path())
            .expect("list root")
            .filter_map(|entry| entry.ok())
            .find(|entry| entry.path().is_dir())
            .expect("the uuid session directory")
            .file_name()
            .to_string_lossy()
            .to_string();
        match store.create_with_journal(wav.clone(), Some(250.0), Some(&uuid_session)) {
            Err(StorageError::Invalid(reason)) => {
                assert!(reason.contains("already exists"), "{reason}")
            }
            other => panic!("expected Invalid, got {other:?}"),
        }

        // save_journal_recovery: a Damaged directory that still holds a
        // manifest file is not the crash window — refused, untouched.
        let damaged_dir = temp.path().join("j_collide");
        std::fs::create_dir_all(&damaged_dir).expect("dir");
        std::fs::write(damaged_dir.join(MANIFEST_FILE), b"{not json").expect("manifest");
        std::fs::write(damaged_dir.join(AUDIO_FILE), b"torn").expect("wav");
        match store.save_journal_recovery("j_collide", wav, Some(250.0), "note") {
            Ok(RecoverySave::Occupied(reason)) => {
                assert!(reason.contains("damaged data"), "{reason}")
            }
            other => panic!("expected Occupied, got {other:?}"),
        }
        assert_eq!(
            std::fs::read(damaged_dir.join(MANIFEST_FILE)).expect("manifest intact"),
            b"{not json".as_slice(),
            "the damaged record's evidence is untouched"
        );

        // The true crash window — WAV torn, manifest never attempted —
        // still heals.
        let window_dir = temp.path().join("j_window");
        std::fs::create_dir_all(&window_dir).expect("dir");
        std::fs::write(window_dir.join(AUDIO_FILE), b"torn before any fsync").expect("wav");
        match store
            .save_journal_recovery("j_window", wav_bytes(0.25), Some(250.0), "healed")
            .expect("heal")
        {
            RecoverySave::Healed(session) => assert_eq!(session.id, "j_window"),
            other => panic!("expected Healed, got {other:?}"),
        }
    }

    /// #215: the persistence seam recovery drives — create when nothing
    /// is there, heal a WAV-only orphan of the journal's name in place,
    /// refuse an occupied name rather than duplicate the take.
    #[test]
    fn save_journal_recovery_creates_heals_and_refuses_occupied_names() {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path()).expect("open");

        // Nothing there: a fresh interrupted session linked to its journal.
        let wav = wav_bytes(0.25);
        match store
            .save_journal_recovery("j_fresh", wav.clone(), Some(250.0), "recovered note")
            .expect("create")
        {
            RecoverySave::Created(session) => {
                assert_eq!(session.id, "j_fresh");
                assert_eq!(session.status, SessionStatus::Interrupted);
                assert_eq!(session.last_error.as_deref(), Some("recovered note"));
                assert_eq!(session.journal_id.as_deref(), Some("j_fresh"));
            }
            other => panic!("expected Created, got {other:?}"),
        }
        assert_eq!(
            std::fs::read(temp.path().join("j_fresh").join(AUDIO_FILE)).expect("wav"),
            wav
        );

        // The crash window: WAV on disk, manifest never published. Healing
        // replaces the audio with the journal-verified bytes and publishes
        // the manifest with the linkage.
        let torn_orphan_wav = wav_bytes(0.1);
        std::fs::create_dir_all(temp.path().join("j_crash")).expect("orphan dir");
        std::fs::write(
            temp.path().join("j_crash").join(AUDIO_FILE),
            &torn_orphan_wav,
        )
        .expect("orphan wav");
        let full_wav = wav_bytes(0.5);
        match store
            .save_journal_recovery("j_crash", full_wav.clone(), Some(500.0), "healed note")
            .expect("heal")
        {
            RecoverySave::Healed(session) => {
                assert_eq!(session.id, "j_crash");
                assert_eq!(session.status, SessionStatus::Interrupted);
                assert_eq!(session.journal_id.as_deref(), Some("j_crash"));
                assert!(session.created_at.len() == 24, "orphan mtime stamp kept");
            }
            other => panic!("expected Healed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read(temp.path().join("j_crash").join(AUDIO_FILE)).expect("wav"),
            full_wav,
            "the torn orphan audio is rebuilt from the journal bytes"
        );
        let healed = store.get("j_crash").expect("get").expect("present");
        assert_eq!(healed.duration_ms, Some(500.0));

        // An occupied name never yields a second session.
        store
            .create_with_journal(wav_bytes(0.1), None, Some("j_taken"))
            .expect("create the occupant");
        match store
            .save_journal_recovery("j_taken", wav_bytes(0.1), None, "note")
            .expect("no error, just a refusal")
        {
            RecoverySave::Occupied(reason) => {
                assert!(reason.contains("already exists"), "{reason}")
            }
            other => panic!("expected Occupied, got {other:?}"),
        }
        // The occupant is exactly as it was.
        let occupant = store.get("j_taken").expect("get").expect("present");
        assert_eq!(occupant.status, SessionStatus::Captured);
        assert!(occupant.last_error.is_none());

        // The crash window with the WAV torn past verification (the power
        // loss struck before the audio fsync completed, so no manifest was
        // ever attempted): the directory classifies as Damaged, and the
        // journal's verified bytes still heal it — one session, not a
        // damaged record plus a recovered duplicate.
        std::fs::create_dir_all(temp.path().join("j_torndir")).expect("dir");
        std::fs::write(
            temp.path().join("j_torndir").join(AUDIO_FILE),
            b"not even a header",
        )
        .expect("torn wav");
        let rebuilt = wav_bytes(0.4);
        match store
            .save_journal_recovery("j_torndir", rebuilt.clone(), Some(400.0), "healed note")
            .expect("heal the damaged directory")
        {
            RecoverySave::Healed(session) => {
                assert_eq!(session.id, "j_torndir");
                assert_eq!(session.status, SessionStatus::Interrupted);
                assert_eq!(session.journal_id.as_deref(), Some("j_torndir"));
            }
            other => panic!("expected Healed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read(temp.path().join("j_torndir").join(AUDIO_FILE)).expect("wav"),
            rebuilt,
            "the unparseable audio is rebuilt from the journal bytes"
        );
        let healed = store.get("j_torndir").expect("get").expect("present");
        assert_eq!(healed.status, SessionStatus::Interrupted);

        // Unsafe ids are rejected before touching the filesystem.
        match store.save_journal_recovery("a/b", wav_bytes(0.1), None, "note") {
            Err(StorageError::Invalid(message)) => {
                assert!(message.contains("journal id"), "{message}")
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }
}
