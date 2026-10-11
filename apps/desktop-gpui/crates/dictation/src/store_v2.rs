//! Storage v2 core (I2, `docs/program/design/e17-native-runtime.md` §4).
//!
//! This is THE store (D14: no backwards compatibility of any kind — the
//! v1 file store and its opt-in flag are deleted; the runtime state
//! machine persists through this store as well). The data root —
//! `<data-root>/starling-gpui/` — holds the v2 layout:
//!
//! ```text
//! <root>/starling.db   SQLite (WAL) transactional metadata
//! <root>/audio/        <captureId>.sj finalized sample journals, or
//!                      <captureId>.flac once compressed (#342)
//! <root>/staging/      <captureId>.sj in-flight journals (§4 step 1)
//! <root>/quarantine/   deliberately-deleted journals (R21 tombstones)
//! <root>/owner.lock    the single-owner lock ([`DataRootLock`])
//! ```
//!
//! # Ownership
//!
//! One process owns a data root: it holds [`DataRootLock`] for its
//! lifetime, and only it opens stores on the root (several handles in
//! that process are fine). Recovery ([`StoreV2::reconcile`]) runs before
//! that process records anything, so every journal it finds is a dead
//! run's, and a `started` recognition attempt no handle in the process
//! holds was left by one.
//!
//! # Schema
//!
//! The §4 "schema direction" tables ([`SCHEMA_SQL`], one place, version
//! [`SCHEMA_VERSION`] in `meta`): `captures`, `recognition_attempts`,
//! `context_snapshots`, `mode_decisions`, `documents`/`revisions`,
//! `deliveries`, `insight_events`, `correction_records`, `tombstones`,
//! `journal_supersessions`, `transcription_intents`, `meta`. This core implements the
//! captures/attempts/tombstones/meta surfaces plus the
//! documents/revisions surface (I5, issue #220:
//! [`StoreV2::upsert_document`] and friends — the documents machine's
//! persistence); the context/deliveries tables still await their owning
//! increments (E03's adapters), extending the schema additively.
//! `extra_json` per row preserves unknown/newer fields **verbatim**: it is stored and returned as the raw
//! text a writer produced, never re-serialized from parsed form on paths
//! that do not touch it, and updates that add fields merge into the parsed
//! object without dropping keys.
//!
//! Forward compatibility (§4): a database whose `meta.schema_version` is
//! **higher** than [`SCHEMA_VERSION`] is refused at open
//! ([`StoreV2Error::SchemaTooNew`]) — this build will neither read nor
//! write a future format; a lower version is upgraded by applying
//! [`SCHEMA_SQL`] (idempotent `CREATE TABLE IF NOT EXISTS`).
//!
//! # Crash protocol (§4, per take)
//!
//! 1. [`StoreV2::begin_take`] creates `staging/<id>.sj` (header fsynced);
//!    frames + boundary records are appended and fsynced on the §3 cadence
//!    by the [`crate::journal`] machinery — acknowledged = boundary fsynced.
//! 2. [`V2Take::finalize`] writes the trailer (length + content hash),
//!    fsyncs the file and the staging directory; [`StoreV2::promote`]
//!    renames into `audio/` and fsyncs both directories.
//! 3. [`StoreV2::commit_capture`] inserts the `captures` row in one SQLite
//!    transaction (`synchronous=FULL`, so the commit is on disk) and WAL-
//!    checkpoints per policy ([`WAL_CHECKPOINT_EVERY_COMMITS`], D11
//!    tunable). **The durable ack to the capture side is the `Ok` return of
//!    the combined [`V2Take::finish`] — only after the commit.**
//! 4. [`StoreV2::gc_staging`] removes staging leftovers whose id already has
//!    a committed row (the rename in step 2 already took the file itself).
//!
//! The step boundaries are individually callable so fault-injection tests
//! can kill between any two steps and assert the documented recovery state.
//!
//! # Recovery (§4)
//!
//! [`StoreV2::reconcile`] reconciles both sides independently:
//! - journal tail past the last valid boundary → physically truncated,
//!   sealed with a checksum-valid trailer, promoted, and the row created
//!   with status `interrupted` **and a gap note** (the discarded tail is
//!   never silently joined — E02);
//! - finalized audio with no row → orphan session: a row is created with
//!   status `interrupted`, linked to the audio (`interrupted`, not `gone`);
//! - row without finalized audio → the row is marked `interrupted`;
//! - tombstoned ids (a `tombstones` row or a file under `quarantine/`,
//!   R21 semantics) stay dead — recovery never resurrects a confirmed
//!   deletion, and an interrupted delete is completed, not half-kept.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};

use crate::audio::{decode_pcm16_wav, pcm16, pcm16_to_f32, request_pcm16, STARLING_SAMPLE_RATE};
use crate::flac;
use crate::journal::{
    self, JournalWriter, read_journal, samples_hash, seal_recovered_journal, sync_dir,
};
use crate::storage::{is_safe_path_component, iso_utc, now_iso};

/// Schema version of `starling.db` this build writes and understands.
/// Bump only with an additive migration path; a DB holding a higher value
/// is refused at open. v2 added `recognition_attempts.created_utc` (the
/// real updated-at source for the summaries); v3 added `insight_events`
/// (#294: per-job processing latency, recorded for Insights #308); v4
/// added `correction_records` and `captures.secure_field`; v5 added
/// `journal_supersessions` (#356); v6 added `transcription_intents` and
/// `audio_holds` (#220); v7 added `pending_intents` and `recovery_seals`
/// (#220: a take's transcription intent survives a crash between its
/// audio's promotion and its metadata commit, and a journal recovery
/// sealed never reads as complete).
pub const SCHEMA_VERSION: u32 = 7;

/// WAL checkpoint policy (D11): run `PRAGMA wal_checkpoint(PASSIVE)` after
/// every N metadata commits. Default 64 — frequent enough that the WAL
/// cannot grow without bound across a session of takes, rare enough that
/// the checkpoint cost stays off the per-take path on slow disks. Tunable
/// via [`StoreV2::set_checkpoint_every_commits`].
pub const WAL_CHECKPOINT_EVERY_COMMITS: u32 = 64;

/// Upper bound on one [`StoreV2::list_records`] page (bounded history
/// reads): callers ask for any `limit`; the store clamps it here, so a
/// listing can never make SQLite materialize the whole history in one
/// query regardless of what a caller passes. The app facade pages at
/// 200; this is the lower-layer backstop.
pub const LIST_PAGE_MAX: usize = 500;

/// Directory names under the v2 root.
const AUDIO_DIR: &str = "audio";
const STAGING_DIR: &str = "staging";
const QUARANTINE_DIR: &str = "quarantine";
/// [`DataRootLock`]'s file under the root.
const OWNER_LOCK_FILE: &str = "owner.lock";
const DB_FILE: &str = "starling.db";
/// Extension of a FLAC encode in progress under `audio/` (#342): written,
/// fsynced and verified there, then renamed onto `<id>.flac`.
const FLAC_TEMP_EXT: &str = "flac-tmp";
/// How old a FLAC temporary must be before reconcile removes it: a live
/// compressor's temp (create → verify → rename, seconds even for an
/// hour-long take) is always younger.
const FLAC_TEMP_GRACE: std::time::Duration = std::time::Duration::from_secs(600);
/// Failed compressions of one take before this instance stops offering it
/// (#342): a journal that cannot be compressed is not re-encoded on every
/// upkeep pass. The next launch tries it again.
const COMPRESSION_ATTEMPTS: u32 = 3;
/// Tombstone ids of audio the retention policy removed are
/// `audio:<captureId>` (kind `audio`): the take's row stays, only its
/// audio is gone. The prefix keeps them out of the capture-id space, so
/// reconcile's dead set never mistakes one for a deleted take.
const AUDIO_TOMBSTONE_PREFIX: &str = "audio:";
/// Prefix of the stamps the retention sweep leaves for the recorder
/// journals it removes from `journals/superseded/` (#356): a copy of a
/// take still kept, never a deleted id — a bare stamp would read as one
/// to reconcile and kill a live take adopted under the journal's name.
const SUPERSEDED_TOMBSTONE_PREFIX: &str = "superseded:";

/// The retention class every take starts in.
pub const STANDARD_CLASS: &str = "standard";
/// The opt-in archival class (#342): its own retention limits. Lossy
/// (Opus) archival encoding is not implemented — takes moved here stay
/// lossless.
pub const ARCHIVAL_CLASS: &str = "archival";

/// Default [`RetentionPolicy::grace`]: no take younger than a day loses
/// its audio to a retention limit, so a take whose transcription just
/// failed always survives long enough to retry (#356).
pub const DEFAULT_RETENTION_GRACE: std::time::Duration =
    std::time::Duration::from_secs(24 * 60 * 60);

/// The §4 schema, in one place. `CREATE ... IF NOT EXISTS` throughout so
/// applying it to an existing same-version database is a no-op.
const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS captures (
    id               TEXT PRIMARY KEY,
    created_utc      TEXT NOT NULL,
    tz               TEXT NOT NULL,
    device           TEXT NOT NULL,
    actual_rate      INTEGER NOT NULL,
    policy           TEXT NOT NULL,
    frame_count      INTEGER NOT NULL,
    ack_sample_index INTEGER NOT NULL,
    journal_hash     TEXT NOT NULL,
    status           TEXT NOT NULL,
    retention_class  TEXT NOT NULL,
    extra_json       TEXT,
    secure_field     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS recognition_attempts (
    id               TEXT PRIMARY KEY,
    capture_id       TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    backend          TEXT NOT NULL,
    model_hash       TEXT,
    language         TEXT,
    options_json     TEXT,
    text             TEXT NOT NULL,
    partial_or_final TEXT NOT NULL,
    status           TEXT NOT NULL,
    timing_json      TEXT,
    extra_json       TEXT,
    created_utc      TEXT
);
CREATE INDEX IF NOT EXISTS idx_recognition_attempts_capture
    ON recognition_attempts(capture_id);
CREATE TABLE IF NOT EXISTS context_snapshots (
    id                TEXT PRIMARY KEY,
    capture_id        TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    descriptor_digest TEXT NOT NULL,
    capabilities      TEXT,
    selection_json    TEXT,
    expiry_utc        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mode_decisions (
    id           TEXT PRIMARY KEY,
    capture_id   TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    mode_id      TEXT NOT NULL,
    mode_version TEXT,
    source       TEXT NOT NULL,
    route_json   TEXT,
    explanation  TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    head_revision INTEGER NOT NULL,
    turn_seq      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    rev_id       TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    base_rev     INTEGER,
    sources_json TEXT,
    text         TEXT NOT NULL,
    status       TEXT NOT NULL,
    provenance   TEXT,
    disposition  TEXT
);
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id   TEXT PRIMARY KEY,
    revision_id   TEXT NOT NULL REFERENCES revisions(rev_id) ON DELETE CASCADE,
    target_json   TEXT NOT NULL,
    compare_token TEXT,
    status        TEXT NOT NULL,
    ack_level     TEXT,
    undo_json     TEXT,
    failure_json  TEXT
);
CREATE TABLE IF NOT EXISTS insight_events (
    event_id     TEXT PRIMARY KEY,
    capture_id   TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    occurred_at  TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_insight_events_capture
    ON insight_events(capture_id);
CREATE TABLE IF NOT EXISTS correction_records (
    id              TEXT PRIMARY KEY,
    capture_id      TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    request_id      TEXT NOT NULL,
    raw_attempt_id  TEXT NOT NULL,
    raw_text        TEXT NOT NULL,
    processed_text  TEXT NOT NULL,
    final_text      TEXT,
    decision        TEXT NOT NULL,
    decision_utc    TEXT NOT NULL,
    mode_id         TEXT,
    mode_version    INTEGER,
    provider_id     TEXT,
    provider_kind   TEXT,
    provider_model  TEXT,
    locality        TEXT,
    transform_kinds TEXT,
    language        TEXT,
    asr_backend     TEXT,
    asr_model_hash  TEXT,
    timings_json    TEXT,
    settings_strength TEXT,
    extra_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_correction_records_capture
    ON correction_records(capture_id);
CREATE TABLE IF NOT EXISTS tombstones (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    deleted_utc TEXT NOT NULL,
    retention   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal_supersessions (
    journal_id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL,
    pending_passes INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS transcription_intents (
    capture_id    TEXT PRIMARY KEY REFERENCES captures(id) ON DELETE CASCADE,
    requested_utc TEXT NOT NULL,
    attempt_id    TEXT,
    rerequested_utc TEXT
);
CREATE TABLE IF NOT EXISTS audio_holds (
    id          TEXT PRIMARY KEY,
    capture_id  TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    holder_pid  INTEGER NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_intents (
    capture_id  TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_seals (
    journal_id      TEXT PRIMARY KEY,
    capture_id      TEXT NOT NULL,
    torn_tail_bytes INTEGER NOT NULL,
    sealed_utc      TEXT NOT NULL
);
";

/// Why a storage v2 operation failed.
#[derive(Debug, thiserror::Error)]
pub enum StoreV2Error {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("storage v2 database error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error(
        "storage v2 database schema version {found} is newer than this build understands \
         ({supported}); upgrade the app before opening this data"
    )]
    SchemaTooNew { found: u32, supported: u32 },
    #[error("storage v2 record is invalid: {0}")]
    Invalid(String),
    #[error("capture {0} was not found")]
    NotFound(String),
    /// The journal handed to [`StoreV2::adopt_journal`] held no verified
    /// samples — a header-only journal from a writer that faulted before
    /// its first boundary. Not a storage fault: the caller falls back to
    /// storing its encoded WAV instead of adopting.
    #[error("capture journal {id} has no verified samples; nothing to adopt")]
    NoVerifiedSamples { id: String },
    #[error("storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),
    #[error("audio error: {0}")]
    Audio(#[from] crate::audio::AudioFormatError),
    #[error("{0}")]
    Flac(#[from] crate::flac::FlacError),
}

/// Lifecycle status of a v2 capture row. Recognition progress lives in
/// `recognition_attempts`; these describe the *take* itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptureStatus {
    /// Finalized journal + committed row; the §4 protocol ran to step 4.
    Complete,
    /// The take ended without a clean stop — crash recovery found it, or
    /// its audio went missing after commit. Audio that exists stays
    /// linked and usable (e17 §2.1 `Interrupted`).
    Interrupted,
}

impl CaptureStatus {
    fn as_str(self) -> &'static str {
        match self {
            CaptureStatus::Complete => "complete",
            CaptureStatus::Interrupted => "interrupted",
        }
    }

    fn parse(text: &str) -> Result<Self, StoreV2Error> {
        match text {
            "complete" => Ok(CaptureStatus::Complete),
            "interrupted" => Ok(CaptureStatus::Interrupted),
            other => Err(StoreV2Error::Invalid(format!(
                "unknown capture status {other:?} (expected complete or interrupted)"
            ))),
        }
    }
}

/// One `captures` row. `extra_json` is the raw stored text: unknown/newer
/// fields preserved verbatim (§4), `None` when the writer wrote NULL.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptureRecord {
    pub id: String,
    pub created_utc: String,
    pub tz: String,
    pub device: String,
    pub actual_rate: u32,
    pub policy: String,
    pub frame_count: u64,
    pub ack_sample_index: u64,
    /// FNV-1a 64 content hash of the journal's sample payloads, hex
    /// (`{:016x}`) — the value sealed into the trailer.
    pub journal_hash: String,
    pub status: CaptureStatus,
    pub retention_class: String,
    pub extra_json: Option<String>,
    /// Captured against a secure/incognito input, or recovered by
    /// [`StoreV2::reconcile`] without its marker: such takes never get
    /// correction records ([`StoreV2::upsert_correction_record`]). The
    /// desktop app has no secure-field capture path; a keyboard/IME
    /// integration marks takes when recording.
    pub secure_field: bool,
}

impl CaptureRecord {
    /// The recovery note an interrupted take carries in `extra_json` (the
    /// gap/salvage note stating exactly what survived).
    pub fn recovery_note(&self) -> Option<String> {
        self.extra_json.as_deref().and_then(|extra| {
            serde_json::from_str::<serde_json::Value>(extra)
                .ok()
                .and_then(|value| {
                    value
                        .get("recovery")
                        .and_then(|note| note.as_str())
                        .map(str::to_string)
                })
        })
    }
}

/// One `recognition_attempts` row (§4 direction). Structured columns carry
/// what querying needs; everything else (v1 transcripts, request ids,
/// timing blobs) rides in `extra_json`/`*_json` text.
#[derive(Clone, Debug)]
pub struct AttemptRecord {
    pub id: String,
    pub capture_id: String,
    pub backend: String,
    pub model_hash: Option<String>,
    pub language: Option<String>,
    pub options_json: Option<String>,
    pub text: String,
    /// "partial" or "final".
    pub partial_or_final: String,
    pub status: String,
    pub timing_json: Option<String>,
    pub extra_json: Option<String>,
    /// When the attempt was inserted, the real updated-at source for
    /// summaries. On insert, `None` means "stamp with now_iso()"
    /// ([`StoreV2::insert_attempt`] applies the default — a caller cannot
    /// write an explicit NULL); readers see `None` only on rows written
    /// before the v2 schema added the column.
    pub created_utc: Option<String>,
}

impl AttemptRecord {
    /// Whether this attempt is a completed final transcript.
    pub fn is_final_transcript(&self) -> bool {
        self.status == "completed" && self.partial_or_final == "final"
    }

    /// The v1-shaped transcript this attempt carries, when it has one: the
    /// verbatim result a writer preserved in `extra_json` (the shape both
    /// the app writes). `None` when the row has no `extra_json` **or when
    /// that JSON is not a transcript** — a failed attempt's
    /// `{"error": ...}` blob, or a corrupted one, deserializes to `None`
    /// just like an absent column (the parse failure is not
    /// distinguishable to the caller).
    pub fn transcript(&self) -> Option<crate::storage::TranscriptionResult> {
        self.extra_json
            .as_deref()
            .and_then(|extra| serde_json::from_str(extra).ok())
    }

    /// The surfaced failure message on a failed attempt.
    pub fn failure_message(&self) -> Option<String> {
        self.extra_json.as_deref().and_then(|extra| {
            serde_json::from_str::<serde_json::Value>(extra)
                .ok()
                .and_then(|value| {
                    value
                        .get("error")
                        .and_then(|error| error.as_str())
                        .map(str::to_string)
                })
        })
    }
}

/// One `revisions` row (§4). The documents machine (I5, issue #220) is
/// the writer; `disposition` distinguishes committed heads from preserved
/// conflict candidates while `status` carries the revision's own
/// lifecycle verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionRow {
    pub rev_id: String,
    pub doc_id: String,
    pub base_revision: Option<u64>,
    /// The revision's provenance (the I3 `Revision`'s
    /// `sourceAttemptIds` + `instructionTemplateId`) encoded as one JSON
    /// object — the column the §4 schema gives this side of the record.
    pub sources_json: Option<String>,
    pub text: String,
    pub status: String,
    pub provenance: Option<String>,
    /// `"committed"` (head, or a past head) or `"preserved"` (conflict
    /// candidate retained for explicit user choice).
    pub disposition: Option<String>,
}

/// One `documents` row with its revisions in insertion order (the
/// [`StoreV2::get_document`] shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentRow {
    pub doc_id: String,
    pub name: String,
    pub head_revision: u64,
    pub turn_seq: u32,
    pub revisions: Vec<RevisionRow>,
}

/// What the user did with one processing proposal:
///
/// - `Accepted` — "Use processed"/"Use anyway" made it the take's head.
/// - `Rejected` — "Dismiss" discarded it unused.
/// - `Reverted` — "Back to raw", or an edit restoring the raw transcript
///   exactly, undid an accepted proposal.
/// - `Edited` — the head was edited after acceptance and no longer equals
///   the proposal (nor the raw).
///
/// A proposal the user never decides on records nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionDecision {
    Accepted,
    Rejected,
    Reverted,
    Edited,
}

impl CorrectionDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            CorrectionDecision::Accepted => "accepted",
            CorrectionDecision::Rejected => "rejected",
            CorrectionDecision::Reverted => "reverted",
            CorrectionDecision::Edited => "edited",
        }
    }

    /// `None` for a value this build does not know.
    pub fn parse(text: &str) -> Option<Self> {
        [
            CorrectionDecision::Accepted,
            CorrectionDecision::Rejected,
            CorrectionDecision::Reverted,
            CorrectionDecision::Edited,
        ]
        .into_iter()
        .find(|decision| decision.as_str() == text)
    }
}

/// One `correction_records` row: the raw transcript, the processing
/// proposal shown for it, and the user's decision, with provenance. One
/// row per take per processing request; a revised decision updates it
/// ([`StoreV2::upsert_correction_record`]). Rows cascade with their
/// capture.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CorrectionRecord {
    pub capture_id: String,
    /// The processing request whose proposal this row is about.
    pub request_id: String,
    /// The recognition attempt whose final text is `raw_text`.
    pub raw_attempt_id: String,
    pub raw_text: String,
    /// The proposal as it was shown.
    pub processed_text: String,
    /// The head text when the decision landed.
    pub final_text: Option<String>,
    pub decision: CorrectionDecision,
    pub decision_utc: String,
    pub mode_id: Option<String>,
    pub mode_version: Option<u32>,
    pub provider_id: Option<String>,
    pub provider_kind: Option<String>,
    pub provider_model: Option<String>,
    pub locality: Option<String>,
    /// JSON array of transform kinds.
    pub transform_kinds: Option<String>,
    pub language: Option<String>,
    /// Filled by the store from the `raw_attempt_id` row on write; the
    /// caller's values are ignored.
    pub asr_backend: Option<String>,
    pub asr_model_hash: Option<String>,
    /// JSON object of per-stage timings (`queued_ms`, `processing_ms`,
    /// `stop_to_result_ms`).
    pub timings_json: Option<String>,
    /// Always NULL until an editing-strength setting exists.
    pub settings_strength: Option<String>,
    pub extra_json: Option<String>,
}

/// A correction record's id: one record per take per processing request.
fn correction_id(capture_id: &str, request_id: &str) -> String {
    format!("{capture_id}#c:{request_id}")
}

/// Metadata known when a take starts (the columns not derived from the
/// journal itself).
#[derive(Clone, Debug, Default)]
pub struct TakeMeta {
    pub tz: String,
    pub device: String,
    pub policy: String,
    pub retention_class: String,
    /// Preserved verbatim in the captures row.
    pub extra_json: Option<String>,
    /// Captured against a secure/incognito input: correction records are
    /// never written for it.
    pub secure_field: bool,
    /// The recorder journal this take is stored in place of (#356): its
    /// id is recorded in the take's commit, so a journal the save did not
    /// get to move aside is never adopted as a second copy.
    pub supersedes_journal: Option<String>,
    /// The take is to be transcribed (#220): a complete commit records
    /// that intent in its own transaction (see
    /// [`StoreV2::claim_transcription`]), so a crash right after the
    /// commit cannot leave the take stored but forgotten.
    pub transcribe: bool,
}

impl TakeMeta {
    /// The defaults a plain dictation take gets.
    pub fn for_device(device: impl Into<String>) -> Self {
        Self {
            tz: local_tz_label(),
            device: device.into(),
            policy: "default".to_string(),
            retention_class: "standard".to_string(),
            extra_json: None,
            secure_field: false,
            supersedes_journal: None,
            transcribe: false,
        }
    }
}

/// A take whose journal is finalized (§4 step 2's trailer + fsyncs done)
/// but not yet promoted or committed. The fault-injection seam between
/// steps 2 and 3.
pub struct FinalizedTake {
    pub id: String,
    pub sample_rate: u32,
    pub total_samples: u64,
    pub content_hash: String,
    pub created_utc: String,
    pub meta: TakeMeta,
}

/// A fully committed take (§4 step 3 done). Returning this value *is* the
/// durable ack to the capture side.
#[derive(Debug)]
pub struct CommittedTake {
    pub record: CaptureRecord,
}

/// The v2 store: SQLite metadata + the audio/staging/quarantine trees.
#[derive(Debug)]
pub struct StoreV2 {
    root: PathBuf,
    conn: Connection,
    commits_since_checkpoint: u32,
    checkpoint_every: u32,
    /// In-flight attempts this handle started (#213), registered
    /// process-wide ([`AttemptClaim`]) until they settle, are deleted, or
    /// the handle drops.
    held_attempts: HashMap<String, AttemptClaim>,
    /// Takes whose audio a caller of this instance is using right now
    /// (#342): id → pin count. A retry pins before it loads the audio and
    /// keeps the pin until its attempt is marked started, so neither
    /// compression nor retention acts in between. In-process only —
    /// another process's take is protected once its attempt row exists.
    audio_pins: HashMap<String, usize>,
    /// Takes whose compression failed in this instance: id → failures.
    /// At [`COMPRESSION_ATTEMPTS`] the take is no longer a candidate.
    compression_failures: HashMap<String, u32>,
    /// Runs once, in [`Self::apply_live_retention_policy`], after the walk
    /// has read a due take and before the removal takes the write lock:
    /// where a peer's commit or file change lands in the race tests.
    #[cfg(test)]
    before_retention_lock: Option<TestHook>,
    /// Runs once, in [`Self::sweep_retention_until`], before a file's
    /// removal takes the write lock.
    #[cfg(test)]
    before_sweep_lock: Option<TestHook>,
    /// The recovery step after which the next recovery stops as if the
    /// process died there (it fails without undoing anything).
    #[cfg(test)]
    crash_after: Option<RecoveryStep>,
}

/// The durable steps recovery takes on a cut-short journal, in order:
/// where the crash tests stop it.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryStep {
    /// The seal is noted in `recovery_seals`.
    SealNoted,
    /// The journal is sealed to its verified prefix.
    Sealed,
    /// The journal is in `audio/` under its capture id.
    Moved,
}

#[cfg(test)]
struct TestHook(Box<dyn FnOnce() + Send>);

#[cfg(test)]
impl std::fmt::Debug for TestHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TestHook")
    }
}

/// Attempt ids started by any store handle in this process (#213).
static LIVE_ATTEMPTS: std::sync::Mutex<std::collections::BTreeSet<String>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

fn live_attempts() -> std::sync::MutexGuard<'static, std::collections::BTreeSet<String>> {
    LIVE_ATTEMPTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One in-flight attempt's registration in [`LIVE_ATTEMPTS`]; dropping it
/// unregisters the attempt.
#[derive(Debug)]
struct AttemptClaim(String);

impl AttemptClaim {
    fn new(attempt_id: &str) -> Self {
        live_attempts().insert(attempt_id.to_string());
        AttemptClaim(attempt_id.to_string())
    }

    fn is_live(attempt_id: &str) -> bool {
        live_attempts().contains(attempt_id)
    }
}

impl Drop for AttemptClaim {
    fn drop(&mut self) {
        live_attempts().remove(&self.0);
    }
}

/// The data root's single-owner lock: an exclusive OS lock on
/// `<root>/owner.lock`, held while this value lives. The OS releases it
/// when the process dies.
#[derive(Debug)]
pub struct DataRootLock {
    _file: File,
}

impl DataRootLock {
    /// Takes the lock on `root` (creating the root). `Ok(None)` when
    /// another holder has it.
    pub fn acquire(root: &Path) -> io::Result<Option<DataRootLock>> {
        std::fs::create_dir_all(root)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(OWNER_LOCK_FILE))?;
        match try_flock_exclusive(&file)? {
            FlockEvidence::Held => Ok(None),
            // Without a lock primitive the root is taken on trust.
            FlockEvidence::Free | FlockEvidence::Unknown => Ok(Some(DataRootLock { _file: file })),
        }
    }
}

/// An in-flight take (§4 step 1): owns the staging journal.
pub struct V2Take {
    writer: JournalWriter<crate::journal::FileSink>,
    created_utc: String,
    meta: TakeMeta,
}

impl StoreV2 {
    /// Opens (creating if absent) the v2 store at `root`
    /// (`<data-root>/starling-gpui`). Creates `audio/`, `staging/`,
    /// `quarantine/` and `starling.db` with the WAL journal and
    /// `synchronous=FULL` — every commit is on disk before it returns,
    /// which is what makes the §4 step-3 ack durable. A database with a
    /// higher schema version is refused ([`StoreV2Error::SchemaTooNew`]).
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, StoreV2Error> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(root.join(AUDIO_DIR))?;
        std::fs::create_dir_all(root.join(STAGING_DIR))?;
        std::fs::create_dir_all(root.join(QUARANTINE_DIR))?;

        let mut conn = Connection::open(root.join(DB_FILE))?;
        let mode: String = conn.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(StoreV2Error::Invalid(format!(
                "could not enable WAL journal mode (got {mode:?})"
            )));
        }
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // The owner process runs several connections; a short busy
        // timeout turns a transient SQLITE_BUSY from another connection's
        // commit into a wait instead of a hard error.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        let db_version = Self::read_schema_version(&conn)?;
        match db_version {
            None => {
                // Fresh database: create everything in one transaction.
                let tx = conn.transaction()?;
                tx.execute_batch(SCHEMA_SQL)?;
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?1)",
                    params![SCHEMA_VERSION.to_string()],
                )?;
                tx.commit()?;
            }
            Some(found) if found == SCHEMA_VERSION => {
                // Same version: schema is already in place; verify the
                // version row is sane and touch nothing else.
            }
            Some(found) if found < SCHEMA_VERSION => {
                // Lower version: apply the (idempotent) schema, add any
                // columns introduced since `found` (`CREATE TABLE IF NOT
                // EXISTS` cannot extend an existing table), and bump.
                let tx = conn.transaction()?;
                tx.execute_batch(SCHEMA_SQL)?;
                if found < 2 {
                    // v1 → v2: attempts gained a creation timestamp. Rows
                    // written before the upgrade keep NULL — readers fall
                    // back to the capture's creation time.
                    let has_created: bool = tx
                        .query_row(
                            "SELECT 1 FROM pragma_table_info('recognition_attempts')
                             WHERE name = 'created_utc'",
                            [],
                            |_| Ok(true),
                        )
                        .optional()?
                        .unwrap_or(false);
                    if !has_created {
                        tx.execute_batch(
                            "ALTER TABLE recognition_attempts ADD COLUMN created_utc TEXT",
                        )?;
                    }
                }
                if found < 4 {
                    // v3 → v4: captures gained the secure/incognito marker;
                    // pre-upgrade rows read as ordinary takes.
                    let has_secure: bool = tx
                        .query_row(
                            "SELECT 1 FROM pragma_table_info('captures')
                             WHERE name = 'secure_field'",
                            [],
                            |_| Ok(true),
                        )
                        .optional()?
                        .unwrap_or(false);
                    if !has_secure {
                        tx.execute_batch(
                            "ALTER TABLE captures ADD COLUMN secure_field INTEGER NOT NULL DEFAULT 0",
                        )?;
                    }
                }
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?1)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![SCHEMA_VERSION.to_string()],
                )?;
                tx.commit()?;
            }
            Some(found) => {
                // Higher version: refuse outright — this build must not
                // read or write a future format (§4 forward-compatible
                // reader). Nothing was mutated.
                return Err(StoreV2Error::SchemaTooNew {
                    found,
                    supported: SCHEMA_VERSION,
                });
            }
        }

        Ok(Self {
            root,
            conn,
            commits_since_checkpoint: 0,
            checkpoint_every: WAL_CHECKPOINT_EVERY_COMMITS,
            held_attempts: HashMap::new(),
            audio_pins: HashMap::new(),
            compression_failures: HashMap::new(),
            #[cfg(test)]
            before_retention_lock: None,
            #[cfg(test)]
            before_sweep_lock: None,
            #[cfg(test)]
            crash_after: None,
        })
    }

    /// `<data-dir>/starling-gpui` — the single data root. The recorder's
    /// live journals scratch under `journals/` beside it (adopted here at
    /// save time).
    pub fn default_root() -> Result<PathBuf, StoreV2Error> {
        Ok(dirs::data_dir()
            .ok_or(crate::storage::StorageError::DataDirUnavailable)?
            .join("starling-gpui"))
    }

    fn read_schema_version(conn: &Connection) -> Result<Option<u32>, StoreV2Error> {
        let has_meta: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'",
                [],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if !has_meta {
            return Ok(None);
        }
        let text: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match text {
            None => Ok(None),
            Some(text) => text
                .parse()
                .map(Some)
                .map_err(|_| StoreV2Error::Invalid(format!(
                    "meta.schema_version {text:?} is not a number"
                ))),
        }
    }

    /// The database's schema version (after any upgrade `open` performed).
    pub fn schema_version(&self) -> Result<u32, StoreV2Error> {
        Ok(Self::read_schema_version(&self.conn)?
            .expect("open guarantees a schema version row"))
    }

    /// The v2 root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Overrides the WAL checkpoint policy (D11: tunable for slow disks).
    /// Takes effect from the next commit.
    pub fn set_checkpoint_every_commits(&mut self, every: u32) {
        self.checkpoint_every = every.max(1);
    }

    fn audio_path(&self, id: &str) -> PathBuf {
        self.root.join(AUDIO_DIR).join(format!("{id}.sj"))
    }

    fn staging_path(&self, id: &str) -> PathBuf {
        self.root.join(STAGING_DIR).join(format!("{id}.sj"))
    }

    fn quarantine_path(&self, id: &str) -> PathBuf {
        self.root.join(QUARANTINE_DIR).join(format!("{id}.sj"))
    }

    /// A compressed take's audio (#342): `audio/<id>.flac`.
    fn flac_path(&self, id: &str) -> PathBuf {
        self.root.join(AUDIO_DIR).join(format!("{id}.{}", flac::FLAC_EXT))
    }

    fn quarantine_flac_path(&self, id: &str) -> PathBuf {
        self.root
            .join(QUARANTINE_DIR)
            .join(format!("{id}.{}", flac::FLAC_EXT))
    }

    /// Whether `audio/` holds any audio for `id`, journal or FLAC.
    fn has_audio(&self, id: &str) -> bool {
        self.audio_path(id).exists() || self.flac_path(id).exists()
    }

    /// Move every audio file `id` has in `audio/` (journal and/or FLAC)
    /// into `quarantine/` — a delete's tombstone commit point (R21).
    fn quarantine_audio(&self, id: &str) -> Result<(), StoreV2Error> {
        let mut moved = false;
        for (from, to) in [
            (self.audio_path(id), self.quarantine_path(id)),
            (self.flac_path(id), self.quarantine_flac_path(id)),
        ] {
            if from.exists() {
                std::fs::create_dir_all(self.root.join(QUARANTINE_DIR))?;
                let _ = std::fs::remove_file(&to); // stale tombstone
                std::fs::rename(&from, &to)?;
                moved = true;
            }
        }
        if moved {
            sync_dir(&self.root.join(AUDIO_DIR))?;
            sync_dir(&self.root.join(QUARANTINE_DIR))?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // §4 crash protocol.
    // ------------------------------------------------------------------

    /// Begins a take (§4 step 1): `staging/<captureId>.sj` created with the
    /// journal header fsynced (a crash here leaves at most an empty valid
    /// journal, which reconciliation reports and keeps). Takes record at
    /// 16 kHz mono (the retained-audio format) unless a later increment
    /// plumbs the device rate through.
    pub fn begin_take(&self, meta: TakeMeta) -> Result<V2Take, StoreV2Error> {
        let id = format!("c_{}", uuid::Uuid::new_v4().simple());
        self.begin_take_with_id(id, 16_000, meta)
    }

    /// [`Self::begin_take`] recording at an explicit sample rate — the
    /// WAV-import shape, whose decoded bytes carry whatever rate the file
    /// had. The id is minted the same way. The returned [`V2Take`] owns its
    /// staging journal: appending, sealing, and finalizing are pure journal
    /// work with no store access, so a caller sharing one store behind a
    /// lock may release the lock for the write phase and retake it for
    /// [`FinalizedTake::commit_marked`].
    pub fn begin_take_at_rate(
        &self,
        sample_rate: u32,
        meta: TakeMeta,
    ) -> Result<V2Take, StoreV2Error> {
        let id = format!("c_{}", uuid::Uuid::new_v4().simple());
        self.begin_take_with_id(id, sample_rate, meta)
    }

    /// [`Self::begin_take`] with a caller-chosen id and journal sample
    /// rate (the WAV-save path, which mints its own id).
    fn begin_take_with_id(
        &self,
        id: String,
        sample_rate: u32,
        meta: TakeMeta,
    ) -> Result<V2Take, StoreV2Error> {
        validate_capture_id(&id)?;
        if let Some(journal_id) = &meta.supersedes_journal {
            // Before any recoverable sample exists (#356): a crash from
            // here on leaves a staging journal reconcile turns into this
            // take, and the recorder journal must already be named as
            // replaced by it. Until that take is stored the record counts
            // for nothing ([`Self::journal_superseded_by`]).
            // A replacement already stored keeps its claim.
            self.conn.execute(
                "INSERT INTO journal_supersessions(journal_id, capture_id) VALUES (?1, ?2)
                 ON CONFLICT(journal_id) DO UPDATE SET capture_id = excluded.capture_id
                 WHERE NOT EXISTS (SELECT 1 FROM captures c
                                   WHERE c.id = journal_supersessions.capture_id)",
                params![journal_id, id],
            )?;
        }
        std::fs::create_dir_all(self.root.join(STAGING_DIR))?;
        let writer =
            JournalWriter::create_named(&self.root.join(STAGING_DIR), id, sample_rate)?;
        Ok(V2Take {
            writer,
            created_utc: now_iso(),
            meta,
        })
    }

    /// §4 step 2's second half: rename `staging/<id>.sj` into `audio/`,
    /// fsyncing both directories so the dirent moves survive a crash.
    /// Refuses to overwrite an existing audio file (a journal is evidence;
    /// there is no truncate path).
    pub fn promote_from_staging(&self, id: &str) -> Result<(), StoreV2Error> {
        validate_capture_id(id)?;
        let staging = self.staging_path(id);
        let audio = self.audio_path(id);
        if audio.exists() {
            return Err(StoreV2Error::Invalid(format!(
                "audio for capture {id} already exists; refusing to overwrite"
            )));
        }
        // Path context on the rename: a bare Permission-denied must name
        // the move it refused (callers chain this into their errors).
        std::fs::rename(&staging, &audio).map_err(|err| {
            StoreV2Error::Io(io::Error::new(
                err.kind(),
                format!("promoting staging journal {staging:?} to {audio:?}: {err}"),
            ))
        })?;
        sync_dir(&self.root.join(AUDIO_DIR))?;
        sync_dir(&self.root.join(STAGING_DIR))?;
        Ok(())
    }

    /// §4 step 3: the SQLite transaction inserting the `captures` row,
    /// committed with `synchronous=FULL`, then the WAL checkpoint per
    /// policy. Returning `Ok` is the durable ack.
    pub fn commit_capture(&mut self, record: &CaptureRecord) -> Result<(), StoreV2Error> {
        self.commit_capture_superseding(record, None, false, false)
    }

    /// Notes, before capture `id`'s audio moves into `audio/`, that its
    /// commit carries the intent to transcribe it: reconcile adopts a take
    /// whose commit a crash cut off with that intent (#220). The commit
    /// clears it.
    fn note_pending_intent(&self, id: &str) -> Result<(), StoreV2Error> {
        self.conn.execute(
            "INSERT INTO pending_intents(capture_id, created_utc) VALUES (?1, ?2)
             ON CONFLICT(capture_id) DO NOTHING",
            params![id, now_iso()],
        )?;
        Ok(())
    }

    fn pending_intent(&self, id: &str) -> Result<bool, StoreV2Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT 1 FROM pending_intents WHERE capture_id = ?1",
                params![id],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    /// Notes that recovery is sealing journal `journal_id` (it was cut
    /// short, its last `torn_tail_bytes` an unfinished write) as capture
    /// `capture_id`, before it does: once sealed it looks like a take that
    /// stopped. Durable (`synchronous=FULL`) before the seal is written; the
    /// commit of `capture_id`'s row clears it.
    fn note_recovery_seal(
        &self,
        journal_id: &str,
        capture_id: &str,
        torn_tail_bytes: u64,
    ) -> Result<(), StoreV2Error> {
        self.conn.execute(
            "INSERT INTO recovery_seals(journal_id, capture_id, torn_tail_bytes, sealed_utc)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(journal_id) DO UPDATE SET
                 capture_id = excluded.capture_id,
                 torn_tail_bytes = MAX(torn_tail_bytes, excluded.torn_tail_bytes)",
            params![journal_id, capture_id, int64(torn_tail_bytes)?, now_iso()],
        )?;
        Ok(())
    }

    /// The torn tail recovery recorded when it sealed journal
    /// `journal_id`, if it did and the adoption has not committed.
    fn recovery_seal(&self, journal_id: &str) -> Result<Option<u64>, StoreV2Error> {
        self.seal_where("journal_id", journal_id)
    }

    /// [`Self::recovery_seal`] by the capture the sealed journal was
    /// moving into: a file under `staging/` or `audio/`.
    fn recovery_seal_for_capture(&self, capture_id: &str) -> Result<Option<u64>, StoreV2Error> {
        self.seal_where("capture_id", capture_id)
    }

    fn seal_where(&self, column: &str, id: &str) -> Result<Option<u64>, StoreV2Error> {
        let torn: Option<i64> = self
            .conn
            .query_row(
                &format!("SELECT MAX(torn_tail_bytes) FROM recovery_seals WHERE {column} = ?1"),
                params![id],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        Ok(torn.map(|torn| torn.max(0) as u64))
    }

    /// Stops a recovery after `step` when a crash test asked for it.
    #[cfg(test)]
    fn crash_point(&mut self, step: RecoveryStep) -> Result<(), StoreV2Error> {
        if self.crash_after == Some(step) {
            self.crash_after = None;
            return Err(StoreV2Error::Invalid(format!("injected crash after {step:?}")));
        }
        Ok(())
    }

    /// Forgets the seals aimed at capture `capture_id`: its journal was
    /// set aside, or it already has its row.
    fn clear_recovery_seals(&self, capture_id: &str) -> Result<(), StoreV2Error> {
        self.conn.execute(
            "DELETE FROM recovery_seals WHERE capture_id = ?1",
            params![capture_id],
        )?;
        Ok(())
    }

    /// [`Self::commit_capture`] that also records, in the same
    /// transaction, the recorder journal the take is stored in place of
    /// ([`TakeMeta::supersedes_journal`]) and, when `transcribe` (the
    /// callers set it only for a take whose audio is complete), the intent
    /// to transcribe it ([`TakeMeta::transcribe`]). `recovered`: a
    /// recovery's commit, which ends the seals aimed at the row.
    fn commit_capture_superseding(
        &mut self,
        record: &CaptureRecord,
        supersedes_journal: Option<&str>,
        transcribe: bool,
        recovered: bool,
    ) -> Result<(), StoreV2Error> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO captures(
                id, created_utc, tz, device, actual_rate, policy, frame_count,
                ack_sample_index, journal_hash, status, retention_class, extra_json,
                secure_field)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                record.id,
                record.created_utc,
                record.tz,
                record.device,
                record.actual_rate,
                record.policy,
                int64(record.frame_count)?,
                int64(record.ack_sample_index)?,
                record.journal_hash,
                record.status.as_str(),
                record.retention_class,
                record.extra_json,
                record.secure_field,
            ],
        )?;
        if let Some(journal_id) = supersedes_journal {
            tx.execute(
                "INSERT INTO journal_supersessions(journal_id, capture_id) VALUES (?1, ?2)
                 ON CONFLICT(journal_id) DO UPDATE SET capture_id = excluded.capture_id",
                params![journal_id, record.id],
            )?;
        }
        if transcribe {
            tx.execute(
                "INSERT INTO transcription_intents(capture_id, requested_utc) VALUES (?1, ?2)",
                params![record.id, now_iso()],
            )?;
        }
        tx.execute(
            "DELETE FROM pending_intents WHERE capture_id = ?1",
            params![record.id],
        )?;
        if recovered {
            tx.execute(
                "DELETE FROM recovery_seals WHERE capture_id = ?1",
                params![record.id],
            )?;
        }
        tx.commit()?;

        self.commits_since_checkpoint += 1;
        if self.commits_since_checkpoint >= self.checkpoint_every {
            self.checkpoint()?;
        }
        Ok(())
    }

    /// §4 step 4: remove staging leftovers whose id already has a
    /// committed row. The step-2 rename already took the live file; this
    /// sweeps files an unusual crash (or a copied staging tree) stranded.
    pub fn gc_staging(&mut self) -> Result<usize, StoreV2Error> {
        let mut swept = 0usize;
        for id in journal_ids_in(&self.root.join(STAGING_DIR)) {
            if self.get_capture(&id)?.is_some() {
                let path = self.staging_path(&id);
                std::fs::remove_file(&path)?;
                sync_dir(&self.root.join(STAGING_DIR))?;
                swept += 1;
            }
        }
        Ok(swept)
    }

    /// Discard one in-flight take's staging journal — the explicit
    /// rollback for a caller driving the §4 steps itself (e.g. the
    /// runtime's samples path) whose write failed before
    /// [`FinalizedTake::commit_marked`]: without this, a failed write
    /// would leave a partial staging journal for reconcile to salvage as
    /// an *interrupted take* — duplicate audio of whatever the caller
    /// stored instead. Returns whether **this call removed** the
    /// journal: `Ok(false)` is the idempotent "already gone" (promoted
    /// past the rename, or raced) — callers report that shape
    /// distinctly, never as a rollback that ran. A committed row for the
    /// id is never touched (its audio lives in `audio/`). Removal
    /// failures (permissions, a directory squatting on the name, I/O
    /// trouble) are errors: the caller must be able to tell "rolled
    /// back" from "the partial staging journal is still there" — the
    /// leftover is exactly what reconcile would salvage as a duplicate.
    /// Once the removal itself succeeded, the journal is gone; the
    /// dirent fsync afterwards is best-effort (the same trade
    /// [`Self::release_lease`] makes after removing the lease files): a
    /// sync failure must not read as "still there" — the crash window it
    /// leaves (an unsynced deletion can resurface after a crash) is
    /// narrower than the misleading-error alternative.
    pub fn discard_staging(&self, id: &str) -> Result<bool, StoreV2Error> {
        validate_capture_id(id)?;
        let path = self.staging_path(id);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                // Best-effort: the journal is gone; only the durability
                // of the deletion's dirent is at stake now.
                let _ = sync_dir(&self.root.join(STAGING_DIR));
                Ok(true)
            }
            // Already gone: promoted past the rename, or raced — the
            // idempotent no-op, reported as such.
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            // Context over structure: nothing walks the source chain (the
            // runtime callers report the message), but the ErrorKind stays
            // inspectable and a bare `IsADirectory` without the path
            // sends triage nowhere.
            Err(err) => {
                return Err(StoreV2Error::Io(io::Error::new(
                    err.kind(),
                    format!("removing staging journal {path:?}: {err}"),
                )))
            }
        }
    }

    /// Runs a passive WAL checkpoint now, resetting the policy counter.
    pub fn checkpoint(&mut self) -> Result<(), StoreV2Error> {
        self.conn
            .query_row("PRAGMA wal_checkpoint(PASSIVE);", [], |_| Ok(()))?;
        self.commits_since_checkpoint = 0;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Row access.
    // ------------------------------------------------------------------

    fn row_to_capture(row: &rusqlite::Row<'_>) -> rusqlite::Result<(CaptureRecord, String)> {
        let status_text: String = row.get(9)?;
        Ok((
            CaptureRecord {
                id: row.get(0)?,
                created_utc: row.get(1)?,
                tz: row.get(2)?,
                device: row.get(3)?,
                actual_rate: row.get::<_, i64>(4)? as u32,
                policy: row.get(5)?,
                frame_count: row.get::<_, i64>(6)? as u64,
                ack_sample_index: row.get::<_, i64>(7)? as u64,
                journal_hash: row.get(8)?,
                // Placeholder: the caller parses `status_text` (which may
                // name a status this build does not know) and either
                // assigns or flags the row as damaged.
                status: CaptureStatus::Complete,
                retention_class: row.get(10)?,
                extra_json: row.get(11)?,
                secure_field: row.get::<_, i64>(12)? != 0,
            },
            status_text,
        ))
    }

    /// Reads one `captures` row. `Ok(None)` when the id is unknown.
    /// Unknown status text surfaces as [`StoreV2Error::Invalid`].
    pub fn get_capture(&self, id: &str) -> Result<Option<CaptureRecord>, StoreV2Error> {
        validate_capture_id(id)?;
        let row = self
            .conn
            .query_row(
                "SELECT id, created_utc, tz, device, actual_rate, policy, frame_count,
                        ack_sample_index, journal_hash, status, retention_class, extra_json,
                        secure_field
                 FROM captures WHERE id = ?1",
                params![id],
                Self::row_to_capture,
            )
            .optional()?;
        match row {
            None => Ok(None),
            Some((mut record, status_text)) => {
                record.status = CaptureStatus::parse(&status_text)?;
                Ok(Some(record))
            }
        }
    }

    /// Updates a capture's status. With `note`, the note is merged into
    /// `extra_json` under `recovery` (existing keys — known or unknown —
    /// are preserved); without it the `extra_json` column is not touched,
    /// keeping unknown fields byte-verbatim.
    pub fn update_capture_status(
        &mut self,
        id: &str,
        status: CaptureStatus,
        note: Option<&str>,
    ) -> Result<(), StoreV2Error> {
        let extra = match note {
            None => None,
            Some(note) => {
                let current = self
                    .get_capture(id)?
                    .ok_or_else(|| StoreV2Error::NotFound(id.to_string()))?
                    .extra_json;
                Some(merge_extra_note(current.as_deref(), note))
            }
        };
        match extra {
            None => self.conn.execute(
                "UPDATE captures SET status = ?1 WHERE id = ?2",
                params![status.as_str(), id],
            )?,
            Some(extra) => self.conn.execute(
                "UPDATE captures SET status = ?1, extra_json = ?2 WHERE id = ?3",
                params![status.as_str(), extra, id],
            )?,
        };
        Ok(())
    }

    /// Inserts a `recognition_attempts` row, stamped with its creation
    /// time (the summary's updated-at source). A savepoint, so it nests
    /// inside a caller's transaction.
    pub fn insert_attempt(&mut self, attempt: &AttemptRecord) -> Result<(), StoreV2Error> {
        let tx = self.conn.savepoint()?;
        insert_attempt_row(&tx, attempt)?;
        tx.commit()?;
        Ok(())
    }

    /// Map one `recognition_attempts` row (column order shared by every
    /// SELECT in this file).
    fn row_to_attempt(row: &rusqlite::Row<'_>) -> rusqlite::Result<AttemptRecord> {
        Ok(AttemptRecord {
            id: row.get(0)?,
            capture_id: row.get(1)?,
            backend: row.get(2)?,
            model_hash: row.get(3)?,
            language: row.get(4)?,
            options_json: row.get(5)?,
            text: row.get(6)?,
            partial_or_final: row.get(7)?,
            status: row.get(8)?,
            timing_json: row.get(9)?,
            extra_json: row.get(10)?,
            created_utc: row.get(11)?,
        })
    }

    /// All attempts for a capture, oldest first (insertion id order is
    /// uuid-random; ordering is by rowid, i.e. insertion order).
    pub fn attempts_for(&self, capture_id: &str) -> Result<Vec<AttemptRecord>, StoreV2Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, capture_id, backend, model_hash, language, options_json, text,
                    partial_or_final, status, timing_json, extra_json, created_utc
             FROM recognition_attempts WHERE capture_id = ?1 ORDER BY rowid",
        )?;
        let rows = stmt.query_map(params![capture_id], Self::row_to_attempt)?;
        let mut attempts = Vec::new();
        for row in rows {
            attempts.push(row?);
        }
        Ok(attempts)
    }

    /// All attempts for every capture in `capture_ids`, oldest first,
    /// grouped by capture id — one query per chunk instead of one per
    /// record (the listing path runs after every save, transcript,
    /// failure, and delete, so the per-record round-trips added up).
    /// Captures with no attempts are simply absent from the map.
    pub fn attempts_grouped_by_capture(
        &self,
        capture_ids: &[String],
    ) -> Result<HashMap<String, Vec<AttemptRecord>>, StoreV2Error> {
        let mut grouped: HashMap<String, Vec<AttemptRecord>> = HashMap::new();
        // Stay well under SQLite's default 999 host-parameter limit.
        for chunk in capture_ids.chunks(500) {
            let placeholders = std::iter::repeat("?")
                .take(chunk.len())
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT id, capture_id, backend, model_hash, language, options_json, text,
                        partial_or_final, status, timing_json, extra_json, created_utc
                 FROM recognition_attempts WHERE capture_id IN ({placeholders}) ORDER BY rowid"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> = chunk
                .iter()
                .map(|id| id as &dyn rusqlite::ToSql)
                .collect();
            let rows = stmt.query_map(params.as_slice(), Self::row_to_attempt)?;
            for row in rows {
                let attempt = row?;
                grouped
                    .entry(attempt.capture_id.clone())
                    .or_default()
                    .push(attempt);
            }
        }
        Ok(grouped)
    }

    // ------------------------------------------------------------------
    // Documents / revisions (§4 `documents`/`revisions` tables — the I5
    // documents-machine persistence, issue #220). Rows are plain
    // metadata: no audio, no per-row files, so these APIs have no
    // filesystem side and no bounded-read concerns beyond the one query
    // each.
    // ------------------------------------------------------------------

    /// Creates or updates one `documents` row. Idempotent on `doc_id`:
    /// a re-commit of the same head (a caller retrying after a crash
    /// window) writes the same row rather than colliding. The durable
    /// head never moves backwards: an older `head_revision` than the
    /// stored one is a no-op, reported as `Ok(false)`.
    pub fn upsert_document(
        &self,
        doc_id: &str,
        name: &str,
        head_revision: u64,
        turn_seq: u32,
    ) -> Result<bool, StoreV2Error> {
        validate_document_id(doc_id)?;
        let changed = self.conn.execute(
            "INSERT INTO documents(doc_id, name, head_revision, turn_seq)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(doc_id) DO UPDATE SET
                name = excluded.name,
                head_revision = excluded.head_revision,
                turn_seq = excluded.turn_seq
             WHERE excluded.head_revision >= documents.head_revision",
            params![doc_id, name, int64(head_revision)?, int64(u64::from(turn_seq))?],
        )?;
        Ok(changed > 0)
    }

    /// Inserts (or idempotently re-stores) one `revisions` row. The
    /// referenced `doc_id` must already exist — `foreign_keys` is ON, so a
    /// revision for an unknown document fails with the SQLite foreign-key
    /// error rather than landing orphaned. `disposition` carries the
    /// documents machine's slot (`"committed"` head / `"preserved"`
    /// conflict candidate); `status` keeps the revision's own lifecycle
    /// status verbatim (e.g. `"candidate"`) — two different claims, which
    /// is why the schema has both columns.
    pub fn store_document_revision(&self, revision: &RevisionRow) -> Result<(), StoreV2Error> {
        validate_document_id(&revision.rev_id)?;
        validate_document_id(&revision.doc_id)?;
        let changed = self.conn.execute(
            "INSERT INTO revisions(rev_id, doc_id, base_rev, sources_json, text,
                                   status, provenance, disposition)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(rev_id) DO UPDATE SET
                base_rev = excluded.base_rev,
                sources_json = excluded.sources_json,
                text = excluded.text,
                status = excluded.status,
                provenance = excluded.provenance,
                disposition = excluded.disposition
             WHERE revisions.doc_id = excluded.doc_id",
            params![
                revision.rev_id,
                revision.doc_id,
                revision.base_revision.map(|base| int64(base)).transpose()?,
                revision.sources_json,
                revision.text,
                revision.status,
                revision.provenance,
                revision.disposition,
            ],
        )?;
        if changed == 0 {
            return Err(StoreV2Error::Invalid(format!(
                "revision {} already belongs to another document",
                revision.rev_id
            )));
        }
        Ok(())
    }

    /// Advances a document's head and lands the head's revision row in
    /// one transaction: the durable head never references a revision
    /// whose row (and text) did not land. Both writes run on `self.conn`
    /// inside the open transaction; the store's single connection sits
    /// behind a Mutex in every embedder, so nothing interleaves.
    pub fn commit_document_head(
        &self,
        name: &str,
        head_revision: u64,
        turn_seq: u32,
        revision: &RevisionRow,
    ) -> Result<(), StoreV2Error> {
        self.commit_document_head_with(name, head_revision, turn_seq, revision, &[])
    }

    /// [`Self::commit_document_head`] plus further revision rows of the
    /// same document (e.g. the proposal the new head accepted), all in one
    /// transaction: either every row lands or none does.
    pub fn commit_document_head_with(
        &self,
        name: &str,
        head_revision: u64,
        turn_seq: u32,
        revision: &RevisionRow,
        also: &[RevisionRow],
    ) -> Result<(), StoreV2Error> {
        if let Some(row) = also.iter().find(|row| row.doc_id != revision.doc_id) {
            return Err(StoreV2Error::Invalid(format!(
                "revision {} belongs to another document",
                row.rev_id
            )));
        }
        let tx = self.conn.unchecked_transaction()?;
        // A refused write returns before `commit`: dropping `tx` rolls
        // the whole set back.
        if !self.upsert_document(&revision.doc_id, name, head_revision, turn_seq)? {
            return Err(StoreV2Error::Invalid(format!(
                "document {} head {head_revision} is older than the durable head",
                revision.doc_id
            )));
        }
        self.store_document_revision(revision)?;
        for row in also {
            self.store_document_revision(row)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Deletes one document and, by cascade, its revisions and their
    /// delivery rows (`revisions` and `deliveries` both declare ON DELETE
    /// CASCADE), so the document's delivery history goes too. `Ok(false)`
    /// when there was nothing to delete. Documents carry no foreign key
    /// to a capture, and deleting a capture does not delete them: an
    /// embedder that keys a document by capture id must call this
    /// alongside `delete_capture` itself.
    pub fn delete_document(&self, doc_id: &str) -> Result<bool, StoreV2Error> {
        validate_document_id(doc_id)?;
        let changed = self
            .conn
            .execute("DELETE FROM documents WHERE doc_id = ?1", params![doc_id])?;
        Ok(changed > 0)
    }

    /// Records one insight event (`packages/contracts/insight-events`)
    /// for a capture. Idempotent on `event_id` like the contract says: a
    /// replay identical in every column is a no-op, anything else under a
    /// known id is an error, never an overwrite. The row cascades away
    /// with its capture, so deleting a take removes its events; an event
    /// for a capture that does not exist (deleted meanwhile) is
    /// `NotFound`. The check, the insert and the comparison run in one
    /// transaction, so the contract holds without a lock around the store.
    pub fn record_insight_event(
        &self,
        event_id: &str,
        capture_id: &str,
        kind: &str,
        occurred_at: &str,
        payload_json: &str,
    ) -> Result<(), StoreV2Error> {
        let tx = self.conn.unchecked_transaction()?;
        if self.get_capture(capture_id)?.is_none() {
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        let inserted = self.conn.execute(
            "INSERT INTO insight_events(event_id, capture_id, type, occurred_at, payload_json)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(event_id) DO NOTHING",
            params![event_id, capture_id, kind, occurred_at, payload_json],
        )?;
        if inserted == 0 {
            let known: (String, String, String, String) = self.conn.query_row(
                "SELECT capture_id, type, occurred_at, payload_json
                 FROM insight_events WHERE event_id = ?1",
                params![event_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;
            let replay = (
                capture_id.to_string(),
                kind.to_string(),
                occurred_at.to_string(),
                payload_json.to_string(),
            );
            if known != replay {
                return Err(StoreV2Error::Invalid(format!(
                    "insight event {event_id} already recorded differently"
                )));
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// One capture's insight events as `(type, payload_json)`, in
    /// insertion order.
    pub fn insight_events_for(&self, capture_id: &str) -> Result<Vec<(String, String)>, StoreV2Error> {
        let mut stmt = self.conn.prepare(
            "SELECT type, payload_json FROM insight_events WHERE capture_id = ?1 ORDER BY rowid",
        )?;
        let rows = stmt.query_map(params![capture_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let mut events = Vec::new();
        for row in rows {
            events.push(row?);
        }
        Ok(events)
    }

    // ------------------------------------------------------------------
    // Correction records.
    // ------------------------------------------------------------------

    /// Writes or revises the correction record for one take and request
    /// (row id `<captureId>#c:<requestId>`). A revision moves only the
    /// decision columns (`final_text`, `decision`, `decision_utc`,
    /// `timings_json`, `extra_json`); the rest stay as first written.
    /// `Ok(false)` without writing for a `secure_field` capture;
    /// `NotFound` when the capture is gone.
    pub fn upsert_correction_record(
        &self,
        record: &CorrectionRecord,
    ) -> Result<bool, StoreV2Error> {
        let Some(capture) = self.get_capture(&record.capture_id)? else {
            return Err(StoreV2Error::NotFound(record.capture_id.clone()));
        };
        if capture.secure_field {
            return Ok(false);
        }
        let (asr_backend, asr_model_hash): (Option<String>, Option<String>) = self
            .conn
            .query_row(
                "SELECT backend, model_hash FROM recognition_attempts WHERE id = ?1",
                params![record.raw_attempt_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .unwrap_or((None, None));
        self.conn.execute(
            "INSERT INTO correction_records(
                id, capture_id, request_id, raw_attempt_id, raw_text, processed_text,
                final_text, decision, decision_utc, mode_id, mode_version, provider_id,
                provider_kind, provider_model, locality, transform_kinds, language,
                asr_backend, asr_model_hash, timings_json, settings_strength, extra_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                     ?16, ?17, ?18, ?19, ?20, ?21, ?22)
             ON CONFLICT(id) DO UPDATE SET
                final_text = excluded.final_text,
                decision = excluded.decision,
                decision_utc = excluded.decision_utc,
                timings_json = excluded.timings_json,
                extra_json = excluded.extra_json",
            params![
                correction_id(&record.capture_id, &record.request_id),
                record.capture_id,
                record.request_id,
                record.raw_attempt_id,
                record.raw_text,
                record.processed_text,
                record.final_text,
                record.decision.as_str(),
                record.decision_utc,
                record.mode_id,
                record.mode_version,
                record.provider_id,
                record.provider_kind,
                record.provider_model,
                record.locality,
                record.transform_kinds,
                record.language,
                asr_backend,
                asr_model_hash,
                record.timings_json,
                record.settings_strength,
                record.extra_json,
            ],
        )?;
        Ok(true)
    }

    /// Revises the decision columns of an existing correction record;
    /// `Ok(false)` when the take has no record for `request_id`.
    pub fn revise_correction_record(
        &self,
        capture_id: &str,
        request_id: &str,
        decision: CorrectionDecision,
        decision_utc: &str,
        final_text: &str,
    ) -> Result<bool, StoreV2Error> {
        let changed = self.conn.execute(
            "UPDATE correction_records SET decision = ?2, decision_utc = ?3, final_text = ?4
             WHERE id = ?1",
            params![
                correction_id(capture_id, request_id),
                decision.as_str(),
                decision_utc,
                final_text,
            ],
        )?;
        Ok(changed > 0)
    }

    /// One capture's correction records, in insertion order. A damaged
    /// row fails the whole read rather than being skipped, so a dataset
    /// read never silently loses pairs.
    pub fn correction_records_for(
        &self,
        capture_id: &str,
    ) -> Result<Vec<CorrectionRecord>, StoreV2Error> {
        validate_capture_id(capture_id)?;
        let mut stmt = self.conn.prepare(
            "SELECT capture_id, request_id, raw_attempt_id, raw_text, processed_text,
                    final_text, decision, decision_utc, mode_id, mode_version, provider_id,
                    provider_kind, provider_model, locality, transform_kinds, language,
                    asr_backend, asr_model_hash, timings_json, settings_strength, extra_json
             FROM correction_records WHERE capture_id = ?1 ORDER BY rowid",
        )?;
        let records = stmt
            .query_map(params![capture_id], |row| {
                let decision: String = row.get(6)?;
                let decision = CorrectionDecision::parse(&decision).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        6,
                        rusqlite::types::Type::Text,
                        format!("unknown correction decision {decision:?}").into(),
                    )
                })?;
                Ok(CorrectionRecord {
                    capture_id: row.get(0)?,
                    request_id: row.get(1)?,
                    raw_attempt_id: row.get(2)?,
                    raw_text: row.get(3)?,
                    processed_text: row.get(4)?,
                    final_text: row.get(5)?,
                    decision,
                    decision_utc: row.get(7)?,
                    mode_id: row.get(8)?,
                    mode_version: row.get(9)?,
                    provider_id: row.get(10)?,
                    provider_kind: row.get(11)?,
                    provider_model: row.get(12)?,
                    locality: row.get(13)?,
                    transform_kinds: row.get(14)?,
                    language: row.get(15)?,
                    asr_backend: row.get(16)?,
                    asr_model_hash: row.get(17)?,
                    timings_json: row.get(18)?,
                    settings_strength: row.get(19)?,
                    extra_json: row.get(20)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(records)
    }

    /// Advances one document's `turn_seq`, creating the row (head 0,
    /// name = id) when a turn is appended to a document no updateHead has
    /// written yet — the documents machine's implicit-document shape. An
    /// existing row's `head_revision` is never touched here.
    pub fn bump_document_turn(&self, doc_id: &str, turn_seq: u32) -> Result<(), StoreV2Error> {
        validate_document_id(doc_id)?;
        self.conn.execute(
            "INSERT INTO documents(doc_id, name, head_revision, turn_seq)
             VALUES (?1, ?1, 0, ?2)
             ON CONFLICT(doc_id) DO UPDATE SET turn_seq = excluded.turn_seq",
            params![doc_id, int64(u64::from(turn_seq))?],
        )?;
        Ok(())
    }

    /// The document that holds revision `rev_id`; `Ok(None)` when no
    /// document does. Revision ids are unique across documents (the
    /// primary key), so a writer checks ownership before claiming one.
    pub fn revision_document(&self, rev_id: &str) -> Result<Option<String>, StoreV2Error> {
        validate_document_id(rev_id)?;
        Ok(self
            .conn
            .query_row(
                "SELECT doc_id FROM revisions WHERE rev_id = ?1",
                params![rev_id],
                |row| row.get(0),
            )
            .optional()?)
    }

    /// Loads one document and its revisions (insertion order). `Ok(None)`
    /// for a document this root has never stored. Revisions of a missing
    /// document cannot exist (the foreign key), so there is no
    /// half-present shape to interpret.
    pub fn get_document(&self, doc_id: &str) -> Result<Option<DocumentRow>, StoreV2Error> {
        validate_document_id(doc_id)?;
        // Both reads run on the one connection, which sits behind a Mutex
        // in every embedder: the document row and its revisions are one
        // consistent snapshot.
        let document = self
            .conn
            .query_row(
                "SELECT doc_id, name, head_revision, turn_seq
                 FROM documents WHERE doc_id = ?1",
                params![doc_id],
                |row| {
                    Ok(DocumentRow {
                        doc_id: row.get(0)?,
                        name: row.get(1)?,
                        head_revision: row.get::<_, i64>(2)?.max(0) as u64,
                        turn_seq: row.get::<_, i64>(3)?.max(0) as u32,
                        revisions: Vec::new(),
                    })
                },
            )
            .optional()?;
        let Some(mut document) = document else {
            return Ok(None);
        };
        let mut stmt = self.conn.prepare(
            "SELECT rev_id, doc_id, base_rev, sources_json, text, status, provenance, disposition
             FROM revisions WHERE doc_id = ?1 ORDER BY rowid",
        )?;
        let rows = stmt.query_map(params![doc_id], |row| {
            Ok(RevisionRow {
                rev_id: row.get(0)?,
                doc_id: row.get(1)?,
                base_revision: row.get::<_, Option<i64>>(2)?.map(|base| base.max(0) as u64),
                sources_json: row.get(3)?,
                text: row.get(4)?,
                status: row.get(5)?,
                provenance: row.get(6)?,
                disposition: row.get(7)?,
            })
        })?;
        for row in rows {
            document.revisions.push(row?);
        }
        Ok(Some(document))
    }

    // ------------------------------------------------------------------
    // Bounded listing + lazy audio (G02 semantics carried to v2).
    // ------------------------------------------------------------------

    /// Metadata-only listing: rows come from SQLite; each record gets a
    /// bounded audio check (file presence + the 13-byte journal header —
    /// never PCM). A damaged row or a missing/corrupt journal surfaces its
    /// reason on that record only; the listing never aborts. The page is
    /// bounded at the store layer too: `limit` is clamped to
    /// [`LIST_PAGE_MAX`], so no caller can make one query materialize the
    /// whole history (the app facade pages at 200; paging policy above the
    /// clamp is the facade's business).
    pub fn list_records(&self, offset: usize, limit: usize) -> Result<CapturePage, StoreV2Error> {
        let limit = limit.min(LIST_PAGE_MAX);
        let total: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM captures", [], |row| row.get(0))?;
        let mut stmt = self.conn.prepare(
            "SELECT id, created_utc, tz, device, actual_rate, policy, frame_count,
                    ack_sample_index, journal_hash, status, retention_class, extra_json,
                    secure_field
             FROM captures ORDER BY created_utc DESC, id LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt.query_map(params![int64(limit as u64)?, int64(offset as u64)?], |row| {
            Self::row_to_capture(row)
        })?;

        let mut records = Vec::new();
        for row in rows {
            let (mut record, status_text) = row?;
            let id = record.id.clone();
            match CaptureStatus::parse(&status_text) {
                Ok(status) => {
                    record.status = status;
                    if !is_safe_path_component(&id) {
                        records.push(ListedCapture::Damaged(DamagedCaptureV2 {
                            id,
                            reason: "capture id is not a safe path component".to_string(),
                        }));
                        continue;
                    }
                    let audio = self.audio_at_rest(&id)?;
                    let problems = self.audio_problems(&record, &audio);
                    records.push(ListedCapture::Capture(CaptureListing {
                        record,
                        problems,
                        audio,
                    }));
                }
                Err(err) => records.push(ListedCapture::Damaged(DamagedCaptureV2 {
                    id,
                    reason: err.to_string(),
                })),
            }
        }

        Ok(CapturePage {
            records,
            total: total as usize,
            offset,
        })
    }

    /// What form a take's audio is kept in (#342): its journal, its
    /// FLAC, removed by the retention policy, or missing. A retention
    /// stamp wins over files still on disk — the policy's unlink follows
    /// the stamp, and a crash between the two is finished by reconcile.
    pub fn audio_at_rest(&self, id: &str) -> Result<AudioAtRest, StoreV2Error> {
        if let Some(utc) = self.audio_retired_utc(id)? {
            return Ok(AudioAtRest::Retired { utc });
        }
        Ok(if self.audio_path(id).exists() {
            AudioAtRest::Journal
        } else if self.flac_path(id).exists() {
            AudioAtRest::Flac
        } else {
            AudioAtRest::Missing
        })
    }

    /// When the retention policy removed `id`'s audio, if it did.
    pub fn audio_retired_utc(&self, id: &str) -> Result<Option<String>, StoreV2Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT deleted_utc FROM tombstones WHERE id = ?1 AND kind = 'audio'",
                params![format!("{AUDIO_TOMBSTONE_PREFIX}{id}")],
                |row| row.get(0),
            )
            .optional()?)
    }

    /// Bounded per-record audio check: the reasons a committed row cannot
    /// currently be played back, without loading any samples. Audio the
    /// retention policy removed is not a problem — the listing reports
    /// it as [`AudioAtRest::Retired`].
    fn audio_problems(&self, record: &CaptureRecord, audio: &AudioAtRest) -> Vec<String> {
        let path = match audio {
            AudioAtRest::Retired { .. } => return Vec::new(),
            AudioAtRest::Flac => {
                return match std::fs::File::open(self.flac_path(&record.id)).and_then(|mut file| {
                    use std::io::Read;
                    let mut magic = [0u8; 4];
                    file.read_exact(&mut magic)?;
                    Ok(magic)
                }) {
                    Ok(magic) if &magic == b"fLaC" => Vec::new(),
                    Ok(_) => vec!["compressed audio does not start with the FLAC marker".to_string()],
                    Err(err) => vec![format!("compressed audio: {err}")],
                };
            }
            AudioAtRest::Journal | AudioAtRest::Missing => self.audio_path(&record.id),
        };
        match std::fs::metadata(&path) {
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                vec!["audio journal is missing — the take was committed but its journal \
                      is gone; the row is kept and marked interrupted"
                    .to_string()]
            }
            Err(err) => vec![format!("audio journal: {err}")],
            Ok(_) => {
                use std::io::Read;
                let mut problems = Vec::new();
                match std::fs::File::open(&path).and_then(|mut file| {
                    let mut header = [0u8; 13];
                    file.read_exact(&mut header)?;
                    Ok(header)
                }) {
                    Ok(header) => {
                        if !journal::is_journal_header(&header) {
                            problems.push(
                                "audio journal does not start with the v1 header".to_string(),
                            );
                        }
                    }
                    Err(err) => problems.push(format!("audio journal header: {err}")),
                }
                problems
            }
        }
    }

    /// Metadata-only half of [`Self::load_audio`]: the on-disk path of a
    /// capture's audio journal. [`StoreV2Error::NotFound`] when the id has
    /// no row; [`StoreV2Error::Invalid`] when the row exists but its journal
    /// file is gone. Deliberately split out so a caller sharing the store
    /// behind a lock can resolve the path under the guard and do the heavy
    /// read + verify + WAV encode (via [`read_audio_journal`]) without it.
    ///
    /// The path is the journal while one exists, else the take's FLAC
    /// (#342); [`read_audio_journal`] reads either, and falls over to the
    /// FLAC when compression replaced the journal between the two calls.
    pub fn audio_journal_path(&self, id: &str) -> Result<PathBuf, StoreV2Error> {
        validate_capture_id(id)?;
        if self.get_capture(id)?.is_none() {
            return Err(StoreV2Error::NotFound(id.to_string()));
        }
        if let Some(utc) = self.audio_retired_utc(id)? {
            return Err(StoreV2Error::Invalid(format!(
                "the audio of capture {id} was removed by the retention policy on {utc}; \
                 its transcript is kept"
            )));
        }
        let path = self.audio_path(id);
        if path.exists() {
            return Ok(path);
        }
        let flac = self.flac_path(id);
        if flac.exists() {
            return Ok(flac);
        }
        Err(StoreV2Error::Invalid(format!(
            "capture {id} has no audio journal on disk"
        )))
    }

    /// Lazily loads one take's audio (the G02 "load on demand" contract):
    /// reads and verifies `audio/<id>.sj`. The verified prefix excludes any
    /// torn tail; `torn_tail_bytes` reports it.
    pub fn load_audio(&self, id: &str) -> Result<JournalAudio, StoreV2Error> {
        let path = self.audio_journal_path(id)?;
        read_audio_journal(&path)
    }

    /// Whether `audio/` holds the journal for `id` — one metadata probe,
    /// no row read, no journal parse. The rollback seam for a caller that
    /// failed between promotion and commit and must learn which side of
    /// the rename the bytes sit on ([`audio_journal_path`] cannot answer
    /// that: it resolves through the row, which in exactly that shape
    /// does not exist yet). A probe that itself errors (`try_exists`, not
    /// `exists`) propagates instead of reading as "not promoted".
    pub fn audio_journal_exists(&self, id: &str) -> Result<bool, StoreV2Error> {
        validate_capture_id(id)?;
        Ok(self.audio_path(id).try_exists()? || self.flac_path(id).try_exists()?)
    }

    // ------------------------------------------------------------------
    // Deletion (R21 semantics on the v2 layout).
    // ------------------------------------------------------------------

    /// Confirmed capture deletion: quarantine the journal (rename into
    /// `quarantine/` + fsyncs — the tombstone commit point), then one
    /// transaction inserting the `tombstones` row and removing the
    /// `captures` row (attempts, insight events and correction records
    /// cascade). A crash between the two is
    /// completed by [`Self::reconcile`]; a crash before the rename leaves
    /// everything live. Idempotent: deleting an unknown id is `Ok`.
    /// Deleting a take is the **only** thing that removes its correction
    /// records — success, failure and retention elsewhere never do.
    pub fn delete_capture(&mut self, id: &str) -> Result<(), StoreV2Error> {
        validate_capture_id(id)?;
        let row = self.get_capture(id)?;
        if row.is_none() && !self.has_audio(id) {
            return Ok(());
        }

        // The database's write lock is taken before any file moves: a
        // compression on another connection publishes its FLAC under that
        // lock, so it either finishes first (and the FLAC is quarantined
        // here) or sees the take gone and publishes nothing.
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;

        // 1. Tombstone: the quarantine rename is the commit point.
        self.quarantine_audio(id)?;

        // 2. Row removal in the same transaction as the tombstone insert.
        // The capture's attempt ids are collected first: the DELETE
        // cascades the rows away, and their in-flight markers (#213) must
        // go with them (released only after the commit took, so a failed
        // delete leaves a live attempt fully owned).
        let attempt_ids: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id FROM recognition_attempts WHERE capture_id = ?1")?;
            let rows = stmt.query_map(params![id], |row| row.get::<_, String>(0))?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        tx.execute(
            "INSERT OR REPLACE INTO tombstones(id, kind, deleted_utc, retention)
             VALUES (?1, 'capture', ?2, 'quarantined')",
            params![id, now_iso()],
        )?;
        tx.execute("DELETE FROM captures WHERE id = ?1", params![id])?;
        tx.commit()?;
        for attempt_id in &attempt_ids {
            self.release_attempt(attempt_id);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Retention sweep (R21: never-delete-until-swept).
    // ------------------------------------------------------------------

    /// The explicit retention sweep — the **only** code path in the store
    /// that unlinks deliberately-deleted content. Policy (frozen, §4):
    /// a confirmed delete quarantines the journal and tombstones the row;
    /// the bytes stay on disk, recoverable, until this sweep is called.
    /// The store itself never calls it: no read path, no reconcile, no
    /// open ever sweeps (that is the never-delete-until-swept contract).
    /// The desktop app calls it on every audio upkeep pass
    /// (`crates/app/src/store.rs`, `audio_upkeep`), so deleted audio is
    /// removed within about one upkeep interval of the delete (#342).
    ///
    /// What it sweeps, per tree:
    ///
    /// - `quarantine/` — v2 tombstoned capture journals
    ///   ([`Self::delete_capture`], interrupted deletes completed by
    ///   [`Self::reconcile`]);
    /// - `journals/deleted/` — recorder journals of takes the user
    ///   deleted, whose ids are tombstoned already (#356);
    /// - `journals/superseded/` — recorder journals a stored (or
    ///   delete-quarantined) take was proven, by reading its audio back,
    ///   to hold every sample of (#356, [`supersede_journal_held_by`]) —
    ///   nothing unproven is ever moved there. Their stamps carry
    ///   [`SUPERSEDED_TOMBSTONE_PREFIX`]: a copy's removal deadens no id.
    ///
    /// Bookkeeping: per file, the `tombstones` row is stamped
    /// `retention = 'swept'` **before** the bytes are unlinked (a
    /// tombstone is created if none exists — the never-resurrect
    /// guarantee must outlive the bytes, including across a crash inside
    /// the sweep; both steps are idempotent, so the next sweep re-attempts
    /// whatever a crash left); the report names every swept file with its
    /// size, and every entry that was left in place with the reason. Live
    /// content — `audio/`, `staging/`, anything not under the two
    /// tombstone trees — is untouched by construction. A deleted take's
    /// audio still pinned ([`Self::pin_audio`]: a read that began before
    /// the delete) stays until a sweep after the pin is released.
    pub fn sweep_retention(&mut self) -> Result<SweepReport, StoreV2Error> {
        self.sweep_retention_until(|| false)
    }

    /// [`Self::sweep_retention`] that ends early, with
    /// [`SweepReport::stopped`], once `stop` says so — asked under the
    /// database's write lock before each removal (a take started
    /// recording).
    pub fn sweep_retention_until(
        &mut self,
        stop: impl Fn() -> bool,
    ) -> Result<SweepReport, StoreV2Error> {
        let mut report = SweepReport::default();
        // Recorder journals a stored take provably holds (#356): a copy
        // of audio history already kept whole. Proven again, per journal,
        // under the write lock its removal holds — the take may have
        // changed since (another connection's compression resamples a take
        // not recorded at 16 kHz), and a copy no longer proven is kept.
        // Swept before quarantine, whose deleted takes' audio may be the
        // proof.
        let superseded = self.root.join("journals").join(SUPERSEDED_SUBDIR);
        let unproven = |store: &Self, path: &Path| {
            (!store.superseded_journal_proven(path))
                .then_some("no stored take is proven to hold this journal's audio any more")
        };
        self.sweep_tree(
            &superseded,
            "journal",
            SUPERSEDED_TOMBSTONE_PREFIX,
            &unproven,
            &stop,
            &mut report,
        )?;
        let in_use = |store: &Self, path: &Path| {
            let id = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or_default();
            store
                .audio_pins
                .contains_key(id)
                .then_some("the deleted recording's audio is still being read")
        };
        self.sweep_tree(
            &self.root.join(QUARANTINE_DIR),
            "capture",
            "",
            &in_use,
            &stop,
            &mut report,
        )?;
        self.sweep_tree(
            &self.root.join("journals").join(DELETED_SUBDIR),
            "journal",
            "",
            &|_, _| None,
            &stop,
            &mut report,
        )?;
        Ok(report)
    }

    /// Whether journal `path` in `superseded/` (#356) is a copy a take's
    /// audio, read back now, still proves ([`Self::journal_copy`]): only
    /// these may be swept. A numbered name (`<id>.<n>.sj`) is proven as
    /// `<id>`.
    fn superseded_journal_proven(&self, path: &Path) -> bool {
        let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("");
        let id = match stem.rsplit_once('.') {
            Some((id, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => id,
            _ => stem,
        };
        read_journal(path).ok().is_some_and(|parsed| {
            matches!(
                self.journal_copy(id, &parsed.samples, parsed.sample_rate),
                Ok(JournalCopy::Stored | JournalCopy::Deleted)
            )
        })
    }

    /// Sweep one tombstone tree into `report` (`kind` is the tombstone
    /// kind rows get: `capture` for v2 quarantine, `journal` for the
    /// journal trees; `stamp_prefix` goes before each file's id in its
    /// stamp, so a tree of copies never deadens the id of a live take;
    /// the files `keep` gives a reason for are left in place; once `stop`
    /// says so, nothing more is removed and the report says stopped).
    ///
    /// Per file, `stop` (before and after `keep`), `keep` and the stamp
    /// run under the database's
    /// write lock, so no other connection deletes, compresses or retires
    /// a take between the decision and the removal. The ordering is
    /// **stamp, then unlink**: the `tombstones` UPSERT (retention
    /// `'swept'`) is committed before the bytes are removed. Both steps
    /// are idempotent, so any crash inside the pair leaves either
    /// file-plus-stamped-row (the next sweep re-attempts the unlink) or
    /// just the stamped row — never the bytes without their tombstone.
    /// That ordering is what makes the never-resurrect guarantee hold for
    /// files that reach the sweep with **no** tombstone row of their own:
    /// quarantine files from `delete_capture`'s own crash window (rename
    /// committed, transaction not — the shape `reconcile`'s
    /// `complete_tombstoned` heals). Reconcile's dead set reads `tombstones` rows ∪ quarantine
    /// files regardless of the retention value, so a stamped-but-not-yet-
    /// unlinked id is already dead to it.
    ///
    /// A prefixed tree (`superseded/`) is the exception: its proof holds
    /// only while the lock does, so its copies are unlinked before the
    /// commit. A crash in between loses only the stamp of a copy, which
    /// deadens no id anyway.
    fn sweep_tree(
        &mut self,
        dir: &Path,
        kind: &str,
        stamp_prefix: &str,
        keep: &dyn Fn(&Self, &Path) -> Option<&'static str>,
        stop: &dyn Fn() -> bool,
        report: &mut SweepReport,
    ) -> Result<(), StoreV2Error> {
        if report.stopped {
            return Ok(());
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(()); // nothing tombstoned under this tree
        };
        // Deterministic order: sweep bookkeeping must be repeatable, not a
        // directory-listing artifact.
        let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
        paths.sort();
        let mut swept_here = false;
        for path in paths {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            let ext = path.extension().and_then(|ext| ext.to_str());
            if ext != Some("sj") && ext != Some(flac::FLAC_EXT) {
                // Not tombstoned audio (hand-dropped junk, a stray file):
                // never-delete-until-swept cuts both ways — the sweep
                // only removes what the tombstone semantics cover.
                report
                    .retained
                    .push((name, "not a .sj journal or .flac audio".to_string()));
                continue;
            }
            if !path.is_file() {
                // A directory (or anything unlink cannot remove as a
                // file) named like a journal must never be stamped swept:
                // the tombstone would permanently deaden that id —
                // reconcile's dead set would suppress recovery and
                // adoption under it forever — while the entry itself
                // survives every later sweep. Retain it unstamped for a
                // human to look at.
                report
                    .retained
                    .push((name, "not a regular file".to_string()));
                continue;
            }
            #[cfg(test)]
            if let Some(hook) = self.before_sweep_lock.take() {
                (hook.0)();
            }
            // Dropped without a commit, the transaction rolls back.
            let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
            if stop() {
                report.stopped = true;
                break;
            }
            if let Some(reason) = keep(self, &path) {
                report.retained.push((name, reason.to_string()));
                continue;
            }
            // The proof reads audio: a take may have started meanwhile.
            if stop() {
                report.stopped = true;
                break;
            }
            let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            let id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default()
                .to_string();
            // 1. Stamp first: the tombstone outlives the bytes, and a
            //    crash after this commit but before the unlink leaves a
            //    dead id whose file the next sweep re-attempts. Created if
            //    the delete never wrote one — the file under a tombstone
            //    tree is itself the deliberate-delete evidence.
            self.conn.execute(
                "INSERT INTO tombstones(id, kind, deleted_utc, retention)
                 VALUES (?1, ?2, ?3, 'swept')
                 ON CONFLICT(id) DO UPDATE SET retention = 'swept'",
                params![format!("{stamp_prefix}{id}"), kind, now_iso()],
            )?;
            // 2. Only now may the bytes go.
            let removed = if stamp_prefix.is_empty() {
                tx.commit()?;
                std::fs::remove_file(&path)
            } else {
                let removed = std::fs::remove_file(&path);
                if removed.is_ok() {
                    tx.commit()?;
                }
                removed
            };
            match removed {
                Ok(()) => {
                    swept_here = true;
                    report.swept.push(SweptFile {
                        id: id.clone(),
                        kind: kind.to_string(),
                        bytes,
                    });
                    report.swept_bytes += bytes;
                }
                Err(err) => report.retained.push((name, err.to_string())),
            }
        }
        if swept_here {
            sync_dir(dir)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Lossless at-rest audio (#342).
    // ------------------------------------------------------------------

    /// Pin `id`'s audio for this instance (#342): until the matching
    /// [`Self::unpin_audio`], it is neither compressed nor retired. Pins
    /// nest.
    pub fn pin_audio(&mut self, id: &str) {
        *self.audio_pins.entry(id.to_string()).or_default() += 1;
    }

    /// Count a failed compression of `id` (its prepare or its commit); at
    /// [`COMPRESSION_ATTEMPTS`] failures [`Self::compression_candidates`]
    /// stops offering it until the store is opened again.
    pub fn note_compression_failure(&mut self, id: &str) {
        *self.compression_failures.entry(id.to_string()).or_default() += 1;
    }

    /// Release one [`Self::pin_audio`] (a release without a pin does
    /// nothing).
    pub fn unpin_audio(&mut self, id: &str) {
        if let Some(count) = self.audio_pins.get_mut(id) {
            *count -= 1;
            if *count == 0 {
                self.audio_pins.remove(id);
            }
        }
    }

    /// Whether `id`'s audio is in use: pinned by a caller of this
    /// instance, a recognition attempt on it is in flight (#356), or it
    /// still waits to be transcribed or a live process holds it (#220).
    /// Its audio is never compressed or retired under it.
    fn audio_in_use(&self, id: &str) -> Result<bool, StoreV2Error> {
        if self.audio_pins.contains_key(id) || self.audio_held(id)? {
            return Ok(true);
        }
        Ok(self
            .conn
            .query_row(
                "SELECT 1 FROM recognition_attempts WHERE capture_id = ?1 AND status = 'started'
                 UNION ALL SELECT 1 FROM transcription_intents WHERE capture_id = ?1
                 LIMIT 1",
                params![id],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    /// Up to `limit` committed takes still kept as journals that may be
    /// compressed now, oldest first: not pinned by an in-flight attempt,
    /// not removed by retention, and long enough to encode. Takes still
    /// recording live in `staging/` (or the recorder's own journal tree)
    /// and never appear here.
    pub fn compression_candidates(
        &self,
        limit: usize,
    ) -> Result<Vec<CompressionJob>, StoreV2Error> {
        let rows: Vec<(String, u32, u64)> = {
            let mut stmt = self.conn.prepare(
                "SELECT id, actual_rate, frame_count FROM captures c
                 WHERE NOT EXISTS (SELECT 1 FROM recognition_attempts a
                                   WHERE a.capture_id = c.id AND a.status = 'started')
                   AND NOT EXISTS (SELECT 1 FROM transcription_intents i
                                   WHERE i.capture_id = c.id)
                   AND NOT EXISTS (SELECT 1 FROM tombstones t
                                   WHERE t.id = ?1 || c.id AND t.kind = 'audio')
                 ORDER BY created_utc, id",
            )?;
            let rows = stmt.query_map(params![AUDIO_TOMBSTONE_PREFIX], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)? as u32,
                    row.get::<_, i64>(2)? as u64,
                ))
            })?;
            rows.collect::<Result<_, _>>()?
        };
        let mut jobs = Vec::new();
        for (id, rate, frames) in rows {
            if jobs.len() >= limit {
                break;
            }
            if !is_safe_path_component(&id)
                || rate == 0
                || self.audio_pins.contains_key(&id)
                || self.audio_held(&id)?
            {
                continue;
            }
            if self.compression_failures.get(&id).copied().unwrap_or(0) >= COMPRESSION_ATTEMPTS {
                continue;
            }
            // Shorter than FLAC's minimum block at 16 kHz: stays a journal.
            if frames.saturating_mul(u64::from(STARLING_SAMPLE_RATE)) / u64::from(rate)
                < flac::MIN_SAMPLES as u64
            {
                continue;
            }
            let journal = self.audio_path(&id);
            if journal.exists() {
                jobs.push(CompressionJob { id, journal });
            }
        }
        Ok(jobs)
    }

    /// Publish a [`prepare_compression`] result (#342): under the store,
    /// re-check that the take is still live, unpinned and still a
    /// journal; rename the verified FLAC onto `audio/<id>.flac` and fsync
    /// the directory; only then remove the journal. A crash at any point
    /// leaves the journal, or the complete FLAC, or both (reconcile
    /// finishes that last shape) — never neither.
    pub fn commit_compression(
        &mut self,
        prepared: PreparedCompression,
    ) -> Result<CompressionOutcome, StoreV2Error> {
        self.commit_compression_unless(prepared, || false)
    }

    /// [`Self::commit_compression`] that publishes nothing when `stop`
    /// says so once the write lock is held (a take started recording
    /// while the publish waited for it): skipped, its temporary removed.
    pub fn commit_compression_unless(
        &mut self,
        prepared: PreparedCompression,
        stop: impl Fn() -> bool,
    ) -> Result<CompressionOutcome, StoreV2Error> {
        let skip = |reason: &str| {
            let _ = std::fs::remove_file(&prepared.temp);
            Ok(CompressionOutcome::Skipped(reason.to_string()))
        };
        let id = prepared.id.as_str();
        // The checks and the publish hold the database's write lock, so
        // no other connection can retire, delete or start a transcription
        // of the take in between.
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        if stop() {
            return skip("a take started recording");
        }
        if self.get_capture(id)?.is_none() {
            return skip("the take was deleted while it was being compressed");
        }
        if self.audio_retired_utc(id)?.is_some() {
            return skip("the retention policy removed the take's audio meanwhile");
        }
        if self.audio_in_use(id)? {
            return skip("a transcription of the take started meanwhile");
        }
        let journal = self.audio_path(id);
        if journal != prepared.journal || !journal.exists() {
            return skip("the take's journal is no longer in place");
        }
        let target = self.flac_path(id);
        std::fs::rename(&prepared.temp, &target).map_err(|err| {
            let _ = std::fs::remove_file(&prepared.temp);
            StoreV2Error::Io(io::Error::new(
                err.kind(),
                format!("publishing compressed audio {target:?}: {err}"),
            ))
        })?;
        sync_dir(&self.root.join(AUDIO_DIR))?;
        // The FLAC is durable and verified: the journal may go, still
        // under the lock. Nothing in the database records the change, so
        // a retention run never learns of it from a commit: it measures
        // the files under the lock instead.
        match std::fs::remove_file(&journal) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
        sync_dir(&self.root.join(AUDIO_DIR))?;
        tx.commit()?;
        Ok(CompressionOutcome::Compressed {
            journal_bytes: prepared.journal_bytes,
            flac_bytes: prepared.flac_bytes,
        })
    }

    /// [`Self::compression_candidates`] + [`prepare_compression`] +
    /// [`Self::commit_compression`] for one take, holding the store
    /// throughout. Callers sharing the store behind a lock run the
    /// encode without it instead.
    pub fn compress_audio(&mut self, id: &str) -> Result<CompressionOutcome, StoreV2Error> {
        validate_capture_id(id)?;
        let Some(job) = self
            .compression_candidates(usize::MAX)?
            .into_iter()
            .find(|job| job.id == id)
        else {
            return Ok(CompressionOutcome::Skipped(
                "not a journal that can be compressed now".to_string(),
            ));
        };
        let prepared = prepare_compression(&job)?;
        self.commit_compression(prepared)
    }

    /// Reconcile's half of the compression protocol: both `<id>.sj` and
    /// `<id>.flac` exist, so a compression stopped after publishing the
    /// FLAC. When the FLAC decodes to exactly the journal's request
    /// PCM16 the journal is removed (the compression completes); anything
    /// else keeps both — the journal stays authoritative — and is
    /// reported. Nothing here ever removes the FLAC, so a peer that is
    /// mid-commit can never be left with neither file.
    fn finish_interrupted_compression(
        &mut self,
        id: &str,
        report: &mut ReconciliationReport,
    ) -> Result<(), StoreV2Error> {
        let journal = self.audio_path(id);
        let verified = read_journal(&journal)
            .map_err(|err| err.to_string())
            .and_then(|parsed| {
                request_pcm16(&parsed.samples, parsed.sample_rate).map_err(|err| err.to_string())
            })
            .and_then(|expected| {
                let file = File::open(self.flac_path(id)).map_err(|err| err.to_string())?;
                let decoded =
                    flac::decode(io::BufReader::new(file)).map_err(|err| err.to_string())?;
                if decoded == expected {
                    Ok(())
                } else {
                    Err("the compressed copy differs from the journal".to_string())
                }
            });
        match verified {
            Ok(()) => {
                // The compressor may have died before its directory fsync:
                // make the FLAC's name durable before the journal goes.
                sync_dir(&self.root.join(AUDIO_DIR))?;
                match std::fs::remove_file(&journal) {
                    Ok(()) => {}
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) => return Err(err.into()),
                }
                sync_dir(&self.root.join(AUDIO_DIR))?;
                report.completed_compressions.push(id.to_string());
            }
            Err(reason) => report.unreadable.push((
                id.to_string(),
                format!("interrupted compression left both files; the journal is kept: {reason}"),
            )),
        }
        Ok(())
    }

    /// Remove FLAC temporaries under `audio/` older than
    /// [`FLAC_TEMP_GRACE`] (best-effort: scratch, never evidence).
    fn remove_stale_flac_temps(&self) {
        let dir = self.root.join(AUDIO_DIR);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut removed = false;
        for path in entries.flatten().map(|entry| entry.path()) {
            if path.extension().and_then(|ext| ext.to_str()) != Some(FLAC_TEMP_EXT) {
                continue;
            }
            let stale = std::fs::metadata(&path)
                .and_then(|meta| meta.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age >= FLAC_TEMP_GRACE);
            if stale && std::fs::remove_file(&path).is_ok() {
                removed = true;
            }
        }
        if removed {
            let _ = sync_dir(&dir);
        }
    }

    // ------------------------------------------------------------------
    // Retention policy (#342): opt-in age and size limits per class.
    // ------------------------------------------------------------------

    /// Move a take into another retention class (e.g. [`ARCHIVAL_CLASS`]),
    /// so that class's limits govern its audio from the next policy run.
    pub fn set_retention_class(&mut self, id: &str, class: &str) -> Result<(), StoreV2Error> {
        validate_capture_id(id)?;
        if class.trim().is_empty() {
            return Err(StoreV2Error::Invalid("retention class must not be empty".to_string()));
        }
        let changed = self.conn.execute(
            "UPDATE captures SET retention_class = ?2 WHERE id = ?1",
            params![id, class],
        )?;
        if changed == 0 {
            return Err(StoreV2Error::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// Bytes `id`'s audio occupies under `audio/`. While a compression
    /// has both its journal and its FLAC on disk, the smaller counts:
    /// one of the two is about to go, and over-counting would remove
    /// audio a limit does not require removing.
    fn audio_bytes(&self, id: &str) -> u64 {
        let size = |path: PathBuf| std::fs::metadata(path).ok().map(|meta| meta.len());
        match (size(self.audio_path(id)), size(self.flac_path(id))) {
            (Some(journal), Some(flac)) => journal.min(flac),
            (journal, flac) => journal.or(flac).unwrap_or(0),
        }
    }

    /// Apply the user's retention limits as of `now` — the one code path
    /// that removes a committed take's audio without a delete. Off unless
    /// [`RetentionPolicy::is_active`]; nothing calls it implicitly.
    ///
    /// Per class, takes are walked newest first. A take is due when it is
    /// older than the class's age limit, or when it and every newer take
    /// of the class together exceed the class's size limit. A due take is
    /// held — and reported with the reason — when it is younger than the
    /// policy's grace, pinned by an in-flight attempt, has never been
    /// transcribed, or is referenced by a document revision or a
    /// correction record (unless the policy includes referenced audio).
    /// A held take still counts toward its class's size; whatever the
    /// limit could not reach is reported in `over_limit`.
    ///
    /// Only the audio goes: the row, attempts, revisions and correction
    /// records stay, and the take lists as [`AudioAtRest::Retired`].
    /// Ordering is stamp, then unlink (the sweep's discipline): the
    /// `audio:<id>` tombstone is committed first, so a crash before the
    /// unlink leaves a take that already reads as retired and whose files
    /// reconcile removes. Takes still recording are not rows yet and are
    /// never considered.
    pub fn apply_retention_policy(
        &mut self,
        policy: &RetentionPolicy,
        now: time::OffsetDateTime,
    ) -> Result<RetentionReport, StoreV2Error> {
        self.apply_live_retention_policy(|| policy.clone(), now)
    }

    /// [`Self::apply_retention_policy`] for a policy the user may change
    /// while the run goes: `current` is read at the start and again
    /// under the write lock before each removal; when it no longer
    /// matches, the run stops ([`RetentionReport::policy_changed`]).
    /// What `current` returns before a removal is kept until that
    /// removal's stamp commits, so a caller whose value holds a lock on
    /// its settings makes a change to them wait for the removal, and the
    /// next check sees the change.
    pub fn apply_live_retention_policy<P: std::borrow::Borrow<RetentionPolicy>>(
        &mut self,
        current: impl Fn() -> P,
        now: time::OffsetDateTime,
    ) -> Result<RetentionReport, StoreV2Error> {
        self.apply_retention_policy_until(current, || false, now)
    }

    /// [`Self::apply_live_retention_policy`] that also asks `stop` under
    /// the write lock before each removal: when it says yes (the app
    /// started recording a take), the run ends there
    /// ([`RetentionReport::stopped`]).
    pub fn apply_retention_policy_until<P: std::borrow::Borrow<RetentionPolicy>>(
        &mut self,
        current: impl Fn() -> P,
        stop: impl Fn() -> bool,
        now: time::OffsetDateTime,
    ) -> Result<RetentionReport, StoreV2Error> {
        let mut report = RetentionReport::default();
        let policy = &current().borrow().clone();
        if !policy.is_active() {
            return Ok(report);
        }
        let grace_cutoff = iso_utc(now - policy.grace);
        for (class, limits) in &policy.limits {
            if !limits.is_active() {
                continue;
            }
            let age_cutoff = limits
                .max_age_days
                .map(|days| iso_utc(now - time::Duration::days(i64::from(days))));
            // Read before the walk: a later change means another
            // connection committed since, and the walk's view is stale.
            let mut seen_version = self.data_version()?;
            let rows: Vec<(String, String)> = {
                let mut stmt = self.conn.prepare(
                    "SELECT id, created_utc FROM captures WHERE retention_class = ?1
                     ORDER BY created_utc DESC, id DESC",
                )?;
                let rows = stmt.query_map(params![class], |row| Ok((row.get(0)?, row.get(1)?)))?;
                rows.collect::<Result<_, _>>()?
            };
            let mut kept_bytes = 0u64;
            // The takes `kept_bytes` counts, newest first.
            let mut counted: Vec<(String, u64)> = Vec::new();
            for (id, created_utc) in rows {
                if !is_safe_path_component(&id) || self.audio_retired_utc(&id)?.is_some() {
                    continue;
                }
                let bytes = self.audio_bytes(&id);
                if bytes == 0 {
                    continue;
                }
                kept_bytes += bytes;
                counted.push((id.clone(), bytes));
                let due = if age_cutoff.as_ref().is_some_and(|cutoff| created_utc < *cutoff) {
                    Some(RetireReason::Age)
                } else if limits.max_total_bytes.is_some_and(|max| kept_bytes > max) {
                    Some(RetireReason::Size)
                } else {
                    None
                };
                let Some(reason) = due else {
                    continue;
                };
                #[cfg(test)]
                if let Some(hook) = self.before_retention_lock.take() {
                    (hook.0)();
                }
                // The hold check and the stamp share the database's write
                // lock: another connection cannot start a transcription,
                // add a revision or a correction record in between.
                let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
                let live = current();
                if live.borrow() != policy {
                    report.policy_changed = true;
                    return Ok(report);
                }
                if stop() {
                    report.stopped = true;
                    return Ok(report);
                }
                // Another connection committed since the walk looked: takes
                // may have moved to another class, been deleted or retired.
                // Drop what this class no longer holds before deciding.
                let version = self.data_version()?;
                if version != seen_version {
                    seen_version = version;
                    let mut still_held = Vec::with_capacity(counted.len());
                    for (counted_id, bytes) in counted.drain(..) {
                        if self.in_retention_class(&counted_id, class)?
                            && self.audio_retired_utc(&counted_id)?.is_none()
                        {
                            still_held.push((counted_id, bytes));
                        }
                    }
                    counted = still_held;
                }
                // Sizes are measured again, now, under the lock. Files
                // change without a commit to announce it: a compression
                // publishes its FLAC and unlinks the journal, and a crash
                // or failed commit after that leaves the files changed and
                // the database not. A size counted during the walk may be
                // too large, and removing by it removes audio the limit
                // does not require removing. An age limit needs only the
                // candidate's own size.
                for (counted_id, bytes) in counted.iter_mut() {
                    if reason == RetireReason::Size || *counted_id == id {
                        *bytes = self.audio_bytes(counted_id);
                    }
                }
                kept_bytes = counted.iter().map(|(_, bytes)| bytes).sum();
                // Moved to another class, deleted or retired since the walk
                // read it: those limits decide, on their own walk. Under a
                // size limit the recount may also show the class fits now.
                let candidate = counted
                    .iter()
                    .position(|(counted_id, _)| *counted_id == id);
                let Some(index) = candidate else {
                    continue;
                };
                if !self.in_retention_class(&id, class)? || self.audio_retired_utc(&id)?.is_some()
                {
                    let (_, gone) = counted.remove(index);
                    kept_bytes -= gone;
                    continue;
                }
                let bytes = counted[index].1;
                if bytes == 0 {
                    continue;
                }
                let still_due = match reason {
                    RetireReason::Age => true,
                    RetireReason::Size => {
                        limits.max_total_bytes.is_some_and(|max| kept_bytes > max)
                    }
                };
                if !still_due {
                    continue;
                }
                if let Some(hold) = self.retention_hold(&id, &created_utc, &grace_cutoff, policy)? {
                    drop(tx);
                    report.held.push(HeldAudio {
                        id,
                        class: class.clone(),
                        bytes,
                        reason: hold,
                    });
                    continue;
                }
                self.stamp_audio_retired(&id, now)?;
                tx.commit()?;
                drop(live);
                self.unlink_audio(&id)?;
                kept_bytes -= bytes;
                counted.retain(|(counted_id, _)| *counted_id != id);
                report.retired_bytes += bytes;
                report.retired.push(RetiredAudio {
                    id,
                    class: class.clone(),
                    bytes,
                    reason,
                });
            }
            if let Some(max) = limits.max_total_bytes {
                let kept_bytes: u64 = counted.iter().map(|(id, _)| self.audio_bytes(id)).sum();
                if kept_bytes > max {
                    report.over_limit.push((class.clone(), kept_bytes - max));
                }
            }
        }
        Ok(report)
    }

    /// SQLite's `data_version`: it changes when another connection
    /// commits, never for this connection's own writes.
    fn data_version(&self) -> Result<i64, StoreV2Error> {
        Ok(self
            .conn
            .query_row("PRAGMA data_version", [], |row| row.get(0))?)
    }

    fn in_retention_class(&self, id: &str, class: &str) -> Result<bool, StoreV2Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT 1 FROM captures WHERE id = ?1 AND retention_class = ?2",
                params![id, class],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    /// [`Self::apply_retention_policy_until`] as of the current time.
    pub fn apply_retention_policy_now<P: std::borrow::Borrow<RetentionPolicy>>(
        &mut self,
        current: impl Fn() -> P,
        stop: impl Fn() -> bool,
    ) -> Result<RetentionReport, StoreV2Error> {
        self.apply_retention_policy_until(current, stop, time::OffsetDateTime::now_utc())
    }

    /// Why a due take keeps its audio, if it does.
    fn retention_hold(
        &self,
        id: &str,
        created_utc: &str,
        grace_cutoff: &str,
        policy: &RetentionPolicy,
    ) -> Result<Option<HoldReason>, StoreV2Error> {
        if created_utc >= grace_cutoff {
            return Ok(Some(HoldReason::Recent));
        }
        if self.audio_in_use(id)? {
            return Ok(Some(HoldReason::InUse));
        }
        let transcribed = self
            .conn
            .query_row(
                "SELECT 1 FROM recognition_attempts
                 WHERE capture_id = ?1 AND status = 'completed' AND partial_or_final = 'final'
                 LIMIT 1",
                params![id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !transcribed {
            return Ok(Some(HoldReason::Untranscribed));
        }
        if policy.include_referenced {
            return Ok(None);
        }
        // A revision references the take when it lives in the take's own
        // document (the app's processing document is keyed by the capture
        // id) or names one of its attempts in its sources.
        let revisions: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM revisions r
             WHERE r.doc_id = ?1
                OR EXISTS (SELECT 1 FROM recognition_attempts a
                           WHERE a.capture_id = ?1
                             AND instr(r.sources_json, '\"' || a.id || '\"') > 0)",
            params![id],
            |row| row.get(0),
        )?;
        let corrections: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM correction_records WHERE capture_id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok((revisions > 0 || corrections > 0).then_some(HoldReason::Referenced {
            revisions: revisions as u32,
            corrections: corrections as u32,
        }))
    }

    /// Stamp `id`'s audio as retired; [`Self::unlink_audio`] follows.
    fn stamp_audio_retired(&self, id: &str, now: time::OffsetDateTime) -> Result<(), StoreV2Error> {
        self.conn.execute(
            "INSERT INTO tombstones(id, kind, deleted_utc, retention)
             VALUES (?1, 'audio', ?2, 'swept')
             ON CONFLICT(id) DO NOTHING",
            params![format!("{AUDIO_TOMBSTONE_PREFIX}{id}"), iso_utc(now)],
        )?;
        Ok(())
    }

    /// Unlink whatever audio `id` has under `audio/` (idempotent).
    fn unlink_audio(&self, id: &str) -> Result<(), StoreV2Error> {
        for path in [self.audio_path(id), self.flac_path(id)] {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(err.into()),
            }
        }
        sync_dir(&self.root.join(AUDIO_DIR))?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Recovery (§4 reconciliation).
    // ------------------------------------------------------------------

    /// Reconciles journal files and metadata rows after a crash. Both
    /// sides are inspected independently; everything is idempotent, so a
    /// crash during reconciliation is repaired by the next run. Run by the
    /// root's owner before it records anything: every staging journal is
    /// a dead run's.
    pub fn reconcile(&mut self) -> Result<ReconciliationReport, StoreV2Error> {
        // Recovery's notes (seals, intents) must be on disk before the
        // files they describe change: never inside a caller's transaction.
        self.refuse_open_transaction("reconcile the store")?;
        let mut report = ReconciliationReport::default();

        // Tombstoned ids outrank everything (R21): the DB row and any file
        // under quarantine/ both mean "deliberately deleted".
        let mut dead: std::collections::HashSet<String> =
            audio_ids_in(&self.root.join(QUARANTINE_DIR)).into_iter().collect();
        {
            let mut stmt = self.conn.prepare("SELECT id FROM tombstones")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            for row in rows {
                dead.insert(row?);
            }
        }
        // Audio the retention policy removed (#342): the rows live on.
        let retired: HashSet<String> = dead
            .iter()
            .filter_map(|id| id.strip_prefix(AUDIO_TOMBSTONE_PREFIX))
            .map(str::to_string)
            .collect();

        // --- staging side ---
        for id in journal_ids_in(&self.root.join(STAGING_DIR)) {
            if !is_safe_path_component(&id) {
                report.unreadable.push((
                    id,
                    "journal id is not a safe path component".to_string(),
                ));
                continue;
            }
            if dead.contains(&id) {
                // A delete raced with a crash before/after promote: the
                // tombstone wins, complete the removal.
                self.complete_tombstoned(&id, &mut report)?;
                continue;
            }
            let path = self.staging_path(&id);
            let parsed = match read_journal(&path) {
                Ok(parsed) => parsed,
                Err(err) => {
                    report.unreadable.push((id, err.to_string()));
                    continue;
                }
            };
            if parsed.samples.is_empty() {
                // Never reached its first boundary: no audio to recover.
                report.empty_journals.push(id);
                continue;
            }
            // A journal an earlier pass sealed and died with reads as
            // finished now: its noted seal says it was cut short, and how
            // much of its tail that pass discarded.
            let earlier_seal = self.recovery_seal_for_capture(&id)?;
            let torn_tail_bytes = earlier_seal
                .map_or(parsed.torn_tail_bytes, |earlier| earlier.max(parsed.torn_tail_bytes));
            let cut = !parsed.finalized || parsed.torn_tail_bytes > 0 || earlier_seal.is_some();
            let sample_rate = parsed.sample_rate;
            let count = parsed.samples.len() as u64;
            let hash = format!("{:016x}", samples_hash(&parsed.samples));

            // Seal the verified prefix (§4: truncate + gap flag) and
            // promote. A journal already finalized (by its recorder, or by
            // a pass that died after sealing it) is left as it is — a
            // second trailer would make it unreadable — so a crash
            // mid-recovery re-runs cleanly; the seal is noted first, so
            // the re-run still knows the take was cut short (#220).
            if cut {
                self.note_recovery_seal(&id, &id, torn_tail_bytes)?;
            }
            #[cfg(test)]
            self.crash_point(RecoveryStep::SealNoted)?;
            if !parsed.finalized || parsed.torn_tail_bytes > 0 {
                seal_recovered_journal(&path, &parsed)?;
            }
            #[cfg(test)]
            self.crash_point(RecoveryStep::Sealed)?;
            self.promote_from_staging(&id)?;
            #[cfg(test)]
            self.crash_point(RecoveryStep::Moved)?;
            if !cut {
                report.promoted_finalized.push(id.clone());
            }

            let note = if torn_tail_bytes > 0 {
                format!(
                    "Recovered from an interrupted take: the last {torn_tail_bytes} bytes of \
                     the journal were an unfinished write and were discarded (gap flagged, \
                     never joined)."
                )
            } else if !cut {
                "Recovered from a take that finished cleanly but was never promoted or \
                 committed (crash between finalize and the metadata commit)."
                    .to_string()
            } else {
                "Recovered from an interrupted take that was never finalized.".to_string()
            };
            let record = CaptureRecord {
                id: id.clone(),
                created_utc: now_iso(),
                tz: local_tz_label(),
                device: String::new(),
                actual_rate: sample_rate,
                policy: "default".to_string(),
                frame_count: count,
                ack_sample_index: count,
                journal_hash: hash,
                status: CaptureStatus::Interrupted,
                retention_class: "standard".to_string(),
                extra_json: None,
                // The take's marker never reached disk: exclude it.
                secure_field: true,
            };
            match self.get_capture(&id)? {
                Some(existing) => {
                    // A row already exists (sealed on an earlier run that
                    // crashed before removing staging — impossible after a
                    // rename, defense for copied trees): keep it, only
                    // ensure the note is present.
                    if existing.status != CaptureStatus::Interrupted {
                        self.update_capture_status(
                            &id,
                            CaptureStatus::Interrupted,
                            Some(&note),
                        )?;
                    }
                    self.clear_recovery_seals(&id)?;
                }
                None => {
                    let record = CaptureRecord {
                        extra_json: Some(merge_extra_note(None, &note)),
                        ..record
                    };
                    // A complete take whose commit was to carry the intent
                    // to transcribe it still does (#220).
                    let transcribe = !cut && self.pending_intent(&id)?;
                    self.commit_capture_superseding(&record, None, transcribe, true)?;
                }
            }
            report.recovered_torn.push(RecoveredTake {
                id,
                torn_tail_bytes,
            });
        }

        // --- audio side ---
        // Abandoned FLAC encodes (#342): the journal beside each is
        // intact, so the temporary is only scratch.
        self.remove_stale_flac_temps();
        let mut awaiting_replacement = Vec::new();
        for id in audio_ids_in(&self.root.join(AUDIO_DIR)) {
            if !is_safe_path_component(&id) {
                report.unreadable.push((
                    id,
                    "journal id is not a safe path component".to_string(),
                ));
                continue;
            }
            if dead.contains(&id) {
                self.complete_tombstoned(&id, &mut report)?;
                continue;
            }
            if retired.contains(&id) {
                // The policy stamped the removal and crashed before the
                // unlink: finish it.
                self.unlink_audio(&id)?;
                report.completed_retirements.push(id);
                continue;
            }
            let path = self.audio_path(&id);
            match self.get_capture(&id)? {
                // Row with finalized audio: healthy, nothing to do.
                Some(_) => {
                    if path.exists() && self.flac_path(&id).exists() {
                        // A compression stopped between publishing the
                        // FLAC and removing the journal.
                        self.finish_interrupted_compression(&id, &mut report)?;
                    }
                }
                None if !path.exists() => {
                    // Compressed audio is only ever written for a
                    // committed row; without one there is nothing to
                    // adopt it into. Kept for a human to look at.
                    report.unreadable.push((
                        id,
                        "compressed audio without a library row; kept in place".to_string(),
                    ));
                }
                None => {
                    // Finalized audio with no row → orphan session:
                    // interrupted status, linked to the audio.
                    if self.reconcile_orphan_audio(&id, true, &mut report)? {
                        awaiting_replacement.push(id);
                    }
                }
            }
        }
        // Recorder journals whose replacement was itself among the orphans
        // above: committed by now, so the proof can be read.
        for id in awaiting_replacement {
            self.reconcile_orphan_audio(&id, false, &mut report)?;
        }

        // --- rows whose audio is gone ---
        let mut dead_rows = Vec::new();
        let mut missing_audio = Vec::new();
        {
            let mut stmt = self.conn.prepare("SELECT id, status FROM captures")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (id, status) = row?;
                if dead.contains(&id) {
                    // A delete that crashed between the quarantine rename
                    // and the transaction: complete it.
                    dead_rows.push(id);
                    continue;
                }
                if !is_safe_path_component(&id) {
                    // Not tombstoned and not touchable: leave the row
                    // as-is rather than aborting the reconciliation.
                    continue;
                }
                if !self.has_audio(&id) && !retired.contains(&id) && status != "interrupted" {
                    missing_audio.push(id);
                }
            }
        }
        for id in dead_rows {
            self.complete_tombstoned(&id, &mut report)?;
        }
        for id in missing_audio {
            self.update_capture_status(
                &id,
                CaptureStatus::Interrupted,
                Some(
                    "The take's audio journal is missing; the row is kept and \
                     marked interrupted.",
                ),
            )?;
            report.marked_interrupted.push(id);
        }

        // A noted intent whose take is stored, or whose audio exists
        // nowhere (its save failed before the move), is done with.
        let pending: Vec<String> = {
            let mut stmt = self.conn.prepare("SELECT capture_id FROM pending_intents")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.collect::<Result<_, _>>()?
        };
        for id in pending {
            let stored = self.get_capture(&id)?.is_some();
            let audio = is_safe_path_component(&id)
                && (self.audio_path(&id).exists() || self.staging_path(&id).exists());
            if stored || !audio {
                self.conn.execute(
                    "DELETE FROM pending_intents WHERE capture_id = ?1",
                    params![id],
                )?;
            }
        }

        Ok(report)
    }

    /// Reconcile's answer to finalized audio `id` in `audio/` without a
    /// row: an interrupted row linked to it — unless it is a recorder
    /// journal whose adoption failed to commit after the take was stored
    /// from memory in its place, and that take's audio is read back
    /// holding every sample of it (#356): then it is moved to
    /// `journals/superseded/`. A replacement that is itself still rowless here is
    /// waited for when `may_wait` (`Ok(true)`); otherwise unproven means
    /// a row of its own.
    fn reconcile_orphan_audio(
        &mut self,
        id: &str,
        may_wait: bool,
        report: &mut ReconciliationReport,
    ) -> Result<bool, StoreV2Error> {
        let id = id.to_string();
        let path = self.audio_path(&id);
        let parsed = match read_journal(&path) {
            Ok(parsed) => parsed,
            Err(err) => {
                report.unreadable.push((id, err.to_string()));
                return Ok(false);
            }
        };
        if parsed.samples.is_empty() {
            report.empty_journals.push(id);
            return Ok(false);
        }
        let aside = match self.journal_copy(&id, &parsed.samples, parsed.sample_rate)? {
            JournalCopy::Unproven => None,
            JournalCopy::Pending if may_wait => return Ok(true),
            JournalCopy::Pending => None,
            JournalCopy::Stored => Some(SUPERSEDED_SUBDIR),
            JournalCopy::Deleted => Some(DELETED_SUBDIR),
        };
        if let Some(aside) = aside {
            // Kept with the recorder's other journals of stored or
            // deleted takes, not offered as a second copy.
            drop(parsed);
            move_journal_aside(&path, &self.root.join("journals").join(aside))?;
            self.clear_recovery_seals(&id)?;
            report.superseded_journals.push(id);
            return Ok(false);
        }
        // A cut-short journal recovery sealed and moved here before a
        // crash cut off its commit reads as finished: its noted seal says
        // otherwise (#220).
        let earlier_seal = self.recovery_seal_for_capture(&id)?;
        let torn_tail_bytes = earlier_seal
            .map_or(parsed.torn_tail_bytes, |earlier| earlier.max(parsed.torn_tail_bytes));
        let cut = !parsed.finalized || parsed.torn_tail_bytes > 0 || earlier_seal.is_some();
        let note = if !cut {
            "Recovered from a take that finished cleanly but was never \
             committed to the library (crash between finalize and the \
             metadata commit)."
                .to_string()
        } else if torn_tail_bytes > 0 {
            format!(
                "Recovered from an orphaned journal with a torn tail of {torn_tail_bytes} \
                 bytes (gap flagged, never joined)."
            )
        } else if earlier_seal.is_some() {
            "Recovered from an interrupted take that was never finalized; audio up to \
             its last confirmed boundary was kept."
                .to_string()
        } else {
            "Recovered from an orphaned journal.".to_string()
        };
        let record = CaptureRecord {
            id: id.clone(),
            created_utc: now_iso(),
            tz: local_tz_label(),
            device: String::new(),
            actual_rate: parsed.sample_rate,
            policy: "default".to_string(),
            frame_count: parsed.samples.len() as u64,
            ack_sample_index: parsed.samples.len() as u64,
            journal_hash: format!("{:016x}", samples_hash(&parsed.samples)),
            status: CaptureStatus::Interrupted,
            retention_class: "standard".to_string(),
            extra_json: Some(merge_extra_note(None, &note)),
            // The take's marker never reached disk: exclude it.
            secure_field: true,
        };
        // A complete take whose commit was to carry the intent to
        // transcribe it still does (#220): the row was written before its
        // audio moved here.
        let transcribe = !cut && self.pending_intent(&id)?;
        self.commit_capture_superseding(&record, None, transcribe, true)?;
        report.orphan_sessions.push(id);
        Ok(false)
    }

    /// Finish a tombstoned id: remove any row, move any live journal (in
    /// `audio/` or `staging/`) into `quarantine/`. The R21 "retry
    /// completes the delete" path.
    fn complete_tombstoned(
        &mut self,
        id: &str,
        report: &mut ReconciliationReport,
    ) -> Result<(), StoreV2Error> {
        self.quarantine_audio(id)?;
        let staging = self.staging_path(id);
        if staging.exists() {
            std::fs::create_dir_all(self.root.join(QUARANTINE_DIR))?;
            let destination = self.quarantine_path(id);
            let _ = std::fs::remove_file(&destination);
            std::fs::rename(&staging, &destination)?;
            sync_dir(&self.root.join(STAGING_DIR))?;
            sync_dir(&self.root.join(QUARANTINE_DIR))?;
        }
        if self.get_capture(id)?.is_some() {
            // The capture's attempt ids go first: the DELETE cascades the
            // attempt rows away, and their markers must not outlive them
            // (#213 review — every row-removal path releases markers).
            let attempt_ids: Vec<String> = {
                let mut stmt = self
                    .conn
                    .prepare("SELECT id FROM recognition_attempts WHERE capture_id = ?1")?;
                let rows = stmt.query_map(params![id], |row| row.get::<_, String>(0))?;
                rows.collect::<Result<Vec<_>, _>>()?
            };
            let tx = self.conn.transaction()?;
            tx.execute(
                "INSERT OR REPLACE INTO tombstones(id, kind, deleted_utc, retention)
                 VALUES (?1, 'capture', ?2, 'quarantined')",
                params![id, now_iso()],
            )?;
            tx.execute("DELETE FROM captures WHERE id = ?1", params![id])?;
            tx.commit()?;
            for attempt_id in &attempt_ids {
                self.release_attempt(attempt_id);
            }
        }
        report.completed_deletes.push(id.to_string());
        Ok(())
    }

    // ------------------------------------------------------------------
    // Daily-use operations.
    // ------------------------------------------------------------------

    /// Adopts a finalized (or faulted) capture journal written elsewhere —
    /// the live-capture path journals to the recorder's own tree and moves
    /// the evidence here instead of re-encoding it. The journal is read and
    /// verified first; a torn tail is sealed to its verified prefix (the
    /// discarded bytes become a gap note, never a silent join); only then
    /// is the file renamed into `audio/` (fsyncing both directories) and
    /// the `captures` row committed.
    ///
    /// # What can happen to the source
    ///
    /// The source is read, verified, and **sealed in place when its tail is
    /// torn or it never finalized** — the pre-adoption byte layout is *not*
    /// preserved. A failure after the seal but before the rename leaves a
    /// sealed-but-unadopted source at its original path: its verified
    /// samples are intact, and callers treat any adoption failure as "no
    /// adoption happened" (the app stores the take from its encoded WAV
    /// instead). A failure after the rename but before the commit leaves
    /// the journal in `audio/` with no `captures` row — repaired by the
    /// next [`Self::reconcile`] (an orphan session), not by restoring the
    /// source. The capture id is the journal's file stem, so an adopted
    /// take stays traceable to its origin.
    pub fn adopt_journal(
        &mut self,
        source: impl AsRef<Path>,
        note: Option<&str>,
    ) -> Result<CaptureRecord, StoreV2Error> {
        self.adopt_journal_transcribed(source, note, false)
    }

    /// [`Self::adopt_journal`], recording the intent to transcribe the
    /// take in the same commit when `transcribe` is set and the journal
    /// lands complete ([`TakeMeta::transcribe`]).
    pub fn adopt_journal_transcribed(
        &mut self,
        source: impl AsRef<Path>,
        note: Option<&str>,
        transcribe: bool,
    ) -> Result<CaptureRecord, StoreV2Error> {
        self.adopt_journal_with(source.as_ref(), None, transcribe, |facts| {
            let recovery_note = match (facts.torn_tail_bytes, facts.was_finalized) {
                (torn, _) if torn > 0 => Some(format!(
                    "Adopted from a capture journal whose last {torn} bytes were an \
                     unfinished write; they were discarded (gap flagged, never joined)."
                )),
                (_, false) => Some(
                    "Adopted from a capture journal that was never finalized; audio up \
                     to the last verified boundary was kept."
                        .to_string(),
                ),
                (_, true) => None,
            };
            [note.map(str::to_string), recovery_note]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" ")
        })
    }

    /// The body of [`Self::adopt_journal`]: `note` words the row's
    /// recovery note from what the verified read found (an empty note
    /// stores none), and `status`, when set, is the row's status in the
    /// same commit — otherwise a torn or unfinalized journal lands
    /// interrupted and a sealed one complete. `transcribe` records the
    /// intent to transcribe the take only when the journal itself is
    /// complete (sealed by its recorder, nothing torn), whatever status
    /// the row gets.
    fn adopt_journal_with(
        &mut self,
        source: &Path,
        status: Option<CaptureStatus>,
        transcribe: bool,
        note: impl FnOnce(&AdoptedJournal) -> String,
    ) -> Result<CaptureRecord, StoreV2Error> {
        let id = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                StoreV2Error::Invalid(format!(
                    "journal path {:?} has no usable file stem",
                    source.display()
                ))
            })?
            .to_string();
        self.adopt_journal_as(source, id, status, transcribe, note)
    }

    /// [`Self::adopt_journal_with`] under capture id `id`.
    fn adopt_journal_as(
        &mut self,
        source: &Path,
        id: String,
        status: Option<CaptureStatus>,
        transcribe: bool,
        note: impl FnOnce(&AdoptedJournal) -> String,
    ) -> Result<CaptureRecord, StoreV2Error> {
        validate_capture_id(&id)?;
        // Its notes must be on disk before the journal changes: inside a
        // caller's transaction a rollback would take them back.
        self.refuse_open_transaction("adopt a capture journal")?;
        if self.get_capture(&id)?.is_some() || self.audio_path(&id).exists() {
            return Err(StoreV2Error::Invalid(format!(
                "destination already holds capture {id}; refusing to overwrite"
            )));
        }

        let mut parsed = read_journal(source).map_err(|err| {
            StoreV2Error::Invalid(format!(
                "capture journal {}: {err}",
                source.display()
            ))
        })?;
        if parsed.samples.is_empty() {
            return Err(StoreV2Error::NoVerifiedSamples { id: id.clone() });
        }

        // An unsealed or torn journal is sealed to its verified prefix
        // first (idempotent), so audio/ only ever holds trailer-valid
        // files. The discarded tail — if any — is recorded as the gap.
        // That it was sealed here is recorded first, durably, under the
        // journal and the capture it moves into: a pass that dies before
        // the adoption commits — before or after the move — leaves a
        // sealed journal that still reads as cut short, never as a take
        // that stopped (#220). The adoption's commit clears it.
        let source_id = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_string();
        let earlier_seal = self.recovery_seal(&source_id)?;
        let torn_tail_bytes = earlier_seal
            .map_or(parsed.torn_tail_bytes, |earlier| earlier.max(parsed.torn_tail_bytes));
        let was_finalized = parsed.finalized && earlier_seal.is_none();
        let cut = !was_finalized || torn_tail_bytes > 0;
        if cut {
            self.note_recovery_seal(&source_id, &id, torn_tail_bytes)?;
        }
        #[cfg(test)]
        self.crash_point(RecoveryStep::SealNoted)?;
        if !parsed.finalized || parsed.torn_tail_bytes > 0 {
            seal_recovered_journal(source, &parsed)?;
            parsed.finalized = true;
            parsed.torn_tail_bytes = 0;
        }
        #[cfg(test)]
        self.crash_point(RecoveryStep::Sealed)?;
        let transcribe = transcribe && !cut;
        if transcribe {
            self.note_pending_intent(&id)?;
        }

        std::fs::create_dir_all(self.root.join(AUDIO_DIR))?;
        let destination = self.audio_path(&id);
        std::fs::rename(source, &destination)?;
        // Durable on both sides: a source name that came back after a
        // power loss would be offered to recovery beside the stored take.
        sync_dir(&self.root.join(AUDIO_DIR))?;
        if let Some(parent) = source.parent() {
            sync_dir(parent)?;
        }
        #[cfg(test)]
        self.crash_point(RecoveryStep::Moved)?;

        let count = parsed.samples.len() as u64;
        let note = note(&AdoptedJournal {
            samples: count,
            sample_rate: parsed.sample_rate,
            torn_tail_bytes,
            was_finalized,
        });
        let record = CaptureRecord {
            id: id.clone(),
            created_utc: now_iso(),
            tz: local_tz_label(),
            device: String::new(),
            actual_rate: parsed.sample_rate,
            policy: "adopted".to_string(),
            frame_count: count,
            ack_sample_index: count,
            journal_hash: format!("{:016x}", samples_hash(&parsed.samples)),
            status: status.unwrap_or(if torn_tail_bytes > 0 || !was_finalized {
                CaptureStatus::Interrupted
            } else {
                CaptureStatus::Complete
            }),
            retention_class: "standard".to_string(),
            extra_json: (!note.is_empty()).then(|| merge_extra_note(None, &note)),
            secure_field: false,
        };
        self.commit_capture_superseding(&record, None, transcribe, true)?;
        Ok(record)
    }

    /// The take stored in place of recorder journal `journal_id`
    /// ([`TakeMeta::supersedes_journal`]), if one was (#356): its row is
    /// committed, or it was stored and then deliberately deleted. A
    /// replacement whose save never committed replaces nothing — the
    /// journal is then still the take's only stored copy.
    pub fn journal_superseded_by(&self, journal_id: &str) -> Result<Option<String>, StoreV2Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT s.capture_id FROM journal_supersessions s
                 WHERE s.journal_id = ?1
                   AND (EXISTS (SELECT 1 FROM captures c WHERE c.id = s.capture_id)
                        OR EXISTS (SELECT 1 FROM tombstones t WHERE t.id = s.capture_id))",
                params![journal_id],
                |row| row.get(0),
            )
            .optional()?)
    }

    /// Where else recorder journal `journal_id`'s confirmed `samples` (at
    /// `rate`) are kept, as proven by reading that copy's audio from disk
    /// (#356) — never by row metadata. A copy that is missing, unreadable,
    /// shorter, at another rate or different in any sample proves
    /// nothing, and neither does a deleted replacement whose audio the
    /// sweep already removed: the journal may then be the only copy, and
    /// is adopted (a duplicate the user can delete beats audio lost).
    fn journal_copy(
        &self,
        journal_id: &str,
        samples: &[f32],
        rate: u32,
    ) -> Result<JournalCopy, StoreV2Error> {
        let held_at = |path: PathBuf| {
            read_audio_journal(&path).is_ok_and(|stored| holds_journal(&stored, samples, rate))
        };
        // Adopted under its own name, with its move out of the recorder's
        // tree undone by a power loss.
        if self.get_capture(journal_id)?.is_some() {
            return Ok(
                if self.audio_retired_utc(journal_id)?.is_none()
                    && held_at(self.audio_path(journal_id))
                {
                    JournalCopy::Stored
                } else {
                    JournalCopy::Unproven
                },
            );
        }
        // This very recording was adopted and then deleted by the user:
        // recorder journal ids are never reused, and only a capture
        // delete leaves a bare `capture` tombstone on one (the sweep
        // stamps its copies under a prefix). An adoption now would only
        // be deleted again by reconcile, the id being dead to it.
        if self.is_deleted_capture(journal_id)? {
            return Ok(JournalCopy::Deleted);
        }
        let replacement: Option<String> = self
            .conn
            .query_row(
                "SELECT capture_id FROM journal_supersessions WHERE journal_id = ?1",
                params![journal_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(replacement) = replacement else {
            return Ok(JournalCopy::Unproven);
        };
        if self.is_tombstoned(&replacement)? {
            // Deleted since: only its quarantined audio, until swept,
            // proves the journal a copy of what the user deleted.
            return Ok(if held_at(self.quarantine_path(&replacement)) {
                JournalCopy::Stored
            } else {
                JournalCopy::Unproven
            });
        }
        if self.get_capture(&replacement)?.is_some() {
            return Ok(
                if self.audio_retired_utc(&replacement)?.is_none()
                    && held_at(self.audio_path(&replacement))
                {
                    JournalCopy::Stored
                } else {
                    JournalCopy::Unproven
                },
            );
        }
        // Not committed: reconcile turns its staging journal, or its audio
        // left in `audio/` without a row, into the take.
        Ok(
            if held_at(self.staging_path(&replacement)) || held_at(self.audio_path(&replacement))
            {
                JournalCopy::Pending
            } else {
                JournalCopy::Unproven
            },
        )
    }

    /// Count one more recovery pass that left recorder journal `id` to its
    /// uncommitted replacement; the passes so far, this one included.
    fn pending_pass(&self, id: &str) -> Result<u32, StoreV2Error> {
        self.conn.execute(
            "UPDATE journal_supersessions SET pending_passes = pending_passes + 1
             WHERE journal_id = ?1",
            params![id],
        )?;
        Ok(self.conn.query_row(
            "SELECT pending_passes FROM journal_supersessions WHERE journal_id = ?1",
            params![id],
            |row| row.get(0),
        )?)
    }

    /// The capture id startup recovery adopts recorder journal `id` as:
    /// its own, unless a take, its audio or a tombstone already holds that
    /// name — then the first free `<id>-recovered[-n]`.
    fn fresh_adoption_id(&self, id: &str) -> Result<String, StoreV2Error> {
        let mut attempt = 0u32;
        loop {
            let candidate = match attempt {
                0 => id.to_string(),
                1 => format!("{id}-recovered"),
                n => format!("{id}-recovered-{n}"),
            };
            if self.get_capture(&candidate)?.is_none()
                && !self.has_audio(&candidate)
                && !self.staging_path(&candidate).exists()
                && !self.is_tombstoned(&candidate)?
            {
                return Ok(candidate);
            }
            attempt += 1;
        }
    }

    /// Whether capture `id` itself was deleted (a `capture` tombstone).
    fn is_deleted_capture(&self, id: &str) -> Result<bool, StoreV2Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT 1 FROM tombstones WHERE id = ?1 AND kind = 'capture'",
                params![id],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    /// Whether `id` was deliberately deleted (or its audio retired).
    fn is_tombstoned(&self, id: &str) -> Result<bool, StoreV2Error> {
        Ok(self
            .conn
            .query_row("SELECT 1 FROM tombstones WHERE id = ?1", params![id], |_| Ok(()))
            .optional()?
            .is_some())
    }

    /// Startup recovery of the recorder's live-capture tree (#356):
    /// `journals_dir` is where takes journal while they record
    /// ([`crate::journal::default_journals_root`]). A journal still there
    /// at startup belongs to a take whose app stopped before saving it —
    /// killed or crashed mid-recording, or between stop and save — and is
    /// adopted into history as an interrupted take whose note says what
    /// survived. [`Self::reconcile`] covers only the store's own trees;
    /// this is the recorder's half.
    ///
    /// What is left alone:
    ///
    /// - a journal whose writer lock is held: a live take, recording in
    ///   this process or another (the writer holds the lock for the
    ///   file's lifetime; the OS frees it when the writer dies);
    /// - a finalized journal written in the last
    ///   [`FINALIZED_ADOPTION_GRACE`]: its take may be between stop and
    ///   save in a live instance, which adopts it itself — a later launch
    ///   recovers it if not;
    /// - a journal with no verified samples (nothing to recover);
    /// - `deleted/`, `superseded/` and anything not named `*.sj`.
    ///
    /// A journal whose take is already stored — the save committed it in
    /// the journal's place, or adopted it, and the app stopped before the
    /// journal left the tree — is moved into `superseded/` instead of
    /// becoming a take again, and one whose take was deleted since into
    /// `deleted/`; but only once that take's audio, read back from disk,
    /// holds every confirmed sample of the journal at its rate. Anything
    /// less is adopted (a duplicate the user can delete beats audio
    /// lost). A journal whose replacement awaits a reconcile that has not
    /// committed it is deferred: reconcile runs first, and adopting the
    /// journal now would leave both once it does — but only for
    /// [`PENDING_REPLACEMENT_PASSES`] passes, so a reconcile that keeps
    /// failing never keeps the take hidden. A stale
    /// `*.sj.creating` scratch name no writer holds is removed: it is an
    /// extra name of a published journal, or an empty file whose writer
    /// died before publishing it.
    ///
    /// A file that is not a readable journal is renamed to
    /// `<name>.unrecognized` beside itself — kept, never repaired or
    /// deleted, and not rescanned. A failed adoption leaves the journal in
    /// place for the next launch. Never deletes anything.
    pub fn recover_capture_journals(
        &mut self,
        journals_dir: &Path,
    ) -> Result<JournalRecovery, StoreV2Error> {
        self.recover_capture_journals_where(journals_dir, |_| true, false)
    }

    /// [`Self::recover_capture_journals`] limited to the journal ids
    /// `wanted` accepts — a second look at the ones an earlier pass
    /// deferred, without making candidates of takes recorded since.
    /// `transcribe_complete`: a journal its recorder finalized cleanly —
    /// a take that stopped but whose process died before storing it — is
    /// adopted with the intent to transcribe it (#220), in the same
    /// commit, as its own store would have done. A journal cut short by
    /// the crash never is: what to do with a partial take is the user's
    /// call.
    pub fn recover_capture_journals_where(
        &mut self,
        journals_dir: &Path,
        wanted: impl Fn(&str) -> bool,
        transcribe_complete: bool,
    ) -> Result<JournalRecovery, StoreV2Error> {
        self.refuse_open_transaction("recover capture journals")?;
        let mut report = JournalRecovery::default();
        // No tree yet is nothing to recover; any other failure to list it
        // must surface, never read as "no interrupted recordings".
        let entries = match std::fs::read_dir(journals_dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(report),
            Err(err) => return Err(err.into()),
        };
        let mut paths: Vec<PathBuf> = entries
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();
        paths.sort();
        let (paths, scratch): (Vec<PathBuf>, Vec<PathBuf>) = paths
            .into_iter()
            .filter(|path| {
                let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
                name.ends_with(".sj") || name.ends_with(".sj.creating")
            })
            .partition(|path| path.extension().and_then(|ext| ext.to_str()) == Some("sj"));
        for path in scratch {
            let Some(id) = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_suffix(".sj.creating"))
                .map(str::to_string)
            else {
                continue;
            };
            if !is_safe_path_component(&id) || !wanted(&id) {
                continue;
            }
            // Only a free lock on a name nobody has touched for a while
            // says no writer is mid-create (the creator locks a moment
            // after creating it); the header is written after publishing,
            // so the scratch name never holds audio a published journal
            // does not.
            let free = modified_age(&path).is_some_and(|age| age >= FINALIZED_ADOPTION_GRACE)
                && File::open(&path).ok().is_some_and(|file| {
                    matches!(try_flock_exclusive(&file), Ok(FlockEvidence::Free))
                });
            if free {
                match std::fs::remove_file(&path) {
                    Ok(()) => {
                        let _ = sync_dir(journals_dir);
                    }
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) => report.failed.push((id, err.to_string())),
                }
            }
        }
        for path in paths {
            let Some(id) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
            else {
                continue;
            };
            if !is_safe_path_component(&id) || !wanted(&id) {
                continue;
            }
            // The probe's handle keeps the lock through the adoption, so
            // a second instance scanning at the same moment sees it held.
            let lock = match File::open(&path) {
                Ok(file) => file,
                Err(err) => {
                    report.failed.push((id, err.to_string()));
                    continue;
                }
            };
            let free = match try_flock_exclusive(&lock) {
                Ok(FlockEvidence::Free) => true,
                Ok(FlockEvidence::Held) | Err(_) => false,
                // No lock primitive (only targets that are neither unix
                // nor windows; a unix filesystem that cannot lock answers
                // with an error, read as held): an unfinalized journal
                // nobody has written to for a while has no writer. A live
                // take writes a boundary at least every quarter second
                // while samples arrive, and its stall watch stops a take
                // whose input goes quiet.
                Ok(FlockEvidence::Unknown) => modified_age(&path)
                    .is_some_and(|age| age >= FINALIZED_ADOPTION_GRACE),
            };
            if !free {
                report.deferred.push(id);
                continue;
            }
            let parsed = match read_journal(&path) {
                Ok(parsed) => parsed,
                Err(journal::JournalReadError::NotAJournal(reason)) => {
                    let aside = path.with_extension("sj.unrecognized");
                    match std::fs::rename(&path, &aside) {
                        Ok(()) => {
                            let _ = sync_dir(journals_dir);
                            report.unrecognized.push((id, reason));
                        }
                        Err(err) => report.failed.push((id, err.to_string())),
                    }
                    continue;
                }
                Err(err) => {
                    report.failed.push((id, err.to_string()));
                    continue;
                }
            };
            if parsed.samples.is_empty() {
                continue;
            }
            // Already a take, or deliberately not one: stored in this
            // journal's place (the app stopped between that commit and
            // moving the journal aside), adopted with its move out of the
            // tree undone by a power loss, or since deleted — each only
            // once that copy's audio is read back holding all of this.
            let aside = match self.journal_copy(&id, &parsed.samples, parsed.sample_rate) {
                Ok(JournalCopy::Unproven) => None,
                Ok(JournalCopy::Stored) => Some(SUPERSEDED_SUBDIR),
                Ok(JournalCopy::Deleted) => Some(DELETED_SUBDIR),
                Ok(JournalCopy::Pending) => {
                    // Its replacement awaits a reconcile that has not
                    // committed it yet: adopting now would leave both. A
                    // reconcile that keeps failing is waited out only for
                    // a few passes — then the journal is adopted, a
                    // possible duplicate rather than a take kept hidden.
                    if self.pending_pass(&id).is_ok_and(|passes| passes <= PENDING_REPLACEMENT_PASSES)
                    {
                        report.deferred.push(id);
                        continue;
                    }
                    None
                }
                Err(err) => {
                    report.failed.push((id, err.to_string()));
                    continue;
                }
            };
            if let Some(aside) = aside {
                drop(parsed);
                match move_journal_aside(&path, &journals_dir.join(aside)) {
                    Ok(()) => {
                        let _ = self.conn.execute(
                            "DELETE FROM recovery_seals WHERE journal_id = ?1",
                            params![id],
                        );
                        if aside == SUPERSEDED_SUBDIR {
                            report.superseded.push(id);
                        } else {
                            report.deleted.push(id);
                        }
                    }
                    Err(err) => report.failed.push((id, err.to_string())),
                }
                continue;
            }
            // A journal recovery sealed has no recorder left to save it.
            let sealed_here = match self.recovery_seal(&id) {
                Ok(seal) => seal.is_some(),
                Err(err) => {
                    report.failed.push((id, err.to_string()));
                    continue;
                }
            };
            if parsed.finalized
                && !sealed_here
                && modified_age(&path).is_none_or(|age| age < FINALIZED_ADOPTION_GRACE)
            {
                report.deferred.push(id);
                continue;
            }
            drop(parsed);
            // A take already under the journal's name whose audio does not
            // hold it (lost, damaged, retired) keeps its row; the journal
            // comes back beside it under a fresh name.
            let as_id = match self.fresh_adoption_id(&id) {
                Ok(as_id) => as_id,
                Err(err) => {
                    report.failed.push((id, err.to_string()));
                    continue;
                }
            };
            match self.adopt_journal_as(&path, as_id, Some(CaptureStatus::Interrupted), transcribe_complete, |facts| {
                recovered_journal_note(facts)
            }) {
                Ok(record) => report.recovered.push(RecoveredJournal {
                    id: record.id,
                    samples: record.frame_count,
                    sample_rate: record.actual_rate,
                }),
                Err(err) => report.failed.push((id, err.to_string())),
            }
            drop(lock);
        }
        Ok(report)
    }

    /// Saves an encoded WAV as a new capture through the full §4 crash
    /// protocol (staging journal → finalize → promote → commit). The
    /// import-audio and quiesce-salvage paths arrive as canonical 16 kHz
    /// WAVs rather than live journals; this writes them through the same
    /// durable steps a take gets. The audio is verified by re-reading the
    /// journal before the row commits. Note this decodes the entire WAV
    /// into an in-memory [`crate::audio::PcmAudio`] while the caller still
    /// holds the encoded bytes — roughly doubling peak memory for long
    /// imports; acceptable at desktop scale, worth knowing for future
    /// bulk-import callers.
    ///
    /// Everything runs under the caller's `&mut self`; a caller that shares
    /// the store behind a lock and wants the journal writes off it drives
    /// the same steps itself ([`Self::begin_take_at_rate`] +
    /// [`V2Take::append_and_seal`] + [`V2Take::finalize`] +
    /// [`FinalizedTake::commit_marked`]).
    pub fn save_wav_capture(
        &mut self,
        wav: &[u8],
        meta: TakeMeta,
    ) -> Result<CommittedTake, StoreV2Error> {
        let pcm = decode_pcm16_wav(wav)?;
        if pcm.samples.is_empty() {
            return Err(StoreV2Error::Invalid(
                "wav has no samples; nothing to store".to_string(),
            ));
        }
        let mut take = self.begin_take_at_rate(pcm.sample_rate, meta)?;
        take.append_and_seal(&pcm.samples)?;
        take.finalize()?.commit_marked(self, CommitMark::Complete)
    }

    /// Marks the start of one recognition attempt on a capture: a
    /// `recognition_attempts` row with status `started`, registered as
    /// this process's in-flight attempt (#213) *before* the row becomes
    /// visible, so the sweep never observes an unowned in-between.
    /// Returns the attempt id. Recognizing an unknown
    /// capture is [`StoreV2Error::NotFound`].
    pub fn begin_recognition(
        &mut self,
        capture_id: &str,
        backend: &str,
        options_json: Option<&str>,
    ) -> Result<String, StoreV2Error> {
        // Its own transaction, never a caller's: the failure path's
        // ROLLBACK below would end that one too.
        self.refuse_open_transaction("begin a recognition attempt")?;
        // The checks and the insert hold the database's write lock: the
        // retention policy (#342), on this connection or another, either
        // sees this attempt and keeps the audio, or retired it first and
        // the attempt is refused — never an attempt on removed audio.
        let begun = self.begin_recognition_in_transaction(capture_id, backend, options_json);
        if !self.conn.is_autocommit() {
            // The guard's own rollback failed too (rusqlite drops that
            // error): try once more, so a failed begin does not leave every
            // later write on this connection inside a dead transaction.
            if let Err(err) = self.conn.execute_batch("ROLLBACK") {
                eprintln!("Rolling back a failed recognition start also failed: {err}");
            }
        }
        begun
    }

    fn begin_recognition_in_transaction(
        &mut self,
        capture_id: &str,
        backend: &str,
        options_json: Option<&str>,
    ) -> Result<String, StoreV2Error> {
        // The transaction guard rolls back on every exit that is not a
        // successful commit — an early error, a failed COMMIT, or an
        // unwind.
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        if self.get_capture(capture_id)?.is_none() {
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        if let Some(utc) = self.audio_retired_utc(capture_id)? {
            return Err(StoreV2Error::Invalid(format!(
                "the audio of capture {capture_id} was removed by the retention policy on \
                 {utc}; it cannot be transcribed again"
            )));
        }
        let id = format!("a_{}", uuid::Uuid::new_v4().simple());
        // Registered from here; it joins `held_attempts` only once the row
        // is committed, and a failed begin drops it.
        let claim = AttemptClaim::new(&id);
        let inserted = insert_attempt_row(
            &tx,
            &AttemptRecord {
                id: id.clone(),
                capture_id: capture_id.to_string(),
                backend: backend.to_string(),
                model_hash: None,
                language: None,
                options_json: options_json.map(str::to_string),
                text: String::new(),
                partial_or_final: "partial".to_string(),
                status: "started".to_string(),
                timing_json: None,
                extra_json: None,
                created_utc: None,
            },
        )
        .and_then(|()| tx.commit().map_err(StoreV2Error::from));
        match inserted {
            Ok(()) => {
                self.held_attempts.insert(id.clone(), claim);
                Ok(id)
            }
            Err(err) => Err(err),
        }
    }

    /// How one recognition attempt ended. Applies to the capture's most
    /// recent `started` attempt — the app guards one in-flight job per
    /// capture, so "latest started" is that job. [`StoreV2Error::NotFound`]
    /// when the capture is gone or no attempt is in flight: the v1 delete
    /// race maps onto exactly that (a delete cascades the attempts away).
    pub fn finish_recognition(
        &mut self,
        capture_id: &str,
        outcome: RecognitionOutcome<'_>,
    ) -> Result<(), StoreV2Error> {
        if self.get_capture(capture_id)?.is_none() {
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        let attempt_id: Option<String> = self
            .conn
            .query_row(
                "SELECT id FROM recognition_attempts
                 WHERE capture_id = ?1 AND status = 'started'
                 ORDER BY rowid DESC LIMIT 1",
                params![capture_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(attempt_id) = attempt_id else {
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        };

        let written = match outcome {
            RecognitionOutcome::Completed { text, extra_json } => self.conn.execute(
                "UPDATE recognition_attempts
                 SET text = ?1, partial_or_final = 'final', status = 'completed',
                     extra_json = COALESCE(?2, extra_json)
                 WHERE id = ?3 AND status = 'started'",
                params![text, extra_json, attempt_id],
            ),
            RecognitionOutcome::Failed { message } => {
                let extra = serde_json::json!({ "error": message });
                self.conn.execute(
                    "UPDATE recognition_attempts
                     SET status = 'failed', extra_json = ?1
                     WHERE id = ?2 AND status = 'started'",
                    params![extra.to_string(), attempt_id],
                )
            }
        };
        let changed = match written {
            Ok(changed) => changed,
            Err(err) => {
                // The settle could not be written (a full disk): nothing
                // works on the attempt any more, so it must not read as
                // in flight (#220) — the stale sweep takes it from here.
                self.release_attempt(&attempt_id);
                return Err(err.into());
            }
        };
        if changed == 0 {
            // A second finisher lost the race to the first: never re-write
            // a terminal row. The attempt is settled either way.
            self.release_attempt(&attempt_id);
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        self.release_attempt(&attempt_id);
        Ok(())
    }

    /// [`Self::finish_recognition`] with the v1-shaped transcript result:
    /// the full result (text, segments, duration, request id) is preserved
    /// verbatim in the attempt row.
    pub fn finish_recognition_transcript(
        &mut self,
        capture_id: &str,
        transcript: &crate::storage::TranscriptionResult,
    ) -> Result<(), StoreV2Error> {
        let text = transcript.text.clone();
        let extra = serde_json::to_string(transcript)
            .map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
        self.finish_recognition(
            capture_id,
            RecognitionOutcome::Completed {
                text: &text,
                extra_json: Some(&extra),
            },
        )
    }

    /// Fails the `started` attempts no handle in this process holds — left
    /// by a run that is gone, or by a settle that could not be written —
    /// with `note`. The write is conditional on the row still reading
    /// `started`, so a settle that commits meanwhile keeps its outcome.
    /// Returns the affected capture ids, each once.
    pub fn interrupt_stale_attempts(&mut self, note: &str) -> Result<Vec<String>, StoreV2Error> {
        let mut started: Vec<(String, String)> = Vec::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT capture_id, id FROM recognition_attempts
                 WHERE status = 'started' ORDER BY capture_id, rowid",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                started.push(row?);
            }
        }
        let extra = serde_json::json!({ "error": note }).to_string();
        let mut swept: Vec<String> = Vec::new();
        let mut swept_captures: HashSet<&str> = HashSet::new();
        for (capture_id, attempt_id) in &started {
            if self.attempt_is_owned(attempt_id) {
                continue;
            }
            let changed = self.conn.execute(
                "UPDATE recognition_attempts
                 SET status = 'failed', extra_json = ?1
                 WHERE id = ?2 AND status = 'started'",
                params![extra, attempt_id],
            )?;
            if changed > 0 && swept_captures.insert(capture_id) {
                // The distinct-capture contract of the old `SELECT
                // DISTINCT` sweep: one entry per capture, however many of
                // its attempts this pass failed.
                swept.push(capture_id.clone());
            }
        }
        Ok(swept)
    }

    /// Ends this handle's hold on `attempt_id` (a no-op for attempts it
    /// does not hold).
    fn release_attempt(&mut self, attempt_id: &str) {
        self.held_attempts.remove(attempt_id);
    }

    /// Whether a handle in this process holds `started` attempt
    /// `attempt_id`.
    pub fn attempt_owned(&self, attempt_id: &str) -> bool {
        self.attempt_is_owned(attempt_id)
    }

    fn attempt_is_owned(&self, attempt_id: &str) -> bool {
        self.held_attempts.contains_key(attempt_id) || AttemptClaim::is_live(attempt_id)
    }
}

/// How long a finalized journal in the recorder's tree is left to the
/// instance that may still be saving it before startup recovery adopts
/// it ([`StoreV2::recover_capture_journals`]). A live save adopts within
/// seconds of the stop.
pub const FINALIZED_ADOPTION_GRACE: std::time::Duration = std::time::Duration::from_secs(60);

/// What the verified read of an adopted journal found.
struct AdoptedJournal {
    samples: u64,
    sample_rate: u32,
    torn_tail_bytes: u64,
    was_finalized: bool,
}

/// The note a take recovered from the recorder's tree carries: how much
/// audio came back, and the durability boundary it came back to.
fn recovered_journal_note(facts: &AdoptedJournal) -> String {
    let seconds = facts.samples as f64 / f64::from(facts.sample_rate.max(1));
    if facts.was_finalized && facts.torn_tail_bytes == 0 {
        return format!(
            "Starling closed after this take stopped but before it was saved. The complete \
             recording ({seconds:.1} s) was recovered."
        );
    }
    let mut note = format!(
        "Starling closed while this take was recording. Recovered {seconds:.1} s: everything \
         the capture journal had confirmed on disk. Audio after its last confirmation is \
         missing — normally only the last moment, as the journal confirms about every quarter \
         second while the disk keeps up."
    );
    if facts.torn_tail_bytes > 0 {
        note.push_str(&format!(
            " An unconfirmed write of {} bytes at the end was discarded (gap flagged, never \
             joined).",
            facts.torn_tail_bytes
        ));
    }
    note
}

/// How long ago `path` was last written; `None` when the filesystem
/// cannot say (or the clock runs behind the file).
fn modified_age(path: &Path) -> Option<std::time::Duration> {
    let modified = std::fs::metadata(path).and_then(|meta| meta.modified()).ok()?;
    std::time::SystemTime::now().duration_since(modified).ok()
}

/// Where else a recorder journal's confirmed audio is kept (#356,
/// [`StoreV2::journal_copy`]).
enum JournalCopy {
    /// Nowhere proven: the journal may be the only copy.
    Unproven,
    /// A take's audio, read back — stored, or quarantined by a delete —
    /// holds every confirmed sample.
    Stored,
    /// This very recording was adopted and then deleted by the user.
    Deleted,
    /// A replacement reconcile has not committed yet holds them.
    Pending,
}

/// Whether `stored` — a take's audio as read back from disk — holds
/// every confirmed sample of a recorder journal (`samples` at `rate`) as
/// a prefix, at the same rate (#356). Samples are compared as the 16-bit
/// request PCM the store keeps audio as at rest (#342), the whole prefix
/// through one path: every stored sample quantizes as the journal's, or
/// every one as the journal's after the PCM16 WAV round trip a save from
/// the in-memory take goes through — never a mix of the two. Anything
/// less — another rate, fewer samples, one differing sample — proves
/// nothing.
fn holds_journal(stored: &JournalAudio, samples: &[f32], rate: u32) -> bool {
    let through = |path: fn(i16) -> i16| {
        samples
            .iter()
            .zip(&stored.samples)
            .all(|(&sample, &kept)| pcm16(kept) == path(pcm16(sample)))
    };
    stored.sample_rate == rate
        && stored.samples.len() >= samples.len()
        && (through(|request| request)
            || through(|request| pcm16(f32::from(request) / 32_768.0)))
}

/// Move recorder journal `journal` into `superseded/` beside it once the
/// take stored in its place — whose audio is at `stored` (its journal or
/// FLAC, [`StoreV2::audio_journal_path`]) — is proven to hold every
/// confirmed sample ([`holds_journal`]). `Ok(true)` when moved; kept,
/// since journals are never deleted outside the retention sweep, which
/// only ever finds proven copies there. Anything unproven — the stored
/// audio unreadable, shorter, at another rate — leaves the journal where
/// it is, for startup recovery to adopt as an interrupted take: a
/// duplicate the user can delete beats audio lost. A journal already
/// gone is fine. A name already taken in `superseded/` gets a numbered
/// one: the kept journal there is never replaced.
pub fn supersede_journal_held_by(journal: &Path, stored: &Path) -> Result<bool, StoreV2Error> {
    let parsed = match read_journal(journal) {
        Ok(parsed) => parsed,
        Err(journal::JournalReadError::Io(err)) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(false);
        }
        Err(err) => return Err(StoreV2Error::Invalid(err.to_string())),
    };
    let held = read_audio_journal(stored)
        .is_ok_and(|stored| holds_journal(&stored, &parsed.samples, parsed.sample_rate));
    drop(parsed);
    let Some(dir) = journal.parent().filter(|_| held) else {
        return Ok(false);
    };
    move_journal_aside(journal, &dir.join(SUPERSEDED_SUBDIR))?;
    Ok(true)
}

/// Move `path` into the `aside` directory, never replacing a kept file.
fn move_journal_aside(path: &Path, aside: &Path) -> Result<(), StoreV2Error> {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
        return Ok(());
    };
    if !path.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(aside)?;
    let stem = path.file_stem().unwrap_or(name).to_string_lossy().to_string();
    let mut attempt = 0u32;
    loop {
        let destination = if attempt == 0 {
            aside.join(name)
        } else {
            aside.join(format!("{stem}.{attempt}.sj"))
        };
        // One atomic move that never replaces what is already kept: no
        // instant has the journal under both names (recovery would offer
        // the tree's copy, sharing the kept one's bytes).
        if rename_noreplace(path, &destination)? {
            break;
        }
        attempt += 1;
    }
    sync_dir(dir)?;
    sync_dir(aside)?;
    Ok(())
}

/// Rename `from` to `to` unless `to` exists: `Ok(false)` when the name is
/// taken. Linux (glibc) asks the kernel for it atomically
/// (`RENAME_NOREPLACE`); a filesystem or platform without that checks
/// first — only [`move_journal_aside`] writes in `superseded/` and
/// `deleted/`, so the check-to-rename window has no competing writer in
/// practice.
fn rename_noreplace(from: &Path, to: &Path) -> io::Result<bool> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let from_c = CString::new(from.as_os_str().as_bytes())?;
        let to_c = CString::new(to.as_os_str().as_bytes())?;
        // SAFETY: two NUL-terminated paths that outlive the call.
        let renamed = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                from_c.as_ptr(),
                libc::AT_FDCWD,
                to_c.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        if renamed == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EEXIST) => return Ok(false),
            // No RENAME_NOREPLACE on this filesystem or kernel.
            Some(libc::EINVAL) | Some(libc::ENOSYS) => {}
            _ => return Err(error),
        }
    }
    match std::fs::symlink_metadata(to) {
        Ok(_) => Ok(false),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            std::fs::rename(from, to).map(|()| true)
        }
        Err(err) => Err(err),
    }
}

/// How many recovery passes leave a recorder journal to a replacement
/// reconcile has not committed (#356) before adopting it anyway: two
/// launches' worth (each runs a startup scan and a later recheck).
const PENDING_REPLACEMENT_PASSES: u32 = 4;

/// Where [`supersede_journal_held_by`] keeps journals, under the
/// recorder's tree.
pub const SUPERSEDED_SUBDIR: &str = "superseded";

/// Where startup recovery keeps the recorder journals of takes the user
/// deleted — the journal's own id carries the capture tombstone — under
/// the recorder's tree (swept like the takes' own audio).
const DELETED_SUBDIR: &str = "deleted";

/// What [`StoreV2::recover_capture_journals`] did.
#[derive(Debug, Default)]
pub struct JournalRecovery {
    /// Takes adopted into history as interrupted.
    pub recovered: Vec<RecoveredJournal>,
    /// Journals left for a live writer, or for the instance that may
    /// still be saving them.
    pub deferred: Vec<String>,
    /// Files that are not readable journals, renamed aside: `(id, why)`.
    pub unrecognized: Vec<(String, String)>,
    /// Journals that could not be adopted this time: `(id, why)`. They
    /// stay where they are for the next launch.
    pub failed: Vec<(String, String)>,
    /// Journals whose take was already stored: moved into `superseded/`.
    /// Housekeeping, not a finding.
    pub superseded: Vec<String>,
    /// Journals of takes adopted under their own name and since deleted
    /// by the user: moved into `deleted/`. Housekeeping, not a finding.
    pub deleted: Vec<String>,
}

/// One take [`StoreV2::recover_capture_journals`] brought back.
#[derive(Debug, Clone)]
pub struct RecoveredJournal {
    pub id: String,
    /// Verified samples recovered — all of them confirmed on disk.
    pub samples: u64,
    pub sample_rate: u32,
}

impl JournalRecovery {
    /// Everything this pass has to say; empty when nothing.
    pub fn summary(&self) -> String {
        [self.recovered_summary(), self.problems()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// What came back, for a notice; empty when nothing did.
    pub fn recovered_summary(&self) -> String {
        let recovered = self.recovered.len();
        if recovered == 0 {
            return String::new();
        }
        format!(
            "Recovered {recovered} recording{} Starling was still recording or saving when it \
             closed; {} in your history as interrupted, ready to play, export or transcribe.",
            if recovered == 1 { "" } else { "s" },
            if recovered == 1 { "it is" } else { "they are" }
        )
    }

    /// What could not be recovered or read, for the error banner; empty
    /// when nothing went wrong.
    pub fn problems(&self) -> String {
        let plural = |count: usize| if count == 1 { "" } else { "s" };
        let mut parts = Vec::new();
        if !self.failed.is_empty() {
            let reasons: Vec<String> = self
                .failed
                .iter()
                .map(|(id, why)| format!("{id}: {why}"))
                .collect();
            parts.push(format!(
                "{} interrupted recording{} could not be recovered yet and stay on disk for the \
                 next launch ({}).",
                self.failed.len(),
                plural(self.failed.len()),
                reasons.join("; ")
            ));
        }
        if !self.unrecognized.is_empty() {
            parts.push(format!(
                "{} file{} in the capture journal folder {} not a readable recording; kept \
                 aside as .unrecognized.",
                self.unrecognized.len(),
                plural(self.unrecognized.len()),
                if self.unrecognized.len() == 1 { "is" } else { "are" }
            ));
        }
        parts.join(" ")
    }
}

/// How one recognition attempt ended ([`StoreV2::finish_recognition`]).
#[derive(Debug, Clone, Copy)]
pub enum RecognitionOutcome<'a> {
    /// The backend returned text; `extra_json` carries the full v1-shaped
    /// transcript (segments, duration, request id) verbatim when the caller
    /// has one.
    Completed {
        text: &'a str,
        extra_json: Option<&'a str>,
    },
    /// The attempt failed; `message` is surfaced with the row.
    Failed { message: &'a str },
}

/// How a finalized take's `captures` row is written at commit time
/// ([`FinalizedTake::commit_marked`]). `Interrupted` writes the status and
/// its recovery note inside the commit transaction itself — there is no
/// second update that a crash could skip (R34: a salvaged take must never
/// rest as a complete take without its note).
#[derive(Debug, Clone)]
pub enum CommitMark {
    Complete,
    Interrupted { note: String },
}

impl V2Take {
    /// The capture id (also the journal file stem).
    pub fn id(&self) -> &str {
        self.writer.id()
    }

    /// Appends samples as frame records (plain writes; they become
    /// acknowledged when a boundary is fsynced).
    pub fn append_frames(&mut self, samples: &[f32]) -> Result<(), StoreV2Error> {
        Ok(self.writer.append_frames(samples)?)
    }

    /// Writes + fsyncs a boundary record. Returns the acknowledged sample
    /// count (§3: acknowledged = boundary fsynced).
    pub fn write_boundary(&mut self) -> Result<u64, StoreV2Error> {
        Ok(self.writer.write_boundary()?)
    }

    /// Appends `samples` in §3-sized chunks and writes the closing
    /// boundary, verifying the acknowledged count covers everything
    /// written (the write half of the WAV-import protocol). Pure journal
    /// work through the take's own writer — no store access, so it needs
    /// no store lock.
    pub fn append_and_seal(&mut self, samples: &[f32]) -> Result<(), StoreV2Error> {
        for chunk in samples.chunks(4096) {
            self.append_frames(chunk)?;
        }
        let acked = self.write_boundary()?;
        if acked != samples.len() as u64 {
            return Err(StoreV2Error::Invalid(format!(
                "journal acknowledged {acked} samples for {} written",
                samples.len()
            )));
        }
        Ok(())
    }

    /// §4 step 2 (first half): trailer with length + content hash, fsync
    /// the file, fsync the staging directory. Consumes the writer; the
    /// take is finalized-but-unpromoted until
    /// [`StoreV2::promote_from_staging`].
    pub fn finalize(mut self) -> Result<FinalizedTake, StoreV2Error> {
        self.writer.finalize()?;
        Ok(FinalizedTake {
            id: self.writer.id().to_string(),
            sample_rate: self.writer.sample_rate(),
            total_samples: self.writer.total_samples(),
            content_hash: format!("{:016x}", self.writer.content_hash()),
            created_utc: self.created_utc,
            meta: self.meta,
        })
    }

    /// §4 steps 2–4 in order: finalize → promote → commit. The returned
    /// [`CommittedTake`] is the durable ack — it exists only after the
    /// SQLite commit.
    pub fn finish(self, store: &mut StoreV2) -> Result<CommittedTake, StoreV2Error> {
        self.finalize()?.commit_marked(store, CommitMark::Complete)
    }
}

impl FinalizedTake {
    /// §4 steps 2 (second half) – 4: promote out of staging, commit the
    /// `captures` row (one transaction), sweep staging. The commit is
    /// where the row first exists — with [`CommitMark::Interrupted`] the
    /// status and recovery note land in that same transaction, so no crash
    /// window can leave a salvaged take looking complete (R34).
    pub fn commit_marked(
        self,
        store: &mut StoreV2,
        mark: CommitMark,
    ) -> Result<CommittedTake, StoreV2Error> {
        let (status, extra_json) = match mark {
            CommitMark::Complete => (CaptureStatus::Complete, self.meta.extra_json),
            CommitMark::Interrupted { note } => (
                CaptureStatus::Interrupted,
                Some(merge_extra_note(self.meta.extra_json.as_deref(), &note)),
            ),
        };
        // Only a complete take is transcribed by itself: a salvaged one
        // waits for the user. The intent is noted before the audio moves,
        // so a crash before the commit below cannot lose it (#220).
        let transcribe = self.meta.transcribe && status == CaptureStatus::Complete;
        if transcribe {
            store.note_pending_intent(&self.id)?;
        }
        store.promote_from_staging(&self.id)?;
        let record = CaptureRecord {
            id: self.id.clone(),
            created_utc: self.created_utc.clone(),
            tz: self.meta.tz.clone(),
            device: self.meta.device.clone(),
            actual_rate: self.sample_rate,
            policy: self.meta.policy.clone(),
            frame_count: self.total_samples,
            ack_sample_index: self.total_samples,
            journal_hash: self.content_hash,
            status,
            retention_class: self.meta.retention_class.clone(),
            extra_json,
            secure_field: self.meta.secure_field,
        };
        store.commit_capture_superseding(
            &record,
            self.meta.supersedes_journal.as_deref(),
            transcribe,
            false,
        )?;
        store.gc_staging()?;
        Ok(CommittedTake { record })
    }
}

// ---------------------------------------------------------------------------
// Supporting types.
// ---------------------------------------------------------------------------

/// One record as [`StoreV2::list_records`] reports it: a readable capture
/// row (with any per-record audio problems surfaced as reasons) or a
/// damaged row that could not be mapped at all. Damaged rows never abort
/// the listing (G02).
#[derive(Clone, Debug)]
pub enum ListedCapture {
    Capture(CaptureListing),
    Damaged(DamagedCaptureV2),
}

#[derive(Clone, Debug)]
pub struct CaptureListing {
    pub record: CaptureRecord,
    /// Reasons the committed audio is currently unusable (missing file,
    /// bad header). Empty for a healthy take.
    pub problems: Vec<String>,
    /// The form the audio is kept in (#342).
    pub audio: AudioAtRest,
}

/// How a committed take's audio is kept (#342).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioAtRest {
    /// The original sample journal (not compressed yet).
    Journal,
    /// Lossless FLAC of the take's request PCM16.
    Flac,
    /// Removed by the retention policy at `utc`; the row, transcripts,
    /// revisions and correction records are kept.
    Retired { utc: String },
    /// Neither file exists and no retention stamp explains it.
    Missing,
}

#[derive(Clone, Debug)]
pub struct DamagedCaptureV2 {
    pub id: String,
    pub reason: String,
}

impl ListedCapture {
    /// The record's id, whatever its state.
    pub fn id(&self) -> &str {
        match self {
            ListedCapture::Capture(listing) => &listing.record.id,
            ListedCapture::Damaged(damaged) => &damaged.id,
        }
    }
}

/// One page of a metadata-only listing.
#[derive(Clone, Debug)]
pub struct CapturePage {
    pub records: Vec<ListedCapture>,
    pub total: usize,
    pub offset: usize,
}

/// Lazily loaded, checksum-verified journal audio.
#[derive(Clone, Debug)]
pub struct JournalAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    /// True when the file ends with its valid trailer.
    pub finalized: bool,
    /// Bytes past the last valid verification point (a torn tail).
    pub torn_tail_bytes: u64,
}

/// What reconciliation found and did.
#[derive(Debug, Default)]
pub struct ReconciliationReport {
    /// Staging journals with a torn tail: sealed to the verified prefix,
    /// promoted, and given an interrupted row with a gap note.
    pub recovered_torn: Vec<RecoveredTake>,
    /// Finalized journals still in staging: promoted into `audio/`.
    pub promoted_finalized: Vec<String>,
    /// Finalized audio with no row: an interrupted row was created
    /// (linked to the audio).
    pub orphan_sessions: Vec<String>,
    /// Rows whose audio is gone: marked interrupted.
    pub marked_interrupted: Vec<String>,
    /// Tombstoned ids whose interrupted deletion was completed.
    pub completed_deletes: Vec<String>,
    /// Journals with zero verified samples: kept in place, no row.
    pub empty_journals: Vec<String>,
    /// `(id, reason)` for files that could not be parsed: kept in place.
    pub unreadable: Vec<(String, String)>,
    /// Takes whose compression stopped after the FLAC was published:
    /// the FLAC verified against the journal, and the journal was
    /// removed (#342). Housekeeping, not a finding.
    pub completed_compressions: Vec<String>,
    /// Takes whose retention removal was stamped but not yet unlinked:
    /// the unlink was finished (#342). Housekeeping, not a finding.
    pub completed_retirements: Vec<String>,
    /// Recorder journals left in `audio/` by an adoption whose commit
    /// failed, after the take was stored from its samples instead and
    /// read back holding all of them: moved to `journals/superseded/`
    /// (#356). Housekeeping, not a finding.
    pub superseded_journals: Vec<String>,
}

impl ReconciliationReport {
    /// Whether anything user-facing happened (the app surfaces a summary
    /// only when there is something to say).
    pub fn has_findings(&self) -> bool {
        !self.recovered_torn.is_empty()
            || !self.promoted_finalized.is_empty()
            || !self.orphan_sessions.is_empty()
            || !self.marked_interrupted.is_empty()
            || !self.completed_deletes.is_empty()
            || !self.unreadable.is_empty()
    }

    /// One-line summary for the app's error banner, in the voice of the v1
    /// recovery report: what was recovered, what was flagged, what was
    /// kept. Counts only — per-record detail lives in the report fields.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        let recovered =
            self.recovered_torn.len() + self.promoted_finalized.len() + self.orphan_sessions.len();
        if recovered > 0 {
            parts.push(format!(
                "recovered {recovered} interrupted recording{} into history",
                if recovered == 1 { "" } else { "s" }
            ));
        }
        if !self.marked_interrupted.is_empty() {
            parts.push(format!(
                "{} recording{} marked interrupted (audio missing)",
                self.marked_interrupted.len(),
                if self.marked_interrupted.len() == 1 {
                    ""
                } else {
                    "s"
                }
            ));
        }
        if !self.completed_deletes.is_empty() {
            parts.push(format!(
                "completed {} pending deletion{}",
                self.completed_deletes.len(),
                if self.completed_deletes.len() == 1 {
                    ""
                } else {
                    "s"
                }
            ));
        }
        if !self.unreadable.is_empty() {
            parts.push(format!(
                "{} journal file{} could not be read and were left in place",
                self.unreadable.len(),
                if self.unreadable.len() == 1 { "" } else { "s" }
            ));
        }
        format!("Startup scan: {}.", parts.join("; "))
    }
}

/// One torn-take recovery, with the discarded tail size (the gap).
#[derive(Debug, Clone)]
pub struct RecoveredTake {
    pub id: String,
    pub torn_tail_bytes: u64,
}

/// What one explicit [`StoreV2::sweep_retention`] run removed and kept.
#[derive(Debug, Default)]
pub struct SweepReport {
    /// Tombstoned content unlinked, in sweep order: the id, which tree it
    /// came from (`capture` = v2 `quarantine/`, `journal` =
    /// `journals/superseded/` or `journals/deleted/`), and the bytes freed.
    pub swept: Vec<SweptFile>,
    /// Total bytes unlinked.
    pub swept_bytes: u64,
    /// The `stop` of [`StoreV2::sweep_retention_until`] said so: the
    /// sweep ended before every tree was done.
    pub stopped: bool,
    /// `(name, reason)` for entries left in place: files that are not
    /// `.sj` journals or `.flac` audio, superseded journals no longer
    /// proven copies, deleted audio still pinned, or removals that failed
    /// (the next sweep retries).
    pub retained: Vec<(String, String)>,
}

/// One swept tombstoned file.
#[derive(Debug, Clone)]
pub struct SweptFile {
    pub id: String,
    /// `capture` (v2 `quarantine/`) or `journal` (`journals/superseded/`,
    /// `journals/deleted/`).
    pub kind: String,
    pub bytes: u64,
}

/// A take whose journal may be replaced by FLAC (#342), from
/// [`StoreV2::compression_candidates`].
#[derive(Clone, Debug)]
pub struct CompressionJob {
    pub id: String,
    journal: PathBuf,
}

/// A verified FLAC encode waiting under `audio/` as a temporary, ready
/// for [`StoreV2::commit_compression`].
#[derive(Debug)]
pub struct PreparedCompression {
    id: String,
    journal: PathBuf,
    temp: PathBuf,
    journal_bytes: u64,
    flac_bytes: u64,
}

impl PreparedCompression {
    /// Abandon the encode: remove its temporary.
    pub fn discard(self) {
        let _ = std::fs::remove_file(&self.temp);
    }
}

/// What one compression did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompressionOutcome {
    /// The journal was replaced by its FLAC.
    Compressed { journal_bytes: u64, flac_bytes: u64 },
    /// Nothing changed, and why.
    Skipped(String),
}

/// The heavy, store-free half of compression (#342): read and verify the
/// journal, derive its request PCM16 ([`request_pcm16`] — exactly what
/// every transcription of the take receives), encode FLAC into a unique
/// temporary beside the journal, fsync it, and decode it back from disk
/// with the independent decoder, refusing anything that is not sample-
/// identical. A caller sharing the store behind a lock runs this without
/// the lock and hands the result to [`StoreV2::commit_compression`].
/// On error, no temporary is left behind.
pub fn prepare_compression(job: &CompressionJob) -> Result<PreparedCompression, StoreV2Error> {
    let parsed = read_journal(&job.journal).map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
    if !parsed.finalized || parsed.torn_tail_bytes > 0 {
        // Journals under audio/ are sealed by construction; anything else
        // is evidence for a human, not input for an encoder.
        return Err(StoreV2Error::Invalid(format!(
            "journal {} is not sealed; it is kept as it is",
            job.journal.display()
        )));
    }
    let journal_bytes = std::fs::metadata(&job.journal)?.len();
    let pcm = request_pcm16(&parsed.samples, parsed.sample_rate)?;
    drop(parsed);
    let encoded = flac::encode(&pcm)?;
    let temp = job.journal.with_file_name(format!(
        "{}.{}.{FLAC_TEMP_EXT}",
        job.id,
        uuid::Uuid::new_v4().simple()
    ));
    let written = (|| -> Result<(), StoreV2Error> {
        use std::io::Write;
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)?;
        file.write_all(&encoded)?;
        file.sync_all()?;
        drop(file);
        let decoded = flac::decode(io::BufReader::new(File::open(&temp)?))?;
        if decoded != pcm {
            return Err(StoreV2Error::Invalid(format!(
                "FLAC of capture {} did not decode to the journal's samples; the journal is kept",
                job.id
            )));
        }
        Ok(())
    })();
    if let Err(err) = written {
        let _ = std::fs::remove_file(&temp);
        return Err(err);
    }
    Ok(PreparedCompression {
        id: job.id.clone(),
        journal: job.journal.clone(),
        temp,
        journal_bytes,
        flac_bytes: encoded.len() as u64,
    })
}

/// Age and size limits for one retention class (#342). `None` is no
/// limit; both `None` leaves the class alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassLimits {
    pub max_age_days: Option<u32>,
    pub max_total_bytes: Option<u64>,
}

impl ClassLimits {
    pub fn is_active(&self) -> bool {
        self.max_age_days.is_some() || self.max_total_bytes.is_some()
    }
}

/// The user's retention policy (#342). The default is off: no class has
/// a limit, and [`StoreV2::apply_retention_policy`] does nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Limits per retention class name ([`STANDARD_CLASS`],
    /// [`ARCHIVAL_CLASS`], …).
    pub limits: std::collections::BTreeMap<String, ClassLimits>,
    /// Also remove audio a document revision or a correction record
    /// references. Off unless the user agreed to it.
    pub include_referenced: bool,
    /// No take younger than this loses its audio.
    pub grace: std::time::Duration,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            limits: std::collections::BTreeMap::new(),
            include_referenced: false,
            grace: DEFAULT_RETENTION_GRACE,
        }
    }
}

impl RetentionPolicy {
    /// Whether any class has a limit.
    pub fn is_active(&self) -> bool {
        self.limits.values().any(ClassLimits::is_active)
    }
}

/// What one [`StoreV2::apply_retention_policy`] run removed and held.
#[derive(Debug, Default)]
pub struct RetentionReport {
    pub retired: Vec<RetiredAudio>,
    pub retired_bytes: u64,
    /// Due takes that kept their audio, with the reason.
    pub held: Vec<HeldAudio>,
    /// `(class, bytes)` a size limit is still exceeded by after the run
    /// (held takes count toward their class).
    pub over_limit: Vec<(String, u64)>,
    /// The policy changed while the run went (the user saved other
    /// limits): it stopped before removing anything the new policy may
    /// not want removed. The caller runs again with the new one.
    pub policy_changed: bool,
    /// The caller's `stop` said yes before a removal, and the run ended.
    pub stopped: bool,
}

/// One take whose audio the policy removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetiredAudio {
    pub id: String,
    pub class: String,
    pub bytes: u64,
    pub reason: RetireReason,
}

/// Which limit made a take due.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetireReason {
    Age,
    Size,
}

/// One due take that kept its audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldAudio {
    pub id: String,
    pub class: String,
    pub bytes: u64,
    pub reason: HoldReason,
}

/// Why a due take keeps its audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldReason {
    /// Younger than the policy's grace.
    Recent,
    /// A recognition attempt is in flight on it.
    InUse,
    /// No attempt ever produced a transcript: the audio is all there is.
    Untranscribed,
    /// Document revisions or correction records reference it.
    Referenced { revisions: u32, corrections: u32 },
}

// ---------------------------------------------------------------------------
// Free helpers.
// ---------------------------------------------------------------------------

/// A local UTC-offset label for the `tz` column; `UTC` when the local
/// offset cannot be determined (multi-threaded sandbox).
fn local_tz_label() -> String {
    time::OffsetDateTime::now_local()
        .map(|now| now.offset().to_string())
        .unwrap_or_else(|_| "UTC".to_string())
}

/// The `recognition_attempts` INSERT shared by
/// [`StoreV2::insert_attempt`] and [`StoreV2::begin_recognition`] (which
/// runs it inside its own write-locked transaction).
fn insert_attempt_row(conn: &Connection, attempt: &AttemptRecord) -> Result<(), StoreV2Error> {
    conn.execute(
        "INSERT INTO recognition_attempts(
            id, capture_id, backend, model_hash, language, options_json, text,
            partial_or_final, status, timing_json, extra_json, created_utc)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            attempt.id,
            attempt.capture_id,
            attempt.backend,
            attempt.model_hash,
            attempt.language,
            attempt.options_json,
            attempt.text,
            attempt.partial_or_final,
            attempt.status,
            attempt.timing_json,
            attempt.extra_json,
            attempt.created_utc.clone().unwrap_or_else(now_iso),
        ],
    )?;
    Ok(())
}

fn validate_capture_id(id: &str) -> Result<(), StoreV2Error> {
    if !is_safe_path_component(id) {
        return Err(StoreV2Error::Invalid(format!(
            "capture id {id:?} must be non-empty and contain no path separators"
        )));
    }
    Ok(())
}

/// Document and revision ids are pure SQLite keys (never path
/// components, unlike capture ids), so they carry no separator rule —
/// but they are client-chosen strings, so a non-empty cap keeps a
/// hostile envelope from making the key the row's only bulk.
fn validate_document_id(id: &str) -> Result<(), StoreV2Error> {
    if id.is_empty() || id.len() > 1024 {
        return Err(StoreV2Error::Invalid(format!(
            "document/revision id must be non-empty and at most 1024 bytes (got {})",
            id.len()
        )));
    }
    Ok(())
}

/// What a lock probe ([`try_flock_exclusive`]) learned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlockEvidence {
    /// The probe acquired the exclusive lock itself (held until the
    /// handle closes): no live holder — the OS frees the lock when its
    /// holder dies.
    Free,
    /// A live holder has the lock.
    Held,
    /// The platform has no lock primitive.
    #[cfg_attr(any(unix, windows), allow(dead_code))]
    Unknown,
}

/// Take a non-blocking exclusive advisory lock on an open file. The lock
/// lives on the open file description, so the OS releases it when the
/// handle closes or its process dies.
#[cfg(unix)]
pub(crate) fn try_flock_exclusive(file: &File) -> io::Result<FlockEvidence> {
    use std::os::fd::AsRawFd;
    // SAFETY: flock(2) on an fd this caller owns and keeps open for the
    // lock's lifetime; no close or hand-off happens here.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(FlockEvidence::Free);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::WouldBlock {
        return Ok(FlockEvidence::Held);
    }
    Err(error)
}

/// Windows: the same probe with `LockFileEx` (exclusive, fail
/// immediately) — released when the holding handle closes or its process
/// dies. The locked byte lies far past EOF on purpose: Windows byte-range
/// locks are mandatory, and recovery reads a journal it probed through a
/// second handle.
#[cfg(windows)]
pub(crate) fn try_flock_exclusive(file: &File) -> io::Result<FlockEvidence> {
    Ok(if windows_lock::try_lock(file)? {
        FlockEvidence::Free
    } else {
        FlockEvidence::Held
    })
}

/// Without any lock primitive there is nothing to ask: the probe answers
/// [`FlockEvidence::Unknown`].
#[cfg(not(any(unix, windows)))]
pub(crate) fn try_flock_exclusive(_file: &File) -> io::Result<FlockEvidence> {
    Ok(FlockEvidence::Unknown)
}

/// The Windows lock primitive behind [`try_flock_exclusive`].
#[cfg(windows)]
mod windows_lock {
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION;
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    /// High dword of the locked offset (2^62): far past any real file
    /// content, so the mandatory lock never covers bytes anyone reads.
    const LOCK_OFFSET_HIGH: u32 = 0x4000_0000;

    /// `Ok(true)` = this handle now holds the lock; `Ok(false)` = another
    /// handle (this process or another) holds it.
    pub(super) fn try_lock(file: &File) -> io::Result<bool> {
        // SAFETY: a zeroed OVERLAPPED is valid input; `std` opens files
        // synchronously, so LockFileEx with FAIL_IMMEDIATELY completes
        // before returning and the OVERLAPPED (only carrying the offset)
        // is not referenced afterwards. The handle stays open for the
        // call (borrowed from `file`).
        let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
        overlapped.Anonymous.Anonymous.OffsetHigh = LOCK_OFFSET_HIGH;
        let locked = unsafe {
            LockFileEx(
                file.as_raw_handle(),
                LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                0,
                1,
                0,
                &mut overlapped,
            )
        };
        if locked != 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
            return Ok(false);
        }
        Err(error)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The Win32 docs say `LockFileEx` wants a handle opened with
        /// GENERIC_READ *or* GENERIC_WRITE; the journal recovery probe
        /// opens read-only and leans on that. Pin it: the first read-only
        /// handle locks (`Ok(true)`) and a second one on the same file
        /// answers held (`Ok(false)`), not an error.
        #[test]
        fn readonly_handles_take_and_hold_the_exclusive_lock() {
            let temp = tempfile::NamedTempFile::new().expect("a temp file");
            let first = File::open(temp.path()).expect("the first read-only open");
            assert!(
                try_lock(&first).expect("the first read-only handle locks"),
                "LockFileEx refused a GENERIC_READ-only handle"
            );
            let second = File::open(temp.path()).expect("the second read-only open");
            match try_lock(&second) {
                Ok(held) => assert!(
                    !held,
                    "the second read-only handle must see the first one's lock"
                ),
                Err(err) => {
                    panic!(
                        "the contended read-only probe errored instead of answering Held: {err}"
                    )
                }
            }
        }
    }
}

fn int64(value: u64) -> Result<i64, StoreV2Error> {
    i64::try_from(value)
        .map_err(|_| StoreV2Error::Invalid(format!("value {value} exceeds SQLite INTEGER")))
}

/// Merge a recovery note into a row's `extra_json` object without dropping
/// existing keys (known or unknown). `NULL` extra becomes a fresh object.
fn merge_extra_note(current: Option<&str>, note: &str) -> String {
    let mut map = current
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    map.insert("recovery".to_string(), serde_json::Value::String(note.to_string()));
    serde_json::to_string(&serde_json::Value::Object(map))
        .expect("a JSON object serializes")
}

/// Read and verify one audio journal from disk (the lock-free half of
/// [`StoreV2::load_audio`]). No store state is touched — only the file at
/// `path` — so a caller sharing the store behind a lock resolves the path
/// via [`StoreV2::audio_journal_path`] under the guard and does this work
/// without it.
///
/// A `.flac` path decodes the compressed take (#342). A journal that is
/// gone by the time it is opened — compression replaced it with its
/// verified FLAC after the caller resolved the path — falls over to that
/// FLAC, so a read racing compression never fails.
pub fn read_audio_journal(path: &Path) -> Result<JournalAudio, StoreV2Error> {
    if path.extension().and_then(|ext| ext.to_str()) == Some(flac::FLAC_EXT) {
        return read_flac_audio(path);
    }
    match read_journal(path) {
        Ok(parsed) => Ok(JournalAudio {
            sample_rate: parsed.sample_rate,
            samples: parsed.samples,
            finalized: parsed.finalized,
            torn_tail_bytes: parsed.torn_tail_bytes,
        }),
        Err(journal::JournalReadError::Io(err)) if err.kind() == io::ErrorKind::NotFound => {
            let compressed = path.with_extension(flac::FLAC_EXT);
            if compressed.exists() {
                read_flac_audio(&compressed)
            } else {
                Err(StoreV2Error::Invalid(err.to_string()))
            }
        }
        Err(err) => Err(StoreV2Error::Invalid(err.to_string())),
    }
}

/// Decode a compressed take into the shape a journal read returns: the
/// request PCM16 at 16 kHz, mapped back through the exact inverse of the
/// quantizer so re-encoding reproduces the same request WAV.
fn read_flac_audio(path: &Path) -> Result<JournalAudio, StoreV2Error> {
    let samples = File::open(path)
        .map_err(StoreV2Error::from)
        .and_then(|file| Ok(flac::decode(io::BufReader::new(file))?))
        .map_err(|err| StoreV2Error::Invalid(format!("{}: {err}", path.display())))?;
    Ok(JournalAudio {
        sample_rate: STARLING_SAMPLE_RATE,
        samples: samples.into_iter().map(pcm16_to_f32).collect(),
        finalized: true,
        torn_tail_bytes: 0,
    })
}

/// Sorted, deduplicated stems of `.sj` journals and `.flac` audio under
/// `dir` (missing dir = empty) — every id with audio in an `audio/` or
/// `quarantine/` tree (#342).
fn audio_ids_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("sj") | Some(flac::FLAC_EXT)
            )
        })
        .filter_map(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// Sorted `.sj` stems under `dir` (missing dir = empty).
fn journal_ids_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("sj"))
        .filter_map(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .collect();
    ids.sort();
    ids
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

mod transcription;
pub use transcription::TranscriptionClaim;

#[cfg(test)]
mod at_rest_tests;

#[cfg(test)]
mod transcription_tests;

#[cfg(test)]
mod journal_recovery_tests;

/// Staging salvages that die at a chosen step (the store's own fault
/// injection). The crash protocol itself is proven through the public
/// entry points in `tests/store_recovery.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Deterministic sample block `len` long.
    fn ramp(len: usize, offset: u32) -> Vec<f32> {
        (0..len)
            .map(|i| ((offset as usize + i) % 997) as f32 * 0.0001)
            .collect()
    }

    fn store_in(dir: &TempDir) -> StoreV2 {
        StoreV2::open(dir.path().join("v2")).expect("open v2 store")
    }

    /// The control: a complete take still in staging whose commit was to
    /// carry its transcription intent, and whose salvage dies at any step,
    /// is stored whole with that intent — its journal is never sealed a
    /// second time.
    #[test]
    fn a_complete_staging_salvage_that_dies_at_any_step_keeps_its_intent() {
        for step in [RecoveryStep::SealNoted, RecoveryStep::Sealed, RecoveryStep::Moved] {
            let dir = TempDir::new().expect("tempdir");
            let samples = ramp(800, 0);
            let id = {
                let mut store = store_in(&dir);
                let mut take = store
                    .begin_take(TakeMeta::for_device("mic"))
                    .expect("begin");
                take.append_and_seal(&samples).expect("append");
                let finalized = take.finalize().expect("finalize");
                // The commit's first step, then the recorder's crash.
                store.note_pending_intent(&finalized.id).expect("intent");
                store.crash_after = Some(step);
                // Nothing is noted or sealed for a complete journal; the
                // step boundaries still stop the salvage there.
                assert!(store.reconcile().is_err(), "{step:?}");
                finalized.id
            };

            let mut store = store_in(&dir);
            store.reconcile().expect("reconcile");
            let record = store.get_capture(&id).expect("get").expect("row");
            assert_eq!(record.frame_count, 800, "{step:?}");
            let audio = store.load_audio(&id).expect("audio");
            assert!(audio.finalized && audio.torn_tail_bytes == 0, "{step:?}: one valid trailer");
            assert_eq!(audio.samples, samples, "{step:?}");
            let note = record.recovery_note().expect("note");
            assert!(note.contains("finished cleanly"), "{step:?}: {note}");
            assert!(store.transcription_wanted(&id).unwrap(), "{step:?}: the intent survives");
        }
    }

    /// #220/#356: a reconcile that dies at any step of salvaging a cut
    /// staging journal — after noting the seal, after sealing it, after
    /// promoting it — leaves a take the next one still stores as
    /// interrupted with its confirmed extent, never as one that finished.
    #[test]
    fn a_staging_salvage_that_dies_at_any_step_still_reads_as_cut() {
        for step in [RecoveryStep::SealNoted, RecoveryStep::Sealed, RecoveryStep::Moved] {
            let dir = TempDir::new().expect("tempdir");
            let confirmed = ramp(800, 0);
            let id = {
                let mut store = store_in(&dir);
                let mut take = store
                    .begin_take(TakeMeta::for_device("mic"))
                    .expect("begin");
                take.append_frames(&confirmed).expect("append");
                take.write_boundary().expect("boundary");
                take.append_frames(&ramp(120, 800)).expect("unconfirmed tail");
                let id = take.id().to_string();
                drop(take); // the recorder's crash
                store.crash_after = Some(step);
                assert!(store.reconcile().is_err(), "{step:?}");
                id
                // ... and the salvage's
            };

            let mut store = store_in(&dir);
            let report = store.reconcile().expect("reconcile");
            assert!(report.promoted_finalized.is_empty(), "{step:?}: {report:?}");
            let record = store.get_capture(&id).expect("get").expect("row");
            assert_eq!(record.status, CaptureStatus::Interrupted, "{step:?}");
            assert_eq!(record.frame_count, 800, "{step:?}: only the verified prefix");
            assert_eq!(store.load_audio(&id).expect("audio").samples, confirmed, "{step:?}");
            let audio = store.load_audio(&id).expect("audio");
            assert!(audio.finalized && audio.torn_tail_bytes == 0, "{step:?}: one valid trailer");
            let note = record.recovery_note().expect("note");
            assert!(!note.contains("finished cleanly"), "{step:?}: {note}");
            assert!(note.contains("gap flagged"), "{step:?}: the torn tail is named: {note}");
            assert!(!store.transcription_wanted(&id).unwrap(), "{step:?}");
            let seals: i64 = store
                .conn
                .query_row("SELECT COUNT(*) FROM recovery_seals", [], |row| row.get(0))
                .expect("count");
            assert_eq!(seals, 0, "{step:?}: the seal is done with");
        }
    }

    // ---- hand-built WAV fixture ---------------------------------------

    // ---- daily-use operations -----------------------------------------

    // ---- review round 1: lease races and degradation ----------------------

    // ---- review round 1: staging rollback ----------------------------------

    // -----------------------------------------------------------------
    // Documents / revisions (§4 tables; the I5 documents-machine
    // persistence, issue #220).
    // -----------------------------------------------------------------

    // ---- correction records ------------------------------------------

}
