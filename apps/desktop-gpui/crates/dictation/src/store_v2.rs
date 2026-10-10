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
//! <root>/attempt-locks/ <attemptId>.lock in-flight recognition markers (#213)
//! <root>/leases/       <ownerId>.lease runtime ownership leases (§4)
//! ```
//!
//! # Ownership (§4 leases)
//!
//! [`StoreV2::acquire_lease`] is the multi-process ownership handshake: a
//! live lease (`pid` + boot id + heartbeat in `leases/`) makes every other
//! process a **client** of the data root instead of a competitor — most
//! visibly, [`StoreV2::reconcile`] run by a client defers the in-flight
//! halves of recovery (staging salvage, orphan adoption) to the live
//! owner instead of sealing and promoting journals the owner may still be
//! writing. A lease dies with its process (the OS releases its flock),
//! and a lease whose heartbeat expired past [`LEASE_HEARTBEAT_TTL`] —
//! or whose boot id no longer matches — is stale and breakable
//! ([`StoreV2::break_stale_leases`], also run inside every acquire).
//! Lease writes never share a fixed `.tmp` name: every scratch file is a
//! unique temporary named after its owner (see [`unique_lease_temp`]).
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
/// `audio_holds` (#220).
pub const SCHEMA_VERSION: u32 = 6;

/// WAL checkpoint policy (D11): run `PRAGMA wal_checkpoint(PASSIVE)` after
/// every N metadata commits. Default 64 — frequent enough that the WAL
/// cannot grow without bound across a session of takes, rare enough that
/// the checkpoint cost stays off the per-take path on slow disks. Tunable
/// via [`StoreV2::set_checkpoint_every_commits`].
pub const WAL_CHECKPOINT_EVERY_COMMITS: u32 = 64;

/// How stale a lease heartbeat may be before the lease is breakable (§4
/// ownership): on hosts where the flock cannot answer, a lease whose last
/// heartbeat is older than this reads as stale and
/// [`StoreV2::break_stale_leases`] may remove it. Where flock answers
/// (every unix desktop target), the OS-level lock decides outright — a
/// held flock means the owner lives no matter what the heartbeat says,
/// and a freed flock means it died no matter how fresh the heartbeat is.
/// Tunable via [`StoreV2::set_lease_ttl`].
pub const LEASE_HEARTBEAT_TTL: std::time::Duration = std::time::Duration::from_secs(30);

/// How young an acquisition/heartbeat temporary under `leases/` must be
/// to be untouchable by the temp sweep: a live writer's temp (between
/// create and its publish rename) is younger than this by construction,
/// so gating removal on this age closes the sweep-vs-acquirer race — a
/// concurrent sweep can never delete a temp the rename is about to
/// publish. An older temp belongs to a writer that stopped moving.
pub const LEASE_TEMP_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

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
/// Per-attempt in-flight markers (#213): one `<attemptId>.lock` file per
/// live recognition attempt, flocked by the owning process for the
/// attempt's lifetime. The startup sweep reads them to decide whether a
/// `started` row still has a live owner.
const ATTEMPT_LOCKS_DIR: &str = "attempt-locks";
/// Runtime ownership leases (§4): one `<ownerId>.lease` identity file and
/// one `<ownerId>.hb` heartbeat file per process holding the data root,
/// the identity flocked for the owner's lifetime.
const LEASES_DIR: &str = "leases";
/// The fixed-name acquisition sentinel under `leases/` (see
/// [`LeaseSentinel`]): flocked across probe-and-publish, never renamed,
/// never carrying content.
const LEASE_SENTINEL_FILE: &str = ".lock";
/// The v1 journal tree's tombstone directory (R21), a sibling of the v2
/// root: `<root>/journals/deleted/`. The v1 store quarantined deleted
/// journals here pending the retention sweep; the sweep still empties it
/// (never-delete-until-swept), but nothing writes into it anymore.
const LEGACY_DELETED_SUBPATH: &str = "journals/deleted";
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
    /// In-flight attempt markers this process holds (#213): attempt id →
    /// the flocked marker file. The flock is the cross-process ownership
    /// signal the startup sweep consults — the file-store analog of the
    /// Electron reference's per-attempt Web Lock. Dropping the handle
    /// (settle, delete, or process exit — crash included) releases the
    /// lock, so a dead owner's marker can never block a later sweep.
    attempt_locks: HashMap<String, File>,
    /// The runtime lease this instance holds (§4), when it is the owner.
    lease: Option<LeaseHandle>,
    /// How stale a foreign lease's heartbeat may be before the lease is
    /// breakable (D11-style tunable; default [`LEASE_HEARTBEAT_TTL`]).
    lease_ttl: std::time::Duration,
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
}

#[cfg(test)]
struct TestHook(Box<dyn FnOnce() + Send>);

#[cfg(test)]
impl std::fmt::Debug for TestHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TestHook")
    }
}

/// The lease this process holds: the owner id and the flocked lease file
/// (keeping the file open is what holds the flock, exactly like the
/// attempt markers). The flock outlives the file's name — acquisition
/// publishes the record with a rename, and the lock rides the inode.
#[derive(Debug)]
struct LeaseHandle {
    owner_id: String,
    file: File,
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
        std::fs::create_dir_all(root.join(ATTEMPT_LOCKS_DIR))?;
        std::fs::create_dir_all(root.join(LEASES_DIR))?;

        let mut conn = Connection::open(root.join(DB_FILE))?;
        let mode: String = conn.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(StoreV2Error::Invalid(format!(
                "could not enable WAL journal mode (got {mode:?})"
            )));
        }
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // Multi-process access is a designed state now (§4 leases: a
        // second process is a client, not an error). A short busy timeout
        // turns a transient SQLITE_BUSY from a peer's commit into a wait
        // instead of a hard error — on every write path, including the
        // sweep's tombstone stamp.
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
            attempt_locks: HashMap::new(),
            lease: None,
            lease_ttl: LEASE_HEARTBEAT_TTL,
            audio_pins: HashMap::new(),
            compression_failures: HashMap::new(),
            #[cfg(test)]
            before_retention_lock: None,
            #[cfg(test)]
            before_sweep_lock: None,
        })
    }

    /// `<data-dir>/starling-gpui` — the single data root. The recorder's
    /// live journals still scratch under `journals/` beside it (adopted
    /// here at save time), and the v1 store's tombstone tree
    /// `journals/deleted/` is swept here (see [`Self::sweep_retention`]).
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
        self.commit_capture_superseding(record, None, false)
    }

    /// [`Self::commit_capture`] that also records, in the same
    /// transaction, the recorder journal the take is stored in place of
    /// ([`TakeMeta::supersedes_journal`]) and, when `transcribe` (the
    /// callers set it only for a take whose audio is complete), the intent
    /// to transcribe it ([`TakeMeta::transcribe`]).
    fn commit_capture_superseding(
        &mut self,
        record: &CaptureRecord,
        supersedes_journal: Option<&str>,
        transcribe: bool,
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
            self.release_attempt_lock(attempt_id);
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
    /// - `journals/deleted/` — the v1 journal tree's tombstones (R21),
    ///   left behind by the deleted v1 store, still awaiting this sweep,
    ///   and recorder journals of takes the user deleted, whose ids are
    ///   tombstoned already (#356);
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
            &self.root.join(LEGACY_DELETED_SUBPATH),
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
    /// the two populations that reach the sweep with **no** tombstone row
    /// of their own: the legacy `journals/deleted/` tree (v1 never wrote
    /// v2 rows — the sweep is their only stamper) and quarantine files
    /// from `delete_capture`'s own crash window (rename committed,
    /// transaction not — the shape `reconcile`'s `complete_tombstoned`
    /// heals). Reconcile's dead set reads `tombstones` rows ∪ quarantine
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
        let skip = |reason: &str| {
            let _ = std::fs::remove_file(&prepared.temp);
            Ok(CompressionOutcome::Skipped(reason.to_string()))
        };
        let id = prepared.id.as_str();
        // The checks and the publish hold the database's write lock, so
        // no other connection can retire, delete or start a transcription
        // of the take in between.
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
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
    /// crash during reconciliation is repaired by the next run.
    ///
    /// **Client mode** (§4 ownership): when this instance holds no lease
    /// and a live foreign owner does, the in-flight halves of recovery —
    /// staging salvage and orphan-session adoption — are **deferred to the
    /// owner** (named in [`ReconciliationReport::deferred_to_live_owner`]):
    /// a staging journal may be a take the owner is writing right now, and
    /// sealing + promoting it out from under a live writer is exactly the
    /// competitor behavior leases exist to prevent. The row-side repairs
    /// (tombstone completion, missing-audio marking) still run — they are
    /// idempotent and cannot touch an in-flight take, whose row does not
    /// exist yet. Recognition attempts are guarded separately, per
    /// attempt, by the #213 markers.
    ///
    /// A lease file that exists but cannot be probed at all still reads
    /// as a live owner (never break what cannot be proven dead) and so
    /// still defers — but it is surfaced in
    /// [`ReconciliationReport::unreadable_leases`], because a single
    /// corrupt lease file must not silently disable crash recovery for
    /// the whole root without a trace.
    pub fn reconcile(&mut self) -> Result<ReconciliationReport, StoreV2Error> {
        let mut report = ReconciliationReport::default();
        let ownership = self.live_foreign_lease()?;
        let foreign_owner = ownership.live;
        report.unreadable_leases = ownership.unreadable;
        // An unanswerable lease defers like a live one (it may be a live
        // owner; sealing its staging would be the competitor behavior),
        // but — unlike a plain deferral — it is a finding: recovery stays
        // disabled until the file is removed or repaired.
        let client_mode = foreign_owner.is_some() || !report.unreadable_leases.is_empty();

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
            if client_mode {
                // Client mode: the live (or unanswerable) owner may be
                // writing this take right now — its salvage is the
                // owner's to run.
                report.deferred_to_live_owner.push(id);
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
            let torn_tail_bytes = parsed.torn_tail_bytes;
            let sample_rate = parsed.sample_rate;
            let count = parsed.samples.len() as u64;
            let hash = format!("{:016x}", samples_hash(&parsed.samples));

            // Seal the verified prefix (§4: truncate + gap flag) and
            // promote. `seal_recovered_journal` is idempotent, so a crash
            // mid-recovery re-runs cleanly.
            seal_recovered_journal(&path, &parsed)?;
            self.promote_from_staging(&id)?;
            if parsed.finalized && torn_tail_bytes == 0 {
                report.promoted_finalized.push(id.clone());
            }

            let note = if torn_tail_bytes > 0 {
                format!(
                    "Recovered from an interrupted take: the last {torn_tail_bytes} bytes of \
                     the journal were an unfinished write and were discarded (gap flagged, \
                     never joined)."
                )
            } else if parsed.finalized {
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
                }
                None => {
                    let record = CaptureRecord {
                        extra_json: Some(merge_extra_note(None, &note)),
                        ..record
                    };
                    self.commit_capture(&record)?;
                }
            }
            report.recovered_torn.push(RecoveredTake {
                id,
                torn_tail_bytes,
            });
        }

        // --- audio side ---
        if !client_mode {
            // Abandoned FLAC encodes (#342): the journal beside each is
            // intact, so the temporary is only scratch.
            self.remove_stale_flac_temps();
        }
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
                        // FLAC and removing the journal (or a peer is in
                        // that window right now: defer to the owner).
                        if client_mode {
                            report.deferred_to_live_owner.push(id);
                        } else {
                            self.finish_interrupted_compression(&id, &mut report)?;
                        }
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
                    // interrupted status, linked to the audio. In client
                    // mode this is deferred too — the owner may sit in the
                    // finalize→commit window, and racing an adoption into
                    // it would make the owner's own commit fail.
                    if client_mode {
                        report.deferred_to_live_owner.push(id);
                        continue;
                    }
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
            report.superseded_journals.push(id);
            return Ok(false);
        }
        let note = if parsed.finalized {
            "Recovered from a take that finished cleanly but was never \
             committed to the library (crash between finalize and the \
             metadata commit)."
                .to_string()
        } else if parsed.torn_tail_bytes > 0 {
            format!(
                "Recovered from an orphaned journal with a torn tail of {} \
                 bytes (gap flagged, never joined).",
                parsed.torn_tail_bytes
            )
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
        self.commit_capture(&record)?;
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
                self.release_attempt_lock(attempt_id);
            }
        }
        report.completed_deletes.push(id.to_string());
        Ok(())
    }

    // ------------------------------------------------------------------
    // Multi-process ownership (§4 leases).
    // ------------------------------------------------------------------

    /// The lease identity file path for one owner.
    fn lease_path(&self, owner_id: &str) -> PathBuf {
        self.root.join(LEASES_DIR).join(format!("{owner_id}.lease"))
    }

    /// The lease heartbeat file path for one owner.
    fn heartbeat_path(&self, owner_id: &str) -> PathBuf {
        self.root.join(LEASES_DIR).join(format!("{owner_id}.hb"))
    }

    /// Tune the stale-lease heartbeat TTL (takes effect on the next
    /// liveness probe; the default is [`LEASE_HEARTBEAT_TTL`]).
    pub fn set_lease_ttl(&mut self, ttl: std::time::Duration) {
        self.lease_ttl = ttl.max(std::time::Duration::from_secs(1));
    }

    /// Acquire the data root's runtime lease (§4 ownership). Outcomes:
    ///
    /// - [`LeaseAcquisition::Owner`] — this instance now owns the root.
    ///   Stale leases are broken first ([`Self::break_stale_leases`]); the
    ///   ids broken in the process ride along. Re-acquiring while already
    ///   the owner renews the heartbeat and keeps the same owner id.
    /// - [`LeaseAcquisition::Client`] — a live foreign owner holds the
    ///   root; this process is a **client, not a competitor**: it must not
    ///   run the mutating halves of startup recovery against the owner's
    ///   in-flight state ([`Self::reconcile`] already defers them), and
    ///   per-attempt recognition ownership stays guarded by the #213
    ///   markers as before.
    /// - [`LeaseAcquisition::UnanswerableLeases`] — a lease file exists
    ///   that can be neither probed nor broken (it will not open or
    ///   parse, and no evidence proves its owner dead). It reads as an
    ///   unanswerable owner — [`Self::reconcile`] would defer to it
    ///   forever — so this instance does **not** publish alongside it:
    ///   becoming a second writer whose own reconciles then defer
    ///   indefinitely is the recovery-disabling state the unreadable
    ///   surface exists to prevent, self-inflicted. The caller surfaces
    ///   the state (repair, warn, or wait) exactly like
    ///   [`ReconciliationReport::unreadable_leases`].
    ///
    /// The probe-and-publish critical section is serialized by the
    /// `leases/.lock` sentinel (a bounded-wait flock, never renamed, so it
    /// is not the fixed-`.tmp` collision this design removes), so two
    /// concurrent acquirers cannot both observe "no live owner" and both
    /// publish. On hosts where flock cannot serialize, a post-publish
    /// tie-break re-check closes the remainder deterministically: if a
    /// live foreign lease appeared with a **smaller** owner id, this
    /// instance releases its own and answers `Client` — both racers
    /// compute the same order, so exactly one stays owner.
    ///
    /// The lease is held until [`Self::release_lease`], the store is
    /// dropped, or the process dies — the flock on the identity file is
    /// the ownership signal, and the OS releases it when the owner dies,
    /// so a crashed owner's lease file is harmless and breakable.
    pub fn acquire_lease(&mut self) -> Result<LeaseAcquisition, StoreV2Error> {
        if let Some(handle) = &self.lease {
            // Already the owner: renew and keep the identity.
            let owner_id = handle.owner_id.clone();
            self.renew_lease_record()?;
            return Ok(LeaseAcquisition::Owner {
                owner_id,
                broke: Vec::new(),
            });
        }
        let leases_dir = self.root.join(LEASES_DIR);
        std::fs::create_dir_all(&leases_dir)?;
        // Serialize probe+publish against concurrent acquirers (the
        // sentinel is dropped — lock released — when this scope ends).
        let _sentinel = LeaseSentinel::acquire(&leases_dir)?;

        let broke = self.break_stale_leases_under_sentinel()?;
        let ownership = self.live_foreign_lease()?;
        if let Some(owner) = ownership.live {
            return Ok(LeaseAcquisition::Client { owner });
        }
        if !ownership.unreadable.is_empty() {
            // An unanswerable lease is never broken and still defers
            // recovery in reconcile; becoming owner on top of it would
            // leave this process writing while its own reconciles defer
            // forever. Surface it instead of publishing a second owner.
            return Ok(LeaseAcquisition::UnanswerableLeases {
                unreadable: ownership.unreadable,
            });
        }

        let owner_id = format!("l_{}", uuid::Uuid::new_v4().simple());
        let started_utc = now_iso();
        let file = self.publish_identity(&leases_dir, &owner_id, &started_utc)?;
        // The heartbeat is a separate, atomically-replaced file: a crash
        // before it lands leaves a lease whose heartbeat reads as ancient
        // — stale after the TTL, which is the correct answer for an owner
        // that died mid-acquire.
        self.write_heartbeat(&leases_dir, &owner_id)?;
        sync_dir(&leases_dir)?;
        self.lease = Some(LeaseHandle {
            owner_id: owner_id.clone(),
            file,
        });
        // Post-publish tie-break (see the method doc): yields only to a
        // live foreign lease with a smaller owner id, so concurrent
        // publishers elect exactly one owner even without the sentinel.
        if let Some(owner) = self.younger_live_foreign_lease(&owner_id)?.live {
            self.release_lease()?;
            return Ok(LeaseAcquisition::Client { owner });
        }
        Ok(LeaseAcquisition::Owner { owner_id, broke })
    }

    /// Create the identity temp, flock it, write the immutable identity
    /// record, and publish it by rename. The rename keeps the inode, so
    /// the flock rides into the published name — the probe any other
    /// process runs on `leases/<owner>.lease` answers `Held` from the
    /// first moment the lease is observable. One retry covers a temp
    /// removed by a race the sentinel could not serialize (flock-less
    /// hosts); each attempt uses a fresh unique temp.
    fn publish_identity(
        &self,
        leases_dir: &Path,
        owner_id: &str,
        started_utc: &str,
    ) -> Result<File, StoreV2Error> {
        let mut last_err = None;
        for _ in 0..2 {
            // §4: no shared fixed `.tmp` name — the scratch file is unique
            // per owner, so two processes can never fight over one temp
            // path.
            let temp = unique_lease_temp(leases_dir, owner_id);
            let result = (|| -> Result<File, StoreV2Error> {
                let mut file = open_flocked_lease_temp(&temp, "identity")?;
                write_lease_identity_content(&mut file, started_utc)?;
                match std::fs::rename(&temp, self.lease_path(owner_id)) {
                    Ok(()) => Ok(file),
                    Err(err) => Err(err.into()),
                }
            })();
            match result {
                Ok(file) => return Ok(file),
                Err(err) => {
                    // The attempt never published, so its scratch file (if
                    // it survived the failure — a refused write, a failed
                    // rename) is garbage: remove it here so a persistent
                    // failure leaves zero temps behind instead of one per
                    // attempt for the grace-period sweep to collect. The
                    // handle (and its flock) went with the closure's
                    // return.
                    let _ = std::fs::remove_file(&temp);
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.expect("two attempts always set the error"))
    }

    /// Renew the held lease's heartbeat (§4). Returns whether this
    /// instance held a lease to renew — `Ok(false)` means it is not the
    /// owner (nothing was written).
    pub fn heartbeat_lease(&mut self) -> Result<bool, StoreV2Error> {
        if self.lease.is_none() {
            return Ok(false);
        }
        self.renew_lease_record()?;
        Ok(true)
    }

    /// Rewrite the heartbeat. The published heartbeat file carries no
    /// lock (the ownership flock lives on the identity file's inode; the
    /// temp that publishes it holds one only between create and the
    /// rename — see [`Self::write_heartbeat`]), so it is replaced
    /// atomically: unique temp, write, fsync, rename. A crash
    /// mid-renewal leaves the *previous* heartbeat — which then ages out
    /// past the TTL, the correct staleness answer — and can never tear
    /// the identity record the pid/boot-id ladder reads.
    fn renew_lease_record(&mut self) -> Result<(), StoreV2Error> {
        let owner_id = {
            let handle = self
                .lease
                .as_ref()
                .ok_or_else(|| StoreV2Error::Invalid("no lease held".to_string()))?;
            handle.owner_id.clone()
        };
        self.write_heartbeat(&self.root.join(LEASES_DIR), &owner_id)
    }

    fn write_heartbeat(&self, leases_dir: &Path, owner_id: &str) -> Result<(), StoreV2Error> {
        let beat = LeaseHeartbeat {
            heartbeat_utc: now_iso(),
            heartbeat_ms: now_epoch_ms(),
        };
        let bytes =
            serde_json::to_vec(&beat).map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
        let temp = unique_lease_temp(leases_dir, owner_id);
        let result = (|| -> Result<(), StoreV2Error> {
            use std::io::Write;
            let mut file = open_flocked_lease_temp(&temp, "heartbeat")?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            // Rename first, drop after: the flock covers the temp from
            // create through the rename that publishes it — the same
            // coverage window the identity temp gets.
            std::fs::rename(&temp, self.heartbeat_path(owner_id))?;
            drop(file);
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result
    }

    /// Release the held lease. The flock goes with the dropped handle and
    /// both files are removed best-effort; by the time this returns the
    /// release has happened, so a directory-sync failure is best-effort
    /// too — it must not read as "release failed" to a retrying caller
    /// (the retry's no-op path would then answer a misleading `Ok`).
    pub fn release_lease(&mut self) -> Result<(), StoreV2Error> {
        if let Some(handle) = self.lease.take() {
            drop(handle.file); // release the flock before unlinking
            let _ = std::fs::remove_file(self.lease_path(&handle.owner_id));
            let _ = std::fs::remove_file(self.heartbeat_path(&handle.owner_id));
            let _ = sync_dir(&self.root.join(LEASES_DIR));
        }
        Ok(())
    }

    /// Every lease on disk: this instance's own (marked `mine`), plus the
    /// foreign ones with their liveness probed. Introspection for callers
    /// deciding whether to repair, warn, or wait.
    pub fn lease_status(&self) -> Result<Vec<LeaseInfo>, StoreV2Error> {
        let own = self.lease.as_ref().map(|handle| handle.owner_id.clone());
        let mut leases = Vec::new();
        for probed in self.probe_leases()? {
            let alive = probed.alive(self.lease_ttl);
            let record = probed.merged_record();
            leases.push(LeaseInfo {
                mine: own.as_deref() == Some(probed.owner_id.as_str()),
                owner_id: probed.owner_id,
                pid: probed.identity.as_ref().map_or(0, |identity| identity.pid),
                record,
                alive,
            });
        }
        Ok(leases)
    }

    /// Break stale leases (§4): remove every lease whose owner is provably
    /// gone — its flock is free (the OS released it when the owner died),
    /// or, where flock cannot answer, its boot id no longer matches, its
    /// heartbeat expired past the TTL, or its recorded pid is dead. A
    /// lease that cannot be proven dead is never broken. Crash leftovers
    /// of the unique acquisition/heartbeat temporaries are swept too, but
    /// only past [`LEASE_TEMP_GRACE`]: a live writer's temp is younger
    /// than that by construction, so a concurrent sweep can never delete
    /// it out from under the rename that publishes it. Returns the broken
    /// owner ids.
    ///
    /// Serialized by the same `leases/.lock` sentinel as
    /// [`Self::acquire_lease`] (this is a public mutating path of its
    /// own): without the sentinel, a direct call racing a concurrent
    /// acquirer's publish window — identity renamed, heartbeat not yet
    /// written, the acquirer holding the sentinel there — could probe the
    /// fresh lease as ancient-hearted and delete it while its flock is
    /// genuinely held, leaving two owners: exactly what the lease design
    /// prevents. [`Self::acquire_lease`] reaches the breaking half
    /// through [`Self::break_stale_leases_under_sentinel`], already
    /// inside its own sentinel scope (flock is per open file
    /// description — self-deadlock if re-taken).
    pub fn break_stale_leases(&mut self) -> Result<Vec<String>, StoreV2Error> {
        self.break_stale_leases_within(LeaseSentinel::ACQUIRE_TIMEOUT)
    }

    /// [`Self::break_stale_leases`] with the sentinel wait bounded by
    /// `timeout` (the production bound; tests inject a short one so a
    /// held sentinel fails the call deterministically instead of via
    /// wall-clock timing).
    fn break_stale_leases_within(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<Vec<String>, StoreV2Error> {
        let leases_dir = self.root.join(LEASES_DIR);
        std::fs::create_dir_all(&leases_dir)?;
        let _sentinel = LeaseSentinel::acquire_with_timeout(&leases_dir, timeout)?;
        self.break_stale_leases_under_sentinel()
    }

    /// The breaking half of [`Self::break_stale_leases`], for callers
    /// already inside the acquisition sentinel's critical section
    /// (only [`Self::acquire_lease`]).
    fn break_stale_leases_under_sentinel(&mut self) -> Result<Vec<String>, StoreV2Error> {
        let own = self.lease.as_ref().map(|handle| handle.owner_id.clone());
        let mut broken = Vec::new();
        let mut removed_any = false;
        for probed in self.probe_leases()? {
            if Some(&probed.owner_id) == own.as_ref() {
                continue; // ours by construction
            }
            if probed.alive(self.lease_ttl) {
                continue;
            }
            let _ = std::fs::remove_file(self.lease_path(&probed.owner_id));
            let _ = std::fs::remove_file(self.heartbeat_path(&probed.owner_id));
            broken.push(probed.owner_id);
            removed_any = true;
        }
        // Temp leftovers: unique names, so nothing legitimate can
        // collide. A temp younger than the grace period is left alone (it
        // may belong to a live writer between create and rename); an
        // older one is removed when its flock is free — or, where flock
        // cannot answer, when it holds no parseable record (a live
        // writer's published files are never `*.tmp`). This keeps
        // flock-less hosts from accumulating crashed writers' temps while
        // never touching a live acquirer's.
        for temp in lease_temps_in(&self.root.join(LEASES_DIR)) {
            let Ok(metadata) = std::fs::metadata(&temp) else {
                continue;
            };
            let young = metadata
                .modified()
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age < LEASE_TEMP_GRACE);
            if young {
                continue;
            }
            if let Ok(mut file) = OpenOptions::new().read(true).open(&temp) {
                match try_flock_exclusive(&file) {
                    // A live writer holds its flock between create and
                    // rename — never touched, however old. Both temp
                    // kinds carry it: the identity temp from
                    // `publish_identity`, the heartbeat temp from
                    // `write_heartbeat` (the grace period below is the
                    // second guard, not the only one).
                    Ok(FlockEvidence::Held) => continue,
                    Ok(FlockEvidence::Free) => {}
                    // Where flock cannot answer, only a record-less temp
                    // is provably not a published lease: published files
                    // never carry the .tmp extension, and an identity or
                    // heartbeat record in a temp means a writer whose
                    // rename is still in flight (or wedged mid-retry) —
                    // leave it for the next sweep after it provably stops
                    // moving.
                    Ok(FlockEvidence::Unknown) => {
                        let holds_record = read_json_file::<LeaseIdentity>(&mut file).is_some()
                            || read_json_file::<LeaseHeartbeat>(&mut file).is_some();
                        if holds_record {
                            continue;
                        }
                    }
                    Err(_) => continue,
                }
            }
            let _ = std::fs::remove_file(&temp);
            removed_any = true;
        }
        if removed_any {
            sync_dir(&self.root.join(LEASES_DIR))?;
        }
        Ok(broken)
    }

    /// The live foreign lease on this root, when one exists — the store
    /// this process should treat itself as a client of — plus any lease
    /// files that could not be probed at all (surfaced by
    /// [`Self::reconcile`] so a corrupt lease cannot silently disable
    /// recovery without a trace). Deterministic: with several live
    /// foreign owners (a race the sentinel usually prevents), the
    /// smallest owner id answers.
    fn live_foreign_lease(&self) -> Result<Ownership, StoreV2Error> {
        let own = self.lease.as_ref().map(|handle| handle.owner_id.clone());
        let mut ownership = Ownership::default();
        for probed in self.probe_leases()? {
            if Some(&probed.owner_id) == own.as_ref() {
                continue;
            }
            if probed.unreadable {
                ownership.unreadable.push((
                    probed.owner_id,
                    "lease file exists but cannot be opened or parsed".to_string(),
                ));
                continue;
            }
            if probed.alive(self.lease_ttl) && ownership.live.is_none() {
                let record = probed.merged_record();
                ownership.live = Some(LeaseInfo {
                    owner_id: probed.owner_id,
                    pid: probed.identity.as_ref().map_or(0, |identity| identity.pid),
                    mine: false,
                    record,
                    alive: true,
                });
            }
        }
        Ok(ownership)
    }

    /// The post-publish tie-break: a live foreign lease with a **smaller**
    /// owner id than ours — the peer this instance must yield to if the
    /// sentinel could not serialize the publishes (see
    /// [`Self::acquire_lease`]). Order is total (unique ids), so both
    /// racers reach the same verdict and exactly one stays owner.
    fn younger_live_foreign_lease(&self, mine: &str) -> Result<Ownership, StoreV2Error> {
        let mut ownership = self.live_foreign_lease()?;
        if let Some(live) = &ownership.live {
            if live.owner_id.as_str() >= mine {
                ownership.live = None;
            }
        }
        Ok(ownership)
    }

    /// Every `*.lease` under `leases/`, probed. The probe opens each file
    /// on its own read-only handle (flock needs no write access — a
    /// read-only open keeps working where read-write would degrade to
    /// `Unknown`), so acquiring-then-dropping the flock on a free file
    /// releases it again.
    fn probe_leases(&self) -> Result<Vec<ProbedLease>, StoreV2Error> {
        let leases_dir = self.root.join(LEASES_DIR);
        let Ok(entries) = std::fs::read_dir(&leases_dir) else {
            return Ok(Vec::new());
        };
        let mut probed = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("lease") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let stem = stem.to_string();
            if !is_safe_path_component(&stem) {
                continue;
            }
            let mut file = match File::open(&path) {
                Ok(file) => file,
                Err(_) => {
                    // Unreadable, but present: probe as unknown — never
                    // decide ownership on an I/O failure — and flag it so
                    // callers can surface the state instead of silently
                    // deferring to an unanswerable owner forever.
                    probed.push(ProbedLease::unreadable(stem));
                    continue;
                }
            };
            let flock = try_flock_exclusive(&file).unwrap_or(FlockEvidence::Unknown);
            let identity = read_lease_identity(&mut file);
            let unreadable = identity.is_none();
            let heartbeat = read_lease_heartbeat(&mut self.heartbeat_path(&stem));
            probed.push(ProbedLease {
                owner_id: stem,
                identity,
                heartbeat,
                flock,
                unreadable,
            });
        }
        probed.sort_by(|left, right| left.owner_id.cmp(&right.owner_id));
        Ok(probed)
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
        let torn_tail_bytes = parsed.torn_tail_bytes;
        let was_finalized = parsed.finalized;
        if !was_finalized || torn_tail_bytes > 0 {
            seal_recovered_journal(source, &parsed)?;
            parsed.finalized = true;
            parsed.torn_tail_bytes = 0;
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
        let complete = was_finalized && torn_tail_bytes == 0;
        self.commit_capture_superseding(&record, None, transcribe && complete)?;
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
                    Ok(()) if aside == SUPERSEDED_SUBDIR => report.superseded.push(id),
                    Ok(()) => report.deleted.push(id),
                    Err(err) => report.failed.push((id, err.to_string())),
                }
                continue;
            }
            if parsed.finalized
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
    /// `recognition_attempts` row with status `started` (v2's analog of the
    /// v1 `mark_attempt` status bump), plus the cross-process in-flight
    /// marker the startup sweep consults (#213) — held *before* the row
    /// becomes visible, so a `started` row is always either owned by a
    /// live process or orphaned by a dead one; the sweep can never observe
    /// the in-between. Returns the attempt id. Recognizing an unknown
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
        // The marker is held from here; it joins `attempt_locks` only once
        // the row is committed. Until then, dropping `marker` releases the
        // flock and `remove_attempt_marker` unlinks the file, so a failed
        // begin never leaves a marker behind.
        let marker = self.open_attempt_marker(&id)?;
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
                self.attempt_locks.insert(id.clone(), marker);
                Ok(id)
            }
            Err(err) => {
                drop(marker);
                self.remove_attempt_marker(&id);
                Err(err)
            }
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
                // in flight (#220) — the stale sweep, or another window,
                // takes it from here.
                self.release_attempt_lock(&attempt_id);
                return Err(err.into());
            }
        };
        if changed == 0 {
            // A second finisher lost the race to the first: never re-write
            // a terminal row. The attempt is settled either way, so its
            // marker must not outlive it (#213).
            self.release_attempt_lock(&attempt_id);
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        // The attempt is settled: its in-flight marker (#213) must not
        // outlive the row that gave it meaning.
        self.release_attempt_lock(&attempt_id);
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

    /// Finishes `started` attempts left behind by a previous run (v2's
    /// analog of the v1 "stuck in Transcribing" startup fix): each is
    /// marked failed with `note`, because the process that started it is
    /// gone — *confirmed* against the attempt's cross-process marker
    /// (#213): an attempt a live instance still owns (its flock is held)
    /// is left alone, the file-store analog of the Electron reference's
    /// `transcriptionInFlight` guard (#144); only an owner whose signal is
    /// gone is treated as interrupted, so one instance's startup can never
    /// phantom-fail another instance's in-flight attempt. The sweep's
    /// marker probe stays a read-only hint; the enforcement is one
    /// conditional write keyed by the exact attempt row, so a settle (or
    /// delete) that commits between the probe and the write cannot
    /// last-writer-lose its outcome to a failure (#162 semantics).
    /// Returns the affected capture ids — each capture once, even when
    /// several of its attempts were swept. Completed attempts are
    /// untouched. The pass also garbage-collects the marker directory (see
    /// [`Self::gc_attempt_markers`]).
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
                // Another window may still own this attempt (#144, #213):
                // only an owner whose signal is gone is treated as
                // interrupted.
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
            // changed == 0: the owner settled (or a delete cascaded)
            // between the probe and the write — nothing stale remains.
            // Either way the attempt is settled now, and a marker for a
            // settled row is disk garbage: take it with the sweep.
            self.release_attempt_lock(attempt_id);
        }
        self.gc_attempt_markers();
        Ok(swept)
    }

    /// The marker file one in-flight recognition attempt owns (#213):
    /// `<root>/attempt-locks/<attemptId>.lock`. Like the Electron
    /// reference's per-attempt Web Lock
    /// (`starling:dictation:<db>:transcribe:<id>:<signal>`), the name is
    /// unique per attempt, so a contender that outlives the first settler
    /// keeps its own signal.
    fn attempt_lock_path(&self, attempt_id: &str) -> PathBuf {
        self.root
            .join(ATTEMPT_LOCKS_DIR)
            .join(format!("{attempt_id}.lock"))
    }

    /// Hold one attempt's cross-process marker (#213): create the lock
    /// file, flock it where the platform has flock, and record this
    /// process's PID in the file — the ownership signal everywhere flock
    /// cannot answer (see [`attempt_owned_from`]). The handle lives in
    /// [`Self::attempt_locks`] — keeping the file open is what holds the
    /// lock — until the attempt settles, the capture is deleted, or the
    /// process exits. The marker directory is created at
    /// [`StoreV2::open`]; this path never recreates it.
    #[cfg(test)]
    fn hold_attempt_lock(&mut self, attempt_id: &str) -> Result<(), StoreV2Error> {
        let file = self.open_attempt_marker(attempt_id)?;
        self.attempt_locks.insert(attempt_id.to_string(), file);
        Ok(())
    }

    /// Create, flock and stamp one attempt's marker without registering
    /// it: the returned handle holds the lock until it is dropped or moved
    /// into [`Self::attempt_locks`]. A failed open leaves no file behind.
    fn open_attempt_marker(&self, attempt_id: &str) -> Result<File, StoreV2Error> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.attempt_lock_path(attempt_id))?;
        let mut file = match try_flock_exclusive(&file) {
            Ok(FlockEvidence::Free) | Ok(FlockEvidence::Unknown) => file,
            Ok(FlockEvidence::Held) => {
                // Unique-per-attempt ids make this unreachable short of
                // external tampering with the locks directory; whatever
                // file sits at this freshly minted id's path is not a live
                // attempt's marker, and the failed hold must not leave it
                // behind (#213 review). A real owner can never share a
                // minted id, so nothing legitimate is unlinked.
                drop(file);
                let _ = std::fs::remove_file(self.attempt_lock_path(attempt_id));
                return Err(StoreV2Error::Io(io::Error::other(format!(
                    "attempt marker {attempt_id} is already held"
                ))));
            }
            Err(err) => {
                // A flock that errored (I/O trouble, not contention) must
                // not leave the freshly created marker behind either — the
                // id is minted, so nothing legitimate is unlinked (#213
                // review, second round).
                drop(file);
                let _ = std::fs::remove_file(self.attempt_lock_path(attempt_id));
                return Err(err.into());
            }
        };
        // Best-effort: on flock platforms the lock decides ownership and
        // the PID is observability; where flock cannot answer, a PID that
        // failed to write degrades that marker to "held" — the safe
        // direction.
        use std::io::Write as _;
        let _ = file.write_all(std::process::id().to_string().as_bytes());
        Ok(file)
    }

    /// Unlink one attempt's marker file. The caller has dropped (or never
    /// held) its handle.
    fn remove_attempt_marker(&self, attempt_id: &str) {
        let _ = std::fs::remove_file(self.attempt_lock_path(attempt_id));
    }

    /// Release one attempt's marker (#213): drop the held flock and remove
    /// the file. Idempotent, and safe for attempts this process never held
    /// (the sweep calls it for markers it found unowned).
    fn release_attempt_lock(&mut self, attempt_id: &str) {
        if !is_safe_path_component(attempt_id) {
            // A row whose id is not a safe file name can never have one of
            // our markers; never follow an id that escaped the locks
            // directory.
            return;
        }
        self.attempt_locks.remove(attempt_id);
        self.remove_attempt_marker(attempt_id);
    }

    /// Whether a live owner holds the attempt's marker (#213): this
    /// process's own registry first (the within-process signal that also
    /// covers flock-less platforms), then the marker itself — flock where
    /// the platform has it, the recorded PID otherwise. A missing marker
    /// means no owner — a `started` row from a pre-marker build stays
    /// sweepable; marker *presence* alone is never the signal, so a stale
    /// file left by a crash cannot block the sweep (the OS released its
    /// flock when the owner died). A marker that cannot be probed at all
    /// (permissions, I/O trouble) is treated as owned: an unreadable
    /// liveness signal must not manufacture an interruption — the same
    /// rule the Electron reference applies when the lock manager cannot
    /// answer.
    /// Whether a live process (this one included) still owns `started`
    /// attempt `attempt_id` — its cross-process marker is held (#213).
    /// A window offered a take another process is transcribing (#220)
    /// leaves it to that process.
    pub fn attempt_owned(&self, attempt_id: &str) -> bool {
        self.attempt_is_owned(attempt_id)
    }

    fn attempt_is_owned(&self, attempt_id: &str) -> bool {
        if self.attempt_locks.contains_key(attempt_id) {
            return true;
        }
        if !is_safe_path_component(attempt_id) {
            return false;
        }
        let mut file = match OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.attempt_lock_path(attempt_id))
        {
            Ok(file) => file,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return false,
            Err(_) => return true,
        };
        let flock = match try_flock_exclusive(&file) {
            Ok(evidence) => evidence,
            Err(_) => FlockEvidence::Unknown,
        };
        // The flock decides when it answered; otherwise the PID the marker
        // records decides.
        let pid_alive = if flock == FlockEvidence::Unknown {
            read_recorded_pid(&mut file).map(process_is_alive)
        } else {
            None
        };
        attempt_owned_from(flock, pid_alive)
    }

    /// Best-effort garbage collection of the marker directory (#213
    /// review): removes markers whose attempt row no longer reads
    /// `started`, so row-removal paths beyond `finish_recognition` and
    /// `delete_capture` (cascades, future reconciliation work) cannot
    /// leak them. Two guards keep it safe: a marker whose flock is held
    /// is never touched — its holder may be a live owner between
    /// acquiring the marker and inserting its row (the order
    /// [`Self::begin_recognition`] guarantees) — and without a flock
    /// answer the marker is removed only when its recorded PID is
    /// provably dead. Errors are ignored: a marker that cannot be
    /// examined today is simply revisited by the next sweep.
    fn gc_attempt_markers(&self) {
        let Ok(entries) = std::fs::read_dir(self.root.join(ATTEMPT_LOCKS_DIR)) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("lock") {
                continue;
            }
            let Some(attempt_id) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if !is_safe_path_component(attempt_id) {
                continue;
            }
            let Ok(mut file) = OpenOptions::new().read(true).write(true).open(&path) else {
                continue;
            };
            match try_flock_exclusive(&file) {
                // Held (or unanswerable with a PID that cannot be proven
                // dead): a live owner may exist — leave the marker alone.
                Ok(FlockEvidence::Held) | Err(_) => continue,
                Ok(FlockEvidence::Unknown) => {
                    if read_recorded_pid(&mut file).map(process_is_alive) != Some(false) {
                        continue;
                    }
                }
                Ok(FlockEvidence::Free) => {}
            }
            let started = self
                .conn
                .query_row(
                    "SELECT status FROM recognition_attempts WHERE id = ?1",
                    params![attempt_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .ok()
                .flatten();
            if started.as_deref() != Some("started") {
                let _ = std::fs::remove_file(&path);
            }
        }
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
        store.promote_from_staging(&self.id)?;
        let (status, extra_json) = match mark {
            CommitMark::Complete => (CaptureStatus::Complete, self.meta.extra_json),
            CommitMark::Interrupted { note } => (
                CaptureStatus::Interrupted,
                Some(merge_extra_note(self.meta.extra_json.as_deref(), &note)),
            ),
        };
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
        // Only a complete take is transcribed by itself: a salvaged one
        // waits for the user.
        let transcribe = self.meta.transcribe && record.status == CaptureStatus::Complete;
        store.commit_capture_superseding(&record, self.meta.supersedes_journal.as_deref(), transcribe)?;
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
    /// In-flight ids (staging journals, orphan candidates) a live foreign
    /// lease owner claimed: this run was a client (§4 ownership) and left
    /// them for the owner's own reconcile. Informational — a deferral is
    /// normal multi-instance operation, not a repair finding, so it does
    /// not contribute to [`Self::has_findings`].
    pub deferred_to_live_owner: Vec<String>,
    /// `(owner, reason)` for lease files that exist under `leases/` but
    /// could not be probed. An unanswerable lease still reads as a live
    /// owner (never break what cannot be proven dead), so recovery
    /// defers to it — this finding is what makes that visible instead of
    /// a silently disabled recovery.
    pub unreadable_leases: Vec<(String, String)>,
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
            || !self.unreadable_leases.is_empty()
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
        if !self.unreadable_leases.is_empty() {
            parts.push(format!(
                "{} ownership lease file{} could not be read; recovery deferred to it \
                 until the file is removed or repaired",
                self.unreadable_leases.len(),
                if self.unreadable_leases.len() == 1 { "" } else { "s" }
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
    /// legacy `journals/deleted/`).
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

/// What one marker probe learned about its flock (#213 review).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlockEvidence {
    /// The probe acquired the exclusive lock itself (releasing it with the
    /// handle): no live owner anywhere — the OS frees the lock when the
    /// owning process dies.
    Free,
    /// A live owner holds the lock.
    Held,
    /// The platform has no flock, or the call could not answer.
    Unknown,
}

/// Whether an attempt reads as owned, given the flock evidence and the
/// liveness of the PID its marker records (#213 review). Pure, so the
/// whole degradation ladder is testable on platforms that always have
/// flock:
///
/// - a held flock decides outright — the owner is alive by construction;
/// - a freed flock decides outright — the owner is dead by construction;
/// - without a flock answer (no flock on the platform, or the call
///   errored) the recorded PID's liveness decides;
/// - with neither answer the marker reads as held: an unanswerable
///   liveness signal must not manufacture an interruption — the same rule
///   the Electron reference applies when the lock manager cannot answer.
fn attempt_owned_from(flock: FlockEvidence, pid_alive: Option<bool>) -> bool {
    match flock {
        FlockEvidence::Held => true,
        FlockEvidence::Free => false,
        FlockEvidence::Unknown => match pid_alive {
            Some(true) => true,
            Some(false) => false,
            None => true,
        },
    }
}

/// The PID recorded in a marker file (#213 review), when one is readable.
fn read_recorded_pid(file: &mut File) -> Option<u32> {
    use std::io::Read as _;
    let mut text = String::new();
    file.read_to_string(&mut text).ok()?;
    text.trim().parse().ok()
}

/// Take a non-blocking exclusive advisory lock on an open file (#213).
/// Because the lock lives on the open file description, the OS releases
/// it when the owning process dies, which is what makes a leftover marker
/// file after a crash harmless.
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
/// immediately) — a lock on the file object that the OS releases when
/// the holding handle closes or its process dies, which is exactly the
/// flock contract the lease and attempt markers rely on (#220: without
/// it a second host broke a live owner's lease). The locked byte lies
/// far past EOF on purpose: Windows byte-range locks are mandatory, and
/// locking the record's own bytes would stop every prober from reading
/// the identity it needs.
#[cfg(windows)]
pub(crate) fn try_flock_exclusive(file: &File) -> io::Result<FlockEvidence> {
    Ok(if windows_lock::try_lock(file)? {
        FlockEvidence::Free
    } else {
        FlockEvidence::Held
    })
}

/// Without any lock primitive there is nothing to ask (#213 review): the
/// probe answers [`FlockEvidence::Unknown`] and the marker's recorded PID
/// decides ownership instead (see [`attempt_owned_from`]).
#[cfg(not(any(unix, windows)))]
pub(crate) fn try_flock_exclusive(_file: &File) -> io::Result<FlockEvidence> {
    Ok(FlockEvidence::Unknown)
}

/// The Windows lock primitive behind [`try_flock_exclusive`] and the
/// lease sentinel.
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

        /// Empirically settles the read-only-handle question (review on
        /// #220): the Win32 docs say `LockFileEx` wants a handle opened
        /// with GENERIC_READ *or* GENERIC_WRITE, so a read-only handle
        /// is *expected* to take the exclusive lock — the read-only
        /// probe opens (`probe_leases`, the stale-temp sweep) lean on
        /// exactly that. Pin it: the first read-only handle locks
        /// (`Ok(true)`) and a second one on the same file answers held
        /// (`Ok(false)`), not an error. If this ever fails on a Windows
        /// runner, those probe opens must switch to read+write.
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

/// Whether the process `pid` is still alive (#213 review) — the fallback
/// ownership signal wherever the flock cannot answer.
#[cfg(unix)]
pub(crate) fn process_is_alive(pid: u32) -> bool {
    // SAFETY: kill(2) with signal 0 performs existence and permission
    // checks only — no signal is ever delivered.
    if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
        return true;
    }
    // EPERM means the process exists but belongs to another user; only
    // ESRCH (no such process) means it is gone.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Windows: open the process for a limited query and read its exit
/// code. A pid that cannot be opened because it belongs to someone else
/// (ACCESS_DENIED) exists; any other open failure means no such
/// process. A query that fails on an opened process is not proof of
/// death — presumed alive, never break what cannot be proven dead.
#[cfg(windows)]
pub(crate) fn process_is_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ACCESS_DENIED};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    /// `STILL_ACTIVE`: the exit code a running process reports.
    const STILL_ACTIVE: u32 = 259;
    // SAFETY: plain Win32 calls with owned out-parameters; the handle is
    // closed on every path that opened it.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return GetLastError() == ERROR_ACCESS_DENIED;
        }
        let mut code = 0u32;
        let queried = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        // NOTE: PID recycling and an exit code of 259 both read as
        // "alive"; liveness here is best-effort, matching unix kill(0).
        // It is only the fallback signal: wherever the flock /
        // LockFileEx probe can speak (the lease and attempt markers),
        // its answer takes precedence over this pid query.
        queried == 0 || code == STILL_ACTIVE
    }
}

/// Without a primitive to ask the OS about a foreign PID (#213
/// review), it is presumed dead: this process's OWN attempts are spared
/// by the in-process registry before any marker is probed, so the only
/// markers read here belong to other processes — unknowable liveness
/// must not turn every crash-orphaned attempt into a permanently stuck
/// `started` row. The cost is that a second live instance on a flock-less
/// platform can have its attempt interrupted, the same degradation the
/// Electron reference accepts without a lock manager; unix keeps the
/// full cross-process guarantee via flock.
#[cfg(not(any(unix, windows)))]
pub(crate) fn process_is_alive(_pid: u32) -> bool {
    false
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
// Leases (§4 ownership): free helpers.
// ---------------------------------------------------------------------------

/// What `acquire_lease` found on the data root.
#[derive(Debug)]
pub enum LeaseAcquisition {
    /// This instance is the owner of the data root. `broke` names the
    /// stale leases broken on the way in.
    Owner { owner_id: String, broke: Vec<String> },
    /// A live foreign owner holds the root: this process is a client, not
    /// a competitor. `owner` identifies the live lease.
    Client { owner: LeaseInfo },
    /// Lease files exist that can be neither probed nor broken (see
    /// [`StoreV2::acquire_lease`]): no ownership was taken. `unreadable`
    /// names each with the reason, the same shape
    /// [`ReconciliationReport::unreadable_leases`] reports.
    UnanswerableLeases { unreadable: Vec<(String, String)> },
}

/// One lease as `leases/` shows it. `pid`, `boot_id` and the heartbeat
/// come from the record the owner wrote; `alive` is the probed verdict
/// (flock first, record fallback — see [`lease_alive_from_probe`]).
#[derive(Debug, Clone)]
pub struct LeaseInfo {
    /// This instance's own lease.
    pub mine: bool,
    pub owner_id: String,
    pub pid: u32,
    pub record: Option<LeaseRecord>,
    pub alive: bool,
}

/// The lease record serialized into `leases/<ownerId>.lease`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseRecord {
    pub pid: u32,
    /// The kernel boot id at acquisition ("" where the host has none): a
    /// pid alone is not identity across a reboot, but pid+boot is.
    pub boot_id: String,
    pub started_utc: String,
    pub heartbeat_utc: String,
    /// Epoch-milliseconds heartbeat — the machine-readable staleness
    /// source (ISO strings are for humans; comparing them is not).
    pub heartbeat_ms: u64,
}

/// The immutable identity record inside `leases/<ownerId>.lease`.
/// Written once into the acquisition temp and published by rename;
/// never rewritten afterwards — heartbeats live in the separate
/// [`LeaseHeartbeat`] file precisely so a mid-heartbeat crash cannot
/// tear the record the pid/boot-id fallback ladder depends on.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct LeaseIdentity {
    pid: u32,
    /// The kernel boot id at acquisition ("" where the host has none): a
    /// pid alone is not identity across a reboot, but pid+boot is.
    boot_id: String,
    started_utc: String,
}

/// The heartbeat record inside `leases/<ownerId>.hb` — replaced
/// atomically (unique temp + rename) on every renewal. The file carries
/// no lock, so replacing it never strands the ownership flock that lives
/// on the identity file's inode.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct LeaseHeartbeat {
    heartbeat_utc: String,
    /// Epoch milliseconds — the machine-readable staleness source (ISO
    /// strings are for humans; comparing them is not).
    heartbeat_ms: u64,
}

/// What one probe of `leases/` concluded about ownership: the live
/// foreign owner to defer to (if any), and the lease files that could
/// not be probed at all — an unanswerable lease still reads as alive
/// (never break what cannot be proven dead), but it is reported so a
/// corrupt file cannot silently disable recovery without a trace.
#[derive(Default)]
struct Ownership {
    live: Option<LeaseInfo>,
    unreadable: Vec<(String, String)>,
}

/// One probed lease: its identity (when the file opens and parses), its
/// latest heartbeat (when the `.hb` file opens and parses), and the flock
/// evidence of the identity file.
struct ProbedLease {
    owner_id: String,
    identity: Option<LeaseIdentity>,
    heartbeat: Option<LeaseHeartbeat>,
    flock: FlockEvidence,
    /// The identity file exists but would not open or parse.
    unreadable: bool,
}

impl ProbedLease {
    fn unreadable(owner_id: String) -> Self {
        Self {
            owner_id,
            identity: None,
            heartbeat: None,
            flock: FlockEvidence::Unknown,
            unreadable: true,
        }
    }

    /// The probed liveness verdict: the flock decides when it answered
    /// (the OS is the truth — a held lock means a live owner, a freed
    /// lock means a dead one, whatever the heartbeat says); the record's
    /// boot id / heartbeat / pid decide only where the flock cannot
    /// answer. A missing heartbeat file reads as an ancient heartbeat —
    /// stale, which is the right answer both for an owner that died
    /// mid-acquire and for one whose renewal crashed.
    fn alive(&self, ttl: std::time::Duration) -> bool {
        match self.flock {
            FlockEvidence::Held => true,
            FlockEvidence::Free => false,
            FlockEvidence::Unknown => match &self.identity {
                // No record to reason about and no lock to ask: never
                // break what cannot be proven dead.
                None => true,
                Some(identity) => lease_alive_from(
                    boot_matches(&identity.boot_id),
                    Some(
                        self.heartbeat
                            .as_ref()
                            .is_some_and(|beat| heartbeat_is_fresh(beat.heartbeat_ms, ttl)),
                    ),
                    Some(process_is_alive(identity.pid)),
                ),
            },
        }
    }

    /// The merged public view of identity + heartbeat.
    fn merged_record(&self) -> Option<LeaseRecord> {
        self.identity.as_ref().map(|identity| LeaseRecord {
            pid: identity.pid,
            boot_id: identity.boot_id.clone(),
            started_utc: identity.started_utc.clone(),
            heartbeat_utc: self
                .heartbeat
                .as_ref()
                .map_or_else(String::new, |beat| beat.heartbeat_utc.clone()),
            heartbeat_ms: self.heartbeat.as_ref().map_or(0, |beat| beat.heartbeat_ms),
        })
    }
}

/// The acquisition sentinel: `leases/.lock`, a **fixed-name** file (it is
/// never renamed and never carries per-owner content, so it does not
/// reintroduce the fixed-`.tmp` collision this design removes) flocked
/// for the duration of the probe-and-publish critical section in
/// [`StoreV2::acquire_lease`], making concurrent acquisitions serialize.
/// Where the platform has no flock the sentinel is a no-op and the
/// post-publish tie-break re-check covers the window.
struct LeaseSentinel {
    #[allow(dead_code)] // the open handle is the lock; nothing reads it
    file: Option<File>,
}

impl LeaseSentinel {
    /// The production wait: long enough for a healthy concurrent
    /// acquisition (a few fsync'd writes) to finish, short enough that a
    /// wedged holder cannot stall startup behind it.
    const ACQUIRE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    /// Take the sentinel with the production timeout (see
    /// [`Self::acquire_with_timeout`]).
    fn acquire(leases_dir: &Path) -> Result<Self, StoreV2Error> {
        Self::acquire_with_timeout(leases_dir, Self::ACQUIRE_TIMEOUT)
    }

    /// Take the sentinel: a non-blocking flock retried under a BOUNDED
    /// wait (flock(2) on unix, `LockFileEx` on Windows; no-op elsewhere). The sentinel coordinates
    /// probe-and-publish only — it is not a liveness primitive — so a
    /// holder wedged mid-acquisition (hung fsync, stuck disk) must not be
    /// able to block every other process's lease acquisition, and
    /// therefore startup, indefinitely: past the bound the acquisition
    /// fails with a distinct error the caller can surface or retry,
    /// instead of blocking forever. Contention with a healthy sibling
    /// acquisition resolves in milliseconds and never sees the bound.
    fn acquire_with_timeout(
        leases_dir: &Path,
        timeout: std::time::Duration,
    ) -> Result<Self, StoreV2Error> {
        // Only the lock call differs between the platforms (flock(2) vs
        // `LockFileEx`); the open/retry/deadline spine is shared below
        // so a fix on one backend cannot drift past the other. No lock
        // primitive at all: the sentinel is a no-op.
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (leases_dir, timeout);
            return Ok(Self { file: None });
        }
        #[cfg(any(unix, windows))]
        Self::take_with_lock(leases_dir, timeout, sentinel_try_lock)
    }

    /// The shared spine of [`Self::acquire_with_timeout`]: open the
    /// sentinel read-write, retry `try_lock` every 25 ms until it takes
    /// the lock or `timeout` passes, then fail with the distinct
    /// wedged-holder error. `Ok(true)` from `try_lock` means this
    /// handle holds the lock; `Ok(false)` means another holder;
    /// `Err` propagates (a genuinely failed lock call is not a timeout).
    #[cfg(any(unix, windows))]
    fn take_with_lock(
        leases_dir: &Path,
        timeout: std::time::Duration,
        mut try_lock: impl FnMut(&File) -> io::Result<bool>,
    ) -> Result<Self, StoreV2Error> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(leases_dir.join(LEASE_SENTINEL_FILE))?;
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if try_lock(&file)? {
                return Ok(Self { file: Some(file) });
            }
            if std::time::Instant::now() >= deadline {
                return Err(StoreV2Error::Invalid(format!(
                    "the lease sentinel stayed busy for more than {timeout:?} — another \
                     lease acquisition appears wedged; retrying may help"
                )));
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }
}

/// The unix sentinel lock call, in the shape
/// [`LeaseSentinel::acquire_with_timeout`]'s shared spine wants:
/// `Ok(true)` = locked, `Ok(false)` = another holder (EWOULDBLOCK),
/// `Err` = anything else.
#[cfg(unix)]
fn sentinel_try_lock(file: &File) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    // SAFETY: flock(2) on an fd this guard owns and keeps open until
    // dropped; no close or hand-off happens here.
    let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if taken == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
        return Ok(false);
    }
    Err(err)
}

/// The Windows sentinel lock call: [`windows_lock::try_lock`] in the
/// same shape as the unix one above.
#[cfg(windows)]
fn sentinel_try_lock(file: &File) -> io::Result<bool> {
    windows_lock::try_lock(file)
}

/// Open a freshly created unique lease temp and flock it — the shared
/// create-and-probe step of the identity publish and the heartbeat
/// replacement (both temps carry the flock from create through the
/// rename that publishes them, so a stalled writer answers `Held` to
/// the stale-lease sweep however long it stalls). A freshly created
/// unique temp is never genuinely contended: a `Held` probe means
/// someone else's temp collided into the unique name (external
/// tampering) — abort; a probe that itself *failed* is reported as the
/// probe failure it is, never as a phantom holder. `kind` names the
/// temp ("identity"/"heartbeat") so triage lands on the right path.
fn open_flocked_lease_temp(temp: &Path, kind: &str) -> Result<File, StoreV2Error> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(temp)?;
    match try_flock_exclusive(&file) {
        Ok(FlockEvidence::Free) | Ok(FlockEvidence::Unknown) => Ok(file),
        Ok(FlockEvidence::Held) => {
            drop(file);
            Err(StoreV2Error::Io(io::Error::other(format!(
                "{kind} lease temp {temp:?} is already held"
            ))))
        }
        Err(err) => {
            drop(file);
            Err(StoreV2Error::Io(io::Error::new(
                err.kind(),
                format!("{kind} lease temp {temp:?}: flock probe failed: {err}"),
            )))
        }
    }
}

/// Write the immutable identity record onto the acquisition temp (the
/// file is published by rename immediately after; nothing rewrites it).
fn write_lease_identity_content(file: &mut File, started_utc: &str) -> Result<(), StoreV2Error> {
    use std::io::Write;
    let record = LeaseIdentity {
        pid: std::process::id(),
        boot_id: boot_id(),
        started_utc: started_utc.to_string(),
    };
    let bytes =
        serde_json::to_vec(&record).map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

/// The identity inside one lease file, when one parses. A torn or
/// unparseable file reads as `None` — the flock then decides, and the
/// probe flags the lease unreadable so the state is surfaced.
fn read_lease_identity(file: &mut File) -> Option<LeaseIdentity> {
    read_json_file(file)
}

/// The heartbeat beside one lease, when one parses. A missing or torn
/// `.hb` reads as `None` — an ancient heartbeat, stale after the TTL.
fn read_lease_heartbeat(path: &Path) -> Option<LeaseHeartbeat> {
    let mut file = File::open(path).ok()?;
    read_json_file(&mut file)
}

/// One small JSON file, parsed in full.
fn read_json_file<T: serde::de::DeserializeOwned>(file: &mut File) -> Option<T> {
    use std::io::{Read, Seek, SeekFrom};
    let mut text = String::new();
    file.seek(SeekFrom::Start(0)).ok()?;
    file.read_to_string(&mut text).ok()?;
    serde_json::from_str(&text).ok()
}

/// A unique temporary under `leases/`, named after its owner (§4: the
/// fixed shared `.tmp` name is replaced by unique temporaries under the
/// lease owner — two processes can never collide on it). Used by both the
/// identity publish and the heartbeat replacement.
fn unique_lease_temp(leases_dir: &Path, owner_id: &str) -> PathBuf {
    leases_dir.join(format!("{owner_id}.{}.tmp", uuid::Uuid::new_v4().simple()))
}

/// Every `*.tmp` scratch file under `leases/` (sorted; missing dir = none).
fn lease_temps_in(leases_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(leases_dir) else {
        return Vec::new();
    };
    let mut temps: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("tmp"))
        .collect();
    temps.sort();
    temps
}

/// The kernel boot id (Linux), or "" where the host does not expose one.
fn boot_id() -> String {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|text| text.trim().to_string())
        .unwrap_or_default()
}

/// Whether a recorded boot id matches the current boot. `None` when
/// either side is unknown — an unknown boot id never proves anything.
fn boot_matches(recorded: &str) -> Option<bool> {
    let current = boot_id();
    if current.is_empty() || recorded.is_empty() {
        None
    } else {
        Some(current == recorded)
    }
}

/// Unix epoch milliseconds; never 0 (0 is the "no heartbeat parsed"
/// sentinel — a pre-epoch or exactly-epoch clock saturates to 1, see
/// [`epoch_ms`]).
fn now_epoch_ms() -> u64 {
    epoch_ms(std::time::SystemTime::now())
}

/// A clock reading as epoch milliseconds. A clock before 1970 saturates
/// to 1 ms past the epoch, never 0 (and an exactly-epoch clock clamps to
/// the same 1): `0` is the "no heartbeat parsed" sentinel (a missing
/// `.hb` reads as `heartbeat_ms = 0` — ancient), so a skewed-but-live
/// owner must not write a heartbeat indistinguishable from "never";
/// against a peer reading the same skewed clock, 1-vs-1 still reads
/// fresh.
fn epoch_ms(now: std::time::SystemTime) -> u64 {
    now.duration_since(std::time::UNIX_EPOCH)
        // The Ok arm clamps too: an exactly-epoch (or sub-millisecond
        // past it) clock yields 0 ms, which is just as much the
        // "no heartbeat parsed" sentinel as the pre-epoch Err arm.
        .map(|since| (since.as_millis() as u64).max(1))
        .unwrap_or(1)
}

/// Whether a heartbeat recorded at `heartbeat_ms` is still fresh under
/// `ttl`. A zero/garbage timestamp reads as ancient — stale.
fn heartbeat_is_fresh(heartbeat_ms: u64, ttl: std::time::Duration) -> bool {
    now_epoch_ms().saturating_sub(heartbeat_ms) <= ttl.as_millis() as u64
}

/// The flock-less liveness ladder, pure so the whole degradation order is
/// testable on platforms that always have flock:
///
/// - a boot mismatch is fatal outright — after a reboot the recorded pid
///   is not the owner, however alive it looks;
/// - an expired heartbeat is fatal (§4: stale after heartbeat expiry);
/// - a dead pid is fatal;
/// - everything else — fresh heartbeat, no dead evidence, or signals that
///   could not answer — reads as alive: breaking a lease requires proof
///   of death, never the absence of proof of life.
fn lease_alive_from(
    same_boot: Option<bool>,
    heartbeat_fresh: Option<bool>,
    pid_alive: Option<bool>,
) -> bool {
    if same_boot == Some(false) {
        return false;
    }
    if heartbeat_fresh == Some(false) {
        return false;
    }
    if pid_alive == Some(false) {
        return false;
    }
    true
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
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

    /// A complete take through the full protocol.
    fn committed_take(store: &mut StoreV2, samples: &[f32]) -> CommittedTake {
        let mut take = store
            .begin_take(TakeMeta::for_device("test-device"))
            .expect("begin take");
        take.append_frames(samples).expect("append");
        take.write_boundary().expect("boundary");
        take.finish(store).expect("finish")
    }

    // ---- schema, versioning, extra_json ------------------------------

    #[test]
    fn schema_initializes_and_reports_its_version() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        assert_eq!(store.schema_version().expect("version"), SCHEMA_VERSION);

        committed_take(&mut store, &ramp(50, 0));

        // Reopening a same-version database neither upgrades nor refuses.
        drop(store);
        let store = store_in(&dir);
        assert_eq!(store.schema_version().expect("version"), SCHEMA_VERSION);
        assert_eq!(
            store
                .list_records(0, 10)
                .expect("list")
                .total,
            1
        );
    }

    #[test]
    fn a_v1_schema_database_is_upgraded_in_place() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(40, 0));
        let id = take.record.id.clone();
        store
            .begin_recognition(&id, "starling:parakeet", None)
            .expect("begin");
        drop(store);

        // Rewind the database to the v1 shape: drop the created_utc
        // column's data by rebuilding the table without it (SQLite cannot
        // drop columns portably) and stamp schema_version 1.
        {
            let conn = Connection::open(dir.path().join("v2").join(DB_FILE)).expect("open");
            conn.execute_batch(
                "CREATE TABLE recognition_attempts_v1 (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    backend TEXT NOT NULL,
                    model_hash TEXT,
                    language TEXT,
                    options_json TEXT,
                    text TEXT NOT NULL,
                    partial_or_final TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timing_json TEXT,
                    extra_json TEXT
                 );
                 INSERT INTO recognition_attempts_v1
                    SELECT id, capture_id, backend, model_hash, language, options_json,
                           text, partial_or_final, status, timing_json, extra_json
                    FROM recognition_attempts;
                 DROP TABLE recognition_attempts;
                 ALTER TABLE recognition_attempts_v1 RENAME TO recognition_attempts;
                 UPDATE meta SET value = '1' WHERE key = 'schema_version';",
            )
            .expect("rewind to v1");
        }

        // Opening upgrades: version bumped, column added, and the
        // pre-upgrade attempt reads back with NULL created_utc (readers
        // fall back to the capture's creation time).
        let mut store = store_in(&dir);
        assert_eq!(store.schema_version().expect("version"), SCHEMA_VERSION);
        let attempts = store.attempts_for(&id).expect("attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].created_utc, None);
        // The upgraded store accepts new work normally.
        store
            .finish_recognition(&id, RecognitionOutcome::Failed { message: "x" })
            .expect("finish on upgraded schema");
    }

    #[test]
    fn attempts_grouped_by_capture_fetches_a_page_in_one_query() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let first = committed_take(&mut store, &ramp(30, 0)).record.id.clone();
        let second = committed_take(&mut store, &ramp(30, 1)).record.id.clone();
        store
            .begin_recognition(&first, "starling:parakeet", None)
            .expect("begin first");
        store
            .begin_recognition(&second, "starling:parakeet", None)
            .expect("begin second");
        store
            .finish_recognition(&first, RecognitionOutcome::Failed { message: "nope" })
            .expect("fail first");
        store
            .begin_recognition(&first, "starling:parakeet", None)
            .expect("retry first");

        let grouped = store
            .attempts_grouped_by_capture(&[first.clone(), second.clone()])
            .expect("grouped");
        assert_eq!(grouped.len(), 2, "both captures present");
        assert_eq!(grouped[&first].len(), 2, "history + retry, oldest first");
        assert_eq!(grouped[&first][0].status, "failed");
        assert_eq!(grouped[&first][1].status, "started");
        assert_eq!(grouped[&second].len(), 1);

        // Ids with no attempts are absent, not empty entries.
        let empty = store
            .attempts_grouped_by_capture(&["c_nope".to_string()])
            .expect("grouped");
        assert!(empty.is_empty());
    }

    #[test]
    fn higher_schema_version_database_is_refused_without_mutation() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(50, 0));
        drop(store);
        // A future build stamped a higher version.
        {
            let conn = Connection::open(dir.path().join("v2").join(DB_FILE)).expect("open");
            conn.execute(
                "UPDATE meta SET value = '99' WHERE key = 'schema_version'",
                [],
            )
            .expect("bump version");
        }

        match StoreV2::open(dir.path().join("v2")) {
            Err(StoreV2Error::SchemaTooNew { found, supported }) => {
                assert_eq!(found, 99);
                assert_eq!(supported, SCHEMA_VERSION);
            }
            other => panic!("expected SchemaTooNew, got {other:?}"),
        }

        // The refusal must not have "fixed" the version: a future build
        // still sees its own 99.
        let conn = Connection::open(dir.path().join("v2").join(DB_FILE)).expect("reopen");
        let version: String = conn
            .query_row("SELECT value FROM meta WHERE key = 'schema_version'", [], |row| {
                row.get(0)
            })
            .expect("version");
        assert_eq!(version, "99");
        // …and the row this build wrote is still there, unread but intact.
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM captures", [], |row| row.get(0))
            .expect("count");
        assert_eq!(rows, 1, "no row was deleted by the refusal: {take:?}");
    }

    #[test]
    fn extra_json_round_trips_unknown_fields_verbatim() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let mut meta = TakeMeta::for_device("test-device");
        meta.extra_json = Some(r#"{"knownField":"kept"}"#.to_string());
        let mut take = store.begin_take(meta).expect("begin");
        take.append_frames(&ramp(100, 0)).expect("append");
        let committed = take.finish(&mut store).expect("finish");
        let id = committed.record.id.clone();

        // Write-side round trip.
        let record = store.get_capture(&id).expect("get").expect("present");
        assert_eq!(record.extra_json.as_deref(), Some(r#"{"knownField":"kept"}"#));

        // A newer writer on the same schema version added a field we do
        // not know about, directly in the column.
        let injected = r#"{"knownField":"kept","newerField":{"nested":[1,2,3]},"z":null}"#;
        store
            .conn
            .execute(
                "UPDATE captures SET extra_json = ?1 WHERE id = ?2",
                params![injected, id],
            )
            .expect("inject");

        let record = store.get_capture(&id).expect("get").expect("present");
        assert_eq!(
            record.extra_json.as_deref(),
            Some(injected),
            "unknown fields are preserved verbatim on read"
        );

        // A status-only update must not touch the column.
        store
            .update_capture_status(&id, CaptureStatus::Complete, None)
            .expect("update status");
        let raw: String = store
            .conn
            .query_row(
                "SELECT extra_json FROM captures WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .expect("raw extra");
        assert_eq!(raw, injected, "untouched extra_json stays byte-identical");

        // A noted update merges without dropping unknown keys.
        store
            .update_capture_status(&id, CaptureStatus::Interrupted, Some("recovery note"))
            .expect("noted update");
        let record = store.get_capture(&id).expect("get").expect("present");
        let extra: serde_json::Value =
            serde_json::from_str(record.extra_json.as_deref().expect("extra")).expect("parse");
        assert_eq!(extra["newerField"]["nested"], serde_json::json!([1, 2, 3]));
        assert_eq!(extra["z"], serde_json::Value::Null);
        assert_eq!(extra["recovery"], "recovery note");
    }

    #[test]
    fn attempts_round_trip_and_cascade_with_their_capture() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(60, 0));
        let id = take.record.id.clone();

        let extra = r#"{"requestId":"req-1","durationSeconds":3.5}"#;
        store
            .insert_attempt(&AttemptRecord {
                id: "a_one".to_string(),
                capture_id: id.clone(),
                backend: "whisper".to_string(),
                model_hash: Some("sha256:abc".to_string()),
                language: Some("en".to_string()),
                options_json: Some(r#"{"beam":1}"#.to_string()),
                text: "hello v2".to_string(),
                partial_or_final: "final".to_string(),
                status: "completed".to_string(),
                timing_json: Some(r#"{"totalMs":120}"#.to_string()),
                extra_json: Some(extra.to_string()),
                created_utc: None,
            })
            .expect("insert attempt");

        // Insert stamped it with a creation time (the summary's real
        // updated-at source), and it round-trips.
        let attempts = store.attempts_for(&id).expect("attempts");
        assert_eq!(attempts.len(), 1);
        assert!(attempts[0].created_utc.is_some(), "stamped on insert");

        let attempts = store.attempts_for(&id).expect("attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].text, "hello v2");
        assert_eq!(attempts[0].extra_json.as_deref(), Some(extra));

        // Deleting the capture cascades the attempts away.
        store.delete_capture(&id).expect("delete");
        assert!(store.attempts_for(&id).expect("attempts").is_empty());
    }

    // ---- §4 crash protocol, one fault per step ------------------------

    #[test]
    fn kill_after_staging_fsync_recovers_interrupted_with_gap_flag() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        // Step 1 only: frames + an fsynced boundary, then an unconfirmed
        // tail (the bytes a crash may or may not have reached disk).
        let mut take = store
            .begin_take(TakeMeta::for_device("mic"))
            .expect("begin");
        let confirmed = ramp(800, 0);
        take.append_frames(&confirmed).expect("append");
        let acked = take.write_boundary().expect("boundary");
        assert_eq!(acked, 800);
        take.append_frames(&ramp(120, 800)).expect("unconfirmed tail");
        let id = take.id().to_string();
        drop(take); // "crash"

        assert!(store.staging_path(&id).exists());
        assert!(!store.audio_path(&id).exists());
        assert!(store.get_capture(&id).expect("get").is_none());

        let report = store.reconcile().expect("reconcile");
        assert_eq!(report.recovered_torn.len(), 1);
        assert_eq!(report.recovered_torn[0].id, id);
        assert!(
            report.recovered_torn[0].torn_tail_bytes > 0,
            "the discarded tail is reported as the gap"
        );

        // The take is usable: sealed audio in audio/, interrupted row.
        let record = store.get_capture(&id).expect("get").expect("row");
        assert_eq!(record.status, CaptureStatus::Interrupted);
        assert_eq!(record.frame_count, 800, "only the verified prefix");
        assert!(record.secure_field, "an unknown marker excludes the take");
        let audio = store.load_audio(&id).expect("load audio");
        assert!(audio.finalized, "recovery sealed a valid trailer");
        assert_eq!(audio.samples, confirmed);
        assert_eq!(audio.torn_tail_bytes, 0, "the sealed file has no torn tail");
        let extra: serde_json::Value =
            serde_json::from_str(record.extra_json.as_deref().expect("note")).expect("parse");
        let note = extra["recovery"].as_str().expect("recovery note");
        assert!(note.contains("gap flagged"), "{note}");

        // Staging is empty; a rerun changes nothing.
        assert!(journal_ids_in(&store.root.join(STAGING_DIR)).is_empty());
        let rerun = store.reconcile().expect("rerun");
        assert!(rerun.recovered_torn.is_empty());
        assert!(rerun.orphan_sessions.is_empty());
    }

    #[test]
    fn kill_after_finalize_in_staging_promotes_and_creates_orphan_row() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        // Step 2's first half only: trailer + fsyncs, no rename, no row.
        let mut take = store
            .begin_take(TakeMeta::for_device("mic"))
            .expect("begin");
        let samples = ramp(500, 0);
        take.append_frames(&samples).expect("append");
        take.write_boundary().expect("boundary");
        let finalized = take.finalize().expect("finalize");
        let id = finalized.id.clone();

        assert!(store.staging_path(&id).exists());
        assert!(!store.audio_path(&id).exists());

        let report = store.reconcile().expect("reconcile");
        assert_eq!(report.recovered_torn.len(), 1, "promoted via the torn path");
        assert_eq!(report.recovered_torn[0].torn_tail_bytes, 0);
        assert_eq!(report.promoted_finalized, vec![id.clone()]);
        assert!(!store.staging_path(&id).exists(), "staging drained");
        let record = store.get_capture(&id).expect("get").expect("row");
        assert_eq!(record.status, CaptureStatus::Interrupted);
        assert_eq!(record.frame_count, 500);
        let audio = store.load_audio(&id).expect("audio");
        assert_eq!(audio.samples, samples);
    }

    #[test]
    fn kill_after_rename_before_commit_creates_orphan_session() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        // Steps 1–2 only: finalized + promoted, transaction never ran.
        let mut take = store
            .begin_take(TakeMeta::for_device("mic"))
            .expect("begin");
        let samples = ramp(300, 0);
        take.append_frames(&samples).expect("append");
        let finalized = take.finalize().expect("finalize");
        store.promote_from_staging(&finalized.id).expect("promote");
        let id = finalized.id;

        assert!(store.audio_path(&id).exists());
        assert!(store.get_capture(&id).expect("get").is_none());

        let report = store.reconcile().expect("reconcile");
        assert_eq!(report.orphan_sessions, vec![id.clone()]);
        let record = store.get_capture(&id).expect("get").expect("orphan row");
        assert_eq!(record.status, CaptureStatus::Interrupted);
        assert_eq!(record.frame_count, 300);
        assert!(record.secure_field, "an unknown marker excludes the take");
        let extra: serde_json::Value =
            serde_json::from_str(record.extra_json.as_deref().expect("note")).expect("parse");
        assert!(
            extra["recovery"]
                .as_str()
                .expect("note")
                .contains("never committed"),
            "the orphan note states the crash window"
        );
        // Linked: row and audio agree on the samples.
        let audio = store.load_audio(&id).expect("audio");
        assert_eq!(audio.samples, samples);
    }

    #[test]
    fn full_protocol_take_commits_and_acks_only_after_commit() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let samples = ramp(1_000, 0);
        let committed = committed_take(&mut store, &samples);
        let id = committed.record.id.clone();

        // Step 4 complete: row + finalized audio + clean staging.
        let record = store.get_capture(&id).expect("get").expect("row");
        assert_eq!(record.status, CaptureStatus::Complete);
        assert_eq!(record.ack_sample_index, 1_000);
        assert_eq!(record.frame_count, 1_000);
        let audio = store.load_audio(&id).expect("audio");
        assert!(audio.finalized);
        assert_eq!(audio.samples, samples);
        assert!(journal_ids_in(&store.root.join(STAGING_DIR)).is_empty());
        // The row's journal hash is exactly the sealed trailer hash.
        let parsed = read_journal(&store.audio_path(&id)).expect("parse");
        assert_eq!(
            record.journal_hash,
            format!("{:016x}", samples_hash(&parsed.samples))
        );

        // The ack cannot precede the commit: a commit that cannot run
        // (here: duplicate primary key, i.e. the row already exists)
        // surfaces an error and leaves the orphan for reconciliation.
        let mut take = store
            .begin_take(TakeMeta::for_device("mic"))
            .expect("begin");
        take.append_frames(&samples).expect("append");
        let finalized = take.finalize().expect("finalize");
        store.promote_from_staging(&finalized.id).expect("promote");
        let mut record = CaptureRecord {
            id: finalized.id.clone(),
            created_utc: finalized.created_utc,
            tz: finalized.meta.tz,
            device: finalized.meta.device,
            actual_rate: finalized.sample_rate,
            policy: finalized.meta.policy,
            frame_count: finalized.total_samples,
            ack_sample_index: finalized.total_samples,
            journal_hash: finalized.content_hash,
            status: CaptureStatus::Complete,
            retention_class: finalized.meta.retention_class,
            extra_json: finalized.meta.extra_json,
            secure_field: finalized.meta.secure_field,
        };
        record.id = id.clone(); // force the primary-key collision
        assert!(
            store.commit_capture(&record).is_err(),
            "a failed commit must not ack"
        );
        // Recovery converges: the second journal becomes an orphan take.
        let report = store.reconcile().expect("reconcile");
        assert_eq!(report.orphan_sessions.len(), 1);
        assert_eq!(report.orphan_sessions[0], finalized.id);
    }

    #[test]
    fn checkpoint_policy_is_tunable_and_survives_reopen() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        assert_eq!(WAL_CHECKPOINT_EVERY_COMMITS, 64, "documented default (D11)");
        store.set_checkpoint_every_commits(2);
        for i in 0..3 {
            committed_take(&mut store, &ramp(10, i));
        }
        store.checkpoint().expect("explicit checkpoint");
        drop(store);

        let mut store = store_in(&dir);
        assert_eq!(store.list_records(0, 10).expect("list").total, 3);
        committed_take(&mut store, &ramp(10, 99));
        assert_eq!(store.list_records(0, 10).expect("list").total, 4);
    }

    // ---- reconciliation matrix ----------------------------------------

    #[test]
    fn reconciliation_matrix_covers_every_documented_state() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        // (1) Torn staging journal.
        let mut torn = store.begin_take(TakeMeta::for_device("mic")).expect("begin");
        torn.append_frames(&ramp(400, 0)).expect("append");
        torn.write_boundary().expect("boundary");
        torn.append_frames(&ramp(50, 400)).expect("tail");
        let torn_id = torn.id().to_string();
        drop(torn);

        // (2) Finalized journal still in staging.
        let mut staged = store.begin_take(TakeMeta::for_device("mic")).expect("begin");
        staged.append_frames(&ramp(200, 0)).expect("append");
        let staged_finalized = staged.finalize().expect("finalize");
        let staged_id = staged_finalized.id;

        // (3) Promoted audio with no row.
        let mut orphan = store.begin_take(TakeMeta::for_device("mic")).expect("begin");
        orphan.append_frames(&ramp(600, 0)).expect("append");
        let orphan_finalized = orphan.finalize().expect("finalize");
        store.promote_from_staging(&orphan_finalized.id).expect("promote");
        let orphan_id = orphan_finalized.id;

        // (4) A committed take whose audio later vanished.
        let lost = committed_take(&mut store, &ramp(700, 0));
        let lost_id = lost.record.id.clone();
        std::fs::remove_file(store.audio_path(&lost_id)).expect("remove audio");

        // (5) A tombstoned take: delete crashed between the quarantine
        // rename and the transaction.
        let deleted = committed_take(&mut store, &ramp(100, 0));
        let deleted_id = deleted.record.id.clone();
        let quarantine = store.quarantine_path(&deleted_id);
        std::fs::rename(store.audio_path(&deleted_id), &quarantine).expect("quarantine rename");
        sync_dir(&store.root.join(QUARANTINE_DIR)).expect("fsync");

        // One healthy control.
        let healthy = committed_take(&mut store, &ramp(50, 0));
        let healthy_id = healthy.record.id.clone();

        let report = store.reconcile().expect("reconcile");

        let recovered: HashSet<String> = report
            .recovered_torn
            .iter()
            .map(|take| take.id.clone())
            .collect();
        assert!(recovered.contains(&torn_id), "torn staging recovered");
        assert!(recovered.contains(&staged_id), "finalized staging promoted");
        assert_eq!(report.orphan_sessions, vec![orphan_id.clone()]);
        assert_eq!(report.marked_interrupted, vec![lost_id.clone()]);
        assert!(report.completed_deletes.contains(&deleted_id));

        // End states:
        let torn_row = store.get_capture(&torn_id).expect("get").expect("row");
        assert_eq!(torn_row.status, CaptureStatus::Interrupted);
        assert_eq!(torn_row.frame_count, 400, "verified prefix only");
        assert!(store.audio_path(&torn_id).exists());

        let staged_row = store.get_capture(&staged_id).expect("get").expect("row");
        assert_eq!(staged_row.status, CaptureStatus::Interrupted);
        assert_eq!(staged_row.frame_count, 200);

        let orphan_row = store.get_capture(&orphan_id).expect("get").expect("row");
        assert_eq!(orphan_row.status, CaptureStatus::Interrupted);

        let lost_row = store.get_capture(&lost_id).expect("get").expect("row kept");
        assert_eq!(lost_row.status, CaptureStatus::Interrupted);

        assert!(store.get_capture(&deleted_id).expect("get").is_none());
        assert!(quarantine.exists(), "the tombstoned bytes await retention");

        let healthy_row = store.get_capture(&healthy_id).expect("get").expect("row");
        assert_eq!(healthy_row.status, CaptureStatus::Complete);

        // Staging is fully drained, and a second pass is a no-op.
        assert!(journal_ids_in(&store.root.join(STAGING_DIR)).is_empty());
        let rerun = store.reconcile().expect("rerun");
        assert!(rerun.recovered_torn.is_empty());
        assert!(rerun.orphan_sessions.is_empty());
        assert!(rerun.marked_interrupted.is_empty());
    }

    #[test]
    fn tombstoned_captures_never_resurrect() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let take = committed_take(&mut store, &ramp(80, 0));
        let id = take.record.id.clone();
        store.delete_capture(&id).expect("delete");

        // A restored backup puts the journal back under audio/ while its
        // tombstone sits in quarantine/ and the DB: it must stay dead.
        std::fs::copy(store.quarantine_path(&id), store.audio_path(&id)).expect("restore file");
        let report = store.reconcile().expect("reconcile");
        assert!(report.completed_deletes.contains(&id));
        assert!(
            store.get_capture(&id).expect("get").is_none(),
            "no resurrection"
        );
        assert!(
            !store.audio_path(&id).exists(),
            "the file went back to quarantine"
        );

        // Idempotent delete; unknown ids are fine.
        store.delete_capture(&id).expect("redelete");
        store.delete_capture("c_never_existed").expect("unknown id");
    }

    #[test]
    fn unreadable_and_empty_journals_are_kept_and_reported() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        std::fs::write(store.staging_path("c_garbage"), b"not a journal").expect("garbage");
        // An empty (header-only) journal: crash right after start.
        store.begin_take(TakeMeta::for_device("mic")).expect("empty take");

        let report = store.reconcile().expect("reconcile");
        assert_eq!(report.unreadable.len(), 1);
        assert_eq!(report.unreadable[0].0, "c_garbage");
        assert_eq!(report.empty_journals.len(), 1);
        // Both kept in place, no rows invented.
        assert!(store.staging_path("c_garbage").exists());
        assert_eq!(store.list_records(0, 10).expect("list").total, 0);
    }

    // ---- bounded listing + lazy audio ---------------------------------

    #[test]
    fn listing_is_metadata_only_and_isolates_damaged_rows() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let good = committed_take(&mut store, &ramp(40, 0)).record.id.clone();
        let missing = committed_take(&mut store, &ramp(40, 1)).record.id.clone();
        std::fs::remove_file(store.audio_path(&missing)).expect("remove audio");
        let corrupt = committed_take(&mut store, &ramp(40, 2)).record.id.clone();
        std::fs::write(store.audio_path(&corrupt), b"garbage bytes").expect("corrupt audio");

        // A row this build cannot even map (a same-version future writer
        // used a new status).
        store
            .conn
            .execute(
                "INSERT INTO captures(id, created_utc, tz, device, actual_rate, policy,
                    frame_count, ack_sample_index, journal_hash, status, retention_class)
                 VALUES ('c_future', '2026-01-01T00:00:00.000Z', 'UTC', '', 16000, 'default',
                    1, 1, '0000000000000000', 'brand-new-status', 'standard')",
                [],
            )
            .expect("future row");

        let page = store.list_records(0, 10).expect("listing never aborts");
        assert_eq!(page.total, 4);
        assert_eq!(page.offset, 0);

        let find = |id: &str| {
            page.records
                .iter()
                .find(|record| record.id() == id)
                .unwrap_or_else(|| panic!("record {id} missing"))
        };
        match find(&good) {
            ListedCapture::Capture(listing) => assert!(listing.problems.is_empty()),
            other => panic!("good record must list clean: {other:?}"),
        }
        match find(&missing) {
            ListedCapture::Capture(listing) => {
                assert!(listing.problems.iter().any(|p| p.contains("missing")))
            }
            other => panic!("missing-audio record must still list: {other:?}"),
        }
        match find(&corrupt) {
            ListedCapture::Capture(listing) => {
                assert!(listing.problems.iter().any(|p| p.contains("header")))
            }
            other => panic!("corrupt-audio record must still list: {other:?}"),
        }
        match find("c_future") {
            ListedCapture::Damaged(damaged) => {
                assert!(damaged.reason.contains("brand-new-status"))
            }
            other => panic!("unmappable row must surface as damaged: {other:?}"),
        }

        // Paging slices the same ordering with a true total.
        let page = store.list_records(1, 2).expect("page");
        assert_eq!(page.total, 4);
        assert_eq!(page.records.len(), 2);

        // Lazy audio: loads on demand, with typed failures.
        let audio = store.load_audio(&good).expect("lazy load");
        assert_eq!(audio.samples.len(), 40);
        match store.load_audio(&missing) {
            Err(StoreV2Error::Invalid(reason)) => assert!(reason.contains("no audio journal")),
            other => panic!("expected Invalid, got {other:?}"),
        }
        match store.load_audio("c_missing_row") {
            Err(StoreV2Error::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
        for bad in ["", "a/b", ".."] {
            assert!(store.load_audio(bad).is_err());
        }
    }

    // ---- hand-built WAV fixture ---------------------------------------

    /// A minimal canonical WAV built by hand (same shape the storage tests
    /// use) so this suite stays independent of the audio encoder.
    fn wav_bytes(samples: &[f32]) -> Vec<u8> {
        // PCM16 mono 16 kHz, one i16 per f32 sample (clamped).
        let data: Vec<i16> = samples
            .iter()
            .map(|&sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        let data_len = (data.len() * 2) as u32;
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for sample in data {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }

    // ---- daily-use operations -----------------------------------------

    /// A finalized capture journal in the recorder's own tree, outside the
    /// store (the live-capture shape: `journals/<id>.sj`).
    fn external_journal(dir: &TempDir, id: &str, samples: &[f32], finalize: bool) -> PathBuf {
        let tree = dir.path().join("capture-tree");
        std::fs::create_dir_all(&tree).expect("capture tree");
        let mut writer =
            JournalWriter::create_named(&tree, id.to_string(), 16_000).expect("writer");
        writer.append_frames(samples).expect("append");
        writer.write_boundary().expect("boundary");
        if finalize {
            writer.finalize().expect("finalize");
        }
        tree.join(format!("{id}.sj"))
    }

    #[test]
    fn adopt_journal_moves_a_finalized_journal_in_and_commits() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let samples = ramp(300, 0);
        let source = external_journal(&dir, "j_adopt", &samples, true);

        let record = store.adopt_journal(&source, None).expect("adopt");
        assert_eq!(record.id, "j_adopt");
        assert_eq!(record.status, CaptureStatus::Complete);
        assert_eq!(record.actual_rate, 16_000);
        assert_eq!(record.frame_count, 300);

        // The file moved; the row points at it; the audio round-trips.
        assert!(!source.exists(), "the journal left the capture tree");
        assert!(store.audio_path("j_adopt").exists());
        let audio = store.load_audio("j_adopt").expect("audio");
        assert!(audio.finalized);
        assert_eq!(audio.samples, samples);
        let stored = store.get_capture("j_adopt").expect("get").expect("row");
        assert_eq!(
            stored.journal_hash,
            format!("{:016x}", samples_hash(&samples))
        );

        // A second adoption of anything under that id refuses to overwrite.
        let again = external_journal(&dir, "j_adopt", &ramp(5, 0), true);
        match store.adopt_journal(&again, None) {
            Err(StoreV2Error::Invalid(reason)) => {
                assert!(reason.contains("refusing to overwrite"), "{reason}")
            }
            other => panic!("expected refusal, got {other:?}"),
        }
    }

    #[test]
    fn adopt_journal_seals_a_torn_tail_and_flags_the_gap() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        // Frames + boundary + an unconfirmed tail, never finalized: the
        // quiesce-fault / kill shape.
        let tree = dir.path().join("capture-tree");
        std::fs::create_dir_all(&tree).expect("tree");
        let mut writer =
            JournalWriter::create_named(&tree, "j_torn".to_string(), 16_000).expect("writer");
        let confirmed = ramp(500, 0);
        writer.append_frames(&confirmed).expect("append");
        writer.write_boundary().expect("boundary");
        writer.append_frames(&ramp(40, 500)).expect("tail");
        drop(writer);
        let source = tree.join("j_torn.sj");

        let record = store
            .adopt_journal(&source, Some("salvage note from the caller"))
            .expect("adopt");
        assert_eq!(record.status, CaptureStatus::Interrupted);
        assert_eq!(record.frame_count, 500, "verified prefix only");
        let extra: serde_json::Value =
            serde_json::from_str(record.extra_json.as_deref().expect("note")).expect("parse");
        let note = extra["recovery"].as_str().expect("note");
        assert!(note.contains("salvage note from the caller"), "{note}");
        assert!(note.contains("gap flagged"), "{note}");

        let audio = store.load_audio("j_torn").expect("audio");
        assert!(audio.finalized, "adoption sealed a valid trailer");
        assert_eq!(audio.samples, confirmed);
        assert_eq!(audio.torn_tail_bytes, 0);
        assert!(!source.exists());
    }

    #[test]
    fn adopt_journal_refuses_empty_journals_without_touching_the_source() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        // Header-only journal: a take that never reached its first boundary.
        let source = external_journal(&dir, "j_empty", &[], true);
        match store.adopt_journal(&source, None) {
            Err(StoreV2Error::NoVerifiedSamples { id }) => {
                assert_eq!(id, "j_empty");
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }
        assert!(source.exists(), "the source is kept for the caller");
        assert!(
            store
                .list_records(0, 10)
                .expect("list")
                .records
                .is_empty()
        );
    }

    #[test]
    fn save_wav_capture_round_trips_through_the_crash_protocol() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let samples = ramp(240, 0);
        let committed = store
            .save_wav_capture(&wav_bytes(&samples), TakeMeta::for_device("import"))
            .expect("save");
        assert_eq!(committed.record.status, CaptureStatus::Complete);
        assert_eq!(committed.record.device, "import");
        assert_eq!(committed.record.frame_count, 240);

        // Metadata-only listing sees it; the audio is bit-identical in the
        // sample domain; staging is drained.
        assert_eq!(store.list_records(0, 10).expect("list").total, 1);
        let audio = store.load_audio(&committed.record.id).expect("audio");
        assert!(audio.finalized);
        assert_eq!(audio.samples.len(), 240);
        assert!(journal_ids_in(&store.root.join(STAGING_DIR)).is_empty());

        // Empty audio is refused honestly (the decoder rejects a zero-length
        // data chunk before anything is written).
        assert!(
            store
                .save_wav_capture(&wav_bytes(&[]), TakeMeta::for_device("x"))
                .is_err()
        );
    }

    #[test]
    fn the_staged_protocol_walks_the_same_steps_without_holding_the_store() {
        // The step-wise protocol a lock-sharing caller drives: begin under
        // the guard, write + finalize with only the take's own journal,
        // commit back under the guard. The end state must be exactly
        // `save_wav_capture`'s (same row shape, same audio, staging swept).
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let samples = ramp(120, 0);

        // `&self` borrows only — the point of the split: nothing here
        // needs `&mut StoreV2` until the commit.
        let mut take = store
            .begin_take_at_rate(16_000, TakeMeta::for_device("staged"))
            .expect("begin");
        let staged_id = take.id().to_string();
        take.append_and_seal(&samples).expect("append + seal");
        let finalized = take.finalize().expect("finalize");
        let committed = finalized
            .commit_marked(&mut store, CommitMark::Complete)
            .expect("commit");

        assert_eq!(committed.record.id, staged_id);
        assert_eq!(committed.record.status, CaptureStatus::Complete);
        assert_eq!(committed.record.frame_count, 120);
        let audio = store.load_audio(&staged_id).expect("audio");
        assert_eq!(audio.samples, samples);
        assert!(journal_ids_in(&store.root.join(STAGING_DIR)).is_empty());
    }

    #[test]
    fn an_interrupted_commit_writes_status_and_note_in_one_transaction() {
        // R34: a salvaged take committed through the WAV path must land as
        // interrupted with its note in the row's own INSERT — there is no
        // follow-up status update a crash between the two could skip.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let samples = ramp(90, 0);

        let mut take = store
            .begin_take_at_rate(16_000, TakeMeta::for_device("salvage"))
            .expect("begin");
        take.append_and_seal(&samples).expect("append");
        let committed = take
            .finalize()
            .expect("finalize")
            .commit_marked(
                &mut store,
                CommitMark::Interrupted {
                    note: "salvaged and kept as this interrupted recording".to_string(),
                },
            )
            .expect("commit");

        let id = committed.record.id.clone();
        assert_eq!(committed.record.status, CaptureStatus::Interrupted);
        let record = store.get_capture(&id).expect("get").expect("row");
        assert_eq!(record.status, CaptureStatus::Interrupted);
        let note = record.recovery_note().expect("the note landed with the row");
        assert!(note.contains("salvaged and kept"), "{note}");
        let audio = store.load_audio(&id).expect("audio");
        assert_eq!(audio.samples, samples);
    }

    #[test]
    fn read_audio_journal_agrees_with_load_audio() {
        // The lock-free read half of `load_audio`: given only the path (as
        // `audio_journal_path` resolves it under a caller's guard), it
        // yields exactly what the store-owned read yields.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(70, 0));
        let id = take.record.id.clone();

        let path = store.audio_journal_path(&id).expect("path");
        let free_read = read_audio_journal(&path).expect("free read");
        let store_read = store.load_audio(&id).expect("store read");
        assert_eq!(free_read.samples, store_read.samples);
        assert_eq!(free_read.sample_rate, store_read.sample_rate);
        assert_eq!(free_read.finalized, store_read.finalized);

        // The metadata-only half keeps load_audio's typed failures.
        match store.audio_journal_path("c_missing_row") {
            Err(StoreV2Error::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn recognition_lifecycle_started_completed_failed_not_found() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(30, 0));
        let id = take.record.id.clone();

        // Started: one partial row.
        store
            .begin_recognition(&id, "starling:parakeet", Some(r#"{"model":"parakeet"}"#))
            .expect("begin");
        let attempts = store.attempts_for(&id).expect("attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].status, "started");
        assert_eq!(attempts[0].partial_or_final, "partial");
        assert_eq!(attempts[0].backend, "starling:parakeet");

        // Completed: the same row becomes final text with verbatim extra.
        let extra = r#"{"text":"hello","segments":[]}"#;
        store
            .finish_recognition(
                &id,
                RecognitionOutcome::Completed {
                    text: "hello",
                    extra_json: Some(extra),
                },
            )
            .expect("finish");
        let attempts = store.attempts_for(&id).expect("attempts");
        assert_eq!(attempts.len(), 1, "one row per attempt, updated in place");
        assert_eq!(attempts[0].status, "completed");
        assert_eq!(attempts[0].partial_or_final, "final");
        assert_eq!(attempts[0].text, "hello");
        assert_eq!(attempts[0].extra_json.as_deref(), Some(extra));

        // A second finish has nothing started to land on.
        match store.finish_recognition(&id, RecognitionOutcome::Failed { message: "late" }) {
            Err(StoreV2Error::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }

        // Failed retry: history keeps the earlier completion above it.
        store.begin_recognition(&id, "starling:parakeet", None).expect("begin 2");
        store
            .finish_recognition(&id, RecognitionOutcome::Failed { message: "offline" })
            .expect("fail");
        let attempts = store.attempts_for(&id).expect("attempts");
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].status, "completed", "insertion order kept");
        assert_eq!(attempts[1].status, "failed");
        let extra: serde_json::Value =
            serde_json::from_str(attempts[1].extra_json.as_deref().expect("extra"))
                .expect("parse");
        assert_eq!(extra["error"], "offline");

        // Unknown captures refuse up front.
        match store.begin_recognition("c_nope", "starling", None) {
            Err(StoreV2Error::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn a_begin_whose_commit_fails_rolls_back_and_leaves_no_marker() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(30, 0));
        let id = take.record.id.clone();

        // A deferred foreign-key violation passes every statement and
        // fails only at COMMIT, which leaves the transaction open: the
        // guard must roll it back.
        store
            .conn
            .execute_batch(
                "CREATE TABLE commit_fault(
                    capture_id TEXT REFERENCES captures(id) DEFERRABLE INITIALLY DEFERRED);
                 CREATE TRIGGER fail_commit AFTER INSERT ON recognition_attempts
                 BEGIN INSERT INTO commit_fault VALUES ('c_missing'); END;",
            )
            .expect("install the commit fault");
        assert!(store.begin_recognition(&id, "starling", None).is_err());
        assert!(
            store.conn.is_autocommit(),
            "no transaction is left open after the failed commit"
        );
        assert!(store.attempts_for(&id).expect("attempts").is_empty());
        assert!(store.attempt_locks.is_empty());
        let markers = std::fs::read_dir(dir.path().join("v2").join(ATTEMPT_LOCKS_DIR))
            .expect("locks dir")
            .count();
        assert_eq!(markers, 0, "the failed begin's marker is removed");

        // Later writes on the same connection still work.
        store
            .conn
            .execute_batch("DROP TRIGGER fail_commit")
            .expect("remove the commit fault");
        store
            .begin_recognition(&id, "starling", None)
            .expect("begin");
        assert_eq!(store.attempts_for(&id).expect("attempts").len(), 1);
    }

    #[test]
    fn finishing_recognition_on_a_deleted_capture_is_not_found() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(30, 0));
        let id = take.record.id.clone();
        store.begin_recognition(&id, "starling", None).expect("begin");

        // The user's delete wins the race (R21): the cascade removes the
        // in-flight attempt with the row.
        store.delete_capture(&id).expect("delete");
        match store.finish_recognition(
            &id,
            RecognitionOutcome::Completed {
                text: "late",
                extra_json: None,
            },
        ) {
            Err(StoreV2Error::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn a_settle_the_database_refuses_no_longer_reads_as_in_flight() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(30, 0)).record.id.clone();
        let attempt = store.begin_recognition(&id, "starling", None).expect("begin");
        assert!(store.attempt_owned(&attempt));
        store
            .conn
            .execute_batch(
                "CREATE TRIGGER refuse_settles BEFORE UPDATE ON recognition_attempts
                 BEGIN SELECT RAISE(ABORT, 'disk full'); END;",
            )
            .expect("trigger");
        assert!(
            store
                .finish_recognition(&id, RecognitionOutcome::Failed { message: "server" })
                .is_err()
        );
        assert!(!store.attempt_owned(&attempt), "nothing works on it any more");
    }

    #[test]
    fn interrupt_stale_attempts_fails_only_started_rows() {
        let dir = TempDir::new().expect("tempdir");
        // The previous run: one attempt left "started" when the process
        // died, one settled before it did.
        let (live, done) = {
            let mut owner = store_in(&dir);
            let live = committed_take(&mut owner, &ramp(30, 0)).record.id.clone();
            let done = committed_take(&mut owner, &ramp(30, 1)).record.id.clone();
            owner
                .begin_recognition(&live, "starling", None)
                .expect("begin live");
            owner
                .begin_recognition(&done, "starling", None)
                .expect("begin done");
            owner
                .finish_recognition(
                    &done,
                    RecognitionOutcome::Completed {
                        text: "kept",
                        extra_json: None,
                    },
                )
                .expect("finish done");
            drop(owner); // the process is gone; its flocks died with it (#213)
            (live, done)
        };

        let mut store = store_in(&dir);
        let stale = store
            .interrupt_stale_attempts("Interrupted before the server returned a transcript.")
            .expect("interrupt");
        assert_eq!(stale, vec![live.clone()]);

        let attempts = store.attempts_for(&live).expect("attempts");
        assert_eq!(attempts[0].status, "failed");
        let extra: serde_json::Value =
            serde_json::from_str(attempts[0].extra_json.as_deref().expect("extra"))
                .expect("parse");
        assert_eq!(
            extra["error"],
            "Interrupted before the server returned a transcript."
        );
        let attempts = store.attempts_for(&done).expect("attempts");
        assert_eq!(attempts[0].status, "completed", "terminal rows untouched");

        // Idempotent: a rerun finds nothing started.
        assert!(store
            .interrupt_stale_attempts("again")
            .expect("rerun")
            .is_empty());
    }

    // ---- cross-instance attempt ownership (#213) -----------------------

    #[test]
    fn a_live_owner_blocks_another_instances_startup_sweep() {
        let dir = TempDir::new().expect("tempdir");
        let mut owner = store_in(&dir);
        let id = committed_take(&mut owner, &ramp(30, 0)).record.id.clone();
        owner
            .begin_recognition(&id, "starling:parakeet", None)
            .expect("begin");

        // A second instance sweeps at ITS startup: the owner's flock is
        // held, so the attempt is not stale — the file-store analog of the
        // Electron reference's `transcriptionInFlight` guard (#144, #213).
        // Instance B must not phantom-fail instance A's in-flight attempt.
        let mut sweeper = store_in(&dir);
        let swept = sweeper
            .interrupt_stale_attempts("Interrupted before the server returned a transcript.")
            .expect("sweep");
        assert!(swept.is_empty(), "a live owner's attempt is skipped: {swept:?}");
        let attempts = sweeper.attempts_for(&id).expect("attempts");
        assert_eq!(attempts[0].status, "started", "no phantom failure");

        // The owner still settles its own attempt after the other
        // instance's sweep ran.
        owner
            .finish_recognition(
                &id,
                RecognitionOutcome::Completed {
                    text: "late",
                    extra_json: None,
                },
            )
            .expect("the owner finishes");
        let attempts = sweeper.attempts_for(&id).expect("attempts");
        assert_eq!(attempts[0].status, "completed");
    }

    #[test]
    fn a_sweep_fails_the_dead_attempts_and_spares_the_live_ones() {
        let dir = TempDir::new().expect("tempdir");
        // One attempt whose owner crashed, one whose owner is live: the
        // sweep must tell them apart by the flock, not the marker file —
        // both markers exist on disk.
        let dead = {
            let mut crashed = store_in(&dir);
            let dead = committed_take(&mut crashed, &ramp(30, 1)).record.id.clone();
            crashed
                .begin_recognition(&dead, "starling", None)
                .expect("begin dead");
            drop(crashed); // crashed: the OS released its flock
            dead
        };
        let mut owner = store_in(&dir);
        let owned = committed_take(&mut owner, &ramp(30, 0)).record.id.clone();
        owner
            .begin_recognition(&owned, "starling", None)
            .expect("begin owned");

        let mut sweeper = store_in(&dir);
        let swept = sweeper
            .interrupt_stale_attempts("Interrupted before the server returned a transcript.")
            .expect("sweep");
        assert_eq!(swept, vec![dead.clone()], "only the dead owner's attempt");
        assert_eq!(
            sweeper.attempts_for(&dead).expect("attempts")[0].status,
            "failed"
        );
        assert_eq!(
            sweeper.attempts_for(&owned).expect("attempts")[0].status,
            "started"
        );
    }

    #[test]
    fn a_started_row_without_a_marker_is_swept() {
        // A row left 'started' by a pre-marker build (or written by hand):
        // marker presence is never the signal — the absence of a live
        // owner is, so a missing marker sweeps.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(30, 0)).record.id.clone();
        store
            .insert_attempt(&AttemptRecord {
                id: "a_handmade".to_string(),
                capture_id: id.clone(),
                backend: "starling".to_string(),
                model_hash: None,
                language: None,
                options_json: None,
                text: String::new(),
                partial_or_final: "partial".to_string(),
                status: "started".to_string(),
                timing_json: None,
                extra_json: None,
                created_utc: None,
            })
            .expect("insert started row");

        let swept = store
            .interrupt_stale_attempts("Interrupted before the server returned a transcript.")
            .expect("sweep");
        assert_eq!(swept, vec![id.clone()]);
        assert_eq!(store.attempts_for(&id).expect("attempts")[0].status, "failed");
    }

    #[test]
    fn the_owning_process_sweep_spares_its_own_live_attempts() {
        // Within one process the held-marker registry is the signal (also
        // the only signal on flock-less platforms): a sweep racing its own
        // store's live attempt must not fail it.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(20, 0)).record.id.clone();
        store
            .begin_recognition(&id, "starling", None)
            .expect("begin");

        let swept = store.interrupt_stale_attempts("note").expect("sweep");
        assert!(swept.is_empty(), "own live attempt: {swept:?}");
        assert_eq!(store.attempts_for(&id).expect("attempts")[0].status, "started");
    }

    #[test]
    fn a_settling_attempt_takes_its_marker_with_it() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(20, 0)).record.id.clone();
        let attempt = store
            .begin_recognition(&id, "starling", None)
            .expect("begin");
        let marker = dir
            .path()
            .join("v2")
            .join(ATTEMPT_LOCKS_DIR)
            .join(format!("{attempt}.lock"));
        assert!(marker.exists(), "held while the attempt is in flight");

        store
            .finish_recognition(&id, RecognitionOutcome::Failed { message: "x" })
            .expect("finish");
        assert!(!marker.exists(), "released with the settle");
    }

    #[test]
    fn deleting_a_capture_releases_its_attempt_markers() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(20, 0)).record.id.clone();
        let attempt = store
            .begin_recognition(&id, "starling", None)
            .expect("begin");
        let marker = dir
            .path()
            .join("v2")
            .join(ATTEMPT_LOCKS_DIR)
            .join(format!("{attempt}.lock"));
        assert!(marker.exists());

        // The user's delete wins the race (R21); the cascaded attempt's
        // marker must not linger as disk garbage.
        store.delete_capture(&id).expect("delete");
        assert!(!marker.exists(), "the marker went with the rows");
    }

    // ---- ownership evidence and its fallbacks (#213 review) -----------

    #[test]
    fn ownership_evidence_falls_back_to_the_recorded_pid_then_to_held() {
        // Platform-agnostic contract of the ladder: the flock decides
        // when it answered — either way, outright; without an answer the
        // recorded PID's liveness decides; with no answer at all the
        // marker reads as held (the safe direction).
        assert!(attempt_owned_from(FlockEvidence::Held, None));
        assert!(
            attempt_owned_from(FlockEvidence::Held, Some(false)),
            "a held flock means a live owner by construction"
        );
        assert!(!attempt_owned_from(FlockEvidence::Free, None));
        assert!(
            !attempt_owned_from(FlockEvidence::Free, Some(true)),
            "a freed flock means a dead owner by construction"
        );
        assert!(attempt_owned_from(FlockEvidence::Unknown, Some(true)));
        assert!(!attempt_owned_from(FlockEvidence::Unknown, Some(false)));
        assert!(attempt_owned_from(FlockEvidence::Unknown, None));
    }

    #[cfg(unix)]
    #[test]
    fn pid_liveness_answers_for_live_and_dead_processes() {
        // The fallback signal has to actually distinguish owners: this
        // process is alive; a child that exited is not (pid reuse inside
        // the test window is not a practical concern).
        assert!(process_is_alive(std::process::id()));
        let mut child = std::process::Command::new("true").spawn().expect("spawn true");
        let pid = child.id();
        child.wait().expect("wait for the child");
        assert!(!process_is_alive(pid), "pid {pid} has exited");
    }

    #[test]
    fn a_held_marker_names_its_owner_pid() {
        // The PID is the ownership fallback wherever flock cannot answer,
        // so every held marker records one.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        store.hold_attempt_lock("a_marker").expect("hold");
        let text = std::fs::read_to_string(
            dir.path()
                .join("v2")
                .join(ATTEMPT_LOCKS_DIR)
                .join("a_marker.lock"),
        )
        .expect("read the marker");
        assert_eq!(text.trim(), std::process::id().to_string());
        store.release_attempt_lock("a_marker");
    }

    #[test]
    fn the_sweep_reports_each_swept_capture_once() {
        // #213 review: the old sweep's DISTINCT-capture contract must
        // survive the per-attempt loop — two swept attempts on one
        // capture fail together, and the capture is reported once.
        let dir = TempDir::new().expect("tempdir");
        let id = {
            let mut owner = store_in(&dir);
            let id = committed_take(&mut owner, &ramp(30, 0)).record.id.clone();
            owner
                .begin_recognition(&id, "starling", None)
                .expect("begin one");
            owner
                .begin_recognition(&id, "starling", None)
                .expect("begin two");
            drop(owner); // both attempts' owner is gone
            id
        };

        let mut store = store_in(&dir);
        let swept = store
            .interrupt_stale_attempts("Interrupted before the server returned a transcript.")
            .expect("sweep");
        assert_eq!(swept, vec![id.clone()], "one entry per capture");
        let attempts = store.attempts_for(&id).expect("attempts");
        assert_eq!(attempts.len(), 2);
        assert!(attempts.iter().all(|attempt| attempt.status == "failed"));
    }

    #[cfg(unix)]
    #[test]
    fn a_contended_marker_hold_leaves_no_marker_behind() {
        // #213 review: the contention path (a holder at a freshly minted
        // id's marker — external tampering by definition, since ids are
        // unique per attempt) must not leave the marker file on disk.
        let dir = TempDir::new().expect("tempdir");
        let mut holder = store_in(&dir);
        holder.hold_attempt_lock("a_tamper").expect("first hold acquires");

        let mut contender = store_in(&dir);
        let marker = contender.attempt_lock_path("a_tamper");
        assert!(marker.exists());
        assert!(
            contender.hold_attempt_lock("a_tamper").is_err(),
            "the held marker contends"
        );
        assert!(
            !marker.exists(),
            "the contended marker is removed with the error"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_sweep_collects_markers_whose_attempts_are_gone() {
        // #213 review: markers can outlive their rows through paths other
        // than finish/delete; the sweep garbage-collects them — but never
        // a flocked marker, whose holder may be a live owner inside
        // begin_recognition's marker-before-row window.
        let dir = TempDir::new().expect("tempdir");
        let locks = dir.path().join("v2").join(ATTEMPT_LOCKS_DIR);
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(20, 0)).record.id.clone();
        let settled = store
            .begin_recognition(&id, "starling", None)
            .expect("begin");
        store
            .finish_recognition(&id, RecognitionOutcome::Failed { message: "x" })
            .expect("settle"); // takes its marker with it

        // A marker for a settled row, planted back (the crash window
        // between the terminal write and the unlink): garbage.
        std::fs::write(
            locks.join(format!("{settled}.lock")),
            std::process::id().to_string(),
        )
        .expect("plant the settled-row marker");
        // A marker with no row at all and a free flock (a crash between
        // marker creation and the row insert): garbage.
        std::fs::write(locks.join("a_orphan.lock"), std::process::id().to_string())
            .expect("plant the row-less marker");
        // A marker with no row whose flock is held: a live owner may be
        // between marker and row — never touched.
        let live = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(locks.join("a_live.lock"))
            .expect("open the live marker");
        assert_eq!(
            try_flock_exclusive(&live).expect("flock"),
            FlockEvidence::Free,
            "the test handle now holds it, like a mid-begin owner"
        );

        store
            .interrupt_stale_attempts("Interrupted before the server returned a transcript.")
            .expect("sweep");

        assert!(
            !locks.join(format!("{settled}.lock")).exists(),
            "the settled row's marker is collected"
        );
        assert!(
            !locks.join("a_orphan.lock").exists(),
            "the row-less unlocked marker is collected"
        );
        assert!(
            locks.join("a_live.lock").exists(),
            "a flocked marker is never collected"
        );
    }

    #[test]
    fn reconciliation_summary_names_its_findings() {
        let report = ReconciliationReport {
            recovered_torn: vec![RecoveredTake {
                id: "c_1".to_string(),
                torn_tail_bytes: 4,
            }],
            orphan_sessions: vec!["c_2".to_string()],
            ..ReconciliationReport::default()
        };
        assert!(report.has_findings());
        let summary = report.summary();
        assert!(summary.starts_with("Startup scan:"), "{summary}");
        assert!(summary.contains("recovered 2 interrupted recordings"), "{summary}");

        let quiet = ReconciliationReport::default();
        assert!(!quiet.has_findings());
    }

    // ---- leases (§4 ownership) ------------------------------------------

    /// The lease files under a root's `leases/`, as file names.
    fn lease_files(root: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(root.join("leases"))
            .expect("read leases dir")
            .flatten()
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(str::to_string)
            })
            .collect();
        names.sort();
        names
    }

    /// Whether a store's staging tree still holds the id's journal.
    fn store_root_has_staging(store: &StoreV2, id: &str) -> bool {
        store
            .root()
            .join("staging")
            .join(format!("{id}.sj"))
            .exists()
    }

    #[test]
    fn a_lease_records_pid_boot_id_and_heartbeat() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();

        let owner_id = match store.acquire_lease().expect("acquire") {
            LeaseAcquisition::Owner { owner_id, broke } => {
                assert!(broke.is_empty(), "a fresh root has nothing to break");
                owner_id
            }
            other => panic!("expected to own a fresh root, got {other:?}"),
        };

        // One published identity, one heartbeat beside it, no temp left
        // behind, and the records name this process, this boot, right now.
        assert!(lease_files(&root).contains(&format!("{owner_id}.lease")));
        let identity: LeaseIdentity = serde_json::from_str(
            &std::fs::read_to_string(root.join("leases").join(format!("{owner_id}.lease")))
                .expect("read lease"),
        )
        .expect("lease identity parses");
        assert_eq!(identity.pid, std::process::id());
        if cfg!(target_os = "linux") {
            assert!(!identity.boot_id.is_empty(), "boot id is recorded on linux");
            assert_eq!(identity.boot_id, boot_id());
        }
        assert!(!identity.started_utc.is_empty());
        let heartbeat: LeaseHeartbeat = serde_json::from_str(
            &std::fs::read_to_string(root.join("leases").join(format!("{owner_id}.hb")))
                .expect("read heartbeat"),
        )
        .expect("heartbeat parses");
        assert!(
            now_epoch_ms().saturating_sub(heartbeat.heartbeat_ms) < 5_000,
            "the heartbeat is current"
        );
        assert!(!heartbeat.heartbeat_utc.is_empty());
    }

    #[test]
    fn lease_temporaries_are_unique_and_never_a_fixed_tmp_name() {
        let dir = TempDir::new().expect("tempdir");
        let leases = dir.path().join("leases");
        std::fs::create_dir_all(&leases).expect("create leases dir");

        let first = unique_lease_temp(&leases, "l_owner");
        let second = unique_lease_temp(&leases, "l_owner");
        assert_ne!(first, second, "every temporary is unique");
        for temp in [&first, &second] {
            let name = temp.file_name().unwrap().to_str().unwrap();
            assert!(
                name.starts_with("l_owner."),
                "the temp is named after its owner: {name}"
            );
            assert!(name.ends_with(".tmp"));
            assert_ne!(name, ".tmp", "no shared fixed temp name exists");
        }
        // And the one real acquisition leaves no temporary behind at all.
        let mut store = StoreV2::open(dir.path().join("v2")).expect("open");
        store.acquire_lease().expect("acquire");
        let files = lease_files(store.root());
        assert!(
            files.iter().all(|name| !name.ends_with(".tmp")),
            "no temp leftovers: {files:?}"
        );
        assert_eq!(
            files.iter().filter(|name| name.ends_with(".lease")).count(),
            1,
            "exactly one published lease: {files:?}"
        );
    }

    #[test]
    fn a_second_process_is_a_client_not_a_competitor() {
        let dir = TempDir::new().expect("tempdir");
        let mut owner = store_in(&dir);
        let owner_id = match owner.acquire_lease().expect("owner acquires") {
            LeaseAcquisition::Owner { owner_id, .. } => owner_id,
            other => panic!("fresh root must be acquirable, got {other:?}"),
        };

        // The owner has an in-flight take: a staging journal it may still
        // be writing.
        let mut take = owner
            .begin_take(TakeMeta::for_device("test-device"))
            .expect("begin take");
        take.append_frames(&ramp(20, 0)).expect("append");
        take.write_boundary().expect("boundary");
        let take_id = take.id().to_string();

        // The second process on the same root: a client, not a competitor.
        let mut client = StoreV2::open(owner.root()).expect("second open");
        match client.acquire_lease().expect("client acquires") {
            LeaseAcquisition::Client { owner } => {
                assert_eq!(owner.owner_id, owner_id);
                assert_eq!(owner.pid, std::process::id());
                assert!(owner.alive);
            }
            other => panic!("a live foreign owner must make this a client, got {other:?}"),
        }

        // The client's reconcile defers the owner's in-flight state
        // instead of sealing and promoting a live take out from under it.
        let report = client.reconcile().expect("client reconcile");
        assert_eq!(report.deferred_to_live_owner, vec![take_id.clone()]);
        assert!(
            store_root_has_staging(&client, &take_id),
            "staging untouched"
        );
        assert!(client.get_capture(&take_id).expect("row").is_none());
        assert!(
            !report.has_findings(),
            "a deferral is not a finding: {report:?}"
        );

        // The owner finishes its take normally afterwards.
        take.finalize()
            .expect("finalize")
            .commit_marked(&mut owner, CommitMark::Complete)
            .expect("commit");
        assert!(owner.get_capture(&take_id).expect("row").is_some());

        // And with the staging drained, a client reconcile has nothing
        // left to defer.
        let report = client.reconcile().expect("client reconcile again");
        assert!(report.deferred_to_live_owner.is_empty());

        // The owner's own lease_status shows itself; the client's view of
        // the same file agrees it is alive and not the client's.
        let owner_view = owner.lease_status().expect("owner status");
        assert_eq!(owner_view.len(), 1);
        assert!(owner_view[0].mine && owner_view[0].alive);
        let client_view = client.lease_status().expect("client status");
        assert_eq!(client_view.len(), 1);
        assert!(!client_view[0].mine && client_view[0].alive);
    }

    #[test]
    fn a_client_reconcile_still_completes_tombstones_and_row_repairs() {
        // Client mode defers *in-flight* salvage only: idempotent row-side
        // repairs (tombstone completion, missing-audio marking) still run —
        // they cannot touch a take whose row does not exist yet.
        let dir = TempDir::new().expect("tempdir");
        let mut owner = store_in(&dir);
        let committed = committed_take(&mut owner, &ramp(30, 0)).record.id.clone();
        let doomed = committed_take(&mut owner, &ramp(30, 1)).record.id.clone();
        owner.delete_capture(&doomed).expect("delete");

        // Row without audio: mark-interrupted repair material.
        std::fs::remove_file(owner.root().join("audio").join(format!("{committed}.sj")))
            .expect("remove audio");

        owner.acquire_lease().expect("owner");
        let mut client = StoreV2::open(owner.root()).expect("client");
        assert!(matches!(
            client.acquire_lease().expect("client"),
            LeaseAcquisition::Client { .. }
        ));

        let report = client.reconcile().expect("client reconcile");
        assert!(
            report.marked_interrupted.contains(&committed),
            "the row-side repair ran: {:?}",
            report
        );
        // The quarantined journal is still waiting for the retention sweep
        // (never-delete-until-swept — a client changes nothing there).
        assert!(owner
            .root()
            .join("quarantine")
            .join(format!("{doomed}.sj"))
            .exists());
    }

    #[test]
    fn reacquiring_keeps_the_owner_id_and_renews_the_heartbeat() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let first = match store.acquire_lease().expect("acquire") {
            LeaseAcquisition::Owner { owner_id, .. } => owner_id,
            other => panic!("{other:?}"),
        };
        let identity_of = |store: &StoreV2| {
            serde_json::from_str::<LeaseIdentity>(
                &std::fs::read_to_string(store.lease_path(&first)).expect("lease"),
            )
            .expect("identity")
        };
        let heartbeat_of = |store: &StoreV2| {
            serde_json::from_str::<LeaseHeartbeat>(
                &std::fs::read_to_string(store.heartbeat_path(&first)).expect("heartbeat"),
            )
            .expect("heartbeat")
        };
        let before_identity = identity_of(&store);
        let before_heartbeat = heartbeat_of(&store);

        assert!(store.heartbeat_lease().expect("heartbeat"));
        let second = match store.acquire_lease().expect("re-acquire") {
            LeaseAcquisition::Owner { owner_id, .. } => owner_id,
            other => panic!("{other:?}"),
        };
        assert_eq!(first, second, "the identity is stable");
        assert_eq!(
            identity_of(&store).started_utc,
            before_identity.started_utc,
            "the identity file is immutable across renewals"
        );
        assert!(
            heartbeat_of(&store).heartbeat_ms >= before_heartbeat.heartbeat_ms,
            "the heartbeat moved forward"
        );
        // Not holding a lease: nothing to renew.
        let mut client = StoreV2::open(store.root()).expect("client");
        assert!(!client.heartbeat_lease().expect("heartbeat without a lease"));
    }

    #[test]
    fn release_lease_hands_ownership_to_the_next_process() {
        let dir = TempDir::new().expect("tempdir");
        let mut first = store_in(&dir);
        first.acquire_lease().expect("first acquires");

        let mut second = StoreV2::open(first.root()).expect("second");
        assert!(matches!(
            second.acquire_lease().expect("blocked while held"),
            LeaseAcquisition::Client { .. }
        ));

        first.release_lease().expect("release");
        assert!(first.lease_status().expect("status").is_empty());
        match second.acquire_lease().expect("second takes over") {
            LeaseAcquisition::Owner { broke, .. } => {
                assert!(broke.is_empty(), "a clean release leaves nothing to break")
            }
            other => panic!("{other:?}"),
        }
    }

    /// Forge a foreign lease on `root` (identity + heartbeat files) with
    /// the given fields. Nothing flocks it — the shape a crashed (or
    /// never-started) owner leaves behind.
    fn forged_lease(root: &Path, owner_id: &str, pid: u32, heartbeat_ms: u64) {
        let identity = LeaseIdentity {
            pid,
            boot_id: boot_id(),
            started_utc: "2026-09-01T00:00:00.000Z".to_string(),
        };
        std::fs::write(
            root.join("leases").join(format!("{owner_id}.lease")),
            serde_json::to_string(&identity).expect("serialize"),
        )
        .expect("write forged lease");
        let heartbeat = LeaseHeartbeat {
            heartbeat_utc: "2026-09-01T00:00:00.000Z".to_string(),
            heartbeat_ms,
        };
        std::fs::write(
            root.join("leases").join(format!("{owner_id}.hb")),
            serde_json::to_string(&heartbeat).expect("serialize"),
        )
        .expect("write forged heartbeat");
    }

    /// Open and flock a lease file the way a live owner would; keep the
    /// returned handle open to keep the lock.
    fn hold_lease_flock(path: &Path) -> File {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .expect("open lease");
        assert_eq!(
            try_flock_exclusive(&file).expect("flock"),
            FlockEvidence::Free,
            "the test holds the lease like a live owner"
        );
        file
    }

    #[test]
    fn a_crash_orphaned_lease_is_stale_and_breakable() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();
        // A lease whose owner died: the OS released its flock when the
        // process did, so however alive the record claims to be, the
        // lease is dead weight a new process breaks on its way in.
        forged_lease(&root, "l_dead", std::process::id(), now_epoch_ms());

        match store.acquire_lease().expect("acquire") {
            LeaseAcquisition::Owner { broke, .. } => {
                assert_eq!(broke, vec!["l_dead".to_string()]);
            }
            other => panic!("the stale lease must not block, got {other:?}"),
        }
        assert_eq!(
            lease_files(&root)
                .into_iter()
                .filter(|name| name.ends_with(".lease"))
                .count(),
            1,
            "only the new owner's lease remains"
        );
        assert!(
            !root.join("leases").join("l_dead.hb").exists(),
            "the broken lease's heartbeat goes with it"
        );
    }

    #[test]
    fn a_torn_lease_record_from_a_dead_owner_is_breakable() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();
        // A crash mid-heartbeat left truncated JSON. The flock (free, the
        // owner is gone) decides; the unparseable record cannot veto it.
        std::fs::write(root.join("leases").join("l_torn.lease"), b"{\"pid\":12")
            .expect("write torn lease");

        let broken = store.break_stale_leases().expect("break");
        assert_eq!(broken, vec!["l_torn".to_string()]);
    }

    #[test]
    fn an_flocked_lease_is_never_broken_however_stale_its_record_reads() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();
        // A live owner that forgot to heartbeat: the held flock is OS
        // truth, and a timestamp must never outvote it.
        forged_lease(&root, "l_zombie", 999_999_999, 0);
        let held = hold_lease_flock(&root.join("leases").join("l_zombie.lease"));

        let broken = store.break_stale_leases().expect("break");
        assert!(broken.is_empty(), "a held lease is alive: {broken:?}");
        drop(held);

        // Once the holder is gone the same file is breakable.
        let broken = store.break_stale_leases().expect("break again");
        assert_eq!(broken, vec!["l_zombie".to_string()]);
    }

    #[test]
    fn the_stale_ladder_requires_proof_of_death() {
        // Pure ladder (the flock-less degradation order; on unix the flock
        // decides before this runs, on other hosts this is the whole
        // rule). Breaking a lease requires proof of death.
        let alive = |boot: Option<bool>, heartbeat: Option<bool>, pid: Option<bool>| {
            lease_alive_from(boot, heartbeat, pid)
        };
        // Boot mismatch: fatal even with a fresh heartbeat and a live pid
        // — after a reboot the recorded pid is not the owner.
        assert!(!alive(Some(false), Some(true), Some(true)));
        // Expired heartbeat: fatal (§4 stale rule).
        assert!(!alive(Some(true), Some(false), Some(true)));
        // Dead pid: fatal.
        assert!(!alive(Some(true), Some(true), Some(false)));
        // Every provable signal healthy: alive.
        assert!(alive(Some(true), Some(true), Some(true)));
        // Unknown signals are never proof of death.
        assert!(alive(None, None, None));
        assert!(alive(None, Some(true), None));
        assert!(alive(Some(true), None, None));
    }

    // ---- retention sweep (R21 never-delete-until-swept) ------------------

    #[test]
    fn sweep_removes_quarantined_journals_and_stamps_tombstones_swept() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(40, 0)).record.id;
        let quarantine = store.root().join("quarantine").join(format!("{id}.sj"));
        store.delete_capture(&id).expect("delete");
        assert!(
            quarantine.exists(),
            "never-delete-until-swept holds pre-sweep"
        );
        let bytes = std::fs::metadata(&quarantine).map(|m| m.len()).unwrap_or(0);
        assert!(bytes > 0);

        let report = store.sweep_retention().expect("sweep");
        assert_eq!(report.swept.len(), 1);
        assert_eq!(report.swept[0].id, id);
        assert_eq!(report.swept[0].kind, "capture");
        assert_eq!(report.swept[0].bytes, bytes);
        assert_eq!(report.swept_bytes, bytes);
        assert!(report.retained.is_empty());
        assert!(!quarantine.exists());
        assert!(store.get_capture(&id).expect("row").is_none());

        // Bookkeeping: the tombstone outlives the bytes and says swept.
        let retention: String = store
            .conn
            .query_row(
                "SELECT retention FROM tombstones WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .expect("tombstone row");
        assert_eq!(retention, "swept");

        // Idempotent: a second sweep has nothing left to do.
        let report = store.sweep_retention().expect("sweep again");
        assert!(report.swept.is_empty() && report.retained.is_empty());
    }

    #[test]
    fn sweep_never_stamps_a_directory_named_like_a_journal() {
        // A directory under quarantine/ named <id>.sj cannot be unlinked
        // as a file; stamping it swept would deaden that id forever —
        // reconcile's dead set would suppress recovery and adoption under
        // it — while the entry itself survives every later sweep. It must
        // be retained, unstamped.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let stray = store.root().join("quarantine").join("stray.sj");

        std::fs::create_dir_all(&stray).expect("stray directory");
        std::fs::write(stray.join("junk"), b"x").expect("content");

        let report = store.sweep_retention().expect("sweep");

        assert!(report.swept.is_empty());
        assert!(stray.exists(), "the directory itself is untouched");
        let retained_reason = report
            .retained
            .iter()
            .find(|(name, _)| name == "stray.sj")
            .map(|(_, reason)| reason.clone())
            .expect("retained entry");
        assert!(
            retained_reason.contains("not a regular file"),
            "{retained_reason}"
        );
        let stamped: Option<String> = store
            .conn
            .query_row(
                "SELECT retention FROM tombstones WHERE id = 'stray'",
                [],
                |row| row.get(0),
            )
            .ok();
        assert!(
            stamped.is_none(),
            "no tombstone may exist for the stray directory's id"
        );
    }

    #[test]
    fn sweep_also_empties_the_legacy_v1_deleted_tree() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let legacy = store.root().join("journals").join("deleted");
        std::fs::create_dir_all(&legacy).expect("create legacy tree");
        std::fs::write(legacy.join("j_old.sj"), b"stale v1 bytes").expect("write");
        std::fs::write(legacy.join("j_older.sj"), b"older v1 bytes").expect("write");

        let report = store.sweep_retention().expect("sweep");
        let swept_ids: Vec<&str> = report.swept.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(swept_ids, vec!["j_old", "j_older"], "sweep order is sorted");
        assert!(report.swept.iter().all(|f| f.kind == "journal"));
        assert!(!legacy.join("j_old.sj").exists());
        assert!(!legacy.join("j_older.sj").exists());

        // Each swept legacy journal got a tombstone: never resurrect,
        // even though no v2 capture row ever named it.
        let tombstoned: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM tombstones WHERE kind = 'journal'",
                [],
                |row| row.get(0),
            )
            .expect("count");
        assert_eq!(tombstoned, 2);
    }

    #[test]
    fn sweep_touches_nothing_but_the_tombstone_trees() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let live = committed_take(&mut store, &ramp(40, 0)).record.id;
        let deleted = committed_take(&mut store, &ramp(40, 1)).record.id;
        store.delete_capture(&deleted).expect("delete");

        // Reconcile (a read-path-adjacent recovery pass) never sweeps.
        store.reconcile().expect("reconcile");
        assert!(
            store
                .root()
                .join("quarantine")
                .join(format!("{deleted}.sj"))
                .exists(),
            "reconcile leaves quarantine for the explicit sweep"
        );

        let report = store.sweep_retention().expect("sweep");
        assert_eq!(
            report.swept.len(),
            1,
            "only the tombstoned journal: {report:?}"
        );
        assert_eq!(report.swept[0].id, deleted);
        // The live take's audio and row are untouched.
        assert!(store
            .root()
            .join("audio")
            .join(format!("{live}.sj"))
            .exists());
        let loaded = store.load_audio(&live).expect("live audio loads");
        assert_eq!(loaded.samples.len(), 40);
    }

    #[test]
    fn a_swept_tombstone_still_blocks_resurrection() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let id = committed_take(&mut store, &ramp(30, 0)).record.id;
        store.delete_capture(&id).expect("delete");
        store.sweep_retention().expect("sweep");

        // A journal with the swept id reappears under audio/ (restored
        // backup, copied disk): the tombstone still wins — completed
        // delete, no resurrection.
        let audio = store.root().join("audio").join(format!("{id}.sj"));
        std::fs::write(&audio, b"resurrected bytes").expect("write");
        let report = store.reconcile().expect("reconcile");
        assert!(report.completed_deletes.contains(&id), "{report:?}");
        assert!(!audio.exists());
        assert!(store.get_capture(&id).expect("row").is_none());
    }

    #[test]
    fn non_journal_entries_in_the_tombstone_trees_are_retained() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let quarantine = store.root().join("quarantine");
        std::fs::write(quarantine.join("notes.txt"), b"hand-dropped junk").expect("write");
        // A directory squatting on a `.sj` name cannot be unlinked as a
        // file — the sweep reports it, never blasts it.
        std::fs::create_dir(quarantine.join("c_dir.sj")).expect("create");

        let report = store.sweep_retention().expect("sweep");
        assert!(report.swept.is_empty());
        let retained: Vec<(String, String)> = report.retained.clone();
        assert_eq!(retained.len(), 2, "{retained:?}");
        assert!(quarantine.join("notes.txt").exists());
        assert!(quarantine.join("c_dir.sj").exists());
    }

    // ---- bounded listing ---------------------------------------------------

    #[test]
    fn listing_pages_are_clamped_at_the_store_layer() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        // Rows straight into SQLite (the journal files are irrelevant to
        // the page bound; 550 > LIST_PAGE_MAX).
        for i in 0..550 {
            let record = CaptureRecord {
                id: format!("c_{i:04}"),
                created_utc: format!("2026-09-20T10:{:02}:{:02}.000Z", i / 60, i % 60),
                tz: "UTC".to_string(),
                device: String::new(),
                actual_rate: 16_000,
                policy: "default".to_string(),
                frame_count: 10,
                ack_sample_index: 10,
                journal_hash: "00".to_string(),
                status: CaptureStatus::Complete,
                retention_class: "standard".to_string(),
                extra_json: None,
                secure_field: false,
            };
            store.commit_capture(&record).expect("insert");
        }

        // However large a limit a caller asks for, one query never
        // materializes more than LIST_PAGE_MAX rows.
        let page = store.list_records(0, 1_000_000).expect("list");
        assert_eq!(page.records.len(), LIST_PAGE_MAX);
        assert_eq!(page.total, 550);
        // Paging continues past the clamp.
        let page = store.list_records(LIST_PAGE_MAX, 1_000_000).expect("list");
        assert_eq!(page.records.len(), 50);
    }

    // ---- review round 1: sweep crash window ------------------------------

    #[test]
    fn a_crash_between_stamp_and_unlink_still_blocks_resurrection() {
        // The sweep stamps the tombstone BEFORE unlinking the bytes. This
        // simulates the crash between the two steps for the population
        // that has no row of its own (the legacy v1 tree): row stamped,
        // file still present — the id must already be dead to reconcile,
        // and the next sweep must complete the unlink.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();
        let legacy = root.join("journals").join("deleted");
        std::fs::create_dir_all(&legacy).expect("create legacy tree");
        std::fs::write(legacy.join("j_old.sj"), b"stale v1 bytes").expect("write");

        // The mid-sweep state: exactly what sweep_tree's step 1 leaves.
        store
            .conn
            .execute(
                "INSERT INTO tombstones(id, kind, deleted_utc, retention)
                 VALUES ('j_old', 'journal', ?1, 'swept')",
                params![now_iso()],
            )
            .expect("stamp");

        // A journal with the swept id reappears under audio/ (restored
        // backup, copied disk): the stamped row already wins — completed
        // delete, re-quarantined, no resurrection.
        std::fs::write(root.join("audio").join("j_old.sj"), b"resurrected bytes")
            .expect("resurrect");
        let report = store.reconcile().expect("reconcile");
        assert!(
            report.completed_deletes.contains(&"j_old".to_string()),
            "{report:?}"
        );
        assert!(!root.join("audio").join("j_old.sj").exists());
        assert!(
            root.join("quarantine").join("j_old.sj").exists(),
            "the resurrected bytes are re-quarantined awaiting the sweep"
        );

        // The next sweep completes the interrupted one: bytes unlinked,
        // row already (and still) swept.
        let report = store.sweep_retention().expect("sweep");
        assert_eq!(report.swept.len(), 2, "legacy file + re-quarantined bytes");
        assert!(!legacy.join("j_old.sj").exists());
        assert!(!root.join("quarantine").join("j_old.sj").exists());
        let retention: String = store
            .conn
            .query_row(
                "SELECT retention FROM tombstones WHERE id = 'j_old'",
                [],
                |row| row.get(0),
            )
            .expect("row");
        assert_eq!(retention, "swept");
    }

    // ---- review round 1: lease races and degradation ----------------------

    #[test]
    fn acquisition_serializes_on_the_leases_sentinel() {
        // While one process holds the sentinel across its
        // probe-and-publish section, a concurrent acquire cannot run its
        // own — it blocks instead of interleaving into the check-then-act
        // window that let two processes both become owners.
        let dir = TempDir::new().expect("tempdir");
        let store = store_in(&dir);
        let root = store.root().to_path_buf();
        let sentinel = LeaseSentinel::acquire(&root.join("leases")).expect("sentinel");

        let acquirer_root = root.clone();
        let acquirer = std::thread::spawn(move || {
            let mut other = StoreV2::open(&acquirer_root).expect("open");
            match other.acquire_lease().expect("acquire") {
                LeaseAcquisition::Owner { owner_id, .. } => owner_id,
                other => panic!("expected Owner once the sentinel frees, got {other:?}"),
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(
            !acquirer.is_finished(),
            "acquire must block while the sentinel is held"
        );
        drop(sentinel);
        let owner_id = acquirer.join().expect("acquisition thread");
        assert!(root
            .join("leases")
            .join(format!("{owner_id}.lease"))
            .exists());
    }

    #[test]
    fn a_wedged_sentinel_holder_fails_the_acquisition_after_a_bounded_wait() {
        // The sentinel is coordination, not liveness: a holder stuck
        // across its critical section must not block another acquisition
        // (and therefore startup) forever — past the bound the
        // acquisition fails with a distinct error instead of hanging.
        // Probed with a short injected deadline while the sentinel is
        // genuinely held.
        let dir = TempDir::new().expect("tempdir");
        let store = store_in(&dir);
        let root = store.root().to_path_buf();
        let _held = LeaseSentinel::acquire(&root.join("leases")).expect("sentinel");

        let taken = LeaseSentinel::acquire_with_timeout(
            &root.join("leases"),
            std::time::Duration::from_millis(60),
        );

        let err = match taken {
            Ok(_) => panic!("must not block past the deadline"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("wedged"), "{err}");
    }

    #[test]
    fn the_post_publish_tie_break_yields_only_to_a_smaller_owner_id() {
        // The sentinel serializes acquirers on unix; where flock cannot,
        // the deterministic tie-break closes the rest: both concurrent
        // publishers compute the same order (yield to the smallest live
        // owner id), so exactly one stays owner. Probed at the mechanism
        // level: two live flocked leases, various vantage points.
        let dir = TempDir::new().expect("tempdir");
        let store = store_in(&dir);
        let root = store.root().to_path_buf();
        forged_lease(&root, "l_aaa", std::process::id(), now_epoch_ms());
        forged_lease(&root, "l_zzz", std::process::id(), now_epoch_ms());
        let small = hold_lease_flock(&root.join("leases").join("l_aaa.lease"));
        let large = hold_lease_flock(&root.join("leases").join("l_zzz.lease"));

        // A late publisher between the two yields to the smaller live id.
        let ownership = store
            .younger_live_foreign_lease("l_mmm")
            .expect("tie-break");
        assert_eq!(
            ownership.live.expect("yields to someone").owner_id,
            "l_aaa",
            "the smallest live owner wins"
        );
        // The smallest live owner itself stays owner (its only peer is
        // larger), and the largest yields to the smallest.
        let ownership = store
            .younger_live_foreign_lease("l_aaa")
            .expect("tie-break from the smallest");
        assert!(ownership.live.is_none(), "the smallest stays owner");
        let ownership = store
            .younger_live_foreign_lease("l_zzz")
            .expect("tie-break from the largest");
        assert_eq!(ownership.live.expect("yields").owner_id, "l_aaa");

        drop(small);
        drop(large);
    }

    #[test]
    fn a_young_acquisition_temp_is_never_swept_but_an_old_one_is() {
        // The grace period closes the sweep-vs-acquirer race: a temp
        // younger than LEASE_TEMP_GRACE may belong to a live writer
        // between create and its publish rename, so the sweep must not
        // touch it — however flock-free it looks. An older temp with no
        // holder is a crashed writer's garbage and goes.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();
        let leases = root.join("leases");

        let young = unique_lease_temp(&leases, "l_young");
        std::fs::write(&young, b"{}").expect("write young temp");
        let old = unique_lease_temp(&leases, "l_old");
        std::fs::write(&old, b"{}").expect("write old temp");
        // Backdate the old temp past the grace period.
        let file = File::options().write(true).open(&old).expect("open old");
        file.set_times(std::fs::FileTimes::new().set_modified(
            std::time::SystemTime::now() - LEASE_TEMP_GRACE - std::time::Duration::from_secs(5),
        ))
        .expect("backdate");

        let broken = store.break_stale_leases().expect("break");
        assert!(broken.is_empty(), "no published leases: {broken:?}");
        assert!(young.exists(), "a young temp is untouchable");
        assert!(!old.exists(), "an old orphan temp is swept");

        // And a temp whose flock is held is never swept, however old.
        let held = unique_lease_temp(&leases, "l_held");
        std::fs::write(&held, b"{}").expect("write held temp");
        let holder = hold_lease_flock(&held);
        let file = File::options().write(true).open(&held).expect("open held");
        file.set_times(std::fs::FileTimes::new().set_modified(
            std::time::SystemTime::now() - LEASE_TEMP_GRACE - std::time::Duration::from_secs(5),
        ))
        .expect("backdate");
        drop(file);
        store.break_stale_leases().expect("break again");
        assert!(held.exists(), "a held temp belongs to a live writer");
        drop(holder);
    }

    #[test]
    fn an_unreadable_lease_defers_recovery_and_is_reported() {
        // A lease file that cannot be probed at all still reads as an
        // owner (never break what cannot be proven dead), so recovery
        // defers — but the state is surfaced, because a single corrupt
        // lease file must not silently disable crash recovery for the
        // whole root.
        let dir = TempDir::new().expect("tempdir");
        let mut owner = store_in(&dir);
        let mut take = owner
            .begin_take(TakeMeta::for_device("test-device"))
            .expect("begin take");
        take.append_frames(&ramp(20, 0)).expect("append");
        take.write_boundary().expect("boundary");
        let take_id = take.id().to_string();
        drop(take);
        drop(owner);

        let mut store = StoreV2::open(dir.path().join("v2")).expect("reopen");
        // A directory squatting on a lease name: present, unopenable.
        std::fs::create_dir(store.root().join("leases").join("l_weird.lease"))
            .expect("create unreadable lease");

        let report = store.reconcile().expect("reconcile");
        assert_eq!(
            report.deferred_to_live_owner,
            vec![take_id.clone()],
            "{report:?}"
        );
        assert!(
            store_root_has_staging(&store, &take_id),
            "the in-flight take is left for the unanswerable owner"
        );
        assert_eq!(report.unreadable_leases.len(), 1, "{report:?}");
        assert_eq!(report.unreadable_leases[0].0, "l_weird");
        assert!(report.has_findings(), "the disabled recovery is visible");
        assert!(report.summary().contains("lease"), "{}", report.summary());
    }

    #[test]
    fn a_torn_or_missing_heartbeat_never_creates_an_unbreakable_lease() {
        // The identity record is immutable and the heartbeat is replaced
        // atomically, so a crash mid-heartbeat cannot leave a lease whose
        // record cannot be parsed: a garbage or absent `.hb` reads as an
        // ancient heartbeat — stale — and once the flock is gone the
        // lease breaks instead of zombie-ing forever.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();

        // Garbage heartbeat, identity intact, owner alive (flock held).
        forged_lease(&root, "l_tornhb", std::process::id(), now_epoch_ms());
        std::fs::write(root.join("leases").join("l_tornhb.hb"), b"{\"heartbeatM")
            .expect("write torn heartbeat");
        let held = hold_lease_flock(&root.join("leases").join("l_tornhb.lease"));
        let broken = store.break_stale_leases().expect("break");
        assert!(broken.is_empty(), "the flock is OS truth while held");

        // Owner gone: the unparseable heartbeat reads as ancient — stale
        // — and the lease breaks rather than surviving forever.
        drop(held);
        let broken = store.break_stale_leases().expect("break");
        assert_eq!(broken, vec!["l_tornhb".to_string()]);

        // A missing heartbeat file (crash between publish and the first
        // renewal) is the same staleness answer.
        forged_lease(&root, "l_nohb", std::process::id(), now_epoch_ms());
        std::fs::remove_file(root.join("leases").join("l_nohb.hb")).expect("drop the heartbeat");
        let held = hold_lease_flock(&root.join("leases").join("l_nohb.lease"));
        assert!(
            store
                .lease_status()
                .expect("status")
                .iter()
                .any(|lease| lease.owner_id == "l_nohb" && lease.alive),
            "alive while the flock is held"
        );
        drop(held);
        let broken = store.break_stale_leases().expect("break");
        assert_eq!(broken, vec!["l_nohb".to_string()]);
    }

    // ---- review round 1: staging rollback ----------------------------------

    #[test]
    fn discard_staging_removes_only_the_named_in_flight_journal() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let mut take = store
            .begin_take(TakeMeta::for_device("test-device"))
            .expect("begin take");
        take.append_frames(&ramp(10, 0)).expect("append");
        let id = take.id().to_string();
        let mut other = store
            .begin_take(TakeMeta::for_device("test-device"))
            .expect("begin other take");
        other.append_frames(&ramp(10, 1)).expect("append");
        let other_id = other.id().to_string();
        drop(other);

        store.discard_staging(&id).expect("discard");
        assert!(
            !store_root_has_staging(&store, &id),
            "the failed take's staging is gone"
        );
        assert!(
            store_root_has_staging(&store, &other_id),
            "an unrelated in-flight take is untouched"
        );
        // The Ok(bool) contract: this call removed it...
        assert!(
            store
                .discard_staging(&other_id)
                .expect("discard other"),
            "a present journal reports removed-by-this-call"
        );
        // ...and a second discard of the now-missing id reports the
        // idempotent already-gone shape, not another removal — the two
        // sides rollback reporting distinguishes.
        assert!(
            !store.discard_staging(&id).expect("discard again"),
            "an already-gone journal reports already-gone, not removed"
        );
        drop(take);

        // A committed row is never touched: its audio lives in `audio/`,
        // and a late discard of its id is a no-op.
        let committed = committed_take(&mut store, &ramp(12, 2));
        store
            .discard_staging(&committed.record.id)
            .expect("discard after commit");
        assert!(
            store
                .root()
                .join("audio")
                .join(format!("{}.sj", committed.record.id))
                .exists(),
            "committed audio survives a late discard"
        );
        assert!(
            store
                .get_capture(&committed.record.id)
                .expect("row")
                .is_some(),
            "the committed row survives a late discard"
        );
    }

    // ---- issue #259: PR #257 review round 2 follow-ups ---------------------

    #[test]
    fn discard_staging_surfaces_a_real_removal_failure() {
        // The rollback contract must be honest: a missing file is the
        // idempotent no-op, but a removal that genuinely fails (here: a
        // directory squatting on the staging name) is an error the caller
        // can act on — otherwise the partial journal that reconcile would
        // salvage as a duplicate survives, silently.
        let dir = TempDir::new().expect("tempdir");
        let store = store_in(&dir);
        let squat = store.root().join(STAGING_DIR).join("c_squat.sj");
        std::fs::create_dir(&squat).expect("directory on the staging name");

        assert!(
            store.discard_staging("c_squat").is_err(),
            "a real removal failure must surface, not read as rolled back"
        );
        assert!(squat.exists(), "the failed removal left it in place");
        // The idempotent no-op stays Ok — and reports already-gone.
        assert!(
            !store
                .discard_staging("c_missing")
                .expect("a missing staging file is the no-op"),
            "a missing file reports already-gone, not removed"
        );
    }

    #[test]
    fn an_unanswerable_lease_blocks_acquisition_instead_of_dual_ownership() {
        // A lease that can be neither probed nor broken is an unanswerable
        // owner: reconcile would defer to it forever, so acquisition must
        // not publish a second owner alongside it — this process would
        // keep writing while its own reconciles defer, the
        // recovery-disabling state self-inflicted on the root's owner.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        // A directory squatting on a lease name: present, unprobeable.
        std::fs::create_dir(store.root().join(LEASES_DIR).join("l_wedge.lease"))
            .expect("wedge lease");

        match store.acquire_lease().expect("acquisition answers") {
            LeaseAcquisition::UnanswerableLeases { unreadable } => {
                assert_eq!(unreadable.len(), 1, "{unreadable:?}");
                assert_eq!(unreadable[0].0, "l_wedge");
            }
            other => panic!("no ownership on top of an unanswerable lease: {other:?}"),
        }
        let mut remaining: Vec<String> = std::fs::read_dir(store.root().join(LEASES_DIR))
            .expect("leases dir")
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.ends_with(".lease"))
            .collect();
        remaining.sort();
        assert_eq!(
            remaining,
            vec!["l_wedge.lease".to_string()],
            "no second owner's lease file was published"
        );

        // The wedge repaired: acquisition proceeds normally.
        std::fs::remove_dir(store.root().join(LEASES_DIR).join("l_wedge.lease"))
            .expect("remove the wedge");
        assert!(matches!(
            store.acquire_lease().expect("acquire after repair"),
            LeaseAcquisition::Owner { .. }
        ));
    }

    #[test]
    fn break_stale_leases_takes_the_acquisition_sentinel() {
        // The public breaking path is a mutating section of its own: it
        // must serialize against a concurrent acquirer's probe-and-publish
        // exactly like acquire_lease. Pinned deterministically — no
        // wall-clock windows: while another holder has the sentinel, a
        // breaking call with a short injected bound fails outright; once
        // the holder drops it, the same call breaks the stale lease.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let root = store.root().to_path_buf();
        forged_lease(&root, "l_dead", std::process::id(), now_epoch_ms());

        let held = LeaseSentinel::acquire(&root.join(LEASES_DIR)).expect("sentinel");
        let err = store
            .break_stale_leases_within(std::time::Duration::from_millis(60))
            .expect_err("breaking must not run under a held sentinel");
        assert!(
            err.to_string().contains("wedged"),
            "the refusal names the wedged sentinel: {err}"
        );
        assert!(
            root.join(LEASES_DIR).join("l_dead.lease").exists(),
            "the stale lease is untouched while the sentinel is held"
        );
        drop(held);

        let broken = store.break_stale_leases().expect("break once free");
        assert_eq!(broken, vec!["l_dead".to_string()]);
    }

    #[test]
    fn a_failed_identity_publish_leaves_no_temporaries_behind() {
        // The publish retry loop must clean its scratch file on EVERY
        // failure arm, not only the flocked-temp arm: a persistent rename
        // failure used to accumulate one orphaned temp per attempt until
        // the grace-period sweep collected them — the failure's footprint
        // is now zero.
        let dir = TempDir::new().expect("tempdir");
        let store = store_in(&dir);
        let leases = store.root().join(LEASES_DIR);
        // A directory where the identity must land: rename(file, dir)
        // fails on every attempt.
        std::fs::create_dir(leases.join("l_stuck.lease")).expect("destination wedge");

        let published = store.publish_identity(&leases, "l_stuck", "2026-09-22T00:00:00.000Z");

        assert!(published.is_err(), "the wedged destination must fail");
        assert!(
            lease_temps_in(&leases).is_empty(),
            "no scratch temp survives the failed attempts"
        );
    }

    #[test]
    fn a_pre_1970_clock_saturates_to_one_not_the_missing_heartbeat_sentinel() {
        // heartbeat_ms = 0 means "no heartbeat parsed" (ancient, stale);
        // a pre-epoch clock must not write its fresh heartbeat as that
        // sentinel — it saturates to 1 ms past the epoch, which still
        // reads fresh against a peer reading the same skewed clock. An
        // exactly-epoch clock clamps to the same 1 for the same reason.
        let skewed = std::time::SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(86_400);
        assert_eq!(epoch_ms(skewed), 1);
        assert_eq!(epoch_ms(std::time::SystemTime::UNIX_EPOCH), 1);
    }

    #[test]
    fn audio_journal_exists_answers_for_rowless_promoted_audio() {
        // The promoted-but-uncommitted shape (a failure between the
        // promoting rename and the row commit): no row, but the bytes sit
        // in audio/. audio_journal_path cannot answer — it resolves
        // through the row — this probe is what rollback callers need to
        // tell which side of the rename the bytes are on.
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let mut take = store
            .begin_take(TakeMeta::for_device("test-device"))
            .expect("begin");
        take.append_frames(&ramp(10, 0)).expect("append");
        let id = take.id().to_string();
        drop(take.finalize().expect("finalize"));
        assert!(
            !store.audio_journal_exists(&id).expect("probe in staging"),
            "still in staging: nothing promoted"
        );
        store.promote_from_staging(&id).expect("promote, no commit");
        assert!(
            store.audio_journal_exists(&id).expect("probe promoted"),
            "the bytes are in audio/ though no row exists"
        );
        assert!(
            store.get_capture(&id).expect("row read").is_none(),
            "no row — exactly the shape the probe exists for"
        );
        assert!(
            !store.audio_journal_exists("c_never").expect("probe unknown"),
            "an unknown id has no audio"
        );
    }

    // -----------------------------------------------------------------
    // Documents / revisions (§4 tables; the I5 documents-machine
    // persistence, issue #220).
    // -----------------------------------------------------------------

    fn sample_revision(rev_id: &str, doc_id: &str, base: u64, text: &str) -> RevisionRow {
        RevisionRow {
            rev_id: rev_id.to_string(),
            doc_id: doc_id.to_string(),
            base_revision: Some(base),
            sources_json: Some(
                serde_json::json!({
                    "attempts": ["att-1"],
                    "instructionTemplateId": "tpl-none",
                })
                .to_string(),
            ),
            text: text.to_string(),
            status: "candidate".to_string(),
            provenance: Some("recognition".to_string()),
            disposition: Some("committed".to_string()),
        }
    }

    #[test]
    fn documents_round_trip_and_survive_reopen() {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path().join("v2");
        {
            let store = StoreV2::open(&root).expect("open");
            store
                .upsert_document("notes", "notes", 1, 1)
                .expect("upsert document");
            store
                .store_document_revision(&sample_revision("rev-1", "notes", 0, "First head."))
                .expect("store revision");
            store
                .store_document_revision(&sample_revision(
                    "rev-2",
                    "notes",
                    0,
                    "Preserved candidate.",
                ))
                .expect("store preserved revision");
            // The CAS advanced the head: the same upsert is the update.
            store
                .upsert_document("notes", "notes", 2, 3)
                .expect("update document");
            store.bump_document_turn("notes", 3).expect("bump turn");
        }
        // A fresh connection (the documents machine hydrating at boot).
        let reopened = StoreV2::open(&root).expect("reopen");
        let doc = reopened
            .get_document("notes")
            .expect("get")
            .expect("the document persisted");
        assert_eq!(doc.name, "notes");
        assert_eq!(doc.head_revision, 2);
        assert_eq!(doc.turn_seq, 3);
        assert_eq!(doc.revisions.len(), 2);
        assert_eq!(doc.revisions[0].rev_id, "rev-1");
        assert_eq!(doc.revisions[1].rev_id, "rev-2");
        // The provenance JSON survived verbatim.
        assert_eq!(
            doc.revisions[0].sources_json.as_deref(),
            sample_revision("rev-1", "notes", 0, "").sources_json.as_deref()
        );
        assert_eq!(doc.revisions[0].text, "First head.");
        // An unknown document is None, not an error.
        assert!(reopened.get_document("nope").expect("get unknown").is_none());
    }

    #[test]
    fn a_revision_for_an_unknown_document_is_refused_by_the_foreign_key() {
        let dir = TempDir::new().expect("tempdir");
        let store = StoreV2::open(dir.path().join("v2")).expect("open");
        let err = store
            .store_document_revision(&sample_revision("rev-x", "ghost", 0, "orphaned"))
            .expect_err("the foreign key must refuse an orphaned revision");
        assert!(
            err.to_string().contains("FOREIGN KEY"),
            "the refusal is the SQLite foreign-key error: {err}"
        );
        // And nothing landed.
        assert!(store.get_document("ghost").expect("get").is_none());
    }

    #[test]
    fn bump_document_turn_creates_and_updates_without_touching_the_head() {
        let dir = TempDir::new().expect("tempdir");
        let store = StoreV2::open(dir.path().join("v2")).expect("open");
        // A turn appended to a document no updateHead ever wrote: the
        // implicit-document shape creates the row at head 0.
        store.bump_document_turn("scratch", 1).expect("implicit doc");
        let doc = store.get_document("scratch").expect("get").expect("row");
        assert_eq!(doc.head_revision, 0);
        assert_eq!(doc.turn_seq, 1);
        assert_eq!(doc.name, "scratch");
        // An existing document's head survives the bump.
        store
            .upsert_document("scratch", "scratch", 4, 1)
            .expect("advance head");
        store.bump_document_turn("scratch", 2).expect("bump");
        let doc = store.get_document("scratch").expect("get").expect("row");
        assert_eq!(doc.head_revision, 4);
        assert_eq!(doc.turn_seq, 2);
    }

    #[test]
    fn the_durable_head_never_moves_backwards() {
        let dir = TempDir::new().expect("tempdir");
        let store = StoreV2::open(dir.path().join("v2")).expect("open");
        assert!(store.upsert_document("doc", "doc", 2, 0).expect("head 2"));
        assert!(!store.upsert_document("doc", "doc", 1, 0).expect("stale head is a no-op"));
        assert_eq!(store.get_document("doc").expect("get").expect("row").head_revision, 2);
        // A stale commit lands neither the head nor its revision row.
        store
            .commit_document_head("doc", 1, 0, &sample_revision("rev-stale", "doc", 0, "stale"))
            .expect_err("a stale head commit is refused");
        let doc = store.get_document("doc").expect("get").expect("row");
        assert_eq!(doc.head_revision, 2);
        assert!(doc.revisions.is_empty(), "the refused commit rolled back");
    }

    #[test]
    fn a_revision_is_never_reparented_to_another_document() {
        let dir = TempDir::new().expect("tempdir");
        let store = StoreV2::open(dir.path().join("v2")).expect("open");
        store.upsert_document("a", "a", 1, 0).expect("doc a");
        store.upsert_document("b", "b", 0, 0).expect("doc b");
        store
            .store_document_revision(&sample_revision("r1", "a", 0, "A's text"))
            .expect("a's revision");
        store
            .commit_document_head("b", 1, 0, &sample_revision("r1", "b", 0, "B's text"))
            .expect_err("a colliding rev_id under another document is refused");
        let a = store.get_document("a").expect("get").expect("row");
        assert_eq!(a.revisions[0].text, "A's text");
        let b = store.get_document("b").expect("get").expect("row");
        assert_eq!(b.head_revision, 0, "the refused commit rolled the head back");
    }

    #[test]
    fn document_ids_are_validated_before_any_sql_runs() {
        let dir = TempDir::new().expect("tempdir");
        let store = StoreV2::open(dir.path().join("v2")).expect("open");
        for bad in ["", &"x".repeat(1025)] {
            assert!(store.upsert_document(bad, "n", 0, 0).is_err());
            assert!(store.bump_document_turn(bad, 1).is_err());
            assert!(store.get_document(bad).is_err());
            let mut revision = sample_revision("rev-1", "notes", 0, "t");
            revision.rev_id = bad.to_string();
            assert!(store.store_document_revision(&revision).is_err());
        }
    }

    #[test]
    fn a_document_is_deleted_with_its_revisions() {
        let dir = TempDir::new().expect("tempdir");
        let store = StoreV2::open(dir.path().join("v2")).expect("open");
        store
            .commit_document_head("take", 1, 0, &sample_revision("take#h1", "take", 0, "raw"))
            .expect("head");
        assert!(store.delete_document("take").expect("delete"));
        assert!(store.get_document("take").expect("get").is_none());
        assert!(!store.delete_document("take").expect("second delete"));
        // The cascade took the revision row: its id is free again.
        store
            .commit_document_head("other", 1, 0, &sample_revision("take#h1", "other", 0, "x"))
            .expect("the rev id no longer belongs to the deleted document");
    }

    #[test]
    fn insight_events_are_idempotent_and_die_with_their_capture() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let take = committed_take(&mut store, &ramp(1600, 0));
        let id = take.record.id.clone();
        let payload = r#"{"type":"processing_recorded","job_id":"req-1"}"#;
        store
            .record_insight_event("proc-req-1", &id, "processing_recorded", "2026-09-24T10:00:00Z", payload)
            .expect("record");
        store
            .record_insight_event("proc-req-1", &id, "processing_recorded", "2026-09-24T10:00:00Z", payload)
            .expect("a byte-identical replay is a no-op");
        store
            .record_insight_event("proc-req-1", &id, "processing_recorded", "2026-09-24T10:00:00Z", "{}")
            .expect_err("a different payload under a known id is a conflict");
        assert_eq!(
            store.insight_events_for(&id).expect("events"),
            vec![("processing_recorded".to_string(), payload.to_string())]
        );
        store.delete_capture(&id).expect("delete");
        assert!(store.insight_events_for(&id).expect("events").is_empty());
        assert!(matches!(
            store.record_insight_event("proc-req-2", &id, "processing_recorded", "2026-09-24T10:00:00Z", payload),
            Err(StoreV2Error::NotFound(_))
        ), "a deleted take's event is NotFound, not a constraint error");
    }

    #[test]
    fn a_version_2_database_gains_the_insight_table() {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path().join("v2");
        {
            let store = StoreV2::open(&root).expect("open");
            store
                .conn
                .execute_batch(
                    "DROP TABLE insight_events;
                     UPDATE meta SET value = '2' WHERE key = 'schema_version';",
                )
                .expect("downgrade to the v2 layout");
        }
        let mut store = StoreV2::open(&root).expect("reopen upgrades");
        assert_eq!(store.schema_version().expect("version"), SCHEMA_VERSION);
        let take = committed_take(&mut store, &ramp(160, 0));
        store
            .record_insight_event("e1", &take.record.id, "processing_recorded", "2026-09-24T10:00:00Z", "{}")
            .expect("the table exists after the upgrade");
    }

    // ---- correction records ------------------------------------------

    /// A committed take with a final transcript attempt, returning
    /// `(capture id, attempt id)`.
    fn transcribed_take(store: &mut StoreV2, meta: TakeMeta) -> (String, String) {
        let mut take = store.begin_take(meta).expect("begin");
        take.append_frames(&ramp(160, 0)).expect("append");
        let id = take.finish(store).expect("finish").record.id;
        store
            .begin_recognition(&id, "starling:parakeet", None)
            .expect("begin");
        store
            .finish_recognition(
                &id,
                RecognitionOutcome::Completed {
                    text: "um so hello there",
                    extra_json: None,
                },
            )
            .expect("finish");
        let attempt = store
            .attempts_for(&id)
            .expect("attempts")
            .into_iter()
            .find(AttemptRecord::is_final_transcript)
            .expect("a final attempt")
            .id;
        (id, attempt)
    }

    fn ordinary_take(store: &mut StoreV2) -> (String, String) {
        transcribed_take(store, TakeMeta::for_device("test-device"))
    }

    /// A correction record for `capture`/`attempt` with full provenance.
    fn correction(capture: &str, attempt: &str, request: &str) -> CorrectionRecord {
        CorrectionRecord {
            capture_id: capture.to_string(),
            request_id: request.to_string(),
            raw_attempt_id: attempt.to_string(),
            raw_text: "um so hello there".to_string(),
            processed_text: "So, hello there.".to_string(),
            final_text: Some("So, hello there.".to_string()),
            decision: CorrectionDecision::Accepted,
            decision_utc: "2026-09-24T10:00:00Z".to_string(),
            mode_id: Some("clean-local".to_string()),
            mode_version: Some(3),
            provider_id: Some("local-s1".to_string()),
            provider_kind: Some("s1".to_string()),
            provider_model: Some("s1-mini".to_string()),
            locality: Some("local".to_string()),
            transform_kinds: Some(r#"["clean"]"#.to_string()),
            language: Some("en".to_string()),
            asr_backend: None,
            asr_model_hash: None,
            timings_json: Some(
                r#"{"queued_ms":0.0,"processing_ms":900.0,"stop_to_result_ms":1234.5}"#.to_string(),
            ),
            settings_strength: None,
            extra_json: Some(r#"{"label":"S1-mini · this computer"}"#.to_string()),
        }
    }

    #[test]
    fn correction_records_round_trip_upsert_and_cascade() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let (id, attempt) = ordinary_take(&mut store);

        let record = correction(&id, &attempt, "req-1");
        assert!(store.upsert_correction_record(&record).expect("upsert"));
        let stored = store.correction_records_for(&id).expect("read");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].decision, CorrectionDecision::Accepted);
        assert_eq!(stored[0].raw_text, "um so hello there");
        assert_eq!(stored[0].processed_text, "So, hello there.");
        assert_eq!(stored[0].final_text.as_deref(), Some("So, hello there."));
        assert_eq!(stored[0].mode_id.as_deref(), Some("clean-local"));
        assert_eq!(stored[0].mode_version, Some(3));
        assert_eq!(stored[0].provider_model.as_deref(), Some("s1-mini"));
        // ASR provenance was filled from the attempt row, not the caller.
        assert_eq!(stored[0].asr_backend.as_deref(), Some("starling:parakeet"));
        assert_eq!(stored[0].asr_model_hash, None);
        assert_eq!(
            stored[0].settings_strength, None,
            "no strength setting exists yet"
        );

        // A revised decision upserts: one row, latest decision wins, the
        // request-pinned columns stay as first written.
        let mut revised = correction(&id, &attempt, "req-1");
        revised.decision = CorrectionDecision::Reverted;
        revised.decision_utc = "2026-09-24T10:02:00Z".to_string();
        revised.final_text = Some("um so hello there".to_string());
        revised.raw_text = "changed raw".to_string();
        revised.processed_text = "Changed.".to_string();
        revised.mode_id = Some("other-mode".to_string());
        revised.provider_model = Some("other-model".to_string());
        assert!(store.upsert_correction_record(&revised).expect("upsert"));
        let stored = store.correction_records_for(&id).expect("read");
        assert_eq!(
            stored.len(),
            1,
            "a revised decision is an update, not a second row"
        );
        assert_eq!(stored[0].decision, CorrectionDecision::Reverted);
        assert_eq!(stored[0].decision_utc, "2026-09-24T10:02:00Z");
        assert_eq!(stored[0].final_text.as_deref(), Some("um so hello there"));
        assert_eq!(stored[0].raw_text, "um so hello there");
        assert_eq!(stored[0].processed_text, "So, hello there.");
        assert_eq!(stored[0].mode_id.as_deref(), Some("clean-local"));
        assert_eq!(stored[0].provider_model.as_deref(), Some("s1-mini"));

        // A revision without the proposal in hand moves only the decision.
        assert!(
            store
                .revise_correction_record(&id, "req-1", CorrectionDecision::Edited, "t", "Edited.")
                .expect("revise")
        );
        let stored = store.correction_records_for(&id).expect("read");
        assert_eq!(stored[0].decision, CorrectionDecision::Edited);
        assert_eq!(stored[0].final_text.as_deref(), Some("Edited."));
        assert_eq!(stored[0].processed_text, "So, hello there.");
        assert!(
            !store
                .revise_correction_record(&id, "req-none", CorrectionDecision::Edited, "t", "x")
                .expect("revise"),
            "nothing to revise"
        );

        // A second proposal of the same take is its own row.
        assert!(
            store
                .upsert_correction_record(&correction(&id, &attempt, "req-2"))
                .expect("upsert")
        );
        assert_eq!(store.correction_records_for(&id).expect("read").len(), 2);

        // Deleting the take removes its correction records (cascade) and
        // a late decision for it lands nowhere.
        store.delete_capture(&id).expect("delete");
        assert!(store.correction_records_for(&id).expect("read").is_empty());
        assert!(matches!(
            store.upsert_correction_record(&revised),
            Err(StoreV2Error::NotFound(_))
        ));
    }

    #[test]
    fn secure_field_captures_never_record_corrections() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let mut meta = TakeMeta::for_device("test-device");
        meta.secure_field = true;
        let (secure_id, attempt) = transcribed_take(&mut store, meta);
        assert!(
            store
                .get_capture(&secure_id)
                .expect("get")
                .expect("present")
                .secure_field
        );

        assert!(
            !store
                .upsert_correction_record(&correction(&secure_id, &attempt, "req-1"))
                .expect("excluded, not failed")
        );
        assert!(
            store
                .correction_records_for(&secure_id)
                .expect("read")
                .is_empty()
        );

        let (normal_id, normal_attempt) = ordinary_take(&mut store);
        assert!(
            store
                .upsert_correction_record(&correction(&normal_id, &normal_attempt, "req-1"))
                .expect("record")
        );
    }

    #[test]
    fn a_version_3_database_gains_correction_records_and_the_secure_flag() {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path().join("v2");
        let (id, attempt) = {
            let mut store = store_in(&dir);
            let pair = ordinary_take(&mut store);
            drop(store);
            // Rewind to the v3 shape: no correction_records table and a
            // captures table without secure_field (rebuilt, because
            // SQLite cannot drop columns portably). Foreign keys stay off
            // during the surgery, like any table rebuild.
            let conn = Connection::open(root.join(DB_FILE)).expect("open");
            conn.execute_batch(
                "PRAGMA foreign_keys=OFF;
                 DROP TABLE correction_records;
                 CREATE TABLE captures_v3 (
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
                    extra_json       TEXT
                 );
                 INSERT INTO captures_v3
                    SELECT id, created_utc, tz, device, actual_rate, policy, frame_count,
                           ack_sample_index, journal_hash, status, retention_class, extra_json
                    FROM captures;
                 DROP TABLE captures;
                 ALTER TABLE captures_v3 RENAME TO captures;
                 UPDATE meta SET value = '3' WHERE key = 'schema_version';",
            )
            .expect("rewind to v3");
            pair
        };

        // Opening upgrades: version bumped, column added, pre-upgrade
        // rows read as ordinary takes (secure_field false), and the
        // pre-upgrade attempt is intact.
        let store = StoreV2::open(&root).expect("reopen upgrades");
        assert_eq!(store.schema_version().expect("version"), SCHEMA_VERSION);
        let capture = store.get_capture(&id).expect("get").expect("present");
        assert!(!capture.secure_field);
        assert_eq!(store.attempts_for(&id).expect("attempts").len(), 1);
        // The upgraded store records corrections normally.
        assert!(
            store
                .upsert_correction_record(&correction(&id, &attempt, "req-1"))
                .expect("the table exists after the upgrade")
        );
        assert_eq!(store.correction_records_for(&id).expect("read").len(), 1);
    }

    #[test]
    fn correction_decision_spellings_round_trip() {
        for decision in [
            CorrectionDecision::Accepted,
            CorrectionDecision::Rejected,
            CorrectionDecision::Reverted,
            CorrectionDecision::Edited,
        ] {
            assert_eq!(CorrectionDecision::parse(decision.as_str()), Some(decision));
        }
        assert_eq!(CorrectionDecision::parse("nope"), None);
    }
}
