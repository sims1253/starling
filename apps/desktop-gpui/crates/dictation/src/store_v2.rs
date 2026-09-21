//! Storage v2 core (I2, `docs/program/design/e17-native-runtime.md` §4).
//!
//! This is THE store (D14: no backwards compatibility of any kind — the
//! v1 file store remains in the tree only where the runtime state machine
//! and journal recovery still reference it). The data root —
//! `<data-root>/starling-gpui/` — holds the v2 layout:
//!
//! ```text
//! <root>/starling.db   SQLite (WAL) transactional metadata
//! <root>/audio/        <captureId>.sj finalized sample journals
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
//! [`V2_SCHEMA_VERSION`] in `meta`): `captures`, `recognition_attempts`,
//! `context_snapshots`, `mode_decisions`, `documents`/`revisions`,
//! `deliveries`, `tombstones`, `meta`. This core implements the
//! captures/attempts/tombstones/meta surfaces; the context/documents/
//! deliveries tables exist so later increments (I5) extend the schema
//! additively instead of rewriting it. `extra_json` per row preserves
//! unknown/newer fields **verbatim**: it is stored and returned as the raw
//! text a writer produced, never re-serialized from parsed form on paths
//! that do not touch it, and updates that add fields merge into the parsed
//! object without dropping keys.
//!
//! Forward compatibility (§4): a database whose `meta.schema_version` is
//! **higher** than [`V2_SCHEMA_VERSION`] is refused at open
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

use rusqlite::{params, Connection, OptionalExtension};

use crate::audio::decode_pcm16_wav;
use crate::journal::{
    self, JournalWriter, read_journal, samples_hash, seal_recovered_journal, sync_dir,
};
use crate::storage::{is_safe_path_component, now_iso};

/// Schema version of `starling.db` this build writes and understands.
/// Bump only with an additive migration path; a DB holding a higher value
/// is refused at open. v2 added `recognition_attempts.created_utc` (the
/// real updated-at source for the summaries).
pub const V2_SCHEMA_VERSION: u32 = 2;

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
/// Runtime ownership leases (§4): one `<ownerId>.lease` file per process
/// holding the data root, flocked for the owner's lifetime.
const LEASES_DIR: &str = "leases";
/// The v1 journal tree's tombstone directory (R21), a sibling of the v2
/// root: `<root>/journals/deleted/`. The v1 store quarantined deleted
/// journals here pending the retention sweep; the sweep still empties it
/// (never-delete-until-swept), but nothing writes into it anymore.
const LEGACY_DELETED_DIRS: &[&str] = &["journals", "deleted"];
const DB_FILE: &str = "starling.db";

/// Environment variable the runtime state machine's capture store uses to
/// opt its persistence into v2 (I2: v2 ships alongside v1 there, with no
/// automatic switchover). The desktop app itself runs v2 unconditionally
/// (D14) and never reads this flag.
pub const STORAGE_V2_FLAG_ENV: &str = "STARLING_STORAGE_V2";

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
    extra_json       TEXT
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
CREATE TABLE IF NOT EXISTS tombstones (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    deleted_utc TEXT NOT NULL,
    retention   TEXT NOT NULL
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
}

/// The lease this process holds: the owner id and the flocked lease file
/// (keeping the file open is what holds the flock, exactly like the
/// attempt markers). The flock outlives the file's name — acquisition
/// publishes the record with a rename, and the lock rides the inode.
#[derive(Debug)]
struct LeaseHandle {
    owner_id: String,
    started_utc: String,
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

        let db_version = Self::read_schema_version(&conn)?;
        match db_version {
            None => {
                // Fresh database: create everything in one transaction.
                let tx = conn.transaction()?;
                tx.execute_batch(SCHEMA_SQL)?;
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?1)",
                    params![V2_SCHEMA_VERSION.to_string()],
                )?;
                tx.commit()?;
            }
            Some(found) if found == V2_SCHEMA_VERSION => {
                // Same version: schema is already in place; verify the
                // version row is sane and touch nothing else.
            }
            Some(found) if found < V2_SCHEMA_VERSION => {
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
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?1)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![V2_SCHEMA_VERSION.to_string()],
                )?;
                tx.commit()?;
            }
            Some(found) => {
                // Higher version: refuse outright — this build must not
                // read or write a future format (§4 forward-compatible
                // reader). Nothing was mutated.
                return Err(StoreV2Error::SchemaTooNew {
                    found,
                    supported: V2_SCHEMA_VERSION,
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
        })
    }

    /// `<data-dir>/starling-gpui` — the same root the v1 store uses for
    /// `sessions/` and `journals/`, so both stores coexist during the
    /// transition.
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
        std::fs::rename(&staging, &audio)?;
        sync_dir(&self.root.join(AUDIO_DIR))?;
        sync_dir(&self.root.join(STAGING_DIR))?;
        Ok(())
    }

    /// §4 step 3: the SQLite transaction inserting the `captures` row,
    /// committed with `synchronous=FULL`, then the WAL checkpoint per
    /// policy. Returning `Ok` is the durable ack.
    pub fn commit_capture(&mut self, record: &CaptureRecord) -> Result<(), StoreV2Error> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO captures(
                id, created_utc, tz, device, actual_rate, policy, frame_count,
                ack_sample_index, journal_hash, status, retention_class, extra_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
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
            ],
        )?;
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
                        ack_sample_index, journal_hash, status, retention_class, extra_json
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
    /// time (the summary's updated-at source).
    pub fn insert_attempt(&mut self, attempt: &AttemptRecord) -> Result<(), StoreV2Error> {
        let tx = self.conn.transaction()?;
        tx.execute(
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
                    ack_sample_index, journal_hash, status, retention_class, extra_json
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
                    let problems = self.audio_problems(&record);
                    records.push(ListedCapture::Capture(CaptureListing { record, problems }));
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

    /// Bounded per-record audio check: the reasons a committed row cannot
    /// currently be played back, without loading any samples.
    fn audio_problems(&self, record: &CaptureRecord) -> Vec<String> {
        let path = self.audio_path(&record.id);
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
    pub fn audio_journal_path(&self, id: &str) -> Result<PathBuf, StoreV2Error> {
        validate_capture_id(id)?;
        if self.get_capture(id)?.is_none() {
            return Err(StoreV2Error::NotFound(id.to_string()));
        }
        let path = self.audio_path(id);
        if !path.exists() {
            return Err(StoreV2Error::Invalid(format!(
                "capture {id} has no audio journal on disk"
            )));
        }
        Ok(path)
    }

    /// Lazily loads one take's audio (the G02 "load on demand" contract):
    /// reads and verifies `audio/<id>.sj`. The verified prefix excludes any
    /// torn tail; `torn_tail_bytes` reports it.
    pub fn load_audio(&self, id: &str) -> Result<JournalAudio, StoreV2Error> {
        let path = self.audio_journal_path(id)?;
        read_audio_journal(&path)
    }

    // ------------------------------------------------------------------
    // Deletion (R21 semantics on the v2 layout).
    // ------------------------------------------------------------------

    /// Confirmed capture deletion: quarantine the journal (rename into
    /// `quarantine/` + fsyncs — the tombstone commit point), then one
    /// transaction inserting the `tombstones` row and removing the
    /// `captures` row (attempts cascade). A crash between the two is
    /// completed by [`Self::reconcile`]; a crash before the rename leaves
    /// everything live. Idempotent: deleting an unknown id is `Ok`.
    pub fn delete_capture(&mut self, id: &str) -> Result<(), StoreV2Error> {
        validate_capture_id(id)?;
        let row = self.get_capture(id)?;
        let audio = self.audio_path(id);
        if row.is_none() && !audio.exists() {
            return Ok(());
        }

        // 1. Tombstone: the quarantine rename is the commit point.
        if audio.exists() {
            std::fs::create_dir_all(self.root.join(QUARANTINE_DIR))?;
            let destination = self.quarantine_path(id);
            let _ = std::fs::remove_file(&destination); // stale tombstone
            std::fs::rename(&audio, &destination)?;
            sync_dir(&self.root.join(AUDIO_DIR))?;
            sync_dir(&self.root.join(QUARANTINE_DIR))?;
        }

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
        Ok(())
    }

    // ------------------------------------------------------------------
    // Retention sweep (R21: never-delete-until-swept).
    // ------------------------------------------------------------------

    /// The explicit retention sweep — the **only** code path in the store
    /// that unlinks deliberately-deleted content. Policy (frozen, §4):
    /// a confirmed delete quarantines the journal and tombstones the row;
    /// the bytes stay on disk, recoverable, until this sweep is called.
    /// Nothing here runs automatically: no read path, no reconcile, no
    /// open ever sweeps (that is the never-delete-until-swept contract).
    ///
    /// What it sweeps, per tree:
    ///
    /// - `quarantine/` — v2 tombstoned capture journals
    ///   ([`Self::delete_capture`], interrupted deletes completed by
    ///   [`Self::reconcile`]);
    /// - `journals/deleted/` — the v1 journal tree's tombstones (R21),
    ///   left behind by the deleted v1 store, still awaiting this sweep.
    ///
    /// Bookkeeping: each swept file removes its bytes and stamps its
    /// `tombstones` row `retention = 'swept'` (a tombstone is created if
    /// none exists — the never-resurrect guarantee must outlive the
    /// bytes); the report names every swept file with its size, and every
    /// entry that was left in place with the reason. Live content —
    /// `audio/`, `staging/`, anything not under the two tombstone trees —
    /// is untouched by construction.
    pub fn sweep_retention(&mut self) -> Result<SweepReport, StoreV2Error> {
        let mut report = SweepReport::default();
        self.sweep_tree(&self.root.join(QUARANTINE_DIR), "capture", &mut report)?;
        self.sweep_tree(
            &self.root.join(LEGACY_DELETED_DIRS[0]).join(LEGACY_DELETED_DIRS[1]),
            "journal",
            &mut report,
        )?;
        Ok(report)
    }

    /// Sweep one tombstone tree into `report` (`kind` is the tombstone
    /// kind rows get: `capture` for v2 quarantine, `journal` for the
    /// legacy v1 tree).
    fn sweep_tree(
        &mut self,
        dir: &Path,
        kind: &str,
        report: &mut SweepReport,
    ) -> Result<(), StoreV2Error> {
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
            if path.extension().and_then(|ext| ext.to_str()) != Some("sj") {
                // Not a tombstoned journal (hand-dropped junk, a stray
                // file): never-delete-until-swept cuts both ways — the
                // sweep only removes what the tombstone semantics cover.
                report.retained.push((name, "not a .sj journal".to_string()));
                continue;
            }
            let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            let id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default()
                .to_string();
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    swept_here = true;
                    report.swept.push(SweptFile {
                        id: id.clone(),
                        kind: kind.to_string(),
                        bytes,
                    });
                    report.swept_bytes += bytes;
                    // The tombstone outlives the bytes: stamp the row
                    // swept, or create it if the delete never wrote one
                    // (the file under a tombstone tree is itself the
                    // deliberate-delete evidence).
                    self.conn.execute(
                        "INSERT INTO tombstones(id, kind, deleted_utc, retention)
                         VALUES (?1, ?2, ?3, 'swept')
                         ON CONFLICT(id) DO UPDATE SET retention = 'swept'",
                        params![id, kind, now_iso()],
                    )?;
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
    pub fn reconcile(&mut self) -> Result<ReconciliationReport, StoreV2Error> {
        let mut report = ReconciliationReport::default();
        let foreign_owner = self.live_foreign_lease()?;

        // Tombstoned ids outrank everything (R21): the DB row and any file
        // under quarantine/ both mean "deliberately deleted".
        let mut dead: std::collections::HashSet<String> =
            journal_ids_in(&self.root.join(QUARANTINE_DIR)).into_iter().collect();
        {
            let mut stmt = self.conn.prepare("SELECT id FROM tombstones")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            for row in rows {
                dead.insert(row?);
            }
        }

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
            if foreign_owner.is_some() {
                // Client mode: the live owner may be writing this take
                // right now — its salvage is the owner's to run.
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
        for id in journal_ids_in(&self.root.join(AUDIO_DIR)) {
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
            let path = self.audio_path(&id);
            match self.get_capture(&id)? {
                // Row with finalized audio: healthy, nothing to do.
                Some(_) => {
                    if !path.exists() {
                        // Raced away between listing and now; nothing to do.
                        continue;
                    }
                }
                None => {
                    // Finalized audio with no row → orphan session:
                    // interrupted status, linked to the audio. In client
                    // mode this is deferred too — the owner may sit in the
                    // finalize→commit window, and racing an adoption into
                    // it would make the owner's own commit fail.
                    if foreign_owner.is_some() {
                        report.deferred_to_live_owner.push(id);
                        continue;
                    }
                    let parsed = match read_journal(&path) {
                        Ok(parsed) => parsed,
                        Err(err) => {
                            report.unreadable.push((id, err.to_string()));
                            continue;
                        }
                    };
                    if parsed.samples.is_empty() {
                        report.empty_journals.push(id);
                        continue;
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
                    };
                    self.commit_capture(&record)?;
                    report.orphan_sessions.push(id);
                }
            }
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
                if !self.audio_path(&id).exists() && status != "interrupted" {
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

    /// Finish a tombstoned id: remove any row, move any live journal (in
    /// `audio/` or `staging/`) into `quarantine/`. The R21 "retry
    /// completes the delete" path.
    fn complete_tombstoned(
        &mut self,
        id: &str,
        report: &mut ReconciliationReport,
    ) -> Result<(), StoreV2Error> {
        let audio = self.audio_path(id);
        if audio.exists() {
            std::fs::create_dir_all(self.root.join(QUARANTINE_DIR))?;
            let destination = self.quarantine_path(id);
            let _ = std::fs::remove_file(&destination);
            std::fs::rename(&audio, &destination)?;
            sync_dir(&self.root.join(AUDIO_DIR))?;
            sync_dir(&self.root.join(QUARANTINE_DIR))?;
        }
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

    /// The lease file path for one owner.
    fn lease_path(&self, owner_id: &str) -> PathBuf {
        self.root.join(LEASES_DIR).join(format!("{owner_id}.lease"))
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
    ///
    /// The lease is held until [`Self::release_lease`], the store is
    /// dropped, or the process dies — the flock on the lease file is the
    /// ownership signal, and the OS releases it when the owner dies, so a
    /// crashed owner's lease file is harmless and breakable.
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
        let broke = self.break_stale_leases()?;
        if let Some(owner) = self.live_foreign_lease()? {
            return Ok(LeaseAcquisition::Client { owner });
        }

        let owner_id = format!("l_{}", uuid::Uuid::new_v4().simple());
        let leases_dir = self.root.join(LEASES_DIR);
        std::fs::create_dir_all(&leases_dir)?;
        // §4: no shared fixed `.tmp` name — the scratch file is unique per
        // owner, so two processes can never fight over one temp path.
        let temp = unique_lease_temp(&leases_dir, &owner_id);
        let mut file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temp)
        {
            Ok(file) => file,
            Err(err) => return Err(err.into()),
        };
        // A freshly created file is never contended; a probe that says
        // otherwise means someone else's temp collided into this unique
        // name (external tampering) — abort without leaving the file.
        match try_flock_exclusive(&file) {
            Ok(FlockEvidence::Free) | Ok(FlockEvidence::Unknown) => {}
            Ok(FlockEvidence::Held) | Err(_) => {
                drop(file);
                let _ = std::fs::remove_file(&temp);
                return Err(StoreV2Error::Io(io::Error::other(format!(
                    "lease temp {temp:?} is already held"
                ))));
            }
        }
        let started_utc = now_iso();
        write_lease_record_content(&mut file, &started_utc)?;
        // Publish. The rename keeps the inode, so the flock taken above
        // rides into the published name — the probe any other process runs
        // on `leases/<owner>.lease` answers Held from the first moment the
        // lease is observable.
        std::fs::rename(&temp, self.lease_path(&owner_id))?;
        sync_dir(&leases_dir)?;
        self.lease = Some(LeaseHandle {
            owner_id: owner_id.clone(),
            started_utc,
            file,
        });
        Ok(LeaseAcquisition::Owner { owner_id, broke })
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

    /// Rewrite the heartbeat on our own flocked lease file, in place —
    /// a rename-based rewrite would replace the inode and strand the
    /// flock the ownership signal lives on.
    fn renew_lease_record(&mut self) -> Result<(), StoreV2Error> {
        let handle = self
            .lease
            .as_mut()
            .ok_or_else(|| StoreV2Error::Invalid("no lease held".to_string()))?;
        write_lease_record_content(&mut handle.file, &handle.started_utc)
    }

    /// Release the held lease (best-effort removal of the file; the flock
    /// goes with the dropped handle). A no-op when this instance holds no
    /// lease.
    pub fn release_lease(&mut self) -> Result<(), StoreV2Error> {
        if let Some(handle) = self.lease.take() {
            drop(handle.file); // release the flock before unlinking
            let _ = std::fs::remove_file(self.lease_path(&handle.owner_id));
            sync_dir(&self.root.join(LEASES_DIR))?;
        }
        Ok(())
    }

    /// Every lease on disk: this instance's own (marked `mine`), plus the
    /// foreign ones with their liveness probed. Introspection for callers
    /// deciding whether to repair, warn, or wait.
    pub fn lease_status(&self) -> Result<Vec<LeaseInfo>, StoreV2Error> {
        let own = self.lease.as_ref().map(|handle| handle.owner_id.clone());
        let mut leases = Vec::new();
        for (stem, record, flock) in self.probe_leases()? {
            let alive = lease_alive_from_probe(record.as_ref(), flock, self.lease_ttl);
            leases.push(LeaseInfo {
                mine: own.as_deref() == Some(stem.as_str()),
                owner_id: stem,
                pid: record.as_ref().map_or(0, |record| record.pid),
                record,
                flock,
                alive,
            });
        }
        Ok(leases)
    }

    /// Break stale leases (§4): remove every lease file whose owner is
    /// provably gone — its flock is free (the OS released it when the
    /// owner died), or, where flock cannot answer, its boot id no longer
    /// matches, its heartbeat expired past the TTL, or its recorded pid is
    /// dead. A lease that cannot be proven dead is never broken. Crash
    /// leftovers of the unique acquisition temporaries (flock-free
    /// `*.tmp`) are swept too. Returns the broken owner ids.
    pub fn break_stale_leases(&mut self) -> Result<Vec<String>, StoreV2Error> {
        let own = self
            .lease
            .as_ref()
            .map(|handle| handle.owner_id.clone());
        let mut broken = Vec::new();
        let mut removed_any = false;
        for (stem, record, flock) in self.probe_leases()? {
            if Some(&stem) == own.as_ref() {
                continue; // ours by construction
            }
            if lease_alive_from_probe(record.as_ref(), flock, self.lease_ttl) {
                continue;
            }
            let _ = std::fs::remove_file(self.lease_path(&stem));
            broken.push(stem);
            removed_any = true;
        }
        // Acquisition-temp leftovers: unique names, so nothing legitimate
        // can collide; only a dead writer's is removed (a live one holds
        // its flock between create and rename).
        for temp in lease_temps_in(&self.root.join(LEASES_DIR)) {
            let file = OpenOptions::new().read(true).write(true).open(&temp);
            if let Ok(file) = file {
                if matches!(try_flock_exclusive(&file), Ok(FlockEvidence::Free)) {
                    let _ = std::fs::remove_file(&temp);
                    removed_any = true;
                }
            }
        }
        if removed_any {
            sync_dir(&self.root.join(LEASES_DIR))?;
        }
        Ok(broken)
    }

    /// The live foreign lease on this root, when one exists — the store
    /// this process should treat itself as a client of.
    fn live_foreign_lease(&self) -> Result<Option<LeaseInfo>, StoreV2Error> {
        let own = self.lease.as_ref().map(|handle| handle.owner_id.clone());
        for (stem, record, flock) in self.probe_leases()? {
            if Some(&stem) == own.as_ref() {
                continue;
            }
            let alive = lease_alive_from_probe(record.as_ref(), flock, self.lease_ttl);
            if alive {
                return Ok(Some(LeaseInfo {
                    owner_id: stem,
                    pid: record.as_ref().map_or(0, |record| record.pid),
                    mine: false,
                    record,
                    flock,
                    alive,
                }));
            }
        }
        Ok(None)
    }

    /// Every `*.lease` under `leases/` with its record and flock evidence.
    /// The probe opens each file on its own handle, so acquiring-then-
    /// dropping the flock on a free file releases it again.
    fn probe_leases(
        &self,
    ) -> Result<Vec<(String, Option<LeaseRecord>, FlockEvidence)>, StoreV2Error> {
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
            let mut file = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(file) => file,
                Err(_) => {
                    // Unreadable, but present: probe as unknown — never
                    // decide ownership on an I/O failure.
                    probed.push((stem, None, FlockEvidence::Unknown));
                    continue;
                }
            };
            let flock = try_flock_exclusive(&file).unwrap_or(FlockEvidence::Unknown);
            let record = read_lease_record(&mut file);
            probed.push((stem, record, flock));
        }
        probed.sort_by(|left, right| left.0.cmp(&right.0));
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
        let source = source.as_ref();
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
        if let Some(parent) = source.parent() {
            sync_dir(parent)?;
        }
        std::fs::rename(source, &destination)?;
        sync_dir(&self.root.join(AUDIO_DIR))?;

        let recovery_note = match (torn_tail_bytes, was_finalized) {
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
        let note = [note.map(str::to_string), recovery_note]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" ");

        let count = parsed.samples.len() as u64;
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
            status: if torn_tail_bytes > 0 || !was_finalized {
                CaptureStatus::Interrupted
            } else {
                CaptureStatus::Complete
            },
            retention_class: "standard".to_string(),
            extra_json: (!note.is_empty()).then(|| merge_extra_note(None, &note)),
        };
        self.commit_capture(&record)?;
        Ok(record)
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
        if self.get_capture(capture_id)?.is_none() {
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        let id = format!("a_{}", uuid::Uuid::new_v4().simple());
        self.hold_attempt_lock(&id)?;
        match self.insert_attempt(&AttemptRecord {
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
        }) {
            Ok(()) => Ok(id),
            Err(err) => {
                // The row never became visible: the marker must not
                // outlive the attempt it was taken for.
                self.release_attempt_lock(&id);
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

        let changed = match outcome {
            RecognitionOutcome::Completed { text, extra_json } => self.conn.execute(
                "UPDATE recognition_attempts
                 SET text = ?1, partial_or_final = 'final', status = 'completed',
                     extra_json = COALESCE(?2, extra_json)
                 WHERE id = ?3 AND status = 'started'",
                params![text, extra_json, attempt_id],
            )?,
            RecognitionOutcome::Failed { message } => {
                let extra = serde_json::json!({ "error": message });
                self.conn.execute(
                    "UPDATE recognition_attempts
                     SET status = 'failed', extra_json = ?1
                     WHERE id = ?2 AND status = 'started'",
                    params![extra.to_string(), attempt_id],
                )?
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
    fn hold_attempt_lock(&mut self, attempt_id: &str) -> Result<(), StoreV2Error> {
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
        self.attempt_locks.insert(attempt_id.to_string(), file);
        Ok(())
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
        let _ = std::fs::remove_file(self.attempt_lock_path(attempt_id));
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
        };
        store.commit_capture(&record)?;
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
    /// In-flight ids (staging journals, orphan candidates) a live foreign
    /// lease owner claimed: this run was a client (§4 ownership) and left
    /// them for the owner's own reconcile. Informational — a deferral is
    /// normal multi-instance operation, not a repair finding, so it does
    /// not contribute to [`Self::has_findings`].
    pub deferred_to_live_owner: Vec<String>,
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
                if self.marked_interrupted.len() == 1 { "" } else { "s" }
            ));
        }
        if !self.completed_deletes.is_empty() {
            parts.push(format!(
                "completed {} pending deletion{}",
                self.completed_deletes.len(),
                if self.completed_deletes.len() == 1 { "" } else { "s" }
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
    /// came from (`capture` = v2 `quarantine/`, `journal` = the legacy
    /// `journals/deleted/`), and the bytes freed.
    pub swept: Vec<SweptFile>,
    /// Total bytes unlinked.
    pub swept_bytes: u64,
    /// `(name, reason)` for entries left in place: files that are not
    /// `.sj` journals, or removals that failed (the next sweep retries).
    pub retained: Vec<(String, String)>,
}

/// One swept tombstoned file.
#[derive(Debug, Clone)]
pub struct SweptFile {
    pub id: String,
    /// `capture` (v2 `quarantine/`) or `journal` (legacy `journals/deleted/`).
    pub kind: String,
    pub bytes: u64,
}

// ---------------------------------------------------------------------------
// Free helpers.
// ---------------------------------------------------------------------------

/// Whether the testing flag is on. Only the runtime state machine reads
/// it (see [`STORAGE_V2_FLAG_ENV`]); the desktop app runs v2
/// unconditionally (D14).
pub fn v2_enabled_for_testing() -> bool {
    v2_enabled_from(std::env::var(STORAGE_V2_FLAG_ENV).ok().as_deref())
}

/// The flag mapping, pure so it is testable without racing the process
/// environment: `1`, `true`, `yes`, `on` (case-insensitive) are on;
/// everything else — including unset — is off.
pub fn v2_enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(text) => matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// A local UTC-offset label for the `tz` column; `UTC` when the local
/// offset cannot be determined (multi-threaded sandbox).
fn local_tz_label() -> String {
    time::OffsetDateTime::now_local()
        .map(|now| now.offset().to_string())
        .unwrap_or_else(|_| "UTC".to_string())
}

fn validate_capture_id(id: &str) -> Result<(), StoreV2Error> {
    if !is_safe_path_component(id) {
        return Err(StoreV2Error::Invalid(format!(
            "capture id {id:?} must be non-empty and contain no path separators"
        )));
    }
    Ok(())
}

/// What one marker probe learned about its flock (#213 review).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlockEvidence {
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
fn try_flock_exclusive(file: &File) -> io::Result<FlockEvidence> {
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

/// Without flock there is no lock to ask (#213 review): the probe answers
/// [`FlockEvidence::Unknown`] and the marker's recorded PID decides
/// ownership instead (see [`attempt_owned_from`]). Every desktop target
/// this port builds today is unix; on flock-less platforms foreign PIDs
/// read as dead so crash-orphaned attempts stay sweepable — the
/// multi-instance spare is then unix-only, matching what the reference
/// can guarantee without a lock manager.
#[cfg(not(unix))]
fn try_flock_exclusive(_file: &File) -> io::Result<FlockEvidence> {
    Ok(FlockEvidence::Unknown)
}

/// Whether the process `pid` is still alive (#213 review) — the fallback
/// ownership signal wherever the flock cannot answer.
#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    // SAFETY: kill(2) with signal 0 performs existence and permission
    // checks only — no signal is ever delivered.
    if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
        return true;
    }
    // EPERM means the process exists but belongs to another user; only
    // ESRCH (no such process) means it is gone.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Without a std primitive to ask the OS about a foreign PID (#213
/// review), it is presumed dead: this process's OWN attempts are spared
/// by the in-process registry before any marker is probed, so the only
/// markers read here belong to other processes — unknowable liveness
/// must not turn every crash-orphaned attempt into a permanently stuck
/// `started` row. The cost is that a second live instance on a flock-less
/// platform can have its attempt interrupted, the same degradation the
/// Electron reference accepts without a lock manager; unix keeps the
/// full cross-process guarantee via flock.
#[cfg(not(unix))]
fn process_is_alive(_pid: u32) -> bool {
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
pub fn read_audio_journal(path: &Path) -> Result<JournalAudio, StoreV2Error> {
    let parsed = read_journal(path).map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
    Ok(JournalAudio {
        sample_rate: parsed.sample_rate,
        samples: parsed.samples,
        finalized: parsed.finalized,
        torn_tail_bytes: parsed.torn_tail_bytes,
    })
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
    /// The flock evidence the probe gathered (crate-internal diagnostic;
    /// `alive` is the verdict callers act on).
    pub(crate) flock: FlockEvidence,
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

/// Write the current lease record onto `file` in place: truncate, write,
/// rewind, fsync. In place on purpose — the flock lives on this open file
/// description, and a rename-based publish would strand it (acquisition
/// publishes via rename *before* anything else can observe the lease;
/// heartbeats must not).
fn write_lease_record_content(
    file: &mut File,
    started_utc: &str,
) -> Result<(), StoreV2Error> {
    use std::io::{Seek, SeekFrom, Write};
    let record = LeaseRecord {
        pid: std::process::id(),
        boot_id: boot_id(),
        started_utc: started_utc.to_string(),
        heartbeat_utc: now_iso(),
        heartbeat_ms: now_epoch_ms(),
    };
    let bytes = serde_json::to_vec(&record)
        .map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

/// The record inside one lease file, when one parses. A torn record (a
/// crash mid-heartbeat) reads as `None` — the flock then decides.
fn read_lease_record(file: &mut File) -> Option<LeaseRecord> {
    use std::io::{Read, Seek, SeekFrom};
    let mut text = String::new();
    file.seek(SeekFrom::Start(0)).ok()?;
    file.read_to_string(&mut text).ok()?;
    serde_json::from_str(&text).ok()
}

/// A unique acquisition temporary under `leases/`, named after its owner
/// (§4: the fixed shared `.tmp` name is replaced by unique temporaries
/// under the lease owner — two processes can never collide on it).
fn unique_lease_temp(leases_dir: &Path, owner_id: &str) -> PathBuf {
    leases_dir.join(format!(
        "{owner_id}.{}.tmp",
        uuid::Uuid::new_v4().simple()
    ))
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

/// Unix epoch milliseconds (0 only if the clock is before 1970).
fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or(0)
}

/// Whether a heartbeat recorded at `heartbeat_ms` is still fresh under
/// `ttl`. A zero/garbage timestamp reads as ancient — stale.
fn heartbeat_is_fresh(heartbeat_ms: u64, ttl: std::time::Duration) -> bool {
    now_epoch_ms().saturating_sub(heartbeat_ms) <= ttl.as_millis() as u64
}

/// The probed verdict on one lease: the flock decides when it answered
/// (the OS is the truth — a held lock means a live owner, a freed lock
/// means a dead one, whatever the heartbeat says); the record's
/// boot id / heartbeat / pid decide only where the flock cannot answer.
fn lease_alive_from_probe(
    record: Option<&LeaseRecord>,
    flock: FlockEvidence,
    ttl: std::time::Duration,
) -> bool {
    match flock {
        FlockEvidence::Held => true,
        FlockEvidence::Free => false,
        FlockEvidence::Unknown => match record {
            // No record to reason about and no lock to ask: never break
            // what cannot be proven dead.
            None => true,
            Some(record) => lease_alive_from(
                boot_matches(&record.boot_id),
                Some(heartbeat_is_fresh(record.heartbeat_ms, ttl)),
                Some(process_is_alive(record.pid)),
            ),
        },
    }
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
        assert_eq!(store.schema_version().expect("version"), V2_SCHEMA_VERSION);

        committed_take(&mut store, &ramp(50, 0));

        // Reopening a same-version database neither upgrades nor refuses.
        drop(store);
        let store = store_in(&dir);
        assert_eq!(store.schema_version().expect("version"), V2_SCHEMA_VERSION);
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
        assert_eq!(store.schema_version().expect("version"), V2_SCHEMA_VERSION);
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
                assert_eq!(supported, V2_SCHEMA_VERSION);
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

    // ---- testing flag ---------------------------------------------------

    #[test]
    fn the_v2_flag_maps_values_and_defaults_off() {
        assert!(!v2_enabled_from(None), "unset = off");
        for off in ["", "0", "false", "no", "off", "garbage"] {
            assert!(!v2_enabled_from(Some(off)), "{off:?} must be off");
        }
        for on in ["1", "true", "TRUE", "Yes", "on"] {
            assert!(v2_enabled_from(Some(on)), "{on:?} must be on");
        }
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
        store.root().join("staging").join(format!("{id}.sj")).exists()
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

        // One published lease file, no temp left behind, and its record
        // names this process, this boot, right now.
        assert_eq!(lease_files(&root), vec![format!("{owner_id}.lease")]);
        let record: LeaseRecord = serde_json::from_str(
            &std::fs::read_to_string(root.join("leases").join(format!("{owner_id}.lease")))
                .expect("read lease"),
        )
        .expect("lease record parses");
        assert_eq!(record.pid, std::process::id());
        if cfg!(target_os = "linux") {
            assert!(!record.boot_id.is_empty(), "boot id is recorded on linux");
            assert_eq!(record.boot_id, boot_id());
        }
        assert!(
            now_epoch_ms().saturating_sub(record.heartbeat_ms) < 5_000,
            "the heartbeat is current"
        );
        assert!(!record.started_utc.is_empty());
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
        assert!(store_root_has_staging(&client, &take_id), "staging untouched");
        assert!(client.get_capture(&take_id).expect("row").is_none());
        assert!(!report.has_findings(), "a deferral is not a finding: {report:?}");

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
        assert!(
            owner
                .root()
                .join("quarantine")
                .join(format!("{doomed}.sj"))
                .exists()
        );
    }

    #[test]
    fn reacquiring_keeps_the_owner_id_and_renews_the_heartbeat() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let first = match store.acquire_lease().expect("acquire") {
            LeaseAcquisition::Owner { owner_id, .. } => owner_id,
            other => panic!("{other:?}"),
        };
        let before: LeaseRecord = serde_json::from_str(
            &std::fs::read_to_string(store.lease_path(&first)).expect("lease"),
        )
        .expect("record");

        assert!(store.heartbeat_lease().expect("heartbeat"));
        let second = match store.acquire_lease().expect("re-acquire") {
            LeaseAcquisition::Owner { owner_id, .. } => owner_id,
            other => panic!("{other:?}"),
        };
        assert_eq!(first, second, "the identity is stable");
        let after: LeaseRecord = serde_json::from_str(
            &std::fs::read_to_string(store.lease_path(&first)).expect("lease"),
        )
        .expect("record");
        assert_eq!(after.started_utc, before.started_utc, "start is preserved");
        assert!(
            after.heartbeat_ms >= before.heartbeat_ms,
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

    /// Forge a foreign lease file on `root` with the given record fields.
    /// Nothing flocks it — the shape a crashed (or never-started) owner
    /// leaves behind.
    fn forged_lease(root: &Path, owner_id: &str, pid: u32, heartbeat_ms: u64) {
        let record = LeaseRecord {
            pid,
            boot_id: boot_id(),
            started_utc: "2026-09-01T00:00:00.000Z".to_string(),
            heartbeat_utc: "2026-09-01T00:00:00.000Z".to_string(),
            heartbeat_ms,
        };
        std::fs::write(
            root.join("leases").join(format!("{owner_id}.lease")),
            serde_json::to_string(&record).expect("serialize"),
        )
        .expect("write forged lease");
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
        let held = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("leases").join("l_zombie.lease"))
            .expect("open");
        assert_eq!(
            try_flock_exclusive(&held).expect("flock"),
            FlockEvidence::Free,
            "the test holds the lease like a live owner"
        );

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
        assert!(quarantine.exists(), "never-delete-until-swept holds pre-sweep");
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
    fn sweep_also_empties_the_legacy_v1_deleted_tree() {
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);
        let legacy = store
            .root()
            .join("journals")
            .join("deleted");
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
        assert_eq!(report.swept.len(), 1, "only the tombstoned journal: {report:?}");
        assert_eq!(report.swept[0].id, deleted);
        // The live take's audio and row are untouched.
        assert!(store.root().join("audio").join(format!("{live}.sj")).exists());
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
        std::fs::write(quarantine.join("notes.txt"), b"hand-dropped junk")
            .expect("write");
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
}
