//! Storage v2 core (I2, `docs/program/design/e17-native-runtime.md` §4).
//!
//! One data root — `<data-root>/starling-gpui/` — holds both stores side by
//! side during the transition: the v1 `sessions/` + `journals/` trees (the
//! app keeps using those by default; no automatic switchover in this
//! increment) and the v2 layout:
//!
//! ```text
//! <root>/starling.db   SQLite (WAL) transactional metadata
//! <root>/audio/        <captureId>.sj finalized sample journals
//! <root>/staging/      <captureId>.sj in-flight journals (§4 step 1)
//! <root>/quarantine/   deliberately-deleted journals (R21 tombstones)
//! ```
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
//!
//! # Migration (additive, reversible, dry-run)
//!
//! [`StoreV2::migrate_v1_dry_run`] / [`StoreV2::migrate_v1_apply`] import
//! the v1 [`crate::storage::FileSessionStore`] directories: copies only
//! (originals are never rewritten), verified by per-record content hashes
//! in the *sample domain* plus counts — not just row counts — with a
//! `migration_report.json` written for user review before any cutover.
//! Unknown v1 manifest fields land in `extra_json`. Rollback
//! ([`StoreV2::rollback_import`]) discards the imported rows and
//! quarantines their journals; the v1 originals are untouched throughout.

use std::io;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension};

use crate::audio::{PcmAudio, decode_pcm16_wav};
use crate::journal::{
    self, JournalWriter, read_journal, samples_hash, seal_recovered_journal, sync_dir,
};
use crate::storage::{
    DictationSession, FileSessionStore, SessionStatus, TranscriptionResult, is_safe_path_component,
    now_iso,
};

/// Schema version of `starling.db` this build writes and understands.
/// Bump only with an additive migration path; a DB holding a higher value
/// is refused at open.
pub const V2_SCHEMA_VERSION: u32 = 1;

/// WAL checkpoint policy (D11): run `PRAGMA wal_checkpoint(PASSIVE)` after
/// every N metadata commits. Default 64 — frequent enough that the WAL
/// cannot grow without bound across a session of takes, rare enough that
/// the checkpoint cost stays off the per-take path on slow disks. Tunable
/// via [`StoreV2::set_checkpoint_every_commits`].
pub const WAL_CHECKPOINT_EVERY_COMMITS: u32 = 64;

/// Directory names under the v2 root.
const AUDIO_DIR: &str = "audio";
const STAGING_DIR: &str = "staging";
const QUARANTINE_DIR: &str = "quarantine";
const DB_FILE: &str = "starling.db";

/// The migration report file written beside the database for user review.
pub const MIGRATION_REPORT_FILE: &str = "migration_report.json";

/// Environment variable that opts the app into the v2 store for testing
/// (I2 ships v2 alongside v1 with no automatic switchover; wiring the app
/// to consult this flag is cutover work, see the module docs).
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
    extra_json       TEXT
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
                // Lower version: apply the (idempotent) schema and bump.
                let tx = conn.transaction()?;
                tx.execute_batch(SCHEMA_SQL)?;
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

    /// [`Self::begin_take`] with a caller-chosen id and journal sample
    /// rate — the migration path (the v1 session id becomes the capture
    /// id so imports are traceable and re-runnable).
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

    /// Inserts a `recognition_attempts` row.
    pub fn insert_attempt(&mut self, attempt: &AttemptRecord) -> Result<(), StoreV2Error> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO recognition_attempts(
                id, capture_id, backend, model_hash, language, options_json, text,
                partial_or_final, status, timing_json, extra_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
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
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// All attempts for a capture, oldest first (insertion id order is
    /// uuid-random; ordering is by rowid, i.e. insertion order).
    pub fn attempts_for(&self, capture_id: &str) -> Result<Vec<AttemptRecord>, StoreV2Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, capture_id, backend, model_hash, language, options_json, text,
                    partial_or_final, status, timing_json, extra_json
             FROM recognition_attempts WHERE capture_id = ?1 ORDER BY rowid",
        )?;
        let rows = stmt.query_map(params![capture_id], |row| {
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
            })
        })?;
        let mut attempts = Vec::new();
        for row in rows {
            attempts.push(row?);
        }
        Ok(attempts)
    }

    fn meta_get(&self, key: &str) -> Result<Option<String>, StoreV2Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?)
    }

    fn meta_set(&mut self, key: &str, value: &str) -> Result<(), StoreV2Error> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Bounded listing + lazy audio (G02 semantics carried to v2).
    // ------------------------------------------------------------------

    /// Metadata-only listing: rows come from SQLite; each record gets a
    /// bounded audio check (file presence + the 13-byte journal header —
    /// never PCM). A damaged row or a missing/corrupt journal surfaces its
    /// reason on that record only; the listing never aborts.
    pub fn list_records(&self, offset: usize, limit: usize) -> Result<CapturePage, StoreV2Error> {
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

    /// Lazily loads one take's audio (the G02 "load on demand" contract):
    /// reads and verifies `audio/<id>.sj`. The verified prefix excludes any
    /// torn tail; `torn_tail_bytes` reports it.
    pub fn load_audio(&self, id: &str) -> Result<JournalAudio, StoreV2Error> {
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
        let parsed =
            read_journal(&path).map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
        Ok(JournalAudio {
            sample_rate: parsed.sample_rate,
            samples: parsed.samples,
            finalized: parsed.finalized,
            torn_tail_bytes: parsed.torn_tail_bytes,
        })
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
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO tombstones(id, kind, deleted_utc, retention)
             VALUES (?1, 'capture', ?2, 'quarantined')",
            params![id, now_iso()],
        )?;
        tx.execute("DELETE FROM captures WHERE id = ?1", params![id])?;
        tx.commit()?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Recovery (§4 reconciliation).
    // ------------------------------------------------------------------

    /// Reconciles journal files and metadata rows after a crash. Both
    /// sides are inspected independently; everything is idempotent, so a
    /// crash during reconciliation is repaired by the next run.
    pub fn reconcile(&mut self) -> Result<ReconciliationReport, StoreV2Error> {
        let mut report = ReconciliationReport::default();

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
                    // interrupted status, linked to the audio.
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
            let tx = self.conn.transaction()?;
            tx.execute(
                "INSERT OR REPLACE INTO tombstones(id, kind, deleted_utc, retention)
                 VALUES (?1, 'capture', ?2, 'quarantined')",
                params![id, now_iso()],
            )?;
            tx.execute("DELETE FROM captures WHERE id = ?1", params![id])?;
            tx.commit()?;
        }
        report.completed_deletes.push(id.to_string());
        Ok(())
    }

    // ------------------------------------------------------------------
    // v1 migration (additive, reversible, dry-run).
    // ------------------------------------------------------------------

    /// Dry-run: classify the v1 store and hash every importable record
    /// without writing anything to v2. The report is written to
    /// `<root>/migration_report.json` for user review.
    pub fn migrate_v1_dry_run(
        &mut self,
        v1_root: impl AsRef<Path>,
    ) -> Result<MigrationReport, StoreV2Error> {
        let v1_root = v1_root.as_ref();
        let batch_id = format!("mig_{}", uuid::Uuid::new_v4().simple());
        let (planned, skipped) = plan_v1_import(v1_root)?;
        let report = MigrationReport::for_plan(&batch_id, v1_root, &planned, skipped, true);
        self.write_migration_report(&report)?;
        Ok(report)
    }

    /// Verified import: each importable v1 record is copied through the §4
    /// crash protocol into v2 (staging journal → finalize → promote →
    /// commit), then verified by re-reading the journal and comparing
    /// sample-domain content hashes and counts. Originals are only ever
    /// read. The verified report is written to
    /// `<root>/migration_report.json`; the batch id it carries is the
    /// rollback handle ([`Self::rollback_import`]).
    pub fn migrate_v1_apply(
        &mut self,
        v1_root: impl AsRef<Path>,
    ) -> Result<MigrationReport, StoreV2Error> {
        let v1_root = v1_root.as_ref();
        let batch_id = format!("mig_{}", uuid::Uuid::new_v4().simple());
        let (planned, skipped) = plan_v1_import(v1_root)?;
        let mut report = MigrationReport::for_plan(&batch_id, v1_root, &planned, skipped, false);

        for planned_record in &planned {
            let entry = report.records.iter_mut().find(|r| r.id == planned_record.id);
            let entry = match entry {
                Some(entry) => entry,
                None => continue,
            };
            match self.import_one(&batch_id, planned_record) {
                Ok(ImportOutcome {
                    journal_hash,
                    sample_hash,
                    sample_count,
                }) => {
                    entry.dest_journal_hash = Some(journal_hash);
                    entry.dest_sample_hash = Some(sample_hash);
                    entry.dest_sample_count = Some(sample_count);
                    entry.verified = Some(
                        entry.dest_sample_hash.as_deref()
                            == Some(entry.source_sample_hash.as_str())
                            && entry.dest_sample_count == Some(entry.source_sample_count),
                    );
                }
                Err(err) => {
                    entry.error = Some(err.to_string());
                    entry.verified = Some(false);
                }
            }
        }

        report.counts.imported = report.records.iter().filter(|r| r.error.is_none()).count();
        report.counts.verified = report
            .records
            .iter()
            .filter(|r| r.verified == Some(true))
            .count();
        self.write_migration_report(&report)?;
        self.meta_set(
            &format!("migration:{batch_id}"),
            &serde_json::to_string(&report)
                .map_err(|err| StoreV2Error::Invalid(err.to_string()))?,
        )?;
        Ok(report)
    }

    fn import_one(
        &mut self,
        batch_id: &str,
        planned: &PlannedRecord,
    ) -> Result<ImportOutcome, StoreV2Error> {
        if !is_safe_path_component(&planned.id) {
            return Err(StoreV2Error::Invalid(format!(
                "v1 session id {:?} is not a safe path component",
                planned.id
            )));
        }
        if self.get_capture(&planned.id)?.is_some() || self.audio_path(&planned.id).exists() {
            return Err(StoreV2Error::Invalid(format!(
                "destination already holds capture {}; refusing to overwrite (re-run after a \
                 rollback if this was a retry)",
                planned.id
            )));
        }

        // Copy through the crash protocol: staging → finalize → promote →
        // commit. A crash mid-import leaves a recoverable orphan, exactly
        // like a take. The v1 session id becomes the capture id.
        let extra = serde_json::json!({
            "importedFrom": "file-v1",
            "importBatch": batch_id,
            "v1": planned.v1_extra,
        });
        let meta = TakeMeta {
            tz: String::new(),
            device: String::new(),
            policy: "v1-import".to_string(),
            retention_class: "standard".to_string(),
            extra_json: Some(
                serde_json::to_string(&extra)
                    .map_err(|err| StoreV2Error::Invalid(err.to_string()))?,
            ),
        };
        let mut take = self.begin_take_with_id(planned.id.clone(), planned.pcm.sample_rate, meta)?;
        for chunk in planned.pcm.samples.chunks(4096) {
            take.append_frames(chunk)?;
        }
        let ack = take.write_boundary()?;
        if ack != planned.pcm.samples.len() as u64 {
            return Err(StoreV2Error::Invalid(format!(
                "journal acknowledged {ack} samples for {} written",
                planned.pcm.samples.len()
            )));
        }
        let finalized = take.finalize()?;
        self.promote_from_staging(&finalized.id)?;
        let record = CaptureRecord {
            id: finalized.id.clone(),
            created_utc: planned.created_utc.clone(),
            tz: finalized.meta.tz,
            device: finalized.meta.device,
            actual_rate: finalized.sample_rate,
            policy: finalized.meta.policy,
            frame_count: finalized.total_samples,
            ack_sample_index: finalized.total_samples,
            journal_hash: finalized.content_hash,
            status: if planned.v1_interrupted {
                CaptureStatus::Interrupted
            } else {
                CaptureStatus::Complete
            },
            retention_class: finalized.meta.retention_class,
            extra_json: finalized.meta.extra_json,
        };
        self.commit_capture(&record)?;

        // Recognition history: one attempt row per retained transcript
        // (history first, current last — the reading order).
        let mut history = planned.transcript_history.clone();
        if let Some(current) = planned.transcript.clone() {
            history.push(current);
        }
        for transcript in history {
            let extra = serde_json::to_string(&transcript)
                .map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
            self.insert_attempt(&AttemptRecord {
                id: format!("a_{}", uuid::Uuid::new_v4().simple()),
                capture_id: planned.id.clone(),
                backend: "v1-import".to_string(),
                model_hash: None,
                language: None,
                options_json: None,
                text: transcript.text,
                partial_or_final: "final".to_string(),
                status: "completed".to_string(),
                timing_json: None,
                extra_json: Some(extra),
            })?;
        }

        // Verify: re-read the journal from disk and hash its verified
        // samples — the destination is checked in the sample domain
        // against the source, not against what we intended to write.
        let parsed = read_journal(&self.audio_path(&planned.id))
            .map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
        Ok(ImportOutcome {
            journal_hash: format!("{:016x}", samples_hash(&parsed.samples)),
            sample_hash: fnv1a_hex_samples(&parsed.samples),
            sample_count: parsed.samples.len() as u64,
        })
    }

    fn write_migration_report(&self, report: &MigrationReport) -> Result<(), StoreV2Error> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
        std::fs::write(self.root.join(MIGRATION_REPORT_FILE), json)?;
        Ok(())
    }

    /// The stored report of a completed migration batch (see
    /// [`Self::migrate_v1_apply`]), for review before cutover and as the
    /// record of what a rollback would discard.
    pub fn migration_report(&self, batch_id: &str) -> Result<Option<MigrationReport>, StoreV2Error> {
        match self.meta_get(&format!("migration:{batch_id}"))? {
            None => Ok(None),
            Some(text) => serde_json::from_str(&text)
                .map(Some)
                .map_err(|err| StoreV2Error::Invalid(format!("stored migration report: {err}"))),
        }
    }

    /// Rollback (reversible migration): discard every row imported by
    /// `batch_id` — their journals are moved into `quarantine/` (never
    /// unlinked; reconciliation treats them as tombstoned) — and drop the
    /// batch's `meta` entry. The v1 originals are of course untouched.
    /// Returns how many captures were discarded.
    pub fn rollback_import(&mut self, batch_id: &str) -> Result<usize, StoreV2Error> {
        let marker = format!("\"importBatch\":\"{batch_id}\"");
        let mut ids = Vec::new();
        {
            let mut stmt = self.conn.prepare("SELECT id, extra_json FROM captures")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })?;
            for row in rows {
                let (id, extra) = row?;
                if extra.as_deref().is_some_and(|text| text.contains(&marker)) {
                    ids.push(id);
                }
            }
        }

        for id in &ids {
            let audio = self.audio_path(id);
            if audio.exists() {
                std::fs::create_dir_all(self.root.join(QUARANTINE_DIR))?;
                let destination = self.quarantine_path(id);
                let _ = std::fs::remove_file(&destination);
                std::fs::rename(&audio, &destination)?;
                sync_dir(&self.root.join(AUDIO_DIR))?;
                sync_dir(&self.root.join(QUARANTINE_DIR))?;
            }
        }

        let tx = self.conn.transaction()?;
        for id in &ids {
            tx.execute(
                "INSERT OR REPLACE INTO tombstones(id, kind, deleted_utc, retention)
                 VALUES (?1, 'migration-rollback', ?2, 'quarantined')",
                params![id, now_iso()],
            )?;
            tx.execute("DELETE FROM captures WHERE id = ?1", params![id])?;
        }
        tx.execute(
            "DELETE FROM meta WHERE key = ?1",
            params![format!("migration:{batch_id}")],
        )?;
        tx.commit()?;
        Ok(ids.len())
    }
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
        let finalized = self.finalize()?;
        store.promote_from_staging(&finalized.id)?;
        let record = CaptureRecord {
            id: finalized.id.clone(),
            created_utc: finalized.created_utc.clone(),
            tz: finalized.meta.tz.clone(),
            device: finalized.meta.device.clone(),
            actual_rate: finalized.sample_rate,
            policy: finalized.meta.policy.clone(),
            frame_count: finalized.total_samples,
            ack_sample_index: finalized.total_samples,
            journal_hash: finalized.content_hash,
            status: CaptureStatus::Complete,
            retention_class: finalized.meta.retention_class,
            extra_json: finalized.meta.extra_json,
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
}

/// One torn-take recovery, with the discarded tail size (the gap).
#[derive(Debug, Clone)]
pub struct RecoveredTake {
    pub id: String,
    pub torn_tail_bytes: u64,
}

/// The JSON migration report produced for user review before any cutover.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationReport {
    /// Report format version (independent of the DB schema version).
    pub schema_version: u32,
    pub generated_at: String,
    pub source_root: String,
    pub dry_run: bool,
    pub batch_id: String,
    pub counts: MigrationCounts,
    pub records: Vec<MigratedRecord>,
    pub skipped_damaged: Vec<SkippedRecord>,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationCounts {
    /// Readable v1 records found (sessions plus recovered orphans).
    pub source_sessions: usize,
    /// v1 records too damaged to read.
    pub source_damaged: usize,
    /// Records copied into v2 without error.
    pub imported: usize,
    /// Records whose re-read destination content hash + count matched the
    /// source exactly.
    pub verified: usize,
    /// Damaged records listed in `skipped_damaged`.
    pub skipped_damaged: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MigratedRecord {
    pub id: String,
    /// "session" (manifest present) or "orphan" (WAV-only directory).
    pub category: String,
    pub v1_status: String,
    /// FNV-1a 64 hex over the source `recording.wav` bytes.
    pub source_wav_hash: String,
    /// FNV-1a 64 hex over the source's decoded f32 samples (little-endian
    /// bit patterns) — the domain the destination is verified in.
    pub source_sample_hash: String,
    pub source_sample_count: u64,
    pub dest_journal_hash: Option<String>,
    pub dest_sample_hash: Option<String>,
    pub dest_sample_count: Option<u64>,
    pub verified: Option<bool>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkippedRecord {
    pub id: String,
    pub reason: String,
}

/// Internal: everything needed to import one v1 record.
struct PlannedRecord {
    id: String,
    category: &'static str,
    v1_interrupted: bool,
    v1_status: String,
    created_utc: String,
    transcript: Option<TranscriptionResult>,
    transcript_history: Vec<TranscriptionResult>,
    pcm: PcmAudio,
    wav_hash: String,
    sample_hash: String,
    v1_extra: serde_json::Value,
}

struct ImportOutcome {
    journal_hash: String,
    sample_hash: String,
    sample_count: u64,
}

impl MigrationReport {
    fn for_plan(
        batch_id: &str,
        source_root: &Path,
        planned: &[PlannedRecord],
        skipped: Vec<SkippedRecord>,
        dry_run: bool,
    ) -> Self {
        let records = planned
            .iter()
            .map(|record| MigratedRecord {
                id: record.id.clone(),
                category: record.category.to_string(),
                v1_status: record.v1_status.clone(),
                source_wav_hash: record.wav_hash.clone(),
                source_sample_hash: record.sample_hash.clone(),
                source_sample_count: record.pcm.samples.len() as u64,
                dest_journal_hash: None,
                dest_sample_hash: None,
                dest_sample_count: None,
                verified: None,
                error: None,
            })
            .collect();
        let counts = MigrationCounts {
            source_sessions: planned.len(),
            source_damaged: skipped.len(),
            imported: 0,
            verified: 0,
            skipped_damaged: skipped.len(),
        };
        Self {
            schema_version: 1,
            generated_at: now_iso(),
            source_root: source_root.display().to_string(),
            dry_run,
            batch_id: batch_id.to_string(),
            counts,
            records,
            skipped_damaged: skipped,
        }
    }

    /// The path this report is written to under the store root.
    pub fn path_for_root(root: &Path) -> PathBuf {
        root.join(MIGRATION_REPORT_FILE)
    }
}

// ---------------------------------------------------------------------------
// Free helpers.
// ---------------------------------------------------------------------------

/// Whether the testing flag is on. The app reads its config/env and calls
/// [`v2_enabled_from`]; I2 ships with the flag off (v1 remains the store
/// of record — see the module docs).
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

/// FNV-1a 64 over raw bytes, hex-encoded — the house content hash.
fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The v1 manifest keys this migration understands; anything else is an
/// unknown field and lands in `extra_json` verbatim.
const KNOWN_V1_MANIFEST_FIELDS: &[&str] = &[
    "schemaVersion",
    "id",
    "createdAt",
    "updatedAt",
    "status",
    "audioFile",
    "durationMs",
    "attemptCount",
    "transcript",
    "transcriptHistory",
    "lastError",
    "journalId",
];

/// Classify a v1 store for migration. Readable records (including
/// recovered orphans — G02's classification) become [`PlannedRecord`]s
/// with source hashes; damaged records become skips with reasons.
/// Originals are only read.
fn plan_v1_import(v1_root: &Path) -> Result<(Vec<PlannedRecord>, Vec<SkippedRecord>), StoreV2Error> {
    let store = FileSessionStore::open(v1_root)?;
    let mut planned = Vec::new();
    let mut skipped = Vec::new();

    for record in store.list_records()? {
        match record {
            crate::storage::ListedRecord::Session(summary) => {
                let Some(session) = store.get(&summary.id)? else {
                    skipped.push(SkippedRecord {
                        id: summary.id.clone(),
                        reason: "record vanished while planning the migration".to_string(),
                    });
                    continue;
                };
                match plan_one(v1_root, &session) {
                    Ok(record) => planned.push(record),
                    Err(reason) => skipped.push(SkippedRecord {
                        id: session.id,
                        reason,
                    }),
                }
            }
            crate::storage::ListedRecord::Damaged(damaged) => {
                skipped.push(SkippedRecord {
                    id: damaged.id,
                    reason: damaged.reason,
                });
            }
        }
    }

    planned.sort_by(|left, right| left.id.cmp(&right.id));
    skipped.sort_by(|left, right| left.id.cmp(&right.id));
    Ok((planned, skipped))
}

/// Build one [`PlannedRecord`]: decode the WAV (the sample-domain hashes
/// are computed over its decoded samples), collect unknown manifest
/// fields.
fn plan_one(v1_root: &Path, session: &DictationSession) -> Result<PlannedRecord, String> {
    let dir = v1_root.join(&session.id);
    let wav_path = dir.join("recording.wav");
    let wav_bytes =
        std::fs::read(&wav_path).map_err(|err| format!("recording.wav: {err}"))?;
    let pcm = decode_pcm16_wav(&wav_bytes)
        .map_err(|err| format!("recording.wav does not decode: {err}"))?;

    // Unknown manifest fields → extra_json, verbatim values.
    let mut v1_extra = serde_json::Map::new();
    if let Ok(text) = std::fs::read_to_string(dir.join("manifest.json")) {
        if let Ok(serde_json::Value::Object(fields)) = serde_json::from_str::<serde_json::Value>(&text)
        {
            let unknown: serde_json::Map<String, serde_json::Value> = fields
                .into_iter()
                .filter(|(key, _)| !KNOWN_V1_MANIFEST_FIELDS.contains(&key.as_str()))
                .collect();
            if !unknown.is_empty() {
                v1_extra.insert("unknownFields".to_string(), unknown.into());
            }
        }
    }
    v1_extra.insert(
        "status".to_string(),
        serde_json::Value::String(status_to_text(session.status).to_string()),
    );
    if let Some(duration) = session.duration_ms {
        v1_extra.insert("durationMs".to_string(), serde_json::json!(duration));
    }
    v1_extra.insert("attemptCount".to_string(), serde_json::json!(session.attempt_count));
    if let Some(error) = &session.last_error {
        v1_extra.insert("lastError".to_string(), serde_json::json!(error));
    }
    if let Some(journal_id) = &session.journal_id {
        v1_extra.insert("journalId".to_string(), serde_json::json!(journal_id));
    }

    let category = if dir.join("manifest.json").exists() {
        "session"
    } else {
        "orphan"
    };
    let samples = &pcm.samples;
    Ok(PlannedRecord {
        id: session.id.clone(),
        category,
        v1_interrupted: session.status == SessionStatus::Interrupted,
        v1_status: status_to_text(session.status).to_string(),
        created_utc: session.created_at.clone(),
        transcript: session.transcript.clone(),
        transcript_history: session.transcript_history.clone(),
        sample_hash: fnv1a_hex_samples(samples),
        wav_hash: fnv1a_hex(&wav_bytes),
        pcm,
        v1_extra: serde_json::Value::Object(v1_extra),
    })
}

/// Sample-domain hash: FNV-1a 64 over the little-endian bit patterns of
/// the f32 samples — identical to what a journal seals into its trailer,
/// so source and destination compare without re-encoding WAV headers.
fn fnv1a_hex_samples(samples: &[f32]) -> String {
    format!("{:016x}", samples_hash(samples))
}

fn status_to_text(status: SessionStatus) -> &'static str {
    match status {
        SessionStatus::Captured => "captured",
        SessionStatus::Transcribing => "transcribing",
        SessionStatus::Transcribed => "transcribed",
        SessionStatus::Failed => "failed",
        SessionStatus::Interrupted => "interrupted",
    }
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
            })
            .expect("insert attempt");

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

    // ---- migration ----------------------------------------------------

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

    /// The mixed v1 fixture: good sessions (one with transcripts and
    /// history, one with unknown manifest fields), an orphan, and every
    /// damaged shape. Returns (tempdir, expected readable ids).
    fn mixed_v1_store() -> (TempDir, Vec<String>) {
        let temp = TempDir::new().expect("tempdir");
        let store = FileSessionStore::open(temp.path().join("sessions"))
            .expect("open v1 store");

        let plain = store
            .create(wav_bytes(&ramp(100, 0)), Some(100.0))
            .expect("plain session");

        let transcribed = store
            .create(wav_bytes(&ramp(150, 100)), None)
            .expect("transcribed session");
        store
            .save_transcript(
                &transcribed.id,
                TranscriptionResult {
                    text: "first pass".to_string(),
                    segments: Vec::new(),
                    duration_seconds: None,
                    request_id: None,
                },
            )
            .expect("first transcript");
        store
            .save_transcript(
                &transcribed.id,
                TranscriptionResult {
                    text: "second pass".to_string(),
                    segments: Vec::new(),
                    duration_seconds: None,
                    request_id: Some("req-2".to_string()),
                },
            )
            .expect("second transcript");

        let interrupted = store
            .create_with_journal(wav_bytes(&ramp(80, 250)), Some(80.0), Some("j_v1"))
            .expect("interrupted session");
        store
            .mark_interrupted(&interrupted.id, "torn tail of 12 bytes")
            .expect("mark interrupted");

        // Unknown manifest fields from a future v1 writer.
        let unknown = store.create(wav_bytes(&ramp(60, 330)), None).expect("unknown");
        let manifest_path = temp
            .path()
            .join("sessions")
            .join(&unknown.id)
            .join("manifest.json");
        let raw = std::fs::read_to_string(&manifest_path).expect("read manifest");
        let mut manifest: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        manifest
            .as_object_mut()
            .expect("object")
            .insert("futureField".into(), serde_json::json!({"a": 1}));
        manifest
            .as_object_mut()
            .expect("object")
            .insert("anotherUnknown".into(), serde_json::json!("kept verbatim"));
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap())
            .expect("rewrite manifest");

        // A WAV-only orphan directory.
        let orphan_dir = temp.path().join("sessions").join("orphan-take");
        std::fs::create_dir_all(&orphan_dir).expect("orphan dir");
        std::fs::write(orphan_dir.join("recording.wav"), wav_bytes(&ramp(70, 400)))
            .expect("orphan wav");

        // Damaged shapes the migration must skip, not choke on.
        let missing_wav = store.create(wav_bytes(&ramp(10, 0)), None).expect("missing wav");
        std::fs::remove_file(
            temp.path().join("sessions").join(&missing_wav.id).join("recording.wav"),
        )
        .expect("remove wav");

        let garbage_dir = temp.path().join("sessions").join("garbage-wav");
        std::fs::create_dir_all(&garbage_dir).expect("garbage dir");
        std::fs::write(garbage_dir.join("recording.wav"), b"not audio").expect("garbage wav");

        let mut ids = vec![
            plain.id.clone(),
            transcribed.id.clone(),
            interrupted.id.clone(),
            unknown.id.clone(),
            "orphan-take".to_string(),
        ];
        ids.sort();
        (temp, ids)
    }

    /// Snapshot every file under `dir` as (path, hash) for byte-identity
    /// checks.
    fn snapshot(dir: &Path) -> Vec<(PathBuf, String)> {
        let mut files = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current).expect("read dir") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let bytes = std::fs::read(&path).expect("read file");
                    files.push((path, fnv1a_hex(&bytes)));
                }
            }
        }
        files.sort();
        files
    }

    #[test]
    fn migration_dry_run_reports_without_touching_either_side() {
        let (v1, expected_ids) = mixed_v1_store();
        let before = snapshot(&v1.path().join("sessions"));
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let report = store
            .migrate_v1_dry_run(v1.path().join("sessions"))
            .expect("dry run");

        assert!(report.dry_run);
        assert_eq!(report.counts.source_sessions, 5, "4 sessions + 1 orphan");
        assert_eq!(report.counts.source_damaged, 2);
        assert_eq!(report.counts.imported, 0, "a dry run imports nothing");
        let got: Vec<String> = {
            let mut ids: Vec<String> = report.records.iter().map(|r| r.id.clone()).collect();
            ids.sort();
            ids
        };
        assert_eq!(got, expected_ids);
        let skipped: Vec<String> = report
            .skipped_damaged
            .iter()
            .map(|record| record.id.clone())
            .collect();
        assert!(skipped.contains(&"garbage-wav".to_string()));
        assert!(
            report
                .skipped_damaged
                .iter()
                .any(|record| record.reason.contains("recording.wav")),
            "damaged reasons are surfaced: {:?}",
            report.skipped_damaged
        );
        // Orphan vs session categorization.
        let orphan = report
            .records
            .iter()
            .find(|record| record.id == "orphan-take")
            .expect("orphan planned");
        assert_eq!(orphan.category, "orphan");
        assert_eq!(orphan.v1_status, "interrupted");

        // Nothing entered v2…
        assert_eq!(store.list_records(0, 10).expect("list").total, 0);
        assert!(journal_ids_in(&store.root.join(AUDIO_DIR)).is_empty());
        // …but the report file exists for review.
        assert!(MigrationReport::path_for_root(&store.root).exists());

        // …and the v1 originals are byte-identical.
        assert_eq!(snapshot(&v1.path().join("sessions")), before);
    }

    #[test]
    fn migration_apply_verifies_hashes_and_preserves_originals() {
        let (v1, _) = mixed_v1_store();
        let sessions = v1.path().join("sessions");
        let before = snapshot(&sessions);
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let report = store.migrate_v1_apply(&sessions).expect("apply");
        assert!(!report.dry_run);
        assert_eq!(report.counts.imported, 5);
        assert_eq!(report.counts.verified, 5, "every record verified");
        assert!(report.records.iter().all(|record| record.error.is_none()));
        for record in &report.records {
            assert_eq!(record.verified, Some(true), "{:?}", record.id);
            assert_eq!(
                record.dest_sample_hash.as_deref(),
                Some(record.source_sample_hash.as_str()),
                "sample-domain hashes match for {}",
                record.id
            );
            assert_eq!(record.dest_sample_count, Some(record.source_sample_count));
            assert!(!record.source_wav_hash.is_empty());
        }

        // The v1 originals were only read.
        assert_eq!(snapshot(&sessions), before, "originals byte-identical");

        // v2 now lists every imported capture, metadata-only.
        let page = store.list_records(0, 10).expect("list");
        assert_eq!(page.total, 5);
        for record in &page.records {
            match record {
                ListedCapture::Capture(listing) => {
                    assert!(listing.problems.is_empty());
                    // Unknown manifest fields survived in extra_json.
                    let extra: serde_json::Value = serde_json::from_str(
                        listing.record.extra_json.as_deref().expect("extra"),
                    )
                    .expect("parse extra");
                    assert_eq!(extra["importedFrom"], "file-v1");
                    assert_eq!(extra["importBatch"], report.batch_id);
                    if listing.record.id == "orphan-take" {
                        assert_eq!(listing.record.status, CaptureStatus::Interrupted);
                        assert_eq!(extra["v1"]["status"], "interrupted");
                    }
                }
                other => panic!("imported records must be clean: {other:?}"),
            }
        }

        // The session with unknown fields carried them verbatim.
        let unknown_row = {
            let mut found = None;
            for record in &page.records {
                if let ListedCapture::Capture(listing) = record {
                    let extra: serde_json::Value = serde_json::from_str(
                        listing.record.extra_json.as_deref().expect("extra"),
                    )
                    .expect("parse");
                    if extra["v1"]["unknownFields"]["futureField"]["a"] == serde_json::json!(1) {
                        found = Some(listing.record.id.clone());
                    }
                }
            }
            found.expect("unknown fields landed in extra_json")
        };
        assert!(!unknown_row.is_empty());

        // The transcribed session's history became attempts (oldest first).
        let with_attempts = page
            .records
            .iter()
            .filter_map(|record| match record {
                ListedCapture::Capture(listing) => Some(listing.record.id.clone()),
                _ => None,
            })
            .find(|id| store.attempts_for(id).expect("attempts").len() == 2)
            .expect("history + current transcript imported");
        let attempts = store.attempts_for(&with_attempts).expect("attempts");
        assert_eq!(attempts[0].text, "first pass");
        assert_eq!(attempts[1].text, "second pass");
        assert_eq!(attempts[1].partial_or_final, "final");
        let extra: serde_json::Value =
            serde_json::from_str(attempts[1].extra_json.as_deref().expect("extra")).expect("parse");
        assert_eq!(extra["requestId"], "req-2");

        // Imported audio loads lazily and is bit-identical to the source
        // samples in the sample domain.
        let audio = store.load_audio(&unknown_row).expect("load imported audio");
        assert!(audio.finalized);
        let expected = decode_pcm16_wav(
            &std::fs::read(sessions.join(&unknown_row).join("recording.wav")).expect("source wav"),
        )
        .expect("decode source");
        assert_eq!(audio.samples, expected.samples);

        // The verified report is on disk for review, with the batch id.
        let raw =
            std::fs::read_to_string(MigrationReport::path_for_root(&store.root)).expect("report");
        let on_disk: MigrationReport = serde_json::from_str(&raw).expect("parse report");
        assert!(!on_disk.dry_run);
        assert_eq!(on_disk.batch_id, report.batch_id);
        assert_eq!(on_disk.counts.verified, 5);

        // A rerun without a rollback refuses to overwrite (no double import).
        let second = store.migrate_v1_apply(&sessions).expect("rerun report");
        assert_eq!(second.counts.imported, 0);
        assert!(second.records.iter().all(|record| record.error.is_some()));
        assert_eq!(store.list_records(0, 10).expect("list").total, 5);
    }

    #[test]
    fn migration_rollback_discards_the_import_and_stays_dead() {
        let (v1, _) = mixed_v1_store();
        let sessions = v1.path().join("sessions");
        let before = snapshot(&sessions);
        let dir = TempDir::new().expect("tempdir");
        let mut store = store_in(&dir);

        let report = store.migrate_v1_apply(&sessions).expect("apply");
        assert_eq!(store.list_records(0, 10).expect("list").total, 5);

        let discarded = store.rollback_import(&report.batch_id).expect("rollback");
        assert_eq!(discarded, 5);
        assert_eq!(store.list_records(0, 10).expect("list").total, 0);
        // The imported journals were quarantined, not unlinked…
        assert_eq!(journal_ids_in(&store.root.join(QUARANTINE_DIR)).len(), 5);
        assert!(journal_ids_in(&store.root.join(AUDIO_DIR)).is_empty());
        // …and reconciliation treats them as tombstoned: no resurrection.
        let rerun = store.reconcile().expect("reconcile");
        assert!(rerun.orphan_sessions.is_empty());
        assert_eq!(store.list_records(0, 10).expect("list").total, 0);
        // The meta entry for the batch is gone.
        assert!(store.meta_get(&format!("migration:{}", report.batch_id)).expect("meta").is_none());

        // Originals still byte-identical, and a fresh import works again.
        assert_eq!(snapshot(&sessions), before);
        let again = store.migrate_v1_apply(&sessions).expect("re-import");
        assert_eq!(again.counts.verified, 5);
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
}
