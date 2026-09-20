//! Durable capture journals (I1 phase 2, `docs/program/design/e17-native-runtime.md`
//! §3 + the journal pieces of §4).
//!
//! Every take is mirrored to an append-only `<take-id>.sj` file under
//! `<data-root>/starling-gpui/journals/` while it is being captured. The
//! journal — not the in-memory accumulation — is the authoritative retained
//! audio: the writer task appends converted mono samples as frame records and,
//! on the §3 cadence (every ≤250 ms or ≤64 KiB, whichever comes first), writes
//! a boundary record and fsyncs. Only samples covered by an fsynced boundary
//! are "acknowledged" ([`crate::recorder::RecorderHandle::acknowledged_samples`]);
//! a plain `write` is never acknowledgment.
//!
//! File format (version 1, all integers little-endian):
//!
//! ```text
//! header:  "STRLNGSJ" (8 bytes magic) | u8 version = 1 | u32 sample rate
//! record:  u8 tag
//!   0x01 FRAME:    u32 sample count | count * 4 bytes of f32 LE payload
//!   0x02 BOUNDARY: u64 cumulative sample count | u64 FNV-1a hash of all
//!                  frame payload bytes so far
//!   0x03 TRAILER:  u64 total sample count | u64 FNV-1a hash of all frame
//!                  payload bytes (must be the final record)
//! ```
//!
//! Crash protocol (§4, journal-only portion):
//! - Frames and boundaries are appended and fsynced on the cadence;
//!   acknowledged = covered by an fsynced boundary.
//! - Clean stop: the trailer (length + content hash) is written, the file is
//!   fsynced and the parent directory fsynced (POSIX; best-effort elsewhere).
//!   The take then hands its audio to the existing session-store path as
//!   today, with the manifest recording the additive `journal_id` linkage.
//! - Startup recovery: a journal without a valid trailer is an interrupted
//!   take. It is recovered to its last checksum-valid boundary — the torn
//!   tail is discarded *logically* (the verified prefix is what becomes the
//!   recovered session's audio) and the gap is recorded in the session note.
//!   The journal file itself is never modified or deleted: it stays in place
//!   as source evidence.
//!
//! RETENTION POLICY (frozen for I1 phase 2, revised by R21): journal files
//! are never deleted by the crash-recovery path — not after the session is
//! saved, not after recovery; that never-delete rule is the *crash* policy.
//! The one exception is an explicit user deletion: a session deleted through
//! the confirmed-delete flow (B05's "permanently removes the audio" warning)
//! takes its linked journal with it — [`delete_session_and_journal`] renames
//! the journal into `journals/deleted/` *before* removing the session row,
//! so the rename is the tombstone commit point and startup recovery can
//! never resurrect a recording the user watched being deleted. Quarantined
//! files stay on disk awaiting the I2 retention sweep (no eager unlink).
//! Journals already linked to a saved session (manifest `journal_id`) are
//! skipped by recovery so a take can never be recovered twice.
//!
//! Deferred to I2/I3 (known gaps, by design of this increment): gap records
//! inside the journal (ring-overflow gaps are flagged live via
//! [`crate::recorder::RecorderHandle::gaps`] but are not serialized into the
//! journal), the `staging/` → `audio/` rename protocol, the SQLite metadata
//! transaction, bounded/streaming journal reads, and journal GC.

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::audio::{PcmAudio, encode_wav_16k};
use crate::storage::{DictationSession, FileSessionStore, StorageError};

/// Extension of journal files under the journals root.
const JOURNAL_EXT: &str = "sj";

/// Subdirectory of the journals root that holds deliberately-deleted
/// journals (R21): a confirmed session deletion renames its linked journal
/// into here instead of unlinking it. The rename is the tombstone — a journal
/// id with a file under `deleted/` is dead forever as far as recovery is
/// concerned; the bytes themselves await the I2 retention sweep.
const DELETED_DIR: &str = "deleted";

/// Format magic at the start of every journal file.
const MAGIC: &[u8; 8] = b"STRLNGSJ";

/// Current on-disk format version (one byte after the magic).
const FORMAT_VERSION: u8 = 1;

/// Header size: magic (8) + version (1) + sample rate (4).
const HEADER_LEN: usize = 13;

/// Record tags.
const TAG_FRAME: u8 = 0x01;
const TAG_BOUNDARY: u8 = 0x02;
const TAG_TRAILER: u8 = 0x03;

/// Boundary/fsync cadence from §3: fsync at least every 250 ms …
const JOURNAL_BOUNDARY_INTERVAL: Duration = Duration::from_millis(250);
/// … or every 64 KiB of appended payload, whichever comes first (tunable).
const JOURNAL_BOUNDARY_BYTES: u64 = 64 * 1024;

/// FNV-1a 64-bit offset basis and prime (dependency-free content hash).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The raw append/sync surface a [`JournalWriter`] owns. Split out so tests
/// can inject faults behind the exact code path production uses.
pub(crate) trait JournalSink: Send {
    /// Appends bytes at the end of the journal.
    fn append(&mut self, bytes: &[u8]) -> io::Result<()>;
    /// Flushes the file's kernel state to disk (`fsync`).
    fn sync(&mut self) -> io::Result<()>;
    /// Makes the journal's directory entry durable (POSIX dir fsync;
    /// best-effort no-op on platforms without it).
    fn sync_parent_dir(&mut self) -> io::Result<()>;
}

/// Real-file sink. The journal file is created with `create_new` so an
/// existing file — or a symlink planted at the name — can never be
/// overwritten: a journal is source evidence, there is deliberately no
/// truncate path.
pub(crate) struct FileSink {
    file: std::fs::File,
    dir: PathBuf,
}

impl FileSink {
    /// Opens a fresh journal file named `<id>.sj` under `dir`.
    fn create(dir: &Path, id: &str) -> io::Result<(Self, PathBuf)> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{id}.{JOURNAL_EXT}"));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok((Self { file, dir: dir.to_path_buf() }, path))
    }
}

impl JournalSink for FileSink {
    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        use std::io::Write;
        self.file.write_all(bytes)
    }

    fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }

    fn sync_parent_dir(&mut self) -> io::Result<()> {
        #[cfg(unix)]
        {
            // POSIX: fsync the directory so the dirent itself survives.
            let dir = std::fs::File::open(&self.dir)?;
            dir.sync_all()?;
        }
        #[cfg(not(unix))]
        {
            // Best-effort elsewhere: the file fsync above is all we can do.
            let _ = &self.dir;
        }
        Ok(())
    }
}

/// The per-take journal writer, owned by the capture writer task. Tracks the
/// cumulative sample count and running content hash so every boundary and
/// the trailer are verifiable on recovery.
pub(crate) struct JournalWriter<S: JournalSink> {
    sink: S,
    id: String,
    path: PathBuf,
    sample_rate: u32,
    total_samples: u64,
    hash: u64,
    bytes_since_boundary: u64,
    last_boundary: Instant,
}

impl<S: JournalSink> JournalWriter<S> {
    /// Builds a writer over an already-opened sink and durably writes the
    /// header (fsync file + parent dir), so a crash immediately after start
    /// leaves at most an empty-but-valid journal — which recovery skips
    /// without creating a session. (Crate-visible for fault-injection
    /// tests that wrap the sink.)
    pub(crate) fn over_sink(
        sink: S,
        id: String,
        path: PathBuf,
        sample_rate: u32,
    ) -> io::Result<Self> {
        let mut writer = Self {
            sink,
            id,
            path,
            sample_rate,
            total_samples: 0,
            hash: FNV_OFFSET,
            bytes_since_boundary: 0,
            last_boundary: Instant::now(),
        };
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.push(FORMAT_VERSION);
        header.extend_from_slice(&sample_rate.to_le_bytes());
        writer.sink.append(&header)?;
        writer.sink.sync()?;
        writer.sink.sync_parent_dir()?;
        Ok(writer)
    }

    /// This journal's id (the `.sj` file stem; recorded in the session
    /// manifest as the additive `journal_id` linkage).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// This journal's file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The device sample rate recorded in the header.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Cumulative sample count appended so far (storage v2, I2: the
    /// `captures.frame_count` source).
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// The running content hash over everything appended so far (storage
    /// v2, I2: the `captures.journal_hash` source — after `finalize` this
    /// is exactly the value sealed into the trailer).
    pub(crate) fn content_hash(&self) -> u64 {
        self.hash
    }

    /// Appends `samples` as one frame record, updating the running content
    /// hash. This is a plain write: the samples become acknowledged only
    /// when a later boundary record is fsynced.
    pub fn append_frames(&mut self, samples: &[f32]) -> io::Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        if samples.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal frame record exceeds the u32 sample count field",
            ));
        }
        let mut record = Vec::with_capacity(5 + samples.len() * 4);
        record.push(TAG_FRAME);
        record.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        let payload_start = record.len();
        record.reserve(samples.len() * 4);
        for &sample in samples {
            record.extend_from_slice(&sample.to_bits().to_le_bytes());
        }
        self.hash = fnv1a(self.hash, &record[payload_start..]);
        self.sink.append(&record)?;
        self.total_samples += samples.len() as u64;
        self.bytes_since_boundary += (record.len() - payload_start) as u64;
        Ok(())
    }

    /// Whether the §3 cadence says a boundary is due.
    pub fn boundary_due(&self) -> bool {
        // The time arm only applies when there are unsynced samples to
        // acknowledge: an idle journal (nothing appended since the last
        // boundary) is never due, however much clock passes — otherwise a
        // slow runner between writer creation and the first append would
        // report due-on-time with zero bytes to confirm.
        self.bytes_since_boundary >= JOURNAL_BOUNDARY_BYTES
            || (self.bytes_since_boundary > 0
                && self.last_boundary.elapsed() >= JOURNAL_BOUNDARY_INTERVAL)
    }

    /// Writes a boundary record covering everything appended so far and
    /// fsyncs. Returns the acknowledged sample count (the cumulative count
    /// this boundary confirms). With nothing new appended it is a clock-
    /// resetting no-op returning the current count.
    pub fn write_boundary(&mut self) -> io::Result<u64> {
        if self.bytes_since_boundary == 0 {
            self.last_boundary = Instant::now();
            return Ok(self.total_samples);
        }
        let mut record = Vec::with_capacity(17);
        record.push(TAG_BOUNDARY);
        record.extend_from_slice(&self.total_samples.to_le_bytes());
        record.extend_from_slice(&self.hash.to_le_bytes());
        self.sink.append(&record)?;
        self.sink.sync()?;
        self.bytes_since_boundary = 0;
        self.last_boundary = Instant::now();
        Ok(self.total_samples)
    }

    /// Finalizes the journal on a clean stop (§4 step 2): the trailer
    /// (length + content hash) validating everything appended so far, then
    /// fsync of the file and the parent directory. Returns the
    /// acknowledged sample count the trailer confirms. A failed finalize
    /// leaves the journal trailer-less — the startup scan then recovers it
    /// as an interrupted take to its last valid boundary, so no
    /// acknowledged audio is lost either way. (The trailer alone is the
    /// final word: the parser treats it like a boundary that must also be
    /// the file's last record, so no separate final boundary is needed.)
    pub fn finalize(&mut self) -> io::Result<u64> {
        let mut record = Vec::with_capacity(17);
        record.push(TAG_TRAILER);
        record.extend_from_slice(&self.total_samples.to_le_bytes());
        record.extend_from_slice(&self.hash.to_le_bytes());
        self.sink.append(&record)?;
        self.sink.sync()?;
        self.sink.sync_parent_dir()?;
        Ok(self.total_samples)
    }
}

impl JournalWriter<FileSink> {
    /// Creates a fresh journal for a new take under `dir`.
    pub fn create(dir: &Path, sample_rate: u32) -> io::Result<Self> {
        let id = format!("j_{}", uuid::Uuid::new_v4().simple());
        Self::create_named(dir, id, sample_rate)
    }

    /// [`Self::create`] with a caller-chosen id (storage v2, I2): the v2
    /// crash protocol stages a journal named after its capture id under
    /// `staging/` and renames it into `audio/`, so the file stem is the
    /// capture id rather than a generated `j_<uuid>`. Crate-visible: the id
    /// becomes a path component, so callers must pass a safe one (the v2
    /// store only ever passes its generated `c_<uuid>` ids).
    pub(crate) fn create_named(
        dir: &Path,
        id: String,
        sample_rate: u32,
    ) -> io::Result<Self> {
        let (sink, path) = FileSink::create(dir, &id)?;
        Self::over_sink(sink, id, path, sample_rate)
    }
}

/// Why a journal file could not be parsed at all.
#[derive(Debug, thiserror::Error)]
pub enum JournalReadError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("not a Starling capture journal: {0}")]
    NotAJournal(String),
}

/// A parsed-and-verified journal: only checksum-confirmed samples are kept.
#[derive(Debug)]
pub(crate) struct ParsedJournal {
    /// Device sample rate from the header.
    pub sample_rate: u32,
    /// Verified samples: everything covered by the last valid boundary, or
    /// by a valid trailer for a finalized journal. The torn tail is excluded.
    pub samples: Vec<f32>,
    /// File length past the last valid boundary (discarded as torn tail).
    pub torn_tail_bytes: u64,
    /// True when a checksum-valid trailer is the final record.
    pub finalized: bool,
    /// How many valid boundary records the journal contains.
    #[cfg_attr(not(test), allow(dead_code))]
    pub boundary_count: usize,
}

/// Parses and verifies a journal file. Recovery rules (§4): frames are
/// accumulated and hashed; a boundary counts only when its cumulative count
/// and hash match the frames before it; the verified prefix ends at the last
/// valid boundary (or at a valid final trailer). Anything after that —
/// partial records, a hash mismatch from corruption, bytes past the trailer —
/// is torn tail. The file is read whole; bounded reads are I2 work.
pub(crate) fn read_journal(path: &Path) -> Result<ParsedJournal, JournalReadError> {
    let data = std::fs::read(path)?;

    if data.len() < HEADER_LEN || &data[..MAGIC.len()] != MAGIC {
        return Err(JournalReadError::NotAJournal(format!(
            "{} does not start with the v1 header",
            path.display()
        )));
    }
    if data[8] != FORMAT_VERSION {
        return Err(JournalReadError::NotAJournal(format!(
            "unsupported journal format version {}",
            data[8]
        )));
    }
    let sample_rate = u32::from_le_bytes([data[9], data[10], data[11], data[12]]);

    let mut samples: Vec<f32> = Vec::new();
    let mut hash = FNV_OFFSET;
    let mut verified_len = 0usize;
    let mut verified_end = HEADER_LEN as u64;
    let mut boundary_count = 0usize;
    let mut finalized = false;

    let mut pos = HEADER_LEN;
    'parse: while pos < data.len() {
        let remaining = data.len() - pos;
        match data[pos] {
            TAG_FRAME => {
                if remaining < 5 {
                    break 'parse;
                }
                let count = u32::from_le_bytes([
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                ]) as usize;
                let payload_len = count.saturating_mul(4);
                // Also guards absurd counts from garbage: a record larger
                // than the bytes actually present is a torn write.
                if remaining < 5 + payload_len {
                    break 'parse;
                }
                let payload = &data[pos + 5..pos + 5 + payload_len];
                hash = fnv1a(hash, payload);
                samples.reserve(count);
                for chunk in payload.chunks_exact(4) {
                    let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    samples.push(f32::from_bits(bits));
                }
                pos += 5 + payload_len;
            }
            TAG_BOUNDARY | TAG_TRAILER => {
                if remaining < 17 {
                    break 'parse;
                }
                let count = u64::from_le_bytes(
                    data[pos + 1..pos + 9]
                        .try_into()
                        .expect("8 bytes parse as u64"),
                );
                let recorded_hash = u64::from_le_bytes(
                    data[pos + 9..pos + 17]
                        .try_into()
                        .expect("8 bytes parse as u64"),
                );
                if count != samples.len() as u64 || recorded_hash != hash {
                    break 'parse;
                }
                if data[pos] == TAG_BOUNDARY {
                    boundary_count += 1;
                    verified_len = samples.len();
                    verified_end = (pos + 17) as u64;
                    pos += 17;
                } else {
                    // Trailer: its checksum is a valid verification point
                    // for everything before it even when bytes follow, but
                    // "finalized" additionally requires it to be the file's
                    // last record. Never parse past a trailer — a journal
                    // appended after its own trailer is malformed, and the
                    // verified prefix ends at the trailer either way.
                    if pos + 17 == data.len() {
                        finalized = true;
                    }
                    verified_len = samples.len();
                    verified_end = (pos + 17) as u64;
                    break 'parse;
                }
            }
            _ => break 'parse,
        }
    }

    samples.truncate(verified_len);
    Ok(ParsedJournal {
        sample_rate,
        samples,
        torn_tail_bytes: data.len() as u64 - verified_end,
        finalized,
        boundary_count,
    })
}

/// The content hash a writer would hold after appending exactly `samples`
/// (FNV-1a over the little-endian f32 payload bytes, in order). Recovery
/// sealing (storage v2, I2) uses this to write a checksum-valid trailer for
/// a verified prefix without re-reading frame records.
pub(crate) fn samples_hash(samples: &[f32]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &sample in samples {
        hash = fnv1a(hash, &sample.to_bits().to_le_bytes());
    }
    hash
}

/// Seal a recovered journal (storage v2, I2): physically truncate the file
/// to the end of its last checksum-valid verification point (boundary or
/// trailer), append a trailer covering the verified samples, and fsync —
/// leaving a finalized journal whose every byte is covered by the trailer's
/// length + hash. The v1 recovery path never modifies journal files (they
/// are source evidence for the v1 store); v2 calls this only on its own
/// staging journals during reconciliation, where §4 specifies "truncate +
/// gap flag" and the discarded tail is reported to the caller so the gap can
/// be flagged on the resulting captures row. Idempotent: sealing an already
/// sealed journal truncates nothing and rewrites an equivalent trailer.
pub(crate) fn seal_recovered_journal(path: &Path, parsed: &ParsedJournal) -> io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};

    let file_len = std::fs::metadata(path)?.len();
    let verified_end = file_len - parsed.torn_tail_bytes;
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.set_len(verified_end)?;
    file.seek(SeekFrom::Start(verified_end))?;

    let mut trailer = Vec::with_capacity(17);
    trailer.push(TAG_TRAILER);
    trailer.extend_from_slice(&(parsed.samples.len() as u64).to_le_bytes());
    trailer.extend_from_slice(&samples_hash(&parsed.samples).to_le_bytes());
    file.write_all(&trailer)?;
    file.sync_all()
}

/// Whether `header` (at least [`HEADER_LEN`] bytes) is a journal this
/// build understands: the v1 magic + format version. Storage v2 uses this
/// for its bounded per-record audio check (I2).
pub(crate) fn is_journal_header(header: &[u8]) -> bool {
    header.len() >= HEADER_LEN
        && &header[..MAGIC.len()] == MAGIC
        && header[MAGIC.len()] == FORMAT_VERSION
}

/// The journals root: `<data-root>/starling-gpui/journals/`, a sibling of
/// the session store's `sessions/` directory.
pub fn default_journals_root() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("starling-gpui")
        .join("journals")
}

/// A manifest-provided journal id must obey the same path-safety rules as a
/// session id before it becomes a filesystem path component. Generated ids
/// (`j_<uuid>`) always pass; [`FileSessionStore::journal_id_of`] already
/// drops unsafe linkages, so this is defense in depth for direct callers.
fn validate_journal_id(id: &str) -> Result<(), StorageError> {
    if !crate::storage::is_safe_path_component(id) {
        return Err(StorageError::Invalid(format!(
            "journal id {id:?} must be non-empty and contain no path separators"
        )));
    }

    Ok(())
}

/// fsync a directory's own entry (POSIX; best-effort no-op elsewhere) so a
/// rename inside it survives a crash. Same contract as
/// [`JournalSink::sync_parent_dir`], standalone for the quarantine path and
/// the storage-v2 staging→audio promotion (I2).
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

/// Quarantine (tombstone) one journal (R21): rename
/// `<journals_root>/<id>.sj` into `<journals_root>/deleted/<id>.sj`, fsyncing
/// both directories. Returns whether a live journal was quarantined — a
/// missing source file is not an error (nothing live to tombstone, e.g. the
/// journal never existed or an earlier delete already took it).
///
/// Why rename rather than unlink: one `rename(2)` atomically removes the
/// journal from the recovery scan *and* records the tombstone, so there is no
/// crash window with a removed journal and no tombstone (the resurrection
/// bug), and no window with a tombstone but a still-live journal. The bytes
/// survive for the I2 retention sweep, matching the storage-v2 direction
/// (`quarantine/` + `tombstones`), and an unlink-before-ack failure mode can
/// never silently break the crash-recovery never-delete rule.
///
/// Any other failure is returned: callers must treat it as "deletion did not
/// happen" and abort before removing the session row.
pub fn quarantine_journal(
    journals_root: &Path,
    journal_id: &str,
) -> Result<bool, StorageError> {
    validate_journal_id(journal_id)?;
    let source = journals_root.join(format!("{journal_id}.{JOURNAL_EXT}"));
    if !source.exists() {
        return Ok(false);
    }

    let quarantine_dir = journals_root.join(DELETED_DIR);
    std::fs::create_dir_all(&quarantine_dir)?;
    let destination = quarantine_dir.join(format!("{journal_id}.{JOURNAL_EXT}"));
    // A stale tombstone under the same id (the journal reappeared live — a
    // restored backup) is replaced: on Windows `rename` refuses to overwrite.
    // A crash in that window leaves the journal live and its session row in
    // place, i.e. deletion simply did not happen — never a resurrection.
    let _ = std::fs::remove_file(&destination);
    std::fs::rename(&source, &destination)?;
    sync_dir(journals_root)?;
    sync_dir(&quarantine_dir)?;
    Ok(true)
}

/// Journal ids tombstoned under `journals/deleted/` (R21): deliberately
/// deleted through a confirmed session deletion, awaiting the I2 retention
/// sweep. An unreadable or missing `deleted/` directory simply holds no
/// tombstones.
fn tombstoned_journal_ids(journals_root: &Path) -> HashSet<String> {
    let Ok(entries) = std::fs::read_dir(journals_root.join(DELETED_DIR)) else {
        return HashSet::new();
    };

    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(JOURNAL_EXT))
        .filter_map(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .collect()
}

/// Confirmed session deletion (R21, the B05 contract: "This permanently
/// removes the audio … no undo"): remove the session row and its retained
/// audio, and take the manifest-linked journal along with it.
///
/// Ordering — tombstone strictly before row removal:
/// 1. Read the session's `journal_id` (metadata-only; never loads the WAV).
/// 2. If the manifest carries a linkage, [`quarantine_journal`] renames the
///    journal into `journals/deleted/` and fsyncs. This is the commit point.
/// 3. Only then does [`FileSessionStore::delete`] remove the row + audio.
///
/// Crash matrix: a crash before step 2 leaves everything in place (the
/// journal stays linked — no resurrection); a crash between 2 and 3 leaves a
/// live row whose journal is quarantined (recovery sees a linked id with no
/// journal file — nothing to recover; the retry delete tolerates the missing
/// source); a crash after 3 is complete. A failure in step 2 aborts before
/// the row removal, so the deletion surfaces as an error with the session
/// intact rather than leaving a live journal that recovery would resurrect.
///
/// Deleting a session without a journal linkage is exactly the old
/// [`FileSessionStore::delete`] behavior — nothing under the journals root is
/// touched.
pub fn delete_session_and_journal(
    store: &FileSessionStore,
    journals_root: &Path,
    session_id: &str,
) -> Result<(), StorageError> {
    if let Some(journal_id) = store.journal_id_of(session_id)? {
        quarantine_journal(journals_root, &journal_id)?;
    }
    store.delete(session_id)
}

/// What the startup scan found and did.
#[derive(Debug, Default)]
pub struct RecoveryReport {
    /// Sessions created from unlinked journals (status `interrupted`).
    pub recovered: Vec<DictationSession>,
    /// Journals with zero verified samples: kept in place, no session made.
    pub empty_journals: Vec<String>,
    /// Files that could not be parsed: `(name, reason)`. Kept in place.
    pub unreadable: Vec<(String, String)>,
}

impl RecoveryReport {
    /// Whether anything user-facing happened (sessions recovered or files
    /// that could not be read).
    pub fn has_findings(&self) -> bool {
        !self.recovered.is_empty() || !self.unreadable.is_empty()
    }

    /// One-line summary for the app's error banner.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.recovered.is_empty() {
            parts.push(format!(
                "recovered {} interrupted recording{} into history",
                self.recovered.len(),
                if self.recovered.len() == 1 { "" } else { "s" }
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

/// The note recorded on a recovered session, stating exactly what survived.
fn recovery_note(parsed: &ParsedJournal) -> String {
    let seconds = parsed.samples.len() as f64 / parsed.sample_rate as f64;
    if parsed.finalized {
        format!(
            "This recording finished cleanly but was never saved to history — the app closed \
             between finalizing the journal and saving the session. {seconds:.1}s of \
             checksum-verified audio were recovered from the journal."
        )
    } else if parsed.torn_tail_bytes > 0 {
        format!(
            "The app closed unexpectedly during this recording: the last {} bytes of the \
             journal were an unfinished write and were discarded. {seconds:.1}s of \
             checksum-verified audio were recovered.",
            parsed.torn_tail_bytes
        )
    } else {
        format!(
            "The app closed unexpectedly before this recording was finalized. {seconds:.1}s \
             of checksum-verified audio were recovered from the journal."
        )
    }
}

/// Startup scan (§4 recovery, journal-only): every `*.sj` under
/// `journals_dir` that is not already linked to a saved session (manifest
/// `journal_id`) and not tombstoned under `journals/deleted/` is recovered —
/// its verified samples become a new session with status `interrupted`,
/// duration computed from the verified sample count at the device rate, and
/// the journal linkage recorded. Journal files are never modified or deleted
/// by recovery; they remain as source evidence.
///
/// A tombstoned id is skipped permanently (R21): the user confirmed its
/// deletion once, so even a journal file that later reappears live under
/// `journals/` (a restored backup, a copied disk) must not resurrect into an
/// interrupted take that contradicts the deletion the user watched.
///
/// A storage failure while saving a recovery aborts the scan with the error
/// (the remaining journals simply stay unlinked and are retried on the next
/// startup) rather than pretending the recovery succeeded.
pub fn recover_interrupted_takes(
    store: &FileSessionStore,
    journals_dir: &Path,
) -> Result<RecoveryReport, StorageError> {
    let mut report = RecoveryReport::default();

    let entries = match std::fs::read_dir(journals_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(report),
        Err(err) => {
            report
                .unreadable
                .push((journals_dir.display().to_string(), err.to_string()));
            return Ok(report);
        }
    };

    // Metadata-only linkage scan: reads manifests, never loads audio (the
    // G02 direction; `list()` would eagerly load every WAV).
    let linked = store.journal_ids()?;
    // Deliberately-deleted ids outrank everything: check the tombstones
    // before deciding a journal is an interrupted take.
    let tombstoned = tombstoned_journal_ids(journals_dir);

    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(JOURNAL_EXT))
        .collect();
    paths.sort(); // deterministic recovery order

    for path in paths {
        let Some(id) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let id = id.to_string();
        if tombstoned.contains(&id) {
            // Deliberately deleted (R21): the tombstone outlives the journal
            // file itself. Never resurrect a confirmed deletion.
            continue;
        }
        if linked.contains(&id) {
            // Already saved once — never recover a take twice.
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
            // A journal that never reached its first boundary (e.g. a crash
            // right after start). No audio to recover; keep the file.
            report.empty_journals.push(id);
            continue;
        }
        if parsed.sample_rate == 0 {
            report
                .unreadable
                .push((id, "journal header has a zero sample rate".to_string()));
            continue;
        }

        // Duration from the verified sample count at the device rate — not
        // from the resampled WAV length, not from wall clock.
        let duration_ms = parsed.samples.len() as f64 * 1000.0 / parsed.sample_rate as f64;
        let note = recovery_note(&parsed);
        let pcm = PcmAudio {
            samples: parsed.samples,
            sample_rate: parsed.sample_rate,
            channels: 1,
        };
        let wav = match encode_wav_16k(&pcm) {
            Ok(wav) => wav,
            Err(err) => {
                report.unreadable.push((
                    id,
                    format!("could not encode the recovered audio: {err}"),
                ));
                continue;
            }
        };

        let session = store.create_with_journal(wav, Some(duration_ms), Some(&id))?;
        let session = store.mark_interrupted(&session.id, note)?;
        report.recovered.push(session);
    }

    Ok(report)
}

#[cfg(test)]
pub(crate) mod testing {
    //! Test-only fault injection behind the production sink.

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::{FileSink, JournalSink};

    /// A sink that fails appends and/or syncs once the shared flags are
    /// set. `fail_appends`-only reproduces a dying disk; sync-only faults
    /// prove acknowledgment depends on the fsync returning, not on the
    /// write reaching the page cache.
    pub(crate) struct FaultSink {
        inner: FileSink,
        fail_appends: Arc<AtomicBool>,
        fail_syncs: Arc<AtomicBool>,
    }

    impl FaultSink {
        /// Fails everything (appends, syncs, dir fsyncs) once `fail` is set.
        pub fn create(
            dir: &std::path::Path,
            id: &str,
            fail: Arc<AtomicBool>,
        ) -> std::io::Result<(Self, std::path::PathBuf)> {
            let (inner, path) = FileSink::create(dir, id)?;
            Ok((
                Self {
                    inner,
                    fail_appends: Arc::clone(&fail),
                    fail_syncs: fail,
                },
                path,
            ))
        }

        /// Fails only `sync` (and the dir fsync) once `fail` is set;
        /// appends keep succeeding.
        pub fn create_sync_faults_only(
            dir: &std::path::Path,
            id: &str,
            fail: Arc<AtomicBool>,
        ) -> std::io::Result<(Self, std::path::PathBuf)> {
            let (inner, path) = FileSink::create(dir, id)?;
            Ok((
                Self {
                    inner,
                    fail_appends: Arc::new(AtomicBool::new(false)),
                    fail_syncs: fail,
                },
                path,
            ))
        }
    }

    impl JournalSink for FaultSink {
        fn append(&mut self, bytes: &[u8]) -> std::io::Result<()> {
            if self.fail_appends.load(Ordering::Relaxed) {
                return Err(std::io::Error::other("injected journal append fault"));
            }
            self.inner.append(bytes)
        }

        fn sync(&mut self) -> std::io::Result<()> {
            if self.fail_syncs.load(Ordering::Relaxed) {
                return Err(std::io::Error::other("injected journal sync fault"));
            }
            self.inner.sync()
        }

        fn sync_parent_dir(&mut self) -> std::io::Result<()> {
            if self.fail_syncs.load(Ordering::Relaxed) {
                return Err(std::io::Error::other("injected journal dir fault"));
            }
            self.inner.sync_parent_dir()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::testing::FaultSink;
    use super::*;
    use crate::audio::decode_pcm16_wav;
    use crate::storage::SessionStatus;
    use tempfile::TempDir;

    /// Build a real writer over a tempdir.
    fn writer_in(dir: &TempDir, rate: u32) -> JournalWriter<FileSink> {
        JournalWriter::create(dir.path(), rate).expect("create journal writer")
    }

    /// Deterministic sample block `len` long.
    fn ramp(len: usize, offset: u32) -> Vec<f32> {
        (0..len)
            .map(|i| ((offset as usize + i) % 997) as f32 * 0.0001)
            .collect()
    }

    /// Offset just past the first boundary record of `bytes` (fixture
    /// arithmetic for exact tail-size assertions).
    fn first_boundary_end(bytes: &[u8]) -> usize {
        let mut pos = HEADER_LEN;
        while pos < bytes.len() {
            match bytes[pos] {
                TAG_FRAME => {
                    let count = u32::from_le_bytes([
                        bytes[pos + 1],
                        bytes[pos + 2],
                        bytes[pos + 3],
                        bytes[pos + 4],
                    ]) as usize;
                    pos += 5 + count * 4;
                }
                TAG_BOUNDARY => return pos + 17,
                _ => panic!("unexpected tag in fixture"),
            }
        }
        panic!("fixture has no boundary record");
    }

    #[test]
    fn journal_round_trip_is_byte_exact() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let id = writer.id().to_string();

        let first = ramp(1_000, 0);
        let second = ramp(500, 1_000);
        let third = ramp(250, 1_500);
        writer.append_frames(&first).expect("append 1");
        writer.write_boundary().expect("boundary 1");
        writer.append_frames(&second).expect("append 2");
        writer.write_boundary().expect("boundary 2");
        writer.append_frames(&third).expect("append 3");
        let acked = writer.finalize().expect("finalize");
        assert_eq!(acked, 1_750);

        let path = dir.path().join(format!("{id}.sj"));
        let parsed = read_journal(&path).expect("parse finalized journal");
        assert!(parsed.finalized, "trailer validates");
        assert_eq!(parsed.sample_rate, 16_000);
        assert_eq!(parsed.boundary_count, 2, "the two cadence boundaries");
        assert_eq!(parsed.torn_tail_bytes, 0);
        let mut expected = first.clone();
        expected.extend_from_slice(&second);
        expected.extend_from_slice(&third);
        assert_eq!(parsed.samples, expected, "samples read back byte-exact");

        // Byte-exact reproducibility: the same append/boundary sequence in a
        // second journal produces the identical file bytes.
        let mut twin = writer_in(&dir, 16_000);
        twin.append_frames(&first).expect("append 1");
        twin.write_boundary().expect("boundary 1");
        twin.append_frames(&second).expect("append 2");
        twin.write_boundary().expect("boundary 2");
        twin.append_frames(&third).expect("append 3");
        twin.finalize().expect("finalize");
        assert_eq!(
            std::fs::read(&path).expect("re-read first journal"),
            std::fs::read(twin.path()).expect("read twin journal"),
            "identical input must produce identical journal bytes"
        );
    }

    #[test]
    fn boundary_due_triggers_on_bytes_and_time() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 48_000);
        assert!(
            !writer.boundary_due(),
            "fresh journal is not due before any bytes"
        );
        // 16 385 samples = 65 540 payload bytes > 64 KiB.
        writer
            .append_frames(&ramp(16_385, 0))
            .expect("append past the byte threshold");
        assert!(writer.boundary_due(), "byte threshold triggers");
        writer.write_boundary().expect("boundary");
        assert!(!writer.boundary_due(), "reset after the boundary");

        // Time trigger: the interval passes while samples are pending (the
        // cadence bounds the acknowledgment latency of UNSYNCED samples —
        // an idle journal with nothing to confirm is never due, however
        // much clock passes; that invariant is what keeps a fresh or silent
        // writer from flapping on timer noise).
        writer
            .append_frames(&ramp(1, 0))
            .expect("append one pending sample");
        assert!(
            !writer.boundary_due(),
            "one sample is under both thresholds"
        );
        std::thread::sleep(JOURNAL_BOUNDARY_INTERVAL + Duration::from_millis(20));
        assert!(writer.boundary_due(), "time threshold triggers on pending samples");
        writer.write_boundary().expect("boundary after the interval");
        assert!(!writer.boundary_due(), "boundary acks the pending sample");
        // …and a boundary with nothing new stays a clock-resetting no-op.
        writer.write_boundary().expect("empty boundary");
        assert!(!writer.boundary_due(), "empty boundary is still a no-op");
    }

    #[test]
    fn torn_tail_truncates_to_the_last_valid_boundary() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let id = writer.id().to_string();

        let confirmed = ramp(800, 0);
        writer.append_frames(&confirmed).expect("append 1");
        writer.write_boundary().expect("boundary 1");

        // Frames appended after the boundary but never boundary-confirmed —
        // the crash scenario.
        writer.append_frames(&ramp(300, 800)).expect("append 2");
        let path = dir.path().join(format!("{id}.sj"));
        let bytes = std::fs::read(&path).expect("read journal");
        let boundary_end = first_boundary_end(&bytes);

        // Corrupt the last 4 bytes (a torn final write): the tail can never
        // become audio because no boundary ever confirmed it, detectable or
        // not.
        let mut corrupted = bytes.clone();
        let last = corrupted.len();
        for byte in &mut corrupted[last - 4..] {
            *byte ^= 0xff;
        }
        std::fs::write(&path, &corrupted).expect("write corrupted journal");
        let parsed = read_journal(&path).expect("parse corrupted journal");
        assert!(!parsed.finalized, "no trailer was written");
        assert_eq!(parsed.samples, confirmed, "verified prefix is exact");
        assert_eq!(parsed.boundary_count, 1);
        assert_eq!(
            parsed.torn_tail_bytes,
            corrupted.len() as u64 - boundary_end as u64,
            "the discarded region is reported in exact bytes"
        );

        // Truncating the file mid-record recovers to the same boundary.
        let truncated = bytes[..bytes.len() - 37].to_vec();
        std::fs::write(&path, &truncated).expect("write truncated journal");
        let parsed = read_journal(&path).expect("parse truncated journal");
        assert_eq!(parsed.samples, confirmed);
        assert_eq!(parsed.boundary_count, 1);
        assert_eq!(
            parsed.torn_tail_bytes,
            truncated.len() as u64 - boundary_end as u64,
            "tail size is exact for the truncated file"
        );
    }

    #[test]
    fn mid_file_corruption_rolls_back_to_the_prior_boundary() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let id = writer.id().to_string();

        let first = ramp(600, 0);
        let second = ramp(600, 600);
        let third = ramp(600, 1_200);
        writer.append_frames(&first).expect("append 1");
        writer.write_boundary().expect("boundary 1");
        writer.append_frames(&second).expect("append 2");
        writer.write_boundary().expect("boundary 2");
        writer.append_frames(&third).expect("append 3");
        writer.write_boundary().expect("boundary 3");
        writer.finalize().expect("finalize");

        let path = dir.path().join(format!("{id}.sj"));
        let mut bytes = std::fs::read(&path).expect("read journal");
        // Corrupt one payload byte inside the *second* frame record: every
        // later boundary and the trailer hash-mismatch, so recovery keeps
        // only the first chunk.
        let second_frame_payload = HEADER_LEN + 5 + first.len() * 4 + 17 + 5;
        bytes[second_frame_payload + 7] ^= 0x55;
        std::fs::write(&path, &bytes).expect("write corrupted journal");

        let parsed = read_journal(&path).expect("parse corrupted journal");
        assert!(!parsed.finalized, "trailer no longer validates");
        assert_eq!(parsed.samples, first, "recovery rolls back to boundary 1");
        assert_eq!(parsed.boundary_count, 1);
    }

    #[test]
    fn garbage_and_foreign_files_are_rejected_as_not_journals() {
        let dir = TempDir::new().expect("tempdir");
        let garbage = dir.path().join("j_garbage.sj");
        std::fs::write(&garbage, b"not a journal at all, just text").expect("write garbage");
        match read_journal(&garbage) {
            Err(JournalReadError::NotAJournal(reason)) => {
                assert!(reason.contains("header"), "{reason}");
            }
            other => panic!("expected NotAJournal, got {other:?}"),
        }

        let short = dir.path().join("j_short.sj");
        std::fs::write(&short, b"STRL").expect("write stub");
        assert!(matches!(
            read_journal(&short),
            Err(JournalReadError::NotAJournal(_))
        ));

        let future = dir.path().join("j_future.sj");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(FORMAT_VERSION + 1);
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        std::fs::write(&future, &bytes).expect("write future journal");
        assert!(matches!(
            read_journal(&future),
            Err(JournalReadError::NotAJournal(_))
        ));
    }

    #[test]
    fn header_only_journal_recovers_no_samples() {
        let dir = TempDir::new().expect("tempdir");
        let writer = writer_in(&dir, 16_000);
        let parsed = read_journal(writer.path()).expect("parse header-only");
        assert!(parsed.samples.is_empty());
        assert!(!parsed.finalized);
        assert_eq!(parsed.torn_tail_bytes, 0);
        assert_eq!(parsed.boundary_count, 0);
    }

    #[test]
    fn bytes_after_a_valid_trailer_mean_not_finalized() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let id = writer.id().to_string();
        let samples = ramp(700, 0);
        writer.append_frames(&samples).expect("append");
        writer.finalize().expect("finalize");

        let path = dir.path().join(format!("{id}.sj"));
        let mut bytes = std::fs::read(&path).expect("read finalized journal");
        bytes.extend_from_slice(b"XX"); // junk past the trailer
        std::fs::write(&path, &bytes).expect("write padded journal");

        let parsed = read_journal(&path).expect("parse padded journal");
        assert!(
            !parsed.finalized,
            "a trailer must be the final record to count"
        );
        assert_eq!(
            parsed.samples, samples,
            "the boundary written by finalize still verifies the samples"
        );
    }

    #[test]
    fn sync_faults_freeze_acknowledgment_at_the_last_good_boundary() {
        let dir = TempDir::new().expect("tempdir");
        let fail = Arc::new(AtomicBool::new(false));
        let id = "j_syncfault".to_string();
        let (sink, path) = FaultSink::create_sync_faults_only(dir.path(), &id, Arc::clone(&fail))
            .expect("open sync-fault journal");
        let mut writer =
            JournalWriter::over_sink(sink, id.clone(), path.clone(), 16_000)
                .expect("writer with header fsynced");

        let first = ramp(1_000, 0);
        writer.append_frames(&first).expect("append 1");
        let acked = writer.write_boundary().expect("boundary 1 syncs");
        assert_eq!(acked, 1_000);

        // From here every sync fails, while appends keep succeeding: bytes
        // still reach the page cache and the file on disk.
        fail.store(true, Ordering::Relaxed);
        writer
            .append_frames(&ramp(500, 1_000))
            .expect("append 2 still writes");
        assert!(
            writer.write_boundary().is_err(),
            "the boundary fsync must fail"
        );
        assert!(writer.finalize().is_err(), "finalize must fail too");

        // What the parser can verify after the fault is physically
        // indeterminate — a write may reach disk even when its fsync
        // reported failure — so the decidable assertions are: the fsync-
        // confirmed prefix always survives verification, anything beyond
        // it only survives by genuinely persisting with a valid checksum,
        // and nothing beyond the written total can appear. (The exact
        // freeze — acknowledged stuck at the last confirmed boundary — is
        // asserted in the recorder-level fault test, where the fault also
        // blocks appends.)
        let parsed = read_journal(&path).expect("parse");
        assert!(
            parsed.samples.len() >= 1_000 && parsed.samples.len() <= 1_500,
            "verified {} samples, expected somewhere in 1 000..=1 500",
            parsed.samples.len()
        );
        assert!(
            parsed.samples.starts_with(&first),
            "the confirmed prefix is always verified"
        );
    }

    #[test]
    fn startup_scan_recovers_sessions_and_skips_linked_or_empty() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");

        // (a) An interrupted journal: confirmed samples + torn tail.
        let mut torn = writer_in(&journals_dir, 24_000);
        let torn_id = torn.id().to_string();
        let confirmed = ramp(2_400, 0); // exactly 100 ms at 24 kHz
        torn.append_frames(&confirmed).expect("append");
        torn.write_boundary().expect("boundary");
        torn.append_frames(&ramp(300, 2_400)).expect("never confirmed");
        drop(torn);

        // (b) A finalized-but-unsaved journal (crash between trailer and
        // session save).
        let mut done = writer_in(&journals_dir, 24_000);
        let done_id = done.id().to_string();
        let done_samples = ramp(4_800, 0); // 200 ms at 24 kHz
        done.append_frames(&done_samples).expect("append");
        done.finalize().expect("finalize");
        drop(done);

        // (c) A journal already linked to a saved session: must be skipped.
        let mut linked = writer_in(&journals_dir, 24_000);
        let linked_id = linked.id().to_string();
        linked.append_frames(&ramp(100, 0)).expect("append");
        linked.finalize().expect("finalize");
        drop(linked);
        let wav = encode_wav_16k(&PcmAudio {
            samples: ramp(100, 0),
            sample_rate: 24_000,
            channels: 1,
        })
        .expect("encode");
        store
            .create_with_journal(wav, Some(5.0), Some(&linked_id))
            .expect("create linked session");

        // (d) A header-only journal: no session, file kept.
        writer_in(&journals_dir, 24_000);

        // (e) A garbage file with the journal extension: reported, kept.
        std::fs::write(journals_dir.path().join("j_garbage.sj"), b"junk")
            .expect("write garbage");

        let report = recover_interrupted_takes(&store, journals_dir.path()).expect("recovery");
        assert_eq!(report.recovered.len(), 2, "torn + finalized-orphan");
        assert_eq!(report.empty_journals.len(), 1);
        assert_eq!(report.unreadable.len(), 1);
        assert!(report.has_findings());
        assert!(
            report.summary().contains("2 interrupted recordings"),
            "{}",
            report.summary()
        );

        let by_journal: HashSet<String> = report
            .recovered
            .iter()
            .map(|session| session.journal_id.clone().expect("linked"))
            .collect();
        assert_eq!(
            by_journal,
            HashSet::from([torn_id.clone(), done_id.clone()])
        );

        for session in &report.recovered {
            assert_eq!(session.status, SessionStatus::Interrupted);
            let decoded = decode_pcm16_wav(&session.wav).expect("recovered wav decodes");
            assert_eq!(decoded.sample_rate, 16_000);
            let note = session.last_error.as_deref().expect("recovery note");
            assert!(note.contains("recovered"), "{note}");
        }

        // Duration comes from the verified sample count at the device rate.
        let torn_session = report
            .recovered
            .iter()
            .find(|session| session.journal_id.as_deref() == Some(torn_id.as_str()))
            .expect("torn session");
        assert_eq!(torn_session.duration_ms, Some(100.0));
        assert!(
            torn_session
                .last_error
                .as_deref()
                .expect("note")
                .contains("unfinished write"),
            "the torn tail must be recorded as the gap: {:?}",
            torn_session.last_error
        );
        let done_session = report
            .recovered
            .iter()
            .find(|session| session.journal_id.as_deref() == Some(done_id.as_str()))
            .expect("finalized session");
        assert_eq!(done_session.duration_ms, Some(200.0));
        assert!(
            done_session
                .last_error
                .as_deref()
                .expect("note")
                .contains("finished cleanly"),
            "the orphan's note must distinguish it from a torn take"
        );

        // Never deleted, never modified: every journal we created still
        // exists, and a rerun recovers nothing new.
        let mut remaining: HashSet<String> = std::fs::read_dir(journals_dir.path())
            .expect("read journals dir")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        for id in [&torn_id, &done_id, &linked_id] {
            assert!(
                remaining.remove(&format!("{id}.sj")),
                "journal {id} must stay in place as source evidence"
            );
        }
        assert!(
            remaining.contains("j_garbage.sj"),
            "garbage is kept for inspection"
        );

        let rerun = recover_interrupted_takes(&store, journals_dir.path()).expect("rerun");
        assert!(
            rerun.recovered.is_empty(),
            "recovery must be idempotent via manifest linkage"
        );
        assert_eq!(rerun.unreadable.len(), 1, "garbage still reported");
        assert_eq!(rerun.empty_journals.len(), 1);
    }

    #[test]
    fn startup_scan_treats_a_missing_journals_dir_as_empty() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");

        // Missing dir: nothing to recover, not an error.
        let report = recover_interrupted_takes(&store, &journals_dir.path().join("nope"))
            .expect("missing dir is fine");
        assert!(!report.has_findings());

        // Empty dir: same.
        let report =
            recover_interrupted_takes(&store, journals_dir.path()).expect("empty dir");
        assert!(!report.has_findings());
    }

    // ---- R21: confirmed deletion vs journal recovery ----

    /// A saved session with a live linked journal, the state the B05 dialog
    /// deletes. Returns (session id, journal id).
    fn linked_session_with_journal(
        store: &FileSessionStore,
        journals_dir: &TempDir,
        samples: usize,
    ) -> (String, String) {
        let mut writer = writer_in(journals_dir, 24_000);
        let journal_id = writer.id().to_string();
        writer.append_frames(&ramp(samples, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        drop(writer);

        let wav = encode_wav_16k(&PcmAudio {
            samples: ramp(samples, 0),
            sample_rate: 24_000,
            channels: 1,
        })
        .expect("encode");
        let session = store
            .create_with_journal(wav, Some(50.0), Some(&journal_id))
            .expect("create linked session");
        assert!(
            journals_dir.path().join(format!("{journal_id}.sj")).exists(),
            "the journal starts live"
        );
        (session.id, journal_id)
    }

    /// R21: a confirmed session deletion takes the linked journal with it —
    /// renamed into `deleted/` (never an eager unlink) — and removes the row
    /// + audio; startup recovery then finds nothing to resurrect, exactly as
    /// the B05 warning promised ("permanently removes the audio").
    #[test]
    fn confirmed_deletion_quarantines_the_linked_journal_and_removes_the_row() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");
        let (session_id, journal_id) =
            linked_session_with_journal(&store, &journals_dir, 1_200);

        delete_session_and_journal(&store, journals_dir.path(), &session_id)
            .expect("confirmed delete");

        // The row and its audio are gone (the old behavior, kept).
        assert!(store.get(&session_id).expect("get").is_none());
        // The journal is no longer live — renamed, not unlinked: the bytes
        // await the I2 retention sweep.
        assert!(
            !journals_dir.path().join(format!("{journal_id}.sj")).exists(),
            "the live journal is gone from the scan path"
        );
        let tombstone = journals_dir
            .path()
            .join(DELETED_DIR)
            .join(format!("{journal_id}.sj"));
        assert!(
            tombstone.exists(),
            "the journal is quarantined as the tombstone"
        );

        // Nothing left to resurrect on the next startup.
        let report = recover_interrupted_takes(&store, journals_dir.path()).expect("recovery");
        assert!(
            !report.has_findings(),
            "the deleted take must stay deleted: {}",
            report.summary()
        );
    }

    /// R21 crash-safety of the ordering. (a) A crash after the tombstone
    /// rename but before the row removal leaves nothing recoverable, and the
    /// retried delete completes. (b) The tombstone is permanent: even a
    /// journal reappearing live under `journals/` with a tombstoned id (a
    /// restored backup, a copied disk) is never resurrected, while an
    /// untombstoned orphan beside it still is.
    #[test]
    fn recovery_skips_tombstoned_ids_forever() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");
        let (session_id, deleted_id) =
            linked_session_with_journal(&store, &journals_dir, 1_200);

        // The control: an untombstoned orphan that must still recover.
        let mut orphan = writer_in(&journals_dir, 24_000);
        let orphan_id = orphan.id().to_string();
        orphan.append_frames(&ramp(2_400, 0)).expect("append");
        orphan.write_boundary().expect("boundary");
        drop(orphan);

        // (a) Crash between the tombstone rename and the row removal: the
        // quarantine is committed, the delete never finished.
        let quarantined =
            quarantine_journal(journals_dir.path(), &deleted_id).expect("tombstone commit");
        assert!(quarantined, "a live journal was quarantined");
        assert!(store.get(&session_id).expect("get").is_some(), "row still present");

        let report = recover_interrupted_takes(&store, journals_dir.path()).expect("recovery");
        let recovered_ids: HashSet<String> = report
            .recovered
            .iter()
            .map(|session| session.journal_id.clone().expect("linked"))
            .collect();
        assert_eq!(
            recovered_ids,
            HashSet::from([orphan_id]),
            "only the orphan recovers; the tombstoned take does not"
        );

        // The delete retries after the restart and completes.
        delete_session_and_journal(&store, journals_dir.path(), &session_id)
            .expect("delete completes after the crash");
        assert!(store.get(&session_id).expect("get").is_none());

        // (b) The journal file reappears live (a restored backup) while its
        // tombstone still sits in deleted/: the id stays dead forever. A
        // fresh orphan beside it is the control that still recovers (the
        // first orphan is linked by pass (a) above).
        let mut second_orphan = writer_in(&journals_dir, 24_000);
        let second_orphan_id = second_orphan.id().to_string();
        second_orphan.append_frames(&ramp(2_400, 5_000)).expect("append");
        second_orphan.write_boundary().expect("boundary");
        drop(second_orphan);

        let quarantined = journals_dir
            .path()
            .join(DELETED_DIR)
            .join(format!("{deleted_id}.sj"));
        std::fs::copy(&quarantined, journals_dir.path().join(format!("{deleted_id}.sj")))
            .expect("restore the file");
        let report = recover_interrupted_takes(&store, journals_dir.path()).expect("recovery");
        let recovered_ids: HashSet<String> = report
            .recovered
            .iter()
            .map(|session| session.journal_id.clone().expect("linked"))
            .collect();
        assert_eq!(
            recovered_ids,
            HashSet::from([second_orphan_id]),
            "the reappeared journal must not resurrect — the id is tombstoned"
        );
        assert!(
            !report.empty_journals.iter().any(|id| id == &deleted_id)
                && !report.unreadable.iter().any(|(id, _)| id == &deleted_id),
            "a tombstoned id is skipped silently, not reported anywhere"
        );
    }

    /// R21: deleting a session without a journal linkage is exactly the old
    /// behavior — the row goes, nothing under the journals root is touched.
    #[test]
    fn deleting_a_session_without_a_journal_touches_nothing_under_journals() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");

        // An unrelated live journal that must not be swept up.
        let mut writer = writer_in(&journals_dir, 24_000);
        let unrelated_id = writer.id().to_string();
        writer.append_frames(&ramp(100, 0)).expect("append");
        writer.finalize().expect("finalize");
        drop(writer);

        let session = store
            .create(
                encode_wav_16k(&PcmAudio {
                    samples: ramp(100, 0),
                    sample_rate: 24_000,
                    channels: 1,
                })
                .expect("encode"),
                Some(10.0),
            )
            .expect("create unlinked session");

        delete_session_and_journal(&store, journals_dir.path(), &session.id)
            .expect("delete");

        assert!(store.get(&session.id).expect("get").is_none());
        assert!(
            journals_dir.path().join(format!("{unrelated_id}.sj")).exists(),
            "unrelated journals are untouched"
        );
        assert!(
            !journals_dir.path().join(DELETED_DIR).exists(),
            "no quarantine dir is created for a journal-less deletion"
        );
    }

    /// R21: only the confirmed-delete entry point may tombstone. The
    /// interrupted-take persistence paths — exactly the store calls the
    /// app's `save_interrupted_take` (R17 salvage) and journal recovery
    /// itself make — must leave the journal live, and the R05 stash path is
    /// in-memory only (it performs no store or journal call at all).
    #[test]
    fn persisting_an_interrupted_take_does_not_tombstone_its_journal() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");
        let (session_id, journal_id) =
            linked_session_with_journal(&store, &journals_dir, 1_200);

        // The salvage/recovery shape: persist + mark interrupted. No
        // confirmed deletion happened anywhere.
        store
            .mark_interrupted(&session_id, "quiesce timeout salvage")
            .expect("mark interrupted");

        assert!(
            journals_dir.path().join(format!("{journal_id}.sj")).exists(),
            "the journal stays live without a confirmed deletion"
        );
        assert!(
            !journals_dir.path().join(DELETED_DIR).exists(),
            "no tombstone was written"
        );

        // The linkage still guards idempotence: nothing new is recovered.
        let report = recover_interrupted_takes(&store, journals_dir.path()).expect("recovery");
        assert!(report.recovered.is_empty(), "linked stays linked");
    }

    /// R21: when the quarantine cannot be committed (here: a regular file
    /// squatting on the `deleted/` dir name), the deletion aborts *before*
    /// the row removal — a half-delete that leaves a live journal behind is
    /// exactly the resurrection bug, and surfacing the error with the
    /// session intact is the honest outcome.
    #[test]
    fn a_failed_tombstone_aborts_deletion_before_the_row_removal() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");
        let (session_id, journal_id) =
            linked_session_with_journal(&store, &journals_dir, 1_200);

        // Sabotage: `deleted` exists as a regular file, so the quarantine
        // dir cannot be created.
        std::fs::write(journals_dir.path().join(DELETED_DIR), b"not a directory")
            .expect("sabotage");

        assert!(
            delete_session_and_journal(&store, journals_dir.path(), &session_id).is_err(),
            "the quarantine failure must surface, not pass silently"
        );
        // The session row and audio survive: the delete did not half-happen.
        assert!(
            store.get(&session_id).expect("get").is_some(),
            "the row must survive a failed tombstone"
        );
        // The journal is still live — and still linked, so even a startup
        // scan in this state recovers nothing.
        assert!(journals_dir.path().join(format!("{journal_id}.sj")).exists());
        let report = recover_interrupted_takes(&store, journals_dir.path()).expect("recovery");
        assert!(report.recovered.is_empty());
    }

    /// R21: a hand-corrupted manifest linkage that is not a safe path
    /// component is treated as no linkage: the deletion proceeds (a weird
    /// manifest must never brick deleting the session) and no path outside
    /// the journals root is ever touched.
    #[test]
    fn an_unsafe_manifest_linkage_is_ignored_rather_than_followed() {
        let store_dir = TempDir::new().expect("store tempdir");
        let store = FileSessionStore::open(store_dir.path()).expect("open store");
        let journals_dir = TempDir::new().expect("journals tempdir");

        let session = store
            .create(
                encode_wav_16k(&PcmAudio {
                    samples: ramp(100, 0),
                    sample_rate: 24_000,
                    channels: 1,
                })
                .expect("encode"),
                Some(10.0),
            )
            .expect("create session");

        // Rewrite the manifest with a traversal linkage.
        let manifest_path = store_dir.path().join(&session.id).join("manifest.json");
        let raw = std::fs::read_to_string(&manifest_path).expect("read manifest");
        let mut manifest: serde_json::Value = serde_json::from_str(&raw).expect("parse manifest");
        manifest
            .as_object_mut()
            .expect("object")
            .insert("journalId".into(), serde_json::json!("../../escape.sj"));
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap())
            .expect("rewrite manifest");

        // The unsafe linkage reads back as "no known linkage"…
        assert_eq!(store.journal_id_of(&session.id).expect("lookup"), None);
        // …so the delete proceeds without touching anything outside.
        delete_session_and_journal(&store, journals_dir.path(), &session.id)
            .expect("delete");
        assert!(store.get(&session.id).expect("get").is_none());
        assert!(
            !journals_dir.path().join(DELETED_DIR).exists(),
            "nothing under the journals root was touched"
        );
    }
}
