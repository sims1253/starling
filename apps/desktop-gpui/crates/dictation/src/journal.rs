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
//! - Clean stop: the trailer (length + content hash) is written and the file
//!   and its parent directory are fsynced. The finished journal is then
//!   adopted by [`crate::store_v2`] as the take's durable audio evidence
//!   (verified, sealed when torn, moved through the §4 crash protocol) —
//!   this module is the raw writer/reader surface, the store owns the
//!   layout protocol.
//! - Recovery of an interrupted take is [`crate::store_v2::StoreV2::reconcile`]'s
//!   job on the store's own trees; a journal left in this tree is either
//!   adopted at save time or is startup-scanned by the store's owner.
//!
//! RETENTION POLICY (frozen, R21): journal files are never deleted by any
//! crash-recovery or read path. A `journals/deleted/` directory left behind
//! by the deleted v1 store holds its tombstoned journals; the explicit
//! retention sweep ([`crate::store_v2::StoreV2::sweep_retention`]) is the
//! only thing that ever unlinks them.
//!
//! Deferred to I2/I3 (known gaps, by design of this increment): gap records
//! inside the journal (ring-overflow gaps are flagged live via
//! [`crate::recorder::RecorderHandle::gaps`] but are not serialized into the
//! journal) and the `staging/` → `audio/` rename protocol as a store-v2
//! concern (`store_v2` owns it; this module's writer stays the raw surface).

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Extension of journal files under the journals root.
const JOURNAL_EXT: &str = "sj";

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

/// The one read buffer a streamed journal parse uses (bounded reads): the
/// parser walks the file through chunks of this size instead of reading
/// it whole, so peak memory is the verified samples plus one chunk.
pub(crate) const JOURNAL_READ_CHUNK: usize = 64 * 1024;

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
/// is torn tail.
///
/// The read is **streaming, not whole-file**: the parser walks the file
/// through one fixed-size buffer ([`JOURNAL_READ_CHUNK`]), so peak memory is
/// the verified samples plus one chunk, never the samples plus the entire
/// file. Semantics match the old whole-file parse with two documented
/// differences, both consequences of the length being pinned to the
/// `metadata()` snapshot taken at open: a file that shrinks *while* being
/// read surfaces as a torn tail (the stream ends early), and a file that
/// *grows* is parsed only up to the snapshot length. A read that fails
/// mid-parse (`EIO`-class) propagates as [`JournalReadError::Io`] exactly
/// like the old whole-file read — never as a torn tail — so a transient
/// I/O failure is retried/reported instead of silently truncating audio.
pub(crate) fn read_journal(path: &Path) -> Result<ParsedJournal, JournalReadError> {
    use std::io::BufReader;
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(JOURNAL_READ_CHUNK, file);
    parse_journal_from(&mut reader, file_len)
}

/// What one bounded read learned: the bytes were filled, the stream ended
/// before they were (`Ok(false)` — torn tail), or the read genuinely
/// failed (`Err` — propagated, never collapsed into end-of-stream).
type Filled = bool;

/// Read exactly `buf.len()` bytes from `reader`.
///
/// `Ok(true)` — filled. `Ok(false)` — the stream ended first (only
/// `Ok(0)` counts as end-of-stream); the caller treats whatever it was
/// building as torn. `Err` — a non-`Interrupted` I/O error, retried on
/// `Interrupted` and propagated otherwise so real failures surface as
/// [`JournalReadError::Io`] like the old whole-file read.
fn read_exact(reader: &mut impl std::io::Read, buf: &mut [u8]) -> io::Result<Filled> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    Ok(true)
}

/// The streaming parser over any reader, pinned to `file_len` (the
/// `metadata()` snapshot the file-reading wrapper takes; tests drive this
/// half directly with scripted readers). See [`read_journal`] for the
/// semantics, including the two snapshot-pin divergences.
fn parse_journal_from(
    reader: &mut impl std::io::Read,
    file_len: u64,
) -> Result<ParsedJournal, JournalReadError> {
    if file_len < HEADER_LEN as u64 {
        return Err(JournalReadError::NotAJournal(
            "journal is shorter than the v1 header".to_string(),
        ));
    }
    let mut header = [0u8; HEADER_LEN];
    if !read_exact(reader, &mut header)? {
        return Err(JournalReadError::NotAJournal(
            "journal ended inside the v1 header".to_string(),
        ));
    }
    if &header[..MAGIC.len()] != MAGIC {
        return Err(JournalReadError::NotAJournal(
            "journal does not start with the v1 header".to_string(),
        ));
    }
    if header[MAGIC.len()] != FORMAT_VERSION {
        return Err(JournalReadError::NotAJournal(format!(
            "unsupported journal format version {}",
            header[MAGIC.len()]
        )));
    }
    let sample_rate = u32::from_le_bytes([header[9], header[10], header[11], header[12]]);

    let mut samples: Vec<f32> = Vec::new();
    let mut hash = FNV_OFFSET;
    let mut verified_len = 0usize;
    let mut verified_end = HEADER_LEN as u64;
    let mut boundary_count = 0usize;
    let mut finalized = false;
    // One scratch buffer for every frame payload: the read stays bounded
    // at this size however long the journal is.
    let mut io_buf = vec![0u8; JOURNAL_READ_CHUNK];

    let mut pos = HEADER_LEN as u64;
    'parse: while pos < file_len {
        let remaining = file_len - pos;
        let mut tag = [0u8; 1];
        if !read_exact(reader, &mut tag)? {
            break 'parse; // stream ended early: torn
        }
        match tag[0] {
            TAG_FRAME => {
                let mut count_buf = [0u8; 4];
                if !read_exact(reader, &mut count_buf)? {
                    break 'parse;
                }
                let count = u32::from_le_bytes(count_buf) as usize;
                let payload_len = count.saturating_mul(4);
                // Also guards absurd counts from garbage: a record larger
                // than the bytes actually present is a torn write.
                if remaining < 5 + payload_len as u64 {
                    break 'parse;
                }
                samples.reserve(count);
                // A frame record is atomic for verification purposes: if
                // the stream ends inside its payload, the bytes already
                // streamed in were hashed and pushed but belong to no
                // verification point — roll the sample vector back to the
                // frame's start so `samples` only ever holds
                // boundary-verified audio (the whole-file parser's guard
                // guaranteed the same by never entering a short record).
                let frame_start_len = samples.len();
                let mut complete = true;
                let mut left = payload_len;
                while left > 0 {
                    let take = left.min(io_buf.len());
                    let (chunk, _) = io_buf.split_at_mut(take);
                    if !read_exact(reader, chunk)? {
                        complete = false;
                        break;
                    }
                    hash = fnv1a(hash, chunk);
                    for bytes in chunk.chunks_exact(4) {
                        let bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        samples.push(f32::from_bits(bits));
                    }
                    left -= take;
                }
                if !complete {
                    samples.truncate(frame_start_len);
                    break 'parse;
                }
                pos += 5 + payload_len as u64;
            }
            TAG_BOUNDARY | TAG_TRAILER => {
                if remaining < 17 {
                    break 'parse;
                }
                let mut record = [0u8; 16];
                if !read_exact(reader, &mut record)? {
                    break 'parse;
                }
                let count = u64::from_le_bytes(
                    record[0..8]
                        .try_into()
                        .expect("8 bytes parse as u64"),
                );
                let recorded_hash = u64::from_le_bytes(
                    record[8..16]
                        .try_into()
                        .expect("8 bytes parse as u64"),
                );
                if count != samples.len() as u64 || recorded_hash != hash {
                    break 'parse;
                }
                if tag[0] == TAG_BOUNDARY {
                    boundary_count += 1;
                    verified_len = samples.len();
                    verified_end = pos + 17;
                    pos += 17;
                } else {
                    // Trailer: its checksum is a valid verification point
                    // for everything before it even when bytes follow, but
                    // "finalized" additionally requires it to be the file's
                    // last record. Never parse past a trailer — a journal
                    // appended after its own trailer is malformed, and the
                    // verified prefix ends at the trailer either way.
                    if pos + 17 == file_len {
                        finalized = true;
                    }
                    verified_len = samples.len();
                    verified_end = pos + 17;
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
        torn_tail_bytes: file_len - verified_end,
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

/// The FNV-1a 64 hash of a journal's VERIFIED sample payloads — the exact
/// value `StoreV2::adopt_journal` stores as the adopted row's
/// `journal_hash` (computed over the verified prefix, never the trailer's
/// unverified claim). Callers use it to prove content identity before
/// treating a row found under a journal's id as that journal's audio: the
/// id alone is the file stem and proves nothing — a stale, renamed or
/// reused journal file under the same id must not satisfy a different
/// take's commit. An unreadable journal yields `Err`, which callers treat
/// as "identity unproven" rather than as a match.
pub fn verified_journal_hash(path: &Path) -> Result<String, JournalReadError> {
    let parsed = read_journal(path)?;
    Ok(format!("{:016x}", samples_hash(&parsed.samples)))
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

/// The journals root: `<data-root>/starling-gpui/journals/` — the
/// recorder's live-capture tree (a sibling of the storage-v2 root's own
/// `audio/` and `staging/` trees). Finished journals are adopted out of
/// here by [`crate::store_v2`].
pub fn default_journals_root() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("starling-gpui")
        .join("journals")
}

/// fsync a directory's own entry so a rename inside it survives a crash.
/// Defined in [`crate::storage`] (the lower layer) and re-exported here
/// for the storage-v2 staging→audio promotion and quarantine paths (I2).
pub(crate) use crate::storage::sync_dir;

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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::testing::FaultSink;
    use super::*;
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

    /// A journal whose payload crosses the streamed-read buffer several
    /// times over parses exactly like a small one: byte-exact samples,
    /// the right boundary count, no torn tail.
    #[test]
    fn streamed_parse_handles_journals_larger_than_the_read_chunk() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let id = writer.id().to_string();

        // 30k samples = 120 KB of payload, ~2x JOURNAL_READ_CHUNK, split
        // across frame records the way the cadence writer emits them.
        let mut expected = Vec::new();
        let mut offset = 0u32;
        while expected.len() < 30_000 {
            let block = ramp(4_096.min(30_000 - expected.len()), offset);
            offset += 4_096 as u32;
            writer.append_frames(&block).expect("append");
            expected.extend_from_slice(&block);
            if expected.len() % 8_192 == 0 {
                writer.write_boundary().expect("boundary");
            }
        }
        let acked = writer.finalize().expect("finalize");
        assert_eq!(acked, expected.len() as u64);

        let parsed = read_journal(&dir.path().join(format!("{id}.sj")))
            .expect("parse large journal");
        assert!(parsed.finalized);
        assert_eq!(parsed.torn_tail_bytes, 0);
        assert_eq!(parsed.samples.len(), 30_000);
        assert_eq!(parsed.samples, expected, "streamed read is byte-exact");
    }

    /// A torn tail after a large verified section: the parser must stop at
    /// the last checksum-valid boundary even when the tail's partial frame
    /// record stretches past the read buffer's edge cases.
    #[test]
    fn streamed_parse_truncates_a_torn_tail_after_a_large_verified_prefix() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let id = writer.id().to_string();

        let verified = ramp(20_000, 0); // 80 KB of verified payload
        writer.append_frames(&verified).expect("append");
        let boundary_end = {
            let path = writer.path().to_path_buf();
            writer.write_boundary().expect("boundary");
            let bytes = std::fs::read(&path).expect("read mid-state");
            let end = first_boundary_end(&bytes);
            assert!(end > JOURNAL_READ_CHUNK, "the fixture crosses the chunk");
            end
        };
        drop(writer); // no finalize: the take died mid-write

        // A torn tail: a frame header claiming 1_000 samples with only
        // 100 bytes of payload actually present.
        let mut torn = Vec::new();
        torn.push(TAG_FRAME);
        torn.extend_from_slice(&1_000u32.to_le_bytes());
        torn.extend_from_slice(&[0xA5u8; 100]);
        let path = dir.path().join(format!("{id}.sj"));
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open for torn append");
        file.write_all(&torn).expect("append torn frame");

        let file_len = std::fs::metadata(&path).expect("len").len();
        let parsed = read_journal(&path).expect("parse torn journal");
        assert!(!parsed.finalized);
        assert_eq!(parsed.samples, verified, "the verified prefix is exact");
        assert_eq!(
            parsed.torn_tail_bytes,
            file_len - boundary_end as u64,
            "the discarded tail is exactly the torn record"
        );
    }

    /// A reader that serves a fixed byte slice and then fails with a
    /// chosen error kind — the test double for a disk failing mid-read.
    struct FailingAfter {
        bytes: std::io::Cursor<Vec<u8>>,
        exhausted: bool,
        error: io::ErrorKind,
    }

    impl FailingAfter {
        fn new(bytes: Vec<u8>, error: io::ErrorKind) -> Self {
            Self {
                bytes: std::io::Cursor::new(bytes),
                exhausted: false,
                error,
            }
        }
    }

    impl std::io::Read for FailingAfter {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.bytes.read(buf)?;
            if n > 0 {
                return Ok(n);
            }
            self.exhausted = true;
            Err(io::Error::new(self.error, "injected mid-read failure"))
        }
    }

    /// A mid-read I/O failure must surface as `JournalReadError::Io` —
    /// the old whole-file contract — never as a torn tail: reconcile
    /// would otherwise seal (physically truncate) and adopt a journal
    /// whose only defect was a transient disk error.
    #[test]
    fn a_mid_read_io_error_surfaces_as_an_error_not_a_torn_tail() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let verified = ramp(3_000, 0);
        writer.append_frames(&verified).expect("append");
        writer.write_boundary().expect("boundary");
        // A second frame the stream will die inside.
        writer.append_frames(&ramp(3_000, 3_000)).expect("append");
        let full = std::fs::read(writer.path()).expect("read journal bytes");
        drop(writer);

        // The reader serves the first half of the file, then errors. The
        // file length stays pinned to the full snapshot, so the parser
        // believes more records exist — exactly the shape an EIO after a
        // successful open + metadata produces.
        let cut = full.len() / 2;
        let mut failing = FailingAfter::new(full[..cut].to_vec(), io::ErrorKind::Other);
        match parse_journal_from(&mut failing, full.len() as u64) {
            Err(JournalReadError::Io(err)) => {
                assert!(err.to_string().contains("injected mid-read failure"))
            }
            other => panic!("expected Io, got {other:?}"),
        }
        assert!(failing.exhausted);
    }

    /// A stream that ends inside a frame payload (the documented
    /// shrink-mid-read shape): the samples already streamed in belong to
    /// no verification point and must be rolled back —
    /// `ParsedJournal.samples` only ever holds boundary-verified audio.
    #[test]
    fn a_stream_ending_mid_payload_never_leaks_unverified_samples() {
        let dir = TempDir::new().expect("tempdir");
        let mut writer = writer_in(&dir, 16_000);
        let verified = ramp(500, 0);
        writer.append_frames(&verified).expect("append");
        writer.write_boundary().expect("boundary");
        // An unverified frame follows the boundary; the stream will cut
        // inside its payload.
        writer.append_frames(&ramp(2_000, 500)).expect("append");
        let full = std::fs::read(writer.path()).expect("read journal bytes");
        drop(writer);

        // Cut the byte stream inside the second frame's payload while the
        // length snapshot still covers the whole file.
        let boundary_end = first_boundary_end(&full);
        let cut = boundary_end + 5 + 400; // inside the frame's payload
        let mut truncated = std::io::Cursor::new(full[..cut].to_vec());
        let parsed =
            parse_journal_from(&mut truncated, full.len() as u64).expect("parse truncates");
        assert!(!parsed.finalized);
        assert_eq!(
            parsed.samples, verified,
            "only the boundary-verified prefix survives"
        );
        assert_eq!(
            parsed.torn_tail_bytes,
            (full.len() - boundary_end) as u64,
            "everything past the last valid boundary is the torn tail"
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

}
