//! Microphone capture, ported from `apps/desktop/src/useRecorder.ts`
//! (`getUserMedia` + `ScriptProcessorNode` + `AnalyserNode`) — see
//! `apps/desktop-gpui/PORT.md`. The transport between the audio callback and
//! the take was redesigned per `docs/program/design/e17-native-runtime.md`
//! §3 (G01/R01/R09); the signal processing (downmix, pre-DSP clip metering,
//! attenuation-only auto gain) is unchanged.
//!
//! Data flow:
//!
//! - The cpal audio callback converts/downmixes to mono f32, meters
//!   clipping on the raw samples (G03), applies the attenuator, and writes
//!   the block into a preallocated single-producer/single-consumer ring.
//!   The callback takes no `Mutex`, sends on no channel, and allocates
//!   nothing once its scratch buffer is warm (R01). Every value it touches
//!   is either callback-local storage or a lock-free atomic.
//! - A dedicated writer task drains the ring continuously (poll cadence
//!   ~25 ms) into the in-memory sample accumulation, recording a gap span
//!   whenever the producer overwrote samples it had not yet consumed
//!   (G01): a full ring overwrites oldest, gaps are flagged via
//!   [`RecorderHandle::gaps`], never silently joined. Phase 2 additionally
//!   mirrors every drained sample into the per-take durable journal
//!   ([`crate::journal`]) with fsynced boundaries: the journal is the
//!   authoritative retained audio, the in-memory accumulation a
//!   convenience. Only samples covered by an fsynced boundary count as
//!   acknowledged — see [`RecorderHandle::acknowledged_samples`]. The
//!   journal append and its fsync run with the consumer lock released, so
//!   accessor polls and the device error callback never wait on storage.
//! - `stop` is an explicit handshake (R09), ordered so the durable journal
//!   and the returned take agree: declare `finalSampleIndex = written_seq`,
//!   drop the CPAL stream, wait (bounded) for `callback_alive == false`,
//!   and only then signal the writer task — whose final drain + journal
//!   finalize therefore observe the complete, stable take. A quiesce
//!   timeout defers teardown instead of racing a callback that may still
//!   be executing: the salvaged accumulation is returned in a typed
//!   [`RecorderError::QuiesceTimeout`], and the `Shared` allocation is
//!   left to the callback's own `Arc` — it keeps writing harmlessly into a
//!   ring nothing reads, and its last access frees the memory. A silent
//!   empty take is structurally impossible: the capture path holds no
//!   mutex that could poison, and the consumer state is always recovered
//!   from a poisoned lock rather than treated as empty.
//!
//! The stream is requested as f32 / 1 channel / 16 kHz when the device
//! supports it; otherwise the device default config is used and any format
//! (integer samples, >1 channels) is converted to mono f32 inside the
//! callback. Resampling to 16 kHz happens in [`crate::audio`] at
//! WAV-encode time, driven by the UI layer — not here.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::journal::{FileSink, JournalSink, JournalWriter};

/// Sample rate requested from the microphone. Devices that cannot capture
/// natively at 16 kHz are used at their own rate; the UI resamples when
/// encoding the WAV.
const PREFERRED_SAMPLE_RATE: u32 = 16_000;

/// Nominal span of the capture ring at the device rate (§3 suggests 1–2 s),
/// rounded to the nearest power-of-two sample count (real devices land in
/// ~1.4–2.1 s). Bounds the memory the live capture can hold between the
/// audio callback and the writer task.
const CAPTURE_RING_SECONDS: f64 = 2.0;

/// How often the writer task wakes to drain the ring when idle. Also the
/// upper bound on how long `stop` needs to join the writer.
const WRITER_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// How long `stop` waits for the audio callback to quiesce after the
/// stream is dropped before degrading to [`RecorderError::QuiesceTimeout`]
/// (R09).
const QUIESCE_TIMEOUT: Duration = Duration::from_secs(2);

/// Peak the capture auto-gain pulls hot input down toward. Chosen to leave
/// encode headroom (the WAV clamp saturates at 1.0) while staying transparent
/// for normal speech.
const AUTO_GAIN_TARGET_PEAK: f32 = 0.72;

/// Samples per gain update; a linear ramp across the block keeps the step
/// inaudible.
const AUTO_GAIN_BLOCK: usize = 256;

/// Fraction of the remaining reduction closed per block when input is hot
/// (~16 ms at 16 kHz, so speech onsets converge within a couple of blocks).
const AUTO_GAIN_ATTACK: f32 = 0.9;

/// Fraction of the distance back to unity per block when input cools down.
const AUTO_GAIN_RELEASE: f32 = 0.06;

/// |sample| at or above this counts as clipped source evidence. i16 full
/// scale converts to 0.99997, so 0.999 catches integer full-scale input
/// while ignoring ±1-LSB flutter below it.
pub const CLIP_THRESHOLD: f32 = 0.999;

/// Clipped-sample fraction above which the UI warns about the source.
pub const CLIP_WARNING_RATIO: f64 = 0.02;

/// Incremental clipping evidence measured on the raw captured samples,
/// before the capture auto-gain touches them (G03): attenuation can pull
/// an already-clipped source below any peak threshold, so the warning must
/// be driven by pre-DSP counts, not by the attenuated copy that is kept.
///
/// Backed by lock-free atomics so the audio callback can update it with no
/// `Mutex` (R01); the ratio only needs to converge, not to be a point-in-
/// time consistent snapshot.
#[derive(Debug, Default)]
pub struct ClipCounters {
    total: AtomicU64,
    clipped: AtomicU64,
}

impl ClipCounters {
    /// Counts full-scale samples against [`CLIP_THRESHOLD`]. A NaN sample is
    /// not evidence of clipping (the comparison is false), matching how the
    /// encoder treats non-finite input.
    pub fn observe(&self, samples: &[f32]) {
        let mut clipped = 0u64;
        for &sample in samples {
            if sample.abs() >= CLIP_THRESHOLD {
                clipped += 1;
            }
        }
        if clipped > 0 {
            self.clipped.fetch_add(clipped, Ordering::Relaxed);
        }
        self.total.fetch_add(samples.len() as u64, Ordering::Relaxed);
    }

    /// Clipped fraction in 0..=1, or 0.0 when nothing was observed (also the
    /// empty-recording case, so no warning can be fabricated).
    pub fn ratio(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.clipped.load(Ordering::Relaxed) as f64 / total as f64
        }
    }
}

/// The capture warning for a source clip `ratio` in 0..=1: `Some(message)`
/// only above [`CLIP_WARNING_RATIO`], with the percentage honestly scaled
/// to 0-100 (G03: the old `{ratio:.0}%` rendered a 3% ratio as 0%). The
/// message describes the unprocessed source and states that attenuation
/// cannot repair it — post-ADC gain is never presented as a fix.
pub fn clipping_warning(ratio: f64) -> Option<String> {
    if !(ratio > CLIP_WARNING_RATIO) {
        return None;
    }

    let percent = (ratio * 100.0).round();
    Some(format!(
        "The microphone input itself was heavily clipped ({percent:.0}% of source samples at \
         full scale, measured before any processing). Lower the input level in your sound \
         settings and record again for a cleaner take — the app's auto-attenuation cannot \
         repair samples that were already clipped at the source."
    ))
}

/// A span of capture sequence indices the producer overwrote before the
/// writer could drain them (§3 G01). `end_sample - start_sample` samples
/// are missing from the take at exactly this position; the surviving
/// samples on either side are joined in the output, but the join is always
/// flagged here — never silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureGap {
    /// First missing sample index (in samples-since-capture-start).
    pub start_sample: u64,
    /// One past the last missing sample index.
    pub end_sample: u64,
}

impl CaptureGap {
    /// How many samples the gap dropped.
    pub fn missing_samples(&self) -> u64 {
        self.end_sample - self.start_sample
    }
}

/// What the durable capture journal did for a take (I1 phase 2). Returned
/// by [`RecorderHandle::stop`] and carried by
/// [`RecorderError::QuiesceTimeout`] so the caller can link the session
/// manifest to its journal (`journal_id`) regardless of how the take ended.
#[derive(Debug, Clone)]
pub struct JournalReport {
    /// Journal id — the stem of `journals/<id>.sj`.
    pub id: String,
    /// Path of the journal file (left in place by every capture outcome;
    /// only a confirmed session deletion quarantines it — R21 — and the
    /// I2 retention sweep owns actual removal).
    pub path: PathBuf,
    /// Device sample rate recorded in the journal header.
    pub sample_rate: u32,
    /// Samples covered by the last fsynced boundary — the honest
    /// "survives a process kill right now" count. On a faulted journal
    /// this is where acknowledgment froze.
    pub acknowledged_samples: u64,
    /// Whether the trailer was written and the file + parent directory
    /// fsynced on a clean stop.
    pub finalized: bool,
    /// Why journaling degraded, when a write/fsync failed: capture kept
    /// running in memory, but acknowledged samples stopped advancing.
    pub fault: Option<String>,
}

/// A cleanly stopped take: the audio, plus the journal that mirrors it.
#[derive(Debug)]
pub struct CapturedTake {
    /// The full take as mono [`crate::audio::PcmAudio`] at the device rate
    /// — exactly what `stop` always returned.
    pub audio: crate::audio::PcmAudio,
    /// The durable journal report, when this capture journaled.
    pub journal: Option<JournalReport>,
}

#[derive(Debug, thiserror::Error)]
pub enum RecorderError {
    #[error("{0}")]
    Device(String),
    #[error("No microphone audio was captured.")]
    Empty,
    /// The stop handshake timed out waiting for the audio callback to
    /// quiesce after the stream was dropped (R09). The acknowledged samples
    /// are preserved in `audio` — a wedged callback must never turn into a
    /// silent empty take. Teardown is deferred: the callback keeps its own
    /// `Arc` to the ring, so `stop` never races a callback that may still
    /// be executing. The journal was already finalized by the writer task
    /// before it exited, so `journal` reports its state (I1 phase 2) — the
    /// caller persists `audio` as an interrupted take and links the journal.
    #[error(
        "The microphone did not stop cleanly within the quiesce timeout; {acknowledged_samples} \
         captured samples are preserved in this error and were not lost."
    )]
    QuiesceTimeout {
        /// Total samples this capture accumulated (everything the writer
        /// drained, including any already handed out via
        /// [`RecorderHandle::drain_chunks`]); the audio in this error is
        /// the not-yet-handed-out remainder.
        acknowledged_samples: u64,
        /// Everything captured and drained but not yet handed out, ready to
        /// encode; `acknowledged_samples` minus this length is what the
        /// caller already owns.
        audio: crate::audio::PcmAudio,
        /// The journal this take wrote (already finalized if it was
        /// healthy), for manifest linkage when salvaging the take.
        journal: Option<JournalReport>,
    },
}

/// Attenuation-only auto gain for the capture path.
///
/// Raw sources can run hot enough to saturate the WAV clamp (a 100% route
/// gain clips close speech at the device, and anything near the ceiling
/// clips as soon as the speaker raises their voice). This pulls recent peaks
/// toward [`AUTO_GAIN_TARGET_PEAK`] with a fast attack and slow release, and
/// never amplifies: input below the target passes at unity gain, bit-exact.
#[derive(Debug)]
pub struct Attenuator {
    gain: f32,
    target: f32,
}

impl Default for Attenuator {
    fn default() -> Self {
        Self {
            gain: 1.0,
            target: 1.0,
        }
    }
}

impl Attenuator {
    /// Applies the current ramp to `samples` in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for block in samples.chunks_mut(AUTO_GAIN_BLOCK) {
            let peak = block.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            let mut target = self.target;
            if peak > AUTO_GAIN_TARGET_PEAK {
                let wanted = (AUTO_GAIN_TARGET_PEAK / peak).min(1.0);
                target += (wanted - target) * AUTO_GAIN_ATTACK;
            } else {
                target += (1.0 - target) * AUTO_GAIN_RELEASE;
            }
            let start = self.gain;
            let delta = target - start;
            let steps = block.len().max(1) as f32;
            for (index, sample) in block.iter_mut().enumerate() {
                *sample *= start + delta * (index as f32 / steps);
            }
            self.gain = target;
            self.target = target;
        }
    }
}

/// Consumer-side take state: everything the writer task has drained from
/// the ring, plus its gap bookkeeping. Touched only by non-RT threads.
#[derive(Default)]
struct ConsumerState {
    /// The attenuated mono stream — exactly the samples the producer
    /// preserved, in capture order, with any overwritten spans joined but
    /// flagged in `gaps`.
    samples: Vec<f32>,
    /// Next ring sequence index to consume.
    read_pos: u64,
    /// Spans the producer overwrote before this consumer read them.
    gaps: Vec<CaptureGap>,
    /// Count of leading samples already handed out via
    /// [`RecorderHandle::drain_chunks`] — the single handout watermark.
    /// Only [`ConsumerState::take_pending`] advances it, so incremental
    /// drainers and `stop` agree on the delivered/pending boundary by
    /// construction instead of each keeping a tally that can double-count
    /// or lose a span.
    delivered: usize,
    /// First device-side error posted by the CPAL error callback (E01/G01).
    stream_error: Option<String>,
    /// Samples already mirrored into the journal (the journal-side
    /// watermark on `samples`). [`ConsumerState::delivered`] tracks what
    /// was handed to callers; this tracks what was made durable.
    journaled: usize,
    /// First journal write/fsync failure (I1 phase 2 storage-fault
    /// honesty): journaling stops, acknowledged samples stop advancing at
    /// the last good boundary, and capture itself keeps running in memory.
    journal_fault: Option<String>,
    /// Set when the writer task finalized the journal (trailer + file and
    /// parent-dir fsyncs) on its exit path.
    journal_finalized: bool,
}

impl ConsumerState {
    /// The accumulated samples not yet handed out.
    fn pending(&self) -> &[f32] {
        &self.samples[self.delivered..]
    }

    /// Hands out every accumulated sample not yet delivered, advancing the
    /// watermark in the one place it is tracked. Both
    /// [`RecorderHandle::drain_chunks`] and [`RecorderHandle::stop`] take
    /// their pending span through here, so an interleaved caller's drained
    /// chunks and `stop`'s final return always reassemble the take exactly.
    fn take_pending(&mut self) -> Vec<f32> {
        let pending = self.pending().to_vec();
        self.delivered = self.samples.len();
        pending
    }
}

/// State shared between the cpal audio callback (producer) and the writer
/// task plus [`RecorderHandle`] (consumers).
///
/// The audio callback touches only `ring`, `written_seq`, `callback_alive`,
/// and `clip` — all lock-free and wait-free (R01). Everything else lives on
/// the consumer side and is guarded by a plain `Mutex` that the callback
/// never holds.
struct Shared {
    /// Preallocated SPSC ring of post-DSP mono samples, stored bit-cast
    /// into `AtomicU32` slots so both sides can touch them through `&Self`
    /// with no data race (Relaxed slot access; publication ordering is
    /// carried by `written_seq` — Release store after the slot writes,
    /// Acquire load before reading them).
    ring: Box<[AtomicU32]>,
    /// `ring.len() - 1`; capacity is a power of two, so wrapping is a mask.
    mask: u64,
    /// Monotonic count of samples ever pushed by the callback — the
    /// sequence frontier, the gap-detection watermark source, and the
    /// `finalSampleIndex` declared by the stop handshake.
    written_seq: AtomicU64,
    /// Samples covered by the last fsynced journal boundary (I1 phase 2):
    /// the acknowledged count, published by the writer task with Release
    /// after each successful boundary fsync. Without a journal nothing is
    /// ever durable, so this stays 0. Always ≤ `written_seq`.
    durable_ack: AtomicU64,
    /// True while a callback invocation is executing. The stop handshake
    /// waits for this to clear after the stream is dropped (R09).
    callback_alive: AtomicBool,
    /// Pre-DSP clipping evidence, updated from the callback (G03).
    clip: ClipCounters,
    /// Consumer-side accumulation and gap bookkeeping.
    consumer: Mutex<ConsumerState>,
    /// Wakes the writer promptly for its final drain at stop.
    wake: Condvar,
    /// Set by `stop` once the producer is quiesced — after the stream is
    /// dropped and `callback_alive` has cleared, or the bounded quiesce
    /// wait has given up — so the writer's exit path (final drain + journal
    /// finalize) observes the complete, stable take (#204). The
    /// `stream.play()` failure teardown in `start_recording_inner` also
    /// sets it directly; no callback ever ran there.
    stopping: AtomicBool,
}

/// Ring slot count for a device rate: [`CAPTURE_RING_SECONDS`] worth of
/// samples rounded to the nearest power of two (the wrap arithmetic needs
/// a power of two), floored at 2 048.
fn ring_capacity(sample_rate: u32) -> usize {
    let target = ((sample_rate as f64 * CAPTURE_RING_SECONDS).round() as usize).max(2_048);
    let above = target.next_power_of_two();
    let below = (above / 2).max(2_048);
    if target - below <= above - target {
        below
    } else {
        above
    }
}

/// Append a gap span to `state`, merging with the previous one when the
/// producer lapped the reader twice in a row (contiguous spans).
fn record_gap(state: &mut ConsumerState, start: u64, end: u64) {
    match state.gaps.last_mut() {
        Some(previous) if previous.end_sample == start => previous.end_sample = end,
        _ => state.gaps.push(CaptureGap {
            start_sample: start,
            end_sample: end,
        }),
    }
}

impl Shared {
    fn new(capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "ring capacity must be a power of two"
        );
        Self {
            ring: (0..capacity)
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            mask: (capacity - 1) as u64,
            written_seq: AtomicU64::new(0),
            durable_ack: AtomicU64::new(0),
            callback_alive: AtomicBool::new(false),
            clip: ClipCounters::default(),
            consumer: Mutex::new(ConsumerState::default()),
            wake: Condvar::new(),
            stopping: AtomicBool::new(false),
        }
    }

    /// Locks the consumer state, recovering from poisoning instead of
    /// treating a poisoned lock as "no data" (R09: the old silent
    /// empty-drain failure mode is gone by construction).
    fn lock_consumer(&self) -> MutexGuard<'_, ConsumerState> {
        self.consumer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Producer side — runs in the audio callback. Writes `mono` at the
    /// current frontier, overwriting the oldest samples past capacity
    /// (§3 overflow policy: keep capturing, bound memory). A block larger
    /// than the whole ring keeps only its tail; the skipped sequence
    /// indices surface to the consumer as a gap, never as stale slot data.
    fn push_block(&self, mono: &[f32]) {
        let cap = self.ring.len();
        let skipped = mono.len().saturating_sub(cap);
        let src = &mono[skipped..];
        let base = self.written_seq.load(Ordering::Relaxed) + skipped as u64;
        let start = (base & self.mask) as usize;
        let first = src.len().min(cap - start);
        for (offset, &sample) in src[..first].iter().enumerate() {
            self.ring[start + offset].store(sample.to_bits(), Ordering::Relaxed);
        }
        if first < src.len() {
            for (offset, &sample) in src[first..].iter().enumerate() {
                self.ring[offset].store(sample.to_bits(), Ordering::Relaxed);
            }
        }
        // Release-publishes the slot writes above; the consumer Acquire-
        // loads this before reading any slot below the frontier.
        self.written_seq
            .store(base + src.len() as u64, Ordering::Release);
    }

    /// Consumer side — drains every published sample into `state`,
    /// detecting overwrites (§3): when the producer has lapped the reader,
    /// the unread span is recorded as a gap and the reader jumps to the
    /// oldest surviving sample. Gaps are flagged, never silently joined.
    ///
    /// Soundness: at any moment with frontier `w`, slot `i` holds sample `i`
    /// iff `w - capacity <= i < w`. The gap clamp enforces exactly that
    /// range before any slot is copied. (A producer lapping the *entire*
    /// ring during the copy itself would need seconds of audio to pass
    /// during a microsecond-scale memcpy — and the stop path only drains
    /// after the callback has quiesced.)
    fn drain_ring(&self, state: &mut ConsumerState) {
        let cap = self.ring.len() as u64;
        loop {
            let frontier = self.written_seq.load(Ordering::Acquire);
            let oldest_alive = frontier.saturating_sub(cap);
            if state.read_pos < oldest_alive {
                record_gap(state, state.read_pos, oldest_alive);
                state.read_pos = oldest_alive;
            }
            if state.read_pos >= frontier {
                return;
            }
            let count = (frontier - state.read_pos) as usize; // <= cap
            let start = (state.read_pos & self.mask) as usize;
            let first = count.min(self.ring.len() - start);
            state.samples.reserve(count);
            for slot in &self.ring[start..start + first] {
                state
                    .samples
                    .push(f32::from_bits(slot.load(Ordering::Relaxed)));
            }
            if first < count {
                for slot in &self.ring[..count - first] {
                    state
                        .samples
                        .push(f32::from_bits(slot.load(Ordering::Relaxed)));
                }
            }
            state.read_pos = frontier;
            // Loop while the producer advanced during the copy; each
            // iteration consumes to a freshly loaded frontier, so this
            // terminates (copying is orders of magnitude faster than
            // capture fills).
        }
    }

    /// Posts a device-side error as typed state the app can query (E01/G01)
    /// — the capture path never prints. Called from the CPAL error
    /// callback, which is not the RT data path; the first error wins.
    fn record_stream_error(&self, message: String) {
        let mut guard = self.lock_consumer();
        if guard.stream_error.is_none() {
            guard.stream_error = Some(message);
        }
    }
}

/// Waits (bounded) for the audio callback to be between invocations after
/// the stream has been dropped. Returns false on timeout.
fn wait_callback_quiesce(shared: &Shared, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while shared.callback_alive.load(Ordering::Acquire) {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// The payload of a failed thread join, as text. Boxed panic payloads print
/// as `Any { .. }` through Debug, which would say nothing.
fn panic_message(join_err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = join_err.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = join_err.downcast_ref::<String>() {
        message.clone()
    } else {
        "its panic payload could not be displayed".to_string()
    }
}

/// The dedicated consumer (§3 G01's writer task). Drains the ring every
/// poll interval so a stalled UI — occluded window, paused animation frames —
/// cannot overflow the capture ring, records gap spans on sequence
/// discontinuities, mirrors drained samples into the per-take durable
/// journal with fsynced boundaries (I1 phase 2), and exits within one
/// interval of `stopping`, finalizing the journal (trailer + fsyncs) on the
/// way out.
///
/// Locking (#203): the consumer lock is held only for in-memory work — the
/// ring drain, the unjournaled-span snapshot, the watermark advance, the
/// fault/state commits. The journal append and its fsync run with the lock
/// released, so accessors ([`RecorderHandle::latest_window`], polled by the
/// render thread every animation frame) and the CPAL error callback
/// ([`Shared::record_stream_error`]) never wait on storage. The journal
/// writer is owned solely by this task, so unlocking around its I/O
/// introduces no journal-side race.
fn writer_loop<S: JournalSink>(shared: Arc<Shared>, mut journal: Option<JournalWriter<S>>) {
    loop {
        {
            let mut guard = shared.lock_consumer();
            shared.drain_ring(&mut guard);
        }
        journal_boundary_step(&shared, &mut journal);
        if shared.stopping.load(Ordering::Acquire) {
            {
                let mut guard = shared.lock_consumer();
                shared.drain_ring(&mut guard);
            }
            journal_finalize_step(&shared, &mut journal);
            return;
        }
        // Sleep one poll interval (or until `stop` wakes us for the final
        // drain), holding no lock while waiting.
        let (guard, _timed_out) = shared
            .wake
            .wait_timeout(shared.lock_consumer(), WRITER_POLL_INTERVAL)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        drop(guard);
    }
}

/// The per-tick journal work: append newly drained samples and, when the
/// §3 cadence fires, write+fsync a boundary and publish the acknowledged
/// count. Storage-fault honesty (§3): a failing write or fsync records the
/// fault in the capture-error slot, drops the journal, and freezes
/// acknowledgment at the last good boundary — capture itself keeps running
/// (the ring keeps being drained into memory, so `stop` still returns the
/// whole take; only the durability claim stops advancing). What remains for
/// a full degrade-and-recover: re-attaching journaling after the fault and
/// bounding the memory accumulation when it never comes back (I2/I3).
///
/// Locking (#203): the unjournaled span is copied out under the consumer
/// lock, the append and the boundary fsync run with the lock released, and
/// the lock is re-acquired only to advance the watermark — by exactly the
/// snapshot's span, never to the (possibly larger) current accumulation,
/// so samples produced while the lock was released cannot be skipped: they
/// simply form the next tick's unjournaled span. `durable_ack` is published
/// only by this task, only after a boundary fsync returned, so it stays
/// monotonic; on a fault it is never stored again (frozen at the last good
/// boundary).
fn journal_boundary_step<S: JournalSink>(
    shared: &Shared,
    journal: &mut Option<JournalWriter<S>>,
) {
    if journal.is_none() {
        return;
    }

    // Snapshot of the unjournaled span: `(start index, samples)`. Immutable
    // once copied — the append and fsync below work on this copy, so a
    // concurrent accessor draining more ring samples into `state.samples`
    // cannot change what this append writes or what the watermark advance
    // covers.
    let snapshot: Option<(usize, Vec<f32>)> = {
        let guard = shared.lock_consumer();
        (guard.journaled < guard.samples.len()).then(|| {
            let from = guard.journaled;
            (from, guard.samples[from..].to_vec())
        })
    };

    let mut fault: Option<std::io::Error> = None;
    let mut acknowledged: Option<u64> = None;

    if let Some((journaled_from, samples)) = snapshot {
        let writer = journal.as_mut().expect("journal presence checked above");
        match writer.append_frames(&samples) {
            Ok(()) => {
                // Re-acquire and advance the watermark by exactly the
                // snapshot's span. `journaled` is written only by this task,
                // so it is still `journaled_from`; `samples.len()` at
                // snapshot time is ≤ the current accumulation, keeping
                // `journaled <= samples.len()` invariant.
                let mut guard = shared.lock_consumer();
                guard.journaled = journaled_from + samples.len();
            }
            Err(err) => fault = Some(err),
        }
    }

    if fault.is_none() {
        let writer = journal.as_mut().expect("journal presence checked above");
        if writer.boundary_due() {
            match writer.write_boundary() {
                Ok(acknowledged_samples) => acknowledged = Some(acknowledged_samples),
                Err(err) => fault = Some(err),
            }
        }
    }

    if let Some(acknowledged_samples) = acknowledged {
        // Monotonic: only this task stores, and only with the journal
        // writer's cumulative count after a successful fsync.
        shared.durable_ack.store(acknowledged_samples, Ordering::Release);
    }
    if let Some(err) = fault {
        let frozen = shared.durable_ack.load(Ordering::Acquire);
        let mut guard = shared.lock_consumer();
        guard.journal_fault = Some(format!(
            "The capture journal failed: {err}. Recording continues, but acknowledged \
             samples are frozen at {frozen} — newly captured audio is not being made durable."
        ));
        drop(guard);
        *journal = None;
    }
}

/// The stop-path journal work: append anything the final drain added, then
/// finalize (final boundary + trailer + file fsync + parent-dir fsync, §4
/// step 2) and publish the acknowledged count. A finalize failure leaves
/// the journal trailer-less — startup recovery then treats it as an
/// interrupted take at its last valid boundary, so no acknowledged audio is
/// lost; the fault is surfaced through the capture-error slot.
///
/// Locking (#203): like the boundary step, the final append and the
/// trailer's fsyncs run with the consumer lock released; the watermark and
/// `journal_finalized` commits re-acquire it.
///
/// Ordering (#204): the writer only reaches here after observing
/// `stopping`, which `stop` stores once the producer is quiesced (or the
/// bounded quiesce wait has given up). The tail this appends and
/// finalizes is therefore the same tail `stop`'s own final drain observes,
/// and the producer can add nothing while the finalize fsyncs run — the
/// journal watermark and the returned take agree by construction.
fn journal_finalize_step<S: JournalSink>(
    shared: &Shared,
    journal: &mut Option<JournalWriter<S>>,
) {
    let Some(mut writer) = journal.take() else {
        return;
    };
    // Snapshot under the lock; append + fsync outside it.
    let (journaled_from, tail) = {
        let guard = shared.lock_consumer();
        (guard.journaled, guard.samples[guard.journaled..].to_vec())
    };

    if let Err(err) = writer.append_frames(&tail) {
        let mut guard = shared.lock_consumer();
        guard.journal_fault = Some(format!(
            "The capture journal failed while writing the final samples: {err}. The take \
             is intact in memory; the journal stays as an interrupted source."
        ));
        return;
    }
    {
        let mut guard = shared.lock_consumer();
        guard.journaled = journaled_from + tail.len();
    }
    match writer.finalize() {
        Ok(acknowledged_samples) => {
            shared
                .durable_ack
                .store(acknowledged_samples, Ordering::Release);
            let mut guard = shared.lock_consumer();
            guard.journal_finalized = true;
        }
        Err(err) => {
            let mut guard = shared.lock_consumer();
            guard.journal_fault = Some(format!(
                "The capture journal could not be finalized: {err}. The take is intact in \
                 memory; the journal stays as an interrupted source and will be recovered to \
                 its last durable boundary on the next startup."
            ));
        }
    }
}

/// Callback-local capture state (R01): owned by the data callback's `FnMut`
/// closure, so the callback mutates it with no locks and allocates nothing
/// once the scratch buffer has seen one block.
struct CallbackState {
    channels: usize,
    attenuator: Attenuator,
    /// Reused downmix scratch: cleared and refilled per block; capacity
    /// stabilizes after the first callback.
    mono: Vec<f32>,
}

impl CallbackState {
    fn new(channels: usize) -> Self {
        Self {
            channels,
            attenuator: Attenuator::default(),
            mono: Vec::new(),
        }
    }

    /// The audio-callback body: convert + downmix, meter clipping on the
    /// raw samples (G03: pre-DSP), attenuate, push into the SPSC ring, and
    /// bracket it all with the quiesce flag. No locks, no channels, no
    /// allocation on the steady path (R01).
    fn process<T>(&mut self, data: &[T], shared: &Shared)
    where
        T: cpal::Sample,
        f32: cpal::FromSample<T>,
    {
        shared.callback_alive.store(true, Ordering::Release);
        downmix_into(data, self.channels, &mut self.mono);
        shared.clip.observe(&self.mono);
        self.attenuator.process(&mut self.mono);
        shared.push_block(&self.mono);
        shared.callback_alive.store(false, Ordering::Release);
    }
}

/// Convert one cpal sample of any supported format to f32 in -1..=1
/// (i16/i32/24-in-i32/u8/u16/f64/… all go through cpal's sample conversion).
fn sample_to_f32<T>(sample: T) -> f32
where
    T: cpal::Sample,
    f32: cpal::FromSample<T>,
{
    sample.to_sample()
}

/// Convert and average interleaved frames down to mono, appending to `out`.
/// Frames shorter than `channels` (a trailing partial frame) are dropped;
/// `channels <= 1` passes through (a zero-channel config is rejected at
/// stream-open, so that arm is only defensive totality for direct calls).
/// `out` is cleared first; its capacity is retained so repeated calls do
/// not allocate.
fn downmix_into<T>(data: &[T], channels: usize, out: &mut Vec<f32>)
where
    T: cpal::Sample,
    f32: cpal::FromSample<T>,
{
    out.clear();
    match channels {
        0 | 1 => out.extend(data.iter().map(|&sample| sample_to_f32(sample))),
        _ => out.extend(
            data.chunks_exact(channels).map(|frame| {
                frame.iter().map(|&sample| sample_to_f32(sample)).sum::<f32>()
                    / channels as f32
            }),
        ),
    }
}

/// Average interleaved f32 frames down to mono into a fresh `Vec` — the
/// test-facing form of [`downmix_into`].
#[cfg(test)]
fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len() / channels.max(1));
    downmix_into(samples, channels, &mut out);
    out
}

/// Identity of the journal this take owns, kept on the handle so `stop`
/// can report linkage even when the writer task died before finalizing.
struct JournalIdentity {
    id: String,
    path: PathBuf,
    rate: u32,
}

/// Live microphone capture handle returned by [`start_recording`].
pub struct RecorderHandle {
    shared: Arc<Shared>,
    stream: Option<cpal::Stream>,
    writer: Option<JoinHandle<()>>,
    sample_rate: u32,
    started_at: Instant,
    quiesce_timeout: Duration,
    journal: Option<JournalIdentity>,
}

impl RecorderHandle {
    /// Actual capture rate of the device (16 kHz when it was available
    /// natively, otherwise the device default).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Fraction of raw (pre-attenuation) samples at full scale for this
    /// capture so far — evidence about the microphone source rather than
    /// the attenuated copy `stop` returns. Read before calling `stop`,
    /// which consumes the handle. Lock-free (R01).
    pub fn source_clip_ratio(&self) -> f64 {
        self.shared.clip.ratio()
    }

    /// Wall-clock time since `start_recording`, like
    /// `performance.now() - startedAt` in the TS hook.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Total samples the audio callback has pushed (the sequence frontier).
    /// Groundwork for §3's acknowledged-sample reporting.
    pub fn captured_sample_count(&self) -> u64 {
        self.shared.written_seq.load(Ordering::Acquire)
    }

    /// Samples covered by the last fsynced journal boundary — the honest
    /// acknowledged count for §3's `capture.progress{ackSamples}` (I1
    /// phase 2). This is what survives a process kill right now; it is
    /// always ≤ [`Self::captured_sample_count`] and advances only after a
    /// boundary record's fsync returned, never on the write alone. Without
    /// a journal (or after a journal fault) nothing is durable, so it
    /// stays at the last fsynced value — 0 if there never was one.
    pub fn acknowledged_samples(&self) -> u64 {
        self.shared.durable_ack.load(Ordering::Acquire)
    }

    /// Gap spans where ring overflow dropped samples, in capture order.
    /// Empty means the take is a continuous stream; any entry means the
    /// output joins surviving audio across a flagged hole (G01: gaps are
    /// surfaced, never silently repaired).
    pub fn gaps(&self) -> Vec<CaptureGap> {
        let mut guard = self.shared.lock_consumer();
        self.shared.drain_ring(&mut guard);
        guard.gaps.clone()
    }

    /// The first capture-path error, if any (E01/G01 + I1 phase 2
    /// storage-fault honesty): the device-side error posted by the CPAL
    /// error callback wins; otherwise the first journal write/fsync
    /// failure is reported. The UI can stop pretending capture is durable
    /// (or live) once this is set.
    pub fn capture_error(&self) -> Option<String> {
        let guard = self.shared.lock_consumer();
        guard
            .stream_error
            .clone()
            .or_else(|| guard.journal_fault.clone())
    }

    /// Drains the mono f32 samples accumulated since the last call, in
    /// order, as one consolidated chunk.
    ///
    /// The simplest UI wiring is to never call this while recording and take
    /// the whole recording from `stop` (exactly like `useRecorder.ts`, which
    /// only reads `chunksRef` at stop). If you do drain per frame, you own
    /// the drained samples: `stop` returns only what is still pending — an
    /// empty `Ok`, never [`RecorderError::Empty`], because the take was
    /// captured; it is just already in your hands.
    pub fn drain_chunks(&self) -> Vec<Vec<f32>> {
        let mut guard = self.shared.lock_consumer();
        self.shared.drain_ring(&mut guard);
        let chunk = guard.take_pending();
        if chunk.is_empty() {
            Vec::new()
        } else {
            vec![chunk]
        }
    }

    /// Last `n` captured samples (fewer until the take fills), for live
    /// level metering with [`crate::fft`]. Non-destructive. Drains the ring
    /// first, so the window is fresh even if the writer task has not ticked
    /// yet.
    pub fn latest_window(&self, n: usize) -> Vec<f32> {
        let mut guard = self.shared.lock_consumer();
        self.shared.drain_ring(&mut guard);
        let start = guard.samples.len().saturating_sub(n);
        guard.samples[start..].to_vec()
    }

    /// Assembles the journal report from the handle's identity plus the
    /// writer task's outcome state (read after the writer is joined).
    fn journal_report(&self, guard: &ConsumerState) -> Option<JournalReport> {
        let identity = self.journal.as_ref()?;
        Some(JournalReport {
            id: identity.id.clone(),
            path: identity.path.clone(),
            sample_rate: identity.rate,
            acknowledged_samples: self.shared.durable_ack.load(Ordering::Acquire),
            finalized: guard.journal_finalized,
            fault: guard.journal_fault.clone(),
        })
    }

    /// Stops the capture with the §3 R09 handshake, ordered so the journal
    /// watermark and the returned take agree (#204):
    ///
    /// 1. declare `finalSampleIndex = written_seq`;
    /// 2. drop the CPAL stream (explicit stop; pause is unsupported on
    ///    some backends, and dropping the stream releases the device either
    ///    way);
    /// 3. wait (bounded by the quiesce timeout) for `callback_alive` to
    ///    clear — the producer is quiesced;
    /// 4. signal the writer task: its exit drain + journal finalize now
    ///    observe exactly the samples step 6 observes. Signaling before the
    ///    stream was dropped let the writer's 25 ms poll land mid-capture
    ///    and finalize a journal that silently missed the tail;
    /// 5. join the writer — its exit path finalized the journal (trailer +
    ///    file/dir fsyncs) when journaling was healthy;
    /// 6. drain anything left (also recovering ring contents if the writer
    ///    died unexpectedly), check the watermark agreement, and return the
    ///    pending samples as mono [`crate::audio::PcmAudio`] at the device
    ///    rate, with the [`JournalReport`] beside them.
    ///
    /// A quiesce timeout returns [`RecorderError::QuiesceTimeout`] with the
    /// acknowledged samples preserved inside the error — never a silent
    /// empty result — and defers teardown: the wedged callback keeps its
    /// own `Arc` to the ring, so `stop` neither races nor frees memory
    /// under a callback that may still be executing. The journal was
    /// already finalized at that point and rides along in the error so the
    /// caller can salvage the take as interrupted and link the journal.
    /// [`RecorderError::Empty`] means the device produced nothing at all;
    /// samples already handed out via [`Self::drain_chunks`] belong to the
    /// caller and do not make the take "empty".
    pub fn stop(mut self) -> Result<CapturedTake, RecorderError> {
        let final_sample_index = self.shared.written_seq.load(Ordering::Acquire);

        if let Some(stream) = self.stream.take() {
            // Explicit stop; pause is unsupported on some backends, and
            // dropping the stream releases the device either way. Backends
            // that join their callback thread make step (3) trivially pass;
            // the bounded wait covers the rest.
            let _ = stream.pause();
            drop(stream);
        }

        let quiesced = wait_callback_quiesce(&self.shared, self.quiesce_timeout);

        // Only now — the producer quiesced, or the bounded wait given up —
        // does the writer get its exit signal (#204): its final drain +
        // journal finalize must not run while the audio callback can still
        // deliver, or the durable copy silently misses the tail while the
        // report claims a finalized journal.
        self.shared.stopping.store(true, Ordering::Release);

        // Wake the writer under the lock so a thread entering wait_timeout
        // cannot miss the notification; it then drains once more,
        // finalizes the journal, and exits. The join happens on both
        // outcomes below: the writer never waits on the audio callback, so
        // even a wedged callback cannot stall it.
        {
            let _guard = self.shared.lock_consumer();
            self.shared.wake.notify_all();
        }
        if let Some(writer) = self.writer.take() {
            // A panic inside the writer loop must be neither swallowed nor
            // allowed to fail the take: it is surfaced through the same
            // first-error-wins slot the device error callback uses, while
            // the drain below still returns every sample that made it into
            // the ring. The journal, if any, stays trailer-less — recovery
            // picks it up as an interrupted take unless the caller links
            // it via the report's id.
            if let Err(join_err) = writer.join() {
                self.shared.record_stream_error(format!(
                    "The capture writer task failed: {}",
                    panic_message(join_err)
                ));
            }
        }

        if !quiesced {
            // Deferred teardown (§3 R09): `callback_alive` never cleared, so
            // a callback invocation may STILL be executing. Do not race it
            // for the ring — no final drain here — and salvage only what
            // the writer already accumulated. The handle's `Arc` is not the
            // last reference: the callback's captured clone keeps `Shared`
            // alive, its further writes land harmlessly in a ring nothing
            // reads anymore, and the callback's last access frees the
            // memory. The typed error carries the accumulated count and
            // the pending audio — plus the finalized journal report, since
            // the writer's exit path already closed it — so nothing
            // acknowledged is lost.
            let mut guard = self.shared.lock_consumer();
            let acknowledged_samples = guard.samples.len() as u64;
            let journal = self.journal_report(&guard);
            let pending = guard.take_pending();
            drop(guard);
            return Err(RecorderError::QuiesceTimeout {
                acknowledged_samples,
                audio: crate::audio::PcmAudio {
                    samples: pending,
                    sample_rate: self.sample_rate,
                    channels: 1,
                },
                journal,
            });
        }

        // Final drain: the producer has quiesced, so everything below the
        // published frontier is stable. This also recovers ring contents
        // even if the writer died unexpectedly. The writer has already
        // appended everything it drained, and the producer added nothing
        // after quiesce, so the journal watermark and this drain agree —
        // enforced here (#204), not merely claimed: a healthy, finalized
        // journal must acknowledge exactly the accumulated take, or the
        // report would vouch for durability the journal does not have.
        let mut guard = self.shared.lock_consumer();
        self.shared.drain_ring(&mut guard);
        let captured = self.shared.written_seq.load(Ordering::Acquire);
        debug_assert!(captured >= final_sample_index);
        if guard.journal_finalized && guard.journal_fault.is_none() {
            debug_assert_eq!(
                self.shared.durable_ack.load(Ordering::Acquire),
                guard.samples.len() as u64,
                "a finalized healthy journal must acknowledge the whole take"
            );
        }

        // Single-sourced tally: `delivered` advanced only in
        // `take_pending`, so the pending span returned here is exactly the
        // complement of whatever `drain_chunks` already handed out. `Empty`
        // is decided by whether anything was ever captured, not by whether
        // the pending span happens to be empty.
        let nothing_captured = guard.samples.is_empty();
        let journal = self.journal_report(&guard);
        let pending = guard.take_pending();
        drop(guard);

        if nothing_captured {
            return Err(RecorderError::Empty);
        }
        Ok(CapturedTake {
            audio: crate::audio::PcmAudio {
                samples: pending,
                sample_rate: self.sample_rate,
                channels: 1,
            },
            journal,
        })
    }
}

/// Opens the default input device and starts a mono capture stream, with
/// no durable journal (tests, probes). The production path is
/// [`start_recording_with_journal`].
pub fn start_recording() -> Result<RecorderHandle, RecorderError> {
    start_recording_inner(None)
}

/// [`start_recording`] with the per-take durable journal (I1 phase 2):
/// converted mono samples are mirrored to
/// `journals_dir/<journal-id>.sj` with fsynced boundaries on the §3
/// cadence; only boundary-covered samples are acknowledged. A journal that
/// cannot even be created (missing permissions, full disk) does not fail
/// the capture — it degrades honestly: recording proceeds in memory, the
/// fault lands in the capture-error slot, and
/// [`RecorderHandle::acknowledged_samples`] stays 0.
pub fn start_recording_with_journal(
    journals_dir: &Path,
) -> Result<RecorderHandle, RecorderError> {
    start_recording_inner(Some(journals_dir))
}

fn start_recording_inner(
    journals_dir: Option<&Path>,
) -> Result<RecorderHandle, RecorderError> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| {
        RecorderError::Device(
            "No microphone was found. Connect an input device and try again.".to_string(),
        )
    })?;

    let supported = pick_input_config(&device)?;
    let sample_format = supported.sample_format();
    let stream_config = supported.config();
    let sample_rate = stream_config.sample_rate.0;

    let shared = Arc::new(Shared::new(ring_capacity(sample_rate)));

    // Open the journal before the stream so a take that starts recording
    // always has its durable sink (or a surfaced fault) from the first
    // sample. create_new means an existing journal can never be stomped.
    let (journal, journal_identity) = match journals_dir
        .map(|dir| JournalWriter::<FileSink>::create(dir, sample_rate))
    {
        Some(Ok(writer)) => {
            let identity = JournalIdentity {
                id: writer.id().to_string(),
                path: writer.path().to_path_buf(),
                rate: writer.sample_rate(),
            };
            (Some(writer), Some(identity))
        }
        Some(Err(err)) => {
            // Degraded start (§3 storage-fault honesty): capture without a
            // journal, with the reason in the capture-error slot.
            shared.lock_consumer().journal_fault = Some(format!(
                "Could not open the capture journal: {err}. Recording continues, but no \
                 samples will be acknowledged as durable."
            ));
            (None, None)
        }
        None => (None, None),
    };

    let stream = open_stream(&device, sample_format, &stream_config, &shared)?;

    // The writer task owns the take for the whole session (§3 G01). Starting
    // it before `play` means the first callback already has a drainer; a
    // failure to spawn it is a device-level error, not a degraded capture.
    let writer = std::thread::Builder::new()
        .name("starling-capture-writer".to_string())
        .spawn({
            let shared = Arc::clone(&shared);
            move || writer_loop(shared, journal)
        })
        .map_err(|err| {
            RecorderError::Device(format!("Could not start the capture writer task: {err}"))
        })?;

    if let Err(err) = stream.play() {
        // Tear the writer down before failing, or it would poll forever.
        shared.stopping.store(true, Ordering::Release);
        {
            let _guard = shared.lock_consumer();
            shared.wake.notify_all();
        }
        let _ = writer.join();
        return Err(RecorderError::Device(format!(
            "Failed to start the microphone stream: {err}"
        )));
    }

    Ok(RecorderHandle {
        shared,
        stream: Some(stream),
        writer: Some(writer),
        sample_rate,
        started_at: Instant::now(),
        quiesce_timeout: QUIESCE_TIMEOUT,
        journal: journal_identity,
    })
}

/// Prefer f32 / 1 channel / 16 kHz when the device offers it; otherwise use
/// the device's default input config (converted in the callback).
fn pick_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig, RecorderError> {
    let ranges: Vec<_> = device
        .supported_input_configs()
        .map_err(|err| {
            RecorderError::Device(format!("Could not query microphone configurations: {err}"))
        })?
        .collect();

    if let Some(range) = ranges.into_iter().find(|range| {
        range.sample_format() == cpal::SampleFormat::F32
            && range.channels() == 1
            && range.min_sample_rate() <= cpal::SampleRate(PREFERRED_SAMPLE_RATE)
            && range.max_sample_rate() >= cpal::SampleRate(PREFERRED_SAMPLE_RATE)
    }) {
        return Ok(range.with_sample_rate(cpal::SampleRate(PREFERRED_SAMPLE_RATE)));
    }

    device.default_input_config().map_err(|err| {
        RecorderError::Device(format!(
            "Could not choose a microphone configuration: {err}"
        ))
    })
}

/// A capture stream needs at least one channel. A zero-channel device
/// config must be rejected loudly at stream-open: letting it through would
/// silently fall into the downmixer's mono passthrough and treat
/// interleaved nothing-at-all as if it were mono samples.
fn reject_zero_channels(channels: u16) -> Result<(), RecorderError> {
    if channels == 0 {
        Err(RecorderError::Device(
            "The microphone reported a zero-channel configuration and cannot be captured \
             from."
                .to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Build the input stream for whichever sample format the config settled on.
fn open_stream(
    device: &cpal::Device,
    sample_format: cpal::SampleFormat,
    config: &cpal::StreamConfig,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, RecorderError> {
    reject_zero_channels(config.channels)?;
    let attempted = match sample_format {
        cpal::SampleFormat::I8 => Some(build_stream::<i8>(device, config, shared)),
        cpal::SampleFormat::I16 => Some(build_stream::<i16>(device, config, shared)),
        cpal::SampleFormat::I32 => Some(build_stream::<i32>(device, config, shared)),
        cpal::SampleFormat::I64 => Some(build_stream::<i64>(device, config, shared)),
        cpal::SampleFormat::U8 => Some(build_stream::<u8>(device, config, shared)),
        cpal::SampleFormat::U16 => Some(build_stream::<u16>(device, config, shared)),
        cpal::SampleFormat::U32 => Some(build_stream::<u32>(device, config, shared)),
        cpal::SampleFormat::U64 => Some(build_stream::<u64>(device, config, shared)),
        cpal::SampleFormat::F32 => Some(build_stream::<f32>(device, config, shared)),
        cpal::SampleFormat::F64 => Some(build_stream::<f64>(device, config, shared)),
        // `SampleFormat` is #[non_exhaustive]; treat unknown formats as
        // unsupported instead of failing to compile.
        _ => None,
    };

    attempted
        .ok_or_else(|| {
            RecorderError::Device(format!(
                "Microphone sample format {sample_format} is not supported."
            ))
        })?
        .map_err(|err| {
            RecorderError::Device(format!("Failed to open the microphone stream: {err}"))
        })
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels as usize;
    let shared = Arc::clone(shared);
    let error_shared = Arc::clone(&shared);
    // R01: everything the data callback mutates lives in this closure
    // capture (FnMut storage) or lock-free atomics on `shared`.
    let mut callback = CallbackState::new(channels);
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            callback.process(data, &shared);
        },
        // E01/G01: device errors are posted as typed state, never printed.
        move |err| error_shared.record_stream_error(err.to_string()),
        // No delivery timeout: the callback cadence is the device's business.
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::testing::FaultSink;
    use tempfile::TempDir;

    /// A `Shared` with an arbitrary (power-of-two) ring capacity for
    /// callback-side simulation without a real device.
    fn test_shared(capacity: usize) -> Arc<Shared> {
        Arc::new(Shared::new(capacity))
    }

    /// A handle wired to a simulated capture: no stream, no writer thread,
    /// no journal.
    fn test_handle(shared: Arc<Shared>, sample_rate: u32) -> RecorderHandle {
        RecorderHandle {
            shared,
            stream: None,
            writer: None,
            sample_rate,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(500),
            journal: None,
        }
    }

    /// Spawn the real writer task against `shared`, without a journal.
    fn spawn_writer(shared: Arc<Shared>) -> JoinHandle<()> {
        spawn_writer_with::<FileSink>(shared, None)
    }

    /// Spawn the real writer task against `shared`, optionally journaling
    /// into `journal` (any sink, for fault injection).
    fn spawn_writer_with<S: JournalSink + 'static>(
        shared: Arc<Shared>,
        journal: Option<JournalWriter<S>>,
    ) -> JoinHandle<()> {
        std::thread::Builder::new()
            .name("test-capture-writer".to_string())
            .spawn(move || writer_loop(shared, journal))
            .expect("spawn writer")
    }

    /// A handle wired to a journaling simulated capture.
    fn journaled_test_handle<S: JournalSink + 'static>(
        shared: Arc<Shared>,
        writer: JournalWriter<S>,
        sample_rate: u32,
    ) -> RecorderHandle {
        let identity = JournalIdentity {
            id: writer.id().to_string(),
            path: writer.path().to_path_buf(),
            rate: writer.sample_rate(),
        };
        let writer_thread = spawn_writer_with(Arc::clone(&shared), Some(writer));
        RecorderHandle {
            shared,
            stream: None,
            writer: Some(writer_thread),
            sample_rate,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(500),
            journal: Some(identity),
        }
    }

    /// Poll until `predicate` holds or ~2 s elapse (the writer task ticks
    /// on its own cadence), then hand back the final value for asserting.
    fn wait_until<T>(mut probe: impl FnMut() -> T, predicate: impl Fn(&T) -> bool) -> T {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let value = probe();
            if predicate(&value) || Instant::now() >= deadline {
                return value;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn downmix_passes_mono_through() {
        let samples = vec![0.25, -0.5, 1.0];
        assert_eq!(downmix_to_mono(&samples, 1), samples);
    }

    #[test]
    fn downmix_averages_stereo_frames() {
        assert_eq!(downmix_to_mono(&[-1.0, 1.0, 0.5, 0.5], 2), vec![0.0, 0.5]);
    }

    #[test]
    fn downmix_averages_multichannel_frames() {
        assert_eq!(
            downmix_to_mono(&[1.0, 1.0, 1.0, -1.0, -1.0, -1.0], 3),
            vec![1.0, -1.0]
        );
    }

    #[test]
    fn downmix_drops_trailing_partial_frame() {
        assert_eq!(downmix_to_mono(&[0.1, 0.9, 0.5], 2), vec![0.5]);
    }

    #[test]
    fn integer_samples_convert_to_f32_range() {
        assert_eq!(sample_to_f32(i16::MIN), -1.0);
        assert!((sample_to_f32(i16::MAX) - 0.99997).abs() < 1e-4);
        assert_eq!(sample_to_f32(0i16), 0.0);
        assert_eq!(sample_to_f32(0u16), -1.0); // u16 silence sits at 32768
        assert_eq!(sample_to_f32(32_768u16), 0.0);
        assert_eq!(sample_to_f32(0i8), 0.0);
        assert_eq!(sample_to_f32(128u8), 0.0);
        assert!((sample_to_f32(i32::MIN) + 1.0).abs() < 1e-9);
        assert!((sample_to_f32(i32::MAX) - 1.0).abs() < 1e-9);
        assert_eq!(sample_to_f32(0.5f64), 0.5);
    }

    #[test]
    fn attenuator_passes_quiet_input_bit_exact() {
        let mut att = Attenuator::default();
        let quiet: Vec<f32> = (0..2_000).map(|i| 0.05 * (i as f32 * 0.03).sin()).collect();
        let mut processed = quiet.clone();
        att.process(&mut processed);
        assert_eq!(processed, quiet);
    }

    #[test]
    fn attenuator_pulls_hot_peaks_toward_target() {
        let mut att = Attenuator::default();
        let mut hot: Vec<f32> = (0..16_000)
            .map(|i| 0.98 * (i as f32 * 0.05).sin())
            .collect();
        att.process(&mut hot);
        // The first block still passes at unity gain (no lookahead), so judge
        // convergence after the attack has had a few blocks to act.
        let converged_peak = hot[1_024..].iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            converged_peak <= AUTO_GAIN_TARGET_PEAK + 0.05,
            "converged peak {converged_peak} should sit near the target"
        );
        assert!(converged_peak > 0.4, "must not crush the signal");
        assert!(
            hot.iter().all(|s| s.abs() <= 0.98 + 1e-6),
            "output must never exceed the input envelope"
        );
    }

    #[test]
    fn attenuator_never_amplifies() {
        let mut att = Attenuator::default();
        let input: Vec<f32> = (0..8_000)
            .map(|i| ((i as f32 * 0.013).sin() * 0.8).clamp(-0.99, 0.99))
            .collect();
        let mut output = input.clone();
        att.process(&mut output);
        for (in_sample, out_sample) in input.iter().zip(&output) {
            assert!(
                out_sample.abs() <= in_sample.abs() + 1e-6,
                "output {out_sample} exceeds input {in_sample}"
            );
        }
    }

    #[test]
    fn attenuator_ramp_is_zipper_free() {
        let mut att = Attenuator::default();
        let mut samples: Vec<f32> = (0..16_000).map(|_| 0.95).collect();
        att.process(&mut samples);
        for pair in samples.windows(2) {
            let jump = (pair[0] - pair[1]).abs();
            assert!(jump < 0.02, "adjacent jump {jump} too large");
        }
        let tail_peak = samples[samples.len() - 1_000..]
            .iter()
            .fold(0.0f32, |m, s| m.max(*s));
        assert!(
            (0.6..=0.95).contains(&tail_peak),
            "steady hot input should settle near the target, tail peak {tail_peak}"
        );
    }

    #[test]
    fn clip_counters_count_full_scale_boundary() {
        let counters = ClipCounters::default();
        counters.observe(&[
            CLIP_THRESHOLD,
            CLIP_THRESHOLD - 0.0001,
            -1.0,
            f32::NAN,
            0.5,
        ]);
        // Exactly at the threshold counts; just under does not; -1.0 counts;
        // NaN is not evidence of clipping; every observed sample is totaled.
        assert_eq!(counters.ratio(), 0.4);
    }

    #[test]
    fn clip_counters_accumulate_across_chunks() {
        let counters = ClipCounters::default();
        assert_eq!(counters.ratio(), 0.0, "empty capture has no clip evidence");
        counters.observe(&[]);
        assert_eq!(counters.ratio(), 0.0, "empty chunks fabricate nothing");

        counters.observe(&[1.0, 1.0, -1.0, 0.1, 0.2, 0.3]);
        counters.observe(&[0.0; 94]);
        assert_eq!(counters.ratio(), 0.03, "3 of 100 across two chunks");
    }

    #[test]
    fn clip_counters_share_across_threads_without_locks() {
        // R01 shape check: pre-DSP metering is updated through `&self`, so
        // the RT callback needs no Mutex around it.
        let counters = Arc::new(ClipCounters::default());
        let writers: Vec<_> = (0..4)
            .map(|_| {
                let counters = Arc::clone(&counters);
                std::thread::spawn(move || {
                    counters.observe(&[1.0, -1.0, 0.0, 0.25]);
                })
            })
            .collect();
        for writer in writers {
            writer.join().expect("counter writer");
        }
        assert_eq!(counters.ratio(), 0.5, "8 of 16 across four threads");
    }

    #[test]
    fn clipping_warning_scales_percentages_and_tests_boundaries() {
        assert!(clipping_warning(0.0).is_none());
        assert!(clipping_warning(f64::NAN).is_none());
        assert!(clipping_warning(-0.5).is_none());
        // Exactly at the threshold stays silent; just past it warns.
        assert!(clipping_warning(CLIP_WARNING_RATIO).is_none());
        let just_over = clipping_warning(0.020001).expect("warns just over the threshold");
        assert!(just_over.contains("2%"), "{just_over}");

        // 0.03 displays as 3%, not the 0% the old `{ratio:.0}%` produced.
        let three_percent = clipping_warning(0.03).expect("warns at 3%");
        assert!(three_percent.contains("3%"), "{three_percent}");
        assert!(!three_percent.contains("0%"), "{three_percent}");

        let everything = clipping_warning(1.0).expect("warns at 100%");
        assert!(everything.contains("100%"), "{everything}");
    }

    #[test]
    fn clipped_source_still_warns_after_attenuation() {
        // G03 acceptance: synthetic full-scale blocks keep a source-clipped
        // warning even when the processed peaks fall below 0.999. The order
        // here mirrors the audio callback: observe raw, then attenuate.
        let counters = ClipCounters::default();
        let mut attenuator = Attenuator::default();
        let mut post_dsp_full_scale = 0usize;
        let mut post_dsp_total = 0usize;

        for block in 0..100 {
            let raw: Vec<f32> = (0..AUTO_GAIN_BLOCK)
                .map(|index| {
                    if (block + index) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            counters.observe(&raw);

            let mut processed = raw;
            attenuator.process(&mut processed);
            post_dsp_total += processed.len();
            post_dsp_full_scale += processed
                .iter()
                .filter(|sample| sample.abs() >= CLIP_THRESHOLD)
                .count();
        }

        // The old post-DSP measurement would sit below the warning threshold
        // (only the first block's head passes at unity gain) …
        let post_ratio = post_dsp_full_scale as f64 / post_dsp_total as f64;
        assert!(
            post_ratio <= CLIP_WARNING_RATIO,
            "fixture must hide clipping from post-DSP counting: {post_ratio}"
        );

        // … while the pre-DSP evidence reports the whole capture.
        let warning = clipping_warning(counters.ratio()).expect("source-clipped warning");
        assert!(warning.contains("100%"), "{warning}");
    }

    #[test]
    fn ring_capacity_scales_with_device_rate() {
        for rate in [8_000u32, 16_000, 22_050, 44_100, 48_000, 96_000] {
            let capacity = ring_capacity(rate);
            assert!(
                capacity.is_power_of_two(),
                "rate {rate}: capacity {capacity} must be a power of two"
            );
            let seconds = capacity as f64 / rate as f64;
            assert!(
                (1.0..=2.2).contains(&seconds),
                "rate {rate}: ring holds {seconds}s, expected ~{CAPTURE_RING_SECONDS}s"
            );
        }
    }

    #[test]
    fn ring_overwrites_oldest_and_flags_the_gap() {
        let shared = test_shared(1_024);
        let mut callback = CallbackState::new(1);
        // Quiet ramp (peak 0.13) passes the attenuator bit-exact, so the
        // drained values are comparable by equality.
        let data: Vec<f32> = (0..1_324u32).map(|i| i as f32 * 0.0001).collect();
        for block in data.chunks(128) {
            callback.process(block, &shared);
        }
        assert_eq!(shared.written_seq.load(Ordering::Acquire), 1_324);

        let mut state = ConsumerState::default();
        shared.drain_ring(&mut state);
        assert_eq!(
            state.gaps,
            vec![CaptureGap {
                start_sample: 0,
                end_sample: 300
            }],
            "the first 300 samples were overwritten before the drain"
        );
        assert_eq!(state.gaps[0].missing_samples(), 300);
        assert_eq!(state.samples.len(), 1_024, "the ring bounds memory");
        assert_eq!(&state.samples, &data[300..], "the newest samples survive");
    }

    #[test]
    fn gap_detected_mid_capture_on_discontinuity() {
        let shared = test_shared(1_024);
        let mut callback = CallbackState::new(1);
        let data: Vec<f32> = (0..2_500u32).map(|i| i as f32 * 0.0001).collect();

        // The consumer keeps up for the first 500 samples…
        for block in data[..500].chunks(125) {
            callback.process(block, &shared);
        }
        let mut state = ConsumerState::default();
        shared.drain_ring(&mut state);
        assert_eq!(state.samples.len(), 500);
        assert!(state.gaps.is_empty());

        // …then stalls while 2 000 more arrive: the ring (1 024) laps, and
        // the survivor set starts at index 2 500 - 1 024 = 1 476.
        for block in data[500..].chunks(125) {
            callback.process(block, &shared);
        }
        shared.drain_ring(&mut state);
        assert_eq!(
            state.gaps,
            vec![CaptureGap {
                start_sample: 500,
                end_sample: 1_476
            }]
        );
        assert_eq!(state.gaps[0].missing_samples(), 976);
        // The join is flagged, and the surviving samples are exactly the
        // newest 1 024 — never a silent blend of straddling data.
        assert_eq!(&state.samples[500..], &data[1_476..]);
        assert_eq!(state.samples.len(), 500 + 1_024);
    }

    #[test]
    fn interleaved_draining_keeps_continuity_without_gaps() {
        let shared = test_shared(1_024);
        let mut callback = CallbackState::new(1);
        let mut state = ConsumerState::default();
        let mut expected = Vec::new();
        for round in 0..10u32 {
            let block: Vec<f32> = (0..256)
                .map(|i| (round * 256 + i) as f32 * 0.0001)
                .collect();
            expected.extend_from_slice(&block);
            callback.process(&block, &shared);
            shared.drain_ring(&mut state);
        }
        assert_eq!(state.samples, expected);
        assert!(state.gaps.is_empty());
        assert_eq!(state.read_pos, 2_560);
    }

    #[test]
    fn oversize_block_keeps_only_the_tail() {
        let shared = test_shared(1_024);
        let mut callback = CallbackState::new(1);
        let data: Vec<f32> = (0..3_000u32).map(|i| i as f32 * 0.0001).collect();
        // A single block larger than the whole ring (not a real device
        // shape, but the push path must stay sound): only the tail fits.
        callback.process(&data, &shared);
        assert_eq!(shared.written_seq.load(Ordering::Acquire), 3_000);

        let mut state = ConsumerState::default();
        shared.drain_ring(&mut state);
        assert_eq!(
            state.gaps,
            vec![CaptureGap {
                start_sample: 0,
                end_sample: 1_976
            }]
        );
        assert_eq!(&state.samples, &data[1_976..]);
    }

    #[test]
    fn writer_thread_drains_ring_for_latest_window() {
        let shared = test_shared(4_096);
        let writer = spawn_writer(Arc::clone(&shared));
        let handle = RecorderHandle {
            shared: Arc::clone(&shared),
            stream: None,
            writer: Some(writer),
            sample_rate: 16_000,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(500),
            journal: None,
        };

        let mut callback = CallbackState::new(1);
        let data: Vec<f32> = (0..2_000u32).map(|i| i as f32 * 0.0001).collect();
        for block in data.chunks(128) {
            callback.process(block, &shared);
        }
        // The writer — not the accessor — must have drained the ring by now.
        std::thread::sleep(WRITER_POLL_INTERVAL * 4);
        assert!(
            shared.lock_consumer().read_pos >= 2_000,
            "writer task should have drained the ring within a few polls"
        );

        assert_eq!(handle.latest_window(64), data[2_000 - 64..]);
        assert_eq!(handle.latest_window(2_048), data[..]);
        assert!(handle.latest_window(0).is_empty());

        let _ = handle.stop(); // joins the writer
    }

    #[test]
    fn stop_handshake_returns_the_full_capture() {
        let shared = test_shared(8_192);
        let writer = spawn_writer(Arc::clone(&shared));
        let handle = RecorderHandle {
            shared: Arc::clone(&shared),
            stream: None,
            writer: Some(writer),
            sample_rate: 16_000,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(500),
            journal: None,
        };

        // Stereo input exercises the downmix path, like a real device; L = R
        // so the mono average is bit-exact.
        let mut callback = CallbackState::new(2);
        let mut expected = Vec::new();
        for block in 0..20usize {
            let start = block * 256;
            let interleaved: Vec<f32> = (0..256)
                .flat_map(|i| {
                    let v = ((start + i) as f32) * 0.0001;
                    [v, v]
                })
                .collect();
            expected.extend((0..256).map(|i| ((start + i) as f32) * 0.0001));
            callback.process(&interleaved, &shared);
        }

        std::thread::sleep(Duration::from_millis(60)); // let the writer tick
        assert_eq!(handle.captured_sample_count(), 5_120);
        assert!(handle.gaps().is_empty(), "no overflow happened");

        let take = handle.stop().expect("clean stop");
        assert_eq!(take.audio.sample_rate, 16_000);
        assert_eq!(take.audio.channels, 1);
        assert_eq!(take.audio.samples, expected);
    }

    #[test]
    fn quiesce_timeout_returns_typed_error_with_samples_intact() {
        let shared = test_shared(8_192);
        let mut callback = CallbackState::new(1);
        let expected: Vec<f32> = (0..1_000u32).map(|i| i as f32 * 0.0001).collect();
        for block in expected.chunks(128) {
            callback.process(block, &shared);
        }

        // A callback that never clears `callback_alive` — a wedged driver
        // thread — must not turn into a silent empty take (R09).
        shared.callback_alive.store(true, Ordering::Release);

        let writer = spawn_writer(Arc::clone(&shared));
        let handle = RecorderHandle {
            shared,
            stream: None,
            writer: Some(writer),
            sample_rate: 48_000,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(40),
            journal: None,
        };

        match handle.stop() {
            Err(RecorderError::QuiesceTimeout {
                acknowledged_samples,
                audio,
                journal,
            }) => {
                assert_eq!(acknowledged_samples, 1_000);
                assert_eq!(audio.sample_rate, 48_000);
                assert_eq!(audio.channels, 1);
                assert_eq!(audio.samples, expected, "samples must be intact");
                assert!(journal.is_none(), "this capture had no journal");
                let message = RecorderError::QuiesceTimeout {
                    acknowledged_samples,
                    audio,
                    journal,
                }
                .to_string();
                assert!(message.contains("1000"), "{message}");
                assert!(message.contains("not lost"), "{message}");
            }
            other => panic!("expected QuiesceTimeout, got {other:?}"),
        }
    }

    #[test]
    fn quiesce_timeout_defers_teardown_and_orphans_the_ring() {
        // §3 quiesce-timeout contract: stop must not race a callback that
        // may still be executing. It returns the salvaged accumulation
        // immediately, and the `Shared` allocation survives behind the
        // producer's own `Arc` — the wedged callback keeps writing
        // harmlessly into the orphaned ring with nothing reading it.
        let shared = test_shared(8_192);
        let mut callback = CallbackState::new(1);
        let expected: Vec<f32> = (0..600u32).map(|i| i as f32 * 0.0001).collect();
        for block in expected.chunks(128) {
            callback.process(block, &shared);
        }
        let writer = spawn_writer(Arc::clone(&shared));
        // Give the writer a tick, then wedge the quiesce flag. Even without
        // the tick, the writer's exit drain inside stop() salvages the ring
        // before the timeout branch reads the accumulation.
        std::thread::sleep(Duration::from_millis(60));
        shared.callback_alive.store(true, Ordering::Release);

        let handle = RecorderHandle {
            // The test keeps the producer's view of the allocation — in a
            // real capture the callback closure holds this clone.
            shared: Arc::clone(&shared),
            stream: None,
            writer: Some(writer),
            sample_rate: 16_000,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(30),
            journal: None,
        };

        let salvaged = match handle.stop() {
            Err(RecorderError::QuiesceTimeout {
                acknowledged_samples,
                audio,
                journal,
            }) => {
                assert_eq!(acknowledged_samples, 600);
                assert!(journal.is_none(), "this capture had no journal");
                audio
            }
            other => panic!("expected QuiesceTimeout, got {other:?}"),
        };
        assert_eq!(salvaged.sample_rate, 16_000);
        assert_eq!(salvaged.channels, 1);
        assert_eq!(salvaged.samples, expected, "the salvaged take is intact");

        // The orphaned ring still accepts the wedged callback's writes; the
        // recorder is gone, nothing observes them, and the salvaged take is
        // unaffected. Completing this without panicking is the no-UB shape:
        // every access goes through the surviving Arc.
        let after: Vec<f32> = (600..900u32).map(|i| i as f32 * 0.0001).collect();
        for block in after.chunks(128) {
            callback.process(block, &shared);
        }
        assert_eq!(shared.written_seq.load(Ordering::Acquire), 900);
        assert_eq!(salvaged.samples, expected);
    }

    #[test]
    fn interleaved_drain_chunks_and_stop_tally_exactly() {
        // Partial-drain accounting: the delivered/pending watermark is
        // single-sourced, so interleaved drain_chunks callers get an exact
        // final tally — no double-counted span, no lost span, and no gap
        // fabricated at a handout boundary.
        let shared = test_shared(4_096);
        let mut callback = CallbackState::new(1);
        let handle = test_handle(Arc::clone(&shared), 16_000);
        let data: Vec<f32> = (0..1_000u32).map(|i| i as f32 * 0.0001).collect();

        let mut collected = Vec::new();
        for (round, block) in data.chunks(125).enumerate() {
            callback.process(block, &shared);
            if round % 2 == 0 {
                for chunk in handle.drain_chunks() {
                    collected.extend(chunk);
                }
            }
        }
        assert!(handle.gaps().is_empty(), "no overflow: no gaps fabricated");
        let take = handle.stop().expect("stop after incremental drains");
        collected.extend_from_slice(&take.audio.samples);
        assert_eq!(collected.len(), 1_000, "exact sample count");
        assert_eq!(collected, data, "drained + pending reassemble the take");
    }

    #[test]
    fn fully_drained_take_is_not_reported_empty() {
        // A caller that took the whole take via drain_chunks owns those
        // samples; stop must not fabricate RecorderError::Empty ("the device
        // produced nothing") for audio it already handed out.
        let shared = test_shared(4_096);
        let mut callback = CallbackState::new(1);
        let handle = test_handle(Arc::clone(&shared), 16_000);
        let data: Vec<f32> = (0..250u32).map(|i| i as f32 * 0.0001).collect();
        for block in data.chunks(125) {
            callback.process(block, &shared);
        }
        assert_eq!(handle.drain_chunks().concat(), data);
        let take = handle.stop().expect("a fully drained take stops cleanly");
        assert!(take.audio.samples.is_empty(), "nothing pending remains");
    }

    #[test]
    fn writer_panic_is_surfaced_and_samples_still_returned() {
        // The writer dying must not be swallowed: the join failure lands in
        // the first-error-wins capture-error slot while the take that made
        // it into the ring is still returned (stop's final drain recovers
        // what the dead writer left behind).
        let shared = test_shared(4_096);
        let mut callback = CallbackState::new(1);
        let expected: Vec<f32> = (0..500u32).map(|i| i as f32 * 0.0001).collect();
        for block in expected.chunks(125) {
            callback.process(block, &shared);
        }

        let writer = std::thread::Builder::new()
            .name("test-panicking-writer".to_string())
            .spawn(|| panic!("writer exploded"))
            .expect("spawn panicking writer");
        let handle = RecorderHandle {
            shared: Arc::clone(&shared),
            stream: None,
            writer: Some(writer),
            sample_rate: 16_000,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(200),
            journal: None,
        };

        let take = handle.stop().expect("samples that made it are returned");
        assert_eq!(take.audio.samples, expected, "ring contents recovered");

        let slot = shared.lock_consumer().stream_error.clone();
        assert!(
            slot.as_deref()
                .is_some_and(|message| message.contains("writer exploded")),
            "the writer failure must be surfaced, got {slot:?}"
        );
    }

    #[test]
    fn zero_channel_configs_are_rejected_not_treated_as_mono() {
        // A device config with no channels must fail stream-open loudly…
        let err = reject_zero_channels(0).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("zero-channel"),
            "{err}"
        );
        // …while every real channel count opens as before.
        for channels in 1..=8u16 {
            assert!(reject_zero_channels(channels).is_ok());
        }
    }

    #[test]
    fn callback_clip_metering_stays_pre_dsp() {
        // G03 through the real callback path: full-scale input keeps a 100%
        // source-clip ratio while the attenuated copy stored in the ring
        // falls below the warning threshold.
        let shared = test_shared(8_192);
        let mut callback = CallbackState::new(1);
        let block: Vec<f32> = (0..AUTO_GAIN_BLOCK)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        for _ in 0..100 {
            callback.process(&block, &shared);
        }
        assert!((shared.clip.ratio() - 1.0).abs() < 1e-9);

        let mut state = ConsumerState::default();
        shared.drain_ring(&mut state);
        let post_dsp = state
            .samples
            .iter()
            .filter(|sample| sample.abs() >= CLIP_THRESHOLD)
            .count() as f64
            / state.samples.len() as f64;
        assert!(
            post_dsp <= CLIP_WARNING_RATIO,
            "fixture must hide clipping from post-DSP counting: {post_dsp}"
        );
    }

    #[test]
    fn drain_chunks_ownership_and_stop_returns_only_pending() {
        let shared = test_shared(4_096);
        let mut callback = CallbackState::new(1);
        let data: Vec<f32> = (0..400u32).map(|i| i as f32 * 0.0001).collect();
        for block in data[..300].chunks(128) {
            callback.process(block, &shared);
        }

        let handle = test_handle(Arc::clone(&shared), 16_000);
        let drained = handle.drain_chunks();
        assert_eq!(drained.len(), 1, "consolidated into one chunk");
        assert_eq!(drained[0], data[..300]);

        for block in data[300..].chunks(128) {
            callback.process(block, &shared);
        }
        let take = handle.stop().expect("stop with pending samples");
        assert_eq!(take.audio.samples, data[300..], "stop returns only the pending");
    }

    #[test]
    fn stream_errors_are_surfaced_as_typed_state() {
        // E01/G01: the CPAL error callback posts typed state instead of
        // printing; the first error wins.
        let shared = test_shared(4_096);
        shared.record_stream_error("input device detached".to_string());
        shared.record_stream_error("a later failure".to_string());
        let handle = test_handle(shared, 16_000);
        assert_eq!(
            handle.capture_error().as_deref(),
            Some("input device detached")
        );
    }

    #[test]
    fn acknowledged_samples_stay_zero_without_a_journal() {
        // Honesty baseline: acknowledgment is a durability claim, so a
        // capture with no journal acknowledges nothing, no matter how much
        // it captured — "acknowledged == accumulated len" is exactly the
        // conflation this accessor exists to remove.
        let shared = test_shared(4_096);
        let mut callback = CallbackState::new(1);
        let data: Vec<f32> = (0..1_000u32).map(|i| i as f32 * 0.0001).collect();
        for block in data.chunks(125) {
            callback.process(block, &shared);
        }
        let writer = spawn_writer(Arc::clone(&shared));
        let handle = RecorderHandle {
            shared,
            stream: None,
            writer: Some(writer),
            sample_rate: 16_000,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(200),
            journal: None,
        };
        std::thread::sleep(Duration::from_millis(60)); // let the writer drain
        assert_eq!(handle.captured_sample_count(), 1_000);
        assert_eq!(handle.acknowledged_samples(), 0);
        let take = handle.stop().expect("stop");
        assert_eq!(take.audio.samples, data, "the take itself is intact");
        assert!(take.journal.is_none());
    }

    #[test]
    fn journaled_capture_advances_ack_only_across_fsynced_boundaries() {
        // Ack semantics through the real writer task with a fault-injecting
        // sink: acknowledged ≤ written always; it advances after a
        // boundary's fsync; a storage fault freezes it at the last good
        // boundary while capture keeps running in memory and the fault is
        // surfaced through the capture-error slot (§3 storage-fault
        // honesty).
        let dir = TempDir::new().expect("tempdir");
        let fail = Arc::new(AtomicBool::new(false));
        let id = "j_recorder_fault".to_string();
        let (sink, path) =
            FaultSink::create(dir.path(), &id, Arc::clone(&fail)).expect("fault sink");
        let writer = JournalWriter::over_sink(sink, id, path.clone(), 16_000)
            .expect("writer with header fsynced");

        let shared = test_shared(32_768);
        let mut callback = CallbackState::new(1);
        let first: Vec<f32> = (0..20_000u32).map(|i| i as f32 * 0.00001).collect();
        // 20 000 samples = 80 000 payload bytes > the 64 KiB cadence: the
        // writer's next tick must append + fsync a boundary.
        callback.process(&first, &shared);
        let handle = journaled_test_handle(Arc::clone(&shared), writer, 16_000);

        let acknowledged =
            wait_until(|| handle.acknowledged_samples(), |acked| *acked == 20_000);
        assert_eq!(acknowledged, 20_000, "boundary fsync published the ack");
        assert!(acknowledged <= handle.captured_sample_count());

        // The disk starts failing. Capture keeps producing; the journal
        // drops; acknowledgment must freeze at the last good boundary.
        fail.store(true, Ordering::Release);
        let more: Vec<f32> = (20_000u32..28_000).map(|i| i as f32 * 0.00001).collect();
        for block in more.chunks(128) {
            callback.process(block, &shared);
        }
        std::thread::sleep(Duration::from_millis(350)); // past the time cadence
        assert_eq!(handle.captured_sample_count(), 28_000, "capture is still live");
        assert_eq!(
            handle.acknowledged_samples(),
            20_000,
            "acknowledgment frozen at the last fsynced boundary"
        );
        let fault = handle.capture_error().expect("fault surfaced");
        assert!(fault.contains("journal"), "{fault}");
        assert!(fault.contains("20 000") || fault.contains("20000"), "{fault}");

        // The take itself is not lost: memory accumulation kept running.
        let take = handle.stop().expect("stop with a faulted journal");
        assert_eq!(take.audio.samples.len(), 28_000);
        let report = take.journal.expect("journal report present");
        assert!(!report.finalized, "a faulted journal has no trailer");
        assert_eq!(report.acknowledged_samples, 20_000);
        assert!(report.fault.as_deref().is_some_and(|f| f.contains("journal")));

        // What the journal file can verify on recovery: exactly the frozen
        // boundary's samples.
        let parsed = crate::journal::read_journal(&path).expect("parse the faulted journal");
        assert_eq!(parsed.samples.len(), 20_000);
        assert_eq!(parsed.samples, first);
        assert!(!parsed.finalized);
    }

    #[test]
    fn stop_finalizes_the_journal_with_a_verifiable_trailer() {
        let dir = TempDir::new().expect("tempdir");
        let writer =
            JournalWriter::<FileSink>::create(dir.path(), 16_000).expect("create journal");

        let shared = test_shared(8_192);
        let mut callback = CallbackState::new(1);
        let expected: Vec<f32> = (0..5_000u32).map(|i| i as f32 * 0.00002).collect();
        for block in expected.chunks(250) {
            callback.process(block, &shared);
        }
        let handle = journaled_test_handle(Arc::clone(&shared), writer, 16_000);

        let acknowledged = wait_until(|| handle.acknowledged_samples(), |acked| *acked > 0);
        assert!(acknowledged <= 5_000, "acknowledged ≤ written at all times");

        let take = handle.stop().expect("clean stop");
        assert_eq!(take.audio.samples, expected);
        let report = take.journal.expect("journal report");
        assert!(report.finalized, "trailer written, file + dir fsynced");
        assert_eq!(report.acknowledged_samples, 5_000);
        assert_eq!(report.fault, None);

        let parsed =
            crate::journal::read_journal(&report.path).expect("parse finalized journal");
        assert!(parsed.finalized);
        assert_eq!(parsed.samples, expected, "journal round-trips byte-exact");
        assert_eq!(parsed.torn_tail_bytes, 0);
    }

    /// A journal sink whose `sync` parks on a test-held gate: the writer
    /// enters the fsync and cannot leave until the test (or a watchdog)
    /// opens the gate. Appends succeed, so the boundary itself is written.
    struct GatedSink {
        in_sync: Arc<AtomicBool>,
        gate: Arc<(Mutex<bool>, Condvar)>,
    }

    impl JournalSink for GatedSink {
        fn append(&mut self, _bytes: &[u8]) -> std::io::Result<()> {
            Ok(())
        }

        fn sync(&mut self) -> std::io::Result<()> {
            self.in_sync.store(true, Ordering::Release);
            let (lock, cv) = &*self.gate;
            let mut open = lock.lock().expect("gate lock");
            while !*open {
                open = cv.wait(open).expect("gate wait");
            }
            self.in_sync.store(false, Ordering::Release);
            Ok(())
        }

        fn sync_parent_dir(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn journal_fsync_in_flight_does_not_stall_the_metering_path() {
        // #203: `latest_window` is polled by the UI render thread every
        // animation frame while recording, and journal boundaries fsync at
        // least every 250 ms — on slow storage, hundreds of ms. With the
        // boundary fsync parked here mid-flight, the accessor must complete
        // without waiting on the consumer lock; the old writer loop held
        // that lock across append + fsync, janking the render thread every
        // boundary.
        let in_sync = Arc::new(AtomicBool::new(false));
        // The gate starts open so `over_sink`'s header fsync can pass; it is
        // closed before any capture data exists.
        let gate = Arc::new((Mutex::new(true), Condvar::new()));
        let writer = JournalWriter::over_sink(
            GatedSink {
                in_sync: Arc::clone(&in_sync),
                gate: Arc::clone(&gate),
            },
            "j_gated".to_string(),
            PathBuf::from("gated-sink-has-no-file.sj"),
            16_000,
        )
        .expect("writer over the gated sink");

        // 32 768-slot ring: 16 385 samples fit without overflowing.
        let shared = test_shared(32_768);
        let handle = journaled_test_handle(Arc::clone(&shared), writer, 16_000);
        let mut callback = CallbackState::new(1);
        // 16 385 samples = 65 540 payload bytes > the 64 KiB boundary
        // cadence: the writer's tick must append and fsync a boundary.
        let expected: Vec<f32> = (0..16_385u32).map(|i| i as f32 * 0.00001).collect();
        for block in expected.chunks(2_000) {
            callback.process(block, &shared);
        }

        // Close the gate (no boundary can be due yet — nothing has been
        // appended), then wait until the writer is parked inside the
        // boundary fsync.
        *gate.0.lock().expect("close the gate") = false;
        wait_until(
            || in_sync.load(Ordering::Acquire),
            |parked| *parked,
        );

        // Watchdog: if `latest_window` does block on the consumer lock,
        // release the fsync anyway so the failure below is a latency
        // assertion instead of a hung suite.
        {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(2));
                *gate.0.lock().expect("watchdog gate") = true;
                gate.1.notify_all();
            });
        }

        let started = Instant::now();
        let window = handle.latest_window(64);
        let metering_latency = started.elapsed();
        assert_eq!(window.len(), 64, "the accessor still reads fresh samples");
        assert!(
            metering_latency < Duration::from_millis(500),
            "latest_window waited {metering_latency:?} behind an in-flight journal fsync"
        );
        // The error-callback path takes the same lock (E01/G01): it must
        // not stall behind the fsync either.
        handle.capture_error();

        // Release the fsync and stop; the memory sink still finalizes, so
        // the watermark agreement over the whole take holds.
        *gate.0.lock().expect("open the gate") = true;
        gate.1.notify_all();
        let take = handle.stop().expect("clean stop after the gate opened");
        assert_eq!(take.audio.samples, expected, "the take itself is intact");
        let report = take.journal.expect("journal report");
        assert!(report.finalized);
        assert_eq!(report.acknowledged_samples, 16_385);
    }

    #[test]
    fn stop_does_not_finalize_the_journal_before_the_producer_quiesces() {
        // #204 regression: `stopping` used to be stored before the stream
        // was dropped, so the writer's 25 ms poll could land mid-capture and
        // run its "final" drain + finalize while the callback was still
        // producing. The tail then reached the returned take but never the
        // journal: `finalized: true` with `acknowledged_samples` short of
        // the take — and a crash-recovery resurrection would silently come
        // back truncated. `stopping` must stay false while the (simulated)
        // callback is in flight, and samples delivered during that window
        // must be part of the finalized journal.
        let dir = TempDir::new().expect("tempdir");
        let writer =
            JournalWriter::<FileSink>::create(dir.path(), 16_000).expect("create journal");

        let shared = test_shared(8_192);
        let mut callback = CallbackState::new(1);
        let head: Vec<f32> = (0..1_000u32).map(|i| i as f32 * 0.0001).collect();
        for block in head.chunks(125) {
            callback.process(block, &shared);
        }
        let handle = journaled_test_handle(Arc::clone(&shared), writer, 16_000);

        // The producer thread simulates one long in-flight callback
        // invocation straddling the stop call: `callback_alive` is held true
        // while `stop` runs, the tail is delivered mid-window, and only then
        // does the invocation complete. (The handle itself is !Send through
        // cpal::Stream, so `stop` stays on this thread.)
        let producer = {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || {
                shared.callback_alive.store(true, Ordering::Release);
                // Give `stop` time to reach its quiesce wait — and, under
                // the old ordering, to have stored `stopping` where the
                // writer's 25 ms poll would already have acted on it.
                std::thread::sleep(Duration::from_millis(120));
                assert!(
                    !shared.stopping.load(Ordering::Acquire),
                    "the writer must not be signaled to finalize while the producer \
                     may still deliver"
                );
                // Deliver the tail as one block, then complete the callback.
                let tail: Vec<f32> = (1_000..1_250u32).map(|i| i as f32 * 0.0001).collect();
                callback.process(&tail, &shared);
                shared.callback_alive.store(false, Ordering::Release);
                tail
            })
        };

        // Enter the handshake only once the callback is in flight, so the
        // quiesce wait is genuinely exercised.
        wait_until(
            || shared.callback_alive.load(Ordering::Acquire),
            |alive| *alive,
        );
        let take = handle.stop().expect("clean stop");
        let tail = producer.join().expect("producer thread");

        let mut expected = head.clone();
        expected.extend_from_slice(&tail);
        assert_eq!(take.audio.samples, expected, "the take carries the tail");

        let report = take.journal.expect("journal report");
        assert!(report.finalized, "clean stop finalizes the journal");
        assert_eq!(
            report.acknowledged_samples,
            expected.len() as u64,
            "the finalized journal must cover the tail the in-flight callback delivered"
        );
        assert_eq!(report.fault, None);

        // The durable copy itself, not just the report's claim.
        let parsed =
            crate::journal::read_journal(&report.path).expect("parse finalized journal");
        assert!(parsed.finalized);
        assert_eq!(parsed.samples, expected, "the journal round-trips the whole take");
        assert_eq!(parsed.torn_tail_bytes, 0);
    }

    #[test]
    fn quiesce_timeout_salvage_persists_an_interrupted_linked_take() {
        // R17 fix, full chain: a wedged callback must not silently discard
        // acknowledged audio. stop() defers teardown and returns the
        // salvaged samples plus the already-finalized journal; the caller
        // (here, the same chain the app runs) persists them as an
        // interrupted session linked to the journal, and the startup scan
        // then recovers nothing — the linkage prevents a duplicate.
        let journals_dir = TempDir::new().expect("journals tempdir");
        let writer = JournalWriter::<FileSink>::create(journals_dir.path(), 16_000)
            .expect("create journal");

        let shared = test_shared(8_192);
        let mut callback = CallbackState::new(1);
        let expected: Vec<f32> = (0..1_000u32).map(|i| i as f32 * 0.0001).collect();
        for block in expected.chunks(128) {
            callback.process(block, &shared);
        }
        std::thread::sleep(Duration::from_millis(60)); // let the writer drain
        shared.callback_alive.store(true, Ordering::Release); // wedge

        let mut wedged = journaled_test_handle(Arc::clone(&shared), writer, 16_000);
        wedged.quiesce_timeout = Duration::from_millis(30);

        match wedged.stop() {
            Err(RecorderError::QuiesceTimeout {
                acknowledged_samples,
                audio,
                journal,
            }) => {
                assert_eq!(acknowledged_samples, 1_000);
                assert_eq!(audio.samples, expected, "salvaged audio intact");
                let report = journal.expect("journal report rides along");
                assert!(report.finalized, "writer finalized before returning");
                assert_eq!(report.acknowledged_samples, 1_000);

                // The app-side salvage: encode, persist, mark interrupted.
                let wav = crate::audio::encode_wav_16k(&audio).expect("encode salvage");
                let store_dir = TempDir::new().expect("store tempdir");
                let store = crate::storage::FileSessionStore::open(store_dir.path())
                    .expect("open store");
                let session = store
                    .create_with_journal(wav, Some(62.5), Some(&report.id))
                    .expect("persist salvage")
                    ;
                let session = store
                    .mark_interrupted(
                        &session.id,
                        "The microphone did not stop cleanly; the salvaged take was kept.",
                    )
                    .expect("mark interrupted");
                assert_eq!(
                    session.status,
                    crate::storage::SessionStatus::Interrupted
                );
                assert_eq!(session.journal_id.as_deref(), Some(report.id.as_str()));

                // No double recovery: the linked journal is skipped.
                let rerun = crate::journal::recover_interrupted_takes(
                    &store,
                    journals_dir.path(),
                )
                .expect("recovery rerun");
                assert!(rerun.recovered.is_empty(), "linked journals are skipped");
            }
            other => panic!("expected QuiesceTimeout, got {other:?}"),
        }
    }

    /// Live round-trip against the real microphone. Skips itself (rather than
    /// failing) in environments without an input device.
    #[test]
    fn live_capture_produces_mono_samples() {
        let handle = match start_recording() {
            Ok(handle) => handle,
            Err(err) => {
                eprintln!("skipping live capture test: {err}");
                return;
            }
        };

        std::thread::sleep(Duration::from_millis(300));

        assert!(handle.elapsed() >= Duration::from_millis(250));
        assert!(handle.sample_rate() > 0);
        assert!(
            !handle.latest_window(64).is_empty(),
            "expected recent samples for level metering"
        );
        assert!(!handle.latest_window(4_096).is_empty());

        let take = handle.stop().expect("stop after live capture");
        let audio = take.audio;
        assert_eq!(audio.channels, 1);
        assert!(!audio.samples.is_empty());
        assert!(
            audio.sample_rate == PREFERRED_SAMPLE_RATE || audio.sample_rate >= 8_000,
            "unexpected device sample rate {}",
            audio.sample_rate
        );
    }
}
