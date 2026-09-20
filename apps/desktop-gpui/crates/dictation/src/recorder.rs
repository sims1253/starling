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
//!   [`RecorderHandle::gaps`], never silently joined. Phase 2 replaces the
//!   accumulation with the on-disk journal at this exact seam.
//! - `stop` is an explicit handshake (R09): declare
//!   `finalSampleIndex = written_seq`, drop the CPAL stream, wait (bounded)
//!   for `callback_alive == false`, join the writer, then drain what is
//!   left. A quiesce timeout degrades to a typed
//!   [`RecorderError::QuiesceTimeout`] carrying the captured audio. A
//!   silent empty take is structurally impossible: the capture path holds
//!   no mutex that could poison, and the consumer state is always
//!   recovered from a poisoned lock rather than treated as empty.
//!
//! The stream is requested as f32 / 1 channel / 16 kHz when the device
//! supports it; otherwise the device default config is used and any format
//! (integer samples, >1 channels) is converted to mono f32 inside the
//! callback. Resampling to 16 kHz happens in [`crate::audio`] at
//! WAV-encode time, driven by the UI layer — not here.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

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

#[derive(Debug, thiserror::Error)]
pub enum RecorderError {
    #[error("{0}")]
    Device(String),
    #[error("No microphone audio was captured.")]
    Empty,
    /// The stop handshake timed out waiting for the audio callback to
    /// quiesce after the stream was dropped (R09). The acknowledged samples
    /// are preserved in `audio` — a wedged callback must never turn into a
    /// silent empty take.
    #[error(
        "The microphone did not stop cleanly within the quiesce timeout; {acknowledged_samples} \
         captured samples are preserved in this error and were not lost."
    )]
    QuiesceTimeout {
        /// Total samples this capture acknowledged (acknowledged-sample
        /// groundwork for §3's `capture.progress{ackSamples}`).
        acknowledged_samples: u64,
        /// Everything captured and drained, ready to encode.
        audio: crate::audio::PcmAudio,
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
    /// [`RecorderHandle::drain_chunks`]; `stop` returns only what is still
    /// pending after that watermark.
    delivered: usize,
    /// First device-side error posted by the CPAL error callback (E01/G01).
    stream_error: Option<String>,
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
    /// True while a callback invocation is executing. The stop handshake
    /// waits for this to clear after the stream is dropped (R09).
    callback_alive: AtomicBool,
    /// Pre-DSP clipping evidence, updated from the callback (G03).
    clip: ClipCounters,
    /// Consumer-side accumulation and gap bookkeeping.
    consumer: Mutex<ConsumerState>,
    /// Wakes the writer promptly for its final drain at stop.
    wake: Condvar,
    /// Set by `stop` before the stream is dropped; the writer exits on it.
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

/// The dedicated consumer (§3 G01's "writer task"; phase 1 accumulates in
/// memory where phase 2 will own the journal). Drains the ring every poll
/// interval so a stalled UI — occluded window, paused animation frames —
/// cannot overflow the capture ring, records gap spans on sequence
/// discontinuities, and exits within one interval of `stopping`.
fn writer_loop(shared: Arc<Shared>) {
    let mut guard = shared.lock_consumer();
    loop {
        shared.drain_ring(&mut guard);
        if shared.stopping.load(Ordering::Acquire) {
            shared.drain_ring(&mut guard);
            return;
        }
        guard = shared
            .wake
            .wait_timeout(guard, WRITER_POLL_INTERVAL)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .0;
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
/// `channels <= 1` passes through. `out` is cleared first; its capacity is
/// retained so repeated calls do not allocate.
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

/// Live microphone capture handle returned by [`start_recording`].
pub struct RecorderHandle {
    shared: Arc<Shared>,
    stream: Option<cpal::Stream>,
    writer: Option<JoinHandle<()>>,
    sample_rate: u32,
    started_at: Instant,
    quiesce_timeout: Duration,
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

    /// Gap spans where ring overflow dropped samples, in capture order.
    /// Empty means the take is a continuous stream; any entry means the
    /// output joins surviving audio across a flagged hole (G01: gaps are
    /// surfaced, never silently repaired).
    pub fn gaps(&self) -> Vec<CaptureGap> {
        let mut guard = self.shared.lock_consumer();
        self.shared.drain_ring(&mut guard);
        guard.gaps.clone()
    }

    /// The first device-side capture error posted by the CPAL error
    /// callback, if any (E01/G01) — the UI can stop pretending capture is
    /// live once this is set.
    pub fn capture_error(&self) -> Option<String> {
        self.shared.lock_consumer().stream_error.clone()
    }

    /// Drains the mono f32 samples accumulated since the last call, in
    /// order, as one consolidated chunk.
    ///
    /// The simplest UI wiring is to never call this while recording and take
    /// the whole recording from `stop` (exactly like `useRecorder.ts`, which
    /// only reads `chunksRef` at stop). If you do drain per frame, you own
    /// the drained samples: `stop` returns only what is still pending.
    pub fn drain_chunks(&self) -> Vec<Vec<f32>> {
        let mut guard = self.shared.lock_consumer();
        self.shared.drain_ring(&mut guard);
        if guard.delivered >= guard.samples.len() {
            return Vec::new();
        }
        let chunk = guard.samples[guard.delivered..].to_vec();
        guard.delivered = guard.samples.len();
        vec![chunk]
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

    /// Stops the capture with the §3 R09 handshake:
    ///
    /// 1. signal the writer task to take its final drain and exit;
    /// 2. declare `finalSampleIndex = written_seq`;
    /// 3. drop the CPAL stream;
    /// 4. wait (bounded by the quiesce timeout) for `callback_alive` to
    ///    clear;
    /// 5. join the writer and drain whatever is left;
    /// 6. return the pending samples as mono [`crate::audio::PcmAudio`] at
    ///    the device rate.
    ///
    /// A quiesce timeout returns [`RecorderError::QuiesceTimeout`] with the
    /// acknowledged samples preserved inside the error — never a silent
    /// empty result. [`RecorderError::Empty`] means the device produced
    /// nothing at all.
    pub fn stop(mut self) -> Result<crate::audio::PcmAudio, RecorderError> {
        self.shared.stopping.store(true, Ordering::Release);
        let final_sample_index = self.shared.written_seq.load(Ordering::Acquire);

        if let Some(stream) = self.stream.take() {
            // Explicit stop; pause is unsupported on some backends, and
            // dropping the stream releases the device either way. Backends
            // that join their callback thread make step (4) trivially pass;
            // the bounded wait covers the rest.
            let _ = stream.pause();
            drop(stream);
        }

        let quiesced = wait_callback_quiesce(&self.shared, self.quiesce_timeout);

        // Wake the writer under the lock so a thread entering wait_timeout
        // cannot miss the notification; it then drains once more and exits.
        {
            let _guard = self.shared.lock_consumer();
            self.shared.wake.notify_all();
        }
        if let Some(writer) = self.writer.take() {
            // Exits within one WRITER_POLL_INTERVAL of `stopping`.
            let _ = writer.join();
        }

        // Final drain: the producer has quiesced (or is wedged — everything
        // below the published frontier is stable either way). This also
        // recovers ring contents even if the writer died unexpectedly.
        {
            let mut guard = self.shared.lock_consumer();
            self.shared.drain_ring(&mut guard);
        }
        let captured = self.shared.written_seq.load(Ordering::Acquire);
        debug_assert!(captured >= final_sample_index);

        let guard = self.shared.lock_consumer();
        let acknowledged_samples = guard.samples.len() as u64;
        let pending = guard.samples[guard.delivered..].to_vec();
        drop(guard);

        let audio = crate::audio::PcmAudio {
            samples: pending,
            sample_rate: self.sample_rate,
            channels: 1,
        };

        if !quiesced {
            return Err(RecorderError::QuiesceTimeout {
                acknowledged_samples,
                audio,
            });
        }
        if audio.samples.is_empty() {
            return Err(RecorderError::Empty);
        }
        Ok(audio)
    }
}

/// Opens the default input device and starts a mono capture stream.
pub fn start_recording() -> Result<RecorderHandle, RecorderError> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| {
        RecorderError::Device(
            "No microphone was found. Connect an input device and try again.".to_string(),
        )
    })?;

    let supported = pick_input_config(&device)?;
    let sample_format = supported.sample_format();
    let stream_config = supported.config();

    let shared = Arc::new(Shared::new(ring_capacity(stream_config.sample_rate.0)));

    let stream = open_stream(&device, sample_format, &stream_config, &shared)?;

    // The writer task owns the take for the whole session (§3 G01). Starting
    // it before `play` means the first callback already has a drainer; a
    // failure to spawn it is a device-level error, not a degraded capture.
    let writer = std::thread::Builder::new()
        .name("starling-capture-writer".to_string())
        .spawn({
            let shared = Arc::clone(&shared);
            move || writer_loop(shared)
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
        sample_rate: stream_config.sample_rate.0,
        started_at: Instant::now(),
        quiesce_timeout: QUIESCE_TIMEOUT,
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

/// Build the input stream for whichever sample format the config settled on.
fn open_stream(
    device: &cpal::Device,
    sample_format: cpal::SampleFormat,
    config: &cpal::StreamConfig,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, RecorderError> {
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

    /// A `Shared` with an arbitrary (power-of-two) ring capacity for
    /// callback-side simulation without a real device.
    fn test_shared(capacity: usize) -> Arc<Shared> {
        Arc::new(Shared::new(capacity))
    }

    /// A handle wired to a simulated capture: no stream, no writer thread.
    fn test_handle(shared: Arc<Shared>, sample_rate: u32) -> RecorderHandle {
        RecorderHandle {
            shared,
            stream: None,
            writer: None,
            sample_rate,
            started_at: Instant::now(),
            quiesce_timeout: Duration::from_millis(500),
        }
    }

    /// Spawn the real writer task against `shared`.
    fn spawn_writer(shared: Arc<Shared>) -> JoinHandle<()> {
        std::thread::Builder::new()
            .name("test-capture-writer".to_string())
            .spawn(move || writer_loop(shared))
            .expect("spawn writer")
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

        let audio = handle.stop().expect("clean stop");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples, expected);
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
        };

        match handle.stop() {
            Err(RecorderError::QuiesceTimeout {
                acknowledged_samples,
                audio,
            }) => {
                assert_eq!(acknowledged_samples, 1_000);
                assert_eq!(audio.sample_rate, 48_000);
                assert_eq!(audio.channels, 1);
                assert_eq!(audio.samples, expected, "samples must be intact");
                let message = RecorderError::QuiesceTimeout {
                    acknowledged_samples,
                    audio,
                }
                .to_string();
                assert!(message.contains("1000"), "{message}");
                assert!(message.contains("not lost"), "{message}");
            }
            other => panic!("expected QuiesceTimeout, got {other:?}"),
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
        let audio = handle.stop().expect("stop with pending samples");
        assert_eq!(audio.samples, data[300..], "stop returns only the pending");
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

        let audio = handle.stop().expect("stop after live capture");
        assert_eq!(audio.channels, 1);
        assert!(!audio.samples.is_empty());
        assert!(
            audio.sample_rate == PREFERRED_SAMPLE_RATE || audio.sample_rate >= 8_000,
            "unexpected device sample rate {}",
            audio.sample_rate
        );
    }
}
