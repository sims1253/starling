//! Microphone capture, ported from `apps/desktop/src/useRecorder.ts`
//! (`getUserMedia` + `ScriptProcessorNode` + `AnalyserNode`) — see
//! `apps/desktop-gpui/PORT.md`.
//!
//! `start_recording` opens the default input device and feeds every cpal audio
//! callback into two sinks, mirroring the two TS streams:
//!
//! - cumulative mono chunks — `drain_chunks` (the `chunksRef` array); `stop`
//!   drains them into the final [`crate::audio::PcmAudio`]
//! - a small ring of recent samples — `latest_window`, for the live level
//!   metering the UI does with [`crate::fft`] (the old `AnalyserNode`)
//!
//! The stream is requested as f32 / 1 channel / 16 kHz when the device
//! supports it; otherwise the device default config is used and any format
//! (integer samples, >1 channels) is converted to mono f32 inside the
//! callback, where an attenuation-only auto gain also keeps hot sources below
//! the WAV clamp. Resampling to 16 kHz happens in [`crate::audio`] at
//! WAV-encode time, driven by the UI layer — not here.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Sample rate requested from the microphone. Devices that cannot capture
/// natively at 16 kHz are used at their own rate; the UI resamples when
/// encoding the WAV.
const PREFERRED_SAMPLE_RATE: u32 = 16_000;

/// Maximum samples kept in the live-level ring (`latest_window`).
const RING_CAPACITY: usize = 4_096;

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
#[derive(Debug, Default)]
pub struct ClipCounters {
    total: u64,
    clipped: u64,
}

impl ClipCounters {
    /// Counts full-scale samples against [`CLIP_THRESHOLD`]. A NaN sample is
    /// not evidence of clipping (the comparison is false), matching how the
    /// encoder treats non-finite input.
    pub fn observe(&mut self, samples: &[f32]) {
        for &sample in samples {
            if sample.abs() >= CLIP_THRESHOLD {
                self.clipped += 1;
            }
        }
        self.total += samples.len() as u64;
    }

    /// Clipped fraction in 0..=1, or 0.0 when nothing was observed (also the
    /// empty-recording case, so no warning can be fabricated).
    pub fn ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.clipped as f64 / self.total as f64
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

#[derive(Debug, thiserror::Error)]
pub enum RecorderError {
    #[error("{0}")]
    Device(String),
    #[error("No microphone audio was captured.")]
    Empty,
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

/// State shared between the cpal audio callback and the [`RecorderHandle`].
struct Shared {
    /// Every mono chunk in capture order. Unbounded, like `chunksRef` in the
    /// TS hook; drained by `drain_chunks`/`stop`.
    chunk_tx: mpsc::Sender<Vec<f32>>,
    chunk_rx: Mutex<mpsc::Receiver<Vec<f32>>>,
    /// Most recent samples, capped at [`RING_CAPACITY`], for level metering.
    ring: Mutex<VecDeque<f32>>,
    /// Capture auto-gain, driven inside the audio callback.
    attenuator: Mutex<Attenuator>,
    /// Pre-DSP clipping evidence, observed inside the audio callback before
    /// the attenuator runs (G03).
    clip: Mutex<ClipCounters>,
}

/// Average interleaved frames down to mono. Frames shorter than `channels`
/// (a trailing partial frame) are dropped; `channels <= 1` passes through.
fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
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

/// Append `chunk` to `ring`, dropping the oldest samples past `capacity`.
fn push_ring(ring: &mut VecDeque<f32>, chunk: &[f32], capacity: usize) {
    for &sample in chunk {
        if ring.len() == capacity {
            ring.pop_front();
        }
        ring.push_back(sample);
    }
}

/// Copy the last `n` samples of `ring`, in order.
fn take_latest(ring: &VecDeque<f32>, n: usize) -> Vec<f32> {
    let start = ring.len().saturating_sub(n);
    ring.iter().skip(start).copied().collect()
}

impl Shared {
    /// Runs inside the cpal audio callback — must stay cheap: one conversion
    /// pass, one channel send, one short ring lock. `mono` is already
    /// downmixed to 1 channel by the caller.
    fn push_chunk(&self, mono: Vec<f32>) {
        // (a) cumulative chunks for the final recording (unbounded channel;
        // send only fails if the receiver is gone, which cannot happen while
        // the handle — and therefore the stream — is alive).
        let _ = self.chunk_tx.send(mono.clone());
        // (b) recent samples for the live level meter.
        if let Ok(mut ring) = self.ring.lock() {
            push_ring(&mut ring, &mono, RING_CAPACITY);
        }
    }
}

/// Live microphone capture handle returned by [`start_recording`].
pub struct RecorderHandle {
    shared: Arc<Shared>,
    stream: Option<cpal::Stream>,
    sample_rate: u32,
    started_at: Instant,
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
    /// which consumes the handle.
    pub fn source_clip_ratio(&self) -> f64 {
        match self.shared.clip.lock() {
            Ok(clip) => clip.ratio(),
            Err(_) => 0.0,
        }
    }

    /// Wall-clock time since `start_recording`, like
    /// `performance.now() - startedAt` in the TS hook.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Drains mono f32 chunks accumulated since the last call, in order.
    ///
    /// The simplest UI wiring is to never call this while recording and take
    /// the whole recording from `stop` (exactly like `useRecorder.ts`, which
    /// only reads `chunksRef` at stop). If you do drain per frame, you own the
    /// drained samples: `stop` returns only the chunks still pending.
    pub fn drain_chunks(&self) -> Vec<Vec<f32>> {
        match self.shared.chunk_rx.lock() {
            Ok(rx) => rx.try_iter().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Last `n` captured samples (fewer until the ring fills), for live level
    /// metering with [`crate::fft`]. Non-destructive.
    pub fn latest_window(&self, n: usize) -> Vec<f32> {
        match self.shared.ring.lock() {
            Ok(ring) => take_latest(&ring, n),
            Err(_) => Vec::new(),
        }
    }

    /// Stops the stream and concatenates every pending chunk into mono
    /// [`crate::audio::PcmAudio`] at the device rate. Errors with
    /// [`RecorderError::Empty`] when nothing was captured.
    pub fn stop(mut self) -> Result<crate::audio::PcmAudio, RecorderError> {
        if let Some(stream) = self.stream.take() {
            // Explicit stop; pause is unsupported on some backends, and
            // dropping the stream releases the device either way.
            let _ = stream.pause();
            drop(stream);
        }

        let mut samples = Vec::new();
        for chunk in self.drain_chunks() {
            samples.extend_from_slice(&chunk);
        }
        if samples.is_empty() {
            return Err(RecorderError::Empty);
        }
        Ok(crate::audio::PcmAudio {
            samples,
            sample_rate: self.sample_rate,
            channels: 1,
        })
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

    let (chunk_tx, chunk_rx) = mpsc::channel();
    let shared = Arc::new(Shared {
        chunk_tx,
        chunk_rx: Mutex::new(chunk_rx),
        ring: Mutex::new(VecDeque::with_capacity(RING_CAPACITY)),
        attenuator: Mutex::new(Attenuator::default()),
        clip: Mutex::new(ClipCounters::default()),
    });

    let stream = open_stream(&device, sample_format, &stream_config, &shared)?;
    stream.play().map_err(|err| {
        RecorderError::Device(format!("Failed to start the microphone stream: {err}"))
    })?;

    Ok(RecorderHandle {
        shared,
        stream: Some(stream),
        sample_rate: stream_config.sample_rate.0,
        started_at: Instant::now(),
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
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let interleaved: Vec<f32> = data.iter().map(|&sample| sample_to_f32(sample)).collect();
            let mut mono = downmix_to_mono(&interleaved, channels);
            // G03: clipping evidence is observed on the raw samples, before
            // the attenuation-only auto gain can hide an already-clipped
            // source below the full-scale threshold.
            if let Ok(mut clip) = shared.clip.lock() {
                clip.observe(&mono);
            }
            if let Ok(mut attenuator) = shared.attenuator.lock() {
                attenuator.process(&mut mono);
            }
            shared.push_chunk(mono);
        },
        |err| eprintln!("starling dictation: microphone stream error: {err}"),
        // No delivery timeout: the callback cadence is the device's business.
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn ring_keeps_order_and_serves_latest_tail() {
        let mut ring = VecDeque::new();
        push_ring(&mut ring, &[1.0, 2.0], RING_CAPACITY);
        push_ring(&mut ring, &[3.0, 4.0], RING_CAPACITY);
        assert_eq!(take_latest(&ring, 10), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(take_latest(&ring, 2), vec![3.0, 4.0]);
        assert_eq!(take_latest(&ring, 0), Vec::<f32>::new());
    }

    #[test]
    fn ring_drops_oldest_when_full() {
        let mut ring = VecDeque::new();
        let first: Vec<f32> = (0..3_000).map(|i| i as f32).collect();
        let second: Vec<f32> = (3_000..6_000).map(|i| i as f32).collect();
        push_ring(&mut ring, &first, RING_CAPACITY);
        push_ring(&mut ring, &second, RING_CAPACITY);

        assert_eq!(ring.len(), RING_CAPACITY);
        assert_eq!(ring.front(), Some(&1_904.0)); // 6000 - 4096 dropped
        assert_eq!(ring.back(), Some(&5_999.0));
        assert_eq!(take_latest(&ring, 3), vec![5_997.0, 5_998.0, 5_999.0]);
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
        let mut counters = ClipCounters::default();
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
        let mut counters = ClipCounters::default();
        assert_eq!(counters.ratio(), 0.0, "empty capture has no clip evidence");
        counters.observe(&[]);
        assert_eq!(counters.ratio(), 0.0, "empty chunks fabricate nothing");

        counters.observe(&[1.0, 1.0, -1.0, 0.1, 0.2, 0.3]);
        counters.observe(&[0.0; 94]);
        assert_eq!(counters.ratio(), 0.03, "3 of 100 across two chunks");
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
        let mut counters = ClipCounters::default();
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
        assert!(!handle.latest_window(RING_CAPACITY).is_empty());

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
