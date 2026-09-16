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
//! callback. Resampling to 16 kHz happens in [`crate::audio`] at WAV-encode
//! time, driven by the UI layer — not here.

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

#[derive(Debug, thiserror::Error)]
pub enum RecorderError {
    #[error("{0}")]
    Device(String),
    #[error("No microphone audio was captured.")]
    Empty,
}

/// State shared between the cpal audio callback and the [`RecorderHandle`].
struct Shared {
    /// Every mono chunk in capture order. Unbounded, like `chunksRef` in the
    /// TS hook; drained by `drain_chunks`/`stop`.
    chunk_tx: mpsc::Sender<Vec<f32>>,
    chunk_rx: Mutex<mpsc::Receiver<Vec<f32>>>,
    /// Most recent samples, capped at [`RING_CAPACITY`], for level metering.
    ring: Mutex<VecDeque<f32>>,
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
            let mono = downmix_to_mono(&interleaved, channels);
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
