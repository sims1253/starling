//! Test doubles shared by the crate's tests and the conformance /
//! integration suites. Production code never touches this module; it
//! exists so scripts (devices, gaps, faults, quiesce timeouts, provider
//! outcomes) can drive the real actors without hardware.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_dictation::recorder::{CaptureGap, CapturedTake, RecorderError, RecorderFault};

use crate::machine::capture::{CaptureSession, CaptureSource, LiveTakeMonitor, LiveTakeStatus};
use crate::provider::FakeJob;

/// How a scripted fake session behaves.
#[derive(Clone)]
pub struct FakeTakeScript {
    pub sample_rate: u32,
    /// Samples acknowledged per second of wall time (the "device" rate
    /// the fake produces).
    pub samples_per_second: u64,
    /// Gap spans to surface, with the delay before each appears.
    pub gaps: Vec<(Duration, CaptureGap)>,
    /// A capture fault to surface after the delay, typed by origin:
    /// [`RecorderFault::Journal`] is non-fatal (one `capture.error{
    /// journal_fault}` event, capture continues),
    /// [`RecorderFault::Device`] is fatal (the take is interrupted).
    /// Fatality follows the variant, never the message text (issue #216).
    pub error_after: Option<(Duration, RecorderFault)>,
    /// What `stop()` returns.
    pub stop: FakeStop,
    /// Amplitude of the synthesized audio (level metering input).
    pub amplitude: f32,
    /// Cap on how many samples the fake synthesizes for a take's audio:
    /// the default keeps every test fast, while a test that needs a
    /// genuinely long WAV encode (one the scheduler must not stall on,
    /// issue #216) raises it to synthesize a multi-minute take.
    pub sample_cap: u64,
    /// How long opening the device takes (a slow device open).
    pub open_delay: Duration,
}

/// What a scripted session's stop handshake does.
#[derive(Clone)]
pub enum FakeStop {
    /// A clean take: `samples.len()` samples, journal acknowledged at the
    /// fsynced boundary.
    Clean { journal_id: String, ack_fraction: f64 },
    /// The R09 quiesce timeout: salvaged samples ride in the error.
    QuiesceTimeout { journal_id: String },
    /// The device produced nothing.
    Empty,
    /// A device error on stop.
    DeviceError(String),
    /// A stop that hands back the take but reports the device failed
    /// just before it (`CapturedTake::device_fault`).
    FaultedClean { journal_id: String, fault: String },
}

impl Default for FakeTakeScript {
    fn default() -> Self {
        FakeTakeScript {
            sample_rate: 16_000,
            samples_per_second: 16_000,
            gaps: Vec::new(),
            error_after: None,
            stop: FakeStop::Clean {
                journal_id: "j_fake01".to_string(),
                ack_fraction: 1.0,
            },
            amplitude: 0.25,
            sample_cap: 64_000,
            open_delay: Duration::ZERO,
        }
    }
}

impl FakeTakeScript {
    /// A clean, gapless, fully-acknowledged take.
    pub fn clean() -> FakeTakeScript {
        FakeTakeScript::default()
    }
}

struct FakeSession {
    script: FakeTakeScript,
    started: Instant,
    /// The sample count the stop handed back and when, once it ran: the
    /// monitor stops there, like a real recorder whose device closed.
    stopped_at: Arc<std::sync::OnceLock<(u64, Duration)>>,
}

/// The fake take as the host's take feed sees it: the same deterministic
/// samples the stop hands back, by index.
struct FakeMonitor {
    script: FakeTakeScript,
    started: Instant,
    stopped_at: Arc<std::sync::OnceLock<(u64, Duration)>>,
}

impl FakeMonitor {
    fn produced(&self) -> u64 {
        match self.stopped_at.get() {
            Some((count, _)) => *count,
            None => elapsed_samples(&self.script, self.started),
        }
    }

    /// The take's age, frozen at the stop.
    fn elapsed(&self) -> Duration {
        match self.stopped_at.get() {
            Some((_, at)) => *at,
            None => self.started.elapsed(),
        }
    }
}

impl LiveTakeMonitor for FakeMonitor {
    fn sample_rate(&self) -> u32 {
        self.script.sample_rate
    }
    fn samples_from(&self, from: usize, max: usize) -> Vec<f32> {
        let count = self.sample_count();
        let samples = fake_samples(&self.script, count as u64);
        let start = from.min(samples.len());
        let end = start.saturating_add(max).min(samples.len());
        samples[start..end].to_vec()
    }
    fn sample_count(&self) -> usize {
        (self.produced().min(self.script.sample_cap)) as usize
    }
    fn status(&self) -> LiveTakeStatus {
        LiveTakeStatus {
            captured: self.produced(),
            acknowledged: ack_of(&self.script, self.produced()),
            clip_ratio: 0.0,
            stalled_ms: Some(0),
            elapsed_ms: self.elapsed().as_millis() as u64,
            fault: match &self.script.error_after {
                Some((delay, fault)) if self.elapsed() >= *delay => Some(fault.clone()),
                _ => None,
            },
            disk: None,
            disk_probe_failing: false,
            route: None,
        }
    }
}

impl CaptureSession for FakeSession {
    fn sample_rate(&self) -> u32 {
        self.script.sample_rate
    }
    fn captured_sample_count(&self) -> u64 {
        elapsed_samples(&self.script, self.started)
    }
    fn acknowledged_samples(&self) -> u64 {
        captured_ack(&self.script, self.started)
    }
    fn source_clip_ratio(&self) -> f64 {
        0.0
    }
    fn gaps(&self) -> Vec<CaptureGap> {
        self.script
            .gaps
            .iter()
            .filter(|(delay, _)| self.started.elapsed() >= *delay)
            .map(|(_, gap)| *gap)
            .collect()
    }
    fn capture_fault(&self) -> Option<RecorderFault> {
        // The fake keeps the fault surfaced once its delay elapses; the
        // recorder's actor-side classification handles first-error-wins.
        match &self.script.error_after {
            Some((delay, fault)) if self.started.elapsed() >= *delay => Some(fault.clone()),
            _ => None,
        }
    }
    fn latest_window(&self, n: usize) -> Vec<f32> {
        vec![self.script.amplitude; n.min(2048)]
    }
    fn monitor(&self) -> Option<Arc<dyn LiveTakeMonitor>> {
        Some(Arc::new(FakeMonitor {
            script: self.script.clone(),
            started: self.started,
            stopped_at: Arc::clone(&self.stopped_at),
        }))
    }
    fn stop(self: Box<Self>) -> Result<CapturedTake, RecorderError> {
        let produced = elapsed_samples(&self.script, self.started);
        let _ = self
            .stopped_at
            .set((produced.max(1), self.started.elapsed()));
        match &self.script.stop {
            FakeStop::Clean {
                journal_id,
                ack_fraction,
            } => {
                let count = produced.max(1);
                let samples = fake_samples(&self.script, count);
                let ack = ((count as f64) * ack_fraction).floor() as u64;
                Ok(CapturedTake {
                    audio: starling_dictation::audio::PcmAudio {
                        samples,
                        sample_rate: self.script.sample_rate,
                        channels: 1,
                    },
                    journal: Some(starling_dictation::recorder::JournalReport {
                        id: journal_id.clone(),
                        path: std::path::PathBuf::from(format!("/tmp/fake/{journal_id}.sj")),
                        sample_rate: self.script.sample_rate,
                        acknowledged_samples: ack,
                        finalized: true,
                        fault: None,
                        liveness: Default::default(),
                    }),
                    device_fault: None,
                })
            }
            FakeStop::QuiesceTimeout { journal_id } => {
                let salvaged = produced;
                let samples = fake_samples(&self.script, salvaged);
                Err(RecorderError::QuiesceTimeout {
                    acknowledged_samples: salvaged,
                    audio: starling_dictation::audio::PcmAudio {
                        samples,
                        sample_rate: self.script.sample_rate,
                        channels: 1,
                    },
                    journal: Some(starling_dictation::recorder::JournalReport {
                        id: journal_id.clone(),
                        path: std::path::PathBuf::from(format!("/tmp/fake/{journal_id}.sj")),
                        sample_rate: self.script.sample_rate,
                        acknowledged_samples: salvaged,
                        finalized: true,
                        fault: None,
                        liveness: Default::default(),
                    }),
                })
            }
            FakeStop::FaultedClean { journal_id, fault } => {
                let count = produced.max(1);
                Ok(CapturedTake {
                    audio: starling_dictation::audio::PcmAudio {
                        samples: fake_samples(&self.script, count),
                        sample_rate: self.script.sample_rate,
                        channels: 1,
                    },
                    journal: Some(starling_dictation::recorder::JournalReport {
                        id: journal_id.clone(),
                        path: std::path::PathBuf::from(format!("/tmp/fake/{journal_id}.sj")),
                        sample_rate: self.script.sample_rate,
                        acknowledged_samples: count,
                        finalized: true,
                        fault: None,
                        liveness: Default::default(),
                    }),
                    device_fault: Some(fault.clone()),
                })
            }
            FakeStop::Empty => Err(RecorderError::Empty),
            FakeStop::DeviceError(message) => Err(RecorderError::Device(message.clone())),
        }
    }
}

fn elapsed_samples(script: &FakeTakeScript, started: Instant) -> u64 {
    ((started.elapsed().as_secs_f64() * script.samples_per_second as f64) as u64).max(1)
}

fn captured_ack(script: &FakeTakeScript, started: Instant) -> u64 {
    ack_of(script, elapsed_samples(script, started))
}

/// What the journal has acknowledged of `produced` samples.
fn ack_of(script: &FakeTakeScript, produced: u64) -> u64 {
    match &script.stop {
        FakeStop::Clean { ack_fraction, .. } => ((produced as f64) * ack_fraction).floor() as u64,
        _ => produced,
    }
}

fn fake_samples(script: &FakeTakeScript, count: u64) -> Vec<f32> {
    // A slow sine so the level meter sees motion and the WAV encoder sees
    // variety; bounded by the script's cap so tests stay fast regardless
    // of count (a test that needs a multi-minute take opts in).
    let count = count.min(script.sample_cap) as usize;
    (0..count)
        .map(|index| script.amplitude * (index as f32 * 0.05).sin())
        .collect()
}

/// The scripted capture source: each `start` pops the next script (in
/// order); an exhausted script fails the device open (the honest "no
/// device" behavior tests can assert against).
pub struct FakeCaptureSource {
    scripts: Mutex<VecDeque<FakeTakeScript>>,
    pub started_takes: Mutex<Vec<String>>,
}

impl FakeCaptureSource {
    pub fn new(scripts: Vec<FakeTakeScript>) -> Arc<Self> {
        Arc::new(FakeCaptureSource {
            scripts: Mutex::new(scripts.into_iter().collect()),
            started_takes: Mutex::new(Vec::new()),
        })
    }

    /// Queues another scripted take (tests schedule scripts as they go).
    pub fn push(&self, script: FakeTakeScript) {
        self.scripts
            .lock()
            .expect("fake capture scripts lock")
            .push_back(script);
    }
}

impl CaptureSource for FakeCaptureSource {
    fn start(
        &self,
        _journals_dir: &std::path::Path,
        policy: &str,
    ) -> Result<Box<dyn CaptureSession>, String> {
        let script = self
            .scripts
            .lock()
            .expect("fake capture scripts lock")
            .pop_front()
            .ok_or_else(|| "No microphone was found. Connect an input device and try again.".to_string())?;
        self.started_takes
            .lock()
            .expect("fake started takes lock")
            .push(policy.to_string());
        std::thread::sleep(script.open_delay);
        Ok(Box::new(FakeSession {
            script,
            started: Instant::now(),
            stopped_at: Arc::default(),
        }))
    }
}

/// Convenience: a provider script of clean completions, one per text.
pub fn clean_provider_script(texts: &[&str]) -> Vec<FakeJob> {
    texts.iter().map(|text| FakeJob::completes_with(text)).collect()
}
