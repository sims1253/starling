//! Test doubles shared by the crate's tests and the conformance /
//! integration suites. Production code never touches this module; it
//! exists so scripts (devices, gaps, faults, quiesce timeouts, provider
//! outcomes) can drive the real actors without hardware.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_dictation::recorder::{CaptureGap, CapturedTake, RecorderError};

use crate::machine::capture::{CaptureSession, CaptureSource};
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
    /// A capture error to surface after the delay; journal-flavored
    /// strings are non-fatal, everything else is a device fault.
    pub error_after: Option<(Duration, String)>,
    /// What `stop()` returns.
    pub stop: FakeStop,
    /// Amplitude of the synthesized audio (level metering input).
    pub amplitude: f32,
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
    fn capture_error(&self) -> Option<String> {
        // The fake keeps the error surfaced once its delay elapses; the
        // recorder's actor-side classification handles first-error-wins.
        match &self.script.error_after {
            Some((delay, message)) if self.started.elapsed() >= *delay => Some(message.clone()),
            _ => None,
        }
    }
    fn latest_window(&self, n: usize) -> Vec<f32> {
        vec![self.script.amplitude; n.min(2048)]
    }
    fn stop(self: Box<Self>) -> Result<CapturedTake, RecorderError> {
        let produced = elapsed_samples(&self.script, self.started);
        match &self.script.stop {
            FakeStop::Clean {
                journal_id,
                ack_fraction,
            } => {
                let count = produced.max(1);
                let samples = fake_samples(count, self.script.amplitude);
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
                    }),
                })
            }
            FakeStop::QuiesceTimeout { journal_id } => {
                let salvaged = produced;
                let samples = fake_samples(salvaged, self.script.amplitude);
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
                    }),
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
    match &script.stop {
        FakeStop::Clean { ack_fraction, .. } => {
            ((elapsed_samples(script, started) as f64) * ack_fraction).floor() as u64
        }
        _ => elapsed_samples(script, started),
    }
}

fn fake_samples(count: u64, amplitude: f32) -> Vec<f32> {
    // A slow sine so the level meter sees motion and the WAV encoder sees
    // variety; bounded so tests stay fast regardless of count.
    let count = count.min(64_000) as usize;
    (0..count)
        .map(|index| {
            amplitude * (index as f32 * 0.05).sin()
        })
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
        Ok(Box::new(FakeSession {
            script,
            started: Instant::now(),
        }))
    }
}

/// Convenience: a provider script of clean completions, one per text.
pub fn clean_provider_script(texts: &[&str]) -> Vec<FakeJob> {
    texts.iter().map(|text| FakeJob::completes_with(text)).collect()
}
