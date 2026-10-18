//! Per-model speed check (#475): how many seconds this machine needs to
//! transcribe one minute of one model's audio, measured offline against
//! the bundled engine.
//!
//! `starling-bench` (`cpp/tools/starling_bench.cpp`) already answers this
//! on the command line; the apps did not expose it. This module is the
//! shared core both can use, with the rules the issue pins down:
//!
//! - **No download, no network.** The probe audio is local. The synthetic
//!   probe is a modulated tone generated on every run (a port of
//!   `tests/fidelity_audio.py`, so the smoke probe and the fidelity
//!   fixtures agree on what "a speech stand-in" means). The
//!   representative number comes from a bundled real-speech clip.
//! - **Two numbers, everything else derived.** What a probe measures is
//!   the audio's duration and the wall time the answer took (request in
//!   to answer out, not the engine's compute time alone). RTFx and the
//!   seconds-per-audio-minute figure are pure functions of those two, so
//!   they are testable across the whole supported range.
//! - **Read-only.** A probe never writes to history, never touches
//!   storage v2's retention (#342), and never reaches the insight-event
//!   stream (#308's aggregates) — a user cannot inflate their own
//!   statistics by speed-checking. The transcript is discarded: this is
//!   a throughput probe, not a quality check, and it is labelled as one.
//! - **Never disturbs the live engine.** A probe spawns its *own*
//!   short-lived sidecar for the model under test and shuts it down. The
//!   active engine, its switch and any in-flight take are untouched, so
//!   "Speed check" is safe to press while dictating.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::audio::{decode_pcm16_wav, encode_wav_16k, PcmAudio, STARLING_SAMPLE_RATE};
use crate::client::{CancelToken, ClientError, StarlingClient};
use crate::engine::catalog::CatalogEntry;
use crate::engine::sidecar::{ReadyStage, Sidecar};
use crate::storage::TranscriptionResult;

/// The bundled real-speech probe clip: LibriSpeech 2086-149220-0033
/// (CC BY 4.0), the same fixture `tests/fixtures/` keeps and the Android
/// e2e test bundles. 7.4 s of read speech — long enough for a stable
/// measurement, short enough that the app's binary pays ~230 KB for it.
/// It is already the canonical request shape (mono 16 kHz PCM16), so a
/// probe against it sends exactly the bytes a real take would.
pub const SPEECH_PROBE_WAV: &[u8] = include_bytes!("../assets/librispeech-2086-149220-0033.wav");

/// Which probe audio produced a measurement. The distinction matters
/// for what the UI may claim, so it travels with the number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeKind {
    /// The synthetic modulated tone. A smoke test that the engine runs
    /// at all: silence and short outputs decode fast, so its RTFx is an
    /// **upper bound**, never real-speech speed.
    Smoke,
    /// The bundled real-speech clip. The representative number.
    Speech,
}

impl ProbeKind {
    /// This probe's name. Private because [`SpeedCheck::describe`] is the
    /// one place a probe is named for a user: broadcasting it invited a
    /// second wording to drift from the first.
    fn label(self) -> &'static str {
        match self {
            ProbeKind::Smoke => "synthetic tone",
            ProbeKind::Speech => "speech clip",
        }
    }

    /// Whether this probe's figure is only an upper bound and must say
    /// so (#475: "Label synthetic results as an upper bound").
    pub fn is_upper_bound(self) -> bool {
        matches!(self, ProbeKind::Smoke)
    }
}

/// One completed probe: the two measured numbers and what they derived.
///
/// `transcribe_ms` is **wall time on the caller's side**, from just
/// before the request to just after the answer — request serialization,
/// the loopback transfer and response parsing included. That is what a
/// user waits, so it is the right number to compare models with, but it
/// is not the engine's compute time alone.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeedCheck {
    /// The model that was measured (catalog id).
    pub model_id: String,
    /// The engine backend that ran it (`Vulkan0`, `CPU`, ...), so a
    /// comparison never silently mixes backends.
    pub backend: String,
    /// Which probe audio this is.
    pub kind: ProbeKind,
    /// Seconds of audio transcribed.
    pub audio_seconds: f64,
    /// Milliseconds of wall time the answer took, request in to answer
    /// out (see the type's doc).
    pub transcribe_ms: f64,
    /// Realtime factor: audio seconds per wall second (higher is
    /// faster). `None` when it cannot be stated (see [`rtf`]).
    pub rtf: Option<f64>,
    /// Seconds needed per one minute of audio — `60 / rtf`. `None` when
    /// `rtf` is.
    pub seconds_per_audio_minute: Option<f64>,
}

impl SpeedCheck {
    /// Builds a record from the two measured numbers. `None` figures are
    /// never zero: an unmeasurable probe reports unknown, and the caller
    /// says so.
    pub fn new(
        model_id: impl Into<String>,
        backend: impl Into<String>,
        kind: ProbeKind,
        audio_seconds: f64,
        transcribe_ms: f64,
    ) -> Self {
        let rtf = rtf(audio_seconds, transcribe_ms);
        Self {
            model_id: model_id.into(),
            backend: backend.into(),
            kind,
            audio_seconds,
            transcribe_ms,
            rtf,
            seconds_per_audio_minute: rtf.map(seconds_per_audio_minute),
        }
    }

    /// The one line a probe renders as, everywhere it is shown.
    ///
    /// This is the *only* formatter for a result — the host carries this
    /// type over the wire and the settings row maps over it. Two
    /// formatters (or a client re-deriving the probe's name) drift, and
    /// the first thing to drift is the sentence that keeps a synthetic
    /// probe from being read as real-speech speed.
    pub fn describe(&self) -> String {
        let Some(rtf) = self.rtf else {
            return format!(
                "{} ({}): could not measure a speed — the engine answered too fast or too \
                 slowly to time.",
                self.kind.label(),
                self.backend
            );
        };
        let mut line = format!(
            "{} ({}): {:.3}x realtime — {:.1} s per minute of audio",
            self.kind.label(),
            self.backend,
            rtf,
            self.seconds_per_audio_minute.unwrap_or(f64::NAN),
        );
        if self.kind.is_upper_bound() {
            line.push_str(" (upper bound: a synthetic tone decodes faster than speech)");
        }
        line
    }
}

/// RTFx = audio seconds per wall second. `None` when the measurement
/// cannot support a ratio — a zero or non-finite duration, a zero or
/// non-finite audio length, or a negative elapsed time. Reporting
/// `None` rather than zero or infinity keeps an instrument failure from
/// reading as a result: "unknown speed" is honest, "0x realtime" or
/// "infinitely fast" is not.
pub fn rtf(audio_seconds: f64, transcribe_ms: f64) -> Option<f64> {
    let seconds = transcribe_ms / 1_000.0;
    if !audio_seconds.is_finite() || !seconds.is_finite() || audio_seconds <= 0.0 || seconds <= 0.0
    {
        return None;
    }
    let rtf = audio_seconds / seconds;
    rtf.is_finite().then_some(rtf)
}

/// How many seconds one minute of audio needs at `rtf`. The inverse of
/// the realtime factor, expressed the way a user asks the question.
pub fn seconds_per_audio_minute(rtf: f64) -> f64 {
    if !rtf.is_finite() || rtf <= 0.0 {
        return f64::NAN;
    }
    60.0 / rtf
}

/// The synthetic smoke probe as WAV bytes: a 220 Hz tone under a 10 Hz
/// envelope, in short speech-length bursts with pauses between them.
///
/// Deliberately not speech. It exercises the same encode/decode path
/// with audio loud enough to avoid a fast-exit silence path, and its
/// output being short is exactly why its figure is an upper bound. The
/// parameters mirror `tests/fidelity_audio.py` so "speech stand-in"
/// means one thing across the tree.
pub fn synthetic_probe_wav() -> Vec<u8> {
    /// Port of `fidelity_audio.py`'s constants.
    const SAMPLE_RATE: u32 = STARLING_SAMPLE_RATE;
    const TONE_HZ: f64 = 220.0;
    const ENVELOPE_HZ: f64 = 10.0;
    /// Well below clipping, so the `pcm16` quantizer's asymmetric
    /// negative scale is the one that decides the rounding.
    const AMPLITUDE: f32 = 12_000.0 / 32_768.0;

    let mut samples = Vec::new();
    let total: f64 = SMOKE_SEGMENTS.iter().map(|&(_, seconds)| seconds).sum();
    assert!(
        (total - SMOKE_SECONDS).abs() < 1e-9,
        "the documented smoke duration must match its segments: {total}"
    );
    for &(kind, seconds) in SMOKE_SEGMENTS.iter() {
        let count = (seconds * f64::from(SAMPLE_RATE)).round() as usize;
        for index in 0..count {
            let value = if kind == SmokeSegment::Silence {
                0.0
            } else {
                let time = index as f64 / f64::from(SAMPLE_RATE);
                let envelope = 0.6 + 0.4 * (2.0 * std::f64::consts::PI * ENVELOPE_HZ * time).sin();
                AMPLITUDE
                    * envelope as f32
                    * (2.0 * std::f64::consts::PI * TONE_HZ * time).sin() as f32
            };
            samples.push(value);
        }
    }
    let audio = PcmAudio {
        samples,
        sample_rate: SAMPLE_RATE,
        channels: 1,
    };
    encode_wav_16k(&audio).expect("a synthetic probe encodes")
}

/// What one smoke-probe segment is. Not a `&str`: a typo in the table
/// below ("slience") would compile and quietly turn a pause into a loud
/// tone — exactly the boundary this probe is shaped to test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SmokeSegment {
    Speech,
    Silence,
}

/// The smoke probe's segments: four bursts with a pause between each,
/// 6.0 s in total. Long enough to time, and shaped like dictation so a
/// model that trims silence still has audio to work on. The figure is
/// normalized per second of audio, so it does not have to match the
/// speech clip's length.
const SMOKE_SEGMENTS: &[(SmokeSegment, f64); 7] = &[
    (SmokeSegment::Speech, 1.2),
    (SmokeSegment::Silence, 0.4),
    (SmokeSegment::Speech, 1.2),
    (SmokeSegment::Silence, 0.4),
    (SmokeSegment::Speech, 1.2),
    (SmokeSegment::Silence, 0.4),
    (SmokeSegment::Speech, 1.2),
];

/// The smoke probe's total duration, straight from [`SMOKE_SEGMENTS`].
const SMOKE_SECONDS: f64 = 6.0;

/// The WAV bytes and the audio duration for `kind`, ready to POST.
pub fn probe_audio(kind: ProbeKind) -> Result<(Arc<Vec<u8>>, f64), SpeedCheckError> {
    let bytes = match kind {
        ProbeKind::Smoke => synthetic_probe_wav(),
        ProbeKind::Speech => SPEECH_PROBE_WAV.to_vec(),
    };
    let audio_seconds = wav_duration_seconds(&bytes).ok_or(SpeedCheckError::ProbeAudio {
        kind,
        reason: "the probe audio is not a decodable mono/stereo PCM16 WAV".to_string(),
    })?;
    Ok((Arc::new(bytes), audio_seconds))
}

/// Seconds of audio in a WAV, via the app's own decoder. `None` for
/// anything it refuses, which is the same answer a transcription would
/// give — a probe must never send audio the engine would reject.
pub fn wav_duration_seconds(wav: &[u8]) -> Option<f64> {
    let PcmAudio {
        samples,
        sample_rate,
        channels,
    } = decode_pcm16_wav(wav).ok()?;
    if channels == 0 || sample_rate == 0 {
        return None;
    }
    let frames = samples.len() / channels as usize;
    (frames > 0).then(|| frames as f64 / f64::from(sample_rate))
}

/// Why a speed check did not produce a figure. These are surface
/// strings for the settings UI, so they name the concrete cause and what
/// to do about it.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum SpeedCheckError {
    #[error("the bundled engine could not be started ({reason})")]
    EngineStart { reason: String },
    #[error("the model {model_id} is not installed; download it first")]
    NotInstalled { model_id: String },
    #[error("the engine stopped before it was ready ({reason})")]
    Warmup { reason: String },
    #[error("the probe audio for the {} probe could not be built ({reason})", kind.label())]
    ProbeAudio { kind: ProbeKind, reason: String },
    #[error("the engine did not answer the {} probe ({reason})", kind.label())]
    Transcribe { kind: ProbeKind, reason: String },
    #[error("the speed check was cancelled")]
    Cancelled,
    /// Loading a second resident model would not fit in the memory
    /// currently free. The numbers are the swap policy's (#363):
    /// `needed` is the conservative resident estimate plus its margin,
    /// `available` what the OS reported free.
    #[error(
        "not enough free memory to measure {model_id} alongside the running engine \
         (needs about {}, {} available); the current model keeps serving",
        fmt_bytes(*needed),
        fmt_bytes(*available)
    )]
    InsufficientMemory {
        model_id: String,
        needed: u64,
        available: u64,
    },
}

/// One decimal place, MB below a thousand and GB above — the same
/// spelling the app's switch refusal uses, so a memory sentence reads
/// the same whichever path produced it.
fn fmt_bytes(bytes: u64) -> String {
    const MB: f64 = 1_000_000.;
    let mb = bytes as f64 / MB;
    if mb >= 1000. {
        format!("{:.1} GB", mb / 1000.)
    } else {
        format!("{:.1} MB", mb)
    }
}

/// Whether loading `entry`'s model as a *second* resident copy fits in
/// `available` free memory, under the same policy a model swap obeys
/// (#363): the probe may load alongside the live engine only when the
/// pair fits without freeing anything — it must never drain or OOM the
/// engine serving the user's takes. `available` is a parameter (the
/// caller reads the OS, or its test override) so the decision itself
/// stays pure. No reading (`None`, macOS) admits the run, matching the
/// swap policy's Unknown handling rather than disabling the feature
/// where no honest number exists.
pub fn check_memory_admission(
    available: Option<u64>,
    entry: &CatalogEntry,
) -> Result<(), SpeedCheckError> {
    use crate::engine::memory::{estimate_resident, swap_plan, SwapPlan, SWAP_MARGIN_BYTES};
    let incoming = estimate_resident(entry.size_bytes);
    let needed = incoming.saturating_add(SWAP_MARGIN_BYTES);
    match swap_plan(available, incoming, None, SWAP_MARGIN_BYTES) {
        SwapPlan::Rolling | SwapPlan::Unknown => Ok(()),
        SwapPlan::Refuse { needed, available } => Err(SpeedCheckError::InsufficientMemory {
            model_id: entry.id.clone(),
            needed,
            available,
        }),
        // Unreachable with nothing to drain (NeedsDrain needs an
        // outgoing estimate), but a probe that cannot fit alongside is
        // refused whatever the policy calls the shortfall.
        SwapPlan::NeedsDrain { .. } => Err(SpeedCheckError::InsufficientMemory {
            model_id: entry.id.clone(),
            needed,
            available: available.unwrap_or(0),
        }),
    }
}

/// Everything a probe run needs about one model, without the app's
/// engine lifecycle: the catalog entry, where its verified file lives,
/// which engine binary to run, and where its stderr goes.
#[derive(Clone, Debug)]
pub struct SpeedCheckTarget {
    pub entry: CatalogEntry,
    /// The engine binary selected for this machine (the app's own
    /// selection, passed through: a probe must run the same backend the
    /// user would get, or its number is not theirs).
    pub engine_path: PathBuf,
    /// `<models_dir>/<entry.file_name>`.
    pub gguf_path: PathBuf,
    /// The log this run's sidecar writes — its own, never the live
    /// engine's: `Sidecar::spawn` truncates the file past 5 MB, so
    /// sharing would let a probe wipe the engine's history.
    pub log_path: PathBuf,
    /// The free-memory reading the run's admission is decided against
    /// instead of the OS's, in the engine config's spelling: `None`
    /// reads the OS (production), `Some(reading)` replays a reading a
    /// caller already admitted against (the host passes its
    /// reservation's, so the library guard cannot race a fresh reading
    /// against the reservation), and `Some(None)` forces the
    /// no-reading path. Tests with a stand-in model file use the seam
    /// instead of depending on the test machine's real free RAM for a
    /// model that never becomes resident.
    pub available_memory_override: Option<Option<u64>>,
}

impl SpeedCheckTarget {
    /// The target for `entry` at `paths`: the model's file where this
    /// machine's models dir keeps it, the engine binary this machine
    /// selected, and the probe's own log.
    pub fn at(entry: CatalogEntry, paths: &crate::engine::ProbePaths) -> Self {
        Self {
            gguf_path: paths.models_dir.join(&entry.file_name),
            entry,
            engine_path: paths.engine_path.clone(),
            log_path: paths.log_path.clone(),
            available_memory_override: None,
        }
    }

    /// The `--model` slug a request must carry (the server accepts that
    /// one form).
    pub fn slug(&self) -> &str {
        &self.entry.slug
    }
}

/// Measures one model, offline, against a sidecar of its own.
///
/// `progress` receives the warmup stages so the UI can say "loading"
/// rather than nothing, and `cancel` aborts the whole run (both probes
/// and the sidecar). The sidecar is stopped on every path, including an
/// error: a probe must never leave an engine process behind.
///
/// The sidecar is driven to **warm** before the first probe, so a
/// model's load time is never counted in `transcribe_ms`: the reported
/// figure is per-inference throughput, which is what makes two models
/// on the same machine comparable. That also means a cold model costs
/// wall time the user waits through — with no number to show for it —
/// and it is the reason a second run of the same model is usually faster
/// than the first.
pub fn run_speed_check(
    target: &SpeedCheckTarget,
    kinds: &[ProbeKind],
    progress: &dyn Fn(ReadyStage),
    cancel: Option<&CancelToken>,
) -> Result<Vec<SpeedCheck>, SpeedCheckError> {
    let cancelled = || match cancel {
        Some(token) if token.is_cancelled() => Some(()),
        _ => None,
    };

    if cancelled().is_some() {
        return Err(SpeedCheckError::Cancelled);
    }
    if !target.gguf_path.is_file() {
        return Err(SpeedCheckError::NotInstalled {
            model_id: target.entry.id.clone(),
        });
    }
    // The probe loads a second resident copy next to whatever engine is
    // serving; the swap policy gets its say before anything spawns, so
    // a machine that can run one model but not two refuses here instead
    // of finding out through an allocation failure mid-take. The host
    // admits (and reserves) before it starts the run and replays its
    // reading through the target; for direct callers this check, over
    // a fresh OS reading, is the guard.
    check_memory_admission(
        target
            .available_memory_override
            .unwrap_or_else(crate::engine::memory::available_memory),
        &target.entry,
    )?;

    let sidecar = Sidecar::spawn(
        &target.engine_path,
        target.slug(),
        &target.gguf_path,
        &target.log_path,
    )
    .map_err(|reason| SpeedCheckError::EngineStart { reason })?;

    // The engine's own cancellation signal: one flag the sidecar polls.
    let engine_cancel = Arc::new(AtomicBool::new(false));
    // Only a caller that asked for cancellation gets a watcher: a run
    // nobody can cancel needs no thread, and nothing to report about
    // one. The flag is the only channel the sidecar understands, so the
    // watcher polls rather than awaits it.
    let watcher = match cancel {
        None => None,
        Some(token) => {
            let cancel_flag = Arc::clone(&engine_cancel);
            // And the watcher's own stop: without it, a token that is never
            // cancelled would keep it polling for the process's life, and
            // joining it on the way out would hang the caller.
            let watching = Arc::new(AtomicBool::new(true));
            let watch_flag = Arc::clone(&watching);
            let token = token.clone();
            // A watcher that cannot start fails the run rather than
            // degrading it: without the watcher nothing carries the
            // caller's cancel into `wait_ready`, so a caller that asked
            // for cancellation (the host, shutting down) could be held
            // for the whole loading budget with a cancel that visibly
            // did nothing. Returning here drops the sidecar, which
            // stops it.
            let handle = std::thread::Builder::new()
                .name("starling-speedcheck-cancel".into())
                .spawn(move || {
                    while watch_flag.load(Ordering::Acquire) {
                        if token.is_cancelled() {
                            cancel_flag.store(true, Ordering::Release);
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                })
                .map_err(|error| SpeedCheckError::EngineStart {
                    reason: format!(
                        "the cancel watcher could not be started ({error}); \
                         a run that cannot be cancelled is not started"
                    ),
                })?;
            Some((Some(handle), watching))
        }
    };
    let _watch_guard = JoinOnDrop(watcher);
    // Declared after the watcher's guard on purpose: drop order is reverse
    // of declaration, so on any return path the sidecar is stopped before
    // the join waits on the watcher. A half-alive engine must not outlive
    // the run that measured it, even for a 50 ms poll.
    let stopped = StopOnDrop(&sidecar);

    let ready = sidecar.wait_ready(target.slug(), Some(&engine_cancel), &|stage| {
        progress(stage)
    });
    let backend = match ready {
        Ok(health) => health.backend,
        Err(error) => {
            if engine_cancel.load(Ordering::Acquire) {
                return Err(SpeedCheckError::Cancelled);
            }
            // The install check before this is check-then-act: the file
            // can go missing between it and the load, and saying so beats
            // a generic crash or timeout as the cause.
            if !target.gguf_path.is_file() {
                return Err(SpeedCheckError::NotInstalled {
                    model_id: target.entry.id.clone(),
                });
            }
            return Err(SpeedCheckError::Warmup {
                reason: error.to_string(),
            });
        }
    };
    if cancelled().is_some() {
        return Err(SpeedCheckError::Cancelled);
    }

    let endpoint = sidecar
        .endpoint()
        .ok_or_else(|| SpeedCheckError::EngineStart {
            reason: "the engine never reported its port".to_string(),
        })?;
    let client = StarlingClient::new(&endpoint, target.slug()).map_err(|reason| {
        SpeedCheckError::EngineStart {
            reason: reason.to_string(),
        }
    })?;

    let mut results = Vec::with_capacity(kinds.len());
    for &kind in kinds {
        if cancelled().is_some() {
            return Err(SpeedCheckError::Cancelled);
        }
        let (wav, audio_seconds) = probe_audio(kind)?;
        // The identity is a per-probe marker, not a real request id:
        // nothing downstream records it, and a probe's latency is not a
        // take's.
        let started = Instant::now();
        let answered = client.transcribe_with_cancel(
            wav,
            &format!("speedcheck-{}", uuid::Uuid::new_v4()),
            cancel,
        );
        let transcribe_ms = started.elapsed().as_secs_f64() * 1_000.0;
        match answered {
            Ok(TranscriptionResult { .. }) => {
                // The transcript is dropped on the floor: correctness
                // is not what this probe is for.
                results.push(SpeedCheck::new(
                    &target.entry.id,
                    &backend,
                    kind,
                    audio_seconds,
                    transcribe_ms,
                ));
            }
            Err(error) => {
                if matches!(error, ClientError::Cancelled) {
                    return Err(SpeedCheckError::Cancelled);
                }
                return Err(SpeedCheckError::Transcribe {
                    kind,
                    reason: error.to_string(),
                });
            }
        }
    }
    // Explicit stop before the report, so a UI that shows the result
    // never shares the machine with a half-shut-down engine. The guard
    // below stops it again on the way out: `Sidecar::stop` is idempotent
    // (it takes the child out of the slot under a mutex), so that second
    // call is a no-op rather than a second kill.
    stopped.stop();
    Ok(results)
}

/// Stops the sidecar when the function returns, however it returns.
/// `Sidecar::stop` is idempotent (it takes the child out of the slot
/// under a mutex), so the explicit stop before the report costs nothing.
struct StopOnDrop<'a>(&'a Sidecar);

impl StopOnDrop<'_> {
    /// The engine is stopped now rather than on the way out, so a UI
    /// showing the result never shares the machine with a half-shut-down
    /// sidecar.
    fn stop(&self) {
        self.0.stop();
    }
}

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.stop();
    }
}

/// Stops and joins the cancel watcher when the function returns, however
/// it returns. `None` when no watcher was started (the caller asked for
/// a run nobody can cancel). Clearing the flag first is what keeps the
/// join bounded: a watcher polling a token nobody is going to cancel
/// would otherwise outlive the run that owns it.
struct JoinOnDrop(Option<(Option<std::thread::JoinHandle<()>>, Arc<AtomicBool>)>);

impl Drop for JoinOnDrop {
    fn drop(&mut self) {
        if let Some((handle, watching)) = self.0.take() {
            watching.store(false, Ordering::Release);
            if let Some(handle) = handle {
                let _ = handle.join();
            }
        }
    }
}

/// Both probes a run performs, in the order they are reported.
pub const ALL_PROBES: &[ProbeKind; 2] = &[ProbeKind::Smoke, ProbeKind::Speech];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtf_is_audio_seconds_per_wall_second() {
        // 30 s of audio in 3 s is 10x realtime.
        assert_close(rtf(30.0, 3_000.0).expect("measurable"), 10.0);
        // 7.4 s of speech in 740 ms is 10x realtime, the round-number case.
        assert_close(rtf(7.435, 743.5).expect("measurable"), 10.0);
        // 60 s of audio in 120 s is half realtime — slower than realtime
        // is a legitimate, reportable result.
        assert_close(rtf(60.0, 120_000.0).expect("measurable"), 0.5);
    }

    /// A figure within one part in a million of `expected`. The RTFx
    /// arithmetic over f64 does not always land on the exact decimal, and
    /// what matters is the measurement, not the last bit.
    fn assert_close(got: f64, expected: f64) {
        let tolerance = expected.abs() * 1e-12;
        assert!(
            (got - expected).abs() <= tolerance,
            "{got} vs {expected} (±{tolerance})"
        );
    }

    #[test]
    fn an_unmeasurable_probe_reports_unknown_not_zero_or_infinity() {
        // Zero elapsed time, zero audio, negative elapsed time.
        assert_eq!(rtf(7.4, 0.0), None);
        assert_eq!(rtf(0.0, 100.0), None);
        assert_eq!(rtf(7.4, -1.0), None);
        // Non-finite either side, including the NaN a broken timer can
        // hand over.
        assert_eq!(rtf(f64::NAN, 100.0), None);
        assert_eq!(rtf(f64::INFINITY, 100.0), None);
        assert_eq!(rtf(7.4, f64::NAN), None);
        assert_eq!(rtf(7.4, f64::INFINITY), None);
    }

    #[test]
    fn seconds_per_audio_minute_is_the_users_question_answered() {
        // 10x realtime: 6 s per audio minute, not 0.6.
        assert_close(seconds_per_audio_minute(10.0), 6.0);
        // Half realtime: 120 s per audio minute.
        assert_close(seconds_per_audio_minute(0.5), 120.0);
        // One realtime minute.
        assert_close(seconds_per_audio_minute(1.0), 60.0);
        // The degenerate cases stay NaN rather than 0, which would read
        // as "free".
        assert!(seconds_per_audio_minute(0.0).is_nan());
        assert!(seconds_per_audio_minute(f64::NAN).is_nan());
    }

    #[test]
    fn a_record_derives_both_figures_and_labels_its_probe() {
        let speech = SpeedCheck::new(
            "parakeet-v3-q4km-s16",
            "Vulkan0",
            ProbeKind::Speech,
            7.435,
            743.5,
        );
        assert_close(speech.rtf.expect("measurable"), 10.0);
        assert_close(speech.seconds_per_audio_minute.expect("measurable"), 6.0);
        assert!(!speech.kind.is_upper_bound());
        let line = speech.describe();
        assert!(line.contains("speech clip"), "{line}");
        assert!(line.contains("10.000x realtime"), "{line}");
        assert!(line.contains("6.0 s per minute of audio"), "{line}");
        assert!(line.contains("Vulkan0"), "{line}");
        // A speech clip must not be marked as an upper bound.
        assert!(!line.contains("upper bound"), "{line}");

        let smoke = SpeedCheck::new(
            "parakeet-v3-q4km-s16",
            "Vulkan0",
            ProbeKind::Smoke,
            SMOKE_SECONDS,
            120.0,
        );
        assert_close(smoke.rtf.expect("measurable"), 50.0);
        assert!(
            smoke.describe().contains("upper bound"),
            "{}",
            smoke.describe()
        );
    }

    #[test]
    fn an_unmeasurable_record_says_so_rather_than_showing_a_number() {
        let unknown = SpeedCheck::new("m", "CPU", ProbeKind::Speech, 7.4, 0.0);
        assert_eq!(unknown.rtf, None);
        assert_eq!(unknown.seconds_per_audio_minute, None);
        let line = unknown.describe();
        assert!(line.starts_with("speech clip"), "{line}");
        assert!(line.contains("could not measure"), "{line}");
        // Unknown, not a number: a row must never read "0x realtime".
        assert!(!line.contains("x realtime"), "{line}");
    }

    #[test]
    fn the_synthetic_probe_is_decodable_16k_mono_of_the_expected_length() {
        let wav = synthetic_probe_wav();
        let audio_seconds = wav_duration_seconds(&wav).expect("the probe decodes");
        let expected: f64 = SMOKE_SEGMENTS.iter().map(|&(_, seconds)| seconds).sum();
        assert!(
            (audio_seconds - expected).abs() < 1e-9,
            "{audio_seconds} vs {expected}"
        );
        let PcmAudio {
            sample_rate,
            channels,
            ..
        } = decode_pcm16_wav(&wav).expect("the probe decodes");
        assert_eq!(sample_rate, STARLING_SAMPLE_RATE);
        assert_eq!(channels, 1);
    }

    #[test]
    fn the_synthetic_probe_is_deterministic_and_not_silent() {
        let first = synthetic_probe_wav();
        let second = synthetic_probe_wav();
        assert_eq!(first, second, "the same build must measure the same probe");
        // A tone loud enough that the quantizer's negative branch fires.
        let PcmAudio { samples, .. } = decode_pcm16_wav(&first).expect("the probe decodes");
        assert!(samples.iter().any(|&sample| sample > 0.0));
        assert!(samples.iter().any(|&sample| sample < 0.0));
        assert!(samples.iter().any(|&sample| sample == 0.0));
    }

    #[test]
    fn the_bundled_speech_clip_is_16k_mono_and_reasonable_length() {
        let audio_seconds = wav_duration_seconds(SPEECH_PROBE_WAV).expect("the clip decodes");
        // Long enough to time, short enough to probe quickly.
        assert!(
            audio_seconds > 1.0 && audio_seconds < 60.0,
            "{audio_seconds}"
        );
        let PcmAudio {
            sample_rate,
            channels,
            ..
        } = decode_pcm16_wav(SPEECH_PROBE_WAV).expect("the clip decodes");
        assert_eq!(sample_rate, STARLING_SAMPLE_RATE);
        assert_eq!(channels, 1);
    }

    #[test]
    fn probe_audio_matches_its_source_for_both_kinds() {
        let (smoke, seconds) = probe_audio(ProbeKind::Smoke).expect("smoke probe");
        assert_eq!(&*smoke, &synthetic_probe_wav());
        assert_close(seconds, SMOKE_SECONDS);

        let (speech, seconds) = probe_audio(ProbeKind::Speech).expect("speech probe");
        assert_eq!(&*speech, SPEECH_PROBE_WAV);
        assert_close(
            seconds,
            wav_duration_seconds(SPEECH_PROBE_WAV).expect("the clip decodes"),
        );
    }

    #[test]
    fn a_model_whose_file_is_missing_is_refused_before_any_engine_starts() {
        // A target that names a missing GGUF: the run fails at the
        // install check, so no sidecar is ever spawned.
        let target = SpeedCheckTarget {
            entry: CatalogEntry::new(
                "missing",
                "Missing",
                "parakeet",
                "http://127.0.0.1/missing.gguf",
                1,
                "00",
                false,
                "not installed",
            ),
            engine_path: PathBuf::from("/nonexistent-engine"),
            gguf_path: PathBuf::from("/nonexistent-model.gguf"),
            log_path: PathBuf::from("/dev/null"),
            available_memory_override: None,
        };
        let error = run_speed_check(&target, ALL_PROBES, &|_| {}, None)
            .expect_err("a missing model is refused");
        assert_eq!(
            error,
            SpeedCheckError::NotInstalled {
                model_id: "missing".to_string()
            }
        );
    }

    #[test]
    fn memory_admission_refuses_when_a_second_resident_copy_cannot_fit() {
        let entry = CatalogEntry::new(
            "big",
            "Big",
            "parakeet",
            "http://127.0.0.1/big.gguf",
            1_000_000_000,
            "00",
            false,
            "a one-gigabyte model",
        );
        // One byte free: the probe may not load beside the live engine,
        // and the refusal carries the policy's numbers.
        let refused = check_memory_admission(Some(1), &entry)
            .expect_err("a second resident model cannot fit in one free byte");
        match refused {
            SpeedCheckError::InsufficientMemory {
                model_id,
                needed,
                available,
            } => {
                assert_eq!(model_id, "big");
                assert!(needed > 1_000_000_000, "{needed}");
                assert_eq!(available, 1);
            }
            other => panic!("expected InsufficientMemory, got {other:?}"),
        }
        // No reading admits rather than disabling the check, and ample
        // memory admits by fitting.
        check_memory_admission(None, &entry).expect("no reading admits");
        check_memory_admission(Some(u64::MAX), &entry).expect("ample memory admits");
    }
}
