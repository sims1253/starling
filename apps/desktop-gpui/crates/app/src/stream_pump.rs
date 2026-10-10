//! Live stream pumping (#357): a take's journaled audio goes out to
//! `/stream` and its previews come back on a worker thread the app owns,
//! not in render callbacks. Live text keeps moving while the main window
//! is minimized or covered, and when the overlay is the only window on
//! screen; the UI only applies the newest preview when it arrives.
//!
//! The worker follows the durability boundary: it sends only
//! journal-acknowledged audio, in capture order, each sample once per
//! connection. Stop takes everything back ([`StreamPump::finish`]): the
//! samples drained so far, how many of them the connection has, the
//! connection itself and why live text stopped, if it did. The stop path
//! sends the remainder and commits, so the server finalizes only the tail;
//! a stream that cannot finish still falls back to uploading the whole
//! take.
//!
//! A connection that fails mid-take is reopened a bounded number of times
//! and the whole take so far is replayed to it, since a new server session
//! starts empty. Its previews stay hidden until they hold as many words as
//! the last one shown, so the staging draft does not step back while the
//! server catches up. The seam is kept small for #220, which moves this
//! worker into the runtime host.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gpui::Context;
use serde_json::{Value, json};
use starling_dictation::recorder::{CaptureTap, RecorderHandle};
use tokio::sync::watch;

use crate::app::StarlingApp;
use crate::live_stream::{LiveStream, Partial, StreamOptions, exact_input_quantum};

/// How often the worker drains the take, sends acknowledged audio and
/// polls for previews (the journal writer polls at the same rate).
const TICK: Duration = Duration::from_millis(25);
/// Reconnects per take before live text gives up for the take.
const RECONNECTS: u32 = 2;
/// The wait before a reconnect.
const RECONNECT_DELAY: Duration = Duration::from_millis(500);
/// The most audio one message carries, in seconds: a catch-up after a slow
/// journal flush, or a replay after a reconnect, goes out as a run of
/// frames the bounded command channel can push back on.
const MAX_FRAME_SECONDS: usize = 1;

/// The connection the worker sends audio on and polls previews from.
pub(crate) trait StreamClient: Send + 'static {
    /// `false` when the audio was not queued: backpressure while
    /// [`Self::is_closed`] is false, a dead connection otherwise.
    fn send_audio(&self, wav: Vec<u8>) -> bool;
    fn is_closed(&self) -> bool;
    /// The newest preview since the last poll; an error ends the
    /// connection.
    fn poll_partial(&self) -> Result<Option<Partial>, String>;
}

/// The take's audio as the worker sees it.
pub(crate) trait AudioTap: Send + 'static {
    /// Samples captured since the last drain, in order.
    fn drain(&self) -> Vec<f32>;
    /// Samples from the start of the take that the journal made durable.
    fn acknowledged(&self) -> u64;
}

impl AudioTap for CaptureTap {
    fn drain(&self) -> Vec<f32> {
        self.drain_chunks().concat()
    }

    fn acknowledged(&self) -> u64 {
        self.acknowledged_samples()
    }
}

/// Opens a fresh connection for a reconnect.
pub(crate) type Connect<C> = Box<dyn FnMut() -> Result<C, String> + Send>;

/// What the worker hands back at stop.
pub(crate) struct Handoff<C> {
    /// Every sample drained from the take, at the device rate; the
    /// recorder's `stop` returns only what follows them.
    pub samples: Vec<f32>,
    /// How many of `samples` the connection already has.
    pub sent: usize,
    /// `None` when live text stopped (see `degradation`) or never had a
    /// connection.
    pub stream: Option<C>,
    /// Why live text stopped mid-take, for the capture warning.
    pub degradation: Option<String>,
}

impl<C> Default for Handoff<C> {
    fn default() -> Self {
        Handoff {
            samples: Vec::new(),
            sent: 0,
            stream: None,
            degradation: None,
        }
    }
}

fn degradation(reason: &str) -> String {
    format!(
        "Live transcription stopped mid-recording ({reason}). The full recording will \
         still be transcribed after you stop."
    )
}

/// Why a step dropped the connection.
enum Failure {
    /// The audio itself cannot be streamed: no reconnect helps.
    Encode,
    Stream(String),
}

/// The worker's state; one [`PumpCore::step`] per tick.
struct PumpCore<C> {
    tap: Box<dyn AudioTap>,
    rate: u32,
    quantum: usize,
    stream: Option<C>,
    connect: Connect<C>,
    samples: Vec<f32>,
    sent: usize,
    reconnects_left: u32,
    retry_at: Option<Instant>,
    last_failure: Option<String>,
    degradation: Option<String>,
    /// Words in the newest preview handed to the UI.
    shown_words: usize,
    /// A replayed connection's previews stay hidden below this many words.
    replay_floor: usize,
    partials: watch::Sender<Option<Partial>>,
    trace: Option<Arc<StreamTrace>>,
}

impl<C: StreamClient> PumpCore<C> {
    fn step(&mut self, now: Instant) {
        self.samples.extend(self.tap.drain());
        self.reconnect_if_due(now);
        if let Err(failure) = self.send_acknowledged() {
            self.fail(failure, now);
        }
        let polled = match self.stream.as_ref() {
            Some(stream) => stream.poll_partial(),
            None => Ok(None),
        };
        match polled {
            Ok(Some(partial)) => self.show(partial),
            Ok(None) => {}
            Err(reason) => self.fail(Failure::Stream(reason), now),
        }
    }

    /// Sends acknowledged audio past the connection's watermark, up to
    /// [`MAX_FRAME_SECONDS`] per message, until it is all queued or the
    /// connection pushes back.
    fn send_acknowledged(&mut self) -> Result<(), Failure> {
        let Some(stream) = self.stream.as_ref() else {
            return Ok(());
        };
        let quantum = self.quantum;
        // Whole input quanta only, so each frame resamples to whole 16 kHz
        // samples and the frames add up to the take without drift.
        let acknowledged = (self.tap.acknowledged() as usize).min(self.samples.len())
            / quantum
            * quantum;
        let frame = ((self.rate as usize).max(1) * MAX_FRAME_SECONDS / quantum).max(1) * quantum;
        while self.sent < acknowledged {
            let end = acknowledged.min(self.sent + frame);
            let wav = starling_dictation::audio::encode_wav_16k_parts(
                &self.samples[self.sent..end],
                self.rate,
                1,
            )
            .map_err(|_| Failure::Encode)?;
            if stream.send_audio(wav) {
                self.sent = end;
            } else if stream.is_closed() {
                return Err(Failure::Stream("stream closed".into()));
            } else {
                // The bounded channel is full: the span stays unsent and
                // the next tick retries it.
                break;
            }
        }
        Ok(())
    }

    fn show(&mut self, partial: Partial) {
        let words = partial.text.split_ascii_whitespace().count();
        if words < self.replay_floor {
            return;
        }
        self.replay_floor = 0;
        self.shown_words = words;
        self.partials.send_replace(Some(partial));
    }

    fn fail(&mut self, failure: Failure, now: Instant) {
        self.stream = None;
        let (reason, retry) = match failure {
            Failure::Encode => ("audio could not be encoded for streaming".to_string(), false),
            Failure::Stream(reason) => (reason, true),
        };
        if let Some(trace) = self.trace.as_ref() {
            trace.log("stream_failed", json!({ "reason": reason, "sent_s": self.seconds(self.sent) }));
        }
        if retry && self.reconnects_left > 0 {
            self.reconnects_left -= 1;
            self.retry_at = Some(now + RECONNECT_DELAY);
            self.last_failure = Some(reason);
        } else {
            self.retry_at = None;
            self.degradation = Some(degradation(&reason));
        }
    }

    fn reconnect_if_due(&mut self, now: Instant) {
        if self.stream.is_some() || self.retry_at.is_none_or(|at| now < at) {
            return;
        }
        self.retry_at = None;
        match (self.connect)() {
            Ok(stream) => {
                if let Some(trace) = self.trace.as_ref() {
                    trace.log("reconnect", json!({ "replay_s": self.seconds(self.samples.len()) }));
                }
                // A new session starts empty: the whole take goes again.
                self.stream = Some(stream);
                self.sent = 0;
                self.replay_floor = self.shown_words;
            }
            Err(reason) => self.fail(Failure::Stream(reason), now),
        }
    }

    fn seconds(&self, samples: usize) -> f64 {
        samples as f64 / f64::from(self.rate.max(1))
    }

    fn handoff(&mut self) -> Handoff<C> {
        let stream = self.stream.take();
        // A reconnect still pending at stop: say why live text went quiet.
        let degradation = self.degradation.take().or_else(|| {
            stream
                .is_none()
                .then(|| self.last_failure.take().map(|reason| degradation(&reason)))
                .flatten()
        });
        Handoff {
            samples: std::mem::take(&mut self.samples),
            sent: std::mem::take(&mut self.sent),
            stream,
            degradation,
        }
    }
}

/// The take's stream worker. Dropping it without [`Self::finish`] stops
/// the thread and drops what it drained.
pub(crate) struct StreamPump<C> {
    core: Arc<Mutex<PumpCore<C>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    trace: Option<Arc<StreamTrace>>,
}

impl<C: StreamClient> StreamPump<C> {
    /// Starts pumping `tap` (captured at `rate`) into `stream`; `connect`
    /// opens a replacement connection after a failure. The receiver holds
    /// the newest preview to show.
    pub(crate) fn start(
        tap: Box<dyn AudioTap>,
        rate: u32,
        stream: C,
        connect: Connect<C>,
        trace: Option<Arc<StreamTrace>>,
    ) -> (Self, watch::Receiver<Option<Partial>>) {
        let (partials, receiver) = watch::channel(None);
        let core = Arc::new(Mutex::new(PumpCore {
            tap,
            rate,
            quantum: exact_input_quantum(rate),
            stream: Some(stream),
            connect,
            samples: Vec::new(),
            sent: 0,
            reconnects_left: RECONNECTS,
            retry_at: None,
            last_failure: None,
            degradation: None,
            shown_words: 0,
            replay_floor: 0,
            partials,
            trace: trace.clone(),
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let core = Arc::clone(&core);
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("starling-stream-pump".into())
                .spawn(move || {
                    while !stop.load(Ordering::Acquire) {
                        core.lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .step(Instant::now());
                        std::thread::park_timeout(TICK);
                    }
                })
                .ok()
        };
        (
            StreamPump {
                core,
                stop,
                thread,
                trace,
            },
            receiver,
        )
    }

    /// Stops the worker after its current step and hands the take back.
    /// Nothing is drained or sent after this returns, so the recorder's
    /// `stop` returns exactly the samples after [`Handoff::samples`].
    pub(crate) fn finish(mut self) -> Handoff<C> {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            // A panicked step leaves its state behind the lock; the
            // samples it drained are still handed back below.
            let _ = thread.join();
        }
        let (handoff, rate) = {
            let mut core = self
                .core
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (core.handoff(), f64::from(core.rate.max(1)))
        };
        if let Some(trace) = self.trace.as_ref() {
            trace.log(
                "stop",
                json!({
                    "drained_s": handoff.samples.len() as f64 / rate,
                    "sent_s": handoff.sent as f64 / rate,
                    "stream": handoff.stream.is_some(),
                }),
            );
        }
        handoff
    }
}

impl<C> Drop for StreamPump<C> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.as_ref() {
            thread.thread().unpark();
        }
    }
}

impl StarlingApp {
    /// Opens the take's live stream at `endpoint` and starts its worker on
    /// `handle`'s audio, with the cadence from the settings. An error is
    /// the reason no live text will show; the take records regardless.
    pub(crate) fn start_stream_pump(
        &mut self,
        handle: &RecorderHandle,
        endpoint: &str,
        cx: &mut Context<Self>,
    ) -> Result<(), String> {
        let trace = StreamTrace::from_env();
        let options = StreamOptions {
            cadence: self.live_preview.effective(),
            trace: trace.clone(),
        };
        let stream = LiveStream::start(endpoint, &options)?;
        let endpoint = endpoint.to_string();
        let (pump, mut previews) = StreamPump::start(
            Box::new(handle.tap()),
            handle.sample_rate(),
            stream,
            Box::new(move || LiveStream::start(&endpoint, &options)),
            trace.clone(),
        );
        self.stream_generation = self.stream_generation.wrapping_add(1);
        let generation = self.stream_generation;
        self.stream_pump = Some(pump);
        self.stream_trace = trace;
        // A foreground task, not a render callback: it runs while the
        // main window is minimized or covered. It ends when the worker
        // does (the sender drops with it).
        cx.spawn(async move |this, cx| {
            while previews.changed().await.is_ok() {
                let Some(partial) = previews.borrow_and_update().clone() else {
                    continue;
                };
                let shown = this.update(cx, |app, cx| {
                    app.show_stream_partial(generation, partial, cx);
                });
                if shown.is_err() {
                    break;
                }
            }
        })
        .detach();
        Ok(())
    }

    /// The newest preview of the running take: into the staging draft
    /// (which keeps the user's edits, #297), or the direct-mode line.
    fn show_stream_partial(&mut self, generation: u64, partial: Partial, cx: &mut Context<Self>) {
        // A preview that was in flight when its take stopped is never
        // shown: the final replaces the draft's live text.
        if generation != self.stream_generation || self.stream_pump.is_none() {
            return;
        }
        if let Some(trace) = self.stream_trace.as_ref() {
            trace.displayed(&partial);
        }
        if self.staging.is_some() {
            self.staging_partial(partial, cx);
        } else {
            self.live_partial = partial.text;
            cx.notify();
        }
    }

    /// Stops the take's worker and hands its audio and connection to the
    /// stop path; empty when the take had no stream.
    pub(crate) fn finish_stream_pump(&mut self) -> Handoff<LiveStream> {
        self.stream_trace = None;
        self.stream_pump
            .take()
            .map(StreamPump::finish)
            .unwrap_or_default()
    }
}

/// The take's stream timeline (#226), on when `STARLING_STREAM_TRACE` is
/// set: `1` or `stderr` writes to stderr, anything else is a file the
/// lines are appended to. One JSON object per line, `ms` since the take
/// started: `start`, every server frame (`partial` with the server's
/// `covered_s`/`audio_s`, `final` with its stop path), every preview the
/// UI applied (`display`), `stream_failed`, `reconnect` and `stop`. Partial
/// age at display is `display.ms - 1000 * covered_s`, measured from the
/// take's start rather than the microphone's.
pub(crate) struct StreamTrace {
    started: Instant,
    out: Mutex<Box<dyn Write + Send>>,
}

impl StreamTrace {
    pub(crate) fn from_env() -> Option<Arc<StreamTrace>> {
        let target = std::env::var("STARLING_STREAM_TRACE").ok()?;
        let out: Box<dyn Write + Send> = match target.trim() {
            "" | "0" => return None,
            "1" | "stderr" => Box::new(std::io::stderr()),
            path => match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(file) => Box::new(file),
                Err(err) => {
                    eprintln!("STARLING_STREAM_TRACE: cannot open {path}: {err}");
                    return None;
                }
            },
        };
        let trace = StreamTrace {
            started: Instant::now(),
            out: Mutex::new(out),
        };
        trace.log("start", json!({}));
        Some(Arc::new(trace))
    }

    #[cfg(test)]
    pub(crate) fn discard() -> StreamTrace {
        StreamTrace {
            started: Instant::now(),
            out: Mutex::new(Box::new(std::io::sink())),
        }
    }

    pub(crate) fn log(&self, event: &str, mut fields: Value) {
        let ms = (self.started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0;
        if let Value::Object(map) = &mut fields {
            map.insert("ev".into(), event.into());
            map.insert("ms".into(), ms.into());
        }
        if let Ok(mut out) = self.out.lock() {
            let _ = writeln!(out, "{fields}");
        }
    }

    /// A text frame from the server.
    pub(crate) fn received(&self, frame: &str) {
        let Ok(payload) = serde_json::from_str::<Value>(frame) else {
            return;
        };
        let trace = payload.get("trace");
        let field = |key: &str| trace.and_then(|trace| trace.get(key)).cloned();
        let words = payload
            .get("text")
            .and_then(Value::as_str)
            .map(|text| text.split_ascii_whitespace().count());
        match payload.get("type").and_then(Value::as_str) {
            Some("partial") => self.log(
                "partial",
                json!({
                    "words": words,
                    "stable_words": payload.get("stable_words"),
                    "covered_s": field("covered_s"),
                    "audio_s": field("audio_s"),
                }),
            ),
            Some("final") => self.log(
                "final",
                json!({
                    "words": words,
                    "audio_s": field("audio_s"),
                    "stop": field("stop").map(|stop| json!({
                        "path": stop.get("path"),
                        "unfinalized_s": stop.get("unfinalized_s"),
                    })),
                    "totals": field("totals"),
                }),
            ),
            Some("error") => self.log("error", json!({ "message": payload.get("message") })),
            _ => {}
        }
    }

    /// A preview the UI applied.
    pub(crate) fn displayed(&self, partial: &Partial) {
        self.log(
            "display",
            json!({
                "words": partial.text.split_ascii_whitespace().count(),
                "covered_s": partial.covered_s,
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use starling_dictation::audio::{decode_pcm16_wav, encode_wav_16k_parts};

    /// A take as the recorder would hand it out: captured samples, the
    /// drain watermark, and the journal's acknowledged count.
    #[derive(Default)]
    struct TakeState {
        captured: Vec<f32>,
        drained: usize,
        acknowledged: u64,
    }

    #[derive(Clone, Default)]
    struct FakeTap(Arc<Mutex<TakeState>>);

    impl FakeTap {
        fn capture(&self, samples: &[f32]) {
            self.0.lock().unwrap().captured.extend_from_slice(samples);
        }

        fn acknowledge_all(&self) {
            let mut take = self.0.lock().unwrap();
            take.acknowledged = take.captured.len() as u64;
        }

        fn acknowledge(&self, samples: u64) {
            self.0.lock().unwrap().acknowledged = samples;
        }

        /// What the recorder's `stop` would return after the pump's drains.
        fn stop(&self) -> Vec<f32> {
            let take = self.0.lock().unwrap();
            take.captured[take.drained..].to_vec()
        }
    }

    impl AudioTap for FakeTap {
        fn drain(&self) -> Vec<f32> {
            let mut take = self.0.lock().unwrap();
            let from = take.drained;
            take.drained = take.captured.len();
            take.captured[from..].to_vec()
        }

        fn acknowledged(&self) -> u64 {
            self.0.lock().unwrap().acknowledged
        }
    }

    /// One fake server session: the audio frames it accepted, previews to
    /// hand out, and its health.
    #[derive(Default)]
    struct Session {
        frames: Vec<Vec<u8>>,
        previews: VecDeque<Result<Partial, String>>,
        closed: bool,
        /// Frames accepted before the channel reports itself full.
        capacity: Option<usize>,
    }

    #[derive(Clone, Default)]
    struct FakeStream(Arc<Mutex<Session>>);

    impl FakeStream {
        fn preview(&self, text: &str, stable_words: usize) {
            self.0.lock().unwrap().previews.push_back(Ok(Partial {
                text: text.into(),
                stable_words,
                covered_s: None,
            }));
        }

        fn fail(&self, reason: &str) {
            self.0.lock().unwrap().previews.push_back(Err(reason.into()));
        }

        /// The PCM the session received, frames joined in order.
        fn audio(&self) -> Vec<i16> {
            pcm(&self.0.lock().unwrap().frames)
        }
    }

    impl StreamClient for FakeStream {
        fn send_audio(&self, wav: Vec<u8>) -> bool {
            let mut session = self.0.lock().unwrap();
            if session.closed || session.capacity.is_some_and(|cap| session.frames.len() >= cap) {
                return false;
            }
            session.frames.push(wav);
            true
        }

        fn is_closed(&self) -> bool {
            self.0.lock().unwrap().closed
        }

        fn poll_partial(&self) -> Result<Option<Partial>, String> {
            let mut session = self.0.lock().unwrap();
            let mut latest = None;
            while let Some(preview) = session.previews.pop_front() {
                latest = Some(preview?);
            }
            Ok(latest)
        }
    }

    fn pcm(frames: &[Vec<u8>]) -> Vec<i16> {
        frames
            .iter()
            .flat_map(|frame| {
                decode_pcm16_wav(frame)
                    .expect("a WAV frame")
                    .samples
                    .into_iter()
                    .map(|sample| (sample * 32768.0).round() as i16)
            })
            .collect()
    }

    /// The PCM a single upload of `samples` would carry.
    fn expected(samples: &[f32]) -> Vec<i16> {
        pcm(&[encode_wav_16k_parts(samples, 16_000, 1).unwrap()])
    }

    fn speech(from: usize, len: usize) -> Vec<f32> {
        (from..from + len)
            .map(|i| ((i % 2_000) as f32 - 1_000.0) / 4_096.0)
            .collect()
    }

    fn core(
        tap: &FakeTap,
        stream: &FakeStream,
        reconnects: Vec<FakeStream>,
    ) -> (PumpCore<FakeStream>, watch::Receiver<Option<Partial>>) {
        let (partials, receiver) = watch::channel(None);
        let mut reconnects = VecDeque::from(reconnects);
        let core = PumpCore {
            tap: Box::new(tap.clone()),
            rate: 16_000,
            quantum: exact_input_quantum(16_000),
            stream: Some(stream.clone()),
            connect: Box::new(move || reconnects.pop_front().ok_or_else(|| "refused".to_string())),
            samples: Vec::new(),
            sent: 0,
            reconnects_left: RECONNECTS,
            retry_at: None,
            last_failure: None,
            degradation: None,
            shown_words: 0,
            replay_floor: 0,
            partials,
            trace: None,
        };
        (core, receiver)
    }

    fn shown(receiver: &mut watch::Receiver<Option<Partial>>) -> Option<String> {
        receiver
            .has_changed()
            .unwrap_or(false)
            .then(|| receiver.borrow_and_update().clone())
            .flatten()
            .map(|partial| partial.text)
    }

    #[test]
    fn the_worker_streams_and_shows_previews_without_any_render() {
        // The occluded/minimized window: nothing renders, the worker
        // thread alone moves audio out and previews in.
        let tap = FakeTap::default();
        let stream = FakeStream::default();
        let (pump, mut previews) = StreamPump::start(
            Box::new(tap.clone()),
            16_000,
            stream.clone(),
            Box::new(|| Err("unused".into())),
            None,
        );
        let audio = speech(0, 40_000);
        tap.capture(&audio);
        tap.acknowledge_all();
        stream.preview("hello there", 1);
        let deadline = Instant::now() + Duration::from_secs(5);
        while (stream.audio().len() < audio.len() || !previews.has_changed().unwrap())
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(shown(&mut previews).as_deref(), Some("hello there"));
        let handoff = pump.finish();
        assert_eq!(handoff.samples, audio);
        assert_eq!(handoff.sent, audio.len());
        assert!(handoff.stream.is_some() && handoff.degradation.is_none());
        assert_eq!(stream.audio(), expected(&audio));
        assert!(
            stream.0.lock().unwrap().frames.len() >= 3,
            "a large span goes out in frames of at most a second"
        );
    }

    #[test]
    fn only_acknowledged_audio_is_sent_across_a_slow_journal() {
        let tap = FakeTap::default();
        let stream = FakeStream::default();
        let (mut core, _) = core(&tap, &stream, Vec::new());
        let audio = speech(0, 30_000);
        let now = Instant::now();
        tap.capture(&audio[..10_000]);
        core.step(now);
        assert!(stream.audio().is_empty(), "nothing durable yet");
        tap.acknowledge(4_000);
        tap.capture(&audio[10_000..]);
        core.step(now);
        assert_eq!(stream.audio(), expected(&audio[..4_000]));
        // The flush stalls; capture runs on. Nothing past the watermark.
        core.step(now);
        core.step(now);
        assert_eq!(stream.audio().len(), 4_000);
        tap.acknowledge(30_000);
        core.step(now);
        assert_eq!(stream.audio(), expected(&audio), "no gap, repeat or reorder");
    }

    #[test]
    fn backpressure_keeps_the_unsent_span_for_the_next_tick() {
        let tap = FakeTap::default();
        let stream = FakeStream::default();
        stream.0.lock().unwrap().capacity = Some(1);
        let (mut core, _) = core(&tap, &stream, Vec::new());
        let audio = speech(0, 48_000);
        tap.capture(&audio);
        tap.acknowledge_all();
        core.step(Instant::now());
        assert_eq!(core.sent, 16_000, "one frame queued, the rest kept");
        stream.0.lock().unwrap().capacity = None;
        core.step(Instant::now());
        assert_eq!(stream.audio(), expected(&audio));
        assert!(core.degradation.is_none(), "backpressure is not a failure");
    }

    #[test]
    fn stop_during_an_inflight_preview_hands_back_everything_and_shows_nothing_more() {
        let tap = FakeTap::default();
        let stream = FakeStream::default();
        let (mut core, mut previews) = core(&tap, &stream, Vec::new());
        let audio = speech(0, 20_000);
        tap.capture(&audio[..12_000]);
        tap.acknowledge(8_000);
        core.step(Instant::now());
        // A preview the server is still producing when Stop lands.
        tap.capture(&audio[12_000..]);
        let handoff = core.handoff();
        stream.preview("stale words", 0);
        assert_eq!(handoff.samples, audio[..12_000]);
        assert_eq!(handoff.sent, 8_000);
        assert!(handoff.stream.is_some());
        // The recorder's stop returns the rest; the stop path sends the
        // remainder after the handoff, on the same connection, so the
        // server has the whole take once, in order.
        let mut take = handoff.samples;
        take.extend(tap.stop());
        assert_eq!(take, audio);
        let live = handoff.stream.unwrap();
        assert!(live.send_audio(encode_wav_16k_parts(&take[handoff.sent..], 16_000, 1).unwrap()));
        assert_eq!(live.audio(), expected(&audio));
        // The worker no longer runs: the stale preview is never shown, and
        // the final comes from the commit, not from it.
        assert_eq!(shown(&mut previews), None);
    }

    #[test]
    fn a_dropped_connection_is_reopened_and_the_take_replayed() {
        let tap = FakeTap::default();
        let first = FakeStream::default();
        let second = FakeStream::default();
        let (mut core, mut previews) = core(&tap, &first, vec![second.clone()]);
        let audio = speech(0, 40_000);
        let start = Instant::now();
        tap.capture(&audio[..20_000]);
        tap.acknowledge_all();
        first.preview("one two three four", 2);
        core.step(start);
        assert_eq!(shown(&mut previews).as_deref(), Some("one two three four"));
        first.0.lock().unwrap().closed = true;
        tap.capture(&audio[20_000..32_000]);
        tap.acknowledge_all();
        core.step(start);
        assert!(core.stream.is_none(), "waiting to reconnect");
        tap.capture(&audio[32_000..]);
        tap.acknowledge_all();
        core.step(start + RECONNECT_DELAY);
        assert_eq!(second.audio(), expected(&audio), "the whole take, once, in order");
        assert_eq!(first.audio(), expected(&audio[..20_000]));
        // Catching up: a shorter preview would step the draft back.
        second.preview("one two", 0);
        core.step(start + RECONNECT_DELAY);
        assert_eq!(shown(&mut previews), None);
        second.preview("one two three four five", 3);
        core.step(start + RECONNECT_DELAY);
        assert_eq!(shown(&mut previews).as_deref(), Some("one two three four five"));
        let handoff = core.handoff();
        assert_eq!(handoff.sent, audio.len());
        assert!(handoff.stream.is_some() && handoff.degradation.is_none());
    }

    #[test]
    fn reconnects_are_bounded_and_the_reason_reaches_the_stop() {
        let tap = FakeTap::default();
        let stream = FakeStream::default();
        let (mut core, _) = core(&tap, &stream, Vec::new());
        let start = Instant::now();
        tap.capture(&speech(0, 8_000));
        tap.acknowledge_all();
        stream.fail("server busy");
        core.step(start);
        // Every reopen is refused; after the last one live text gives up.
        for attempt in 1..=RECONNECTS {
            core.step(start + RECONNECT_DELAY * attempt);
        }
        assert!(core.retry_at.is_none());
        let handoff = core.handoff();
        assert!(handoff.stream.is_none());
        assert_eq!(handoff.samples.len(), 8_000, "the drained audio stays");
        assert!(handoff.degradation.unwrap().contains("refused"));
    }

    #[test]
    fn a_reconnect_pending_at_stop_still_explains_itself() {
        let tap = FakeTap::default();
        let stream = FakeStream::default();
        let (mut core, _) = core(&tap, &stream, Vec::new());
        stream.fail("connection reset");
        core.step(Instant::now());
        let handoff = core.handoff();
        assert!(handoff.stream.is_none());
        assert!(handoff.degradation.unwrap().contains("connection reset"));
    }

    #[test]
    fn edits_during_partial_revision_survive_worker_previews() {
        // The #297 contract end to end through the worker: previews the
        // worker hands over revise the live tail, never the user's edit.
        use starling_processing::live::LiveSegmenter;
        use starling_processing::staging::Draft;
        let tap = FakeTap::default();
        let stream = FakeStream::default();
        let (mut core, mut previews) = core(&tap, &stream, Vec::new());
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        fn apply(
            previews: &mut watch::Receiver<Option<Partial>>,
            draft: &mut Draft,
            live: &mut LiveSegmenter,
        ) {
            if let Some(partial) = previews.borrow_and_update().clone() {
                live.partial(draft, &partial.text, partial.stable_words);
            }
        }
        stream.preview("we meet at noon", 1);
        core.step(Instant::now());
        apply(&mut previews, &mut draft, &mut live);
        assert_eq!(draft.text(), "we meet at noon");
        // The user adds a word while the tail is still revisable.
        live.cut();
        let at = draft.text().chars().count();
        draft.insert(at, " [sharp]");
        // The server revises a word before the edit and keeps going.
        stream.preview("we meat at noon then", 1);
        core.step(Instant::now());
        apply(&mut previews, &mut draft, &mut live);
        let text = draft.text();
        assert!(text.contains("meat"), "{text}");
        let edit = text.find("[sharp]").expect("the edit survives");
        assert!(text.find("noon").unwrap() < edit, "{text}");
        assert!(text.ends_with("then"), "{text}");
        assert!(live.finish(&mut draft, "we meat at noon then"));
        assert!(draft.text().contains("[sharp]"));
    }
}
