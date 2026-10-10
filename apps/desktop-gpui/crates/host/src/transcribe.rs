//! Transcription in the host (#220): the host transcribes the takes it
//! stores, and any stored take an app asks it to transcribe again.
//!
//! **Takes.** A take binds its engine when it starts (#363): a lease on
//! the ready built-in engine, or the user's server, held until its
//! transcription ends. While it records, its journaled audio streams to
//! that engine's `/stream` ([`crate::live`]) and the previews go to the
//! watching apps as [`Frame::LiveText`]. When it ends, the rest of its
//! audio goes out and the stream is committed; once the take is stored,
//! its job claims it in the store ([`StoreV2::claim_transcription`]),
//! takes the stream's final — or uploads the stored take when the stream
//! could not finish — and settles the attempt. The store held the intent
//! to transcribe since the take's own commit, so a host that dies before
//! the claim, or mid-request, leaves it for the next host, which claims
//! it at startup: transcribed once either way.
//!
//! **Requests.** [`Frame::Transcribe`] transcribes a stored take with the
//! current engine, an installed model (the app switched the engine to it;
//! the job waits for that model to serve) or the user's server: a new
//! attempt beside the earlier ones (#356), never an intent.
//!
//! Every step is published as a [`Frame::Transcription`] to the watching
//! apps; the one the result is for (`yours`) delivers or offers it.
//! Delivery itself stays in the app: it needs the app's window focus.
//!
//! [`Frame::LiveText`]: crate::frame::Frame::LiveText
//! [`Frame::Transcribe`]: crate::frame::Frame::Transcribe
//! [`Frame::Transcription`]: crate::frame::Frame::Transcription

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use starling_dictation::audio::{encode_wav_16k, encode_wav_16k_parts, PcmAudio};
use starling_dictation::client::{CancelToken, ClientError, StarlingClient};
use starling_dictation::storage::TranscriptionResult;
use starling_dictation::store_v2::{
    read_audio_journal, RecognitionOutcome, StoreV2, StoreV2Error, TranscriptionClaim,
};
use starling_runtime::machine::capture::{LiveTakeMonitor, TakeRecord};

use crate::engine::{EngineHost, Target, Want};
use crate::frame::{LivePartial, TranscribeWith, TranscriptionState};
use crate::live::pump::{AudioTap, PumpUpdate, StreamPump};
use crate::live::stream::{LiveStream, StreamOptions};
use crate::live::trace::StreamTrace;
use crate::server::ConnState;
use crate::takes::TakeHub;

/// How many transcriptions run at once (one engine serves them; a second
/// keeps a retry from waiting behind a long take).
const MAX_JOBS: usize = 2;

/// How long a transcription waits for an engine to serve.
pub const DEFAULT_ENGINE_WAIT: Duration = crate::engine::DEFAULT_READY_WAIT;

/// How often stored takes still waiting to be transcribed are looked for
/// again, and how old their intent must be: a take stored a moment ago is
/// about to be transcribed by its own job.
const RESCAN: Duration = Duration::from_secs(60);
const RESCAN_AGED: Duration = Duration::from_secs(120);

/// How long shutdown waits for running transcriptions to give up.
const SHUTDOWN_JOIN: Duration = Duration::from_secs(5);

/// The transcriber's handle: what the take feed and the connections tell
/// it. Cloneable; the work runs on its own thread.
#[derive(Clone)]
pub(crate) struct TranscriberLink {
    tx: Sender<Msg>,
}

impl TranscriberLink {
    fn send(&self, msg: Msg) {
        let _ = self.tx.send(msg);
    }

    pub(crate) fn started(&self, take: &str, monitor: Option<Arc<dyn LiveTakeMonitor>>) {
        self.send(Msg::Started {
            take: take.to_string(),
            monitor,
        });
    }

    pub(crate) fn ended(&self, take: &str, record: Option<Arc<TakeRecord>>) {
        self.send(Msg::Ended {
            take: take.to_string(),
            record,
        });
    }

    /// Take `take` was stored as `stored_id` (`None`: not stored, or not
    /// complete — nothing to transcribe), for `owner`.
    pub(crate) fn persisted(
        &self,
        take: &str,
        stored_id: Option<String>,
        owner: Option<Arc<ConnState>>,
    ) {
        self.send(Msg::Persisted {
            take: take.to_string(),
            stored_id,
            owner,
        });
    }

    /// `conn` asks for the transcription stored take `stored_id` waits
    /// for; its result is that connection's to act on.
    pub(crate) fn due(&self, conn: &Arc<ConnState>, stored_id: String) {
        self.send(Msg::Due {
            conn: Arc::clone(conn),
            stored_id,
        });
    }

    pub(crate) fn request(
        &self,
        conn: &Arc<ConnState>,
        req: String,
        stored_id: String,
        with: TranscribeWith,
    ) {
        self.send(Msg::Request {
            conn: Arc::clone(conn),
            req,
            stored_id,
            with,
        });
    }
}

enum Msg {
    Started {
        take: String,
        monitor: Option<Arc<dyn LiveTakeMonitor>>,
    },
    Ended {
        take: String,
        record: Option<Arc<TakeRecord>>,
    },
    Persisted {
        take: String,
        stored_id: Option<String>,
        owner: Option<Arc<ConnState>>,
    },
    Request {
        conn: Arc<ConnState>,
        req: String,
        stored_id: String,
        with: TranscribeWith,
    },
    Due {
        conn: Arc<ConnState>,
        stored_id: String,
    },
    JobDone {
        stored_id: String,
        last: Option<FinalFrame>,
    },
    Shutdown,
}

/// What the transcriber needs from the host.
pub struct TranscriberConfig {
    pub data_root: PathBuf,
    /// The desktop settings (the live preview cadence); `None`: defaults.
    pub settings_path: Option<PathBuf>,
    /// The engine the host attached; `None`: no engine, every
    /// transcription fails and says so.
    pub engine: Option<Arc<EngineHost>>,
    pub engine_wait: Duration,
}

/// The running transcriber.
pub(crate) struct Transcriber {
    link: TranscriberLink,
    busy: Arc<AtomicUsize>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Transcriber {
    /// Starts the transcriber: it first claims every stored take a
    /// previous host left waiting.
    pub(crate) fn start(
        config: TranscriberConfig,
        hub: Arc<TakeHub>,
    ) -> Result<Transcriber, StoreV2Error> {
        let store = Arc::new(Mutex::new(StoreV2::open(&config.data_root)?));
        let (tx, rx) = std::sync::mpsc::channel();
        let busy = Arc::new(AtomicUsize::new(0));
        let link = TranscriberLink { tx };
        let coordinator = Coordinator {
            inbox: rx,
            link: link.clone(),
            jobs: JobContext {
                store,
                engine: config.engine,
                engine_wait: config.engine_wait,
                hub,
            },
            settings_path: config.settings_path,
            live: HashMap::new(),
            queue: VecDeque::new(),
            running: HashMap::new(),
            busy: Arc::clone(&busy),
        };
        let thread = std::thread::Builder::new()
            .name("starling-host-transcriber".to_string())
            .spawn(move || coordinator.run())
            .map_err(StoreV2Error::Io)?;
        Ok(Transcriber {
            link,
            busy,
            thread: Mutex::new(Some(thread)),
        })
    }

    pub(crate) fn link(&self) -> TranscriberLink {
        self.link.clone()
    }

    /// Whether a transcription is queued or running.
    pub(crate) fn busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst) > 0
    }

    /// Cancels running transcriptions (their takes stay due in the store
    /// for the next host) and joins the transcriber. Idempotent.
    pub(crate) fn shutdown(&self) {
        self.link.send(Msg::Shutdown);
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }
}

/// A take recording or just ended, and its live stream.
struct LiveState {
    /// The engine the take bound at its start.
    target: Option<Target>,
    pump: Option<StreamPump<LiveStream>>,
    /// After the end: the stream with the take's remainder sent and the
    /// final asked for, from the thread that sent it.
    finished: Option<Receiver<Option<LiveStream>>>,
}

/// One transcription to run.
struct Job {
    stored_id: String,
    take: Option<String>,
    req: Option<String>,
    /// The audio hold a request takes when the host accepts it: the take
    /// is kept from every process's upkeep until its attempt starts, even
    /// once the window that asked is gone.
    hold: Option<String>,
    /// Who the result is for: the take's owner or the requester.
    owner: Option<Arc<ConnState>>,
    source: Source,
}

enum Source {
    /// A take this host recorded, with the engine it bound.
    Take {
        target: Option<Target>,
        stream: Option<Receiver<Option<LiveStream>>>,
    },
    /// A stored take waiting to be transcribed (a previous host's).
    Due,
    /// An app's [`crate::frame::Frame::Transcribe`].
    Request(TranscribeWith),
}

struct Coordinator {
    inbox: Receiver<Msg>,
    link: TranscriberLink,
    jobs: JobContext,
    settings_path: Option<PathBuf>,
    live: HashMap<String, LiveState>,
    queue: VecDeque<Job>,
    running: HashMap<String, (JoinHandle<()>, CancelToken)>,
    busy: Arc<AtomicUsize>,
}

impl Coordinator {
    fn run(mut self) {
        self.rescan(Duration::ZERO);
        self.dispatch();
        self.busy
            .store(self.queue.len() + self.running.len(), Ordering::SeqCst);
        let mut next_scan = Instant::now() + RESCAN;
        loop {
            let wait = next_scan.saturating_duration_since(Instant::now());
            match self.inbox.recv_timeout(wait) {
                Ok(Msg::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
                Ok(msg) => self.handle(msg),
                Err(RecvTimeoutError::Timeout) => {
                    self.rescan(RESCAN_AGED);
                    next_scan = Instant::now() + RESCAN;
                }
            }
            self.dispatch();
            self.busy
                .store(self.queue.len() + self.running.len(), Ordering::SeqCst);
        }
        self.queue.clear();
        self.live.clear();
        for (_, cancel) in self.running.values() {
            cancel.cancel();
        }
        let deadline = Instant::now() + SHUTDOWN_JOIN;
        for (stored_id, (thread, _)) in self.running.drain() {
            while !thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if thread.is_finished() {
                let _ = thread.join();
            } else {
                eprintln!(
                    "starling-runtime-host: the transcription of {stored_id} did not stop in \
                     time; left running past shutdown"
                );
            }
        }
        self.busy.store(0, Ordering::SeqCst);
    }

    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Started { take, monitor } => self.take_started(take, monitor),
            Msg::Ended { take, record } => self.take_ended(&take, record),
            Msg::Persisted {
                take,
                stored_id,
                owner,
            } => {
                let live = self.live.remove(&take);
                let Some(stored_id) = stored_id else {
                    return;
                };
                let (target, stream) = match live {
                    Some(live) => (live.target, live.finished),
                    None => (None, None),
                };
                self.enqueue(Job {
                    stored_id,
                    take: Some(take),
                    req: None,
                    hold: None,
                    owner,
                    source: Source::Take { target, stream },
                });
            }
            Msg::Request {
                conn,
                req,
                stored_id,
                with,
            } => {
                if self.known(&stored_id) {
                    self.jobs.publish(
                        &stored_id,
                        None,
                        Some(&req),
                        None,
                        TranscriptionState::Refused {
                            message: "This recording is being transcribed already.".to_string(),
                        },
                        Some(&conn),
                    );
                    return;
                }
                let hold = match self.jobs.store().hold_audio(&stored_id) {
                    Ok(hold) => Some(hold),
                    // A take that is gone is refused when the job runs.
                    Err(StoreV2Error::NotFound(_)) => None,
                    Err(err) => {
                        eprintln!(
                            "starling-runtime-host: could not hold {stored_id}'s audio for its \
                             retry: {err}"
                        );
                        None
                    }
                };
                self.queue.push_back(Job {
                    stored_id,
                    take: None,
                    req: Some(req),
                    hold,
                    owner: Some(conn),
                    source: Source::Request(with),
                });
            }
            // Queued once: one that is queued or running already runs
            // the transcription, and a take with nothing due is left as
            // it is (its claim finds no intent).
            Msg::Due { conn, stored_id } => self.enqueue(Job {
                stored_id,
                take: None,
                req: None,
                hold: None,
                owner: Some(conn),
                source: Source::Due,
            }),
            Msg::JobDone { stored_id, last } => {
                if let Some((thread, _)) = self.running.remove(&stored_id) {
                    let _ = thread.join();
                }
                if let Some(last) = last {
                    self.jobs.publish(
                        &last.stored_id,
                        last.take.as_deref(),
                        last.req.as_deref(),
                        last.attempt.as_deref(),
                        last.state,
                        last.owner.as_ref(),
                    );
                }
            }
            Msg::Shutdown => {}
        }
    }

    /// A take started: bind its engine and open its live stream.
    fn take_started(&mut self, take: String, monitor: Option<Arc<dyn LiveTakeMonitor>>) {
        let target = self.jobs.engine.as_ref().and_then(|engine| engine.bind_now());
        let mut state = LiveState {
            target,
            pump: None,
            finished: None,
        };
        match (state.target.as_ref(), monitor) {
            (Some(target), Some(monitor)) => {
                match self.start_pump(&take, &target.endpoint, monitor) {
                    Ok(pump) => state.pump = Some(pump),
                    Err(reason) => self.jobs.hub.live_text(
                        &take,
                        None,
                        Some(format!(
                            "Live transcription is unavailable ({reason}); the recording will be \
                             transcribed in full after you stop."
                        )),
                    ),
                }
            }
            (None, _) => self.jobs.hub.live_text(
                &take,
                None,
                Some(
                    "The built-in engine is not ready, so this recording shows no live text. It \
                     is saved either way, and transcribed after you stop once the engine is \
                     ready."
                        .to_string(),
                ),
            ),
            (Some(_), None) => {}
        }
        self.live.insert(take, state);
    }

    fn start_pump(
        &self,
        take: &str,
        endpoint: &str,
        monitor: Arc<dyn LiveTakeMonitor>,
    ) -> Result<StreamPump<LiveStream>, String> {
        let trace = StreamTrace::from_env();
        let cadence = match self.settings_path.as_deref() {
            Some(path) => starling_dictation::settings::Settings::load(path)
                .live_preview
                .effective(),
            None => starling_dictation::settings::LivePreviewSettings::default().effective(),
        };
        let options = StreamOptions {
            cadence,
            trace: trace.clone(),
        };
        let rate = monitor.sample_rate();
        let stream = LiveStream::start(endpoint, &options)?;
        let endpoint = endpoint.to_string();
        let hub = Arc::clone(&self.jobs.hub);
        let take_id = take.to_string();
        let (pump, _live) = StreamPump::start(
            Box::new(MonitorTap {
                monitor,
                drained: AtomicUsize::new(0),
            }),
            rate,
            stream,
            Box::new(move || LiveStream::start(&endpoint, &options)),
            trace,
            Some(Arc::new(move |update| match update {
                PumpUpdate::Partial(partial) => hub.live_text(
                    &take_id,
                    Some(LivePartial {
                        text: partial.text,
                        stable_words: partial.stable_words,
                        covered_s: partial.covered_s,
                    }),
                    None,
                ),
                PumpUpdate::Degraded(reason) => hub.live_text(&take_id, None, Some(reason)),
            })),
        )?;
        Ok(pump)
    }

    /// A take ended: its stream stops, and — when it kept audio — the rest
    /// of it goes out and the final is asked for, on a thread of its own.
    fn take_ended(&mut self, take: &str, record: Option<Arc<TakeRecord>>) {
        let Some(state) = self.live.get_mut(take) else {
            return;
        };
        let Some(pump) = state.pump.take() else {
            return;
        };
        let handoff = pump.finish();
        if let Some(reason) = handoff.degradation.clone() {
            self.jobs.hub.live_text(take, None, Some(reason));
        }
        let (Some(record), Some(stream)) = (record, handoff.stream) else {
            return;
        };
        let sent = handoff.sent;
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("starling-stream-finish".to_string())
            .spawn(move || {
                let samples = &record.samples;
                let remainder_sent = sent >= samples.len()
                    || encode_wav_16k_parts(&samples[sent..], record.sample_rate.max(1), 1)
                        .map(|wav| stream.send_audio(wav))
                        .unwrap_or(false);
                let committed = sent <= samples.len()
                    && remainder_sent
                    && !stream.is_closed()
                    && stream.commit();
                let _ = tx.send(committed.then_some(stream));
            });
        if spawned.is_ok() {
            state.finished = Some(rx);
        }
    }

    fn known(&self, stored_id: &str) -> bool {
        self.running.contains_key(stored_id)
            || self.queue.iter().any(|job| job.stored_id == stored_id)
    }

    fn enqueue(&mut self, job: Job) {
        if !self.known(&job.stored_id) {
            self.queue.push_back(job);
        }
    }

    /// Queues every stored take whose intent is at least `aged` old and
    /// that nobody is transcribing.
    fn rescan(&mut self, aged: Duration) {
        let due = match self
            .jobs
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .transcriptions_due(aged)
        {
            Ok(due) => due,
            Err(err) => {
                eprintln!(
                    "starling-runtime-host: could not look for takes waiting to be \
                     transcribed: {err}"
                );
                return;
            }
        };
        for stored_id in due {
            self.enqueue(Job {
                stored_id,
                take: None,
                req: None,
                hold: None,
                owner: None,
                source: Source::Due,
            });
        }
    }

    fn dispatch(&mut self) {
        while self.running.len() < MAX_JOBS {
            let Some(job) = self.queue.pop_front() else {
                return;
            };
            let stored_id = job.stored_id.clone();
            let cancel = CancelToken::new();
            let context = self.jobs.clone();
            let link = self.link.clone();
            let token = cancel.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("starling-transcribe-{stored_id}"))
                .spawn(move || {
                    let stored_id = job.stored_id.clone();
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        context.run(job, &token)
                    }));
                    if outcome.is_err() {
                        eprintln!(
                            "starling-runtime-host: the transcription of {stored_id} panicked; \
                             it is retried when the host next starts"
                        );
                    }
                    link.send(Msg::JobDone {
                        stored_id,
                        last: outcome.ok().flatten(),
                    });
                });
            match spawned {
                Ok(thread) => {
                    self.running.insert(stored_id, (thread, cancel));
                }
                Err(err) => {
                    eprintln!(
                        "starling-runtime-host: no thread for the transcription of {stored_id} \
                         ({err}); it is retried later"
                    );
                    return;
                }
            }
        }
    }
}

/// A recording take's audio as the stream worker drains it: from the
/// recorder, without taking it from the stop.
struct MonitorTap {
    monitor: Arc<dyn LiveTakeMonitor>,
    drained: AtomicUsize,
}

impl AudioTap for MonitorTap {
    fn drain(&self) -> Vec<f32> {
        let from = self.drained.load(Ordering::SeqCst);
        let samples = self.monitor.samples_from(from, usize::MAX);
        self.drained.store(from + samples.len(), Ordering::SeqCst);
        samples
    }

    fn acknowledged(&self) -> u64 {
        self.monitor.acknowledged()
    }
}

/// A job's last frame, published by the coordinator once the job is
/// retired.
struct FinalFrame {
    stored_id: String,
    take: Option<String>,
    req: Option<String>,
    owner: Option<Arc<ConnState>>,
    attempt: Option<String>,
    state: TranscriptionState,
}

/// What a running transcription shares.
#[derive(Clone)]
struct JobContext {
    store: Arc<Mutex<StoreV2>>,
    engine: Option<Arc<EngineHost>>,
    engine_wait: Duration,
    hub: Arc<TakeHub>,
}

/// A request's audio hold, released when its job is done with it.
struct HeldAudio {
    store: Arc<Mutex<StoreV2>>,
    hold: String,
}

impl Drop for HeldAudio {
    fn drop(&mut self) {
        let released = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .release_audio_hold(&self.hold);
        if let Err(err) = released {
            // Void once this host exits (its holder is gone).
            eprintln!("starling-runtime-host: releasing an audio hold failed: {err}");
        }
    }
}

/// How a recognition failed: the sentence history keeps, and whether the
/// server could not be reached.
struct Failure {
    message: String,
    transport: bool,
}

impl Failure {
    fn local(message: impl Into<String>) -> Failure {
        Failure {
            message: message.into(),
            transport: false,
        }
    }
}

impl JobContext {
    fn store(&self) -> std::sync::MutexGuard<'_, StoreV2> {
        self.store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[allow(clippy::too_many_arguments)]
    fn publish(
        &self,
        stored_id: &str,
        take: Option<&str>,
        req: Option<&str>,
        attempt: Option<&str>,
        state: TranscriptionState,
        owner: Option<&Arc<ConnState>>,
    ) {
        self.hub
            .transcription(stored_id, take, req, attempt, state, owner);
    }

    fn wait_target(&self, want: &Want, cancel: &CancelToken) -> Result<Target, String> {
        match &self.engine {
            Some(engine) => engine.wait_target(want, self.engine_wait, cancel, None),
            None => Err(
                "No transcription engine is set up for Starling's recording service; the \
                 recording is saved — retry once one is."
                    .to_string(),
            ),
        }
    }

    /// Runs `job`; its last frame comes back to be published once the job
    /// is retired (a client that hears it may ask again at once).
    fn run(&self, job: Job, cancel: &CancelToken) -> Option<FinalFrame> {
        let (stored_id, take, req, owner) = (
            job.stored_id.clone(),
            job.take.clone(),
            job.req.clone(),
            job.owner.clone(),
        );
        let last = std::cell::RefCell::new(None);
        self.run_job(job, cancel, &|attempt: Option<&str>, state: TranscriptionState| {
            if state.is_final() {
                *last.borrow_mut() = Some((attempt.map(str::to_string), state));
            } else {
                self.publish(
                    &stored_id,
                    take.as_deref(),
                    req.as_deref(),
                    attempt,
                    state,
                    owner.as_ref(),
                );
            }
        });
        last.into_inner().map(|(attempt, state)| FinalFrame {
            stored_id,
            take,
            req,
            owner,
            attempt,
            state,
        })
    }

    fn run_job(
        &self,
        job: Job,
        cancel: &CancelToken,
        publish: &dyn Fn(Option<&str>, TranscriptionState),
    ) {
        let Job {
            stored_id,
            hold,
            source,
            ..
        } = job;
        // Released however the job ends: past this point its attempt (or
        // nothing at all) holds the audio.
        let _hold = hold.map(|hold| HeldAudio {
            store: Arc::clone(&self.store),
            hold,
        });
        let retry = matches!(source, Source::Request(_));
        // What the job runs on, should its engine go away mid-request.
        let want = match &source {
            Source::Request(TranscribeWith::Model { model_id }) => Want::Model(model_id.clone()),
            _ => Want::Current,
        };
        let (target, stream) = match source {
            Source::Take {
                target: Some(target),
                stream,
            } => (Ok(target), stream),
            // Started before an engine served: one fresh look now.
            Source::Take {
                target: None,
                stream,
            } => (self.wait_target(&Want::Current, cancel), stream),
            Source::Due => (self.wait_target(&Want::Current, cancel), None),
            Source::Request(TranscribeWith::Current) => {
                (self.wait_target(&Want::Current, cancel), None)
            }
            Source::Request(TranscribeWith::Model { model_id }) => {
                (self.wait_target(&Want::Model(model_id), cancel), None)
            }
            Source::Request(TranscribeWith::Server { endpoint, model }) => (
                Target::manual(&endpoint, &model)
                    .map_err(|err| format!("Your server's endpoint is not usable: {err}")),
                None,
            ),
        };
        if cancel.is_cancelled() {
            return;
        }
        // A retry that cannot run leaves the take as it is (#356).
        let target = match (target, retry) {
            (Err(reason), true) => {
                publish(
                    None,
                    TranscriptionState::Refused {
                        message: format!("{reason} The recording is unchanged."),
                    },
                );
                return;
            }
            (target, _) => target,
        };
        let backend = match &target {
            Ok(target) => target.backend.clone(),
            Err(_) => starling_dictation::storage::BackendLabel::Engine {
                model_id: "none".to_string(),
            }
            .to_string(),
        };
        // A request for a take still waiting to be transcribed (an import,
        // or one its own job has not reached) is that transcription: it
        // claims the take, so the intent is not run a second time.
        let wanted = retry && self.store().transcription_wanted(&stored_id).unwrap_or(false);
        let begun = if retry && !wanted {
            self.store()
                .begin_recognition(&stored_id, &backend, None)
                .map(Some)
        } else {
            self.store()
                .claim_transcription(&stored_id, &backend, None)
                .map(|claim| match claim {
                    TranscriptionClaim::Claimed { attempt_id } => Some(attempt_id),
                    // Transcribed already, or another claimant has it.
                    TranscriptionClaim::NotWanted | TranscriptionClaim::Held => None,
                })
        };
        let attempt = match begun {
            Ok(Some(attempt)) => attempt,
            Ok(None) if retry => {
                publish(
                    None,
                    TranscriptionState::Refused {
                        message: "This recording is being transcribed already.".to_string(),
                    },
                );
                return;
            }
            Ok(None) => return,
            Err(StoreV2Error::NotFound(_)) => {
                publish(None, TranscriptionState::Gone);
                return;
            }
            Err(err) => {
                // Nothing started: a take's intent stays for the next try.
                publish(
                    None,
                    TranscriptionState::Refused {
                        message: format!("The recording could not be transcribed: {err}"),
                    },
                );
                return;
            }
        };
        publish(
            Some(&attempt),
            TranscriptionState::Started {
                backend: backend.clone(),
            },
        );
        let result = match target {
            Err(reason) => Err(Failure::local(reason)),
            Ok(target) => self.recognize(&stored_id, target, &want, stream, cancel),
        };
        if cancel.is_cancelled() {
            // Shutting down: the attempt is left started with its marker
            // released, so the next host fails it and claims the take again.
            return;
        }
        let blank = result
            .as_ref()
            .is_ok_and(|result| result.text.trim().is_empty());
        let settled = match &result {
            Ok(transcript) => self.store().finish_attempt_transcript(&attempt, transcript),
            Err(failure) => self.store().finish_attempt(
                &attempt,
                RecognitionOutcome::Failed {
                    message: &failure.message,
                },
            ),
        };
        let state = match (settled, result) {
            (Err(StoreV2Error::NotFound(_)), _) => TranscriptionState::Gone,
            (Err(err), Ok(_)) => TranscriptionState::Failed {
                message: format!("The transcript could not be saved to history: {err}"),
                transport: false,
            },
            (Err(err), Err(failure)) => TranscriptionState::Failed {
                message: format!(
                    "{}. Local history update also failed: {err}",
                    failure.message.trim_end_matches('.')
                ),
                transport: failure.transport,
            },
            (Ok(()), Ok(transcript)) => TranscriptionState::Completed {
                // A blank retry is kept, but the take shows the earlier
                // words (#356).
                kept_earlier: retry && blank && self.shows_words(&stored_id),
                text: transcript.text,
            },
            (Ok(()), Err(failure)) => TranscriptionState::Failed {
                message: failure.message,
                transport: failure.transport,
            },
        };
        publish(Some(&attempt), state);
    }

    /// Whether the take's shown transcript has words in it.
    fn shows_words(&self, stored_id: &str) -> bool {
        let attempts = self.store().attempts_for(stored_id).unwrap_or_default();
        attempts
            .iter()
            .rev()
            .filter(|attempt| attempt.is_final_transcript())
            .any(|attempt| !attempt.text.trim().is_empty())
    }

    /// The take's transcript: its live stream's final when the stream
    /// finished, else the stored take uploaded in full.
    fn recognize(
        &self,
        stored_id: &str,
        mut target: Target,
        want: &Want,
        stream: Option<Receiver<Option<LiveStream>>>,
        cancel: &CancelToken,
    ) -> Result<TranscriptionResult, Failure> {
        let stream = stream.and_then(|finished| finished.recv().ok().flatten());
        if let Some(stream) = stream {
            match stream.final_result() {
                Ok(result) => return Ok(result),
                // Kept for the log: a silent fallback hides stream trouble.
                Err(err) => eprintln!(
                    "starling-runtime-host: the live stream of {stored_id} did not finish \
                     ({err}); uploading the recording in full"
                ),
            }
        }
        let wav = match load_wav(&self.store, stored_id) {
            Ok(Some(wav)) => wav,
            Ok(None) => return Err(Failure::local("The recording is no longer in history.")),
            Err(err) => {
                return Err(Failure::local(format!(
                    "The recording could not be read back for transcription ({err})."
                )))
            }
        };
        let mut retried = false;
        loop {
            let outcome = StarlingClient::new(&target.endpoint, &target.model)
                .and_then(|client| client.with_timeout_ms(request_timeout_ms(wav.len())))
                .and_then(|client| client.transcribe_with_cancel(Arc::clone(&wav), stored_id, Some(cancel)));
            match outcome {
                Ok(result) => return Ok(result),
                // The built-in engine went away under the request (a crash,
                // or the app that owned the sidecar exiting): once more on
                // its replacement — the same model, so the attempt's label
                // stays true.
                Err(ClientError::Transport(_)) if target.builtin && !retried && !cancel.is_cancelled() => {
                    retried = true;
                    let failed = target.engine();
                    let replacement = match &self.engine {
                        Some(engine) => engine
                            .wait_target(want, self.engine_wait, cancel, failed.as_ref())
                            .ok()
                            .filter(|replacement| replacement.backend == target.backend),
                        None => None,
                    };
                    match replacement {
                        Some(replacement) => target = replacement,
                        None => {
                            return Err(client_failure(true, &ClientError::Transport(String::new())))
                        }
                    }
                }
                Err(err) => return Err(client_failure(target.builtin, &err)),
            }
        }
    }
}

/// What a failed request says (#363): a transport failure on the
/// built-in engine's own endpoint means the engine went away.
fn client_failure(builtin: bool, err: &ClientError) -> Failure {
    let transport = matches!(err, ClientError::Transport(_) | ClientError::Timeout(_));
    let message = if builtin && transport {
        "The built-in engine stopped while transcribing; the recording is saved — retry when \
         the engine is ready."
            .to_string()
    } else {
        err.to_string()
    };
    Failure { message, transport }
}

/// The stored take as the 16 kHz WAV every transcription receives;
/// `Ok(None)` when it is gone. Read once more when the first read fails:
/// the app's upkeep may have just replaced the journal with its FLAC.
fn load_wav(store: &Mutex<StoreV2>, id: &str) -> Result<Option<Arc<Vec<u8>>>, String> {
    let mut last = String::new();
    for _ in 0..2 {
        let path = match store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .audio_journal_path(id)
        {
            Ok(path) => path,
            Err(StoreV2Error::NotFound(_)) => return Ok(None),
            Err(err) => return Err(err.to_string()),
        };
        match read_audio_journal(&path) {
            Ok(journal) => {
                return encode_wav_16k(&PcmAudio {
                    samples: journal.samples,
                    sample_rate: journal.sample_rate,
                    channels: 1,
                })
                .map(|wav| Some(Arc::new(wav)))
                .map_err(|err| err.to_string())
            }
            Err(err) => last = err.to_string(),
        }
    }
    Err(last)
}

/// The request timeout for one upload (#356): the client's 180 s default,
/// plus the take's own length, capped at the client's 10-minute limit — a
/// long take is not timed out by a budget sized for short ones, and a
/// hung engine still fails in bounded time.
pub fn request_timeout_ms(wav_bytes: usize) -> u64 {
    const BASE_MS: u64 = 180_000;
    const MAX_MS: u64 = 600_000;
    // 16 kHz mono PCM16: 32 000 bytes a second, after the 44-byte header.
    let audio_ms = (wav_bytes.saturating_sub(44) as u64) / 32;
    (BASE_MS + audio_ms).min(MAX_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_takes_get_a_longer_but_bounded_request_timeout() {
        assert_eq!(request_timeout_ms(44), 180_000);
        // A minute of audio adds a minute.
        assert_eq!(request_timeout_ms(44 + 32_000 * 60), 240_000);
        assert_eq!(request_timeout_ms(usize::MAX / 2), 600_000);
    }

    #[test]
    fn a_dead_built_in_engine_is_named_and_a_server_error_is_quoted() {
        let gone = client_failure(true, &ClientError::Transport("refused".into()));
        assert!(gone.transport);
        assert!(gone.message.contains("built-in engine stopped"));
        let refused = client_failure(false, &ClientError::Transport("refused".into()));
        assert!(refused.transport);
        assert_eq!(refused.message, "refused");
        let answered = client_failure(
            true,
            &ClientError::Http {
                status: 500,
                message: "boom".into(),
            },
        );
        assert!(!answered.transport);
        assert_eq!(answered.message, "boom");
    }
}
