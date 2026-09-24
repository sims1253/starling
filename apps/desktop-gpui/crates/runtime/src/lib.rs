//! `starling-runtime` — the E17 native runtime, Mode A (increment I3).
//!
//! One in-process library owning the five I0 state machines behind the
//! versioned envelope (E17 §1 Mode A): **capture** (writer task + actor),
//! **jobs** (scheduler + supervised workers), **context/mode** (context
//! service), **documents** (document service) and **delivery** (delivery
//! service). The GPUI app (a later increment) holds a [`RuntimeClient`] —
//! commands in, events + snapshots out, never shared mutable state — and
//! this crate depends on `starling-dictation` for the hardened recorder
//! (`recorder.rs`) and the transcription client (`client.rs`), never the
//! other way around.
//!
//! The frozen I0 contract lives in `packages/contracts/runtime-protocol/`
//! (schemas + fixtures); the executable oracle (`tests/runtime_protocol.py`)
//! is ported here as [`protocol::tables`] (the transition tables as data)
//! + [`protocol::replay`] (trace replay). The conformance suite in
//! `tests/conformance.rs` replays every fixture through this port; the
//! live actors enforce the same tables through [`machine::MachineCore`],
//! so the event stream a running runtime produces is oracle-legal by
//! construction.
//!
//! # Rejection semantics (documented interpretation)
//!
//! v1 defines `runtime.nack{unsupported_version}` (envelope version) and
//! `jobs.rejected{queue_full | resource_limits | duplicate_submission}`
//! (admission) as the only wire-level rejections. Every other refusal —
//! a command illegal in the current machine state, a non-monotonic `seq`,
//! a submit against an unfrozen route or unknown capture — surfaces
//! synchronously through the [`Result`] of [`RuntimeClient::send`] /
//! [`RuntimeClient::send_raw`] instead of being silently absorbed. A
//! command whose machine has an outcome-pending command unresolved is
//! likewise rejected (`pending_unresolved`).
//!
//! # What the stub adapters honestly do
//!
//! - [`machine::context::StubContextProvider`] synthesizes clearly-labeled
//!   target snapshots (`descriptor` prefixed `stub:`) — the real target
//!   adapters are E03's platform work (issue #221: IBus/Fcitx, TSF/UIA,
//!   macOS accessibility), which plugs into the [`RuntimeConfig`]
//!   `context_provider` seam landed with the machine itself (proven
//!   end-to-end over the I4 transport by the host crate's adapter suite).
//! - [`machine::delivery::StubDeliveryAdapter`] prepares and revalidates
//!   (bookkeeping it can do honestly) but **never confirms**: `apply`
//!   fails with `no_delivery_adapter` and suggests the copy fallback, and
//!   every action is recorded in its public log. No fake confirmations.
//!   The real insertion adapters are E03's, on the `delivery_adapter`
//!   seam.
//! - [`machine::docs::MemoryDocumentStore`] keeps documents in memory and
//!   is the default so test construction stays side-effect-free; the I5
//!   wiring (issue #220) adds [`machine::docs::V2DocumentStore`] over
//!   storage v2's `documents`/`revisions` tables — the host's production
//!   config opens it at the data root, and the machine hydrates from it
//!   on first touch per document.

pub mod bus;
pub mod channel;
pub mod machine;
pub mod protocol;
pub mod provider;
#[cfg(any(test, feature = "test-doubles"))]
pub mod testing;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use bus::{EventBus, EventSub};
use machine::capture::{
    CaptureActor, CaptureConfig, CaptureMsg, CaptureSource, CaptureStore, DeviceCaptureSource,
    InMemoryCaptureStore, TakeRegistry,
};
use machine::context::{ContextActor, ContextMsg, ContextProvider, RouteFreezer, StubContextProvider};
use machine::delivery::{DeliveryActor, DeliveryMsg, DeliveryAdapter, StubDeliveryAdapter};
use machine::docs::{DocsActor, DocsMsg, DocumentStore, MemoryDocumentStore, RevisionRegistry};
use machine::jobs::{JobsActor, JobsMsg, JobsSnapshot};
use machine::{MachineView, Receipt, Rejection};
use protocol::{Command, Event, JobLimits};
use provider::{TranscriptionProvider, TransformProcessor};

/// What the router thread accepts from clients.
enum RouterMsg {
    /// A raw wire message (the NACK path: `v != 1` answers
    /// `runtime.nack{unsupported_version}` with `corr` = the rejected id).
    Raw(serde_json::Value, machine::ReceiptTx),
    /// A typed command envelope.
    Typed {
        corr: Option<String>,
        seq: Option<u64>,
        command: Command,
        reply: machine::ReceiptTx,
    },
    Shutdown,
}

/// The runtime's configuration.
pub struct RuntimeConfig {
    /// Bounded capacity of each machine actor's inbox.
    pub command_capacity: usize,
    /// Bounded capacity per event subscriber.
    pub event_capacity: usize,
    /// Capture actor tuning (journal root, progress cadence).
    pub capture: CaptureConfig,
    /// Initial admission limits for the jobs scheduler.
    pub jobs_limits: JobLimits,
    /// The inference provider (production: [`provider::StarlingProvider`];
    /// tests: [`provider::FakeProvider`]).
    pub provider: Arc<dyn TranscriptionProvider>,
    /// The processing seam for `jobs.transform` (#294) (production:
    /// [`provider::PipelineProcessor`] over the configured providers;
    /// tests: [`provider::FakeProcessor`]). The default fails every
    /// transform honestly.
    pub processor: Arc<dyn TransformProcessor>,
    /// The capture device source (production:
    /// [`machine::capture::DeviceCaptureSource`]).
    pub capture_source: Arc<dyn CaptureSource>,
    /// Where finished/salvaged takes are persisted. The default is the
    /// in-memory store so that constructing a config — tests do it freely
    /// — never touches the user's data root; a production embedder
    /// overrides this with storage v2 (the I4 host opens it at its
    /// explicit root; a root-less embedder uses
    /// [`default_capture_store`]).
    pub capture_store: Arc<dyn CaptureStore>,
    /// The documents persistence seam. The default is the in-memory
    /// store (test construction stays side-effect-free, same philosophy
    /// as `capture_store`); the I5 wiring (issue #220) overrides it with
    /// [`machine::docs::V2DocumentStore`] at the data root — the I4
    /// host's production config does exactly that.
    pub document_store: Arc<dyn DocumentStore>,
    /// The external delivery seam (stub in Mode A).
    pub delivery_adapter: Arc<dyn DeliveryAdapter>,
    /// The context snapshot source (stub in Mode A).
    pub context_provider: Arc<dyn ContextProvider>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            command_capacity: 64,
            event_capacity: 1024,
            capture: CaptureConfig::default(),
            jobs_limits: JobLimits {
                max_queued: 8,
                max_concurrent: 2,
                per_route: Vec::new(),
            },
            provider: Arc::new(provider::UnconfiguredProvider),
            processor: Arc::new(provider::UnconfiguredProcessor),
            capture_source: Arc::new(DeviceCaptureSource),
            capture_store: InMemoryCaptureStore::new(),
            document_store: MemoryDocumentStore::new(),
            delivery_adapter: StubDeliveryAdapter::new(),
            context_provider: StubContextProvider::new(),
        }
    }
}

impl RuntimeConfig {
    /// Overrides the inference provider.
    pub fn with_provider(mut self, provider: Arc<dyn TranscriptionProvider>) -> Self {
        self.provider = provider;
        self
    }

    /// Overrides the processing seam for `jobs.transform` (#294).
    pub fn with_processor(mut self, processor: Arc<dyn TransformProcessor>) -> Self {
        self.processor = processor;
        self
    }

    /// Overrides the capture device source.
    pub fn with_capture_source(mut self, source: Arc<dyn CaptureSource>) -> Self {
        self.capture_source = source;
        self
    }

    /// Overrides the capture persistence store.
    pub fn with_capture_store(mut self, store: Arc<dyn CaptureStore>) -> Self {
        self.capture_store = store;
        self
    }

    /// Overrides the documents persistence store.
    pub fn with_document_store(mut self, store: Arc<dyn DocumentStore>) -> Self {
        self.document_store = store;
        self
    }

    /// Overrides the delivery adapter.
    pub fn with_delivery_adapter(mut self, adapter: Arc<dyn DeliveryAdapter>) -> Self {
        self.delivery_adapter = adapter;
        self
    }

    /// Overrides the context provider.
    pub fn with_context_provider(mut self, provider: Arc<dyn ContextProvider>) -> Self {
        self.context_provider = provider;
        self
    }

    /// Overrides the jobs admission limits.
    pub fn with_jobs_limits(mut self, limits: JobLimits) -> Self {
        self.jobs_limits = limits;
        self
    }

    /// Overrides the capture actor tuning.
    pub fn with_capture_config(mut self, config: CaptureConfig) -> Self {
        self.capture = config;
        self
    }
}

/// The production capture persistence (D14: storage v2 is THE store — no
/// opt-in flag, no v1 fallback): [`machine::capture::V2CaptureStore`] at
/// `StoreV2`'s default data root. The in-memory store is returned only
/// when no data root can be opened at all — a degenerate host, not a
/// second backend to switch to.
///
/// Data-visibility note (D14, docs/program/DECISIONS.md): the v1
/// `FileSessionStore`, its reader and the migration flow were removed
/// deliberately, by user directive — the product is pre-release and there
/// is no user data to be compatible with. Takes persisted under the old
/// v1 layout are therefore invisible to a v2 root by design, not by
/// accident; if that ever changes, the directive (not this seam) is what
/// changes. This is not a migration entry point and must not grow one.
///
/// Embedder-facing: nothing inside this workspace calls it in
/// production, and the I4 service host deliberately never will — it owns
/// an explicit data root (its `--root` CLI, its lease) and opens
/// [`machine::capture::V2CaptureStore`] there directly, so a root that
/// will not open is a startup refusal rather than a silent slide onto
/// this function's in-memory fallback. The function remains the recipe
/// for embedders *without* a root-ownership surface: the GPUI in-process
/// switchover passes `.with_capture_store(default_capture_store())` at
/// its entry point. It is pinned, not left to rot, by
/// `starling-runtime-host`'s `tests/default_store.rs`, which constructs
/// it through the public API, persists a take that survives a reopen of
/// the default root, and boots a runtime configured with it (#259).
pub fn default_capture_store() -> Arc<dyn CaptureStore> {
    if let Ok(root) = starling_dictation::store_v2::StoreV2::default_root() {
        if let Ok(store) = machine::capture::V2CaptureStore::open(root) {
            return Arc::new(store);
        }
    }
    InMemoryCaptureStore::new()
}

/// The whole-runtime projection: per-machine states, the frozen-route
/// registry, and the jobs scheduler's internals. Read-only; owned by the
/// machines, copied out on demand.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeSnapshot {
    pub capture: MachineView,
    pub context: MachineView,
    pub docs: MachineView,
    pub delivery: MachineView,
    pub jobs: JobsSnapshot,
    pub frozen_routes: Vec<String>,
}

/// A running runtime. Dropping it shuts the machines down (threads join;
/// in-flight provider workers finish on their own and report into a now
/// closed inbox — surfaced on stderr, since no scheduler remains to
/// receive them).
pub struct Runtime {
    router: channel::Sender<RouterMsg>,
    handles: Vec<JoinHandle<()>>,
    bus: Arc<EventBus>,
    views: SharedViews,
    frozen_routes: machine::context::FrozenRoutes,
}

#[derive(Clone)]
struct SharedViews {
    capture: machine::ViewSlot,
    context: machine::ViewSlot,
    docs: machine::ViewSlot,
    delivery: machine::ViewSlot,
    jobs: Arc<Mutex<JobsSnapshot>>,
}

/// The UI's handle: send commands, receive events + snapshots. Cheap to
/// clone (shares the bounded queues).
#[derive(Clone)]
pub struct RuntimeClient {
    router: channel::Sender<RouterMsg>,
    bus: Arc<EventBus>,
    views: SharedViews,
    frozen_routes: machine::context::FrozenRoutes,
}

impl Runtime {
    /// Boots the five machine actors and returns the runtime with its
    /// client handle.
    pub fn start(config: RuntimeConfig) -> (Runtime, RuntimeClient) {
        let bus = Arc::new(EventBus::new(config.event_capacity));
        let frozen_routes = machine::context::FrozenRoutes::default();
        let registry: TakeRegistry = Arc::default();
        let revisions: RevisionRegistry = Arc::default();

        let (capture_tx, capture_rx) = channel::bounded(config.command_capacity);
        let (context_tx, context_rx) = channel::bounded(config.command_capacity);
        let (docs_tx, docs_rx) = channel::bounded(config.command_capacity);
        let (delivery_tx, delivery_rx) = channel::bounded(config.command_capacity);
        let (jobs_tx, jobs_rx) = channel::bounded(config.command_capacity);

        let views = SharedViews {
            capture: machine::view_slot(&protocol::tables::CAPTURE),
            context: machine::view_slot(&protocol::tables::CONTEXT),
            docs: machine::view_slot(&protocol::tables::DOCS),
            delivery: machine::view_slot(&protocol::tables::DELIVERY),
            jobs: Arc::new(Mutex::new(JobsSnapshot {
                state: "Idle".to_string(),
                limits: config.jobs_limits.clone(),
                waiting: 0,
                active: 0,
                jobs: Vec::new(),
                violations: Vec::new(),
            })),
        };

        let mut handles = Vec::new();

        // Context first: the capture actor holds a freezer onto it.
        let context_actor = ContextActor::new(
            context_rx,
            Arc::clone(&bus),
            Arc::clone(&views.context),
            Arc::clone(&config.context_provider),
            Arc::new(machine::context::DefaultRoutePolicy),
            Arc::clone(&frozen_routes),
        );
        handles.push(spawn("starling-context", move || context_actor.run()));

        let freezer = RouteFreezer::new(context_tx.clone());
        let capture_actor = CaptureActor::new(
            capture_rx,
            // The persist workers' self-addressed report channel (the
            // same inbox, from the sender side — issue #249).
            capture_tx.clone(),
            Arc::clone(&bus),
            Arc::clone(&views.capture),
            Arc::clone(&config.capture_source),
            Arc::clone(&config.capture_store),
            Arc::clone(&registry),
            config.capture.clone(),
            freezer,
        );
        handles.push(spawn("starling-capture", move || capture_actor.run()));

        let jobs_actor = JobsActor::new(
            jobs_rx,
            jobs_tx.clone(),
            Arc::clone(&bus),
            Arc::clone(&views.jobs),
            Arc::clone(&config.provider),
            Arc::clone(&config.processor),
            Arc::clone(&registry),
            Arc::clone(&frozen_routes),
            config.jobs_limits.clone(),
        );
        handles.push(spawn("starling-jobs", move || jobs_actor.run()));

        let docs_actor = DocsActor::new(
            docs_rx,
            Arc::clone(&bus),
            Arc::clone(&views.docs),
            Arc::clone(&revisions),
            Arc::clone(&config.document_store),
        );
        handles.push(spawn("starling-docs", move || docs_actor.run()));

        let delivery_actor = DeliveryActor::new(
            delivery_rx,
            Arc::clone(&bus),
            Arc::clone(&views.delivery),
            Arc::clone(&revisions),
            Arc::clone(&config.delivery_adapter),
        );
        handles.push(spawn("starling-delivery", move || delivery_actor.run()));

        // The router: one bounded entry point that validates envelopes,
        // applies the seq rules, and hands commands to the owning machine.
        let (router_tx, router_rx) = channel::bounded(config.command_capacity);
        let router = Router {
            inbox: router_rx,
            capture: capture_tx,
            context: context_tx,
            docs: docs_tx,
            delivery: delivery_tx,
            jobs: jobs_tx,
            bus: Arc::clone(&bus),
            command_frontier: Mutex::new(HashMap::new()),
        };
        handles.push(spawn("starling-router", move || router.run()));

        let runtime = Runtime {
            router: router_tx,
            handles,
            bus,
            views: views.clone(),
            frozen_routes,
        };
        let client = RuntimeClient {
            router: runtime.router.clone(),
            bus: Arc::clone(&runtime.bus),
            views,
            frozen_routes: Arc::clone(&runtime.frozen_routes),
        };
        (runtime, client)
    }

    /// A fresh event subscription (bounded; backpressure applies).
    pub fn subscribe(&self) -> EventSub {
        self.bus.subscribe()
    }

    /// The runtime projection snapshot.
    pub fn snapshot(&self) -> RuntimeSnapshot {
        snapshot_of(&self.views, &self.frozen_routes)
    }

    /// Shuts the runtime down and joins the machine threads.
    pub fn shutdown(self) {
        let _ = self.router.try_send(RouterMsg::Shutdown);
        drop(self.router);
        for handle in self.handles {
            let _ = handle.join();
        }
    }
}

fn spawn<F>(name: &str, run: F) -> JoinHandle<()>
where
    F: FnOnce() + Send + 'static,
{
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(run)
        .expect("runtime thread spawn")
}

fn snapshot_of(views: &SharedViews, frozen: &machine::context::FrozenRoutes) -> RuntimeSnapshot {
    RuntimeSnapshot {
        capture: views.capture.lock().expect("capture view").clone(),
        context: views.context.lock().expect("context view").clone(),
        docs: views.docs.lock().expect("docs view").clone(),
        delivery: views.delivery.lock().expect("delivery view").clone(),
        jobs: views.jobs.lock().expect("jobs view").clone(),
        frozen_routes: {
            let routes = frozen.lock().expect("frozen routes lock");
            let mut names: Vec<String> = routes.keys().cloned().collect();
            names.sort();
            names
        },
    }
}

struct Router {
    inbox: channel::Receiver<RouterMsg>,
    capture: channel::Sender<CaptureMsg>,
    context: channel::Sender<ContextMsg>,
    docs: channel::Sender<DocsMsg>,
    delivery: channel::Sender<DeliveryMsg>,
    jobs: channel::Sender<JobsMsg>,
    bus: Arc<EventBus>,
    /// Per-stream command-side seq frontier (strictly increasing).
    command_frontier: Mutex<HashMap<String, u64>>,
}

impl Router {
    fn run(mut self) {
        loop {
            match self.inbox.recv() {
                Ok(RouterMsg::Raw(value, reply)) => self.handle_raw(value, reply),
                Ok(RouterMsg::Typed {
                    corr,
                    seq,
                    command,
                    reply,
                }) => self.route_typed(corr, seq, command, reply),
                Ok(RouterMsg::Shutdown) | Err(channel::RecvError::Closed) => break,
                Err(channel::RecvError::Timeout) => unreachable!("recv has no timeout"),
            }
        }
        // Propagate shutdown to every machine.
        let _ = self.capture.try_send(CaptureMsg::Shutdown);
        let _ = self.context.try_send(ContextMsg::Shutdown);
        let _ = self.docs.try_send(DocsMsg::Shutdown);
        let _ = self.delivery.try_send(DeliveryMsg::Shutdown);
        let _ = self.jobs.try_send(JobsMsg::Shutdown);
    }

    fn handle_raw(&mut self, value: serde_json::Value, reply: machine::ReceiptTx) {
        // Version first: a receiver that cannot parse `v` must not apply
        // the payload; it answers runtime.nack{unsupported_version}
        // carrying corr = the rejected message's id.
        let version_supported = value
            .get("v")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|v| protocol::SUPPORTED_VERSIONS.contains(&v));
        if !version_supported {
            if let Some(nack) = protocol::nack_for(&value) {
                let corr = nack
                    .get("corr")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                let _ = self.bus.emit(
                    Event::RuntimeNack {
                        reason: protocol::NackReason,
                    },
                    corr.as_deref(),
                );
            }
            let _ = reply.try_send(Err(Rejection::UnsupportedVersion {
                id: value
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            }));
            return;
        }
        let errors = protocol::envelope_errors(&value);
        if !errors.is_empty() {
            let _ = reply.try_send(Err(Rejection::InvalidEnvelope(errors.join("; "))));
            return;
        }
        let type_ = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let payload = value.get("payload").cloned().unwrap_or(serde_json::Value::Null);
        let command = match Command::from_parts(&type_, &payload) {
            Ok(command) => command,
            Err(message) => {
                let _ = reply.try_send(Err(Rejection::InvalidPayload(message)));
                return;
            }
        };
        let corr = value
            .get("corr")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let seq = value.get("seq").and_then(serde_json::Value::as_u64);
        self.route_typed(corr, seq, command, reply);
    }

    fn route_typed(
        &mut self,
        corr: Option<String>,
        seq: Option<u64>,
        command: Command,
        reply: machine::ReceiptTx,
    ) {
        // Per-stream strictly increasing seq on the command side; a
        // violation is rejected, never absorbed (duplicates and reorders
        // must be surfaced).
        if let Some(seq) = seq {
            let stream = corr
                .clone()
                .unwrap_or_else(|| "__commands__".to_string());
            let stale = {
                let frontier = self.command_frontier.lock().expect("frontier lock");
                frontier
                    .get(&stream)
                    .filter(|last| seq <= **last)
                    .copied()
            };
            if let Some(last) = stale {
                let _ = reply.try_send(Err(Rejection::SeqNotMonotonic {
                    detail: format!(
                        "{} seq {seq} on stream {stream:?} follows seq {last}",
                        command.type_name()
                    ),
                }));
                return;
            }
            self.command_frontier
                .lock()
                .expect("frontier lock")
                .insert(stream, seq);
        }
        let inbound = machine::Inbound {
            id: bus::new_id("cmd"),
            ts: bus::now_ts(),
            corr,
            seq,
            command,
            reply,
        };
        macro_rules! forward {
            ($sender:expr, $variant:path) => {
                match $sender.try_send($variant(inbound)) {
                    Ok(()) => {}
                    Err(channel::TrySendError::Full($variant(back))) => {
                        let _ = back.reply.try_send(Err(Rejection::InboxFull));
                    }
                    Err(channel::TrySendError::Closed($variant(back))) => {
                        let _ = back.reply.try_send(Err(Rejection::Closed));
                    }
                    Err(_) => {}
                }
            };
        }
        match inbound.command.machine() {
            "capture" => forward!(self.capture, CaptureMsg::Command),
            "jobs" => forward!(self.jobs, JobsMsg::Command),
            "context" => forward!(self.context, ContextMsg::Command),
            "docs" => forward!(self.docs, DocsMsg::Command),
            "delivery" => forward!(self.delivery, DeliveryMsg::Command),
            other => {
                let _ = inbound.reply.try_send(Err(Rejection::UnknownMessageType(
                    other.to_string(),
                )));
            }
        }
    }
}

impl RuntimeClient {
    /// Sends a typed command. `corr` names the correlation stream (for a
    /// capture take it is the take's id; for a job it is the job's id).
    /// The runtime assigns `id`, `ts` and a strictly increasing `seq`.
    /// `Ok` means the owning machine accepted the command; outcome events
    /// follow on the event stream.
    pub fn send(
        &self,
        corr: Option<&str>,
        command: Command,
    ) -> Result<Receipt, Rejection> {
        let corr = corr.map(str::to_string);
        let seq = self.bus.next_seq(corr.as_deref());
        let (tx, rx) = channel::bounded(1);
        if self
            .router
            .try_send(RouterMsg::Typed {
                corr,
                seq: Some(seq),
                command,
                reply: tx,
            })
            .is_err()
        {
            return Err(Rejection::Closed);
        }
        match rx.recv() {
            Ok(result) => result,
            Err(_) => Err(Rejection::Closed),
        }
    }

    /// Sends a raw wire message — the envelope-level path, including the
    /// NACK contract: `v != 1` answers `runtime.nack{unsupported_version}`
    /// (corr = the rejected id) on the event stream and rejects here.
    pub fn send_raw(&self, value: serde_json::Value) -> Result<Receipt, Rejection> {
        let (tx, rx) = channel::bounded(1);
        if self.router.try_send(RouterMsg::Raw(value, tx)).is_err() {
            return Err(Rejection::Closed);
        }
        match rx.recv() {
            Ok(result) => result,
            Err(_) => Err(Rejection::Closed),
        }
    }

    /// A fresh bounded event subscription.
    pub fn subscribe(&self) -> EventSub {
        self.bus.subscribe()
    }

    /// Allocates the next `seq` on `corr`'s stream from the same frontier
    /// [`Self::send`] and the event side use. The I4 service host calls
    /// this for IPC clients that sent their envelope without a `seq`:
    /// sequence assignment stays at the point that owns the frontier, so a
    /// reconnecting client cannot collide with the stream positions a dead
    /// connection already consumed (host-assigned numbering continues
    /// monotonically across renderer restarts).
    ///
    /// Caller contract: call once per envelope, immediately before
    /// [`Self::send_raw`], and only when the envelope carries no `seq` of
    /// its own — the assignment is spent either way. Do not mix
    /// host-assigned seqs with client-supplied ones on the same
    /// correlation stream: the router's per-stream `command_frontier`
    /// (a separate map from the bus frontier this method draws on) is
    /// what enforces monotonicity, and interleaving the two numberings
    /// can trip [`crate::machine::Rejection::SeqNotMonotonic`].
    ///
    /// The pairing is **not atomic**: between this call and
    /// [`Self::send_raw`], another sender on the same corr stream may
    /// consume the next seq and route it first, so the raced envelope
    /// arrives out of allocation order and is rejected with
    /// `SeqNotMonotonic`. A caller bridging multiple upstream
    /// connections (the I4 host does) must therefore serialize
    /// assign-then-send per corr stream — one in-flight assignment per
    /// stream at a time keeps allocation order and submission order
    /// identical. The rejection itself is the diagnostic for a caller
    /// that skips that discipline or mixes the two numbering forms on
    /// one stream.
    pub fn assign_seq(&self, corr: Option<&str>) -> u64 {
        self.bus.next_seq(corr)
    }

    /// The runtime projection snapshot.
    pub fn snapshot(&self) -> RuntimeSnapshot {
        snapshot_of(&self.views, &self.frozen_routes)
    }
}
