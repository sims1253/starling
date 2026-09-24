//! #220's renderer-kill acceptance: the host tolerates its renderer
//! **process** being killed at any point — during the handshake, with a
//! command in flight, mid-stream holding a live take, and during the
//! host's own shutdown — and every kill costs exactly the projection:
//! the connection slot returns (a crashlooping renderer never exhausts
//! the cap), the endpoint stays unwedged, the lease is never orphaned (a
//! killed **host** with a live renderer breaks no lease the successor
//! cannot take), and acknowledged audio plus durable documents ride in
//! storage v2.
//!
//! The renderer is `renderer-double`, a real child process the suite
//! SIGKILLs at the phase its mode names (see that binary's docs) — not
//! an in-process `drop`, which is the graceful shape the sibling `ipc`
//! suite already covers. The host under test is the real `server::serve`
//! (or, for the host-death case, the real host binary).

#![cfg(unix)] // the kill signals and the fill mode's socket shrink are
              // unix surfaces; the Windows transport runs elsewhere.

use std::io::Read;
use std::path::Path;
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
use starling_runtime::machine::Receipt;
use starling_runtime::protocol::{Command, Revision};
use starling_runtime::provider::FakeProvider;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::HostClient;
use starling_runtime_host::{serve, HostConfig};

/// The renderer double (a bin target of this crate — cargo exports the
/// built path to integration tests).
const DOUBLE: &str = env!("CARGO_BIN_EXE_renderer-double");

/// The real host binary — the host-death test kills this, not an
/// in-process handle, so the lease's flock releases the way it only does
/// at process death.
const HOST_BIN: &str = env!("CARGO_BIN_EXE_starling-runtime-host");

// --------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------- //

/// The config shape the sibling suites use: transport defaults, real v2
/// persistence at the root, scripted capture/provider so a take is
/// drivable without hardware.
fn kill_config(root: &Path, source: Arc<FakeCaptureSource>, provider: Arc<FakeProvider>) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"));
    config.runtime = config
        .runtime
        .with_capture_source(source)
        .with_provider(provider)
        .with_capture_store(Arc::new(
            V2CaptureStore::open(root).expect("v2 store opens"),
        ))
        .with_capture_config(CaptureConfig {
            journals_dir: root.join("journals"),
            poll_interval: Duration::from_millis(10),
            ..CaptureConfig::default()
        });
    config
}

fn connect_with_retry(path: &Path) -> HostClient {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match HostClient::connect(path) {
            Ok(client) => return client,
            Err(err) => {
                if Instant::now() >= deadline {
                    panic!("no host at {} within 5s: {err}", path.display());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

/// Collects events until `predicate` matches (or the deadline panics).
fn until(
    client: &HostClient,
    label: &str,
    predicate: impl Fn(&starling_runtime_host::client::EventWire) -> bool,
    deadline: Duration,
) {
    let start = Instant::now();
    loop {
        match client.recv_event_timeout(Duration::from_millis(20)) {
            Ok(event) => {
                if predicate(&event) {
                    return;
                }
            }
            Err(starling_runtime::channel::RecvError::Timeout) => {
                if start.elapsed() >= deadline {
                    panic!("timed out waiting for {label}");
                }
            }
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

/// Spawns the renderer double in `mode`, stdout+stderr piped.
fn spawn_double(socket: &Path, mode: &str) -> DoubleRenderer {
    let child = ProcessCommand::new(DOUBLE)
        .arg("--socket")
        .arg(socket)
        .arg("--mode")
        .arg(mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("renderer double spawns");
    DoubleRenderer { child }
}

/// Spawns the double with one extra `--take <corr>` argument.
fn spawn_double_take(socket: &Path, mode: &str, take: &str) -> DoubleRenderer {
    let child = ProcessCommand::new(DOUBLE)
        .arg("--socket")
        .arg(socket)
        .arg("--mode")
        .arg(mode)
        .arg("--take")
        .arg(take)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("renderer double spawns");
    DoubleRenderer { child }
}

/// The child process plus the discipline every test owes it: `ready` is
/// awaited bounded, a failing setup surfaces the double's stderr, and
/// Drop kills the process so a failing test never leaks it.
struct DoubleRenderer {
    child: Child,
}

impl DoubleRenderer {
    /// Waits (bounded) for the double's single `ready` line — the phase
    /// signal after which the kill lands exactly where the test means
    /// it. A double that exited early fails with its stderr, not with a
    /// timeout mystery.
    fn wait_ready(&mut self, socket: &Path) {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let mut stdout = self.child.stdout.take().expect("piped stdout");
        std::thread::spawn(move || {
            let mut line = String::new();
            // One line only (the double prints exactly one); EOF and
            // empty reads surface as an empty line.
            let _ = std::io::BufRead::read_line(&mut std::io::BufReader::new(&mut stdout), &mut line);
            let _ = tx.send(line);
        });
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(line) if line.trim() == "ready" => return,
                Ok(other) => {
                    self.die_with_stderr(format!(
                        "renderer double printed {other:?} instead of ready"
                    ));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(status) = self.child.try_wait().expect("try wait the double") {
                        self.die_with_stderr(format!("renderer double exited early: {status}"));
                    }
                    if Instant::now() > deadline {
                        self.die_with_stderr(format!(
                            "renderer double never reached ready at {}",
                            socket.display()
                        ));
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    self.die_with_stderr("renderer double's stdout closed".to_string());
                }
            }
        }
    }

    fn die_with_stderr(&mut self, message: String) -> ! {
        let mut stderr = String::new();
        if let Some(mut pipe) = self.child.stderr.take() {
            let _ = pipe.read_to_string(&mut stderr);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        panic!("{message}; stderr: {stderr}");
    }

    /// The hard kill: SIGKILL — no handler, no cleanup, the way a
    /// renderer actually dies.
    fn kill(&mut self) {
        self.child.kill().expect("SIGKILL the renderer double");
        let _ = self.child.wait();
    }
}

impl Drop for DoubleRenderer {
    fn drop(&mut self) {
        // Best-effort: a passing test already killed the child (wait
        // reaped it); kill on an exited process is a no-op error.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Waits (bounded) for the endpoint socket to disappear after shutdown.
fn assert_endpoint_removed(socket: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while socket.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(!socket.exists(), "the endpoint {} was not removed", socket.display());
}

// --------------------------------------------------------------------- //
// Kill during the handshake
// --------------------------------------------------------------------- //

/// A renderer killed between connect and its first frame — before the
/// hello was even read — must not hold its connection slot past the
/// kernel's EOF. The idle deadline is tightened to 60s here so the only
/// thing that can return the slot is the connection's own death; with
/// the cap at 1, the proof of return is the next client's admission.
#[test]
fn a_renderer_killed_during_the_handshake_returns_its_slot_and_the_host_serves() {
    let root = tempfile::tempdir().unwrap();
    let mut config = kill_config(
        root.path(),
        FakeCaptureSource::new(vec![]),
        FakeProvider::new(vec![]),
    );
    config.max_connections = 1;
    config.first_frame_idle = Duration::from_secs(60);
    let mut host = serve(config).expect("host serves");
    let socket = host.socket_path().to_path_buf();

    let mut renderer = spawn_double(&socket, "raw");
    renderer.wait_ready(&socket);
    renderer.kill();

    // The slot returns via the reader's EOF, well inside the idle bound:
    // a leaked slot would answer too_many_connections forever.
    let successor = connect_with_retry(&socket);
    successor
        .send(
            None,
            Command::JobsSetLimits(starling_runtime::protocol::JobLimits {
                max_queued: 4,
                max_concurrent: 1,
                per_route: vec![],
            }),
        )
        .expect("the host serves after the handshake kill");

    drop(successor);
    host.shutdown();
}

/// The same kill in a crashloop: eight consecutive connect-and-die
/// renderers at the harshest phase, with the cap at 2 — the double holds
/// one slot, the probe the other, and a single leaked slot in any cycle
/// refuses the next cycle's probe. The registry the accept loop sweeps
/// stays bounded with it (finished pairs are reaped on every accept).
#[test]
fn a_crashlooping_renderer_never_leaks_connection_slots() {
    let root = tempfile::tempdir().unwrap();
    let mut config = kill_config(
        root.path(),
        FakeCaptureSource::new(vec![]),
        FakeProvider::new(vec![]),
    );
    config.max_connections = 2;
    let mut host = serve(config).expect("host serves");
    let socket = host.socket_path().to_path_buf();

    for cycle in 0..8 {
        let mut renderer = spawn_double(&socket, "raw");
        renderer.wait_ready(&socket);
        // The probe shares the cap with the live double; the next command
        // proves the pair is not merely connected but served.
        let probe = connect_with_retry(&socket);
        probe
            .send(
                None,
                Command::JobsSetLimits(starling_runtime::protocol::JobLimits {
                    max_queued: 2 + cycle,
                    max_concurrent: 1,
                    per_route: vec![],
                }),
            )
            .unwrap_or_else(|err| panic!("cycle {cycle}: the host stopped serving: {err}"));
        drop(probe);
        renderer.kill();
        // Let the EOF land before the next cycle's admission so the kill,
        // not a scheduling race, is what each cycle exercises.
        std::thread::sleep(Duration::from_millis(120));
    }

    // Still serving, still within the cap, after the whole loop.
    let final_client = connect_with_retry(&socket);
    let snapshot = final_client.snapshot().expect("snapshot after the crashloop");
    assert_eq!(snapshot["jobs"]["limits"]["maxQueued"], 2 + 7);
    drop(final_client);
    host.shutdown();
}

// --------------------------------------------------------------------- //
// Kill with a command in flight
// --------------------------------------------------------------------- //

/// A renderer that dies between sending a command and reading its
/// receipt: the command still executes (its envelope was accepted), the
/// orphaned receipt costs nothing (the writer unblocks on the dead
/// socket and the connection tears down), and the host keeps serving.
/// The distinctive limits are the durable evidence the command ran.
#[test]
fn a_renderer_killed_with_a_command_in_flight_costs_only_the_receipt() {
    let root = tempfile::tempdir().unwrap();
    let config = kill_config(
        root.path(),
        FakeCaptureSource::new(vec![]),
        FakeProvider::new(vec![]),
    );
    let mut host = serve(config).expect("host serves");
    let socket = host.socket_path().to_path_buf();

    let mut renderer = spawn_double(&socket, "command");
    renderer.wait_ready(&socket);
    // Wait until the host has observably routed the command, then kill —
    // no fixed sleep guessing at the reader.
    let probe = connect_with_retry(&socket);
    let deadline = Instant::now() + Duration::from_secs(5);
    while probe.snapshot().expect("probe snapshot")["jobs"]["limits"]["maxQueued"] != 3 {
        assert!(Instant::now() < deadline, "the renderer's command was never routed");
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(probe);
    renderer.kill();

    let successor = connect_with_retry(&socket);
    let snapshot = successor.snapshot().expect("snapshot");
    assert_eq!(
        snapshot["jobs"]["limits"]["maxQueued"], 3,
        "the killed renderer's command executed ({})",
        snapshot["jobs"]["limits"]
    );
    assert_eq!(snapshot["jobs"]["limits"]["maxConcurrent"], 1);

    drop(successor);
    host.shutdown();
}

// --------------------------------------------------------------------- //
// Kill mid-stream (a live take held by the dying process)
// --------------------------------------------------------------------- /

/// The #220 acceptance with a real process kill: the renderer drove a
/// take to acknowledged audio and died mid-stream — the take keeps
/// recording (Mode B: a dead renderer costs nothing durable), a fresh
/// client resynchronizes from the snapshot and stops it, and the storage
/// v2 row is the durable witness. The sibling `ipc` suite proves the
/// same shape with an in-process drop; this is the process-death
/// version.
#[test]
fn a_renderer_process_killed_mid_take_leaves_a_durable_take_and_a_serving_host() {
    let root = tempfile::tempdir().unwrap();
    let config = kill_config(
        root.path(),
        FakeCaptureSource::new(vec![FakeTakeScript::clean()]),
        FakeProvider::new(vec![]),
    );
    let mut host = serve(config).expect("host serves");
    let socket = host.socket_path().to_path_buf();

    let mut renderer = spawn_double_take(&socket, "take", "take_kill");
    // Ready here means: route frozen, take started, audio acknowledged.
    renderer.wait_ready(&socket);
    renderer.kill();

    let successor = connect_with_retry(&socket);
    let snapshot = successor.snapshot().expect("snapshot after the kill");
    assert_eq!(
        snapshot["capture"]["state"], "Recording",
        "the take outlives the renderer process"
    );

    successor
        .send(
            Some("take_kill"),
            Command::CaptureStop { drain: Some(true) },
        )
        .expect("stop accepted over the new connection");
    until(
        &successor,
        "capture.stopped",
        |event| {
            event.type_name() == "capture.stopped" && event.corr() == Some("take_kill")
        },
        Duration::from_secs(10),
    );
    assert_eq!(successor.snapshot().unwrap()["capture"]["state"], "Persisted");

    // The durable witness: the row is in storage v2, not in any process.
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).expect("store opens");
    let page = store.list_records(0, 10).expect("list");
    assert_eq!(page.total, 1, "exactly one take persisted");
    match &page.records[0] {
        starling_dictation::store_v2::ListedCapture::Capture(listing) => {
            assert_eq!(
                listing.record.status,
                starling_dictation::store_v2::CaptureStatus::Complete
            );
        }
        starling_dictation::store_v2::ListedCapture::Damaged(damaged) => {
            panic!("the surviving take must be healthy: {damaged:?}")
        }
    }

    drop(successor);
    host.shutdown();
}

// --------------------------------------------------------------------- //
// Kill during the host's own shutdown
// --------------------------------------------------------------------- //

/// A wedged renderer (the fill shape: snapshot flood, never reads, the
/// host's writer parked mid-reply) is SIGKILLed **inside** the host's
/// shutdown drain window. Shutdown must still complete in bounded time
/// with the full teardown — machines joined, endpoint removed, lease
/// released (a successor owns the root immediately after).
#[test]
fn shutdown_completes_when_the_renderer_is_killed_mid_drain() {
    let root = tempfile::tempdir().unwrap();
    let config = kill_config(
        root.path(),
        FakeCaptureSource::new(vec![]),
        FakeProvider::new(vec![]),
    );
    let mut host = serve(config).expect("host serves");
    let socket = host.socket_path().to_path_buf();

    let mut renderer = spawn_double(&socket, "fill");
    renderer.wait_ready(&socket);
    // Give the host's reader a moment to answer the flood so the writer
    // is the parked side (the drain's whole point).
    std::thread::sleep(Duration::from_millis(500));

    let socket_path = host.socket_path().to_path_buf();
    let started = Instant::now();
    let drain = std::thread::spawn(move || host.shutdown());
    // Inside the 2s drain window: the bye is queued, the writer parked,
    // shutdown waiting on the connection threads.
    std::thread::sleep(Duration::from_millis(300));
    renderer.kill();
    drain.join().expect("shutdown thread");

    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(45),
        "shutdown completed within the bound ({elapsed:?})"
    );
    assert_endpoint_removed(&socket_path);

    // The lease was released with everything else: a successor owns the
    // root now (and the successor serve also proves no wedged residue on
    // the endpoint path).
    let successor = serve(kill_config(
        root.path(),
        FakeCaptureSource::new(vec![]),
        FakeProvider::new(vec![]),
    ))
    .expect("the lease was released for a successor");
    drop(successor);
}

// --------------------------------------------------------------------- //
// The host's own death with a live renderer (no orphaned lease)
// --------------------------------------------------------------------- /

/// The ownership machinery exercised from the renderer side: the real
/// host binary serves a renderer, is SIGKILLed (flock releases at
/// process death), and the successor — here in-process, the same
/// `serve` the binary wraps — takes over the endpoint and serves the
/// reconnected renderer. No lease is orphaned by the dying host, and
/// the renderer's stale connection observably ends (it is owed nothing
/// durable: documents it wrote live in storage v2).
#[test]
fn a_killed_host_with_a_live_renderer_leaves_no_orphan_lease_and_the_successor_serves() {
    let root = tempfile::tempdir().unwrap();
    let runtime_dir = root.path().join("endpoints");

    // The real binary, its own process, its own lease.
    let mut host_process = ProcessCommand::new(HOST_BIN)
        .arg("--root")
        .arg(root.path())
        .arg("--runtime-dir")
        .arg(&runtime_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("host binary spawns");
    let config = HostConfig::new(root.path(), runtime_dir);
    let socket = config.socket_path();
    let renderer = connect_with_retry(&socket);

    // The renderer durably writes a document through the host process —
    // the evidence that outlives both the connection and the host.
    let revision = Revision {
        rev_id: "rev-1".into(),
        base_revision: 0,
        source_attempt_ids: vec![],
        instruction_template_id: "tpl-none".into(),
        text: "Written before the host died.".into(),
        status: "candidate".into(),
        provenance: "manual".into(),
    };
    renderer
        .send(
            Some("doc-1"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: revision,
            },
        )
        .expect("the renderer wrote through the host process");
    until(
        &renderer,
        "docs.headUpdated",
        |event| event.type_name() == "docs.headUpdated",
        Duration::from_secs(5),
    );

    // Hard kill the host process — flock auto-released, stale socket
    // file left behind, the renderer's connection severed under it.
    host_process.kill().expect("SIGKILL the host process");
    let _ = host_process.wait();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if renderer.is_closed() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the renderer's connection never noticed the host died"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
    drop(renderer);

    // The successor takes the broken lease and the dead endpoint, and
    // the reconnected renderer reads its document back — hydrated from
    // storage v2, not from any process's memory.
    let mut successor_config = kill_config(
        root.path(),
        FakeCaptureSource::new(vec![]),
        FakeProvider::new(vec![]),
    );
    successor_config.runtime = successor_config.runtime.with_document_store(Arc::new(
        starling_runtime::machine::docs::V2DocumentStore::open(root.path())
            .expect("v2 documents store opens"),
    ));
    let mut successor = serve(successor_config).expect("the successor owns the root");
    let reconnected = connect_with_retry(successor.socket_path());
    match reconnected
        .send(
            Some("doc-2"),
            Command::DocsGet {
                doc_id: "notes".into(),
                page: 0,
            },
        )
        .expect("get accepted")
    {
        Receipt::Served(view) => {
            assert_eq!(view["found"], true, "{view}");
            assert_eq!(view["headRevision"], 1, "{view}");
            assert_eq!(view["revisions"][0]["text"], "Written before the host died.");
        }
        other => panic!("expected a served view, got {other:?}"),
    }

    drop(reconnected);
    successor.shutdown();
}
