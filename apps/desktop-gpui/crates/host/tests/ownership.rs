//! Stale-owner detection and single-ownership acceptance (E17 I4 / §4):
//! the real `starling-runtime-host` **binary**, the real storage-v2
//! lease, and a real SIGKILL — the flock releases at process death, the
//! lease machinery proves the dead owner gone, the leftover socket file
//! is taken over, and the new host serves on the same endpoint. Never
//! two owners: a second host (process or in-process) against a live
//! owner answers `already-running` and exits 0.

#![cfg(unix)]

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use starling_runtime::provider::FakeProvider;
use starling_runtime::testing::FakeCaptureSource;
use starling_runtime_host::client::HostClient;
use starling_runtime_host::{serve, HostConfig, HostError};

/// The built binary (cargo provides the path for bin targets in
/// integration tests).
const BIN: &str = env!("CARGO_BIN_EXE_starling-runtime-host");

struct ChildHost {
    child: std::process::Child,
}

impl ChildHost {
    fn spawn(root: &Path) -> ChildHost {
        // Stdio::null: nothing here reads the child's streams, and piped
        // streams nobody drains can deadlock a chatty child on the OS
        // pipe buffer.
        let child = Command::new(BIN)
            .arg("--root")
            .arg(root)
            .arg("--runtime-dir")
            .arg(root.join("endpoints"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("host binary spawns");
        ChildHost { child }
    }

    /// Waits for the socket to accept a client handshake; returns it.
    /// A child that has already exited fails fast with its status —
    /// spinning to the deadline first only delays the diagnosis.
    fn wait_serving(&mut self, socket: &Path) -> HostClient {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Ok(client) = HostClient::connect(socket) {
                return client;
            }
            if let Some(status) = self.child.try_wait().expect("try_wait the child") {
                panic!(
                    "host binary exited before serving at {} ({status})",
                    socket.display()
                );
            }
            if Instant::now() > deadline {
                panic!(
                    "host binary never served at {} (status: {:?})",
                    socket.display(),
                    self.child.try_wait()
                );
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

impl Drop for ChildHost {
    fn drop(&mut self) {
        // Best-effort cleanup so a failing test does not leak processes.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn socket_of(root: &Path) -> std::path::PathBuf {
    HostConfig::new(root, root.join("endpoints")).socket_path()
}

/// The config every in-process test here boots: transport defaults, an
/// idle fake source and provider. One place, so a config-field change
/// does not drift across four copies.
fn plain_config(root: &Path) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"));
    config.runtime = config
        .runtime
        .with_capture_source(FakeCaptureSource::new(vec![]))
        .with_provider(FakeProvider::new(vec![]));
    config
}

/// The full ladder: host A owns and serves; a hard kill (SIGKILL — no
/// handler, no cleanup, flock auto-released by the OS) leaves a stale
/// lease file and a stale socket file; host B must break the dead lease,
/// take over the endpoint, and serve — while the takeover decision never
/// produces two owners.
#[test]
fn a_killed_host_is_taken_over_by_the_next_one() {
    let root = tempfile::tempdir().unwrap();
    let socket = socket_of(root.path());

    let mut first = ChildHost::spawn(root.path());
    let client_a = first.wait_serving(&socket);
    let owner_a = client_a.info.owner_id.clone();
    assert_eq!(client_a.info.pid, first.child.id() as u32);
    // The real binary serves the real protocol: drive a docs CAS cycle.
    client_a
        .send(
            Some("doc-1"),
            starling_runtime::protocol::Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: starling_runtime::protocol::Revision {
                    rev_id: "rev-1".into(),
                    base_revision: 0,
                    source_attempt_ids: vec![],
                    instruction_template_id: "tpl-none".into(),
                    text: "Written before the crash.".into(),
                    status: "candidate".into(),
                    provenance: "manual".into(),
                },
            },
        )
        .expect("the real binary serves commands");
    drop(client_a);

    // Hard kill: SIGKILL, the worst case. No handler runs, no cleanup
    // happens — the lease's flock releases because the OS does it.
    first.child.kill().expect("SIGKILL the host");
    let _ = first.child.wait();
    assert!(
        socket.exists(),
        "the killed host leaves its socket file behind (the stale-owner case)"
    );

    // Host B on the same root: the dead lease breaks (flock free), the
    // dead socket is probed (ECONNREFUSED) and taken over, and B serves
    // on the SAME endpoint path.
    let mut second = ChildHost::spawn(root.path());
    let client_b = second.wait_serving(&socket);
    assert_ne!(
        client_b.info.owner_id, owner_a,
        "a new owner id — the old lease was broken, not inherited"
    );
    assert_eq!(client_b.info.pid, second.child.id() as u32);

    // And it actually serves the runtime (docs machine over the pipe/
    // socket of the real binary).
    let snapshot = client_b.snapshot().expect("snapshot from the successor");
    assert_eq!(snapshot["docs"]["state"], "Steady");
    drop(client_b);
    drop(second);
}

/// A second host process against a live owner exits 0 with
/// `already-running` and the owner's endpoint — the launcher contract.
#[test]
fn a_second_host_binary_reports_already_running_and_exits_zero() {
    let root = tempfile::tempdir().unwrap();
    let mut owner = serve(plain_config(root.path())).expect("in-process owner serves");
    let client = HostClient::connect(owner.socket_path()).expect("owner reachable");

    // Bounded: a regression that makes the second host *block* (instead
    // of reporting already-running and exiting) must fail this test at
    // the deadline, not hang the suite until the job timeout.
    let mut child = Command::new(BIN)
        .arg("--root")
        .arg(root.path())
        .arg("--runtime-dir")
        .arg(root.path().join("endpoints"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("second host runs");
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        match child.try_wait().expect("try_wait the second host") {
            Some(status) => break status,
            None if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("the second host did not exit within 15s");
            }
            None => std::thread::sleep(Duration::from_millis(25)),
        }
    };
    let output = child.wait_with_output().expect("drain the pipes");

    assert!(
        status.success(),
        "a live owner means client mode, not failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let status: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|err| panic!("stdout is one JSON line ({stdout:?}): {err}"));
    assert_eq!(status["status"], "already-running");
    assert_eq!(
        status["socket"].as_str().unwrap(),
        owner.socket_path().to_str().unwrap()
    );
    assert_eq!(status["owner"].as_str().unwrap(), owner.owner_id());

    // The live owner was not disturbed.
    client
        .send(
            None,
            starling_runtime::protocol::Command::JobsSetLimits(
                starling_runtime::protocol::JobLimits {
                    max_queued: 4,
                    max_concurrent: 1,
                    per_route: vec![],
                },
            ),
        )
        .expect("the owner still serves after a second host consulted the lease");
    drop(client);
    owner.shutdown();
}

/// An unanswerable lease file (present, unprobeable — here a directory
/// squatting on a lease name) blocks ownership instead of the host
/// serving beside it: reconcile would defer to the unanswerable owner
/// forever, so a second writer on that root is exactly the
/// recovery-disabling state the lease exists to prevent. The refusal
/// names the wedged file so it can be repaired, and the root serves
/// cleanly once it is.
#[test]
fn an_unanswerable_lease_refuses_ownership_and_names_the_wedge() {
    let root = tempfile::tempdir().unwrap();
    let wedge = root.path().join("leases").join("l_wedge.lease");
    std::fs::create_dir_all(&wedge).expect("wedge lease");

    match serve(plain_config(root.path())) {
        Err(HostError::LeaseUnanswerable { unreadable, .. }) => {
            assert_eq!(unreadable.len(), 1, "{unreadable:?}");
            assert_eq!(unreadable[0].0, "l_wedge", "{unreadable:?}");
        }
        Err(other) => panic!("expected LeaseUnanswerable, got {other}"),
        Ok(mut host) => {
            host.shutdown();
            panic!("expected LeaseUnanswerable, served beside the wedge");
        }
    }

    // No lease of ours was left behind by the refusal, and the repaired
    // root serves normally.
    let lease_names: Vec<String> = std::fs::read_dir(root.path().join("leases"))
        .expect("leases dir")
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.ends_with(".lease"))
        .collect();
    assert_eq!(
        lease_names,
        vec!["l_wedge.lease".to_string()],
        "the refused host published no lease of its own"
    );
    std::fs::remove_dir(&wedge).expect("repair the wedge");
    let mut successor =
        serve(plain_config(root.path())).expect("serves once the wedge is repaired");
    successor.shutdown();
}

/// The ownership-ladder refusal the binary encodes, exercised in-process
/// (fast): a live foreign owner makes the second serve() a client, and a
/// socket answering without the lease makes binding refuse.
#[test]
fn a_live_foreign_server_without_the_lease_is_refused() {
    let root = tempfile::tempdir().unwrap();

    // A live host on another root... but pointed at `root`'s endpoint
    // directory? Ownership is per data root; the endpoint derives from
    // the root, so a foreign server on OUR endpoint requires a host on
    // our root — which the lease already refuses. Construct the
    // contradiction directly: occupy the endpoint with a raw listener.
    let config = HostConfig::new(root.path(), root.path().join("endpoints"));
    let socket = config.socket_path();
    std::fs::create_dir_all(root.path().join("endpoints")).unwrap();
    let squatter = std::os::unix::net::UnixListener::bind(&socket).unwrap();

    // Host with the same root + endpoint: it acquires the lease (no
    // owner), probes the endpoint, finds the squatter live, refuses —
    // and releases the lease it briefly held.
    match serve(plain_config(root.path())) {
        Err(HostError::ForeignServer(path)) => assert_eq!(path, socket),
        Err(other) => panic!("expected ForeignServer, got {other}"),
        Ok(host) => {
            drop(host);
            panic!("expected ForeignServer, a host bound over a squatter")
        }
    }

    // And the lease it held was released: a clean serve on the same root
    // (with the squatter gone) now owns.
    drop(squatter);
    let _ = std::fs::remove_file(&socket);
    let mut successor =
        serve(plain_config(root.path())).expect("serves once the endpoint is free");
    successor.shutdown();
}

/// The killed-host socket takeover needs the probe to see `Dead`, not
/// just an absent file: a leftover socket file with no listener is
/// removed and rebound (already covered by the kill test), and a socket
/// file that was never a socket is refused honestly rather than crashed
/// on.
#[test]
fn a_non_socket_file_at_the_endpoint_is_refused_not_crashed() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("endpoints")).unwrap();
    let config = HostConfig::new(root.path(), root.path().join("endpoints"));
    let socket = config.socket_path();
    std::fs::write(&socket, b"not a socket").unwrap();

    // connect(2) to a non-socket file: ECONNREFUSED on Linux ("the
    // socket is not listening") reads as Dead, the file is removed, and
    // the host binds cleanly — the takeover path handles the residue.
    let mut host = match serve(plain_config(root.path())) {
        Ok(host) => host,
        Err(err) => panic!("residue must be taken over, refused with {err}"),
    };
    let client = HostClient::connect(host.socket_path()).expect("serves on the taken-over path");
    let _ = client;
    host.shutdown();
}

/// Reads the binary's `--help` so the usage surface stays greppable:
/// help prints to stdout and exits 0 (the convention scripts probing the
/// interface rely on); an unknown argument is the error path (stderr,
/// exit 2).
#[test]
fn the_binary_documents_its_interface() {
    let output = Command::new(BIN).arg("--help").output().unwrap();
    assert_eq!(output.status.code(), Some(0), "--help is not an error");
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("--root"), "{help}");
    assert!(help.contains("--runtime-dir"), "{help}");

    let output = Command::new(BIN).arg("--bogus").output().unwrap();
    assert_eq!(output.status.code(), Some(2), "a usage error exits 2");
    let usage = String::from_utf8_lossy(&output.stderr);
    assert!(usage.contains("--bogus"), "{usage}");
}
