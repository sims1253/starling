//! The I5 adapter wiring over the I4 transport (issue #220): the
//! context/mode and delivery machines run end-to-end through **real,
//! injected adapters** — not the honest stubs — proving the seam E03's
//! platform adapters (issue #221: IBus/Fcitx, TSF/UIA, macOS
//! accessibility) plug into is wired through the whole production stack.
//!
//! The adapter pair here is a real implementation of the seam contract
//! against a real target the suite can observe: a file. The context
//! provider digests the file's actual bytes; the delivery adapter's
//! compare token is that digest, revalidated immediately before apply
//! (a changed file is a conflict, never a blind overwrite), and its
//! insert really edits the file and confirms with an honest evidence
//! level (the write was read back). Nothing synthesizes authority: the
//! same discipline an OS adapter will follow, at a target a test can
//! read. The stubs remain the unwired default; this suite pins the
//! wiring an adapter's own increment will ride on.

#![cfg(unix)] // matches the sibling suites (one transport, one CI job).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_runtime::machine::context::ContextProvider;
use starling_runtime::machine::delivery::{
    DeliveryAdapter, InsertEvidence, InsertionFailure, Revalidation,
};
use starling_runtime::protocol::{Command, Revision, Span, TargetSnapshotData};
use starling_runtime::provider::FakeProvider;
use starling_runtime::testing::FakeCaptureSource;
use starling_runtime_host::client::{ClientError, EventWire, HostClient};
use starling_runtime_host::{serve, HostConfig, HostHandle};

// --------------------------------------------------------------------- //
// The real adapters (a file target)
// --------------------------------------------------------------------- //

/// FNV-1a 64 over the target's bytes — not cryptographic, exactly like
/// the endpoint-name hash: it needs to detect *change*, not prove
/// identity to a third party (the revalidation comparison is ours).
fn digest_of(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn read_target(target_ref: &str) -> Result<Vec<u8>, String> {
    std::fs::read(target_ref).map_err(|err| format!("target read failed: {err}"))
}

/// A context provider over a real file target: `source` is the path,
/// the snapshot's digest is the file's actual content hash, and the
/// selection spans the whole file (the honest claim an append-shaped
/// target can make).
struct FileContextProvider;

/// A far-future RFC3339 expiry: this adapter has no wall-clock
/// dependency of its own (the host crate does not depend on `time`);
/// the machine's auto-expiry only ever fires *at* an expiry, so a
/// distant one keeps the snapshot live for the test's lifetime the same
/// way a real adapter's now+TTL would.
const FAR_FUTURE_EXPIRY: &str = "2099-01-01T00:00:00Z";

impl ContextProvider for FileContextProvider {
    fn snapshot(&self, source: &str) -> Result<TargetSnapshotData, String> {
        let contents = read_target(source)?;
        Ok(TargetSnapshotData {
            descriptor: format!("file:{source}"),
            digest: digest_of(&contents),
            capabilities: vec!["text-insert".to_string()],
            selection_range: Span { start_offset: 0, end_offset: contents.len() as u64 },
            offset_encoding: "utf-8".to_string(),
            expiry: FAR_FUTURE_EXPIRY.to_string(),
        })
    }
}

/// A delivery adapter over the same file target. `prepare` freezes the
/// content digest as the compare token; `revalidate` compares the live
/// digest against it immediately before apply; `insert` appends the
/// revision's text and reads the file back — the evidence level says
/// exactly that, never "the user saw it land".
struct FileDeliveryAdapter {
    log: Mutex<Vec<String>>,
}

impl FileDeliveryAdapter {
    fn new() -> Arc<Self> {
        Arc::new(FileDeliveryAdapter {
            log: Mutex::new(Vec::new()),
        })
    }

    fn record(&self, entry: String) {
        self.log.lock().expect("adapter log lock").push(entry);
    }

    /// Every insert this adapter performed, in order (the no-auto-apply
    /// assertion reads this).
    fn log(&self) -> Vec<String> {
        self.log.lock().expect("adapter log lock").clone()
    }
}

impl DeliveryAdapter for FileDeliveryAdapter {
    fn prepare(&self, target_ref: &str) -> Result<String, String> {
        let token = digest_of(&read_target(target_ref)?);
        self.record(format!("prepared {target_ref} token={token}"));
        Ok(token)
    }

    fn revalidate(&self, target_ref: &str, compare_token: &str) -> Revalidation {
        let actual = digest_of(&read_target(target_ref).unwrap_or_default());
        if actual == compare_token {
            Revalidation::Unchanged
        } else {
            Revalidation::Changed {
                expected: compare_token.to_string(),
                actual,
            }
        }
    }

    fn insert(
        &self,
        delivery_id: &str,
        target_ref: &str,
        text: &str,
    ) -> Result<InsertEvidence, InsertionFailure> {
        self.record(format!("insert {delivery_id} -> {target_ref}"));
        let write = std::fs::OpenOptions::new()
            .append(true)
            .open(target_ref)
            .and_then(|mut file| {
                use std::io::Write;
                file.write_all(text.as_bytes())?;
                file.flush()
            });
        if let Err(err) = write {
            return Err(InsertionFailure {
                reason: format!("target write failed: {err}"),
                fallback_suggested: true,
            });
        }
        // Read-back evidence: we verified the bytes are in the target —
        // nothing more (no synthetic key acceptance, no "user saw it").
        let read_back = read_target(target_ref)
            .map(|bytes| bytes.ends_with(text.as_bytes()))
            .unwrap_or(false);
        if !read_back {
            return Err(InsertionFailure {
                reason: "target read-back mismatch".to_string(),
                fallback_suggested: true,
            });
        }
        Ok(InsertEvidence {
            level: "file-readback".to_string(),
        })
    }

    fn describe(&self) -> String {
        "file target (real read/write/re-read)".to_string()
    }
}

// --------------------------------------------------------------------- //
// Harness
// --------------------------------------------------------------------- //

fn until(
    client: &HostClient,
    label: &str,
    predicate: impl Fn(&EventWire) -> bool,
    deadline: Duration,
) -> EventWire {
    let start = Instant::now();
    loop {
        match client.recv_event_timeout(Duration::from_millis(20)) {
            Ok(event) => {
                if predicate(&event) {
                    return event;
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

fn boot(root: &Path) -> (HostHandle, HostClient, Arc<FileDeliveryAdapter>) {
    let delivery = FileDeliveryAdapter::new();
    let mut config = HostConfig::new(root, root.join("endpoints"));
    config.runtime = config
        .runtime
        .with_capture_source(FakeCaptureSource::new(vec![]))
        .with_provider(FakeProvider::new(vec![]))
        .with_context_provider(Arc::new(FileContextProvider))
        .with_delivery_adapter(Arc::clone(&delivery) as Arc<dyn DeliveryAdapter>);
    let host = serve(config).expect("host serves");
    let client = connect_with_retry(host.socket_path());
    (host, client, delivery)
}

fn revision(rev_id: &str, text: &str) -> Revision {
    Revision {
        rev_id: rev_id.to_string(),
        base_revision: 0,
        source_attempt_ids: vec![],
        instruction_template_id: "tpl-none".to_string(),
        text: text.to_string(),
        status: "candidate".to_string(),
        provenance: "recognition".to_string(),
    }
}

/// Commits one revision as a document head and returns its id — the
/// delivery machine only prepares against committed heads.
fn commit_head(client: &HostClient, text: &str) -> String {
    let rev = revision("rev-adapters", text);
    let rev_id = rev.rev_id.clone();
    client
        .send(
            Some("doc-adapters"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: rev,
            },
        )
        .expect("updateHead accepted");
    until(
        client,
        "docs.headUpdated",
        |event| event.type_name() == "docs.headUpdated",
        Duration::from_secs(5),
    );
    rev_id
}

fn prepare(client: &HostClient, revision_id: &str, target: &Path) -> (String, String) {
    client
        .send(
            Some("dlv-adapters"),
            Command::DeliveryPrepare {
                revision_id: revision_id.to_string(),
                target_ref: target.display().to_string(),
            },
        )
        .expect("prepare accepted");
    let prepared = until(
        client,
        "delivery.prepared",
        |event| event.type_name() == "delivery.prepared",
        Duration::from_secs(5),
    );
    let delivery_id = prepared.payload()["deliveryId"]
        .as_str()
        .expect("deliveryId")
        .to_string();
    let token = prepared.payload()["compareToken"]
        .as_str()
        .expect("compareToken")
        .to_string();
    (delivery_id, token)
}

// --------------------------------------------------------------------- //
// The wiring, pinned over IPC
// --------------------------------------------------------------------- //

/// `context.snapshot` resolves through the injected provider over the
/// transport: the event's descriptor and digest are the real file's —
/// the stub's `stub:` labels are gone, and digesting real bytes is the
/// observable difference.
#[test]
fn context_snapshots_resolve_through_an_injected_adapter_over_ipc() {
    let root = tempfile::tempdir().unwrap();
    let target: PathBuf = root.path().join("target.txt");
    std::fs::write(&target, b"real target contents").expect("write target");

    let (mut host, client, _delivery) = boot(root.path());
    client
        .send(
            Some("ctx-adapters"),
            Command::ContextSnapshot {
                source: target.display().to_string(),
            },
        )
        .expect("snapshot accepted");
    let event = until(
        &client,
        "context.targetSnapshot",
        |event| event.type_name() == "context.targetSnapshot",
        Duration::from_secs(5),
    );
    assert_eq!(
        event.payload()["descriptor"],
        format!("file:{}", target.display()),
        "the descriptor names the real target"
    );
    assert_eq!(
        event.payload()["digest"],
        digest_of(b"real target contents"),
        "the digest is over the file's actual bytes"
    );
    assert_eq!(event.payload()["selectionRange"]["endOffset"], 20u64);
    assert_eq!(event.payload()["offsetEncoding"], "utf-8");

    drop(client);
    host.shutdown();
}

/// The full delivery path through the injected adapter: prepare freezes
/// the digest, apply revalidates it, inserts the revision's text into
/// the real target, and confirms **stating its evidence** — the file
/// really changed, and the wire says exactly how much that proves.
#[test]
fn delivery_applies_through_an_injected_adapter_over_ipc() {
    let root = tempfile::tempdir().unwrap();
    let target: PathBuf = root.path().join("target.txt");
    std::fs::write(&target, b"before ").expect("write target");

    let (mut host, client, delivery) = boot(root.path());
    let revision_id = commit_head(&client, "inserted by the file adapter.");
    let (delivery_id, token) = prepare(&client, &revision_id, &target);
    assert_eq!(
        token,
        digest_of(b"before "),
        "the compare token froze the target's digest at prepare"
    );

    client
        .send(
            Some("dlv-adapters"),
            Command::DeliveryApply { delivery_id },
        )
        .expect("apply accepted");
    // The machine walks SubmittedUnconfirmed then Confirmed on one
    // stream — collect the window and assert both ends of it.
    let mut saw_submitted = false;
    let confirmed = loop {
        let event = until(
            &client,
            "delivery.confirmed | delivery.submittedUnconfirmed",
            |event| {
                event.type_name() == "delivery.confirmed"
                    || event.type_name() == "delivery.submittedUnconfirmed"
            },
            Duration::from_secs(5),
        );
        match event.type_name() {
            "delivery.submittedUnconfirmed" => saw_submitted = true,
            _ => break event,
        }
    };
    assert!(
        saw_submitted,
        "the unconfirmed edge crossed before the confirmation"
    );
    assert_eq!(
        confirmed.payload()["evidenceLevel"],
        "file-readback",
        "the confirmation states the adapter's honest evidence: {}",
        confirmed.payload()
    );

    // The real target changed: the revision's text is physically there.
    let contents = std::fs::read(&target).expect("read target");
    assert!(
        String::from_utf8_lossy(&contents).starts_with("before inserted by the file adapter."),
        "the adapter inserted the revision's text into the real target: {contents:?}"
    );
    // Exactly one insert, from apply (the log is the adapter's own word).
    let inserts = delivery
        .log()
        .iter()
        .filter(|entry| entry.starts_with("insert"))
        .count();
    assert_eq!(inserts, 1, "one apply, one insert: {:?}", delivery.log());

    drop(client);
    host.shutdown();
}

/// The conflict path: the target changed between freeze and apply — the
/// revalidation immediately before apply sees it, the delivery lands in
/// `conflict` with both digests on the wire, and **the file was never
/// written** (no blind overwrite of a target that moved).
#[test]
fn a_changed_target_conflicts_and_the_target_is_never_blindly_written() {
    let root = tempfile::tempdir().unwrap();
    let target: PathBuf = root.path().join("target.txt");
    std::fs::write(&target, b"version A").expect("write target");

    let (mut host, client, _delivery) = boot(root.path());
    let revision_id = commit_head(&client, "the stale writer's text.");
    let (delivery_id, frozen_token) = prepare(&client, &revision_id, &target);

    // The target moves out-of-band (another writer, between freeze and
    // the user-initiated apply).
    std::fs::write(&target, b"version B (someone else edited)").expect("the target changes");

    client
        .send(
            Some("dlv-adapters"),
            Command::DeliveryApply { delivery_id },
        )
        .expect("apply accepted");
    let conflict = until(
        &client,
        "delivery.conflict",
        |event| event.type_name() == "delivery.conflict",
        Duration::from_secs(5),
    );
    assert_eq!(
        conflict.payload()["expectedTarget"],
        frozen_token.as_str(),
        "the conflict names the frozen target"
    );
    assert_eq!(
        conflict.payload()["actualTarget"],
        digest_of(b"version B (someone else edited)"),
        "the conflict names the live target"
    );

    let contents = std::fs::read(&target).expect("read target");
    assert!(
        !String::from_utf8_lossy(&contents).contains("the stale writer's text."),
        "the changed target was never written: {contents:?}"
    );

    drop(client);
    host.shutdown();
}

/// No auto-apply: preparing a delivery never inserts (the §2.5
/// invariant an OS adapter inherits, not one each re-proves), and a
/// prepare against an unknown revision is refused at the machine
/// boundary — never turned into an empty insertion.
#[test]
fn prepare_alone_never_inserts_and_unknown_revisions_are_refused() {
    let root = tempfile::tempdir().unwrap();
    let target: PathBuf = root.path().join("target.txt");
    std::fs::write(&target, b"untouched").expect("write target");

    let (mut host, client, delivery) = boot(root.path());
    let revision_id = commit_head(&client, "prepared but never applied.");
    let _ = prepare(&client, &revision_id, &target);

    // Nothing has happened to the target, and the adapter never inserted.
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        std::fs::read(&target).unwrap(),
        b"untouched",
        "prepare alone touched the target"
    );
    assert!(
        delivery.log().iter().all(|entry| !entry.starts_with("insert")),
        "insert ran without apply: {:?}",
        delivery.log()
    );

    // An unknown revision never reaches the adapter.
    match client.send(
        Some("dlv-adapters"),
        Command::DeliveryPrepare {
            revision_id: "rev-never-committed".into(),
            target_ref: target.display().to_string(),
        },
    ) {
        Err(ClientError::Rejected(
            starling_runtime::machine::Rejection::UnknownRevision { revision_id },
        )) => assert_eq!(revision_id, "rev-never-committed"),
        other => panic!("expected an UnknownRevision refusal, got {other:?}"),
    }

    drop(client);
    host.shutdown();
}
