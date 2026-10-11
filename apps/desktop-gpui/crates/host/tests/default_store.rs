//! The production host wires storage v2 at its explicit root and
//! refuses to start when that root will not open.

#![cfg(unix)] // matches the sibling suites

use starling_runtime_host::HostConfig;

/// The production entry point opts into the durable store through an
/// explicit root, and an unopenable root is a startup refusal, never a
/// silently non-durable host.
#[test]
fn production_host_wires_the_durable_store_and_refuses_to_degrade() {
    let dir = tempfile::tempdir().expect("host root temp dir");
    let runtime_dir = dir.path().join("endpoints");

    let config = HostConfig::production(dir.path(), Some(runtime_dir.clone()))
        .expect("production config at an openable root");
    assert_eq!(
        config.runtime.capture_store.describe(),
        "storage-v2",
        "the production entry point wires the durable store"
    );
    // The I5 wiring (issue #220): the documents machine persists through
    // the same root's `documents`/`revisions` tables — never the
    // session-scoped in-memory default.
    assert_eq!(
        config.runtime.document_store.describe(),
        "storage-v2",
        "the production entry point wires the durable document store"
    );

    // A root occupied by a regular file cannot open (create_dir_all
    // fails) — production must error, never fall back.
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"not a directory").expect("write the blocker");
    let err = match HostConfig::production(&blocker, Some(runtime_dir)) {
        Ok(_) => panic!("an unopenable root must be a startup refusal"),
        Err(err) => err,
    };
    assert!(
        // Couples to config.rs's error wording — `HostConfig::production`
        // returns `String`, so a phrase is the only discriminator; if the
        // message is rephrased, update this with it.
        err.contains("will not open"),
        "the refusal names the root: {err}"
    );
}
