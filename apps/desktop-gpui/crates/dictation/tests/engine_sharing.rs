//! Integration: two managers sharing one state dir — the second app
//! instance attaches to the first's sidecar and takes over when the
//! first shuts down (plan test 7).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use starling_dictation::engine::{EngineManager, EnginePhase};

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const TAKEOVER_TIMEOUT: Duration = Duration::from_secs(20);

/// Plan test 7: a second manager (same state_dir) attaches to the
/// first's sidecar (no new process, `owned == false`); shutting the
/// first down makes the second take over by spawning its own.
#[test]
fn second_instance_attaches_and_takes_over() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(8, 100_000);
    let addr = spawn_model_server(vec![("f.gguf".to_string(), Arc::new(bytes.clone()))]);
    let entry_f = entry("model-f", "f.gguf", addr, &bytes);
    install(&models_dir, &entry_f, &bytes);
    let catalog = vec![entry_f];

    let first = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog.clone()),
        Some("model-f".to_string()),
    );
    let ready = wait_until(&first, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("first instance Ready");
    let original = ready.active.clone().expect("first instance active");
    assert!(original.owned);
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "one sidecar after the first launch");
    }

    // A second app instance with the same state dir reuses the sidecar.
    let second = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog),
        Some("model-f".to_string()),
    );
    let attached = wait_until(&second, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready && s.active.is_some()
    })
    .expect("second instance becomes Ready");
    let view = attached.active.expect("second instance active engine");
    assert!(!view.owned, "the second instance attaches, not spawns");
    assert_eq!(view.pid, original.pid, "no new process was started");
    assert_eq!(view.model_id, "model-f");
    assert_eq!(view.endpoint, original.endpoint);
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "attaching still means exactly one sidecar");
    }

    // The owner shuts down; the second instance takes over by spawning
    // its own sidecar.
    first.shutdown();
    let taken_over = wait_until(&second, TAKEOVER_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready
            && s.active
                .as_ref()
                .is_some_and(|active| active.owned && active.pid != original.pid)
    })
    .expect("the attached instance takes over after the owner exits");
    assert_eq!(taken_over.active.unwrap().model_id, "model-f");
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "take-over keeps exactly one sidecar");
    }

    let lease = second.lease().expect("lease after take-over");
    let (status, _) = http_get(endpoint_port(lease.endpoint()), "/health").expect("health");
    assert_eq!(status, 200);
    drop(lease);
    second.shutdown();
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 0);
    }
}
