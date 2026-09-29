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

/// #363: "the open take finishes on the old server" holds across app
/// instances. The owner switches away while an attached instance holds
/// a take on the shared engine: the engine keeps serving until that
/// lease drops, and only then stops.
#[test]
fn owner_switch_waits_for_an_attached_instances_take() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes_a = model_bytes(30, 120_000);
    let bytes_b = model_bytes(31, 130_000);
    let addr = spawn_model_server(vec![
        ("a.gguf".to_string(), Arc::new(bytes_a.clone())),
        ("b.gguf".to_string(), Arc::new(bytes_b.clone())),
    ]);
    let entry_a = entry("model-a", "a.gguf", addr, &bytes_a);
    let entry_b = entry("model-b", "b.gguf", addr, &bytes_b);
    install(&models_dir, &entry_a, &bytes_a);
    install(&models_dir, &entry_b, &bytes_b);
    let catalog = vec![entry_a, entry_b];

    let first = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog.clone()),
        Some("model-a".to_string()),
    );
    let ready = wait_until(&first, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("first instance Ready");
    let old = ready.active.expect("first instance active");
    let second = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog),
        Some("model-a".to_string()),
    );
    let attached = wait_until(&second, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("second instance Ready");
    assert!(!attached.active.expect("attached engine").owned);

    // The attached instance's open take.
    let held = second.lease().expect("lease on the shared engine");
    first.activate("model-b");
    wait_until(&first, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready
            && s.active.as_ref().is_some_and(|active| active.model_id == "model-b")
    })
    .expect("the owner switches to B");
    std::thread::sleep(Duration::from_millis(600));
    let old_alive = pid_alive(old.pid);
    let health = http_get(endpoint_port(held.endpoint()), "/health");

    // Once the take ends, the owner stops the old engine.
    drop(held);
    let stopped = wait_until(&first, TAKEOVER_TIMEOUT, |s| {
        s.active.as_ref().is_some_and(|active| active.model_id == "model-b") && !pid_alive(old.pid)
    });
    second.shutdown();
    first.shutdown();
    assert!(old_alive, "the old engine must outlive the attached take");
    assert!(
        health.is_some_and(|(status, _)| status == 200),
        "the attached take's endpoint keeps serving"
    );
    assert!(
        stopped.is_some() || !cfg!(target_os = "linux"),
        "the old engine stops after the attached take ends"
    );
}
