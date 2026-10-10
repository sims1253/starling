//! Integration: two managers sharing one state dir — the second app
//! instance attaches to the first's sidecar and takes over when the
//! first shuts down (plan test 7).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use starling_dictation::engine::{EngineManager, EnginePhase, SwapDecision};

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
    // Watched continuously: the old engine must stay up the whole time
    // the attached take is open, not just at one sampled instant.
    let mut old_alive = true;
    let watch_until = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < watch_until {
        old_alive &= pid_alive(old.pid);
        std::thread::sleep(Duration::from_millis(100));
    }
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

/// Two models A and B, both installed, served by one model server.
fn two_models(
    root: &std::path::Path,
) -> (
    std::path::PathBuf,
    starling_dictation::engine::CatalogEntry,
    starling_dictation::engine::CatalogEntry,
) {
    let models_dir = root.join("models");
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
    (models_dir, entry_a, entry_b)
}

/// #362 step 3: an instance that starts with no model and then
/// activates the model another instance already serves attaches to that
/// sidecar through the same protocol as a launch — no second process.
#[test]
fn first_activation_attaches_to_the_shared_sidecar() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let state_dir = root.path().join("state");
    let (models_dir, entry_a, entry_b) = two_models(root.path());
    let catalog = vec![entry_a, entry_b];

    let first = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog.clone()),
        Some("model-a".to_string()),
    );
    let owner = wait_until(&first, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("first instance Ready")
        .active
        .expect("owner engine");
    let second = EngineManager::start(config(&engine_dir, &models_dir, &state_dir, catalog), None);
    wait_until(&second, READY_TIMEOUT, |s| s.phase == EnginePhase::NoModel)
        .expect("second instance starts without a model");

    second.activate("model-a");
    let attached = wait_until(&second, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready && s.active.is_some()
    })
    .and_then(|s| s.active);
    let count = count_engine_processes(engine_dir.to_str().unwrap());
    second.shutdown();
    first.shutdown();
    let attached = attached.expect("the second instance becomes Ready on A");
    assert!(!attached.owned, "first activation attaches instead of spawning");
    assert_eq!(attached.pid, owner.pid);
    if let Some(count) = count {
        assert_eq!(count, 1, "still exactly one sidecar");
    }
}

/// #363 memory policy: stopping an attached engine frees nothing (its
/// owner keeps it running), so it must not be credited as unloadable. A
/// reading that only fits the incoming model after an unload therefore
/// refuses the switch instead of loading both models.
#[test]
fn an_attached_engine_is_not_counted_as_freeable_memory() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let state_dir = root.path().join("state");
    let (models_dir, entry_a, entry_b) = two_models(root.path());
    let catalog = vec![entry_a, entry_b];

    let first = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog.clone()),
        Some("model-a".to_string()),
    );
    wait_until(&first, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("first instance Ready");
    let mut tight = config(&engine_dir, &models_dir, &state_dir, catalog);
    // Fits B only if A's ~384 MiB were freed (see engine_drain.rs).
    tight.available_memory_override = Some(Some(600 * 1024 * 1024));
    let second = EngineManager::start(tight, Some("model-a".to_string()));
    let attached = wait_until(&second, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("second instance Ready")
        .active
        .expect("attached engine");
    assert!(!attached.owned);

    second.activate("model-b");
    let decided = wait_until(&second, READY_TIMEOUT, |s| {
        s.pending_decision.is_some() && s.switch.is_none()
    });
    let count = count_engine_processes(engine_dir.to_str().unwrap());
    let still_attached = second
        .snapshot()
        .active
        .is_some_and(|active| active.model_id == "model-a" && !active.owned);
    second.shutdown();
    first.shutdown();
    let decided = decided.expect("the switch ends with a decision");
    assert!(
        matches!(decided.pending_decision, Some(SwapDecision::Refused { .. })),
        "got {:?}",
        decided.pending_decision
    );
    assert!(still_attached, "the attached engine keeps serving");
    if let Some(count) = count {
        assert_eq!(count, 1, "B was never loaded next to A");
    }
}

/// #220: a backend change on an instance attached to another's sidecar
/// does not start a second one beside it (a reload spawns unshared, so
/// both would stay resident): the attached engine keeps serving, the
/// instance says why, and the owner's sidecar stays the only one.
#[test]
fn a_backend_change_while_attached_starts_no_second_sidecar() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(9, 100_000);
    let addr = spawn_model_server(vec![("g.gguf".to_string(), Arc::new(bytes.clone()))]);
    let entry_g = entry("model-g", "g.gguf", addr, &bytes);
    install(&models_dir, &entry_g, &bytes);
    let catalog = vec![entry_g];

    let owner = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog.clone()),
        Some("model-g".to_string()),
    );
    let original = wait_until(&owner, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("owner Ready")
        .active
        .expect("owner active");
    let attached = EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, catalog),
        Some("model-g".to_string()),
    );
    wait_until(&attached, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready && s.active.as_ref().is_some_and(|a| !a.owned)
    })
    .expect("the second instance attaches");

    attached.set_backend_override(Some(starling_dictation::engine::Backend::Cpu));
    let told = wait_until(&attached, READY_TIMEOUT, |s| s.last_error.is_some())
        .expect("the attached instance says why nothing reloaded");
    assert!(
        told.last_error.as_deref().unwrap_or("").contains("another Starling process"),
        "{:?}",
        told.last_error
    );
    // Long enough for a reload to have spawned and warmed.
    std::thread::sleep(Duration::from_secs(2));
    let after = attached.snapshot();
    assert!(after.switch.is_none(), "no reload runs");
    let active = after.active.expect("still serving");
    assert!(!active.owned);
    assert_eq!(active.pid, original.pid, "still the owner's sidecar");
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "one sidecar, not two");
    }
    attached.shutdown();
    owner.shutdown();
}
