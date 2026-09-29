//! Integration: drain-then-swap with an injected memory reading (plan
//! test 9) — NeedsDrain decision, confirm, wait for the lease to drop,
//! switch on the drained slot.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use starling_dictation::engine::{EnginePhase, InstallState, SwapDecision, SwapMode, SwitchStage};

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const SHORT_TIMEOUT: Duration = Duration::from_secs(10);

/// Plan test 9: with a memory reading that only fits the pair after the
/// outgoing model is unloaded, activation surfaces NeedsDrain; after
/// confirmation it waits for the open take's lease, stops the old
/// engine, and completes the switch (mode Drain).
#[test]
fn drain_swap_waits_for_the_take_then_switches() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);

    let bytes_a = model_bytes(30, 120_000);
    let bytes_b = model_bytes(31, 130_000);
    let addr = spawn_model_server(vec![
        ("a.gguf".to_string(), Arc::new(bytes_a.clone())),
        ("b.gguf".to_string(), Arc::new(bytes_b.clone())),
    ]);
    let entry_a = entry("model-a", "a.gguf", addr, &bytes_a);
    let entry_b = entry("model-b", "b.gguf", addr, &bytes_b);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");
    install(&models_dir, &entry_a, &bytes_a);
    install(&models_dir, &entry_b, &bytes_b);

    let mut config = config(&engine_dir, &models_dir, &state_dir, vec![entry_a, entry_b]);
    // Inject a reading between "rolling fits" and "pair fits only after
    // draining": with the ~384 MiB resident estimates and the 512 MiB
    // margin, 600 MiB needs the drain.
    config.available_memory_override = Some(Some(600 * 1024 * 1024));

    let manager =
        starling_dictation::engine::EngineManager::start(config, Some("model-a".to_string()));
    let ready =
        wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready).expect("Ready on A");
    let old = ready.active.clone().expect("active A");
    assert_eq!(old.model_id, "model-a");

    // An open take holds a lease on A.
    let lease_a = manager.lease().expect("lease on A");

    manager.activate("model-b");
    let snapshot = wait_until(&manager, SHORT_TIMEOUT, |s| {
        matches!(s.pending_decision, Some(SwapDecision::NeedsDrain { .. }))
    })
    .expect("tight memory surfaces NeedsDrain");
    if let Some(SwapDecision::NeedsDrain { needed, available }) = &snapshot.pending_decision {
        assert_eq!(*available, 600 * 1024 * 1024);
        assert!(
            *needed > *available,
            "the decision shows why draining is needed (needed {needed}, available {available})"
        );
    }
    assert_eq!(
        snapshot.phase,
        EnginePhase::Ready,
        "A keeps serving while the decision is open"
    );

    // Confirm: the switch waits for the take to finish.
    manager.confirm_drain_swap();
    let waiting = wait_until(&manager, SHORT_TIMEOUT, |s| {
        s.switch
            .as_ref()
            .is_some_and(|switch| switch.stage == SwitchStage::WaitingForTake)
    })
    .expect("confirmed drain waits for the open take");
    assert!(
        waiting.pending_decision.is_none(),
        "the decision is consumed"
    );
    // The old engine still serves the take.
    assert!(pid_alive(old.pid), "the leased engine stays up");
    let (status, _) = http_get(endpoint_port(&old.endpoint), "/health").expect("A answers");
    assert_eq!(status, 200);

    // The take finishes (lease drops) — the switch completes.
    drop(lease_a);
    let done = wait_until(&manager, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready
            && s.active.as_ref().is_some_and(|a| a.model_id == "model-b")
            && s.switch.is_none()
    })
    .expect("drain swap completes after the lease drops");
    let report = done.last_switch.as_ref().expect("switch report");
    assert_eq!(report.mode, SwapMode::Drain);
    assert_eq!(report.from.as_deref(), Some("model-a"));
    assert_eq!(report.to, "model-b");
    assert!(!pid_alive(old.pid), "the old engine was stopped");
    assert_eq!(install_of(&done, "model-b"), InstallState::Installed);

    // New leases serve B.
    let lease_b = manager.lease().expect("lease on B");
    assert_eq!(lease_b.model_id(), "model-b");
    drop(lease_b);

    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "the drain swap ends with one sidecar");
    }
    manager.shutdown();
}

/// The refuse path: when even draining cannot free enough memory, the
/// decision surfaces with numbers and nothing changes — no spawn, the
/// old model keeps serving, no switch runs.
#[test]
fn refused_swap_changes_nothing() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);

    let bytes_a = model_bytes(40, 120_000);
    let bytes_b = model_bytes(41, 130_000);
    let addr = spawn_model_server(vec![
        ("a.gguf".to_string(), Arc::new(bytes_a.clone())),
        ("b.gguf".to_string(), Arc::new(bytes_b.clone())),
    ]);
    let entry_a = entry("model-a", "a.gguf", addr, &bytes_a);
    let entry_b = entry("model-b", "b.gguf", addr, &bytes_b);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");
    install(&models_dir, &entry_a, &bytes_a);
    install(&models_dir, &entry_b, &bytes_b);

    let mut config = config(&engine_dir, &models_dir, &state_dir, vec![entry_a, entry_b]);
    // Far too little memory even after draining: ~896 MiB would be
    // needed, ~100 MiB is available and draining frees ~384 MiB more.
    config.available_memory_override = Some(Some(100 * 1024 * 1024));

    let manager =
        starling_dictation::engine::EngineManager::start(config, Some("model-a".to_string()));
    let ready =
        wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready).expect("Ready on A");
    let old = ready.active.clone().expect("active A");

    manager.activate("model-b");
    let snapshot = wait_until(&manager, SHORT_TIMEOUT, |s| {
        matches!(s.pending_decision, Some(SwapDecision::Refused { .. }))
    })
    .expect("a hopeless memory situation surfaces Refused");
    if let Some(SwapDecision::Refused { needed, available }) = &snapshot.pending_decision {
        assert_eq!(*available, 100 * 1024 * 1024);
        assert!(
            needed > available,
            "needed {needed} must exceed {available}"
        );
    }
    // Nothing changed: A still serves, no switch is running.
    assert_eq!(snapshot.phase, EnginePhase::Ready);
    let active = snapshot.active.expect("A is still active");
    assert_eq!(active.model_id, "model-a");
    assert_eq!(active.pid, old.pid);
    assert_eq!(snapshot.switch, None, "the refused switch ended");
    assert!(pid_alive(old.pid), "the old engine was never touched");
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "a refused swap spawns nothing");
    }
    // Confirming a refused swap does nothing.
    manager.confirm_drain_swap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(manager.snapshot().active.unwrap().model_id, "model-a");

    manager.shutdown();
}
