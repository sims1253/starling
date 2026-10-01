//! Integration: model switches against the contract fixture — lease
//! draining across a rolling cutover and rapid repeated switching (plan
//! tests 3, 4).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use starling_dictation::engine::{EnginePhase, SwapMode};

const READY_TIMEOUT: Duration = Duration::from_secs(30);

fn two_model_setup(
    root: &std::path::Path,
) -> (
    starling_dictation::engine::CatalogEntry,
    starling_dictation::engine::CatalogEntry,
) {
    let bytes_a = model_bytes(10, 120_000);
    let bytes_b = model_bytes(20, 130_000);
    let addr = spawn_model_server(vec![
        ("a.gguf".to_string(), Arc::new(bytes_a.clone())),
        ("b.gguf".to_string(), Arc::new(bytes_b.clone())),
    ]);
    let entry_a = entry("model-a", "a.gguf", addr, &bytes_a);
    let entry_b = entry("model-b", "b.gguf", addr, &bytes_b);
    let models_dir = root.join("models");
    install(&models_dir, &entry_a, &bytes_a);
    install(&models_dir, &entry_b, &bytes_b);
    (entry_a, entry_b)
}

/// Plan test 3: switch A→B while holding a lease on A. After cutover
/// new leases point at B; A stays alive while the lease is held;
/// dropping the lease stops A; exactly one owned sidecar remains.
#[test]
fn switch_drains_the_old_engine_when_its_last_lease_drops() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let (entry_a, entry_b) = two_model_setup(root.path());
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let manager = starling_dictation::engine::EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, vec![entry_a, entry_b]),
        Some("model-a".to_string()),
    );
    let ready =
        wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready).expect("Ready on A");
    let old = ready.active.clone().expect("active A");
    assert_eq!(old.model_id, "model-a");

    // A take is open on A.
    let lease_a = manager.lease().expect("lease on A");

    manager.activate("model-b");
    let snapshot = wait_until(&manager, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready
            && s.active.as_ref().is_some_and(|a| a.model_id == "model-b")
            && s.switch.is_none()
    })
    .expect("cutting over to B keeps Ready with B active");
    assert_eq!(snapshot.switch, None, "the switch completed");

    // The report records from A to B as a rolling swap.
    let report = snapshot.last_switch.as_ref().expect("switch report");
    assert_eq!(report.from.as_deref(), Some("model-a"));
    assert_eq!(report.to, "model-b");
    assert_eq!(report.mode, SwapMode::Rolling);
    assert!(report.duration.as_nanos() > 0);

    // New leases point at B.
    let lease_b = manager.lease().expect("lease on B");
    assert_eq!(lease_b.model_id(), "model-b");
    assert_ne!(lease_b.endpoint(), lease_a.endpoint());
    let (status, _) = http_get(endpoint_port(lease_b.endpoint()), "/health").expect("health on B");
    assert_eq!(status, 200);

    // A's process is still alive while the lease is held...
    assert!(pid_alive(old.pid), "draining engine stays up for its lease");
    let (status, _) = http_get(endpoint_port(&old.endpoint), "/health").expect("A still answers");
    assert_eq!(status, 200);

    // ...and stops within 3 s of the lease dropping.
    drop(lease_a);
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while pid_alive(old.pid) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!pid_alive(old.pid), "A stops within 3 s of lease drop");

    // Exactly one owned sidecar remains: B.
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "exactly one engine process after the drain");
    }
    drop(lease_b);
    manager.shutdown();
}

/// Plan test 4: rapid switching A→B→A→B ends with exactly one live
/// owned sidecar, the right model, no leftover processes.
#[test]
fn rapid_switching_leaves_exactly_one_sidecar() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let (entry_a, entry_b) = two_model_setup(root.path());
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let manager = starling_dictation::engine::EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, vec![entry_a, entry_b]),
        Some("model-a".to_string()),
    );
    wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready).expect("Ready on A");

    // Fire the switches as fast as the app could.
    manager.activate("model-b");
    manager.activate("model-a");
    manager.activate("model-b");

    let snapshot = wait_until(&manager, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready
            && s.active.as_ref().is_some_and(|a| a.model_id == "model-b")
            && s.switch.is_none()
    })
    .expect("settles on B");
    let final_pid = snapshot.active.as_ref().expect("active B").pid;

    // Give any drain watcher a moment, then count processes: only the
    // final sidecar may remain.
    std::thread::sleep(Duration::from_millis(500));
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "rapid switching leaks no processes");
    }
    assert!(pid_alive(final_pid), "the final sidecar is the one counted");

    let lease = manager.lease().expect("lease serves B");
    assert_eq!(lease.model_id(), "model-b");
    drop(lease);
    manager.shutdown();

    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while pid_alive(final_pid) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!pid_alive(final_pid));
}

/// #363: "the next take uses the new model". A backend reload that is
/// still starting when the user activates another model is cancelled by
/// that activation; it must never cut over afterwards and bring the
/// previous model back.
#[cfg(unix)]
#[test]
fn a_slow_backend_reload_does_not_undo_a_later_activation() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    // The CPU reload of A starts slowly; B starts at normal speed.
    let engine_dir = stage_delayed_engine_dir(root.path(), &fixture, "a.gguf", 3);
    let (entry_a, entry_b) = two_model_setup(root.path());
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");
    let manager = starling_dictation::engine::EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, vec![entry_a, entry_b]),
        Some("model-a".to_string()),
    );
    // The initial start of A is slowed too.
    wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready).expect("Ready on A");

    manager.set_backend_override(Some(starling_dictation::engine::Backend::Cpu));
    wait_until(&manager, READY_TIMEOUT, |s| s.switch.is_some()).expect("the reload starts");
    manager.activate("model-b");
    wait_until(&manager, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready
            && s.active.as_ref().is_some_and(|active| active.model_id == "model-b")
    })
    .expect("B becomes active");
    // Longer than the reload's delayed start.
    std::thread::sleep(Duration::from_secs(4));
    let final_state = manager.snapshot();
    manager.shutdown();
    assert_eq!(
        final_state.active.map(|active| active.model_id).as_deref(),
        Some("model-b"),
        "the superseded reload must not reactivate A"
    );
}

/// A crash of the old engine during a switch schedules a restart; the
/// switch completing first must win, and the restart must not revert it.
#[cfg(unix)]
#[test]
fn a_crash_restart_does_not_revert_a_completed_switch() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_delayed_engine_dir(root.path(), &fixture, "b.gguf", 2);
    let (entry_a, entry_b) = two_model_setup(root.path());
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");
    let mut config = config(&engine_dir, &models_dir, &state_dir, vec![entry_a, entry_b]);
    // Longer than B's delayed start, so B completes while the restart
    // of A is still pending.
    config.backoff_schedule = Some(vec![Duration::from_secs(4)]);
    let manager =
        starling_dictation::engine::EngineManager::start(config, Some("model-a".to_string()));
    let ready =
        wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready).expect("Ready on A");
    let old = ready.active.expect("active A");

    manager.activate("model-b");
    wait_until(&manager, READY_TIMEOUT, |s| s.switch.is_some()).expect("the switch starts");
    unsafe {
        assert_eq!(libc::kill(old.pid as i32, libc::SIGKILL), 0);
    }
    wait_until(&manager, READY_TIMEOUT, |s| {
        s.phase == EnginePhase::Ready
            && s.active.as_ref().is_some_and(|active| active.model_id == "model-b")
    })
    .expect("B becomes active");
    // Past A's restart backoff.
    std::thread::sleep(Duration::from_secs(5));
    let final_state = manager.snapshot();
    manager.shutdown();
    assert_eq!(
        final_state.active.map(|active| active.model_id).as_deref(),
        Some("model-b"),
        "the pending restart of A must not revert the switch to B"
    );
}
