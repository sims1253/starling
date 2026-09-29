//! Integration: first-run lifecycle with the contract fixture — no
//! model, download+verify+activate, bad downloads, crash restart, and
//! shutdown hygiene (plan tests 1, 2, 5, 8).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use starling_dictation::engine::{EnginePhase, InstallState};

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const SHORT_TIMEOUT: Duration = Duration::from_secs(10);

/// Plan test 1: start with no model → NoModel; activate → download →
/// verify → Ready; `/health` on the lease endpoint answers.
#[test]
fn no_model_then_download_activate_ready() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(1, 150_000);
    let addr = spawn_model_server(vec![("a.gguf".to_string(), Arc::new(bytes.clone()))]);
    let entry_a = entry("model-a", "a.gguf", addr, &bytes);

    let manager = starling_dictation::engine::EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, vec![entry_a.clone()]),
        None,
    );

    let snapshot = wait_until(&manager, SHORT_TIMEOUT, |s| s.phase == EnginePhase::NoModel)
        .expect("reaches NoModel with no persisted model");
    assert!(snapshot.active.is_none());
    assert_eq!(install_of(&snapshot, "model-a"), InstallState::NotInstalled);
    assert!(snapshot.backend.is_some(), "backend selection ran");
    assert_eq!(
        snapshot.backend.unwrap().backend,
        starling_dictation::engine::Backend::Cpu
    );

    manager.activate("model-a");
    let snapshot = wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("activating downloads, verifies, loads, warms");
    let active = snapshot.active.clone().expect("active engine");
    assert_eq!(active.model_id, "model-a");
    assert!(active.owned);
    assert_eq!(install_of(&snapshot, "model-a"), InstallState::Installed);

    // The model file landed with its marker.
    assert!(models_dir.join(&entry_a.file_name).is_file());
    assert!(models_dir
        .join(format!("{}.verified", entry_a.file_name))
        .is_file());

    // The lease answers health on its endpoint.
    let lease = manager.lease().expect("lease on a ready engine");
    assert_eq!(lease.model_id(), "model-a");
    assert_eq!(lease.slug(), "parakeet");
    assert_eq!(lease.provenance(), "engine:model-a");
    let port = endpoint_port(lease.endpoint());
    let (status, body) = http_get(port, "/health").expect("health on lease endpoint");
    assert_eq!(status, 200);
    assert!(
        body.contains("\"model\":\"parakeet\""),
        "health body: {body}"
    );
    assert!(body.contains("\"warm\":true"), "health body: {body}");
    drop(lease);

    // Only this test's engine process exists.
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "exactly one sidecar for one model");
    }
    manager.shutdown();
}

/// Plan test 2: checksum-mismatch download → Failed install, no active
/// change (and no stray files).
#[test]
fn checksum_mismatch_download_fails_the_install_only() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(2, 80_000);
    let addr = spawn_model_server(vec![("bad.gguf".to_string(), Arc::new(bytes))]);
    let entry_bad = entry_with_wrong_digest("model-bad", "bad.gguf", addr, &model_bytes(3, 80_000));

    let manager = starling_dictation::engine::EngineManager::start(
        config(
            &engine_dir,
            &models_dir,
            &state_dir,
            vec![entry_bad.clone()],
        ),
        None,
    );
    wait_until(&manager, SHORT_TIMEOUT, |s| s.phase == EnginePhase::NoModel)
        .expect("NoModel before any download");

    manager.download("model-bad");
    let snapshot = wait_until(&manager, SHORT_TIMEOUT, |s| {
        matches!(install_of(s, "model-bad"), InstallState::Failed(_))
    })
    .expect("mismatching download lands in Failed");
    match install_of(&snapshot, "model-bad") {
        InstallState::Failed(message) => assert!(message.contains("checksum"), "{message}"),
        other => panic!("expected Failed install, got {other:?}"),
    }

    // Nothing was installed, nothing is active, and the failed attempt
    // left neither the final file nor the partial.
    assert!(snapshot.active.is_none());
    assert_eq!(snapshot.phase, EnginePhase::NoModel);
    assert!(!models_dir.join(&entry_bad.file_name).exists());
    assert!(!models_dir
        .join(format!("{}.part", entry_bad.file_name))
        .exists());
    assert!(manager.lease().is_none());

    manager.shutdown();
}

/// Plan test 5: kill -9 the active sidecar → Restarting → Ready on a
/// new port.
#[test]
fn killed_sidecar_restarts_on_a_new_port() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(4, 120_000);
    let addr = spawn_model_server(vec![("c.gguf".to_string(), Arc::new(bytes.clone()))]);
    let entry_c = entry("model-c", "c.gguf", addr, &bytes);
    install(&models_dir, &entry_c, &bytes);

    let manager = starling_dictation::engine::EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, vec![entry_c]),
        Some("model-c".to_string()),
    );
    let ready = wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("initial Ready");
    let old = ready.active.clone().expect("active engine");
    let first_port = endpoint_port(&old.endpoint);

    // SIGKILL the sidecar.
    #[cfg(unix)]
    unsafe {
        assert_eq!(libc::kill(old.pid as i32, libc::SIGKILL), 0);
    }

    let snapshot = wait_until(&manager, Duration::from_secs(8), |s| {
        matches!(s.phase, EnginePhase::Restarting { .. })
    })
    .expect("crash surfaces as Restarting");
    if let EnginePhase::Restarting { attempt, .. } = &snapshot.phase {
        assert_eq!(*attempt, 1, "first crash backs off once");
    }

    let ready_again = wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("restarts to Ready");
    let new = ready_again.active.expect("active engine after restart");
    assert_ne!(new.pid, old.pid, "restart gets a new process");
    assert_ne!(
        endpoint_port(&new.endpoint),
        first_port,
        "restart gets a new port"
    );
    assert_eq!(new.model_id, "model-c");

    let lease = manager.lease().expect("lease works after restart");
    let (status, _) = http_get(endpoint_port(lease.endpoint()), "/health").expect("health");
    assert_eq!(status, 200);
    drop(lease);
    manager.shutdown();
}

/// Plan test 8: `shutdown()` leaves no child processes; idempotent.
#[test]
fn shutdown_leaves_no_child_processes() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(5, 90_000);
    let addr = spawn_model_server(vec![("d.gguf".to_string(), Arc::new(bytes.clone()))]);
    let entry_d = entry("model-d", "d.gguf", addr, &bytes);
    install(&models_dir, &entry_d, &bytes);

    let manager = starling_dictation::engine::EngineManager::start(
        config(&engine_dir, &models_dir, &state_dir, vec![entry_d]),
        Some("model-d".to_string()),
    );
    let ready = wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("Ready before shutdown");
    let pid = ready.active.expect("active").pid;

    manager.shutdown();
    manager.shutdown(); // idempotent

    assert!(!pid_alive(pid), "the sidecar must be gone after shutdown");
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 0, "no engine processes remain after shutdown");
    }
    // The registry was ours and is released.
    assert!(!state_dir.join("sidecar.json").exists());
}
