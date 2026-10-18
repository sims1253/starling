//! Integration: the per-model speed check (#475) against the contract
//! fixture engine. Covers what a fake engine can actually prove of the
//! issue's acceptance list:
//!
//! - both probes run offline against a sidecar the check owns, and the
//!   derived figures come back;
//! - the transcript is discarded, nothing is persisted, and the engine
//!   the check started is gone afterwards;
//! - the live engine manager is not disturbed while a check runs;
//! - an already-cancelled check starts nothing at all.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use starling_dictation::engine::EnginePhase;
use starling_dictation::speedcheck::{self, ProbeKind, SpeedCheckTarget};

const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// Everything a check needs about one installed model, pointed at the
/// test's own staged engine.
fn target(
    engine_dir: &Path,
    models_dir: &Path,
    state_dir: &Path,
    id: &str,
    file: &str,
) -> SpeedCheckTarget {
    SpeedCheckTarget {
        entry: starling_dictation::engine::CatalogEntry::new(
            id,
            id,
            "parakeet",
            &format!("http://127.0.0.1/{file}"),
            1,
            "00",
            false,
            "integration test model",
        ),
        engine_path: engine_dir.join("starling-serve-cpu"),
        gguf_path: models_dir.join(file),
        log_path: state_dir.join("logs/engine-speedcheck.log"),
        // The fixture's model file is a few KiB stand-in that never
        // becomes resident; forcing the no-reading path keeps these
        // tests about plumbing, not about the test machine's free RAM.
        // Admission itself is covered by the unit tests and the host's.
        available_memory_override: Some(None),
    }
}

/// The checks every measurement must satisfy, whatever the numbers: the
/// audio that was sent is the audio that was timed, the derived figures
/// agree with each other, and the speech probe is the representative one.
fn assert_sane(check: &starling_dictation::speedcheck::SpeedCheck) {
    let expected_seconds = speedcheck::probe_audio(check.kind)
        .expect("the probe's own audio")
        .1;
    assert_eq!(check.audio_seconds, expected_seconds, "{check:?}");
    assert!(check.transcribe_ms > 0.0, "{check:?}");
    let rtf = check.rtf.expect("a timed probe measures a realtime factor");
    assert!(rtf > 0.0, "{check:?}");
    let per_minute = check
        .seconds_per_audio_minute
        .expect("a realtime factor answers the per-minute question");
    // The two figures are two spellings of the same measurement, so
    // they stay consistent rather than each being merely positive.
    assert!(
        (per_minute - 60.0 / rtf).abs() < 1e-9 * 60.0 / rtf,
        "{check:?}: {per_minute} vs 60/{rtf}"
    );
}

/// Waits until at most `want` processes carry `needle` in their command
/// line; `None` off Linux, where /proc is not the process table.
fn wait_engine_count(needle: &Path, want: usize) -> bool {
    let Some(mut count) = count_engine_processes(needle.to_str().unwrap()) else {
        // Off Linux there is no /proc to read: nothing to wait for.
        return true;
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while count > want && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
        // A re-count that fails keeps the previous count rather than
        // reading as zero: ending the wait early would let this pass
        // while the engine process is still alive.
        if let Some(fresh) = count_engine_processes(needle.to_str().unwrap()) {
            count = fresh;
        }
    }
    count <= want
}

#[test]
fn both_probes_measure_offline_and_leave_no_engine_behind() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");
    std::fs::create_dir_all(&models_dir).expect("models dir");

    // The fixture accepts any existing file as its GGUF.
    std::fs::write(models_dir.join("probe.gguf"), model_bytes(3, 4_096)).expect("model file");
    let target = target(&engine_dir, &models_dir, &state_dir, "probe", "probe.gguf");

    let results = speedcheck::run_speed_check(&target, speedcheck::ALL_PROBES, &|_| {}, None)
        .expect("the check runs");

    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(results[0].kind, ProbeKind::Smoke);
    assert_eq!(results[1].kind, ProbeKind::Speech);
    for check in &results {
        assert_sane(check);
        assert_eq!(check.model_id, "probe");
        assert!(
            !check.backend.is_empty(),
            "the backend that ran it is named: {check:?}"
        );
    }
    // The speech figure is the representative one; the smoke figure is
    // labelled as the upper bound it is.
    assert!(
        results[0].kind.is_upper_bound(),
        "a synthetic tone is faster than speech by construction"
    );
    assert!(!results[1].kind.is_upper_bound());

    // The sidecar the check started is gone, and nothing else either.
    assert!(
        wait_engine_count(&engine_dir, 0),
        "a check leaves no engine process behind"
    );
}

#[test]
fn the_check_does_not_disturb_the_live_engine_manager() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(5, 4_096);
    let addr = spawn_model_server(vec![("live.gguf".to_string(), Arc::new(bytes.clone()))]);
    let live = entry("live", "live.gguf", addr, &bytes);
    let mut live_config = config(&engine_dir, &models_dir, &state_dir, vec![live.clone()]);
    // Same stand-in reasoning as the probe target below: the live
    // fixture model never becomes resident, so its activation must not
    // depend on the test machine's real free RAM either.
    live_config.available_memory_override = Some(None);
    let manager =
        starling_dictation::engine::EngineManager::start(live_config, None);
    manager.activate("live");
    let ready = wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("the real engine warms");
    let live_engine = ready.active.expect("active engine");
    assert_eq!(live_engine.model_id, "live");
    let live_endpoint = live_engine.endpoint.clone();
    let live_pid = live_engine.pid;
    // One engine process: the live one, before any check. The count is
    // /proc-only, so it is guarded rather than asserted — a `Some(1)`
    // cannot be answered off Linux.
    if let Some(count) = count_engine_processes(engine_dir.to_str().unwrap()) {
        assert_eq!(count, 1, "only the live engine runs before a check");
    }

    // An installed model of another id, checked while the live engine
    // keeps serving.
    std::fs::write(models_dir.join("probe.gguf"), model_bytes(9, 4_096)).expect("probe model file");
    let target = target(&engine_dir, &models_dir, &state_dir, "probe", "probe.gguf");
    let results = speedcheck::run_speed_check(&target, speedcheck::ALL_PROBES, &|_| {}, None)
        .expect("the check runs while the live engine serves");
    for check in &results {
        assert_sane(check);
        assert_eq!(check.model_id, "probe");
    }

    // The live engine is untouched: same process, same endpoint, still
    // ready. Only the check's own sidecar ever came and went.
    assert!(pid_alive(live_pid), "the live engine keeps running");
    let after = wait_until(&manager, READY_TIMEOUT, |s| s.phase == EnginePhase::Ready)
        .expect("the live engine is still ready");
    let active = after.active.expect("active engine");
    assert_eq!(active.model_id, "live");
    assert_eq!(active.endpoint, live_endpoint, "the endpoint never moved");
    assert_eq!(active.pid, live_pid, "the live process was never replaced");
    assert!(
        wait_engine_count(&engine_dir, 1),
        "only the live engine is left"
    );

    manager.shutdown();
    assert!(
        wait_engine_count(&engine_dir, 0),
        "the live engine stops on shutdown"
    );
    assert!(!pid_alive(live_pid));
}

/// A completed check must not outlive its own cancel watcher. The
/// watcher polls the caller's token, so a token nobody cancels would
/// keep it polling for the life of the process — and joining it on the
/// way out would hang the caller forever, not merely leak a thread.
/// With the stop flag missing this test hangs rather than fails, which
/// is precisely the regression it is for.
#[test]
fn a_completed_check_returns_despite_a_token_nobody_cancels() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");
    std::fs::create_dir_all(&models_dir).expect("models dir");
    std::fs::write(models_dir.join("probe.gguf"), model_bytes(11, 4_096)).expect("model file");
    let target = target(&engine_dir, &models_dir, &state_dir, "probe", "probe.gguf");

    let token = starling_dictation::client::CancelToken::new();
    let results = starling_dictation::speedcheck::run_speed_check(
        &target,
        speedcheck::ALL_PROBES,
        &|_| {},
        Some(&token),
    )
    .expect("the check finishes with a live, never-cancelled token");
    assert_eq!(results.len(), 2, "{results:?}");
    assert!(
        !token.is_cancelled(),
        "the run must not cancel the caller's token"
    );

    // What this pins is that the run *returns* rather than hanging on the
    // watcher's join, and that no engine process is left behind. It does
    // not count threads — the watcher cannot be observed from here.
    assert!(
        wait_engine_count(&engine_dir, 0),
        "a check leaves no engine process behind"
    );
}

#[test]
fn an_already_cancelled_check_starts_nothing() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_engine_dir(root.path(), &fixture);
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");
    std::fs::create_dir_all(&models_dir).expect("models dir");
    std::fs::write(models_dir.join("probe.gguf"), model_bytes(7, 4_096)).expect("model file");
    let target = target(&engine_dir, &models_dir, &state_dir, "probe", "probe.gguf");

    let cancelled = starling_dictation::client::CancelToken::new();
    cancelled.cancel();
    let error =
        speedcheck::run_speed_check(&target, speedcheck::ALL_PROBES, &|_| {}, Some(&cancelled))
            .expect_err("a cancelled check does not run");
    assert_eq!(
        error,
        starling_dictation::speedcheck::SpeedCheckError::Cancelled
    );
    // Nothing was started, so there is nothing to stop either.
    assert!(
        wait_engine_count(&engine_dir, 0),
        "a cancelled check spawns no engine"
    );
}
