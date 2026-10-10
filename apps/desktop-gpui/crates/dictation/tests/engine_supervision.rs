//! Integration: crash-loop supervision against a fake engine that
//! passes `--version` probing but dies when spawned to serve (plan
//! test 6).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;

const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// An engine whose `starling-serve-cpu` is a shell script: it answers
/// `--version` like the real server (so probing and selection pass) but
/// exits 1 whenever it is spawned to serve.
fn stage_dying_engine(root: &std::path::Path) -> std::path::PathBuf {
    let dir = root.join("engines");
    std::fs::create_dir_all(&dir).expect("create engines dir");
    let engine = dir.join("starling-serve-cpu");
    // The fake answers with the CURRENTLY expected ABI (bumping
    // EXPECTED_ENGINE_ABI must not break this success-path fixture) and an
    // older version string, so probing passes and serving dies.
    write_executable(
        &engine,
        &format!(
            "#!/bin/sh\ncase \"$1\" in\n  --version)\n    printf 'starling-serve 0.1.0\\nabi-version: {}\\nbackend: cpu\\nsupported-models: parakeet s1\\n'\n    exit 0\n    ;;\n  *) exit 1 ;;\nesac\n",
            starling_dictation::engine::EXPECTED_ENGINE_ABI
        ),
    );
    let sha = starling_dictation::engine::bundle::sha256_file(&engine).expect("hash");
    std::fs::write(
        dir.join("SHA256SUMS.txt"),
        format!("{sha}  starling-serve-cpu\n"),
    )
    .expect("sums");
    std::fs::write(
        dir.join("engines.json"),
        format!(
            r#"{{"version":"0.1.0","abi":{},"engines":[{{"backend":"cpu","file":"starling-serve-cpu"}}]}}"#,
            starling_dictation::engine::EXPECTED_ENGINE_ABI
        ),
    )
    .expect("manifest");
    dir
}

/// Plan test 6: the engine passes probing, dies on every serve spawn;
/// after 5 crashes within 5 minutes the manager gives up with
/// `Failed(CrashLoop)` until `retry()`.
#[test]
fn crash_loop_stops_after_five_crashes() {
    let root = tempfile::tempdir().expect("tempdir");
    let engine_dir = stage_dying_engine(root.path());
    let models_dir = root.path().join("models");
    let state_dir = root.path().join("state");

    let bytes = model_bytes(7, 60_000);
    let addr = spawn_model_server(vec![("e.gguf".to_string(), Arc::new(bytes.clone()))]);
    let entry_e = entry("model-e", "e.gguf", addr, &bytes);
    install(&models_dir, &entry_e, &bytes);

    let mut config = config(&engine_dir, &models_dir, &state_dir, vec![entry_e]);
    // Test-only knob: retry fast so five crashes fit in seconds.
    config.backoff_schedule = Some(vec![Duration::from_millis(50); 8]);

    let manager =
        starling_dictation::engine::EngineManager::start(config, Some("model-e".to_string()));

    let snapshot = wait_until(&manager, READY_TIMEOUT, |s| {
        matches!(
            s.phase,
            starling_dictation::engine::EnginePhase::Failed(
                starling_dictation::engine::EngineFailure::CrashLoop { .. }
            )
        )
    })
    .expect("five crashes trip the crash loop");
    if let starling_dictation::engine::EnginePhase::Failed(
        starling_dictation::engine::EngineFailure::CrashLoop { last_stderr },
    ) = &snapshot.phase
    {
        // The sentence names what to do.
        let sentence = starling_dictation::engine::EngineFailure::CrashLoop {
            last_stderr: last_stderr.clone(),
        }
        .to_string();
        assert!(sentence.contains("Choose Retry"), "{sentence}");
    }
    assert!(
        snapshot.active.is_none(),
        "the crash loop leaves no active engine"
    );
    assert!(manager.lease().is_none());

    manager.shutdown();
}
