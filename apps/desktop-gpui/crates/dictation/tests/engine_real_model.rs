//! Real-engine end-to-end run and switch measurement (#362, #363).
//!
//! Ignored by default: it downloads real catalog models (hundreds of MB to
//! GB) from Hugging Face and runs a real `starling-serve`. Run it on the
//! machine to be measured (the #348 reference notebook for the #363
//! acceptance numbers):
//!
//! ```bash
//! scripts/stage-engines.sh --cpu <build>/starling-serve [--vulkan <build-vk>/starling-serve] --out /tmp/engines
//! STARLING_E2E_ENGINE_DIR=/tmp/engines \
//! STARLING_E2E_MODELS_DIR=$HOME/.local/share/starling-gpui/models \
//! STARLING_E2E_SWITCH="parakeet-v3-q4km-s16,moss-2b-q4e8" \
//!   cargo test -p starling-dictation --test engine_real_model -- --ignored --nocapture
//! ```
//!
//! The first id is activated from nothing and must transcribe the fixture
//! WAV; each further id is a runtime switch while a lease on the previous
//! model is held (the in-flight take), which must still transcribe on the
//! old model after the cutover. Every switch prints its mode, duration and
//! peak summed engine RSS — the numbers #363 asks for.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_dictation::client::StarlingClient;
use starling_dictation::engine::{
    default_catalog, EngineConfig, EngineManager, EnginePhase, EngineSnapshot,
};

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn wait_for(
    manager: &EngineManager,
    what: &str,
    timeout: Duration,
    done: impl Fn(&EngineSnapshot) -> bool,
) -> EngineSnapshot {
    let deadline = Instant::now() + timeout;
    let mut last_generation = u64::MAX;
    loop {
        let snapshot = manager.snapshot();
        if manager.generation() != last_generation {
            last_generation = manager.generation();
            eprintln!(
                "  [{what}] phase={:?} switch={:?} error={:?}",
                snapshot.phase,
                snapshot.switch.as_ref().map(|s| &s.stage),
                snapshot.last_error
            );
        }
        if done(&snapshot) {
            return snapshot;
        }
        if let EnginePhase::Failed(failure) = &snapshot.phase {
            panic!("{what}: engine failed: {failure}");
        }
        assert!(Instant::now() < deadline, "{what}: timed out");
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn transcribe(endpoint: &str, slug: &str, wav: &Arc<Vec<u8>>) -> String {
    let client = StarlingClient::new(endpoint, slug)
        .and_then(|client| client.with_timeout_ms(600_000))
        .expect("client");
    client
        .transcribe(wav.clone(), &format!("e2e-{}", uuid::Uuid::new_v4()))
        .expect("transcription")
        .text
}

#[test]
#[ignore = "downloads real models and runs a real engine; see module docs"]
fn real_engine_activates_transcribes_and_switches() {
    let engine_dir = env_path("STARLING_E2E_ENGINE_DIR").expect("set STARLING_E2E_ENGINE_DIR");
    let models_dir = env_path("STARLING_E2E_MODELS_DIR").expect("set STARLING_E2E_MODELS_DIR");
    let ids: Vec<String> = std::env::var("STARLING_E2E_SWITCH")
        .unwrap_or_else(|_| "parakeet-v3-q4km-s16".to_string())
        .split(',')
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .collect();
    let state_dir = tempfile::tempdir().expect("state dir");
    let wav = Arc::new(
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../../tests/fixtures/2086-149220-0033.wav"),
        )
        .expect("fixture wav"),
    );

    let config = EngineConfig {
        engine_dir: Some(engine_dir),
        models_dir,
        state_dir: state_dir.path().to_path_buf(),
        catalog: default_catalog(),
        backend_override: None,
        icd_dirs: None,
        available_memory_override: None,
        backoff_schedule: None,
    };
    let manager = EngineManager::start(config, None);
    let snapshot = wait_for(&manager, "select", Duration::from_secs(60), |s| {
        s.phase == EnginePhase::NoModel
    });
    eprintln!("backend: {:?}; notices: {:?}", snapshot.backend, snapshot.notices);

    let mut held = None;
    for (index, id) in ids.iter().enumerate() {
        let started = Instant::now();
        manager.activate(id);
        let snapshot = wait_for(&manager, id, Duration::from_secs(3600), |s| {
            s.phase == EnginePhase::Ready
                && s.switch.is_none()
                && s.active.as_ref().is_some_and(|active| &active.model_id == id)
        });
        eprintln!(
            "activated {id} in {:.1}s on device {:?}; report: {:?}",
            started.elapsed().as_secs_f64(),
            snapshot.active.as_ref().and_then(|a| a.device.clone()),
            snapshot.last_switch
        );

        // The take opened before this switch finishes on its own model.
        if let Some((lease, old_id)) = held.take() {
            let lease: starling_dictation::engine::EngineLease = lease;
            let text = transcribe(lease.endpoint(), lease.slug(), &wav);
            eprintln!("held take on {old_id} after cutover: {text:?}");
            assert!(!text.trim().is_empty(), "the old model must still serve the held take");
            drop(lease);
        }

        let lease = manager.lease().expect("lease on the active engine");
        let text = transcribe(lease.endpoint(), lease.slug(), &wav);
        eprintln!("{id} transcript: {text:?}");
        assert!(!text.trim().is_empty(), "{id} produced no text");
        if index + 1 < ids.len() {
            held = Some((lease, id.clone()));
        }
    }

    manager.shutdown();
}
