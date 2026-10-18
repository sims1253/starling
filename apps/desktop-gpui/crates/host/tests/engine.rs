//! #220: the supervised engine attaches to the runtime host, not the
//! renderer. The host owns the bundled-engine supervisor; jobs run on
//! it over the real IPC transport; a renderer that dies mid-job costs
//! neither the job nor the engine; the host's shutdown stops the
//! engine; a host that turns out to be a client never starts one.
//!
//! The engine is the real `starling-serve` contract fixture (the same
//! deterministic server the dictation crate's engine lifecycle suites
//! drive), found via `STARLING_CONTRACT_BIN` or the repo build dir. Tests
//! that need it skip without it — unless `STARLING_REQUIRE_CONTRACT_FIXTURE`
//! is set (CI), which turns a missing fixture into a failure.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_dictation::engine::bundle::sha256_file;
use starling_dictation::engine::{CatalogEntry, EngineConfig, EnginePhase, EXPECTED_ENGINE_ABI};
use starling_dictation::settings::{EngineMode, Settings};
use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
use starling_runtime::protocol::Command;
use starling_runtime::provider::FakeProvider;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::{EventWire, HostClient};
use starling_runtime_host::engine::{
    self, EngineChoice, EngineIntent, EngineReply, EngineRequest, EngineStatus, SpeedCheckStatus,
};
use starling_runtime_host::{serve, HostConfig, HostError};

#[path = "common/fake_engine.rs"]
mod fake_engine;

const MODEL_ID: &str = "fixture-model";

fn fixture() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("STARLING_CONTRACT_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    // The repo build dir serves portable runs too: cargo names the
    // fixture for the target OS (`.exe` on Windows), so probe the
    // platform's name instead of silently skipping where the fixture
    // exists next to it.
    let build = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../build/native-cpu");
    let name = if cfg!(windows) {
        "starling-serve-contract-fixture.exe"
    } else {
        "starling-serve-contract-fixture"
    };
    let default = build.join(name);
    if default.is_file() {
        return Some(default);
    }
    if std::env::var_os("STARLING_REQUIRE_CONTRACT_FIXTURE").is_some() {
        panic!(
            "STARLING_REQUIRE_CONTRACT_FIXTURE is set but the contract fixture is missing \
             (STARLING_CONTRACT_BIN or {})",
            default.display()
        );
    }
    println!("skipping: contract fixture {} not found", default.display());
    None
}

/// An engine config over temp dirs: `engines/` staged with `engine`
/// (the fixture, or nothing), one installed catalog model, no Vulkan
/// ICDs, short restart backoff.
fn engine_config(root: &Path, engine: Option<&Path>) -> EngineConfig {
    engine_config_with_models(root, engine, &[MODEL_ID])
}

/// [`engine_config`] with an explicit catalog: every named model
/// installed, distinct bytes per model (a switch between them moves to
/// a genuinely different file).
fn engine_config_with_models(root: &Path, engine: Option<&Path>, models: &[&str]) -> EngineConfig {
    let engines = root.join("engines");
    std::fs::create_dir_all(&engines).unwrap();
    if let Some(fixture) = engine {
        let staged = engines.join("starling-serve-cpu");
        std::fs::copy(fixture, &staged).unwrap();
        let sha = sha256_file(&staged).unwrap();
        std::fs::write(
            engines.join("SHA256SUMS.txt"),
            format!("{sha}  starling-serve-cpu\n"),
        )
        .unwrap();
        // The manifest's abi derives from the app's expected ABI so a
        // future bump can't leave this fixture staging a bundle its own
        // engine refuses (#397 follow-up).
        std::fs::write(
            engines.join("engines.json"),
            format!(
                r#"{{"version":"0.1.0","abi":{EXPECTED_ENGINE_ABI},"engines":[{{"backend":"cpu","file":"starling-serve-cpu"}}]}}"#
            ),
        )
        .unwrap();
    }
    let models_dir = root.join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    let mut catalog = Vec::new();
    for (index, id) in models.iter().enumerate() {
        let file_name = format!("{id}.gguf");
        let model_file = models_dir.join(&file_name);
        std::fs::write(&model_file, vec![(7 + index * 13) as u8; 4096]).unwrap();
        let sha = sha256_file(&model_file).unwrap();
        std::fs::write(models_dir.join(format!("{file_name}.verified")), &sha).unwrap();
        catalog.push(CatalogEntry::new(
            id,
            "Fixture",
            "parakeet",
            &format!("http://127.0.0.1:9/{file_name}"),
            4096,
            &sha,
            false,
            "host engine test model",
        ));
    }
    EngineConfig {
        engine_dir: Some(engines),
        models_dir,
        state_dir: root.join("engine-state"),
        catalog,
        backend_override: None,
        icd_dirs: Some(Vec::new()),
        available_memory_override: Some(None),
        backoff_schedule: Some(vec![Duration::from_millis(100)]),
    }
}

fn host_config(root: &Path, engine: EngineChoice) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints")).with_engine(engine);
    config.runtime = config
        .runtime
        .with_capture_source(FakeCaptureSource::new(vec![
            FakeTakeScript::clean(),
            FakeTakeScript::clean(),
        ]))
        // The engine replaces this; were it ever used, jobs would fail
        // `script_exhausted` and the assertions below would say so.
        .with_provider(FakeProvider::new(vec![]))
        .with_capture_store(Arc::new(
            V2CaptureStore::open(root).expect("v2 store opens"),
        ))
        .with_capture_config(CaptureConfig {
            journals_dir: root.join("journals"),
            poll_interval: Duration::from_millis(10),
            ..CaptureConfig::default()
        });
    config
}

fn connect(path: &Path) -> HostClient {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match HostClient::connect(path) {
            Ok(client) => return client,
            Err(err) if Instant::now() >= deadline => panic!("no host: {err}"),
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

fn until(client: &HostClient, label: &str, predicate: impl Fn(&EventWire) -> bool) -> EventWire {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut seen = Vec::new();
    loop {
        match client.recv_event_timeout(Duration::from_millis(20)) {
            Ok(event) if predicate(&event) => return event,
            Ok(event) => seen.push(format!("{} {}", event.type_name(), event.payload())),
            Err(starling_runtime::channel::RecvError::Timeout) => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {label}; saw {seen:?}"
                )
            }
            Err(other) => panic!("event stream error: {other:?} waiting for {label}; saw {seen:?}"),
        }
    }
}

/// Freezes a route, records a take and submits a job for it.
fn record_and_submit(client: &HostClient, take: &str, job: &str) {
    record_and_submit_after(client, take, job, || {});
}

/// [`record_and_submit`], running `before_submit` between the take's
/// stop and the job's submit.
fn record_and_submit_after(
    client: &HostClient,
    take: &str,
    job: &str,
    before_submit: impl FnOnce(),
) {
    client
        .send(
            Some("ctx"),
            Command::ContextSnapshot {
                source: "vscode".into(),
            },
        )
        .expect("snapshot accepted");
    until(client, "context.targetSnapshot", |e| {
        e.type_name() == "context.targetSnapshot"
    });
    client
        .send(
            Some("ctx"),
            Command::ModeSet {
                mode: "code-guidance".into(),
                source: starling_runtime::protocol::Manual,
            },
        )
        .expect("mode accepted");
    until(client, "mode.decision", |e| {
        e.type_name() == "mode.decision"
    });
    client
        .send(
            Some(take),
            Command::CaptureStart {
                policy: "push-to-talk".into(),
            },
        )
        .expect("start accepted");
    until(client, "capture.progress", |e| {
        e.type_name() == "capture.progress"
    });
    client
        .send(Some(take), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    until(client, "capture.stopped", |e| {
        e.type_name() == "capture.stopped"
    });
    before_submit();
    client
        .send(
            Some(job),
            Command::JobsSubmit {
                capture_ref: take.into(),
                route: "local-default".into(),
                budget: "standard".into(),
            },
        )
        .expect("submit accepted");
}

fn job_outcome(client: &HostClient, job: &str) -> EventWire {
    until(client, "the job's outcome", |e| {
        (e.type_name() == "jobs.completed" || e.type_name() == "jobs.failed")
            && e.corr() == Some(job)
    })
}

/// Whether anything still accepts TCP connections at the engine's
/// loopback endpoint (`http://127.0.0.1:<port>`).
fn endpoint_answers(endpoint: &str) -> bool {
    let address = endpoint.trim_start_matches("http://").trim_end_matches('/');
    std::net::TcpStream::connect_timeout(
        &address.parse().expect("loopback socket address"),
        Duration::from_millis(200),
    )
    .is_ok()
}

#[test]
fn jobs_run_on_the_host_owned_engine_which_outlives_its_renderer() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let engine = EngineChoice::Builtin {
        config: engine_config(root.path(), Some(&fixture)),
        active_model: Some(MODEL_ID.into()),
    };
    let mut host = serve(host_config(root.path(), engine)).expect("host serves");
    let manager = host
        .engine()
        .expect("builtin mode supervises an engine")
        .clone();

    // The host started the engine itself — no renderer exists yet.
    wait_ready(&manager);
    let active = manager.snapshot().active.expect("a ready engine is active");

    // An observer is connected (and greeted) first; a renderer then
    // submits a job and dies straight away — the fixture engine answers
    // fast, so the observer must already be listening.
    let peer = connect(host.socket_path());
    let _ = peer.snapshot().expect("observer greeted");
    let renderer = connect(host.socket_path());
    record_and_submit(&renderer, "take_e1", "job_e1");
    drop(renderer);

    let outcome = job_outcome(&peer, "job_e1");
    assert_eq!(
        outcome.type_name(),
        "jobs.completed",
        "{}",
        outcome.payload()
    );
    assert_eq!(outcome.payload()["backend"], format!("engine:{MODEL_ID}"));
    assert!(
        outcome.payload()["text"]
            .as_str()
            .unwrap_or_default()
            .contains("Keep auth"),
        "the fixture engine's transcript: {}",
        outcome.payload()
    );

    // The renderer's death never touched the engine: same process, still
    // serving — and a second job runs on it.
    let after = manager.snapshot();
    assert_eq!(after.phase, EnginePhase::Ready);
    assert_eq!(after.active.as_ref().map(|a| a.pid), Some(active.pid));
    record_and_submit(&peer, "take_e2", "job_e2");
    assert_eq!(job_outcome(&peer, "job_e2").type_name(), "jobs.completed");
    drop(peer);

    // Host shutdown stops the engine it owns.
    assert!(endpoint_answers(&active.endpoint));
    host.shutdown();
    let deadline = Instant::now() + Duration::from_secs(10);
    while endpoint_answers(&active.endpoint) {
        assert!(
            Instant::now() < deadline,
            "the engine at {} outlived its host",
            active.endpoint
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// An engine that cannot run fails jobs honestly and retryably — the
/// take is durable and the same job succeeds once the engine serves.
#[test]
fn an_unusable_engine_fails_jobs_retryably() {
    let root = tempfile::tempdir().unwrap();
    // No engine staged: the supervisor fails with NoBundledEngine.
    let engine = EngineChoice::Builtin {
        config: engine_config(root.path(), None),
        active_model: Some(MODEL_ID.into()),
    };
    let mut host = serve(host_config(root.path(), engine)).expect("host serves");
    let client = connect(host.socket_path());
    record_and_submit(&client, "take_u", "job_u");
    let outcome = job_outcome(&client, "job_u");
    assert_eq!(outcome.type_name(), "jobs.failed", "{}", outcome.payload());
    assert_eq!(outcome.payload()["reason"], "engine_unavailable");
    assert_eq!(outcome.payload()["retryable"], true);
    drop(client);
    host.shutdown();
}

/// A host that finds a live owner is a client: it must not start a
/// second engine supervisor beside the owner's.
#[test]
fn a_host_that_is_a_client_never_starts_an_engine() {
    let root = tempfile::tempdir().unwrap();
    let mut owner = serve(host_config(root.path(), EngineChoice::None)).expect("owner serves");
    let config = engine_config(root.path(), None);
    let state_dir = config.state_dir.clone();
    let second = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ));
    assert!(matches!(second, Err(HostError::OwnerLive { .. })));
    assert!(
        !state_dir.exists(),
        "the client host touched the engine state dir"
    );
    owner.shutdown();
}

/// The host's engine dies with no warning (SIGKILL: no drain, nothing
/// tells the host first). A job submitted right then hits the dead
/// engine; it must still complete — the provider retries once on the
/// engine the supervisor restarts.
#[cfg(unix)] // the abrupt kill is a signal
#[test]
fn a_job_survives_the_engine_dying_abruptly() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let engine = EngineChoice::Builtin {
        config: engine_config(root.path(), Some(&fixture)),
        active_model: Some(MODEL_ID.into()),
    };
    let mut host = serve(host_config(root.path(), engine)).expect("host serves");
    let manager = host.engine().expect("builtin mode").clone();
    wait_ready(&manager);
    let killed = manager.snapshot().active.unwrap();

    let client = connect(host.socket_path());
    record_and_submit_after(&client, "take_a", "job_a", || {
        // Re-read right before the kill: signal only the engine that is
        // still the active one (endpoint and pid), never a pid the
        // supervisor already replaced (or the OS recycled).
        let current = manager.snapshot().active.expect("engine still active");
        assert_eq!(
            (current.endpoint.as_str(), current.pid),
            (killed.endpoint.as_str(), killed.pid),
            "the sidecar was replaced before the kill"
        );
        // SAFETY: kill(2) on the sidecar's pid; no memory is involved.
        assert_eq!(unsafe { libc::kill(killed.pid as i32, libc::SIGKILL) }, 0);
        let deadline = Instant::now() + Duration::from_secs(5);
        while endpoint_answers(&killed.endpoint) {
            assert!(
                Instant::now() < deadline,
                "the killed sidecar still answers"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    });

    let outcome = job_outcome(&client, "job_a");
    assert_eq!(
        outcome.type_name(),
        "jobs.completed",
        "{}",
        outcome.payload()
    );
    assert_eq!(outcome.payload()["backend"], format!("engine:{MODEL_ID}"));
    let now = manager.snapshot().active.expect("an engine serves again");
    assert_ne!(
        now.pid, killed.pid,
        "the job ran on the replacement engine"
    );
    drop(client);
    host.shutdown();
}

fn wait_ready(manager: &starling_dictation::engine::EngineManager) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while manager.snapshot().phase != EnginePhase::Ready {
        assert!(
            Instant::now() < deadline,
            "engine never became ready: {:?}",
            manager.snapshot()
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// #220: the host follows the settings file while it runs. A manual
/// endpoint change is applied without a restart — the runtime's
/// provider is rebuilt against the new endpoint. Unit-level (the
/// watcher and the engine host alone; no server and no IPC: an
/// endpoint that validates is all the swap needs to observe).
#[test]
fn a_manual_endpoint_change_is_applied_without_a_restart() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("settings.json");

    let mut settings = Settings::default_settings();
    settings.engine.mode = EngineMode::Manual;
    settings.endpoint = "http://127.0.0.1:8181".into();
    settings.save(&path).unwrap();

    let mut runtime = starling_runtime::RuntimeConfig::default();
    let host = engine::attach(
        EngineChoice::Manual {
            endpoint: settings.endpoint.clone(),
            model: settings.model.clone(),
        },
        &mut runtime,
    )
    .expect("manual mode attaches");
    assert_eq!(host.label(), "manual:http://127.0.0.1:8181");

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watcher = engine::watch_settings(
        Arc::clone(&host),
        path.clone(),
        Duration::from_millis(20),
        Arc::clone(&stop),
    )
    .expect("watcher spawns");

    settings.endpoint = "http://127.0.0.1:9199".into();
    settings.save(&path).unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    while host.label() != "manual:http://127.0.0.1:9199" {
        assert!(
            Instant::now() < deadline,
            "the endpoint change was never picked up (still {})",
            host.label()
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    // A file that disappears is not a choice: the host keeps the
    // last-known endpoint rather than falling back to the load
    // defaults (a deletion racing the read must not flip the mode).
    std::fs::remove_file(&path).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        host.label(),
        "manual:http://127.0.0.1:9199",
        "a missing settings file must not change the engine"
    );

    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    watcher.join().expect("the watcher stops on its stop flag");
    host.shutdown();
}

/// #220: a torn settings write is a non-event, not a choice. The file
/// truncated mid-document or emptied — what a non-atomic writer looks
/// like between two identical reads — must not fall back to the load
/// defaults (which would replace a working manual provider with the
/// bundled-engine none); the host keeps its last-applied choice and
/// follows the next valid write.
#[test]
fn a_torn_settings_write_keeps_the_last_engine_choice() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("settings.json");

    let mut settings = Settings::default_settings();
    settings.engine.mode = EngineMode::Manual;
    settings.endpoint = "http://127.0.0.1:8181".into();
    settings.save(&path).unwrap();

    let mut runtime = starling_runtime::RuntimeConfig::default();
    let host = engine::attach(
        EngineChoice::Manual {
            endpoint: settings.endpoint.clone(),
            model: settings.model.clone(),
        },
        &mut runtime,
    )
    .expect("manual mode attaches");

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watcher = engine::watch_settings(
        Arc::clone(&host),
        path.clone(),
        Duration::from_millis(20),
        Arc::clone(&stop),
    )
    .expect("watcher spawns");

    // Truncated mid-document.
    std::fs::write(&path, r#"{"engine":{"mo"#).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        host.label(),
        "manual:http://127.0.0.1:8181",
        "a truncated settings file must not change the engine"
    );

    // Emptied — the same non-event.
    std::fs::write(&path, b"").unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        host.label(),
        "manual:http://127.0.0.1:8181",
        "an empty settings file must not change the engine"
    );

    // The next valid write is followed again.
    settings.endpoint = "http://127.0.0.1:9195".into();
    settings.save(&path).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while host.label() != "manual:http://127.0.0.1:9195" {
        assert!(
            Instant::now() < deadline,
            "the valid write after the torn ones was never picked up (still {})",
            host.label()
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    watcher.join().expect("the watcher stops on its stop flag");
    host.shutdown();
}

/// #220: a change between the startup load (which resolved the host's
/// initial engine choice) and the watcher's start is not frozen out: the
/// host reads the file once more before it accepts a connection — the
/// baseline its watcher compares with — and converges to it. The startup
/// choice here names the old endpoint while the file already names a new
/// one, exactly that gap.
#[test]
fn a_change_before_the_watcher_starts_is_still_applied() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("settings.json");

    // What the file said when the host's startup resolved its choice…
    let mut settings = Settings::default_settings();
    settings.engine.mode = EngineMode::Manual;
    settings.endpoint = "http://127.0.0.1:8181".into();
    // …and what it says by the time the host serves: the user moved the
    // manual endpoint while the host was starting.
    settings.endpoint = "http://127.0.0.1:9196".into();
    settings.save(&path).unwrap();

    let host_setup = host_config(
        root.path(),
        EngineChoice::Manual {
            endpoint: "http://127.0.0.1:8181".into(),
            model: "parakeet".into(),
        },
    )
    .with_settings_path(&path)
    .with_settings_poll(Duration::from_millis(20));
    let mut host = serve(host_setup).expect("host serves");
    assert_eq!(
        host.engine_label(),
        "manual:http://127.0.0.1:9196",
        "the file's newer choice serves before any app connects"
    );
    host.shutdown();
}

/// #220: the host follows a builtin activeModel change while it runs.
/// The watcher activates the new model on the host's own manager (the
/// host-side twin of the app's Activate action), and a job submitted
/// after the switch runs on the model the file names.
#[test]
fn an_active_model_change_switches_the_host_s_engine() {
    let Some(fixture) = fixture() else { return };
    let other = "fixture-model-2";
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_models(root.path(), Some(&fixture), &[MODEL_ID, other]);
    let path = root.path().join("settings.json");
    let mut settings = Settings::default_settings();
    settings.engine.mode = EngineMode::Builtin;
    settings.engine.active_model = Some(MODEL_ID.into());
    settings.save(&path).unwrap();

    let host_setup = host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    )
    .with_settings_path(&path)
    .with_settings_poll(Duration::from_millis(20));
    let mut host = serve(host_setup).expect("host serves");
    let manager = host.engine().expect("builtin mode supervises an engine");
    wait_ready(&manager);
    assert_eq!(
        manager
            .snapshot()
            .active
            .expect("an engine is active")
            .model_id,
        MODEL_ID
    );

    settings.engine.active_model = Some(other.into());
    settings.save(&path).unwrap();

    // #363's switch protocol runs on the host's own manager: the
    // incoming model loads on a second sidecar, the endpoint cuts
    // over, the old engine drains.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let snapshot = manager.snapshot();
        if snapshot.phase == EnginePhase::Ready
            && snapshot
                .active
                .as_ref()
                .is_some_and(|active| active.model_id == other)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the model switch never landed: {:?}",
            snapshot
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    let client = connect(host.socket_path());
    record_and_submit(&client, "take_s", "job_s");
    let outcome = job_outcome(&client, "job_s");
    assert_eq!(
        outcome.type_name(),
        "jobs.completed",
        "{}",
        outcome.payload()
    );
    assert_eq!(outcome.payload()["backend"], format!("engine:{other}"));
    drop(client);
    host.shutdown();
}

/// #220: the host follows a mode change while it runs. builtin→manual
/// stops routing jobs to the engine (a submitted job goes to the manual
/// endpoint) and stops the engine the host supervised — the host-side
/// twin of the app's `apply_engine_mode_change`.
#[test]
fn a_mode_switch_to_manual_stops_using_the_engine() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config(root.path(), Some(&fixture));
    let path = root.path().join("settings.json");
    let mut settings = Settings::default_settings();
    settings.engine.mode = EngineMode::Builtin;
    settings.engine.active_model = Some(MODEL_ID.into());
    settings.save(&path).unwrap();

    let host_setup = host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    )
    .with_settings_path(&path)
    .with_settings_poll(Duration::from_millis(20));
    let mut host = serve(host_setup).expect("host serves");
    let manager = host.engine().expect("builtin mode supervises an engine");
    wait_ready(&manager);
    let active = manager.snapshot().active.expect("a ready engine is active");
    assert!(endpoint_answers(&active.endpoint), "the engine serves");

    // The file moves to the user's own server — a port that nothing
    // answers on, so the job's failure reason is the observable
    // routing proof.
    let dead_port = {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        probe.local_addr().unwrap().port()
    };
    settings.engine.mode = EngineMode::Manual;
    settings.endpoint = format!("http://127.0.0.1:{dead_port}/manual");
    settings.save(&path).unwrap();

    // The mode switch takes the engine away: no manager, no live
    // sidecar at the old endpoint.
    let deadline = Instant::now() + Duration::from_secs(10);
    while host.engine().is_some() || endpoint_answers(&active.endpoint) {
        assert!(
            Instant::now() < deadline,
            "the engine outlived the mode switch"
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    let client = connect(host.socket_path());
    record_and_submit(&client, "take_m", "job_m");
    let outcome = job_outcome(&client, "job_m");
    assert_eq!(outcome.type_name(), "jobs.failed", "{}", outcome.payload());
    // The manual endpoint's transport failure, not one of the engine's
    // reasons — the proof the job left the engine.
    assert_eq!(outcome.payload()["reason"], "transport_error");
    drop(client);
    host.shutdown();
}

/// The next take-feed frame matching `predicate`.
fn until_take(
    client: &HostClient,
    label: &str,
    predicate: impl Fn(&starling_runtime_host::client::TakeWire) -> bool,
) -> starling_runtime_host::client::TakeWire {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut seen = Vec::new();
    loop {
        match client.recv_take_timeout(Duration::from_millis(20)) {
            Ok(frame) if predicate(&frame) => return frame,
            Ok(frame) => seen.push(format!("{frame:?}")),
            Err(starling_runtime::channel::RecvError::Timeout) => {
                assert!(Instant::now() < deadline, "timed out waiting for {label}; saw {seen:?}")
            }
            Err(other) => panic!("take feed error: {other:?} waiting for {label}"),
        }
        while let Ok(_event) = client.try_recv_event() {}
    }
}

/// #220 + #356: a retry with another installed model runs on the host's
/// engine once it serves that model — the app switches the engine (the
/// settings say so), the host follows and transcribes; a model the engine
/// never serves leaves the take as it is.
#[test]
fn a_retry_with_another_model_runs_once_the_host_s_engine_serves_it() {
    use starling_runtime_host::client::TakeWire;
    use starling_runtime_host::frame::{TranscribeWith, TranscriptionState};
    let Some(fixture) = fixture() else { return };
    let other = "fixture-model-2";
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_models(root.path(), Some(&fixture), &[MODEL_ID, other]);
    let path = root.path().join("settings.json");
    let mut settings = Settings::default_settings();
    settings.engine.mode = EngineMode::Builtin;
    settings.engine.active_model = Some(MODEL_ID.into());
    settings.save(&path).unwrap();
    let host_setup = host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    )
    .with_settings_path(&path)
    .with_settings_poll(Duration::from_millis(20))
    .with_engine_wait(Duration::from_secs(3));
    let mut host = serve(host_setup).expect("host serves");
    wait_ready(&host.engine().expect("builtin mode supervises an engine"));
    let app = connect(host.socket_path());
    app.take_watch().expect("watching");

    // A take the host transcribes on the engine it serves.
    app.send(Some("take_m"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until_take(&app, "a status tick", |frame| {
        matches!(frame, TakeWire::Live { status: Some(_), .. })
    });
    app.send(Some("take_m"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    let stored = until_take(&app, "the take stored", |frame| {
        matches!(frame, TakeWire::Persisted { stored_id: Some(_), .. })
    });
    let TakeWire::Persisted { stored_id: Some(id), .. } = stored else { unreachable!() };
    let started = until_take(&app, "its transcription starting", |frame| {
        matches!(frame, TakeWire::Transcription { state: TranscriptionState::Started { .. }, .. })
    });
    assert!(matches!(
        started,
        TakeWire::Transcription { state: TranscriptionState::Started { ref backend }, .. }
            if backend == &format!("engine:{MODEL_ID}")
    ));
    until_take(&app, "its transcription", |frame| {
        matches!(frame, TakeWire::Transcription { req: None, state, .. } if state.is_final())
    });

    // The app switches the engine to the other model, then asks.
    settings.engine.active_model = Some(other.into());
    settings.save(&path).unwrap();
    app.transcribe("r_other", &id, TranscribeWith::Model { model_id: other.into() })
        .unwrap();
    let done = until_take(&app, "the retry", |frame| {
        matches!(frame, TakeWire::Transcription { req: Some(req), state, .. } if req == "r_other" && state.is_final())
    });
    assert!(
        matches!(&done, TakeWire::Transcription { state: TranscriptionState::Completed { .. }, .. }),
        "{done:?}"
    );
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    let backends: Vec<_> = store
        .attempts_for(&id)
        .unwrap()
        .into_iter()
        .map(|attempt| (attempt.backend, attempt.status))
        .collect();
    assert_eq!(
        backends,
        vec![
            (format!("engine:{MODEL_ID}"), "completed".to_string()),
            (format!("engine:{other}"), "completed".to_string()),
        ]
    );
    drop(store);

    // A model the engine never serves: nothing is attempted.
    app.transcribe("r_none", &id, TranscribeWith::Model { model_id: "not-installed".into() })
        .unwrap();
    let refused = until_take(&app, "the refusal", |frame| {
        matches!(frame, TakeWire::Transcription { req: Some(req), state, .. } if req == "r_none" && state.is_final())
    });
    assert!(
        matches!(&refused, TakeWire::Transcription { state: TranscriptionState::Refused { message }, .. } if message.contains("unchanged")),
        "{refused:?}"
    );
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    assert_eq!(store.attempts_for(&id).unwrap().len(), 2);
    drop(store);
    drop(app);
    host.shutdown();
}

/// The file that holds a staged engine (see [`stage_held_engine_dir`]):
/// while it exists under `root`, a wrapper serving a `.gguf` waits
/// instead of exec'ing the fixture.
#[cfg(unix)]
fn engine_hold(root: &Path) -> PathBuf {
    root.join("engine-hold")
}

/// Stages the engine dir with a wrapper that waits for the hold file
/// under `root` to disappear before the real fixture, whenever its
/// arguments mention a model file, then re-hashes it. A speed check
/// against it stays `Running` until the test removes the file —
/// deterministic, unlike a fixed sleep racing the engine feed's
/// 200 ms poll on a loaded runner (which could let the check finish
/// before a test that thinks it is concurrent ever starts). The live
/// engine passes the same wrapper, so tests observe it serving first
/// and only then create the hold file. The loop still gives up after
/// ~120 s, so a test that panics with the hold in place leaves no
/// immortal child behind.
#[cfg(unix)]
fn stage_held_engine_dir(root: &Path, fixture: &Path) {
    use std::io::Write;

    let engines = root.join("engines");
    std::fs::create_dir_all(&engines).unwrap();
    let staged = engines.join("starling-serve-cpu");
    let quoted = format!("'{}'", fixture.display().to_string().replace('\'', r"'\''"));
    let hold = format!(
        "'{}'",
        engine_hold(root).display().to_string().replace('\'', r"'\''")
    );
    let script = format!(
        "#!/bin/sh\ncase \"$*\" in *.gguf*) n=0; while [ -f {hold} ] && [ $n -lt 1200 ]; do sleep 0.1; n=$((n+1)); done ;; esac\nexec {quoted} \"$@\"\n"
    );
    // Written by a `sh` child rather than in this process: a sibling
    // test spawning while such a descriptor is open copies it into a
    // child that holds it until exec, which fails with ETXTBSY ("Text
    // file busy") under any name the file is later renamed to.
    let mut writer = std::process::Command::new("sh")
        .args(["-c", "cat > \"$0\" && chmod 755 \"$0\""])
        .arg(&staged)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    writer
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    assert!(writer.wait().unwrap().success(), "stage the held engine");

    let sha = sha256_file(&staged).unwrap();
    std::fs::write(
        engines.join("SHA256SUMS.txt"),
        format!("{sha}  starling-serve-cpu\n"),
    )
    .unwrap();
    std::fs::write(
        engines.join("engines.json"),
        format!(
            r#"{{"version":"0.1.0","abi":{EXPECTED_ENGINE_ABI},"engines":[{{"backend":"cpu","file":"starling-serve-cpu"}}]}}"#
        ),
    )
    .unwrap();
}

/// [`engine_config_with_models`] with a held engine: the model is
/// installed, but while [`engine_hold`] exists under `root` a fresh
/// sidecar does not become ready, so a check stays observably
/// `Running` until the test removes the hold.
#[cfg(unix)]
fn engine_config_with_held_models(
    root: &Path,
    fixture: &Path,
    models: &[&str],
) -> EngineConfig {
    stage_held_engine_dir(root, fixture);
    let models_dir = root.join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    let mut catalog = Vec::new();
    for (index, id) in models.iter().enumerate() {
        let file_name = format!("{id}.gguf");
        let model_file = models_dir.join(&file_name);
        std::fs::write(&model_file, vec![(7 + index * 13) as u8; 4096]).unwrap();
        let sha = sha256_file(&model_file).unwrap();
        std::fs::write(models_dir.join(format!("{file_name}.verified")), &sha).unwrap();
        catalog.push(CatalogEntry::new(
            id,
            "Fixture",
            "parakeet",
            &format!("http://127.0.0.1:9/{file_name}"),
            4096,
            &sha,
            false,
            "host engine test model",
        ));
    }
    EngineConfig {
        engine_dir: Some(root.join("engines")),
        models_dir,
        state_dir: root.join("engine-state"),
        catalog,
        backend_override: None,
        icd_dirs: Some(Vec::new()),
        available_memory_override: Some(None),
        backoff_schedule: Some(vec![Duration::from_millis(100)]),
    }
}

/// A connection following the take feed — and with it the engine's
/// status pushes.
fn watching(path: &Path) -> HostClient {
    let client = connect(path);
    client.take_watch().expect("watching");
    client
}

/// The next engine status pushed to `client` that satisfies `predicate`.
fn until_engine(
    client: &HostClient,
    label: &str,
    predicate: impl Fn(&EngineStatus) -> bool,
) -> EngineStatus {
    use starling_runtime_host::client::TakeWire;
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut last = None;
    loop {
        match client.recv_take_timeout(Duration::from_millis(20)) {
            Ok(TakeWire::Engine(status)) if predicate(&status) => return *status,
            Ok(TakeWire::Engine(status)) => last = Some(status),
            Ok(_) => {}
            Err(starling_runtime::channel::RecvError::Timeout) => assert!(
                Instant::now() < deadline,
                "timed out waiting for {label}; last engine status {last:#?}"
            ),
            Err(other) => panic!("take feed error: {other:?} waiting for {label}"),
        }
        while let Ok(_event) = client.try_recv_event() {}
    }
}

/// Whether `status` shows the built-in engine serving `model_id` with
/// no switch under way.
fn serving(status: &EngineStatus, model_id: &str) -> bool {
    status.snapshot.as_ref().is_some_and(|snapshot| {
        snapshot.phase == EnginePhase::Ready
            && snapshot.switch.is_none()
            && snapshot
                .active
                .as_ref()
                .is_some_and(|active| active.model_id == model_id)
    })
}

/// The engine settings as the app sends them: the built-in engine on
/// `backend`, or the user's server at `endpoint`.
fn intent(mode: EngineMode, backend: Option<starling_dictation::engine::Backend>, endpoint: &str) -> EngineIntent {
    EngineIntent {
        mode,
        active_model: Some(MODEL_ID.into()),
        backend_override: backend,
        endpoint: endpoint.into(),
        model: "parakeet".into(),
    }
}

fn wait_until_gone(endpoint: &str, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while endpoint_answers(endpoint) {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// #220: the app's Activate reaches the host's engine over the socket.
/// Every watching window hears the switch, and the engine a take still
/// holds drains — it serves that take, and stops once it is done.
#[test]
fn an_activation_over_the_socket_switches_and_drains_the_host_s_engine() {
    let Some(fixture) = fixture() else { return };
    let other = "fixture-model-2";
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_models(root.path(), Some(&fixture), &[MODEL_ID, other]);
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    let second_window = watching(host.socket_path());
    let first = until_engine(&app, "the first model serving", |status| serving(status, MODEL_ID))
        .snapshot
        .and_then(|snapshot| snapshot.active)
        .expect("an engine is active");

    // A take holds the engine (as a transcription does).
    let lease = host.engine().expect("builtin").lease().expect("a lease on the engine");
    let reply = app
        .engine(EngineRequest::Activate {
            model_id: other.into(),
        })
        .expect("answered");
    assert!(matches!(reply, EngineReply::Activating { .. }), "{reply:?}");
    for window in [&app, &second_window] {
        let switched = until_engine(window, "the second model serving", |status| {
            serving(status, other)
        });
        let active = switched.snapshot.unwrap().active.unwrap();
        assert_ne!(active.pid, first.pid, "a second engine serves the new model");
    }
    assert!(
        endpoint_answers(&first.endpoint),
        "the old engine drains: the open take still has it"
    );
    drop(lease);
    wait_until_gone(&first.endpoint, "the drained engine never stopped");
    drop((app, second_window));
    host.shutdown();
}

/// #220: "Use CPU engine" reloads the host's engine live, with no
/// restart of the host and no second sidecar left behind: the new
/// engine serves, the old one (no take holds it) stops.
#[test]
fn a_backend_change_over_the_socket_reloads_the_engine_live() {
    use starling_dictation::engine::Backend;
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config(root.path(), Some(&fixture));
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    let ready = until_engine(&app, "the engine serving", |status| serving(status, MODEL_ID));
    assert_eq!(ready.backend_override, None);
    let first = ready.snapshot.and_then(|snapshot| snapshot.active).unwrap();

    let reply = app
        .engine(EngineRequest::Configure {
            intent: intent(EngineMode::Builtin, Some(Backend::Cpu), ""),
        })
        .expect("answered");
    let EngineReply::Done { revision } = reply else {
        panic!("the backend change applies: {reply:?}")
    };
    assert!(revision > ready.revision, "the settings revision moves on");
    let reloaded = until_engine(&app, "the reloaded engine serving", |status| {
        status.backend_override == Some(Backend::Cpu)
            && serving(status, MODEL_ID)
            && status
                .snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.active.as_ref())
                .is_some_and(|active| active.pid != first.pid)
    });
    let active = reloaded.snapshot.unwrap().active.unwrap();
    assert!(endpoint_answers(&active.endpoint), "the new engine serves");
    wait_until_gone(&first.endpoint, "the replaced engine was left running beside the new one");
    drop(app);
    host.shutdown();
}

/// #220: the user's own server, set in the app, transcribes the host's
/// takes at once; switching back to the built-in engine starts it again
/// on the host's own paths.
#[test]
fn a_manual_endpoint_configured_over_the_socket_transcribes_takes() {
    use fake_engine::{FakeEngine, Reply, StreamMode};
    use starling_runtime_host::client::TakeWire;
    use starling_runtime_host::frame::TranscriptionState;
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config(root.path(), Some(&fixture));
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    let first = until_engine(&app, "the engine serving", |status| serving(status, MODEL_ID))
        .snapshot
        .and_then(|snapshot| snapshot.active)
        .unwrap();

    let server = FakeEngine::start(vec![Reply::Text("from my server".into())], StreamMode::Refuse);
    let reply = app
        .engine(EngineRequest::Configure {
            intent: intent(EngineMode::Manual, None, &server.endpoint()),
        })
        .expect("answered");
    assert!(matches!(reply, EngineReply::Done { .. }), "{reply:?}");
    let manual = until_engine(&app, "manual mode", |status| status.mode == EngineMode::Manual);
    assert_eq!(manual.label, format!("manual:{}", server.endpoint()));
    assert!(manual.snapshot.is_none());
    assert!(host.engine().is_none(), "the built-in engine is gone");
    wait_until_gone(&first.endpoint, "the built-in engine outlived the switch to manual");

    // The host's own transcription of the take, the one every app take
    // gets (a `jobs.submit` beside it would be a second request: only
    // the agent's asks, which the host does not transcribe, use one).
    app.send(Some("take_manual"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until_take(&app, "a status tick", |frame| {
        matches!(frame, TakeWire::Live { status: Some(_), .. })
    });
    app.send(Some("take_manual"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    let done = until_take(&app, "its transcription", |frame| {
        matches!(frame, TakeWire::Transcription { req: None, state, .. } if state.is_final())
    });
    assert!(
        matches!(&done, TakeWire::Transcription { state: TranscriptionState::Completed { text, .. }, .. } if text == "from my server"),
        "{done:?}"
    );
    assert_eq!(server.batch_requests(), 1, "the take went to the user's server, once");

    // Back to the built-in engine.
    let reply = app
        .engine(EngineRequest::Configure {
            intent: intent(EngineMode::Builtin, None, &server.endpoint()),
        })
        .expect("answered");
    assert!(matches!(reply, EngineReply::Done { .. }), "{reply:?}");
    until_engine(&app, "the built-in engine serving again", |status| {
        status.mode == EngineMode::Builtin && serving(status, MODEL_ID)
    });
    assert_eq!(server.batch_requests(), 1, "nothing transcribed the take again");
    drop(app);
    host.shutdown();
}

/// Serves `bytes` at `/<name>` in slow chunks, so a download's progress
/// is observable.
fn slow_model_server(name: &'static str, bytes: Vec<u8>) -> std::net::SocketAddr {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => break,
                }
            }
            if !String::from_utf8_lossy(&head).contains(name) {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n");
                continue;
            }
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                bytes.len()
            );
            for chunk in bytes.chunks(bytes.len().div_ceil(10)) {
                if stream.write_all(chunk).is_err() {
                    break;
                }
                let _ = stream.flush();
                std::thread::sleep(Duration::from_millis(150));
            }
        }
    });
    addr
}

/// #220: a download the app asks for runs in the host; its progress
/// reaches the watching app, a delete while it runs is refused with the
/// manager's own sentence, and the model ends up installed. No engine is
/// needed for any of it.
#[test]
fn a_download_over_the_socket_reports_its_progress() {
    use starling_dictation::engine::InstallState;
    let root = tempfile::tempdir().unwrap();
    let mut config = engine_config(root.path(), None);
    let bytes: Vec<u8> = (0..400_000u32).map(|i| (i % 251) as u8).collect();
    let staged = root.path().join("expected.bin");
    std::fs::write(&staged, &bytes).unwrap();
    let sha = sha256_file(&staged).unwrap();
    let addr = slow_model_server("remote.gguf", bytes.clone());
    config.catalog.push(CatalogEntry::new(
        "remote-model",
        "Remote",
        "parakeet",
        &format!("http://{addr}/remote.gguf"),
        bytes.len() as u64,
        &sha,
        false,
        "a model to download",
    ));
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: None,
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    let install = |status: &EngineStatus| {
        status.snapshot.as_ref().and_then(|snapshot| {
            snapshot
                .models
                .iter()
                .find(|model| model.id == "remote-model")
                .map(|model| model.install.clone())
        })
    };
    until_engine(&app, "the catalog", |status| {
        install(status) == Some(InstallState::NotInstalled)
    });

    let reply = app
        .engine(EngineRequest::Download {
            model_id: "remote-model".into(),
        })
        .expect("answered");
    assert!(matches!(reply, EngineReply::Done { .. }), "{reply:?}");
    // Progress from the first half: the delete below must not race the
    // download's last chunks on a loaded machine.
    let progress = until_engine(&app, "early download progress", |status| {
        matches!(
            install(status),
            Some(InstallState::Downloading { done, total }) if done > 0 && done * 2 < total
        )
    });
    assert!(matches!(
        install(&progress),
        Some(InstallState::Downloading { total, .. }) if total == bytes.len() as u64
    ));
    match app
        .engine(EngineRequest::Delete {
            model_id: "remote-model".into(),
        })
        .expect("answered")
    {
        EngineReply::Refused { message } => assert!(message.contains("downloading"), "{message}"),
        other => panic!("a delete mid-download must be refused: {other:?}"),
    }
    until_engine(&app, "the model installed", |status| {
        install(status) == Some(InstallState::Installed)
    });
    assert_eq!(
        std::fs::read(root.path().join("models/remote.gguf")).unwrap(),
        bytes
    );
    drop(app);
    host.shutdown();
}

/// #220: a switch the memory policy refuses reaches the app as the
/// manager's decision, and the current model keeps serving.
#[test]
fn a_memory_refusal_reaches_the_app() {
    use starling_dictation::engine::SwapDecision;
    let Some(fixture) = fixture() else { return };
    let other = "fixture-model-2";
    let root = tempfile::tempdir().unwrap();
    let mut config = engine_config_with_models(root.path(), Some(&fixture), &[MODEL_ID, other]);
    // Barely any memory free: the second model fits neither beside the
    // first nor after it.
    config.available_memory_override = Some(Some(1024));
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    until_engine(&app, "the first model serving", |status| serving(status, MODEL_ID));
    app.engine(EngineRequest::Activate {
        model_id: other.into(),
    })
    .expect("answered");
    let refused = until_engine(&app, "the refusal", |status| {
        status
            .snapshot
            .as_ref()
            .is_some_and(|snapshot| matches!(snapshot.pending_decision, Some(SwapDecision::Refused { .. })))
    });
    let snapshot = refused.snapshot.unwrap();
    assert_eq!(snapshot.active.unwrap().model_id, MODEL_ID, "the current model keeps serving");
    drop(app);
    host.shutdown();
}

/// #475: a check that refuses before it starts any work must land as
/// `Failed`, not stranding `Running`.
///
/// `run_speed_check` refuses a model whose file has gone missing before
/// it spawns anything, so the worker reaches its terminal state far
/// faster than the thread starting it can reach its next line. Writing
/// `Running` *after* the spawn therefore overwrites that `Failed` and
/// leaves a status no live worker will ever clear: no button on any row,
/// every retry refused, until the host restarts. With that ordering this
/// test waits for a `Failed` that never comes and times out.
#[test]
fn a_check_that_refuses_before_it_starts_lands_as_failed() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_models(root.path(), Some(&fixture), &[MODEL_ID]);
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    until_engine(&app, "the first model serving", |status| {
        serving(status, MODEL_ID)
    });

    // The engine is serving, and only now does the file go away — the
    // state a delete actually races into. The host has already loaded the
    // weights, so removing the file takes its engine down with it; the
    // speed check must not strand the status on the way out.
    let model_file = root.path().join("models").join(format!("{MODEL_ID}.gguf"));
    std::fs::remove_file(&model_file).unwrap();

    let reply = app
        .engine(EngineRequest::SpeedCheck {
            model_id: MODEL_ID.into(),
        })
        .expect("answered");
    assert!(matches!(reply, EngineReply::Done { .. }), "{reply:?}");

    let settled = until_engine(&app, "the refusal recorded", |status| {
        matches!(
            status.speed_check,
            Some(SpeedCheckStatus::Failed { .. } | SpeedCheckStatus::Reported { .. })
        )
    });
    match settled.speed_check.as_ref().expect("settled") {
        SpeedCheckStatus::Failed { message, .. } => {
            // The missing-file refusal, specifically. `contains("installed")`
            // alone is almost a tautology with the alternative below.
            assert!(
                message.to_lowercase().contains("not installed"),
                "expected the missing-model refusal, got: {message}"
            );
        }
        other => panic!("expected Failed for a model with no file, got {other:?}"),
    }
    // Reaching that assertion through `until_engine` is the proof: a
    // status stranded at `Running` would have timed the wait out.
    drop(app);
    host.shutdown();
}

/// #475: the app's SpeedCheck reaches the host's engine over the socket,
/// and the answer comes back as an engine status push — not as the reply,
/// which only says "started". One line per probe, and the synthetic one
/// marked as the upper bound it is.
///
/// Unix-only because it needs the staged hold to see the transient
/// `Running` push: a `sh` wrapper cannot slow a Windows engine, and the
/// host feed polls every 200 ms while the contract fixture finishes a
/// probe faster than that. The refusal test below runs everywhere.
#[cfg(unix)]
#[test]
fn a_speed_check_over_the_socket_reports_through_the_engine_feed() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_held_models(root.path(), &fixture, &[MODEL_ID]);
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    let second_window = watching(host.socket_path());
    until_engine(&app, "the first model serving", |status| {
        serving(status, MODEL_ID)
    });

    // Hold the probe's sidecar at its wrapper: the check now stays
    // `Running` until this test releases it, however loaded the runner.
    std::fs::write(engine_hold(root.path()), "").unwrap();
    let reply = app
        .engine(EngineRequest::SpeedCheck {
            model_id: MODEL_ID.into(),
        })
        .expect("answered");
    assert!(
        matches!(reply, EngineReply::Done { .. }),
        "the reply only acknowledges the start; the numbers arrive as pushes: {reply:?}"
    );

    // Both watching windows hear it, in the same order.
    let running = until_engine(&app, "the check running", |status| {
        matches!(status.speed_check, Some(SpeedCheckStatus::Running { .. }))
    });
    assert_eq!(
        running.speed_check.as_ref().and_then(|s| match s {
            SpeedCheckStatus::Running { model_id } => Some(model_id.as_str()),
            _ => None,
        }),
        Some(MODEL_ID)
    );
    // The live engine before the probe's own sidecar exists.
    let live_pid = |status: &EngineStatus| {
        status
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.active.as_ref())
            .map(|active| active.pid)
    };
    let pid_before = live_pid(&running);

    // Release the probe: now it may finish and report.
    std::fs::remove_file(engine_hold(root.path())).unwrap();
    let reported = until_engine(&app, "the check reported", |status| {
        matches!(status.speed_check, Some(SpeedCheckStatus::Reported { .. }))
    });
    let results = match reported.speed_check.as_ref().expect("reported") {
        SpeedCheckStatus::Reported { results, .. } => results.clone(),
        other => panic!("expected Reported, got {other:?}"),
    };
    assert_eq!(results.len(), 2, "{results:?}");
    assert!(
        results[0].kind.is_upper_bound(),
        "the synthetic tone is the upper bound it is labelled as: {results:?}"
    );
    assert!(!results[1].kind.is_upper_bound(), "{results:?}");
    let line = results[1].describe();
    assert!(line.contains("speech clip"), "{line}");
    assert!(line.contains("realtime"), "{line}");
    assert!(!line.contains("upper bound"), "{line}");

    let also_reported =
        until_engine(&second_window, "the second window's report", |status| {
            matches!(status.speed_check, Some(SpeedCheckStatus::Reported { .. }))
        });
    assert_eq!(also_reported.speed_check, reported.speed_check);

    // The live engine is the one before, still serving the same model:
    // a probe runs its own sidecar and leaves it stopped. Read from the
    // pushes already in hand — a check that changed nothing pushes
    // nothing new — and from the host's own manager, not from a wait for
    // a push that would not come.
    assert_eq!(live_pid(&reported), pid_before, "the live engine's process never changed");
    let manager = host.engine().expect("builtin");
    let snapshot = manager.snapshot();
    assert_eq!(
        snapshot.active.as_ref().map(|active| active.pid),
        pid_before,
        "the host's engine is still the one it was"
    );
    assert_eq!(
        snapshot.active.map(|active| active.model_id),
        Some(MODEL_ID.to_string()),
        "and it still serves the same model"
    );
    drop((app, second_window));
    host.shutdown();
}

/// #475: a second check while one runs is refused, not queued — two
/// probes fighting over the CPU would measure each other, and the feed
/// carries exactly one progress line.
#[cfg(unix)]
#[test]
fn a_second_speed_check_is_refused_while_one_runs() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_held_models(root.path(), &fixture, &[MODEL_ID]);
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    until_engine(&app, "the first model serving", |status| {
        serving(status, MODEL_ID)
    });

    // Held at its wrapper, the first check is still running whenever
    // the second request lands — however long the runner took.
    std::fs::write(engine_hold(root.path()), "").unwrap();
    // Nothing running yet: the first request is taken.
    assert!(matches!(
        app.engine(EngineRequest::SpeedCheck {
            model_id: MODEL_ID.into()
        })
        .expect("answered"),
        EngineReply::Done { .. }
    ));
    let _running = until_engine(&app, "the check running", |status| {
        matches!(status.speed_check, Some(SpeedCheckStatus::Running { .. }))
    });
    let refused = app
        .engine(EngineRequest::SpeedCheck {
            model_id: MODEL_ID.into(),
        })
        .expect("answered");
    match refused {
        EngineReply::Refused { message } => {
            assert!(message.contains("already running"), "{message}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    // The first check still finishes and reports once released.
    std::fs::remove_file(engine_hold(root.path())).unwrap();
    let _reported = until_engine(&app, "the first check reported", |status| {
        matches!(status.speed_check, Some(SpeedCheckStatus::Reported { .. }))
    });
    drop(app);
    host.shutdown();
}

/// #475: a check for a model that is not in the catalog is refused with
/// the model named, and nothing is pushed.
#[test]
fn a_speed_check_for_an_unknown_model_is_refused() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_models(root.path(), Some(&fixture), &[MODEL_ID]);
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    until_engine(&app, "the first model serving", |status| {
        serving(status, MODEL_ID)
    });

    let reply = app
        .engine(EngineRequest::SpeedCheck {
            model_id: "not-a-model".into(),
        })
        .expect("answered");
    match reply {
        EngineReply::Refused { message } => {
            assert!(message.contains("not-a-model"), "{message}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    // Nothing was left running behind the refusal — a stuck `Running`
    // would hide the button on every row and refuse this retry.
    assert!(matches!(
        app.engine(EngineRequest::SpeedCheck {
            model_id: MODEL_ID.into(),
        })
        .expect("answered"),
        EngineReply::Done { .. }
    ));
    drop(app);
    host.shutdown();
}

/// #475: the probe loads a second resident copy of the model beside the
/// live engine, so it obeys the same memory policy as a swap (#363).
/// With essentially nothing free, the check is refused before it
/// records `Running` or spawns anything — and the serving model is
/// untouched.
#[test]
fn a_speed_check_is_refused_when_a_second_resident_copy_cannot_fit() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let mut config = engine_config_with_models(root.path(), Some(&fixture), &[MODEL_ID]);
    config.available_memory_override = Some(Some(1024));
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    let before = until_engine(&app, "the first model serving", |status| {
        serving(status, MODEL_ID)
    });
    let live_pid = before
        .snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.active.as_ref())
        .map(|active| active.pid);

    let reply = app
        .engine(EngineRequest::SpeedCheck {
            model_id: MODEL_ID.into(),
        })
        .expect("answered");
    let message = match reply {
        EngineReply::Refused { message } => message,
        other => panic!("expected a memory refusal, got {other:?}"),
    };
    assert!(message.contains("memory"), "{message}");

    // Recorded on the model's row, not only in the reply — and never as
    // `Running`: nothing was started that would clear a running state.
    let settled = until_engine(&app, "the memory refusal recorded", |status| {
        matches!(status.speed_check, Some(SpeedCheckStatus::Failed { .. }))
    });
    match settled.speed_check.as_ref().expect("recorded") {
        SpeedCheckStatus::Failed {
            model_id,
            message: recorded,
        } => {
            assert_eq!(model_id, MODEL_ID);
            assert_eq!(recorded, &message);
        }
        other => panic!("expected Failed, got {other:?}"),
    }
    // The live engine never blinked.
    let after = settled
        .snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.active.as_ref())
        .map(|active| active.pid);
    assert_eq!(after, live_pid, "the serving engine is the same process");
    drop(app);
    host.shutdown();
}

/// How many processes carry `needle` in their command line (Linux
/// `/proc`); `None` where there is no process table to read.
#[cfg(unix)]
fn count_processes_with(needle: &str) -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let mut count = 0;
        for entry in std::fs::read_dir("/proc").ok()? {
            let Ok(entry) = entry else { continue };
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            if let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) {
                if String::from_utf8_lossy(&cmdline).contains(needle) {
                    count += 1;
                }
            }
        }
        Some(count)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = needle;
        None
    }
}

/// #475: a probe load and a model switch admitted together must not
/// each spend the whole free-memory budget a pre-load snapshot
/// showed. Both models are one GiB sparse files (resident estimate
/// 1.5 GiB; needed with margin exactly the 2 GiB reported free), so
/// either load alone fits — together they cannot, before either
/// allocates. Whichever is admitted second must be decided against
/// what remains: the probe refuses, and the switch is asked to drain
/// instead of rolling.
#[cfg(unix)]
#[test]
fn a_probe_and_a_switch_cannot_spend_the_same_free_memory_twice() {
    use starling_dictation::engine::{SwapDecision, SwitchStage};
    let Some(fixture) = fixture() else { return };
    for switch_first in [true, false] {
        let root = tempfile::tempdir().unwrap();
        let mut config =
            engine_config_with_held_models(root.path(), &fixture, &[MODEL_ID, "other"]);
        // Sparse files make the catalog's sizes true without writing a
        // GiB per model; the `.verified` markers stay as staged.
        for entry in &mut config.catalog {
            entry.size_bytes = 1024 * 1024 * 1024;
            std::fs::OpenOptions::new()
                .write(true)
                .open(config.models_dir.join(format!("{}.gguf", entry.id)))
                .unwrap()
                .set_len(entry.size_bytes)
                .unwrap();
        }
        config.available_memory_override = Some(Some(2 * 1024 * 1024 * 1024));
        let engines = root.path().join("engines/starling-serve-cpu");
        let engines = engines.to_str().unwrap().to_string();
        let mut host = serve(host_config(
            root.path(),
            EngineChoice::Builtin {
                config,
                active_model: Some(MODEL_ID.into()),
            },
        ))
        .expect("host serves");
        let app = watching(host.socket_path());
        until_engine(&app, "the first model serving", |status| {
            serving(status, MODEL_ID)
        });
        // Both loads now park at their wrappers until released, so the
        // second admission is decided while the first has allocated
        // nothing — where a fresh snapshot alone would still show the
        // full 2 GiB free.
        std::fs::write(engine_hold(root.path()), "").unwrap();

        if switch_first {
            app.engine(EngineRequest::Activate {
                model_id: "other".into(),
            })
            .expect("answered");
            // The held wrapper process existing proves admission already
            // happened (it is spawned only after the policy step), unlike
            // the Loading stage, which is published before the switch
            // worker runs — on machines without a readable process
            // table, fall back to the stage.
            if count_processes_with(&engines).is_some() {
                let deadline = Instant::now() + Duration::from_secs(15);
                while count_processes_with(&engines) != Some(1) {
                    assert!(
                        Instant::now() < deadline,
                        "the admitted switch's held wrapper never appeared"
                    );
                    std::thread::sleep(Duration::from_millis(50));
                }
            } else {
                until_engine(&app, "the switch loading", |status| {
                    status.snapshot.as_ref().is_some_and(|snapshot| {
                        snapshot
                            .switch
                            .as_ref()
                            .is_some_and(|switch| switch.stage == SwitchStage::Loading)
                    })
                });
            }
            let reply = app
                .engine(EngineRequest::SpeedCheck {
                    model_id: MODEL_ID.into(),
                })
                .expect("answered");
            match reply {
                EngineReply::Refused { message } => {
                    assert!(message.contains("memory"), "{message}")
                }
                other => panic!("the probe must refuse behind the switch's load: {other:?}"),
            }
        } else {
            assert!(matches!(
                app.engine(EngineRequest::SpeedCheck {
                    model_id: MODEL_ID.into()
                })
                .expect("answered"),
                EngineReply::Done { .. }
            ));
            // `Running` is recorded only after the reservation, so
            // observing it means the probe's claim is registered.
            until_engine(&app, "the check running", |status| {
                matches!(status.speed_check, Some(SpeedCheckStatus::Running { .. }))
            });
            app.engine(EngineRequest::Activate {
                model_id: "other".into(),
            })
            .expect("answered");
            // Not admitted rolling beside the probe: the switch must
            // ask to drain (which loads nothing until confirmed).
            until_engine(&app, "the switch asked to drain", |status| {
                status.snapshot.as_ref().is_some_and(|snapshot| {
                    matches!(snapshot.pending_decision, Some(SwapDecision::NeedsDrain { .. }))
                })
            });
        }
        if count_processes_with(&engines).is_some() {
            assert_eq!(
                count_processes_with(&engines),
                Some(1),
                "exactly one admitted load is in flight (switch_first={switch_first})"
            );
        }
        drop(app);
        host.shutdown();
        std::fs::remove_file(engine_hold(root.path())).unwrap();
    }
}

/// #475: the reverse admission order. Nothing serves (the big model
/// crash-looped), a 2.625 GiB probe is admitted and held, then a tiny
/// target is activated — it fits beside the probe, so it is admitted.
/// Cancelling that switch must not now spawn the 1.5 GiB *recovery* of
/// the previous model beside the still-held probe (2.625 + 1.5 > the
/// 4 GiB free): recovery waits until the probe's claim is gone, even
/// though the switch's reservation already covered its bytes — the
/// reservation sizes the claim, the recovery spawn still has to fit
/// beside claims admitted after it.
#[cfg(unix)]
#[test]
fn recovery_waits_for_a_probe_admitted_before_the_switch() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let mut config =
        engine_config_with_held_models(root.path(), &fixture, &[MODEL_ID, "small", "probe"]);
    for entry in &mut config.catalog {
        entry.size_bytes = match entry.id.as_str() {
            MODEL_ID => 1024 * 1024 * 1024,
            "probe" => 2 * 1024 * 1024 * 1024,
            _ => 4096,
        };
        std::fs::OpenOptions::new()
            .write(true)
            .open(config.models_dir.join(format!("{}.gguf", entry.id)))
            .unwrap()
            .set_len(entry.size_bytes)
            .unwrap();
    }
    config.available_memory_override = Some(Some(4 * 1024 * 1024 * 1024));
    let big_gguf = config.models_dir.join(format!("{MODEL_ID}.gguf"));
    let probe_gguf = config.models_dir.join("probe.gguf");
    let small_gguf = config.models_dir.join("small.gguf");
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    until_engine(&app, "the big model serving", |status| {
        serving(status, MODEL_ID)
    });

    // Crash-loop the big model out of service: five kills inside the
    // window trip the loop guard, leaving last_active_model = the big
    // model with nothing serving.
    let manager = host.engine().expect("builtin");
    let wait_snapshot = |label: &str, ok: &dyn Fn(&starling_dictation::engine::EngineSnapshot) -> bool| {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let snapshot = manager.snapshot();
            if ok(&snapshot) {
                return snapshot;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {label}");
            std::thread::sleep(Duration::from_millis(25));
        }
    };
    for _ in 0..5 {
        let pid = wait_snapshot("an engine to kill", &|s| s.active.is_some())
            .active
            .expect("active")
            .pid;
        assert!(
            std::process::Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status()
                .unwrap()
                .success(),
            "kill the engine"
        );
        wait_snapshot("the engine's death", &move |s| {
            s.active.as_ref().is_none_or(|active| active.pid != pid)
        });
    }
    until_engine(&app, "the crash-loop failure", |status| {
        status.snapshot.as_ref().is_some_and(|snapshot| {
            matches!(snapshot.phase, EnginePhase::Failed(_)) && snapshot.active.is_none()
        })
    });

    // Probe first, held; then the small target, also held.
    std::fs::write(engine_hold(root.path()), "").unwrap();
    assert!(matches!(
        app.engine(EngineRequest::SpeedCheck {
            model_id: "probe".into()
        })
        .expect("answered"),
        EngineReply::Done { .. }
    ));
    let count = |path: &std::path::Path| count_processes_with(path.to_str().unwrap());
    let wait_count = |path: &std::path::Path, want: usize, label: &str| {
        if count(path).is_none() {
            return; // no readable process table: the model-level waits still ran
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        while count(path) != Some(want) {
            assert!(Instant::now() < deadline, "timed out waiting for {label}");
            std::thread::sleep(Duration::from_millis(50));
        }
    };
    wait_count(&probe_gguf, 1, "the probe held");
    app.engine(EngineRequest::Activate {
        model_id: "small".into(),
    })
    .expect("answered");
    wait_count(&small_gguf, 1, "the small switch held");

    app.engine(EngineRequest::CancelSwitch).expect("answered");
    // Recovery of the big model must not spawn beside the held probe.
    // The window starts only once the cancelled target has exited —
    // recovery runs immediately after that, so spawning then is
    // observable; starting the clock at the CancelSwitch reply would
    // let slow teardown, not coordination, pass the test.
    wait_count(&small_gguf, 0, "the cancelled target exited");
    if count(&big_gguf).is_some() {
        let settle = Instant::now() + Duration::from_secs(1);
        while Instant::now() < settle {
            assert_eq!(
                count(&big_gguf),
                Some(0),
                "recovery spawned beside the held probe"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    // Release everything: the probe finishes, its claim drops, and
    // recovery brings the big model back.
    std::fs::remove_file(engine_hold(root.path())).unwrap();
    until_engine(&app, "the big model recovered", |status| {
        serving(status, MODEL_ID)
    });
    drop(app);
    host.shutdown();
}

/// #475: a check that is still measuring when the host shuts down stops
/// with the host. The probe is held at its wrapper for the whole test,
/// so shutdown can only return promptly — and leave nothing running —
/// if it cancels the worker and stops its sidecar rather than
/// detaching both.
#[cfg(unix)]
#[test]
fn shutdown_stops_a_running_speed_check() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config_with_held_models(root.path(), &fixture, &[MODEL_ID]);
    let mut host = serve(host_config(
        root.path(),
        EngineChoice::Builtin {
            config,
            active_model: Some(MODEL_ID.into()),
        },
    ))
    .expect("host serves");
    let app = watching(host.socket_path());
    until_engine(&app, "the first model serving", |status| {
        serving(status, MODEL_ID)
    });

    std::fs::write(engine_hold(root.path()), "").unwrap();
    assert!(matches!(
        app.engine(EngineRequest::SpeedCheck {
            model_id: MODEL_ID.into()
        })
        .expect("answered"),
        EngineReply::Done { .. }
    ));
    until_engine(&app, "the check running", |status| {
        matches!(status.speed_check, Some(SpeedCheckStatus::Running { .. }))
    });
    let engines = root.path().join("engines/starling-serve-cpu");
    let engines = engines.to_str().unwrap().to_string();
    // The probe's wrapper is up and held (Linux-observable).
    if let Some(count) = count_processes_with(&engines) {
        assert!(count >= 1, "the held probe wrapper is running");
    }

    drop(app);
    // The hold is never released: this returns because shutdown
    // cancels the check and stops its sidecar, not because the probe
    // finished.
    host.shutdown();

    if count_processes_with(&engines).is_some() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut count = count_processes_with(&engines).unwrap();
        while count > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            count = count_processes_with(&engines).unwrap();
        }
        assert_eq!(count, 0, "no probe process outlives shutdown");
    }
    std::fs::remove_file(engine_hold(root.path())).unwrap();
}
