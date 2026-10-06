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
use starling_dictation::engine::{CatalogEntry, EngineConfig, EnginePhase};
use starling_dictation::settings::{EngineMode, Settings};
use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
use starling_runtime::protocol::Command;
use starling_runtime::provider::FakeProvider;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::{EventWire, HostClient};
use starling_runtime_host::engine::{self, EngineChoice};
use starling_runtime_host::{serve, HostConfig, HostError};

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
        std::fs::write(
            engines.join("engines.json"),
            r#"{"version":"0.1.0","abi":8,"engines":[{"backend":"cpu","file":"starling-serve-cpu"}]}"#,
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
    assert!(active.owned, "the host owns the sidecar it started");

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

/// The engine the host attached to (not started) dies under the host
/// with no warning — the desktop app owned the shared sidecar and was
/// hard-killed, so the sidecar's `--parent-pid` watchdog ended it
/// abruptly (SIGKILL here: no retire step, nothing tells the attached
/// host first). A job submitted right then hits the dead engine; it must
/// still complete — the provider retries once on whatever engine the
/// supervisors bring back (a restart or the host's takeover).
#[cfg(unix)] // the abrupt kill is a signal; the fixture is unix-built
#[test]
fn a_job_survives_the_attached_engine_dying_abruptly() {
    let Some(fixture) = fixture() else { return };
    let root = tempfile::tempdir().unwrap();
    let config = engine_config(root.path(), Some(&fixture));

    // The "renderer" owns the sidecar: it started first.
    let app_engine =
        starling_dictation::engine::EngineManager::start(config.clone(), Some(MODEL_ID.into()));
    wait_ready(&app_engine);
    assert!(app_engine.snapshot().active.unwrap().owned);

    let engine = EngineChoice::Builtin {
        config,
        active_model: Some(MODEL_ID.into()),
    };
    let mut host = serve(host_config(root.path(), engine)).expect("host serves");
    let manager = host.engine().expect("builtin mode").clone();
    wait_ready(&manager);
    let attached = manager.snapshot().active.unwrap();
    assert!(!attached.owned, "the host attached to the app's sidecar");

    let client = connect(host.socket_path());
    record_and_submit_after(&client, "take_a", "job_a", || {
        // Re-read right before the kill: the config restarts a dead
        // engine on a 100 ms backoff, so the earlier snapshot could be
        // stale — signal only the engine that is still the attached one
        // (endpoint and pid), never a pid the supervisor already
        // replaced (or the OS recycled).
        let current = manager.snapshot().active.expect("engine still active");
        assert_eq!(
            (current.endpoint.as_str(), current.pid),
            (attached.endpoint.as_str(), attached.pid),
            "the sidecar was replaced before the kill"
        );
        // SAFETY: kill(2) on the sidecar's pid; no memory is involved.
        assert_eq!(unsafe { libc::kill(attached.pid as i32, libc::SIGKILL) }, 0);
        let deadline = Instant::now() + Duration::from_secs(5);
        while endpoint_answers(&attached.endpoint) {
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
        now.pid, attached.pid,
        "the job ran on the replacement engine"
    );
    drop(client);
    host.shutdown();
    app_engine.shutdown();
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
    );

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
