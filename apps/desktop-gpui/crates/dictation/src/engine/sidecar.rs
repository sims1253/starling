//! One supervised `starling-serve` process (#362 step 3).
//!
//! The spawn contract with the server (`cpp/serve/main.cpp`): start with
//! `--model <slug> --gguf <path> --host 127.0.0.1 --port 0
//! --no-eager-load --parent-pid <our pid>`, wait for the single stdout
//! line `STARLING_SERVE_LISTENING 127.0.0.1:<port>`, then drive readiness
//! over HTTP (`/v1/models` -> `POST /warmup` -> `/health`). The
//! `--parent-pid` watchdog makes the server exit if this whole process
//! dies — the orphan backstop underneath the supervisor's own stop
//! handling.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

/// How long the announce line may take after spawn.
pub const ANNOUNCE_TIMEOUT: Duration = Duration::from_secs(30);
/// `/v1/models` readiness budget.
const MODELS_BUDGET: Duration = Duration::from_secs(15);
/// How long the loading phase may take before the attempt is given up.
pub const LOADING_CAP: Duration = Duration::from_secs(15 * 60);
/// The stderr ring buffer keeps this many lines.
const STDERR_RING_LINES: usize = 64;
/// The shared engine log is truncated when it has grown past this.
const LOG_TRUNCATE_ABOVE: u64 = 5 * 1024 * 1024;
/// `CREATE_NO_WINDOW`: spawning a console process on Windows must not
/// flash a terminal window in the user's face.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Progress reports from [`Sidecar::wait_ready`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadyStage {
    /// Waiting for the announce line.
    Starting,
    /// The model is loading (`phase=loading`/`unloaded`).
    Loading,
    /// Loaded, warmup in flight (`loaded && !warm`).
    Warming,
}

/// The `/health` readiness ladder (a pure mapping of the additive
/// supervision fields, #362).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthLadder {
    /// `phase=loading` (or still `unloaded` before the warmup kick).
    Loading,
    /// `loaded && !warm`: warmup in flight.
    Warming,
    /// `warm`: the engine serves.
    Warmed,
}

/// Why a sidecar did not reach ready. `Crashed` is the only restartable
/// failure — `LoadFailed` is deterministic (the model cannot load in
/// this process), and auto-restarting it would just burn a loop.
#[derive(Clone, Debug, PartialEq)]
pub enum ReadyError {
    /// The process exited before announcing, with its stderr.
    Crashed { status: String, stderr_tail: String },
    /// No announce line within [`ANNOUNCE_TIMEOUT`].
    AnnounceTimeout,
    /// The process reported a bind/listen failure before announcing.
    BindFailed(String),
    /// `/health` carried a non-null `load_error`.
    LoadFailed(String),
    /// The caller cancelled the wait (a superseded switch, shutdown).
    Cancelled,
    /// Readiness polling timed out or the HTTP side never came up.
    TimedOut(String),
}

impl ReadyError {
    /// Whether a restart could plausibly help (the process itself died).
    pub fn is_crash(&self) -> bool {
        matches!(self, ReadyError::Crashed { .. })
    }
}

impl std::fmt::Display for ReadyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadyError::Crashed {
                status,
                stderr_tail,
            } => {
                write!(f, "the engine exited {status}: {stderr_tail}")
            }
            ReadyError::AnnounceTimeout => {
                write!(f, "the engine never reported its address")
            }
            ReadyError::BindFailed(message) => {
                write!(f, "the engine could not bind a loopback port: {message}")
            }
            ReadyError::LoadFailed(message) => {
                write!(f, "the model failed to load: {message}")
            }
            ReadyError::Cancelled => write!(f, "the start was cancelled"),
            ReadyError::TimedOut(message) => write!(f, "{message}"),
        }
    }
}

/// The `/health` fields the supervisor cares about (additive
/// supervision fields, #362).
#[derive(Clone, Debug, PartialEq)]
pub struct HealthSnapshot {
    pub model: String,
    pub loaded: bool,
    pub warm: bool,
    pub busy: bool,
    pub phase: String,
    pub queue_depth: u64,
    /// The compile-time family before load, the actual device after
    /// (`Vulkan0`, `CPU`, `contract-fixture`, ...).
    pub backend: String,
    pub load_error: Option<String>,
}

impl HealthSnapshot {
    /// Parses the `/health` document; `Err` carries what was wrong.
    pub fn parse(body: &str) -> Result<HealthSnapshot, String> {
        let value: Value =
            serde_json::from_str(body).map_err(|error| format!("invalid health JSON: {error}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| "health JSON is not an object".to_string())?;
        let string = |key: &str| {
            object
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("health JSON field {key} missing"))
        };
        let boolean = |key: &str| {
            object
                .get(key)
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("health JSON field {key} missing"))
        };
        Ok(HealthSnapshot {
            model: string("model")?,
            loaded: boolean("loaded")?,
            warm: boolean("warm")?,
            busy: boolean("busy")?,
            phase: string("phase")?,
            queue_depth: object
                .get("queue_depth")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            backend: string("backend")?,
            load_error: object
                .get("load_error")
                .and_then(Value::as_str)
                .map(str::to_string),
        })
    }

    /// Maps a health snapshot onto the readiness ladder. Pure, so the
    /// mapping is unit-testable against canned JSON.
    pub fn stage(&self) -> Result<HealthLadder, ReadyError> {
        if let Some(error) = &self.load_error {
            return Err(ReadyError::LoadFailed(error.clone()));
        }
        if self.warm {
            return Ok(HealthLadder::Warmed);
        }
        if self.loaded {
            return Ok(HealthLadder::Warming);
        }
        Ok(HealthLadder::Loading)
    }
}

/// A tiny loopback HTTP helper: the async reqwest client driven on a
/// private current-thread runtime (the `client.rs` pattern). One per
/// sidecar plus one for the manager's attach polls.
pub(crate) struct LoopbackHttp {
    client: reqwest::Client,
    runtime: tokio::runtime::Runtime,
}

impl LoopbackHttp {
    pub(crate) fn new() -> LoopbackHttp {
        LoopbackHttp {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("reqwest client for loopback probes builds"),
            runtime: tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("current-thread runtime builds"),
        }
    }

    /// GETs `url` and returns the body text; short timeout because these
    /// are loopback supervision probes.
    pub(crate) fn get_text(&self, url: &str) -> Result<String, String> {
        self.runtime.block_on(async {
            let response = self
                .client
                .get(url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .map_err(|error| error.to_string())?;
            let status = response.status().as_u16();
            let body = response.text().await.map_err(|error| error.to_string())?;
            if (200..300).contains(&status) {
                Ok(body)
            } else {
                Err(format!("HTTP {status}"))
            }
        })
    }

    /// POSTs `url` and returns the status code.
    pub(crate) fn post(&self, url: &str) -> Result<u16, String> {
        self.runtime.block_on(async {
            let response = self
                .client
                .post(url)
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .map_err(|error| error.to_string())?;
            Ok(response.status().as_u16())
        })
    }
}

/// The shared stderr tail: a bounded ring plus the count of dropped
/// lines, so crash reports stay small but informative.
#[derive(Default)]
struct StderrRing {
    lines: std::collections::VecDeque<String>,
}

impl StderrRing {
    fn push(&mut self, line: String) {
        if self.lines.len() >= STDERR_RING_LINES {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    fn tail(&self) -> String {
        let joined = self
            .lines
            .iter()
            .rev()
            .take(8)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join(" | ");
        if joined.trim().is_empty() {
            "(no output)".to_string()
        } else {
            joined
        }
    }
}

/// One running engine process. Clone-safe: the child and stderr ring are
/// shared, and [`Sidecar::stop`] is idempotent.
pub struct Sidecar {
    pid: u32,
    port: Mutex<Option<u16>>,
    child: Arc<Mutex<Option<Child>>>,
    stderr_ring: Arc<Mutex<StderrRing>>,
    /// The last `/health.backend` value seen.
    device: Mutex<Option<String>>,
    /// The announce line receiver, handed over by [`Sidecar::spawn`] and
    /// consumed exactly once by [`Sidecar::wait_ready`].
    announce_rx: Mutex<Option<mpsc::Receiver<u16>>>,
    #[allow(dead_code)]
    log_path: PathBuf,
    http: LoopbackHttp,
}

impl Sidecar {
    /// Spawns the server as described in the module docs. Returns before
    /// the announce line; call [`Sidecar::wait_ready`] to drive the
    /// process to a warm engine.
    ///
    /// `log_path` receives the process's stderr (shared across
    /// sidecars); it is truncated when it has grown past 5 MiB.
    pub fn spawn(
        engine_path: &Path,
        slug: &str,
        gguf: &Path,
        log_path: &Path,
    ) -> Result<Sidecar, String> {
        let mut command = Command::new(engine_path);
        command.args([
            "--model",
            slug,
            "--gguf",
            &gguf.display().to_string(),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--no-eager-load",
            "--parent-pid",
            &std::process::id().to_string(),
        ]);
        command.stdin(Stdio::null());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command
            .spawn()
            .map_err(|error| format!("could not start the engine process: {error}"))?;
        let pid = child.id();

        if let Some(parent) = log_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        // Truncate an oversized shared log at open (keep the newest run
        // readable; the full history lives in the journal, not here).
        if let Ok(meta) = fs::metadata(log_path) {
            if meta.len() > LOG_TRUNCATE_ABOVE {
                let _ = fs::write(log_path, b"");
            }
        }
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .ok();

        // Announce reader: exactly one `STARLING_SERVE_LISTENING` line is
        // expected on stdout; everything else is logged as stderr-like
        // noise (the server keeps stdout for machine lines only).
        let (announce_tx, announce_rx) = mpsc::channel::<u16>();
        if let Some(stdout) = child.stdout.take() {
            std::thread::Builder::new()
                .name("starling-engine-stdout".into())
                .spawn(move || {
                    let mut reader = BufReader::new(stdout);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        match reader.read_line(&mut line) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                if let Some(rest) =
                                    line.trim().strip_prefix("STARLING_SERVE_LISTENING ")
                                {
                                    if let Some(port) = rest
                                        .rsplit(':')
                                        .next()
                                        .and_then(|port| port.parse::<u16>().ok())
                                    {
                                        let _ = announce_tx.send(port);
                                    }
                                }
                            }
                        }
                    }
                })
                .map_err(|error| format!("could not start the announce reader: {error}"))?;
        }

        // Stderr reader: ring buffer + shared log file.
        let stderr_ring = Arc::new(Mutex::new(StderrRing::default()));
        if let Some(stderr) = child.stderr.take() {
            let ring = Arc::clone(&stderr_ring);
            std::thread::Builder::new()
                .name("starling-engine-stderr".into())
                .spawn(move || {
                    let mut reader = BufReader::new(stderr);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        match reader.read_line(&mut line) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                let trimmed = line.trim_end().to_string();
                                ring.lock().map(|mut ring| ring.push(trimmed.clone())).ok();
                                if let Some(log) = &log {
                                    let mut log = log;
                                    let _ = writeln!(log, "{trimmed}");
                                }
                            }
                        }
                    }
                })
                .map_err(|error| format!("could not start the stderr reader: {error}"))?;
        }

        Ok(Sidecar {
            pid,
            port: Mutex::new(None),
            child: Arc::new(Mutex::new(Some(child))),
            stderr_ring,
            device: Mutex::new(None),
            announce_rx: Mutex::new(Some(announce_rx)),
            log_path: log_path.to_path_buf(),
            http: LoopbackHttp::new(),
        })
    }

    /// The process id (valid until [`Sidecar::stop`]).
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// `http://127.0.0.1:<port>` once announced.
    pub fn endpoint(&self) -> Option<String> {
        self.port
            .lock()
            .ok()
            .and_then(|port| *port)
            .map(|port| format!("http://127.0.0.1:{port}"))
    }

    /// The device the engine actually runs on (from `/health.backend`),
    /// once known: the compile-time family before load, `Vulkan0`/`CPU`
    /// after (#362 runtime truth).
    pub fn device(&self) -> Option<String> {
        self.device.lock().ok().and_then(|device| device.clone())
    }

    fn record_device(&self, device: &str) {
        if let Ok(mut slot) = self.device.lock() {
            *slot = Some(device.to_string());
        }
    }

    /// The last few stderr lines (crash context).
    pub fn stderr_tail(&self) -> String {
        self.stderr_ring
            .lock()
            .map(|ring| ring.tail())
            .unwrap_or_else(|_| "(stderr unavailable)".to_string())
    }

    /// Drives the process to a warm engine:
    ///
    /// 1. wait for the announce line (30 s, cancel-aware);
    /// 2. poll `GET /v1/models` until it lists `slug` (100 ms cadence,
    ///    15 s budget);
    /// 3. `POST /warmup` (the deferred load starts here);
    /// 4. poll `GET /health` (250 ms) until warm, mapping
    ///    [`HealthSnapshot::stage`]; a nonzero `load_error` is
    ///    deterministic `LoadFailed`.
    ///
    /// `progress` receives the stage changes (the manager maps them to
    /// `EnginePhase`), `cancel` aborts within ~100 ms.
    pub fn wait_ready(
        &self,
        slug: &str,
        cancel: Option<&AtomicBool>,
        progress: &dyn Fn(ReadyStage),
    ) -> Result<HealthSnapshot, ReadyError> {
        progress(ReadyStage::Starting);
        let port = self.wait_announce(cancel)?;
        if let Ok(mut slot) = self.port.lock() {
            *slot = Some(port);
        }
        let base = format!("http://127.0.0.1:{port}");

        // /v1/models readiness.
        let models_deadline = Instant::now() + MODELS_BUDGET;
        loop {
            if cancelled(cancel) {
                return Err(ReadyError::Cancelled);
            }
            if let Some(status) = self.exited_status() {
                return Err(self.crash(status));
            }
            if let Ok(body) = self.http.get_text(&format!("{base}/v1/models")) {
                if models_list_slug(&body, slug) {
                    break;
                }
            }
            if Instant::now() >= models_deadline {
                return Err(ReadyError::TimedOut(format!(
                    "the engine did not list the model {slug} within 15 s"
                )));
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        // Warmup (deferred load). The POST triggers load+warmup; a
        // transport error before the server is warm is retried within
        // the loading budget rather than failing the whole attempt.
        let loading_deadline = Instant::now() + LOADING_CAP;
        let mut warmup_sent_at: Option<Instant> = None;
        loop {
            if cancelled(cancel) {
                return Err(ReadyError::Cancelled);
            }
            if let Some(status) = self.exited_status() {
                return Err(self.crash(status));
            }
            let send_warmup = match warmup_sent_at {
                None => true,
                Some(at) => at.elapsed() > Duration::from_secs(5),
            };
            if send_warmup {
                if let Ok(status) = self.http.post(&format!("{base}/warmup")) {
                    if (200..300).contains(&status) {
                        warmup_sent_at = Some(Instant::now());
                    }
                }
            }
            if let Ok(body) = self.http.get_text(&format!("{base}/health")) {
                if let Ok(health) = HealthSnapshot::parse(&body) {
                    if health.model == slug || health.model.is_empty() {
                        match health.stage() {
                            Ok(HealthLadder::Loading) => progress(ReadyStage::Loading),
                            Ok(HealthLadder::Warming) => progress(ReadyStage::Warming),
                            Ok(HealthLadder::Warmed) => {
                                progress(ReadyStage::Warming);
                                self.record_device(&health.backend);
                                return Ok(health);
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
            }
            if Instant::now() >= loading_deadline {
                return Err(ReadyError::TimedOut(
                    "loading did not finish within 15 minutes".to_string(),
                ));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn wait_announce(&self, cancel: Option<&AtomicBool>) -> Result<u16, ReadyError> {
        let announce_rx = self
            .announce_rx
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        let Some(announce_rx) = announce_rx else {
            // A second wait_ready on the same sidecar has no channel left;
            // treat it as the timeout it effectively is.
            return Err(ReadyError::AnnounceTimeout);
        };
        let deadline = Instant::now() + ANNOUNCE_TIMEOUT;
        loop {
            if cancelled(cancel) {
                return Err(ReadyError::Cancelled);
            }
            if let Some(status) = self.exited_status() {
                let tail = self.stderr_tail();
                if tail.contains("bind")
                    || tail.contains("listen")
                    || tail.contains("Address already in use")
                {
                    return Err(ReadyError::BindFailed(tail));
                }
                return Err(ReadyError::Crashed {
                    status,
                    stderr_tail: tail,
                });
            }
            match announce_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(port) => return Ok(port),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Reader ended without the line: wait out the deadline
                    // in case exit detection has not caught up yet.
                }
            }
            if Instant::now() >= deadline {
                self.kill();
                return Err(ReadyError::AnnounceTimeout);
            }
        }
    }

    /// Whether the process has exited, as a display string. Reaps the
    /// child when it has (the pid is gone either way).
    pub fn exited_status(&self) -> Option<String> {
        let mut guard = self.child.lock().ok()?;
        let child = guard.as_mut()?;
        match child.try_wait() {
            Ok(Some(status)) => {
                *guard = None;
                Some(format!("{status}"))
            }
            Ok(None) => None,
            Err(_) => None,
        }
    }

    fn crash(&self, status: String) -> ReadyError {
        ReadyError::Crashed {
            status,
            stderr_tail: self.stderr_tail(),
        }
    }

    /// Fetches `/health` from the announced endpoint (attach polling,
    /// device refresh). `None` when the endpoint is unknown or the
    /// request failed.
    pub fn health(&self) -> Option<HealthSnapshot> {
        let endpoint = self.endpoint()?;
        let body = self.http.get_text(&format!("{endpoint}/health")).ok()?;
        let health = HealthSnapshot::parse(&body).ok()?;
        self.record_device(&health.backend);
        Some(health)
    }

    /// Stops the process: SIGTERM, up to 1.5 s grace, then SIGKILL
    /// (Unix) / `kill()` (Windows); always reaps. Idempotent and safe to
    /// call from any thread.
    pub fn stop(&self) {
        let Some(mut child) = self.child.lock().ok().and_then(|mut guard| guard.take()) else {
            return;
        };
        #[cfg(unix)]
        {
            let pid = self.pid as i32;
            unsafe {
                libc::kill(pid, libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_millis(1500);
            while Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        let _ = child.wait();
                        return;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                    Err(_) => break,
                }
            }
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = child.kill();
        }
        let _ = child.wait();
    }

    fn kill(&self) {
        if let Some(mut child) = self.child.lock().ok().and_then(|mut guard| guard.take()) {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Dropping the last handle stops the process: a leaked sidecar must not
/// outlive the manager that spawned it (#362 acceptance: quitting leaves
/// no server process behind). `stop` is idempotent, so a sidecar that
/// was already stopped (or already exited) is a no-op.
impl Drop for Sidecar {
    fn drop(&mut self) {
        self.stop();
    }
}

fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
}

/// `{object: "list", data: [{id: "<slug>"}]}` must name the served slug
/// before transcription requests are sent (they 404 otherwise).
fn models_list_slug(body: &str, slug: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    value
        .get("data")
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.get("id").and_then(Value::as_str) == Some(slug))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_json_maps_to_stages() {
        let loading = HealthSnapshot::parse(
            r#"{"status":"ok","model":"parakeet","loaded":false,"busy":false,
                "phase":"loading","queue_depth":0,"backend":"cpu","warm":false,"load_error":null}"#,
        )
        .expect("parses");
        assert_eq!(loading.stage(), Ok(HealthLadder::Loading));

        let warming = HealthSnapshot::parse(
            r#"{"status":"ok","model":"parakeet","loaded":true,"busy":false,
                "phase":"ready","queue_depth":0,"backend":"cpu","warm":false,"load_error":null}"#,
        )
        .expect("parses");
        assert_eq!(warming.stage(), Ok(HealthLadder::Warming));

        let ready = HealthSnapshot::parse(
            r#"{"status":"ok","model":"parakeet","loaded":true,"busy":false,
                "phase":"ready","queue_depth":2,"backend":"Vulkan0","warm":true,"load_error":null}"#,
        )
        .expect("parses");
        assert_eq!(ready.stage(), Ok(HealthLadder::Warmed));
        assert!(ready.warm);

        let failed = HealthSnapshot::parse(
            r#"{"status":"ok","model":"parakeet","loaded":false,"busy":false,
                "phase":"unloaded","queue_depth":0,"backend":"cpu","warm":false,"load_error":"bad gguf"}"#,
        )
        .expect("parses");
        assert_eq!(
            failed.stage(),
            Err(ReadyError::LoadFailed("bad gguf".to_string()))
        );

        // A Vulkan build that found no device reports the CPU device
        // after load — surfaced as a notice by the manager, parsed here.
        let fallen_back = HealthSnapshot::parse(
            r#"{"status":"ok","model":"parakeet","loaded":true,"busy":false,
                "phase":"ready","queue_depth":0,"backend":"CPU","warm":true,"load_error":null}"#,
        )
        .expect("parses");
        assert_eq!(fallen_back.backend, "CPU");

        assert!(HealthSnapshot::parse("not json").is_err());
        assert!(HealthSnapshot::parse("{}").is_err());
    }

    #[test]
    fn models_list_check_requires_the_slug() {
        let body = r#"{"object":"list","data":[{"id":"parakeet"},{"id":"s1"}]}"#;
        assert!(models_list_slug(body, "parakeet"));
        assert!(models_list_slug(body, "s1"));
        assert!(!models_list_slug(body, "moss"));
        assert!(!models_list_slug(
            r#"{"object":"list","data":[]}"#,
            "parakeet"
        ));
        assert!(!models_list_slug("garbage", "parakeet"));
    }

    #[test]
    fn stderr_ring_keeps_only_the_tail() {
        let mut ring = StderrRing::default();
        for index in 0..(STDERR_RING_LINES + 10) {
            ring.push(format!("line {index}"));
        }
        let tail = ring.tail();
        assert!(tail.contains(&format!("line {}", STDERR_RING_LINES + 9)));
        assert!(!tail.contains("line 0 |"));
    }
}
