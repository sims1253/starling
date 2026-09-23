//! `default_capture_store` pinned for embedders (#259 — PR #257 review
//! thread 4065952668): the constructor is embedder-facing by design and
//! stays uncalled by production inside this workspace, because the I4
//! host deliberately does not route through it — see
//! `production_host_wires_the_durable_store_and_refuses_to_degrade`:
//! a lease-owning service must refuse to start when its data root will
//! not open, never slide onto this constructor's in-memory fallback.
//! What must not happen is the seam silently rotting while nothing
//! exercises it. The tests pin the embedder recipe end to end —
//! construct through the public API, resolve storage v2 at the default
//! root, persist a take that survives a reopen of that root, and boot a
//! runtime configured with it — so the constructor stays valid for the
//! root-less embedders it exists for (the GPUI in-process switchover).

#![cfg(unix)] // matches the sibling suites; the recipe module below is
              // further gated to Linux, where the `dirs` data root
              // honors XDG_DATA_HOME.

use starling_runtime_host::HostConfig;

/// The host's side of thread 4065952668: the production entry point
/// opts into the durable store (the review's literal ask) through an
/// explicit root, and an unopenable root is a startup refusal — the
/// fail-loud contract that is exactly why the host must NOT call
/// `default_capture_store()`, whose in-memory fallback would turn this
/// refusal into a silently non-durable host.
#[test]
fn production_host_wires_the_durable_store_and_refuses_to_degrade() {
    let dir = tempfile::tempdir().expect("host root temp dir");
    let runtime_dir = dir.path().join("endpoints");

    let config = HostConfig::production(dir.path(), Some(runtime_dir.clone()))
        .expect("production config at an openable root");
    assert_eq!(
        config.runtime.capture_store.describe(),
        "storage-v2",
        "the production entry point wires the durable store"
    );

    // A root occupied by a regular file cannot open (create_dir_all
    // fails) — production must error, never fall back.
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"not a directory").expect("write the blocker");
    let err = match HostConfig::production(&blocker, Some(runtime_dir)) {
        Ok(_) => panic!("an unopenable root must be a startup refusal"),
        Err(err) => err,
    };
    assert!(
        // Couples to config.rs's error wording — `HostConfig::production`
        // returns `String`, so a phrase is the only discriminator; if the
        // message is rephrased, update this with it.
        err.contains("will not open"),
        "the refusal names the root: {err}"
    );
}

/// The full embedder recipe, exercised against a redirected default
/// root: `default_capture_store()` must hand back storage v2 at
/// `<XDG_DATA_HOME>/starling-gpui` (proving the no-argument contract
/// resolves the *default* root), a take committed through the returned
/// `Arc<dyn CaptureStore>` must survive a reopen of that root, and the
/// documented call shape — `RuntimeConfig::default()
/// .with_capture_store(default_capture_store())` — must boot a runtime.
///
/// Linux-only: macOS and Windows `dirs::data_dir()` are not
/// env-redirectable, and the test must never touch a real user data
/// root. The module gate also keeps its helpers out of the builds where
/// they would be dead code.
#[cfg(target_os = "linux")]
mod embedder_recipe {
    use std::path::PathBuf;

    use starling_dictation::store_v2::{CaptureStatus, ListedCapture, StoreV2};
    use starling_runtime::machine::capture::{TakeRecord, TakeStatus};
    use starling_runtime::{default_capture_store, Runtime, RuntimeConfig};

    /// Env is process-global and the harness runs tests in parallel
    /// threads: any test in this binary that touches `XDG_DATA_HOME`
    /// must hold this lock for as long as its redirect is live, making
    /// the one-writer invariant structural rather than documented.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Sets `XDG_DATA_HOME` for the test body and restores the prior
    /// value (or unsets it) on drop — panic-safe. Holds [`ENV_LOCK`]
    /// until drop, so a future env-dependent test in this binary
    /// serializes against this one by construction, not convention.
    struct XdgRedirect {
        prior: Option<std::ffi::OsString>,
        /// Never read — its Drop (releasing [`ENV_LOCK`]) is the whole
        /// point. A named field, not a tuple position, so the never-read
        /// lint and the intent agree.
        _env_lock: std::sync::MutexGuard<'static, ()>,
    }

    impl XdgRedirect {
        fn to(dir: &std::path::Path) -> Self {
            let _env_lock = ENV_LOCK.lock().expect("env lock not poisoned");
            let prior = std::env::var_os("XDG_DATA_HOME");
            std::env::set_var("XDG_DATA_HOME", dir);
            Self { prior, _env_lock }
        }
    }

    impl Drop for XdgRedirect {
        fn drop(&mut self) {
            // `Drop::drop` runs before any field drops, so the env is
            // restored while the lock is still held — the release then
            // happens in `_env_lock`'s own drop, after the restore.
            match self.prior.take() {
                Some(value) => std::env::set_var("XDG_DATA_HOME", value),
                None => std::env::remove_var("XDG_DATA_HOME"),
            }
        }
    }

    /// The minimal take the sibling `scripted_take` suite commits directly
    /// (no journal evidence — the samples path through the §4 protocol).
    fn take_record(id: &str, samples: &[f32]) -> TakeRecord {
        TakeRecord {
            id: id.to_string(),
            device: "default-input".to_string(),
            policy: "push-to-talk".to_string(),
            samples: samples.to_vec(),
            sample_rate: 16_000,
            gaps: Vec::new(),
            acknowledged_samples: samples.len() as u64,
            final_sample_index: samples.len() as u64,
            journal: None,
            status: TakeStatus::Complete,
            sample_duration_ms: samples.len() as f64 * 1000.0 / 16_000.0,
            wall_clock_ms: samples.len() as f64 * 1000.0 / 16_000.0,
            capture_id: id.to_string(),
        }
    }

    #[test]
    fn default_capture_store_pins_the_embedder_recipe() {
        let xdg = tempfile::tempdir().expect("redirected XDG_DATA_HOME");
        let _redirect = XdgRedirect::to(xdg.path());
        let root: PathBuf = xdg.path().join("starling-gpui");

        // Construct + identify: the durable store, not the fallback.
        // The root gate runs BEFORE any commit: if the redirect were
        // ever broken, the freshly-created redirected root would not
        // exist and the test aborts before a take could be written
        // anywhere but under the tempdir.
        let store = default_capture_store();
        assert_eq!(store.describe(), "storage-v2", "the default root opened");
        assert!(
            root.is_dir() && root.starts_with(xdg.path()),
            "the no-argument contract resolved the redirected root {}",
            root.display()
        );

        // Persist a take with no journal evidence (the samples path — the
        // §4 crash protocol, not a hand-built row).
        let samples: Vec<f32> = (0..200).map(|i| (i % 53) as f32 * 0.003).collect();
        store
            .commit_take(&take_record("take_default_store", &samples))
            .expect("commit through the default store");
        drop(store);

        // Durability: reopen the root through store v2's own public API —
        // the row and its promoted audio must be there.
        let reopened = StoreV2::open(&root).expect("reopen the default root");
        let page = reopened.list_records(0, 10).expect("list the root");
        assert_eq!(page.total, 1, "exactly one take persisted");
        let ListedCapture::Capture(listing) = &page.records[0] else {
            panic!("expected a readable capture, got {:?}", page.records[0]);
        };
        assert_eq!(listing.record.status, CaptureStatus::Complete);
        assert!(
            listing.problems.is_empty(),
            "no audio problems: {:?}",
            listing.problems
        );
        assert!(
            reopened
                .audio_journal_exists(&listing.record.id)
                .expect("audio journal probe"),
            "the take's audio was promoted into the root"
        );
        drop(reopened);

        // The documented call shape boots: one config line is the whole
        // wiring for a root-less embedder.
        let (runtime, client) =
            Runtime::start(RuntimeConfig::default().with_capture_store(default_capture_store()));
        let snapshot = client.snapshot();
        assert!(
            snapshot.frozen_routes.is_empty(),
            "a freshly booted runtime has no frozen routes"
        );
        runtime.shutdown();
    }
}
