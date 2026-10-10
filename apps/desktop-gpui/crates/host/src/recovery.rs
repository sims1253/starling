//! Startup recovery beyond the store's reconcile (#220, from the app):
//! the host owns the recorder's journal tree, so it — not the app — fails
//! recognition attempts a previous run left started and brings back the
//! takes a killed process left in that tree (#356). Before #220 the app
//! ran these at its own startup, which let a second app instance adopt a
//! journal another instance was still writing.
//!
//! What recovery found reaches the app through the take feed
//! ([`crate::takes::TakeHub::set_recovery`]): problems for the error
//! banner, brought-back takes for a notice. Journals the first pass left
//! to a save that may have been under way get a second look once that
//! save would long have finished ([`recheck_later`]).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use starling_dictation::store_v2::{ReconciliationReport, StoreV2, FINALIZED_ADOPTION_GRACE};

use crate::frame::HostRecovery;

/// The note a stale recognition attempt gets at startup — the wording of
/// the v1 "stuck in Transcribing" fix.
pub const STALE_ATTEMPT_NOTE: &str =
    "Interrupted before the server returned a transcript. Your audio is ready to retry.";

/// The recorder's journal tree for a data root.
pub fn journals_dir(root: &Path) -> PathBuf {
    root.join("journals")
}

/// What the startup pass found, and the journals to look at again.
pub struct StartupRecovery {
    pub recovery: HostRecovery,
    pub recheck: Vec<String>,
}

/// The startup pass on the lease-holding store: `reconciliation` is the
/// owner reconcile [`crate::serve`] already ran (`reconcile_error` when it
/// failed). The repairs are independent, so all run whatever the others
/// did.
pub fn recover(
    store: &mut StoreV2,
    journals: &Path,
    reconciliation: &ReconciliationReport,
    reconcile_error: Option<String>,
) -> StartupRecovery {
    let mut problems: Vec<String> = reconcile_error.into_iter().collect();
    if reconciliation.has_findings() {
        problems.push(reconciliation.summary());
    }
    if let Err(err) = store.interrupt_stale_attempts(STALE_ATTEMPT_NOTE) {
        problems.push(format!("Could not recover interrupted recordings: {err}"));
    }
    let (notice, recheck) = match store.recover_capture_journals_where(journals, |_| true) {
        Ok(recovery) => {
            let found = recovery.problems();
            if !found.is_empty() {
                problems.push(found);
            }
            (recovery.recovered_summary(), recovery.deferred)
        }
        Err(err) => {
            problems.push(format!("Could not scan for interrupted recordings: {err}"));
            (String::new(), Vec::new())
        }
    };
    StartupRecovery {
        recovery: HostRecovery {
            notice,
            problems: problems
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
        },
        recheck,
    }
}

/// The second look (#356): only the deferred ids, once a save that may
/// have been under way at startup would long have finished — a take this
/// host started recording since is never a recovery candidate.
pub fn recheck(store: &mut StoreV2, journals: &Path, ids: &[String]) -> HostRecovery {
    match store.recover_capture_journals_where(journals, |id| ids.iter().any(|wanted| wanted == id)) {
        Ok(recovery) => HostRecovery {
            notice: recovery.recovered_summary(),
            problems: recovery.problems(),
        },
        Err(err) => HostRecovery {
            notice: String::new(),
            problems: format!("Could not scan for interrupted recordings: {err}"),
        },
    }
}

/// How long after startup [`recheck`] runs.
pub const RECHECK_AFTER: Duration = Duration::from_secs(2);

/// Runs [`recheck`] on its own thread after
/// [`FINALIZED_ADOPTION_GRACE`] (+ [`RECHECK_AFTER`]) and hands the
/// findings to `report`; returns at once when there is nothing to look at.
pub fn recheck_later(
    store: Arc<Mutex<StoreV2>>,
    journals: PathBuf,
    ids: Vec<String>,
    report: impl FnOnce(HostRecovery) + Send + 'static,
) {
    if ids.is_empty() {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("starling-host-recheck".to_string())
        .spawn(move || {
            std::thread::sleep(FINALIZED_ADOPTION_GRACE + RECHECK_AFTER);
            let found = {
                let mut store = store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                recheck(&mut store, &journals, &ids)
            };
            report(found);
        });
    if let Err(err) = spawned {
        eprintln!("starling-runtime-host: cannot schedule the journal recheck: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starling_dictation::audio::{encode_wav_16k, PcmAudio};
    use starling_dictation::store_v2::TakeMeta;

    fn wav(samples: usize) -> Vec<u8> {
        encode_wav_16k(&PcmAudio {
            samples: (0..samples).map(|i| (i % 50) as f32 * 0.01).collect(),
            sample_rate: 16_000,
            channels: 1,
        })
        .expect("wav")
    }

    fn take_with_attempt(store: &mut StoreV2) -> String {
        let id = store
            .save_wav_capture(&wav(800), TakeMeta::for_device("test"))
            .expect("save")
            .record
            .id;
        store
            .begin_recognition(&id, "starling:parakeet", None)
            .expect("begin");
        id
    }

    fn latest_status(store: &StoreV2, id: &str) -> String {
        store
            .attempts_for(id)
            .expect("attempts")
            .last()
            .expect("an attempt")
            .status
            .clone()
    }

    /// A journal the recorder left when its process died mid-take: the
    /// first `confirmed` samples under a fsynced boundary, the rest not.
    fn killed_journal(journals: &Path, id: &str, confirmed: &[f32], unconfirmed: &[f32]) {
        let scratch = tempfile::tempdir().unwrap();
        let writer = StoreV2::open(scratch.path()).expect("scratch store");
        let mut take = writer.begin_take(TakeMeta::for_device("test")).expect("begin");
        take.append_frames(confirmed).expect("append");
        take.write_boundary().expect("boundary");
        take.append_frames(unconfirmed).expect("tail");
        let staged = scratch.path().join("staging").join(format!("{}.sj", take.id()));
        drop(take);
        std::fs::create_dir_all(journals).expect("journals tree");
        std::fs::rename(&staged, journals.join(format!("{id}.sj"))).expect("place journal");
    }

    #[test]
    fn an_attempt_a_dead_run_left_started_is_failed_ready_to_retry() {
        let root = tempfile::tempdir().unwrap();
        let mut crashed = StoreV2::open(root.path()).unwrap();
        let id = take_with_attempt(&mut crashed);
        // That run's attempt markers die with it.
        drop(crashed);

        let mut owner = StoreV2::open(root.path()).unwrap();
        let found = recover(
            &mut owner,
            &journals_dir(root.path()),
            &ReconciliationReport::default(),
            None,
        );
        assert_eq!(found.recovery.problems, "", "a healthy store has nothing to say");
        assert_eq!(found.recovery.notice, "");
        assert_ne!(latest_status(&owner, &id), "started", "the stale attempt was failed");
    }

    #[test]
    fn an_attempt_a_live_process_still_runs_is_spared() {
        let root = tempfile::tempdir().unwrap();
        let mut live = StoreV2::open(root.path()).unwrap();
        let id = take_with_attempt(&mut live);
        let mut owner = StoreV2::open(root.path()).unwrap();
        recover(
            &mut owner,
            &journals_dir(root.path()),
            &ReconciliationReport::default(),
            None,
        );
        assert_eq!(latest_status(&owner, &id), "started", "#213: not stale while its owner lives");
    }

    #[test]
    fn a_take_killed_mid_recording_comes_back_with_what_it_confirmed() {
        let root = tempfile::tempdir().unwrap();
        let journals = journals_dir(root.path());
        let confirmed: Vec<f32> = (0..24_000).map(|i| ((i as f32) * 0.01).sin() * 0.2).collect();
        killed_journal(&journals, "j_killed", &confirmed, &[0.05; 640]);
        let mut owner = StoreV2::open(root.path()).unwrap();
        let found = recover(&mut owner, &journals, &ReconciliationReport::default(), None);
        assert!(found.recovery.notice.contains("Recovered 1 recording"), "{}", found.recovery.notice);
        assert!(found.recheck.is_empty());
        let record = owner.get_capture("j_killed").unwrap().expect("the recovered take");
        assert_eq!(record.frame_count, 24_000, "exactly what was confirmed");
    }

    #[test]
    fn a_failed_reconcile_is_reported_and_the_other_repairs_still_run() {
        let root = tempfile::tempdir().unwrap();
        let mut crashed = StoreV2::open(root.path()).unwrap();
        let id = take_with_attempt(&mut crashed);
        drop(crashed);
        let mut owner = StoreV2::open(root.path()).unwrap();
        let found = recover(
            &mut owner,
            &journals_dir(root.path()),
            &ReconciliationReport::default(),
            Some("Could not recover interrupted recordings: disk on fire".to_string()),
        );
        assert!(found.recovery.problems.contains("disk on fire"));
        assert_ne!(latest_status(&owner, &id), "started");
    }
}
