//! Compression and retention races (#342) that interleave a second store
//! handle — or a retry, a delete, a pin — inside one call. The behaviour
//! itself (FLAC fidelity, the compression crash windows, the retention
//! limits) is proven through the public entry points in
//! `tests/store_recovery.rs`; these stay beside the store because each
//! needs a peer to act mid-call.
use super::*;
use tempfile::TempDir;

fn store_in(dir: &TempDir) -> StoreV2 {
    StoreV2::open(dir.path().join("v2")).expect("open v2 store")
}

fn take_at(store: &mut StoreV2, rate: u32, samples: &[f32]) -> String {
    let mut take = store
        .begin_take_at_rate(rate, TakeMeta::for_device("test-device"))
        .expect("begin take");
    take.append_and_seal(samples).expect("append");
    take.finish(store).expect("finish").record.id
}

/// Speech-ish signal: an enveloped tone plus a little noise.
fn speechy(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let t = i as f32 / 16_000.0;
            let envelope = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 10.0 * t).sin();
            0.35 * envelope * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + (state as f32 / u32::MAX as f32 - 0.5) * 0.01
        })
        .collect()
}

fn files(store: &StoreV2, id: &str) -> (bool, bool) {
    (store.audio_path(id).exists(), store.flac_path(id).exists())
}

fn temps(store: &StoreV2) -> Vec<PathBuf> {
    std::fs::read_dir(store.root().join(AUDIO_DIR))
        .expect("audio dir")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(FLAC_TEMP_EXT))
        .collect()
}

fn compressed(outcome: CompressionOutcome) {
    assert!(
        matches!(outcome, CompressionOutcome::Compressed { .. }),
        "{outcome:?}"
    );
}

fn complete(store: &mut StoreV2, id: &str) -> String {
    store
        .begin_recognition(id, "engine:test", None)
        .expect("begin");
    store
        .finish_recognition(
            id,
            RecognitionOutcome::Completed {
                text: "hello there",
                extra_json: None,
            },
        )
        .expect("finish");
    store
        .attempts_for(id)
        .expect("attempts")
        .into_iter()
        .find(AttemptRecord::is_final_transcript)
        .expect("final attempt")
        .id
}

/// A transcribed take created `days_ago` days back.
fn aged_take(store: &mut StoreV2, days_ago: i64, len: usize) -> String {
    let id = take_at(store, 16_000, &speechy(len, days_ago as u32 + 1));
    complete(store, &id);
    backdate(store, &id, days_ago);
    id
}

fn backdate(store: &StoreV2, id: &str, days_ago: i64) {
    let created = iso_utc(time::OffsetDateTime::now_utc() - time::Duration::days(days_ago));
    store
        .conn
        .execute(
            "UPDATE captures SET created_utc = ?2 WHERE id = ?1",
            params![id, created],
        )
        .expect("backdate");
}

fn policy(class: &str, limits: ClassLimits) -> RetentionPolicy {
    let mut policy = RetentionPolicy::default();
    policy.limits.insert(class.to_string(), limits);
    policy
}

fn age(days: u32) -> ClassLimits {
    ClassLimits {
        max_age_days: Some(days),
        max_total_bytes: None,
    }
}

fn retired_ids(report: &RetentionReport) -> Vec<String> {
    report.retired.iter().map(|item| item.id.clone()).collect()
}

fn held(report: &RetentionReport, id: &str) -> Option<HoldReason> {
    report
        .held
        .iter()
        .find(|item| item.id == id)
        .map(|item| item.reason)
}

// ---- FLAC fidelity ----------------------------------------------------

#[test]
fn audio_pinned_by_an_attempt_is_never_compressed_under_it() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(16_000, 7));
    store
        .begin_recognition(&id, "engine:test", None)
        .expect("begin");
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    assert!(matches!(
        store.compress_audio(&id).expect("compress"),
        CompressionOutcome::Skipped(_)
    ));
    store
        .finish_recognition(
            &id,
            RecognitionOutcome::Failed {
                message: "engine crashed",
            },
        )
        .expect("finish");

    // A retry that starts while the encode runs off the lock wins.
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    store
        .begin_recognition(&id, "engine:test", None)
        .expect("retry");
    assert!(matches!(
        store.commit_compression(prepared).expect("commit"),
        CompressionOutcome::Skipped(_)
    ));
    assert_eq!(files(&store, &id), (true, false));
    assert!(temps(&store).is_empty(), "the abandoned temp is removed");
}

#[test]
fn deleting_during_compression_and_after_it_keeps_r21_semantics() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    // Deleted while the encode runs off the lock.
    let racing = take_at(&mut store, 16_000, &speechy(16_000, 9));
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    store.delete_capture(&racing).expect("delete");
    assert!(matches!(
        store.commit_compression(prepared).expect("commit"),
        CompressionOutcome::Skipped(_)
    ));
    assert!(temps(&store).is_empty());
    assert!(store.quarantine_path(&racing).exists());

    // Deleted after compression: the FLAC is quarantined, stays dead, and
    // the sweep removes it.
    let done = take_at(&mut store, 16_000, &speechy(16_000, 10));
    compressed(store.compress_audio(&done).expect("compress"));
    store.delete_capture(&done).expect("delete");
    assert!(store.quarantine_flac_path(&done).exists());
    assert_eq!(files(&store, &done), (false, false));
    store.reconcile().expect("reconcile");
    assert!(
        store.get_capture(&done).expect("row").is_none(),
        "never resurrected"
    );
    let report = store.sweep_retention().expect("sweep");
    let swept: Vec<&str> = report.swept.iter().map(|file| file.id.as_str()).collect();
    assert!(
        swept.contains(&racing.as_str()) && swept.contains(&done.as_str()),
        "{swept:?}"
    );
    assert!(!store.quarantine_flac_path(&done).exists());

    // A quarantined FLAC whose delete crashed before the row went is
    // completed, not resurrected.
    let crashed = take_at(&mut store, 16_000, &speechy(16_000, 11));
    compressed(store.compress_audio(&crashed).expect("compress"));
    std::fs::rename(
        store.flac_path(&crashed),
        store.quarantine_flac_path(&crashed),
    )
    .expect("quarantine rename");
    let report = store.reconcile().expect("reconcile");
    assert!(report.completed_deletes.contains(&crashed));
    assert!(store.get_capture(&crashed).expect("row").is_none());
}

#[test]
fn an_in_process_pin_holds_compression_and_retention() {
    // A retry pins before it loads the audio and keeps the pin until its
    // attempt is marked started: nothing acts in between.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    store.pin_audio(&id);
    store.pin_audio(&id);
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    let now = time::OffsetDateTime::now_utc();
    let report = store
        .apply_retention_policy(&policy(STANDARD_CLASS, age(30)), now)
        .expect("apply");
    assert!(report.retired.is_empty());
    assert_eq!(held(&report, &id), Some(HoldReason::InUse));

    // A compression prepared before the pin does not publish under it.
    store.unpin_audio(&id);
    store.unpin_audio(&id);
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    store.pin_audio(&id);
    assert!(matches!(
        store.commit_compression(prepared).expect("commit"),
        CompressionOutcome::Skipped(_)
    ));
    assert_eq!(files(&store, &id), (true, false));

    // Pins nest; the last release frees the take.
    store.pin_audio(&id);
    store.unpin_audio(&id);
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    store.unpin_audio(&id);
    store.unpin_audio(&id); // an extra release does nothing
    compressed(store.compress_audio(&id).expect("compress"));
    let report = store
        .apply_retention_policy(&policy(STANDARD_CLASS, age(30)), now)
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![id]);
}

#[test]
fn retention_decides_after_a_peer_connection_commits() {
    // Another connection on the same root (another process) is starting a
    // transcription of the take. The hold check and the stamp take the
    // write lock first, so they wait for the peer and see its attempt
    // instead of retiring the audio under it.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "INSERT INTO recognition_attempts
             (id, capture_id, backend, text, partial_or_final, status)
         VALUES ('peer-attempt', ?1, 'engine:peer', '', 'final', 'started')",
        params![id],
    )
    .expect("peer attempt");

    // Runs after the walk has read the due take and before the sweep
    // asks for the write lock.
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        peer.execute_batch("COMMIT").expect("peer commit");
    })));
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, age(30)),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert!(report.retired.is_empty(), "{report:?}");
    assert_eq!(held(&report, &id), Some(HoldReason::InUse));
    assert_eq!(
        store.audio_at_rest(&id).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_policy_changed_mid_run_stops_before_removing_more() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let older = aged_take(&mut store, 120, 16_000);
    let old = aged_take(&mut store, 90, 16_000);
    let limited = policy(STANDARD_CLASS, age(30));
    // The user lifts the limit as soon as the first removal is visible.
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    let report = store
        .apply_live_retention_policy(
            || {
                let removed_one = peer
                    .query_row(
                        "SELECT 1 FROM tombstones WHERE kind = 'audio' LIMIT 1",
                        [],
                        |_| Ok(()),
                    )
                    .optional()
                    .expect("peer read")
                    .is_some();
                if removed_one {
                    RetentionPolicy::default()
                } else {
                    limited.clone()
                }
            },
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert!(report.policy_changed);
    // Walked newest first: `old` went under the limit as it stood then.
    assert_eq!(retired_ids(&report), vec![old]);
    assert_eq!(
        store.audio_at_rest(&older).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_class_change_by_a_peer_mid_run_is_respected() {
    // A peer moves a due take into an unlimited class while the sweep
    // waits for the write lock: the take is not removed.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "UPDATE captures SET retention_class = ?2 WHERE id = ?1",
        params![id, ARCHIVAL_CLASS],
    )
    .expect("peer class change");

    // Runs after the walk has read the due take and before the sweep
    // asks for the write lock.
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        peer.execute_batch("COMMIT").expect("peer commit");
    })));
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, age(30)),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert!(report.retired.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&id).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_size_limit_recounts_after_a_peer_moves_a_newer_take_away() {
    // Two old takes under a limit that fits one. While the sweep waits to
    // retire the older, a peer moves the newer into the archival class:
    // the class now fits, so the older keeps its audio.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let older = aged_take(&mut store, 120, 16_000);
    let newer = aged_take(&mut store, 90, 16_000);
    let limit = store.audio_bytes(&older).max(store.audio_bytes(&newer));
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "UPDATE captures SET retention_class = ?2 WHERE id = ?1",
        params![newer, ARCHIVAL_CLASS],
    )
    .expect("peer class change");

    // Runs after the walk has read the due take and before the sweep
    // asks for the write lock.
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        peer.execute_batch("COMMIT").expect("peer commit");
    })));
    let report = store
        .apply_retention_policy(
            &policy(
                STANDARD_CLASS,
                ClassLimits {
                    max_age_days: None,
                    max_total_bytes: Some(limit),
                },
            ),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert!(report.retired.is_empty(), "{report:?}");
    assert!(report.over_limit.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&older).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(
        store.audio_at_rest(&newer).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_take_a_peer_moved_away_is_never_removed_later_in_the_walk() {
    // Three takes under a limit that fits one. While the sweep waits to
    // retire the middle one, a peer moves the oldest into the archival
    // class: the sweep reaches it after its recount and must still not
    // remove it under the standard limit.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let oldest = aged_take(&mut store, 150, 16_000);
    let middle = aged_take(&mut store, 120, 16_000);
    let newest = aged_take(&mut store, 90, 16_000);
    let limit = store.audio_bytes(&newest);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "UPDATE captures SET retention_class = ?2 WHERE id = ?1",
        params![oldest, ARCHIVAL_CLASS],
    )
    .expect("peer class change");

    // Runs after the walk has read the due take and before the sweep
    // asks for the write lock.
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        peer.execute_batch("COMMIT").expect("peer commit");
    })));
    let report = store
        .apply_retention_policy(
            &policy(
                STANDARD_CLASS,
                ClassLimits {
                    max_age_days: None,
                    max_total_bytes: Some(limit),
                },
            ),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![middle]);
    assert_eq!(
        store.audio_at_rest(&oldest).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(
        store.audio_at_rest(&newest).expect("state"),
        AudioAtRest::Journal
    );
}

/// Two takes for the compression races: `older` and `newer` are both
/// journals, `newer` has a verified FLAC prepared, and the size limit is
/// exactly `newer`'s FLAC plus `older`'s journal — so the class fits once
/// `newer` is counted at its FLAC's size, and is over the limit while it
/// is counted at its journal's.
struct CompressionRace {
    older: String,
    newer: String,
    prepared: PreparedCompression,
    limit: u64,
}

fn compression_race(store: &mut StoreV2) -> CompressionRace {
    let older = aged_take(store, 120, 16_000);
    let newer = aged_take(store, 90, 16_000);
    let job = store
        .compression_candidates(10)
        .expect("candidates")
        .into_iter()
        .find(|job| job.id == newer)
        .expect("newer is a candidate");
    let prepared = prepare_compression(&job).expect("prepare");
    let limit = prepared.flac_bytes + store.audio_bytes(&older);
    assert!(store.audio_bytes(&newer) + store.audio_bytes(&older) > limit);
    CompressionRace {
        older,
        newer,
        prepared,
        limit,
    }
}

fn size_limit(limit: u64) -> RetentionPolicy {
    policy(
        STANDARD_CLASS,
        ClassLimits {
            max_age_days: None,
            max_total_bytes: Some(limit),
        },
    )
}

#[test]
fn a_peer_compression_mid_sweep_is_counted_at_its_new_size() {
    // The sweep counted the newer take as its journal; a peer then
    // compresses it, holding the write lock across its publish as
    // `commit_compression` does. With the FLAC's size the class fits, so
    // the older take keeps its audio.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let race = compression_race(&mut store);
    let (journal, flac) = (store.audio_path(&race.newer), store.flac_path(&race.newer));
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    let temp = race.prepared.temp.clone();
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        peer.execute_batch("BEGIN IMMEDIATE")
            .expect("peer write lock");
        std::fs::rename(&temp, &flac).expect("publish");
        std::fs::remove_file(&journal).expect("journal unlink");
        peer.execute_batch("COMMIT").expect("peer commit");
    })));
    let report = store
        .apply_retention_policy(&size_limit(race.limit), time::OffsetDateTime::now_utc())
        .expect("apply");
    assert!(report.retired.is_empty(), "{report:?}");
    assert!(report.over_limit.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&race.older).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(
        store.audio_at_rest(&race.newer).expect("state"),
        AudioAtRest::Flac
    );
}

#[test]
fn a_compression_whose_transaction_rolled_back_is_counted_at_its_files() {
    // A peer's compression publishes the FLAC and unlinks the journal,
    // then its transaction rolls back (a failed commit): the files
    // changed and no commit says so. The sweep must still count the
    // newer take at its FLAC's size: the class total is exactly the
    // limit, and the older take keeps its audio.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let race = compression_race(&mut store);
    let (journal, flac) = (store.audio_path(&race.newer), store.flac_path(&race.newer));
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    let temp = race.prepared.temp.clone();
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        peer.execute_batch("BEGIN IMMEDIATE")
            .expect("peer write lock");
        std::fs::rename(&temp, &flac).expect("publish");
        std::fs::remove_file(&journal).expect("journal unlink");
        peer.execute_batch("ROLLBACK").expect("peer rollback");
    })));
    let report = store
        .apply_retention_policy(&size_limit(race.limit), time::OffsetDateTime::now_utc())
        .expect("apply");
    assert!(report.retired.is_empty(), "{report:?}");
    assert!(report.over_limit.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&race.older).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(files(&store, &race.newer), (false, true));
}

#[test]
fn a_compressor_that_died_after_publishing_is_counted_at_its_flac() {
    // The peer compressor dies after the FLAC's rename, before its journal
    // unlink and its commit: both files are on disk, nothing committed.
    // The FLAC counts (the smaller of the two), so the class fits; the
    // next reconcile completes the compression.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let race = compression_race(&mut store);
    let flac = store.flac_path(&race.newer);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    let temp = race.prepared.temp.clone();
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        peer.execute_batch("BEGIN IMMEDIATE")
            .expect("peer write lock");
        std::fs::rename(&temp, &flac).expect("publish");
        // The process dies: its connection closes without a commit.
        drop(peer);
    })));
    let report = store
        .apply_retention_policy(&size_limit(race.limit), time::OffsetDateTime::now_utc())
        .expect("apply");
    assert!(report.retired.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&race.older).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(files(&store, &race.newer), (true, true));
    let reconciled = store.reconcile().expect("reconcile");
    assert_eq!(reconciled.completed_compressions, vec![race.newer.clone()]);
    assert_eq!(files(&store, &race.newer), (false, true));
}

#[test]
fn a_compression_that_shrank_a_take_mid_sweep_still_lets_a_real_excess_go() {
    // The re-measure removes only what is no longer due: when the class is
    // over the limit even at the FLAC's size, the older take still goes.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let race = compression_race(&mut store);
    let (journal, flac) = (store.audio_path(&race.newer), store.flac_path(&race.newer));
    let temp = race.prepared.temp.clone();
    store.before_retention_lock = Some(TestHook(Box::new(move || {
        std::fs::rename(&temp, &flac).expect("publish");
        std::fs::remove_file(&journal).expect("journal unlink");
    })));
    let report = store
        .apply_retention_policy(
            &size_limit(race.limit - 1),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![race.older]);
}

/// A policy whose drop records whether `id`'s retirement is already
/// visible to another connection.
struct ObservedPolicy<'a> {
    policy: RetentionPolicy,
    peer: &'a Connection,
    id: &'a str,
    seen: &'a std::cell::RefCell<Vec<bool>>,
}

impl std::borrow::Borrow<RetentionPolicy> for ObservedPolicy<'_> {
    fn borrow(&self) -> &RetentionPolicy {
        &self.policy
    }
}

impl Drop for ObservedPolicy<'_> {
    fn drop(&mut self) {
        let retired = self
            .peer
            .query_row(
                "SELECT 1 FROM tombstones WHERE id = ?1",
                params![format!("{AUDIO_TOMBSTONE_PREFIX}{}", self.id)],
                |_| Ok(()),
            )
            .optional()
            .expect("peer read")
            .is_some();
        self.seen.borrow_mut().push(retired);
    }
}

#[test]
fn the_policy_read_before_a_removal_is_held_until_the_removal_commits() {
    // A caller whose policy value holds the lock on its settings makes a
    // change to them wait until the removal it was checked for is
    // committed: the value is dropped only after the stamp is visible.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    let seen = std::cell::RefCell::new(Vec::new());
    let limited = policy(STANDARD_CLASS, age(30));
    let report = store
        .apply_live_retention_policy(
            || ObservedPolicy {
                policy: limited.clone(),
                peer: &peer,
                id: &id,
                seen: &seen,
            },
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![id.clone()]);
    // The start-of-run read, then the read for the removal.
    assert_eq!(*seen.borrow(), vec![false, true]);
}

#[test]
fn a_delete_waits_for_the_write_lock_before_moving_any_file() {
    // A compression on another connection publishes under the write lock;
    // a delete that moved files without it could quarantine the journal
    // and miss the FLAC published a moment later.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(16_000, 5));
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    store
        .conn
        .busy_timeout(std::time::Duration::from_millis(50))
        .expect("short busy timeout");
    assert!(store.delete_capture(&id).is_err());
    assert_eq!(files(&store, &id), (true, false));
    assert!(store.get_capture(&id).expect("row").is_some());
    peer.execute_batch("ROLLBACK").expect("peer rollback");
    store.delete_capture(&id).expect("delete");
    assert_eq!(files(&store, &id), (false, false));
    assert!(store.get_capture(&id).expect("row").is_none());
}

