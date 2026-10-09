//! Service lifecycle tests over a recorded in-memory backend, plus the
//! `pactl` output parsers. Nothing here shells out.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::*;

/// Starts with one device, `speakers`, stereo at [`START`], unmuted, as
/// the default.
struct FakeBackend {
    default_device: Mutex<String>,
    /// Devices missing from the map are gone.
    devices: Mutex<HashMap<String, OutputSnapshot>>,
    calls: Mutex<Vec<Call>>,
    unsupported: Mutex<Option<String>>,
    /// How many upcoming `apply` calls fail without changing anything.
    failing_applies: Mutex<u32>,
    /// How many upcoming `apply` calls change the device, then fail.
    lost_acks: Mutex<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    ActiveOutput,
    Output(String),
    Apply(String, OutputChange),
}

impl FakeBackend {
    fn new() -> Arc<FakeBackend> {
        let backend = FakeBackend {
            default_device: Mutex::new("speakers".to_string()),
            devices: Mutex::default(),
            calls: Mutex::default(),
            unsupported: Mutex::default(),
            failing_applies: Mutex::new(0),
            lost_acks: Mutex::new(0),
        };
        backend.add_device("speakers", &[START, START], false);
        Arc::new(backend)
    }

    fn add_device(&self, id: &str, volumes: &[u32], muted: bool) {
        self.devices.lock().unwrap().insert(
            id.to_string(),
            OutputSnapshot {
                device: id.to_string(),
                volumes: volumes.to_vec(),
                muted,
            },
        );
    }

    /// A manual change made outside Starling.
    fn user_sets_volumes(&self, id: &str, volumes: &[u32]) {
        self.change(id, &volume_change(volumes));
    }

    fn user_sets_muted(&self, id: &str, muted: bool) {
        self.change(id, &mute_change(muted));
    }

    fn change(&self, id: &str, change: &OutputChange) -> bool {
        let mut devices = self.devices.lock().unwrap();
        let Some(device) = devices.get_mut(id) else {
            return false;
        };
        if let Some(volumes) = &change.volumes {
            device.volumes = volumes.clone();
        }
        if let Some(muted) = change.set_muted {
            device.muted = muted;
        }
        true
    }

    fn device(&self, id: &str) -> OutputSnapshot {
        self.devices.lock().unwrap()[id].clone()
    }

    /// The calls since the last `take_calls`.
    fn take_calls(&self) -> Vec<Call> {
        std::mem::take(&mut *self.calls.lock().unwrap())
    }

    fn lookup(&self, id: &str) -> Result<OutputSnapshot, PlaybackError> {
        if let Some(reason) = self.unsupported.lock().unwrap().clone() {
            return Err(PlaybackError::Unsupported(reason));
        }
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| PlaybackError::NoSuchDevice(format!("{id} is gone")))
    }

    fn take_one(counter: &Mutex<u32>) -> bool {
        let mut counter = counter.lock().unwrap();
        let hit = *counter > 0;
        *counter = counter.saturating_sub(1);
        hit
    }
}

impl PlaybackBackend for FakeBackend {
    fn active_output(&self) -> Result<OutputSnapshot, PlaybackError> {
        self.calls.lock().unwrap().push(Call::ActiveOutput);
        let id = self.default_device.lock().unwrap().clone();
        self.lookup(&id)
    }

    fn output(&self, id: &str) -> Result<OutputSnapshot, PlaybackError> {
        self.calls
            .lock()
            .unwrap()
            .push(Call::Output(id.to_string()));
        self.lookup(id)
    }

    fn apply(&self, id: &str, change: &OutputChange) -> Result<(), PlaybackError> {
        if Self::take_one(&self.failing_applies) {
            return Err(PlaybackError::Backend("refused".to_string()));
        }
        self.calls
            .lock()
            .unwrap()
            .push(Call::Apply(id.to_string(), change.clone()));
        if !self.change(id, change) {
            return Err(PlaybackError::NoSuchDevice(format!("{id} is gone")));
        }
        if Self::take_one(&self.lost_acks) {
            return Err(PlaybackError::Backend("timed out".to_string()));
        }
        Ok(())
    }
}

/// About 70%, deliberately not a whole percent.
const START: u32 = 45900;

fn raw(percent: u32) -> u32 {
    percent * VOLUME_NORM / 100
}

fn start() -> (Arc<FakeBackend>, PlaybackAttenuation, PlaybackHandle) {
    let backend = FakeBackend::new();
    let service = PlaybackAttenuation::start(backend.clone());
    let handle = service.handle();
    (backend, service, handle)
}

fn mute() -> PlaybackSettings {
    PlaybackSettings {
        during_recording: PlaybackMode::Mute,
        ..PlaybackSettings::default()
    }
}

fn lower(percent: u8) -> PlaybackSettings {
    PlaybackSettings {
        during_recording: PlaybackMode::Lower,
        lower_level_percent: percent,
    }
}

fn volume_change(volumes: &[u32]) -> OutputChange {
    OutputChange {
        volumes: Some(volumes.to_vec()),
        ..OutputChange::default()
    }
}

fn mute_change(muted: bool) -> OutputChange {
    OutputChange {
        set_muted: Some(muted),
        ..OutputChange::default()
    }
}

fn read() -> Call {
    Call::Output("speakers".to_string())
}

fn set_volumes(volumes: &[u32]) -> Call {
    Call::Apply("speakers".to_string(), volume_change(volumes))
}

fn set_muted(muted: bool) -> Call {
    Call::Apply("speakers".to_string(), mute_change(muted))
}

fn notice_kinds(handle: &PlaybackHandle) -> Vec<NoticeKind> {
    handle
        .take_notices()
        .iter()
        .map(|notice| notice.kind)
        .collect()
}

#[test]
fn off_touches_nothing() {
    let (backend, mut service, handle) = start();
    let lease = handle.begin(&PlaybackSettings::default());
    handle.end(lease);
    service.shutdown();
    assert_eq!(backend.take_calls(), []);
}

#[test]
fn mute_mutes_and_restores() {
    let (backend, _service, handle) = start();
    let lease = handle.begin(&mute());
    handle.flush();
    assert_eq!(backend.take_calls(), [Call::ActiveOutput, set_muted(true)]);

    handle.end(lease);
    handle.flush();
    assert_eq!(backend.take_calls(), [read(), set_muted(false)]);
    assert!(!backend.device("speakers").muted);
}

#[test]
fn lower_caps_each_channel_and_restores_the_exact_snapshot() {
    let (backend, _service, handle) = start();
    backend.user_sets_volumes("speakers", &[0, START]);
    let lease = handle.begin(&lower(30));
    handle.end(lease);
    handle.flush();
    assert_eq!(
        backend.take_calls(),
        [
            Call::ActiveOutput,
            set_volumes(&[0, raw(30)]),
            read(),
            set_volumes(&[0, START]),
        ]
    );
}

#[test]
fn a_user_change_within_the_same_percent_is_kept() {
    let (backend, _service, handle) = start();
    let lease = handle.begin(&lower(30));
    handle.flush();
    backend.user_sets_volumes("speakers", &[raw(30) + 140, raw(30) + 140]);
    handle.end(lease);
    handle.flush();
    assert_eq!(
        backend.device("speakers").volumes,
        [raw(30) + 140, raw(30) + 140]
    );
    assert_eq!(notice_kinds(&handle), []);
}

/// The volume changed under a muted, lowered take; the mute still restores
/// and the volume is reported, never rewritten.
fn changed_volume_under_mute_and_lower(changed: &[u32]) -> (Vec<Call>, Vec<NoticeKind>) {
    let (backend, _service, handle) = start();
    let first = handle.begin(&mute());
    let second = handle.begin(&lower(30));
    handle.flush();
    backend.user_sets_volumes("speakers", changed);
    backend.take_calls();
    handle.end(first);
    handle.end(second);
    handle.flush();
    assert_eq!(backend.device("speakers").volumes, changed);
    (backend.take_calls(), notice_kinds(&handle))
}

#[test]
fn a_partly_changed_volume_is_left_alone_and_reported() {
    assert_eq!(
        changed_volume_under_mute_and_lower(&[raw(25), raw(30)]),
        (
            vec![read(), set_muted(false)],
            vec![NoticeKind::RestoreFailed]
        )
    );
}

#[test]
fn a_changed_channel_count_is_left_alone_and_reported() {
    // Shrinking and growing alike.
    for changed in [&[raw(30)][..], &[raw(30), raw(30), raw(30)]] {
        assert_eq!(
            changed_volume_under_mute_and_lower(changed),
            (
                vec![read(), set_muted(false)],
                vec![NoticeKind::RestoreFailed]
            )
        );
    }
}

#[test]
fn an_already_muted_output_is_never_unmuted() {
    let (backend, _service, handle) = start();
    backend.user_sets_muted("speakers", true);
    let lease = handle.begin(&mute());
    handle.end(lease);
    handle.flush();
    assert_eq!(backend.take_calls(), [Call::ActiveOutput]);
    assert!(backend.device("speakers").muted);
}

#[test]
fn a_volume_at_or_below_the_level_is_untouched() {
    let (backend, _service, handle) = start();
    backend.user_sets_volumes("speakers", &[raw(15), raw(30)]);
    let lease = handle.begin(&lower(30));
    handle.end(lease);
    handle.flush();
    assert_eq!(backend.take_calls(), [Call::ActiveOutput]);
    assert_eq!(backend.device("speakers").volumes, [raw(15), raw(30)]);
}

#[test]
fn every_take_discovers_the_output_afresh() {
    let (backend, _service, handle) = start();
    for _ in 0..3 {
        let lease = handle.begin(&mute());
        handle.end(lease);
    }
    handle.flush();
    let cycle = [
        Call::ActiveOutput,
        set_muted(true),
        read(),
        set_muted(false),
    ];
    assert_eq!(
        backend.take_calls(),
        [cycle.clone(), cycle.clone(), cycle].concat()
    );
}

#[test]
fn overlapping_takes_share_one_snapshot_and_the_last_end_restores() {
    let (backend, _service, handle) = start();
    let first = handle.begin(&mute());
    let second = handle.begin(&mute());
    handle.flush();
    // The second take re-reads but neither re-snapshots nor re-mutes.
    assert_eq!(
        backend.take_calls(),
        [Call::ActiveOutput, set_muted(true), read()]
    );

    handle.end(first);
    handle.flush();
    assert_eq!(backend.take_calls(), []);
    assert!(backend.device("speakers").muted);

    handle.end(second);
    handle.flush();
    assert_eq!(backend.take_calls(), [read(), set_muted(false)]);
}

#[test]
fn a_stale_end_does_not_restore_a_newer_take() {
    let (backend, _service, handle) = start();
    let first = handle.begin(&mute());
    handle.end(first);
    let second = handle.begin(&mute());
    handle.flush();
    backend.take_calls();

    handle.end(first);
    handle.flush();
    assert_eq!(backend.take_calls(), []);
    assert!(backend.device("speakers").muted);

    handle.end(second);
    handle.flush();
    assert_eq!(backend.take_calls(), [read(), set_muted(false)]);
}

#[test]
fn overlapping_modes_combine_to_the_quieter_outcome() {
    let (backend, _service, handle) = start();
    let first = handle.begin(&mute());
    let second = handle.begin(&lower(30));
    handle.end(first);
    handle.flush();
    assert_eq!(
        backend.take_calls(),
        [
            Call::ActiveOutput,
            set_muted(true),
            read(),
            set_volumes(&[raw(30), raw(30)]),
        ]
    );
    assert!(
        backend.device("speakers").muted,
        "the second take still owns the mute"
    );

    handle.end(second);
    handle.flush();
    assert_eq!(
        backend.take_calls(),
        [
            read(),
            Call::Apply(
                "speakers".to_string(),
                OutputChange {
                    volumes: Some(vec![START, START]),
                    set_muted: Some(false),
                }
            ),
        ]
    );
}

#[test]
fn an_observed_volume_change_is_never_written_again() {
    let (backend, _service, handle) = start();
    let first = handle.begin(&lower(40));
    handle.flush();
    backend.user_sets_volumes("speakers", &[raw(25), raw(25)]);
    let second = handle.begin(&lower(20));
    handle.flush();
    // Back to exactly our earlier value: still the user's.
    backend.user_sets_volumes("speakers", &[raw(40), raw(40)]);
    let third = handle.begin(&lower(20));
    for lease in [first, second, third] {
        handle.end(lease);
    }
    handle.flush();
    assert_eq!(
        backend.take_calls(),
        [
            Call::ActiveOutput,
            set_volumes(&[raw(40), raw(40)]),
            read(),
            read(),
        ]
    );
    assert_eq!(backend.device("speakers").volumes, [raw(40), raw(40)]);
}

#[test]
fn an_observed_unmute_is_never_written_again() {
    let (backend, _service, handle) = start();
    let first = handle.begin(&mute());
    handle.flush();
    backend.user_sets_muted("speakers", false);
    let second = handle.begin(&mute());
    handle.flush();
    backend.user_sets_muted("speakers", true);
    handle.end(first);
    handle.end(second);
    handle.flush();
    assert_eq!(
        backend.take_calls(),
        [Call::ActiveOutput, set_muted(true), read()]
    );
    assert!(backend.device("speakers").muted);
}

#[test]
fn manual_changes_during_a_take_are_kept() {
    let (backend, _service, handle) = start();
    let lease = handle.begin(&lower(30));
    handle.flush();
    backend.user_sets_volumes("speakers", &[raw(55), raw(55)]);
    handle.end(lease);

    let lease = handle.begin(&mute());
    handle.flush();
    backend.user_sets_muted("speakers", false);
    handle.end(lease);
    handle.flush();

    // Each restore reads the device and writes nothing back.
    assert_eq!(
        backend.take_calls(),
        [
            Call::ActiveOutput,
            set_volumes(&[raw(30), raw(30)]),
            read(),
            Call::ActiveOutput,
            set_muted(true),
            read(),
        ]
    );
    assert_eq!(backend.device("speakers").volumes, [raw(55), raw(55)]);
    assert!(!backend.device("speakers").muted);
}

#[test]
fn a_manual_change_back_to_our_value_is_restored() {
    // Unobserved, it is indistinguishable from no change at all; stuck
    // attenuation is the worse failure.
    let (backend, _service, handle) = start();
    let lease = handle.begin(&mute());
    handle.flush();
    backend.user_sets_muted("speakers", false);
    backend.user_sets_muted("speakers", true);
    handle.end(lease);
    handle.flush();
    assert!(!backend.device("speakers").muted);
}

#[test]
fn a_write_that_landed_despite_an_error_is_restored() {
    let (backend, _service, handle) = start();
    *backend.lost_acks.lock().unwrap() = 2;
    let first = handle.begin(&mute());
    let second = handle.begin(&lower(30));
    handle.flush();
    assert_eq!(
        notice_kinds(&handle),
        [NoticeKind::AdjustFailed, NoticeKind::AdjustFailed]
    );
    handle.end(first);
    handle.end(second);
    handle.flush();
    let speakers = backend.device("speakers");
    assert_eq!(speakers.volumes, [START, START]);
    assert!(!speakers.muted);
}

#[test]
fn a_failed_write_keeps_the_earlier_one_restorable() {
    let (backend, _service, handle) = start();
    let first = handle.begin(&lower(40));
    handle.flush();
    *backend.failing_applies.lock().unwrap() = 1;
    let second = handle.begin(&lower(20));
    handle.end(first);
    handle.end(second);
    handle.flush();
    assert_eq!(backend.device("speakers").volumes, [START, START]);
}

#[test]
fn a_removed_output_restores_nothing_and_reports_it() {
    let (backend, _service, handle) = start();
    let lease = handle.begin(&mute());
    handle.flush();
    backend.devices.lock().unwrap().clear();
    backend.add_device("headphones", &[raw(50), raw(50)], false);
    *backend.default_device.lock().unwrap() = "headphones".to_string();
    backend.take_calls();

    handle.end(lease);
    handle.flush();
    assert_eq!(backend.take_calls(), [read()]);
    assert_eq!(notice_kinds(&handle), [NoticeKind::OutputRemoved]);
}

#[test]
fn a_switched_default_output_restores_the_original_device() {
    let (backend, _service, handle) = start();
    let lease = handle.begin(&mute());
    handle.flush();
    backend.add_device("headphones", &[raw(50), raw(50)], false);
    *backend.default_device.lock().unwrap() = "headphones".to_string();

    handle.end(lease);
    handle.flush();
    assert!(!backend.device("speakers").muted);
    assert_eq!(backend.device("headphones").volumes, [raw(50), raw(50)]);
}

#[test]
fn a_failing_restore_is_retried() {
    let (backend, _service, handle) = start();
    let lease = handle.begin(&mute());
    handle.flush();
    *backend.failing_applies.lock().unwrap() = RESTORE_ATTEMPTS - 1;
    handle.end(lease);
    handle.flush();
    assert!(!backend.device("speakers").muted);
    assert_eq!(notice_kinds(&handle), []);
}

#[test]
fn a_restore_that_keeps_failing_tells_the_user() {
    let (backend, _service, handle) = start();
    let lease = handle.begin(&mute());
    handle.flush();
    *backend.failing_applies.lock().unwrap() = RESTORE_ATTEMPTS;
    handle.end(lease);
    handle.flush();
    let notices = handle.take_notices();
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].kind, NoticeKind::RestoreFailed);
    assert!(notices[0].message.contains("speakers"), "{notices:?}");
}

#[test]
fn a_failed_adjustment_is_reported_and_nothing_is_restored() {
    let (backend, _service, handle) = start();
    *backend.failing_applies.lock().unwrap() = 1;
    let lease = handle.begin(&mute());
    handle.end(lease);
    handle.flush();
    assert_eq!(notice_kinds(&handle), [NoticeKind::AdjustFailed]);
    assert_eq!(backend.take_calls(), [Call::ActiveOutput, read()]);
}

#[test]
fn shutdown_restores_a_live_take() {
    let (backend, mut service, handle) = start();
    let _lease = handle.begin(&mute());
    handle.flush();
    service.shutdown();
    assert!(!backend.device("speakers").muted);
}

#[test]
fn an_unsupported_system_is_reported() {
    let (backend, _service, handle) = start();
    *backend.unsupported.lock().unwrap() = Some("pactl not found".to_string());
    let lease = handle.begin(&mute());
    handle.end(lease);
    handle.flush();
    assert_eq!(notice_kinds(&handle), [NoticeKind::AdjustFailed]);
    assert_eq!(
        handle.unsupported_reason().as_deref(),
        Some("pactl not found")
    );
    assert_eq!(backend.take_calls(), [Call::ActiveOutput]);
}

#[test]
fn undrained_notices_are_bounded() {
    let (backend, _service, handle) = start();
    *backend.unsupported.lock().unwrap() = Some("none".to_string());
    for _ in 0..NOTICE_LIMIT + 5 {
        handle.begin(&mute());
    }
    handle.flush();
    assert_eq!(handle.take_notices().len(), NOTICE_LIMIT);
}

#[test]
fn parses_the_default_sink() {
    let info = "\
Server Name: PulseAudio (on PipeWire 0.3.85)
Default Sink: alsa_output.pci-0000_00_1f.3.analog-stereo
Default Source: alsa_input.pci-0000_00_1f.3.analog-stereo
";
    assert_eq!(
        parse_default_sink(info).as_deref(),
        Some("alsa_output.pci-0000_00_1f.3.analog-stereo")
    );
    assert_eq!(
        parse_default_sink("Default Source: alsa_input.pci-0\n"),
        None
    );
    assert_eq!(parse_default_sink("Default Sink: \n"), None);
}

#[test]
fn parses_raw_channel_volumes() {
    let stereo = "\
Volume: front-left: 26214 /  40% / -24.00 dB,   front-right: 100270 / 153% / 11.00 dB
        balance 0.43
";
    assert_eq!(parse_raw_volumes(stereo), [26214, 100270]);
    assert_eq!(
        parse_raw_volumes("Volume: mono: 65536 / 100% / 0.00 dB\n"),
        [65536]
    );
    let surround = "Volume: front-left: 1 / 0% / -inf dB,   front-right: 2 / 0% / -inf dB,   \
                    rear-left: 3 / 0% / -inf dB,   rear-right: 4 / 0% / -inf dB,   \
                    front-center: 5 / 0% / -inf dB,   lfe: 6 / 0% / -inf dB\n";
    assert_eq!(parse_raw_volumes(surround), [1, 2, 3, 4, 5, 6]);
    assert!(parse_raw_volumes("Volume: / % / dB").is_empty());
    assert!(parse_raw_volumes("").is_empty());
}

#[test]
fn parses_mute() {
    assert_eq!(parse_mute("Mute: yes\n"), Some(true));
    assert_eq!(parse_mute("Mute: no\n"), Some(false));
    assert_eq!(parse_mute("Mute: maybe\n"), None);
    assert_eq!(parse_mute(""), None);
}
