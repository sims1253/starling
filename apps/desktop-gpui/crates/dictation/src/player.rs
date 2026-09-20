//! WAV playback, ported from the `<audio>` element behavior in
//! `apps/desktop/src/App.tsx` (transcript drawer playback control) — see
//! `apps/desktop-gpui/PORT.md`.
//!
//! One rodio output stream is kept alive for the [`Player`]'s lifetime;
//! `play` decodes the WAV into a fresh [`rodio::Sink`], replacing any current
//! playback (like assigning a new `src` to the audio element). `is_playing`
//! is an `AtomicBool` that a single long-lived watcher thread clears once
//! the current sink drains (R10: one thread for the player's lifetime, not
//! one per playback).

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

/// How often the watcher checks whether playback has drained. It only polls
/// while a sink is actually live; idle, it blocks on the wake channel.
const WATCHER_POLL: Duration = Duration::from_millis(50);

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct PlayerError(pub String);

/// Playback state shared between the public methods and the watcher thread.
///
/// `current` pairs each sink with the generation ("epoch") that started it so
/// a stale watcher can never clear `playing` for a newer playback: play/stop
/// bump the epoch first, and watchers exit as soon as their epoch is no
/// longer the one stored.
struct PlaybackState {
    current: Mutex<Option<(u64, rodio::Sink)>>,
    epoch: AtomicU64,
    playing: AtomicBool,
}

impl PlaybackState {
    fn lock_current(&self) -> MutexGuard<'_, Option<(u64, rodio::Sink)>> {
        // The lock is only ever held for short, panic-free sections; recover
        // from a poisoned lock rather than poisoning the player forever.
        self.current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The rodio handle plus the glue that keeps the output device open.
///
/// The cpal stream behind rodio is `!Send`, so it is created on — and owned
/// by — a dedicated owner thread in `Player::new`; only the (Send + Sync)
/// [`rodio::OutputStreamHandle`] crosses back. Everything stored here is
/// Send + Sync, which keeps [`Player`] movable into gpui state.
struct Inner {
    handle: rodio::OutputStreamHandle,
    state: Arc<PlaybackState>,
    /// Wakes the playback watcher when a new sink is published. The sender
    /// disconnects when the player drops, telling the watcher to exit.
    watcher_wake: mpsc::SyncSender<()>,
    /// Disconnects when the player drops, telling the owner thread to exit
    /// (which ends playback and releases the device).
    _shutdown: mpsc::Sender<()>,
}

/// Plays WAV bytes on the default output device.
pub struct Player {
    inner: Arc<Inner>,
}

impl Player {
    pub fn new() -> Result<Self, PlayerError> {
        let (handle_tx, handle_rx) = mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();
        std::thread::Builder::new()
            .name("starling-audio-owner".into())
            .spawn(move || {
                let (stream, handle) = match rodio::OutputStream::try_default() {
                    Ok(opened) => opened,
                    Err(err) => {
                        let _ = handle_tx.send(Err(PlayerError(format!(
                            "No audio output device is available: {err}"
                        ))));
                        return;
                    }
                };
                if handle_tx.send(Ok(handle)).is_err() {
                    return; // the player was dropped before it was used
                }
                let _stream = stream; // owned: keeps playback alive
                let _ = shutdown_rx.recv();
            })
            .map_err(|err| PlayerError(format!("Could not start the audio thread: {err}")))?;

        let handle = handle_rx
            .recv()
            .map_err(|_| {
                PlayerError("Audio thread exited before opening an output device.".to_string())
            })?
            .map_err(|err| PlayerError(err.0))?;

        // R10: the one and only playback watcher, alive for the player's
        // lifetime. It sleeps on the wake channel until a playback starts,
        // then polls at WATCHER_POLL until that sink drains.
        let state = Arc::new(PlaybackState {
            current: Mutex::new(None),
            epoch: AtomicU64::new(0),
            playing: AtomicBool::new(false),
        });
        let (watcher_wake, watcher_rx) = mpsc::sync_channel::<()>(1);
        let watcher_state = Arc::clone(&state);
        std::thread::Builder::new()
            .name("starling-playback".into())
            .spawn(move || watch_playback(watcher_state, watcher_rx))
            .map_err(|err| PlayerError(format!("Could not start the playback watcher: {err}")))?;

        Ok(Self {
            inner: Arc::new(Inner {
                handle,
                state,
                watcher_wake,
                _shutdown: shutdown_tx,
            }),
        })
    }

    /// Non-blocking: decodes the WAV, replaces any current playback and
    /// returns. Errors leave semantics simple — decoding happens before the
    /// current playback is touched, so a bad file is a no-op.
    pub fn play(&self, wav: &[u8]) -> Result<(), PlayerError> {
        let source = rodio::Decoder::new(Cursor::new(wav.to_vec()))
            .map_err(|err| PlayerError(format!("Could not decode WAV: {err}")))?;

        let epoch = self.inner.state.epoch.fetch_add(1, Ordering::Relaxed) + 1;

        // Replace whatever is playing, atomically: stop the old sink, build
        // the new one and publish it under a single lock hold so two `play`
        // calls can never interleave and orphan a sink.
        let mut current = self.inner.state.lock_current();
        if let Some((_, old)) = current.take() {
            old.stop();
        }
        let sink = match rodio::Sink::try_new(&self.inner.handle) {
            Ok(sink) => sink,
            Err(err) => {
                self.inner.state.playing.store(false, Ordering::Relaxed);
                return Err(PlayerError(format!("Could not start playback: {err}")));
            }
        };
        sink.append(source);
        self.inner.state.playing.store(true, Ordering::Relaxed);
        *current = Some((epoch, sink));
        drop(current);

        // Wake the watcher so it starts polling this sink for drain. A
        // token queued while it is already polling only costs one extra
        // idle check, so a lost/ignored token is harmless.
        let _ = self.inner.watcher_wake.try_send(());
        Ok(())
    }

    /// Stops the current playback, if any. Safe to call when idle.
    pub fn stop(&self) {
        self.inner.state.epoch.fetch_add(1, Ordering::Relaxed);
        if let Some((_, sink)) = self.inner.state.lock_current().take() {
            sink.stop();
        }
        self.inner.state.playing.store(false, Ordering::Relaxed);
    }

    /// True while a `play` is audible — from `play` until the sink drains or
    /// `stop`/`play` replaces it.
    pub fn is_playing(&self) -> bool {
        self.inner.state.playing.load(Ordering::Relaxed)
    }
}

/// Body of the player's single long-lived watcher thread (R10).
///
/// Idle, it blocks on `wake` — no polling, no wakeups. A published playback
/// wakes it; it then polls at `WATCHER_POLL` until the current sink drains
/// (clearing `playing` and dropping the sink, double-checked against the
/// epoch in case a newer playback replaced it mid-check), or until there is
/// nothing current (stopped/replaced), at which point it goes back to
/// sleep. The channel disconnecting (player dropped) ends the thread.
fn watch_playback(state: Arc<PlaybackState>, wake: mpsc::Receiver<()>) {
    while wake.recv().is_ok() {
        loop {
            std::thread::sleep(WATCHER_POLL);
            let drained_epoch = {
                let current = state.lock_current();
                match current.as_ref() {
                    // rodio 0.20: `Sink::empty()` (len() == 0), no `is_empty`.
                    Some((epoch, sink)) if sink.empty() => Some(*epoch),
                    // Still audible: poll again.
                    Some(_) => None,
                    // Stopped or replaced: back to sleep until the next wake.
                    None => break,
                }
            };

            let Some(epoch) = drained_epoch else {
                continue;
            };

            state.playing.store(false, Ordering::Relaxed);
            let mut current = state.lock_current();
            // A playback may have been published between the check and this
            // lock; only the generation we saw drain may be removed.
            if matches!(current.as_ref(), Some((active, _)) if *active == epoch) {
                *current = None;
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Canonical 44-byte-header WAV: PCM16 mono 16 kHz silence, the shape the
    /// app records in (`crate::audio::encode_wav_16k` output). Written by hand
    /// so the tests don't depend on the `audio` module.
    fn silence_wav_16k(duration_seconds: f64) -> Vec<u8> {
        const SAMPLE_RATE: u32 = 16_000;
        let num_samples = (SAMPLE_RATE as f64 * duration_seconds).round() as u32;
        let data_len = num_samples * 2; // PCM16

        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.resize(44 + data_len as usize, 0); // silence
        wav
    }

    fn wait_until_idle(player: &Player, limit: Duration) -> bool {
        let started = Instant::now();
        while player.is_playing() {
            if started.elapsed() > limit {
                return false;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        true
    }

    #[test]
    fn wav_helper_produces_canonical_header() {
        let wav = silence_wav_16k(0.5);
        assert_eq!(wav.len(), 44 + 16_000); // 8_000 samples * 2 bytes
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1); // PCM
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1); // mono
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16_000
        );
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16); // bits
        assert_eq!(&wav[36..40], b"data");
    }

    #[test]
    fn new_opens_the_default_output_device() {
        match Player::new() {
            Ok(player) => assert!(!player.is_playing()),
            Err(err) => {
                // Only skip when there is genuinely no output device.
                eprintln!("skipping: no audio output device: {err}");
            }
        }
    }

    #[test]
    fn play_silence_reports_playing_then_finishes() {
        // Headless CI has no output device: skip rather than panic (the
        // device-dependent behaviors are covered wherever audio exists).
        let Ok(player) = Player::new() else {
            eprintln!("skipping: no audio output device");

            return;
        };

        assert!(!player.is_playing());

        player.play(&silence_wav_16k(0.5)).expect("play silence");
        assert!(player.is_playing());

        assert!(
            wait_until_idle(&player, Duration::from_secs(10)),
            "playback should drain shortly after the 0.5s clip ends"
        );
        assert!(!player.is_playing());
    }

    #[test]
    fn play_replaces_current_playback() {
        // Headless CI has no output device: skip rather than panic (the
        // device-dependent behaviors are covered wherever audio exists).
        let Ok(player) = Player::new() else {
            eprintln!("skipping: no audio output device");

            return;
        };

        player.play(&silence_wav_16k(30.0)).expect("play long clip");
        player
            .play(&silence_wav_16k(0.25))
            .expect("play short clip");

        assert!(
            wait_until_idle(&player, Duration::from_secs(5)),
            "the short clip should replace the 30s one and finish quickly"
        );
    }

    #[test]
    fn stop_stops_playback_immediately() {
        // Headless CI has no output device: skip rather than panic (the
        // device-dependent behaviors are covered wherever audio exists).
        let Ok(player) = Player::new() else {
            eprintln!("skipping: no audio output device");

            return;
        };

        player.play(&silence_wav_16k(30.0)).expect("play long clip");
        player.stop();
        assert!(!player.is_playing());
    }

    #[test]
    fn play_rejects_bytes_that_are_not_wav() {
        // Headless CI has no output device: skip rather than panic (the
        // device-dependent behaviors are covered wherever audio exists).
        let Ok(player) = Player::new() else {
            eprintln!("skipping: no audio output device");

            return;
        };

        assert!(player.play(&[0u8; 16]).is_err());
        assert!(!player.is_playing());
    }
}
