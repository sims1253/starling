//! WAV playback, ported from the `<audio>` element behavior in
//! `apps/desktop/src/App.tsx` (transcript drawer playback control) — see
//! `apps/desktop-gpui/PORT.md`.
//!
//! One rodio output stream is kept alive for the [`Player`]'s lifetime;
//! `play` decodes the WAV into a fresh [`rodio::Sink`], replacing any current
//! playback (like assigning a new `src` to the audio element). `is_playing`
//! is an `AtomicBool` that a watcher thread clears once the sink drains.

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

/// How often the watcher checks whether playback has drained.
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

        Ok(Self {
            inner: Arc::new(Inner {
                handle,
                state: Arc::new(PlaybackState {
                    current: Mutex::new(None),
                    epoch: AtomicU64::new(0),
                    playing: AtomicBool::new(false),
                }),
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

        // Watcher thread: clears `playing` once this generation's sink drains.
        let state = Arc::clone(&self.inner.state);
        if let Err(err) = std::thread::Builder::new()
            .name("starling-playback".into())
            .spawn(move || watch_until_drained(state, epoch))
        {
            // No watcher means `is_playing` would never settle — stop rather
            // than leave a stuck flag behind.
            self.stop();
            return Err(PlayerError(format!(
                "Could not start playback thread: {err}"
            )));
        }
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

/// Body of the per-playback watcher thread. Exits as soon as `epoch` is no
/// longer the active generation (replaced or stopped); otherwise clears the
/// playing flag and drops the drained sink.
fn watch_until_drained(state: Arc<PlaybackState>, epoch: u64) {
    loop {
        std::thread::sleep(WATCHER_POLL);
        let current = state.lock_current();
        match current.as_ref() {
            Some((active, sink)) if *active == epoch => {
                // rodio 0.20: `Sink::empty()` (len() == 0), no `is_empty`.
                if sink.empty() {
                    drop(current);
                    state.playing.store(false, Ordering::Relaxed);
                    let mut current = state.lock_current();
                    if matches!(current.as_ref(), Some((active, _)) if *active == epoch) {
                        *current = None;
                    }
                    return;
                }
            }
            // Replaced by a newer playback, stopped, or not ours anymore.
            _ => return,
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
        let player = Player::new().expect("player with output device");
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
        let player = Player::new().expect("player with output device");
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
        let player = Player::new().expect("player with output device");
        player.play(&silence_wav_16k(30.0)).expect("play long clip");
        player.stop();
        assert!(!player.is_playing());
    }

    #[test]
    fn play_rejects_bytes_that_are_not_wav() {
        let player = Player::new().expect("player with output device");
        assert!(player.play(&[0u8; 16]).is_err());
        assert!(!player.is_playing());
    }
}
