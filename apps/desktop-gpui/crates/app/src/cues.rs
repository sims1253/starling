//! Optional start/stop cues (#221): two short tones, rising when the
//! microphone is really listening and falling when it has closed.
//!
//! - **The start cue tracks real audio.** It plays on the activation
//!   machine's `Listening` effect — the take's first captured samples —
//!   so a failed start, a take cancelled before audio arrived, or a
//!   late callback from an older take never plays it.
//! - **The stop cue closes what the start cue opened.** It plays only
//!   for a take whose start cue played, only after its capture has
//!   stopped (so it is never recorded into that take), and never once a
//!   newer take has started (whose microphone could hear it). A cue still
//!   sounding when a take starts is silenced before its microphone opens.
//! - **Attenuation (#361) does not swallow them.** Each cue waits until
//!   playback is as the user left it
//!   ([`PlaybackHandle::settled`](starling_dictation::playback::PlaybackHandle::settled):
//!   an earlier take's restore has been carried out and worked), and with
//!   a start cue to play, lowering or muting waits until it has played. A
//!   cue that cannot get there within [`SETTLE_LIMIT`] is dropped, and a
//!   dropped start cue drops its stop cue.
//!
//! The start cue plays while the microphone is open: through speakers
//! it can be heard in the first ~0.15 s of the take, which is where a
//! take is silent anyway. Headphones keep it out entirely. Keeping it out
//! of the captured audio itself would take the recorder's cooperation.
//!
//! The tones are synthesized once, in memory; there are no sound files.

use std::sync::OnceLock;
use std::time::Duration;

use gpui::{AppContext, Context, Timer};
use starling_dictation::settings::{FeedbackSettings, PlaybackMode};

use crate::activation::TakeId;
use crate::app::StarlingApp;

const SAMPLE_RATE: u32 = 44_100;

/// Each of a cue's two tones, and the pause between them.
const TONE: Duration = Duration::from_millis(60);
const PAUSE: Duration = Duration::from_millis(20);

/// How long a cue waits at most for playback to be as the user left it;
/// past it the cue is dropped.
const SETTLE_LIMIT: Duration = Duration::from_millis(1500);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cue {
    Start,
    Stop,
}

impl Cue {
    /// The two tones' frequencies, in Hz: up a fourth for start, down a
    /// fifth for stop.
    fn tones(self) -> [f32; 2] {
        match self {
            Cue::Start => [660., 880.],
            Cue::Stop => [880., 587.33],
        }
    }

    /// The cue's synthesized WAV, built on first use.
    pub(crate) fn wav(self) -> &'static [u8] {
        static START: OnceLock<Vec<u8>> = OnceLock::new();
        static STOP: OnceLock<Vec<u8>> = OnceLock::new();
        match self {
            Cue::Start => START.get_or_init(|| synthesize(Cue::Start)),
            Cue::Stop => STOP.get_or_init(|| synthesize(Cue::Stop)),
        }
    }
}

/// A cue's full length.
pub(crate) fn cue_length() -> Duration {
    TONE * 2 + PAUSE
}

/// The cue as a 16-bit mono WAV: two sine tones with 8 ms raised-cosine
/// fades (no clicks), peaking at half of full scale.
pub(crate) fn synthesize(cue: Cue) -> Vec<u8> {
    let rate = SAMPLE_RATE as f32;
    let tone_len = (TONE.as_secs_f32() * rate) as usize;
    let pause_len = (PAUSE.as_secs_f32() * rate) as usize;
    let fade_len = (0.008 * rate) as usize;
    let mut samples = Vec::with_capacity(tone_len * 2 + pause_len);
    for (index, frequency) in cue.tones().into_iter().enumerate() {
        if index > 0 {
            samples.extend(std::iter::repeat_n(0i16, pause_len));
        }
        for n in 0..tone_len {
            let edge = n.min(tone_len - 1 - n);
            let envelope = if edge < fade_len {
                0.5 - 0.5 * (std::f32::consts::PI * edge as f32 / fade_len as f32).cos()
            } else {
                1.
            };
            let value = (std::f32::consts::TAU * frequency * n as f32 / rate).sin();
            samples.push((value * envelope * 0.5 * f32::from(i16::MAX)) as i16);
        }
    }
    wav_bytes(&samples, SAMPLE_RATE)
}

fn wav_bytes(samples: &[i16], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

/// The playback gain for a volume setting: squared, so the slider feels
/// even to the ear.
pub(crate) fn cue_gain(percent: u8) -> f32 {
    let fraction = f32::from(percent.min(100)) / 100.;
    fraction * fraction
}

/// Whether attenuation waits for the start cue: only when there is a cue
/// to protect and something to attenuate.
pub(crate) fn attenuation_waits_for_cue(feedback: &FeedbackSettings, mode: PlaybackMode) -> bool {
    feedback.cues && mode != PlaybackMode::Off
}

/// Which cues may play, take by take. Pure.
#[derive(Debug, Default)]
pub(crate) struct CueGate {
    /// The take whose start cue waits for playback to settle.
    starting: Option<TakeId>,
    /// The take whose start cue played and whose stop cue has not.
    announced: Option<TakeId>,
    /// An ended take whose stop cue waits for playback to be restored.
    stop_pending: Option<TakeId>,
}

impl CueGate {
    /// A take is about to open its microphone: cues still waiting for an
    /// earlier take are dropped.
    pub(crate) fn started(&mut self) {
        self.starting = None;
        self.announced = None;
        self.stop_pending = None;
    }

    /// `take` reported its first samples while `active` is the live take.
    /// Returns whether a start cue is due once playback settles.
    pub(crate) fn listening(
        &mut self,
        take: TakeId,
        active: Option<TakeId>,
        enabled: bool,
    ) -> bool {
        if !enabled || active != Some(take) {
            return false;
        }
        self.starting = Some(take);
        true
    }

    /// Playback settled for `take`'s start cue (`enabled` is false when it
    /// did not settle in time or cues were turned off). Returns whether to
    /// play it now: only while `take` is still the live take.
    pub(crate) fn start_settled(
        &mut self,
        take: TakeId,
        active: Option<TakeId>,
        enabled: bool,
    ) -> bool {
        if self.starting != Some(take) {
            return false;
        }
        self.starting = None;
        if !enabled || active != Some(take) {
            return false;
        }
        self.announced = Some(take);
        true
    }

    /// `take` stopped recording (finished or cancelled). Returns whether a
    /// stop cue is due once playback settles.
    pub(crate) fn ended(&mut self, take: TakeId) -> bool {
        if self.starting == Some(take) {
            self.starting = None;
        }
        if self.announced != Some(take) {
            return false;
        }
        self.announced = None;
        self.stop_pending = Some(take);
        true
    }

    /// Playback settled after `take` ended. Returns whether to play its
    /// stop cue now: not if a newer take started meanwhile.
    pub(crate) fn settled(&mut self, take: TakeId, active: Option<TakeId>, enabled: bool) -> bool {
        if self.stop_pending != Some(take) {
            return false;
        }
        self.stop_pending = None;
        enabled && active.is_none()
    }
}

// ---- App glue -------------------------------------------------------------

impl StarlingApp {
    fn play_cue(&mut self, cue: Cue, volume_percent: u8) {
        let Some(player) = self.player.as_ref() else {
            return;
        };
        if let Err(err) = player.play_cue(cue.wav(), cue_gain(volume_percent)) {
            eprintln!("Could not play the {cue:?} cue: {err}");
        }
    }

    /// A take is about to open its microphone: no cue may sound into it.
    pub(crate) fn cue_take_starting(&mut self) {
        self.cue_gate.started();
        if let Some(player) = self.player.as_ref() {
            player.stop_cues();
        }
    }

    /// The take's first samples arrived: the start cue once playback has
    /// settled, then (when it waited for the cue) the attenuation.
    pub(crate) fn cue_listening(&mut self, take: TakeId, cx: &mut Context<Self>) {
        let due = self
            .cue_gate
            .listening(take, self.recording_take, self.feedback.cues);
        if !due {
            // Cues were turned off mid-take: nothing to wait for.
            self.begin_deferred_attenuation(take);
            return;
        }
        let settled = self.playback.handle().settled();
        cx.spawn(async move |this, cx| {
            let restored = cx
                .background_spawn(async move { settled.recv_timeout(SETTLE_LIMIT) == Ok(true) })
                .await;
            let played = this
                .update(cx, |app, _| {
                    let enabled = app.feedback.cues && restored;
                    let play = app
                        .cue_gate
                        .start_settled(take, app.recording_take, enabled);
                    if play {
                        app.play_cue(Cue::Start, app.feedback.cue_volume_percent);
                    }
                    play
                })
                .unwrap_or(false);
            if played {
                Timer::after(cue_length()).await;
            }
            this.update(cx, |app, _| app.begin_deferred_attenuation(take))
                .ok();
        })
        .detach();
    }

    /// The start cue has played (or will not): attenuation that waited for
    /// it begins, if `take` is still recording and nothing began it
    /// already.
    fn begin_deferred_attenuation(&mut self, take: TakeId) {
        if self.recording_take == Some(take) && self.playback_lease.is_none() {
            self.playback_lease = Some(self.playback.handle().begin(&self.playback_settings));
        }
    }

    /// The take stopped recording (its capture has ended): its stop cue
    /// follows once playback has been restored.
    pub(crate) fn cue_take_ended(&mut self, take: TakeId, cx: &mut Context<Self>) {
        if !self.cue_gate.ended(take) {
            return;
        }
        let settled = self.playback.handle().settled();
        cx.spawn(async move |this, cx| {
            // A restore that has not finished in time (or a service that
            // shut down) drops the cue: it would play into a lowered or
            // muted output, or long after the take.
            let restored = cx
                .background_spawn(async move { settled.recv_timeout(SETTLE_LIMIT).is_ok() })
                .await;
            this.update(cx, |app, _| {
                let enabled = app.feedback.cues && restored;
                if app
                    .cue_gate
                    .settled(take, app.activation.active_take(), enabled)
                {
                    app.play_cue(Cue::Stop, app.feedback.cue_volume_percent);
                }
            })
            .ok();
        })
        .detach();
    }

    /// The settings dialog's preview: both cues at the draft volume. Not
    /// while a take records: its microphone would hear them.
    pub(crate) fn preview_cues(&mut self, cx: &mut Context<Self>) {
        if self.recording_take.is_some() {
            return;
        }
        let volume = self.draft_cue_volume.read(cx).value();
        self.play_cue(Cue::Start, volume);
        cx.spawn(async move |this, cx| {
            Timer::after(cue_length() + Duration::from_millis(250)).await;
            this.update(cx, |app, _| {
                if app.recording_take.is_none() {
                    app.play_cue(Cue::Stop, volume);
                }
            })
            .ok();
        })
        .detach();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Take `take` is live, reached readiness, and its start cue played.
    fn heard(gate: &mut CueGate, take: TakeId) {
        gate.started();
        assert!(gate.listening(take, Some(take), true));
        assert!(gate.start_settled(take, Some(take), true));
    }

    #[test]
    fn the_start_cue_plays_once_and_only_for_the_live_take() {
        let mut gate = CueGate::default();
        gate.started();
        assert!(gate.listening(1, Some(1), true));
        assert!(gate.start_settled(1, Some(1), true));
        assert!(!gate.start_settled(1, Some(1), true), "played once");
        // A late callback from an older take, or one with no live take.
        assert!(!gate.listening(0, Some(1), true));
        assert!(!gate.listening(2, None, true));
        // Disabled cues never play.
        let mut gate = CueGate::default();
        assert!(!gate.listening(1, Some(1), false));
        assert!(!gate.start_settled(1, Some(1), true));
        assert!(!gate.ended(1), "no start cue, no stop cue");
    }

    #[test]
    fn a_failed_start_or_a_cancel_before_audio_plays_no_cue() {
        let mut gate = CueGate::default();
        // Take 1 failed to start: no Listening ever arrives.
        gate.started();
        assert!(!gate.ended(1));
        // Take 2 was cancelled before its first samples.
        gate.started();
        assert!(!gate.ended(2));
        assert!(!gate.settled(2, None, true));
    }

    #[test]
    fn a_take_that_ends_while_its_start_cue_waits_plays_neither_cue() {
        let mut gate = CueGate::default();
        gate.started();
        assert!(gate.listening(1, Some(1), true));
        // Cancelled before playback settled.
        assert!(!gate.ended(1));
        assert!(!gate.start_settled(1, None, true));
        assert!(!gate.settled(1, None, true));
    }

    #[test]
    fn a_start_cue_that_cannot_settle_is_dropped_with_its_stop_cue() {
        // An earlier take's restore did not finish in time (or failed):
        // the cue would play into a muted output.
        let mut gate = CueGate::default();
        gate.started();
        assert!(gate.listening(1, Some(1), true));
        assert!(!gate.start_settled(1, Some(1), false));
        assert!(!gate.ended(1));
    }

    #[test]
    fn the_start_cue_waits_for_its_own_take_only() {
        // Take 1's settle arrives after take 2 started: neither plays
        // take 1's cue, and take 2's own cue still can.
        let mut gate = CueGate::default();
        gate.started();
        assert!(gate.listening(1, Some(1), true));
        gate.started();
        assert!(!gate.start_settled(1, Some(2), true));
        assert!(gate.listening(2, Some(2), true));
        assert!(!gate.start_settled(1, Some(2), true));
        assert!(gate.start_settled(2, Some(2), true));
    }

    #[test]
    fn the_stop_cue_follows_a_heard_start_once_playback_settled() {
        let mut gate = CueGate::default();
        heard(&mut gate, 1);
        assert!(gate.ended(1));
        assert!(!gate.ended(1), "ending twice is one stop cue");
        assert!(gate.settled(1, None, true));
        assert!(!gate.settled(1, None, true), "played once");
    }

    #[test]
    fn a_cancelled_take_that_was_heard_still_closes_with_the_stop_cue() {
        // The microphone opened (the start cue said so) and has closed.
        let mut gate = CueGate::default();
        heard(&mut gate, 4);
        assert!(gate.ended(4));
        assert!(gate.settled(4, None, true));
    }

    #[test]
    fn a_rapid_re_press_drops_the_previous_stop_cue() {
        let mut gate = CueGate::default();
        heard(&mut gate, 1);
        assert!(gate.ended(1));
        // Take 2 starts before take 1's playback settled: its microphone
        // would hear the stop cue.
        heard(&mut gate, 2);
        assert!(!gate.settled(1, Some(2), true));
        // Even if take 2 already ended, take 1's stale settle stays quiet.
        assert!(gate.ended(2));
        assert!(!gate.settled(1, None, true));
        assert!(gate.settled(2, None, true));
    }

    #[test]
    fn a_stop_cue_is_skipped_when_a_take_is_live_or_cues_were_turned_off() {
        let mut gate = CueGate::default();
        heard(&mut gate, 1);
        assert!(gate.ended(1));
        assert!(!gate.settled(1, Some(2), true));

        let mut gate = CueGate::default();
        heard(&mut gate, 1);
        assert!(gate.ended(1));
        assert!(!gate.settled(1, None, false));
    }

    #[test]
    fn attenuation_waits_only_when_a_cue_would_be_attenuated() {
        let cues = FeedbackSettings {
            cues: true,
            ..FeedbackSettings::default()
        };
        let quiet = FeedbackSettings::default();
        assert!(attenuation_waits_for_cue(&cues, PlaybackMode::Mute));
        assert!(attenuation_waits_for_cue(&cues, PlaybackMode::Lower));
        assert!(!attenuation_waits_for_cue(&cues, PlaybackMode::Off));
        assert!(!attenuation_waits_for_cue(&quiet, PlaybackMode::Mute));
    }

    #[test]
    fn the_cues_are_short_click_free_wavs_that_differ() {
        for cue in [Cue::Start, Cue::Stop] {
            let wav = synthesize(cue);
            assert_eq!(&wav[..4], b"RIFF");
            assert_eq!(&wav[8..16], b"WAVEfmt ");
            let samples: Vec<i16> = wav[44..]
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
            assert!(
                (seconds - cue_length().as_secs_f32()).abs() < 0.002,
                "{seconds}"
            );
            // Faded in and out: silent edges, and never above half scale.
            assert!(samples.first().unwrap().abs() < 50 && samples.last().unwrap().abs() < 50);
            let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap();
            assert!(peak > 10_000 && peak <= (i16::MAX / 2) as u16 + 1, "{peak}");
        }
        assert_ne!(synthesize(Cue::Start), synthesize(Cue::Stop));
        assert_eq!(Cue::Start.wav(), synthesize(Cue::Start).as_slice());
    }

    #[test]
    fn the_volume_curve_is_squared_and_clamped() {
        assert_eq!(cue_gain(0), 0.);
        assert_eq!(cue_gain(100), 1.);
        assert!((cue_gain(50) - 0.25).abs() < 1e-6);
        assert_eq!(cue_gain(250), 1.);
    }
}
