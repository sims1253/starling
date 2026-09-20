//! Capture probe: records N seconds through the same code path the app
//! uses and prints level, clipping, and gap statistics, to diagnose clipped
//! or lossy captures.
//! Run: cargo run -p starling-dictation --example capture_probe -- [seconds] [out.wav]

use std::time::Duration;

use starling_dictation::audio::encode_wav_16k;
use starling_dictation::recorder::{RecorderError, CLIP_THRESHOLD, start_recording};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seconds: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let out = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/probe-capture.wav".into());

    let handle = start_recording().expect("start recording");
    eprintln!("capturing: sample_rate={}", handle.sample_rate());
    std::thread::sleep(Duration::from_secs(seconds));

    // Mid-session health: the same accessors the app can poll while the
    // recorder is live. A device-side failure (E01/G01) and any ring
    // overflow gaps (G01) are typed state, never silent.
    if let Some(error) = handle.capture_error() {
        eprintln!("capture error: {error}");
    }
    let gaps = handle.gaps();
    if gaps.is_empty() {
        eprintln!("gaps: none (continuous take)");
    } else {
        for gap in &gaps {
            eprintln!(
                "gap: [{}, {}) — {} samples missing (survivors joined, flagged)",
                gap.start_sample,
                gap.end_sample,
                gap.missing_samples()
            );
        }
    }

    // The stop handshake (R09): Ok carries the pending samples; a wedged
    // callback degrades to QuiesceTimeout with everything acknowledged
    // still intact inside the error, so even that path keeps the take.
    let audio = match handle.stop() {
        Ok(audio) => audio,
        Err(RecorderError::QuiesceTimeout {
            acknowledged_samples,
            audio,
        }) => {
            eprintln!(
                "quiesce timeout: the callback never quiesced; {acknowledged_samples} \
                 acknowledged samples were salvaged"
            );
            audio
        }
        Err(err) => {
            eprintln!("capture failed: {err}");
            std::process::exit(1);
        }
    };

    let peak = audio.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let clipped = audio
        .samples
        .iter()
        .filter(|s| s.abs() >= CLIP_THRESHOLD)
        .count();
    let ratio = if audio.samples.is_empty() {
        0.0
    } else {
        clipped as f64 / audio.samples.len() as f64
    };
    let rms = if audio.samples.is_empty() {
        0.0
    } else {
        (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt()
    };
    // Same rounding as the library's clipping_warning: a rounded
    // percentage, not the truncated `100 * clipped / len` integer divide.
    eprintln!(
        "samples={} rate={} peak={peak:.4} rms={rms:.4} clipped={clipped} ({:.0}%)",
        audio.samples.len(),
        audio.sample_rate,
        (ratio * 100.0).round()
    );

    let wav = encode_wav_16k(&audio).expect("encode");
    std::fs::write(&out, wav).expect("write wav");
    eprintln!("wrote {out}");
}
