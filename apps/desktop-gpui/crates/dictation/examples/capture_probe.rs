//! Temporary capture probe: records N seconds via the same code path the app
//! uses and prints level statistics, to diagnose clipped captures.
//! Run: cargo run -p starling-dictation --example capture_probe -- [seconds] [out.wav]

use std::time::Duration;

use starling_dictation::audio::encode_wav_16k;
use starling_dictation::recorder::start_recording;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seconds: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let out = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/probe-capture.wav".into());

    let handle = start_recording().expect("start recording");
    eprintln!(
        "capturing: sample_rate={} fmt-path selected",
        handle.sample_rate()
    );
    std::thread::sleep(Duration::from_secs(seconds));
    let audio = handle.stop().expect("stop");

    let peak = audio.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let rms =
        (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt();
    let clipped = audio.samples.iter().filter(|s| s.abs() >= 0.999).count();
    eprintln!(
        "samples={} rate={} peak={:.4} rms={:.4} clipped={} ({}%)",
        audio.samples.len(),
        audio.sample_rate,
        peak,
        rms,
        clipped,
        100 * clipped / audio.samples.len()
    );

    let wav = encode_wav_16k(&audio).expect("encode");
    std::fs::write(&out, wav).expect("write wav");
    eprintln!("wrote {out}");
}
