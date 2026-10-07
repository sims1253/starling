//! Microphone selection probe (#222): lists the capture devices the host
//! reports, then records a short take through the app's own start path
//! with the given preference and prints the route it resolved to, the raw
//! level, and any start problem — the evidence behind the settings
//! picker's preferred/active/fallback display.
//! Run: cargo run -p starling-dictation --example mic_probe -- [preferred-device|-] [seconds]

use std::time::Duration;

use starling_dictation::microphone::{list_input_devices, SignalLevel};
use starling_dictation::recorder::{start_capture, CaptureRequest};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let preferred = args.get(1).filter(|arg| arg.as_str() != "-").cloned();
    let seconds: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.5);

    match list_input_devices() {
        Ok(devices) => {
            println!("devices: {}", devices.len());
            for device in &devices {
                let marker = if device.is_default { " (default)" } else { "" };
                println!("  - {}{marker}", device.name);
            }
        }
        Err(err) => println!("listing failed: {err}"),
    }

    println!(
        "preferred: {}",
        preferred.as_deref().unwrap_or("<follow system default>")
    );
    let handle = match start_capture(CaptureRequest {
        journals_dir: None,
        preferred_device: preferred.as_deref(),
    }) {
        Ok(handle) => handle,
        Err(err) => {
            println!("start failed: {:?}", err.problem);
            println!("  message:  {}", err.problem.message());
            println!("  recovery: {}", err.problem.recovery());
            if let Some(route) = err.route {
                println!("  route: {route:?}");
            }
            return;
        }
    };
    let route = handle.input_route().cloned();
    println!("route: {route:?}");
    if let Some(notice) = route.as_ref().and_then(|route| route.notice()) {
        println!("notice: {notice}");
    }
    std::thread::sleep(Duration::from_secs_f64(seconds));
    println!("stalled for: {:?}", handle.input_stalled_for());
    println!("source peak: {:.5}", handle.source_peak());
    if let Some(fault) = handle.capture_fault() {
        println!("fault: {fault:?} (fatal: {})", fault.is_fatal());
    }
    match handle.stop() {
        Ok(take) => {
            let level = SignalLevel::measure(&take.audio.samples);
            println!(
                "take: {} samples @ {} Hz, peak {:.5}, rms {:.1} dBFS, silent: {}",
                take.audio.samples.len(),
                take.audio.sample_rate,
                level.peak,
                level.rms_dbfs(),
                level.is_silent()
            );
        }
        Err(err) => println!("stop: {err}"),
    }
}
