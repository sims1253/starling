//! Live Wayland check: types into a GTK entry (`zenity --entry`) through
//! the virtual-keyboard backend. Opt-in, and only against a disposable
//! compositor: it sends keys to whatever is focused.
//!
//! ```text
//! niri -c <config> -- sleep infinity   # a nested compositor, e.g. wayland-2
//! STARLING_WAYLAND_IT=1 WAYLAND_DISPLAY=wayland-2 \
//!     cargo test -p starling-insertion --test wayland_integration
//! ```
//!
//! Needs `zenity` and `wtype` (the latter only presses Return to close the
//! dialog: the backend never types Enter).

#![cfg(target_os = "linux")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use starling_insertion::wayland::WaylandBackend;
use starling_insertion::{InsertError, InsertionBackend, EVIDENCE_SYNTHETIC_KEYS};

fn enabled() -> bool {
    std::env::var_os("STARLING_WAYLAND_IT").is_some()
        && std::env::var_os("WAYLAND_DISPLAY").is_some()
}

#[test]
fn wayland_types_unicode_into_a_gtk_entry() {
    if !enabled() {
        eprintln!(
            "skipped: set STARLING_WAYLAND_IT=1 and WAYLAND_DISPLAY (a disposable compositor)"
        );
        return;
    }
    let backend = WaylandBackend::new();
    backend
        .availability()
        .expect("the compositor offers virtual keyboards");

    let mut dialog = Command::new("zenity")
        .args(["--entry", "--title=starling-it", "--text=type here"])
        .env_remove("DISPLAY")
        .env("GDK_BACKEND", "wayland")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("zenity");
    // The new window takes focus once mapped.
    std::thread::sleep(Duration::from_millis(2500));

    let target = backend.capture().unwrap();
    assert!(
        target.target_ref.starts_with("wl:"),
        "{}",
        target.target_ref
    );
    let text = "Café, naïve 😀 – 42 ÄÖÜ ß!";
    assert_eq!(
        backend.insert(&target, "never\nEnter"),
        Err(InsertError::MultilineUnsupported)
    );
    let receipt = backend.insert(&target, text).expect("typed");
    assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);

    // A caller's stop ends typing before the next key: once after the
    // lock, then once per key, so the fifth check stops after three keys.
    let checks = std::cell::Cell::new(0);
    let stop = || {
        checks.set(checks.get() + 1);
        (checks.get() >= 5).then_some(InsertError::TargetIsStarling)
    };
    assert_eq!(
        backend.insert_guarded(&target, " stopped", &stop),
        Err(InsertError::PartialDelivery {
            delivered_chars: 3,
            total_chars: 8,
            cause: Box::new(InsertError::TargetIsStarling),
        })
    );

    std::thread::sleep(Duration::from_millis(300));
    let pressed = Command::new("wtype")
        .args(["-k", "Return"])
        .env_remove("DISPLAY")
        .status()
        .expect("wtype");
    assert!(pressed.success());

    let deadline = Instant::now() + Duration::from_secs(5);
    while dialog.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "the dialog did not close");
        std::thread::sleep(Duration::from_millis(50));
    }
    let mut out = String::new();
    dialog
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    assert_eq!(out.trim_end_matches('\n'), format!("{text} st"));
}
