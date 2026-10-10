//! The AT-SPI field reader against a real toolkit field (opt-in):
//!
//! ```text
//! STARLING_ATSPI_IT=1 STARLING_ATSPI_EXPECT=<text|protected> \
//!   [STARLING_ATSPI_BEFORE=<text before the caret>] [STARLING_ATSPI_TYPE=<text>] \
//!   cargo test -p starling-insertion --test atspi_live -- --nocapture
//! ```
//!
//! Run it inside a nested compositor with its own session bus and
//! accessibility bus, a GTK window with the field focused. It locates the
//! focused field the way a take start does (no pid: Wayland), reads it,
//! and with `STARLING_ATSPI_TYPE` types that text through the Wayland
//! virtual keyboard and reads the field again.
#![cfg(target_os = "linux")]

use starling_insertion::atspi::AtspiReader;
use starling_insertion::wayland::WaylandBackend;
use starling_insertion::{
    FieldReader, InsertionBackend, Surrounding, SurroundingText, TargetSnapshot,
};

fn before(text: &str) -> Surrounding {
    Surrounding::Text(SurroundingText {
        before: text.to_string(),
        after: String::new(),
        selection: None,
    })
}

#[test]
fn a_real_focused_field_is_located_and_read() {
    if std::env::var_os("STARLING_ATSPI_IT").is_none() {
        eprintln!("skipped: set STARLING_ATSPI_IT=1 in a nested session");
        return;
    }
    let expect = std::env::var("STARLING_ATSPI_EXPECT").expect("STARLING_ATSPI_EXPECT");
    let reader = AtspiReader::new(Vec::new());
    let wayland = WaylandBackend::with_excluded_pids(Vec::new());
    let target: TargetSnapshot = wayland.capture().expect("a Wayland capture");
    assert_eq!(target.pid, None);

    let started = std::time::Instant::now();
    let field = reader
        .locate(&target)
        .expect("the focused field is located");
    eprintln!("located {field:?} in {:?}", started.elapsed());
    let started = std::time::Instant::now();
    let read = reader.read(&target, &field);
    eprintln!("read {read:?} in {:?}", started.elapsed());

    match expect.as_str() {
        "protected" => assert_eq!(read, Surrounding::Protected),
        "text" => {
            let expected = std::env::var("STARLING_ATSPI_BEFORE").unwrap_or_default();
            assert_eq!(read, before(&expected));
            if let Ok(typed) = std::env::var("STARLING_ATSPI_TYPE") {
                wayland.insert(&target, &typed).expect("typed");
                std::thread::sleep(std::time::Duration::from_millis(300));
                let after = reader.read(&target, &field);
                eprintln!("after typing: {after:?}");
                assert_eq!(after, before(&format!("{expected}{typed}")));
            }
        }
        other => panic!("unknown STARLING_ATSPI_EXPECT {other}"),
    }
}
