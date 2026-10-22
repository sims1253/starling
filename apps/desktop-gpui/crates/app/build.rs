//! Embeds the third-party notices that scripts/licenses/notices.py writes
//! into notices/ (packaging runs it before `cargo build`). Without them a
//! development build embeds `null` and the Licenses page says how to
//! generate them; STARLING_THIRD_PARTY_NOTICES points at another file and
//! STARLING_REQUIRE_THIRD_PARTY_NOTICES=1 (packaging) makes them mandatory.

use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("third-party-notices.json");
    println!("cargo:rerun-if-env-changed=STARLING_THIRD_PARTY_NOTICES");
    let source = match env::var_os("STARLING_THIRD_PARTY_NOTICES") {
        Some(path) => PathBuf::from(path),
        None => manifest.join("notices").join("third-party-notices.json"),
    };
    // The directory, not the file: a watched path that does not exist
    // reruns this script on every build, and generating the file changes
    // the directory.
    println!(
        "cargo:rerun-if-changed={}",
        source.parent().unwrap_or(&manifest).display()
    );
    // Packaging sets STARLING_REQUIRE_THIRD_PARTY_NOTICES=1, so a build that
    // is distributed can never embed the development fallback.
    println!("cargo:rerun-if-env-changed=STARLING_REQUIRE_THIRD_PARTY_NOTICES");
    let required = env::var_os("STARLING_REQUIRE_THIRD_PARTY_NOTICES").is_some_and(|v| v == "1");
    let notices = match fs::read(&source) {
        Ok(notices) => notices,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound && !required => b"null".to_vec(),
        Err(err) => panic!(
            "third-party notices {}: {err} (run python3 scripts/licenses/notices.py)",
            source.display()
        ),
    };
    fs::write(&out, notices).expect("write the embedded third-party notices");
}
