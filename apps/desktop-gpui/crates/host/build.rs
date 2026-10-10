//! Stamps the host's build (#220): an app and the host it reaches
//! compare stamps, so a newer app replaces an older host that idles and
//! a newer host refuses an older app instead of serving it a protocol it
//! does not speak.
//!
//! The id is a content hash of everything the host runs — this crate and
//! the path crates it builds on, plus the lockfile — so two builds of the
//! same code agree and any change to it differs. The app crate is not
//! part of it: the app binary runs the host as `--runtime-host`, and a
//! change only to the app's own code leaves the host it runs (and its
//! wire) as it was. This script is hashed too: a change to what it
//! hashes is a new build. `built` orders two different stamps: the time
//! this script last ran, which is when the hashed sources last changed.
//! A file that cannot be read fails the build: a stamp that skipped it
//! could equal one of different code. So does a clock before 1970: a
//! stamp built at "0" would order before every other build.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let watched = [
        manifest.join("build.rs"),
        manifest.join("src"),
        manifest.join("Cargo.toml"),
        manifest.join("../runtime/src"),
        manifest.join("../runtime/Cargo.toml"),
        manifest.join("../dictation/src"),
        manifest.join("../dictation/Cargo.toml"),
        manifest.join("../processing/src"),
        manifest.join("../processing/Cargo.toml"),
        // The contracts the processing crate compiles in (`include_str!`).
        manifest.join("../../../../packages/contracts/mode-routing"),
        manifest.join("../../Cargo.lock"),
    ];
    let mut hash = Fnv::new();
    for path in &watched {
        println!("cargo:rerun-if-changed={}", path.display());
        fold(&mut hash, &manifest, path);
    }
    let built = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|err| panic!("build stamp: the build clock is before 1970: {err}"))
        .as_secs();
    println!("cargo:rustc-env=STARLING_HOST_BUILD_ID={:016x}", hash.0);
    println!("cargo:rustc-env=STARLING_HOST_BUILT={built}");
}

/// Folds `path` (a file, or a directory walked in name order) into
/// `hash`: each file's path relative to the manifest, then its bytes.
/// Panics (failing the build) on anything it cannot read.
fn fold(hash: &mut Fnv, base: &Path, path: &Path) {
    let unreadable = |err: std::io::Error| -> ! {
        panic!("build stamp: cannot read {}: {err}", path.display())
    };
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .unwrap_or_else(|err| unreadable(err))
            .map(|entry| entry.map(|entry| entry.path()).unwrap_or_else(|err| unreadable(err)))
            .collect();
        entries.sort();
        for entry in entries {
            fold(hash, base, &entry);
        }
    } else {
        let bytes = std::fs::read(path).unwrap_or_else(|err| unreadable(err));
        let name = path.strip_prefix(base).unwrap_or(path);
        hash.write(name.to_string_lossy().as_bytes());
        hash.write(&[0]);
        hash.write(&bytes);
    }
}

/// FNV-1a, 64-bit: stable across toolchains (std's hasher is not).
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}
