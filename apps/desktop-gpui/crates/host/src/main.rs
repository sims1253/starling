//! The `starling-runtime-host` binary: one per user session. The whole
//! command line lives in [`starling_runtime_host::cli`], which the desktop
//! app also runs as `starling-gpui --runtime-host`.

fn main() {
    std::process::exit(starling_runtime_host::cli::run(std::env::args().skip(1).collect()));
}
