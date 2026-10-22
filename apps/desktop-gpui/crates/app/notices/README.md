`third-party-notices.json` here is generated, not committed: run
`python3 scripts/licenses/notices.py` from the repository root (packaging does,
before `cargo build`). `build.rs` embeds it, and the Licenses page in the
settings lists every component the app ships with its license text.
