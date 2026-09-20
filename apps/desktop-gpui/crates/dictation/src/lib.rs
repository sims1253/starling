//! Rust port of `@starling/dictation` plus native capture/playback.
//!
//! Modules are ports of the TypeScript sources under `packages/dictation/src`
//! (same repository) and of the native request behavior in
//! `apps/desktop/electron/ipc.ts`. See `apps/desktop-gpui/PORT.md`.

pub mod audio;
pub mod client;
pub mod fft;
pub mod fidelity;
pub mod journal;
pub mod player;
pub mod recorder;
pub mod settings;
pub mod storage;
