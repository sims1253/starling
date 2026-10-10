//! Rust port of `@starling/dictation` plus native capture/playback.
//!
//! Modules are ports of the TypeScript sources under `packages/dictation/src`
//! (same repository) and of the native request behavior in
//! `apps/desktop/electron/ipc.ts`. See `apps/desktop-gpui/PORT.md`.

pub mod audio;
pub mod client;
pub mod disk;
pub mod engine;
pub mod fft;
pub mod fidelity;
pub mod flac;
pub mod journal;
pub mod microphone;
pub mod playback;
pub mod player;
pub mod recorder;
pub mod settings;
pub mod storage;
pub mod store_v2;
