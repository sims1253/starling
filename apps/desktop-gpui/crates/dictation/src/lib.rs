//! Dictation core for the desktop app plus native capture/playback.
//!
//! Modules started as ports of the removed TypeScript `@starling/dictation`
//! library and the Electron request path (`apps/desktop/electron/ipc.ts`);
//! they are now the only implementation. See `apps/desktop-gpui/PORT.md`.

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
