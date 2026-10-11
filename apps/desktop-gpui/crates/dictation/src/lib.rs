//! Dictation logic and IO without UI: audio prep, the transcription
//! client, storage v2, settings, the bundled engine, capture and playback.

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
pub mod process;
pub mod recorder;
pub mod settings;
pub mod storage;
pub mod store_v2;
