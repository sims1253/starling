//! `starling-processing` — staged drafts and text processing (#293).
//!
//! - [`contract`]: the #293 records (mode entry, provider declaration,
//!   transform request/result) and the provider choice, ported from the
//!   executable oracle `tests/mode_routing.py`.
//! - [`staging`]: the staged draft of one take (typed regions, immutable
//!   raw attempts, proposals pinned to a revision), ported from
//!   `tests/staging.py`.
//! - [`transforms`]: the deterministic step (spoken commands, snippets),
//!   ported from `tests/spoken_commands.py` (#294).
//! - [`providers`] + [`http`]: the model step behind one contract: S1-mini
//!   on a local starling-serve, OpenAI-compatible, Anthropic, Gemini.
//! - [`pipeline`]: plan → request → run, one [`contract::TransformResult`]
//!   per job with its timing; [`insight`] turns it into the
//!   `processing_recorded` event.
//!
//! The contract data lives in `packages/contracts/mode-routing/`; the
//! conformance tests here read those fixtures in place, so the Rust port
//! and the Python oracle can never test against different copies.

pub mod contract;
pub mod http;
pub mod insight;
pub mod pipeline;
pub mod prompt;
pub mod providers;
pub mod staging;
pub mod transforms;

pub use starling_dictation::client::CancelToken;
