//! `starling-processing` — staged drafts and text processing (#293).
//!
//! - [`contract`]: the #293 records (mode entry, provider declaration,
//!   transform request/result) and the provider choice, ported from the
//!   executable oracle `tests/mode_routing.py`.
//! - [`boundary`]: the delivery-time insertion-boundary rules (leading
//!   space, first-letter case), frozen in
//!   `packages/contracts/insertion-boundary/`.
//! - [`routing`]: the routing oracle itself (leading mode phrases, the
//!   literal escape, rule conflicts), ported from the same file.
//! - [`staging`]: the staged draft of one take (typed regions, immutable
//!   raw attempts, proposals pinned to a revision), ported from
//!   `tests/staging.py`.
//! - [`transforms`]: the deterministic step (spoken commands, snippets),
//!   ported from `tests/spoken_commands.py` (#294).
//! - [`instructions`]: the trailing spoken instruction grammar, ported
//!   from `tests/spoken_instructions.py`.
//! - [`providers`] + [`http`]: the model step behind one contract: S1-mini
//!   on a local starling-serve, or any OpenAI-compatible chat endpoint.
//! - [`pipeline`]: plan → request → run, one [`contract::TransformResult`]
//!   per job with its timing; [`insight`] turns it into the
//!   `processing_recorded` event.
//! - [`live`]: a `/stream` take (running text + stable word count) as
//!   draft operations, for the live staging editor (#297).
//!
//! The contract data lives in `packages/contracts/mode-routing/`; the
//! conformance tests here read those fixtures in place, so the Rust port
//! and the Python oracle can never test against different copies.

pub mod boundary;
pub mod contract;
pub mod http;
pub mod insight;
pub mod instructions;
pub mod live;
pub mod pipeline;
pub mod prompt;
pub mod providers;
pub mod routing;
pub mod staging;
pub mod transforms;

pub use starling_dictation::client::CancelToken;
