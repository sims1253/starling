//! `starling-processing` — staged drafts and text processing (#293).
//!
//! - [`contract`]: the #293 records (mode entry, provider declaration,
//!   transform request/result) and the provider choice, ported from the
//!   executable oracle `tests/mode_routing.py`.
//! - [`staging`]: the staged draft of one take (typed regions, immutable
//!   raw attempts, proposals pinned to a revision), ported from
//!   `tests/staging.py`.
//!
//! The contract data lives in `packages/contracts/mode-routing/`; the
//! conformance tests here read those fixtures in place, so the Rust port
//! and the Python oracle can never test against different copies.

pub mod contract;
pub mod staging;
