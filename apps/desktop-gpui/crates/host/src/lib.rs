//! `starling-runtime-host` — the E17 Mode B service host (increment I4).
//!
//! One process per user session owns the runtime: it acquires the
//! storage-v2 lease (§4 ownership — the designed lease acquirer #257
//! made possible), binds the OS-local authenticated endpoint, boots the
//! [`starling_runtime::Runtime`] with storage v2 as THE capture store
//! (D14), and serves any number of short-lived renderer clients.
//! Renderers are projections: they send commands and receive events +
//! snapshots over [`client::HostClient`], and killing one costs nothing
//! durable — acknowledged audio lives in journals and SQLite, and a
//! reconnected client resynchronizes from the snapshot (the
//! renderer-kill acceptance this crate's test suite proves over the real
//! transport).
//!
//! # Layout
//!
//! - [`frame`] — the length-prefixed frame around the I3 envelope
//!   (no second wire format; transport control only).
//! - [`auth`] + [`platform`] — peer authentication and the per-OS
//!   transport: UDS + `SO_PEERCRED`/`LOCAL_PEERCRED` on unix, a DACL'd
//!   named pipe on Windows (compile-unverified there — see that module).
//! - [`limits`] — per-connection size/rate budgets.
//! - [`server`] — the ownership ladder (lease → endpoint → runtime),
//!   connection lifecycle, event fan-out, supervised shutdown.
//! - [`client`] — the client library GPUI and the Electron adapter will
//!   hold.
//!
//! # Worker supervision boundary (recorded precisely)
//!
//! The design (§1 Mode B) has supervised **C++ engine worker processes**
//! attach to the runtime. What exists today and what this host owns:
//!
//! - The host constructs the runtime and its provider — worker lifetime
//!   is host lifetime, not renderer lifetime. A renderer disconnect
//!   never touches a worker: the IPC suite proves a submitted job keeps
//!   running and completes to another connected client after its
//!   submitting client dies.
//! - The jobs machine's scheduler already supervises **in-process**
//!   workers per §2.2 (bounded concurrency, crash demotion to
//!   `Failed{retryable}` — proven by `starling-runtime`'s
//!   `scripted_take` suite).
//! - What does not exist yet: any engine-process attach interface. There
//!   is no C++ engine worker protocol or entrypoint in-tree (the ggml
//!   submodule has no server shape), so a subprocess provider adapter
//!   and its supervisor would be speculative code against an interface
//!   nobody defined. When E03/I5 defines the engine's attach surface, it
//!   lands as a `TranscriptionProvider` adapter plus process
//!   supervision in this host; nothing here blocks it.
//!
//! # What remains outside this crate (the consuming increments)
//!
//! Neither the GPUI app (`crates/app`) nor the Electron comparison
//! adapter embeds the runtime today — the app talks to
//! `starling-dictation` directly, and the Mode A embed was deliberately
//! left as "the consuming increment's wiring" when I3 merged (same note
//! as `default_capture_store`). Both become [`client::HostClient`]
//! holders in their own switchover increments; the client library here
//! is that surface.

pub mod auth;
pub mod client;
pub mod config;
pub mod frame;
pub mod limits;
pub mod platform;
pub mod server;

pub use config::{default_data_root, HostConfig};
pub use server::{serve, HostError, HostHandle};
