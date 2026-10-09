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
//!   (no second wire format; transport control plus the host-level
//!   agent-dictation ask frames).
//! - [`auth`] + [`platform`] — peer authentication and the per-OS
//!   transport: UDS + `SO_PEERCRED`/`LOCAL_PEERCRED` on unix, a DACL'd
//!   named pipe on Windows (compile-unverified there — see that module).
//! - [`limits`] — per-connection size/rate budgets.
//! - [`server`] — the ownership ladder (lease → endpoint → runtime),
//!   connection lifecycle, event fan-out, supervised shutdown.
//! - [`engine`] — the supervised inference engine attached to the
//!   runtime, following the desktop settings file while the host
//!   serves.
//! - [`agent`] — the agent-dictation ask surface: the client allowlist
//!   and the broker that serializes asks behind a prompt-visibility
//!   gate and drives the existing capture/jobs path.
//! - [`mcp`] — the MCP stdio server behind the `mcp-dictation` binary,
//!   bridging `tools/call` onto [`agent`]'s ask frames.
//! - [`client`] — the client library the GPUI app will hold (the
//!   Electron comparison app has since been removed from the tree).
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
//! - The C++ engine process attaches here ([`engine`], #220): the host
//!   owns the bundled `starling-serve` sidecar's supervisor
//!   (`starling_dictation::engine::EngineManager` — readiness, crash
//!   restart, model switching) per the user's engine settings, and the
//!   jobs machine reaches it through [`engine::EngineProvider`], which
//!   leases the active engine per job. The sidecar's `--parent-pid` is
//!   the host, so it dies with the host, never with a renderer.
//!
//! # What remains outside this crate (the consuming increments)
//!
//! The GPUI app (`crates/app`) does not hold a [`client::HostClient`]
//! yet — it still talks to `starling-dictation` directly (recorder,
//! uploads, its own engine manager), and the switchover is its own
//! increment: the client library here is that surface, and the engine
//! supervisor it would hand over already runs here and shares the app's
//! sidecar through the engine registry in the meantime. The Electron
//! comparison app the design names as a second client has been removed
//! from the tree, so there is no second adapter to switch.

pub mod agent;
pub mod auth;
pub mod client;
pub mod config;
pub mod engine;
pub mod frame;
pub mod limits;
pub mod mcp;
pub mod platform;
pub mod server;

pub use config::{default_data_root, HostConfig};
pub use server::{serve, HostError, HostHandle};
