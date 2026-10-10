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
//! - [`client`] — the client library the GPUI app holds (#220).
//! - [`takes`] — the take feed: the app's projection of the takes this
//!   host records (status, audio, stored rows, orphans), beside the
//!   envelope like the ask frames.
//! - [`transcribe`] — transcription in the host: live text while a take
//!   records, its transcript once stored, and retries apps ask for.
//! - [`capture`] — the production capture source (the desktop settings'
//!   microphone, journaled, disk-watched).
//! - [`recovery`] — owner-side startup recovery beyond reconcile (stale
//!   attempts, the recorder's journal tree).
//! - [`cli`] — the command line, shared by this crate's binary and the
//!   app's `--runtime-host`.
//! - [`live`] — live transcription while a take records: the `/stream`
//!   client and the worker pumping the take's audio into it.
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
//! # What the app holds and what remains (#220)
//!
//! The GPUI app (`crates/app`) is a client: it starts this host from its
//! own executable when none serves (`starling-gpui --runtime-host`),
//! records every take through the capture machine and the take feed,
//! and never opens a recorder, a journal or the store lease itself. It
//! still transcribes its takes (uploads, the live stream, retries) and
//! writes their attempt rows through its own store handle — a store
//! client, the multi-process shape storage v2 is built for — and runs
//! its own engine manager; the host it starts runs with `--engine
//! none`. Moving transcription, store writes and the engine into the
//! host are the next increments. The Electron comparison app the design
//! names as a second client has been removed from the tree.

pub mod agent;
pub mod auth;
pub mod capture;
pub mod cli;
pub mod client;
pub mod config;
pub mod engine;
pub mod frame;
pub mod limits;
pub mod live;
pub mod mcp;
pub mod platform;
pub mod recovery;
pub mod server;
pub mod takes;
pub mod transcribe;

pub use config::{default_data_root, HostConfig};
pub use server::{serve, HostError, HostHandle};
