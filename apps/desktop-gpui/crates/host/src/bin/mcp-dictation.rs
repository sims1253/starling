//! `mcp-dictation` — the MCP stdio server a coding agent launches
//! (issue #309). Stdin/stdout are the agent's JSON-RPC 2.0;
//! everything else (errors, diagnostics) goes to stderr, because
//! stdout belongs to the protocol alone.
//!
//! The process connects to the Starling runtime host over the same
//! authenticated IPC transport as the desktop app, identifies itself
//! with `--client` and its token (the `STARLING_MCP_TOKEN`
//! environment variable, or `--token` as a fallback) against the
//! host's allowlist (default deny — see `starling_runtime_host::agent`),
//! and then serves exactly one tool, `ask_user_dictation`, bridging
//! `tools/call` onto the host's ask surface.
//!
//! Exit contract:
//! - `0` on a clean stdin EOF (the agent closed the session);
//! - `1` when the host is unreachable or the client is not
//!   allowlisted (stderr says which — the agent surfaces it).
//!
//! See `docs/mcp-dictation.md` for the Claude Code / Codex setup.

use std::io::BufReader;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime_host::agent::ALLOWLIST_FILE;
use starling_runtime_host::client::HostClient;
use starling_runtime_host::mcp::{AskSink, McpServer};
use starling_runtime_host::{default_data_root, platform};

/// The environment variable the allowlist token is read from — the
/// preferred channel: `--token` on the command line is world-readable
/// through `/proc/<pid>/cmdline`, an environment variable is not.
const TOKEN_ENV: &str = "STARLING_MCP_TOKEN";

fn usage() -> ! {
    eprintln!(
        "mcp-dictation — the Starling dictation MCP server (issue #309)

USAGE:
    mcp-dictation --client <name> [options]

    The token is read from the {TOKEN_ENV} environment variable
    (preferred — argv is world-readable via /proc/<pid>/cmdline).

OPTIONS:
    --client <name>   the allowlisted agent client name (see the host's
                      {ALLOWLIST_FILE})
    --token <token>   that client's token (fallback when {TOKEN_ENV}
                      is unset)
    --socket <path>   the runtime host's IPC endpoint
                      (default: derived from the default data root)
    --root <dir>      the storage v2 data root whose host to use
                      (default: the platform default root)
    --help, -h        this text

The host must be running (the desktop app or starling-runtime-host);
this server is the agent-side bridge, not a host itself."
    );
    std::process::exit(2);
}

/// The ask sink: one live host connection. `ask` is fire-and-forget on
/// the wire (the answer arrives on the asks stream minutes later);
/// `cancel` routes the agent's cancellation to the host, which stops
/// the microphone if the take is live.
struct HostSink {
    client: Arc<HostClient>,
}

impl AskSink for HostSink {
    fn ask(&self, req: &str, questions: Vec<String>, timeout_ms: u64) -> Result<(), String> {
        self.client
            .ask_user(req, &questions, timeout_ms)
            .map_err(|err| err.to_string())
    }

    fn cancel(&self, req: &str, reason: &str) {
        let _ = self.client.ask_cancel(req, reason);
    }
}

fn main() {
    let mut client_name: Option<String> = None;
    let mut token: Option<String> = None;
    let mut socket: Option<std::path::PathBuf> = None;
    let mut root: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--client" => client_name = Some(value_of(&mut args, &flag)),
            "--token" => token = Some(value_of(&mut args, &flag)),
            "--socket" => socket = Some(std::path::PathBuf::from(value_of(&mut args, &flag))),
            "--root" => root = Some(std::path::PathBuf::from(value_of(&mut args, &flag))),
            "--help" | "-h" => usage(),
            other => {
                eprintln!("unknown argument {other:?}");
                usage();
            }
        }
    }
    let Some(client_name) = client_name else {
        eprintln!("--client is required (the allowlisted agent client name)");
        usage();
    };
    // The environment variable first; the flag is the fallback.
    let token = std::env::var(TOKEN_ENV).ok().or(token);
    let Some(token) = token else {
        eprintln!("a token is required: set {TOKEN_ENV} (or pass --token)");
        usage();
    };
    let socket = socket.unwrap_or_else(|| {
        let root = root.unwrap_or_else(||
            // The same default the host binary serves; a mismatched
            // root fails at connect with the endpoint missing, which
            // is the honest error.
            default_data_root().unwrap_or_else(|err| {
                eprintln!("mcp-dictation: {err}");
                std::process::exit(1);
            }));
        platform::socket_path(&platform::default_runtime_dir(), &root)
    });

    // Connect with a short bounded retry: the agent may start this
    // server in the same breath as the host.
    let client = {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match HostClient::connect(&socket) {
                Ok(client) => break client,
                Err(err) if Instant::now() >= deadline => {
                    eprintln!(
                        "mcp-dictation: no Starling host at {} within 5s: {err}; \
                         start the desktop app or starling-runtime-host first",
                        socket.display()
                    );
                    std::process::exit(1);
                }
                Err(_) => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    };
    let client = Arc::new(client);

    // The allowlist gate, before a single tool call: a client that is
    // not allowlisted fails fast with the refusal on stderr.
    if let Err(err) = client.agent_hello(&client_name, &token) {
        eprintln!(
            "mcp-dictation: the host refused client {client_name:?}: {err}; \
             add the client to the host's {ALLOWLIST_FILE}"
        );
        std::process::exit(1);
    }

    let server = McpServer::new();
    // A `Stdout` handle (not a lock guard) crosses the thread; the
    // writer thread is the process's only stdout writer, so per-call
    // internal locking still leaves every line atomic. Diagnostics go
    // to stderr — stdout belongs to the protocol alone.
    let writer = server.spawn_writer(std::io::stdout());

    // The completion pump: host ask results → JSON-RPC tool results.
    // On connection loss every pending call fails honestly and the
    // server keeps serving (initialize/tools/list need no host).
    let pump = {
        let server = server.clone();
        let client = Arc::clone(&client);
        std::thread::Builder::new()
            .name("starling-mcp-drain".to_string())
            .spawn(move || loop {
                match client.recv_ask_timeout(Duration::from_millis(250)) {
                    Ok(result) => server.complete(&result.req, result.outcome),
                    Err(starling_runtime::channel::RecvError::Timeout) => {
                        if client.is_closed() {
                            server.fail_pending(&client.close_reason());
                            return;
                        }
                    }
                    Err(starling_runtime::channel::RecvError::Closed) => {
                        server.fail_pending(&client.close_reason());
                        return;
                    }
                }
            })
            .expect("mcp drain spawn")
    };

    let sink = HostSink {
        client: Arc::clone(&client),
    };
    // stdin to EOF: answers everything; on return (EOF) every pending
    // ask is cancelled — the deliberate disconnect rule.
    server.serve_read(BufReader::new(std::io::stdin().lock()), &sink);
    server.cancel_pending(&sink, "the agent closed the MCP session");
    // End the host connection explicitly (the drain thread holds its
    // own client handle, so dropping this one would not): the host's
    // broker sees the disconnect and aborts any take the cancels could
    // not reach (the belt-and-braces pair of rules — see
    // docs/mcp-dictation.md), and the drain thread's poll sees the
    // connection closed and exits.
    client.close();
    // Drop the writer's sender so the writer thread drains its queue
    // and exits too — without this both joins below would wait forever.
    server.shutdown_writer();
    let _ = pump.join();
    let _ = writer.join();
    std::process::exit(0);
}

fn value_of(args: &mut impl Iterator<Item = String>, flag: &str) -> String {
    match args.next() {
        Some(value) => value,
        None => {
            eprintln!("{flag} needs a value");
            usage();
        }
    }
}
