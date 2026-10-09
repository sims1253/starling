//! `mcp-dictation`: the MCP stdio server a coding agent launches.
//! Stdout carries only JSON-RPC; diagnostics go to stderr.
//!
//! It connects to the running host, identifies itself with `--client`
//! and the token from `STARLING_MCP_TOKEN` (or `--token`), and serves
//! `ask_user_dictation`. Exits 0 on stdin EOF and 1 when the host is
//! unreachable or refuses the client. See `docs/mcp-dictation.md`.

use std::io::BufReader;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::channel::RecvError;
use starling_runtime_host::agent::ALLOWLIST_FILE;
use starling_runtime_host::client::HostClient;
use starling_runtime_host::mcp::{AskSink, McpServer};
use starling_runtime_host::{default_data_root, platform};

/// Preferred over `--token`, which other users can read from the
/// process's command line.
const TOKEN_ENV: &str = "STARLING_MCP_TOKEN";

fn usage() -> ! {
    eprintln!(
        "mcp-dictation — the Starling dictation MCP server

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
    let token = std::env::var(TOKEN_ENV).ok().or(token);
    let Some(token) = token else {
        eprintln!("a token is required: set {TOKEN_ENV} (or pass --token)");
        usage();
    };
    let socket = socket.unwrap_or_else(|| {
        let root = root.unwrap_or_else(|| {
            default_data_root().unwrap_or_else(|err| {
                eprintln!("mcp-dictation: {err}");
                std::process::exit(1);
            })
        });
        platform::socket_path(&platform::default_runtime_dir(), &root)
    });

    // The agent may start this server in the same breath as the host.
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

    if let Err(err) = client.agent_hello(&client_name, &token) {
        eprintln!(
            "mcp-dictation: the host refused client {client_name:?}: {err}; \
             add the client to the host's {ALLOWLIST_FILE}"
        );
        std::process::exit(1);
    }

    let server = McpServer::new(std::io::stdout());

    // Host results → tool results. If the host connection dies, pending
    // calls fail and later ones are refused, but the session goes on.
    {
        let server = server.clone();
        let client = Arc::clone(&client);
        std::thread::Builder::new()
            .name("starling-mcp-results".to_string())
            .spawn(move || loop {
                match client.recv_ask_timeout(Duration::from_millis(250)) {
                    Ok(result) => server.complete(&result.req, result.outcome),
                    Err(RecvError::Timeout) if !client.is_closed() => {}
                    Err(_) => {
                        server.fail_pending(&client.close_reason());
                        return;
                    }
                }
            })
            .expect("mcp results thread spawns");
    }

    let sink = HostSink { client };
    server.serve_read(BufReader::new(std::io::stdin().lock()), &sink);
    // Stdin EOF: the agent ended the session. Exiting closes the host
    // connection, and the host cancels this connection's asks.
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
