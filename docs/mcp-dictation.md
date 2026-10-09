# Coding-agent dictation over MCP

Part of [#292]; implements the runtime-host slice of [#309]: a coding
agent (Claude Code, Codex, …) asks the user a question by voice, the
desktop app shows the prompt, and the agent receives the spoken answer
as the tool result.

```
coding agent ──JSON-RPC 2.0 (stdio)── mcp-dictation
                                          │ AgentHello/AskUser (host IPC,
                                          │ same-user authenticated)
                                          ▼
                                    runtime host ── ShowPrompt/PromptAck/PromptDone
                                          │           (the app connection: the overlay)
                                          ▼
                                    capture machine → storage v2 → jobs machine
                                          │
                                          ▼
                                    AskResult { Answered(text) }  → the agent
```

There is exactly one tool: `ask_user_dictation(questions: string[],
timeout_ms?: integer)`. It never reads history or documents ([#224] is
the separate explicit-sharing scope) and returns the finalized **raw**
transcript.

## Protocol & safety model

- **Prompt-visibility gate.** The host shows the questions to the app
  connections (`ShowPrompt`) and sends `capture.start` only after one
  of them acks `PromptAck { visible: true }`. No ack within
  `min(10 s, timeout_ms)` fails the ask with `no_prompt_ack` and the
  microphone is never touched; with no app connected the ask fails
  immediately. Only a connection the prompt was shown to may ack it,
  only the acking connection may dismiss or finish it, and an agent
  connection sending prompt frames is closed.
- **Allowlist, default deny.** MCP over stdio has no client identity,
  so the boundary is a token the user provisions in
  `<data_root>/mcp-clients.json` and in the agent's MCP config (via
  `STARLING_MCP_TOKEN`; argv is readable by other users). Unknown
  client, wrong token, or no file → `auth_failed` and the connection
  closes. A malformed file (including duplicate names) refuses host
  startup. The token travels over the same-user-authenticated IPC
  transport, so this decides *which agents may use the microphone*; it
  does not defend against other processes running as the same user.
- **Queueing.** Asks from any number of `tools/call`s or agents are
  serialized: one prompt and one capture at a time, up to 4 waiting,
  `queue_full` beyond that.
- **Timeout.** `timeout_ms` (default 120 000, 1 000–600 000) runs from
  when the ask leaves the queue until the microphone closes; expiry
  aborts the take with the no-answer `timeout`. A captured take is
  never discarded on the clock.
- **Cancel and disconnect.** `notifications/cancelled`, or the MCP
  server's host connection ending for any reason (including `kill -9`),
  aborts the take with the no-answer `agent_cancelled`. The app
  dismissing the prompt, or disconnecting, mid-take stops it with
  `user_cancelled`.
- **Result mapping.** An answer is returned as plain text. A no-answer
  (timeout, cancel, decline) returns `isError: true` with
  `"No answer: <reason>."` so it cannot be mistaken for spoken words.
  Failures (no prompt ack, queue full, capture or transcription failed)
  return `isError: true` with `"Error [<code>]: <message>."`. Malformed
  arguments are JSON-RPC `-32602`.

## Setup

You need a running host (the desktop app, or `starling-runtime-host`),
a token, and the agent registration.

1. **Allow the client.** Write `<data_root>/mcp-clients.json` in the
   host's storage v2 root with a fresh random token per client (e.g.
   `openssl rand -hex 32`):

   ```json
   {
     "version": 1,
     "clients": [
       { "name": "claude-code", "token": "<paste-your-token>" },
       { "name": "codex", "token": "<another-token>" }
     ]
   }
   ```

   The file is read once at startup; restart the host after editing it.

2. **Register the command with the agent.** The binary is
   `mcp-dictation` (`cargo build -p starling-runtime-host` in
   `apps/desktop-gpui`).

   **Claude Code** (project or user scope):

   ```bash
   claude mcp add --transport stdio starling-dictation --env STARLING_MCP_TOKEN=<paste-your-token> -- \
     /path/to/mcp-dictation --client claude-code
   ```

   or in `.mcp.json` / `~/.claude.json`:

   ```json
   {
     "mcpServers": {
       "starling-dictation": {
         "type": "stdio",
         "command": "/path/to/mcp-dictation",
         "args": ["--client", "claude-code"],
         "env": {"STARLING_MCP_TOKEN": "<paste-your-token>"}
       }
     }
   }
   ```

   **Codex** (`~/.codex/config.toml`):

   ```toml
   [mcp_servers.starling-dictation]
   command = "/path/to/mcp-dictation"
   args = ["--client", "codex"]
   env = { "STARLING_MCP_TOKEN" = "<paste-your-token>" }
   ```

   The server derives the host endpoint from the default data root;
   `--socket <path>` or `--root <dir>` override it (see `--help`).

## What is deliberately not here

- **HTTP transport.** Claude Code's HTTP tool timeout is shorter than
  a human speaking, so the server is stdio only.
- **The app-side prompt.** The GPUI app does not hold a `HostClient`
  yet, so nothing in the app shows prompts today. The seam is the
  `ShowPrompt`/`PromptAck`/`PromptDone`/`HidePrompt` frames and
  `HostClient::recv_ui_timeout`/`prompt_ack`/`prompt_done`; until the
  overlay exists, some other client has to play the app side.
- **Processed output** ([#294]).
- **A live Claude Code / Codex session** as acceptance; the tests use
  fake peers and the scripted capture source.

[#292]: https://github.com/sims1253/starling/issues/292
[#309]: https://github.com/sims1253/starling/issues/309
[#224]: https://github.com/sims1253/starling/issues/224
[#294]: https://github.com/sims1253/starling/issues/294
