# Coding-agent dictation over MCP

Part of [#292]; implements the runtime-host slice of [#309]: a coding
agent (Claude Code, Codex, …) asks the user a question by voice, the
desktop app shows the prompt, and the agent receives the spoken answer
as the tool result.

```
coding agent ──JSON-RPC 2.0 (stdio)── starling-mcp-dictation
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
timeout_ms?: integer)`. It never reads history or documents — explicit
sharing with agents is [#224]'s separate scope — and it never returns
processed text: the answer is the finalized **raw** transcript (a
processing mode would have to be selected through the existing
processing configuration, which no host config wires to agent asks
today; see *What is deliberately not here* below).

## Protocol & safety model

- **Prompt-visibility gate.** No agent can start the microphone without
  a visible prompt. The host broker shows the questions to the app
  connection (`ShowPrompt`) and sends `capture.start` **only after**
  the app acks visibility (`PromptAck { visible: true }`). With no ack
  within a bound (10 s, clamped by the ask's remaining budget) the ask
  fails with the typed error `no_prompt_ack` and the mic is never
  touched; with no app connection attached the ask fails immediately.
  A prompt dismissed mid-take (`visible: false` after the take began)
  stops the capture.
- **Per-client allowlist, default deny.** MCP over stdio has no strong
  client identity (whoever can spawn a process can speak the
  protocol), so the boundary is a shared secret the *user* provisions:
  `<data_root>/mcp-clients.json` names each agent client and its
  token, and the same token is registered in the agent's MCP config.
  The token rides the host's same-user-authenticated IPC transport (UDS
  `SO_PEERCRED` / a DACL'd named pipe), so it is not exposed
  cross-user; but this is a *user intent* boundary (which agents may
  summon the mic), not a defense against malware already running as
  the same user. Unknown client, wrong token, or no allowlist file →
  the agent hello is refused (`auth_failed`) and the connection
  closes. A malformed allowlist file refuses host startup (fail
  closed, loudly).
- **Queueing.** Concurrent `tools/call`s (and asks from several
  agents) are serialized by the host broker: one visible prompt, one
  live capture at a time; the rest wait in a bounded queue (4) and
  past it are refused with `queue_full`. The timeout budget of a
  queued ask starts when it is admitted (a queued ask shows no prompt,
  so the user-facing clock starts when one could).
- **Cancel.** An MCP `notifications/cancelled` (or the agent closing
  stdin) cancels the ask: a live take is aborted (`capture.abort`),
  the prompt hides, and the caller receives the typed no-answer
  `agent_cancelled`.
- **Timeout.** `timeout_ms` (default 120 000, bounds 1 000–600 000)
  covers admission → prompt → speaking → transcription; expiry aborts
  a live take and returns the typed no-answer `timeout`.
- **Disconnect.** An ask is bound to the connection that sent it: that
  connection dying — EOF, crash, `kill -9` of the MCP server — cancels
  its asks the same way an explicit cancel would. This rule is
  enforced by the *host*, so it holds even when the MCP server process
  is gone before it could say anything.
- **No-answer vs error (the MCP mapping).** `Answered` returns the
  transcript as plain text. A *completed* ask with no transcript
  (timeout, agent cancel, user cancel, decline) returns
  `isError: true` with a `"No answer: <reason>."` text — the choice is
  deliberate: an error result can never be mistaken for spoken words.
  Refusals and failures (not allowlisted, no prompt ack, queue full,
  capture/transcription failed) return `isError: true` with
  `"Error [<code>]: <message>."`. Malformed tool arguments are refused
  with JSON-RPC `-32602`.

## Setup

You need: a running host (the desktop app, or `starling-runtime-host`),
one token, and the agent registration. The MCP server is stdio-only on
purpose — Claude Code's HTTP tool timeout is shorter than a human
speaking.

1. **Create the token and allow the client** — write
   `<data_root>/mcp-clients.json` next to the host's storage v2 root
   (the desktop app's data directory; the default root is what
   `starling-runtime-host` serves with no `--root`). Use a fresh random
   token, e.g. `openssl rand -hex 32`:

   ```json
   {
     "version": 1,
     "clients": [
       { "name": "claude-code", "token": "<paste-your-token>" },
       { "name": "codex", "token": "<another-token>" }
     ]
   }
   ```

   Missing file = deny all (the unconfigured host serves; MCP clients
   are refused). Restart the host after editing the file (it is read
   once at startup).

2. **Register the command with the agent.** The binary is
   `starling-mcp-dictation` from the desktop workspace
   (`cargo build -p starling-runtime-host` in `apps/desktop-gpui`).

   **Claude Code** (project or user scope):

   ```bash
   claude mcp add --transport stdio starling-dictation -- \
     /path/to/starling-mcp-dictation --client claude-code --token <paste-your-token>
   ```

   or in `.mcp.json` / `~/.claude.json`:

   ```json
   {
     "mcpServers": {
       "starling-dictation": {
         "type": "stdio",
         "command": "/path/to/starling-mcp-dictation",
         "args": ["--client", "claude-code", "--token", "<paste-your-token>"],
         "env": {}
       }
     }
   }
   ```

   **Codex** (`~/.codex/config.toml`):

   ```toml
   [mcp_servers.starling-dictation]
   command = "/path/to/starling-mcp-dictation"
   args = ["--client", "codex", "--token", "<paste-your-token>"]
   ```

   The server derives the host endpoint from the default data root;
   `--socket <path>` or `--root <dir>` override it. `--help` documents
   everything.

3. **Ask.** The agent now sees `ask_user_dictation`; the desktop app
   shows the prompt (see the caveat below), the user speaks, the agent
   gets the text.

## What is deliberately not here

- **HTTP transport.** stdio only (#309 notes Claude Code's HTTP tool
  timeout is too short for speaking).
- **App-side prompt polish.** The frame protocol (`ShowPrompt`,
  `PromptAck`, `PromptDone`, `HidePrompt` in the host's IPC) and its
  client API (`HostClient::recv_ui_timeout` / `prompt_ack` /
  `prompt_done`) are the seam the GPUI overlay plugs into; the GPUI app
  does not hold a `HostClient` yet (the E17 switchover is its own
  increment), so today a connected client — a test, a tray helper —
  plays the app side. Until the app implements the overlay, register
  the ask surface only where you control the acking side.
- **Processed output.** The tool returns the raw transcript; wiring a
  processing mode (#294) into agent asks is follow-up work.
- **Real-agent E2E.** The suites prove the host and MCP layers with
  fake transport peers and the scripted capture source; a live Claude
  Code / Codex session is the remaining acceptance check of #309.

[#292]: https://github.com/sims1253/starling/issues/292
[#309]: https://github.com/sims1253/starling/issues/309
[#224]: https://github.com/sims1253/starling/issues/224
[#294]: https://github.com/sims1253/starling/issues/294
