# Starling desktop

The desktop client records mono microphone audio, converts it to PCM16 16 kHz WAV, saves the WAV locally in IndexedDB, and then sends it to the selected server. A failed or interrupted request leaves the recording available to play, download, and retry. Transcript text is displayed and exported exactly as returned by the server.

When the server is a native `starling-serve` and "Live streaming transcript" is enabled in settings, recording additionally streams PCM16 chunks over the server's `WS /stream` endpoint and shows partial transcripts while you speak. Each chunk is journaled to local storage before it is streamed, the WAV is finalized before the streamed result is accepted, and any stream failure falls back to the batch upload of the saved WAV. The streaming connection assumes a loopback or trusted endpoint without authentication, like the native server's default; batch uploads keep the proxy-auth support of the shared client.

A finished transcript can also be refined by any OpenAI-compatible chat endpoint — a local model such as Ollama at `http://127.0.0.1:11434/v1`, a llama.cpp or LM Studio server, or a hosted API such as `https://api.openai.com/v1`. Refinement is explicit and opt-in per take: configure the endpoint under "Transcript refinement" in settings, then press Refine in the transcript drawer. The refined copy is stored and labeled as a separate field on the same session; the raw transcript — what the drawer leads with — is never rewritten, and the text export appends the refined copy below it instead of replacing anything. A completion whose termination metadata says it is not whole — `finish_reason: "length"` (truncated at the output limit), `"content_filter"`, or an explicit model refusal — is rejected as a failed refinement with the previous refined copy left in place; providers that omit `finish_reason` (common for local servers) are trusted. A refinement API key is stored as OS-secret-store-encrypted ciphertext (Electron `safeStorage`) only when the host really has one — on Linux, `basic_text` and unidentified backends are detected as unprotected rather than silently used. When no protected store is available (the browser preview, or a desktop host without a secret store), the key is kept for the current session only and never written to disk unless you explicitly tick "Store API key unencrypted" in settings; that plaintext choice, and the per-save protection status, are shown in the settings dialog, so prefer local endpoints that need no key at all.

Refinements can chain across takes as a thread. Pressing "Refine in thread" on a finished take joins it to the active thread — a fresh one is created when none is active — and pressing "Refine with thread context" on a threaded take reuses the thread. Each threaded turn is refined against the thread's current text: the model receives the previous refined text plus the newly dictated turn, which may be an edit instruction ("make that more formal") or additional dictation, and returns the complete updated text as that turn's refined copy. Threading is explicit and visible: takes are never threaded automatically, the history pane labels each threaded take and names the active thread, and "start new thread" only clears the active-thread hint so the next threaded refine opens a fresh one — no session is mutated or deleted. Every turn keeps its own untouched raw transcript; each turn's refined copy is the thread's state at that turn.

## Run the browser preview

Start `starling-serve` on `127.0.0.1:8181`, then from the repository root run:

```bash
pnpm install
pnpm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8181`, including WebSocket upgrades on `/api/stream` for live streaming. Set `STARLING_API_TARGET` before the dev command to change the proxy destination.

## Run the native app

Install the workspace dependencies, then run the Electron shell with live renderer reload:

```bash
pnpm run desktop
```

- macOS asks for microphone access using the packaged `NSMicrophoneUsageDescription`.
- Windows microphone privacy settings must allow desktop apps.
- Linux needs a working PipeWire or PulseAudio microphone source.

The native app sends batch requests through its sandboxed Electron preload bridge, so a local server does not need browser CORS headers. The live streaming socket connects directly from the renderer, which works for loopback endpoints; WebSockets carry no credentials, so authenticated remote proxies should keep streaming off and use batch uploads. `Cmd+Shift+Space` on macOS or `Ctrl+Shift+Space` on Windows/Linux focuses Starling and toggles recording. It does not inject text into another app; copying and export are explicit actions.

Create an unpacked application directory with `pnpm --filter @starling/desktop package --dir`, or build the platform installer with `pnpm --filter @starling/desktop package`. Packaging targets macOS, Windows, AppImage, and Debian packages. Each installer is built on its native host.

Batch-mode audio (imports, or recordings with live streaming disabled or unavailable) is held in memory while the microphone is live and becomes durable when recording stops, before the upload starts; a process or device crash during that capture still cannot be recovered in this foundation. Live-streamed recordings journal each chunk to IndexedDB as it is captured, so a crash mid-recording leaves recoverable audio, and journals are recovered as retryable sessions on the next start.

If local storage fails, Starling keeps each unsaved WAV in memory and shows a
download button for it. Downloading does not discard that copy: a download can
be cancelled. The copy remains until you choose Discard unsaved or close the app.

Set `STARLING_DIAGNOSTICS=1` while launching a development build to print measured time-to-ready and per-process memory in the terminal. These figures vary by OS, GPU process, and machine.

The renderer and Electron shell use TypeScript 7 and Effect v4. See
[TypeScript development](../../docs/typescript.md) for the pinned release candidate,
Oxlint and anti-slop rules, formatting, and validation commands.
