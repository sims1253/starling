# Starling desktop

The desktop client records mono microphone audio, converts it to PCM16 16 kHz WAV, saves the WAV locally in IndexedDB, and then sends it to the selected server. A failed or interrupted request leaves the recording available to play, download, and retry. Transcript text is displayed and exported exactly as returned by the server.

## Run the browser preview

Start `starling-serve` on `127.0.0.1:8181`, then from the repository root run:

```bash
pnpm install
pnpm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8181`. Set `STARLING_API_TARGET` before the dev command to change the proxy destination.

## Run the native app

Install the workspace dependencies, then run the Electron shell with live renderer reload:

```bash
pnpm run desktop
```

- macOS asks for microphone access using the packaged `NSMicrophoneUsageDescription`.
- Windows microphone privacy settings must allow desktop apps.
- Linux needs a working PipeWire or PulseAudio microphone source.

The native app sends requests through its sandboxed Electron preload bridge, so a local server does not need browser CORS headers. `Cmd+Shift+Space` on macOS or `Ctrl+Shift+Space` on Windows/Linux focuses Starling and toggles recording. It does not inject text into another app; copying and export are explicit actions.

Create an unpacked application directory with `pnpm --filter @starling/desktop package --dir`, or build the platform installer with `pnpm --filter @starling/desktop package`. Packaging targets macOS, Windows, AppImage, and Debian packages. Each installer is built on its native host.

Audio is held in memory while the microphone is live and becomes durable when recording stops, before the upload starts. A process or device crash during an active recording cannot be recovered in this foundation.

If local storage fails, Starling keeps each unsaved WAV in memory and shows a
download button for it. Downloading does not discard that copy: a download can
be cancelled. The copy remains until you choose Discard unsaved or close the app.

Set `STARLING_DIAGNOSTICS=1` while launching a development build to print measured time-to-ready and per-process memory in the terminal. These figures vary by OS, GPU process, and machine.

The renderer and Electron shell use TypeScript 7 and Effect v4. See
[TypeScript development](../../docs/typescript.md) for the pinned release candidate,
Oxlint and anti-slop rules, formatting, and validation commands.
