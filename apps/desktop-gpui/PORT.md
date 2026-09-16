# Starling desktop — gpui port

A native (Rust + [gpui](https://github.com/zed-industries/zed/tree/main/crates/gpui))
port of the Electron desktop app (`apps/desktop`), for side-by-side comparison of
look, feel, and performance. The TypeScript sources of truth live in this same
worktree: `packages/dictation/src/*.ts` (logic), `apps/desktop/src/App.tsx` +
`src/useRecorder.ts` + `src/styles.css` (UI), `apps/desktop/electron/*.ts`
(native shell + native HTTP path).

## Layout

- `crates/dictation` (`starling-dictation`) — pure logic + IO, no gpui. Ports of
  `@starling/dictation` and of the Electron main-process request behavior:
  - `audio.rs` ← `packages/dictation/src/audio.ts`
  - `client.rs` ← `apps/desktop/electron/ipc.ts` request programs + `client.ts` response normalization
  - `fidelity.rs` ← `packages/dictation/src/fidelity.ts`
  - `storage.rs` ← `packages/dictation/src/storage.ts` (file-backed instead of IndexedDB)
  - `settings.rs` ← the `localStorage` keys used by `App.tsx`
  - `fft.rs` — frequency magnitudes for the waveform (replaces `AnalyserNode`)
  - `recorder.rs` — mic capture (replaces `getUserMedia`/`ScriptProcessorNode`)
  - `player.rs` — WAV playback (replaces the `<audio>` element)
- `crates/app` (`starling-gpui`) — gpui UI, no logic beyond wiring.

## Build / test

The `rustup` shims misbehave in this environment (argv[0] confusion). Always run
cargo via the toolchain path:

```bash
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
cd /home/m0hawk/Documents/starling-gpui/apps/desktop-gpui
cargo test -p starling-dictation        # lib tests, fast, no gpui
cargo run -p starling-gpui              # launch the app (debug)
cargo run -p starling-gpui --release    # launch for perf comparison
STARLING_DIAGNOSTICS=1 cargo run -p starling-gpui   # startup + RSS on first draw
```

Workspace members are `crates/*` only. The pnpm workspace ignores this dir (no
package.json). Never touch files outside `apps/desktop-gpui`.

## Behavior contract

### Transcription client (mirror the Electron native path, not the browser one)

`StarlingClient { base_url, protocol, model, timeout }`, errors via `thiserror`
(`Input`, `Transport`, `Timeout { timeout_ms }`, `Http { status, message }`,
`Protocol`). Endpoint must be http(s) URL without credentials; trailing `/`
trimmed. Timeouts: default 180s, allowed 1ms..=600s. Redirects are blocked
(`redirect: manual` in TS): any 3xx is an error "Server redirect blocked (…)".

- starling protocol:
  - `health`: `GET {base}/health` → `{status, phase?, model?, loaded?, busy?, queue_depth?}`
  - `transcribe`: `POST {base}/transcribe` with raw WAV body, headers
    `x-request-id: {id}` + `content-type: audio/wav`
- openai protocol:
  - `health`: `GET {base}/v1/models` → `{data: [{id}, …]}`; map to health
    `{status: "ok", phase: "ready", busy: false, model: first id}`
  - `transcribe`: `POST {base}/v1/audio/transcriptions` multipart:
    `file` = `recording.wav` (audio/wav), `model` (default `parakeet`),
    `response_format=json`
- transcribe response (both): `{text, segments?: [{text, start_s?|start?, end_s?|end?}], duration_s?|duration?, request_id?}`;
  normalize to `TranscriptionResult { text, segments: Vec<TranscriptionSegment{text, start_seconds, end_seconds}>, duration_seconds: Option<f64>, request_id: Option<String> }`
  (fallback request id: body → `x-request-id` header → the sent id). Invalid
  timestamps (missing, negative start, end < start) and negative duration are
  `Protocol` errors.
- request id rules: non-empty, no `\r`/`\n`, must not start with `#`.
- audio payload limits: >= 44 bytes and <= 256 MiB (`Input` error otherwise).
- HTTP error body detail extraction order: `{detail}` → `{error: string}` →
  `{error: {message}}` → `{message}`; else `Server returned {status}: {first 500 chars}`.

### Audio (mirror audio.ts exactly: dependency-free, deterministic)

`PcmAudio { samples: Vec<f32>, sample_rate: u32, channels: u16 }` (interleaved
-1..=1). `mix_to_mono`, `resample_to_16k` (linear interpolation, same rounding),
`encode_wav_16k` (canonical 44-byte header, PCM16, 16 kHz mono, asymmetric clamp:
negatives `*0x8000`, positives `*0x7fff`, round-half-away like JS `Math.round`),
`decode_pcm16_wav` (PCM-only, 16-bit, RIFF chunks walked with padding, same
error messages), `prepare_wav_16k(bytes) -> PreparedWav { wav: Vec<u8>,
duration_ms, duration_seconds }`.

### Storage (file-backed IndexedDB equivalent)

Root: `dirs::data_dir()/starling-gpui/sessions`. One dir per session:
`{id}/manifest.json` + `{id}/recording.wav`. Manifest mirrors
`DictationSessionManifest` (schemaVersion 1, audioFile "recording.wav",
camelCase JSON). Session ids are UUID v4. Statuses: `captured | transcribing |
transcribed | failed`. Store API: `create`, `get`, `list` (sorted by
`updated_at` desc), `mark_attempt` (status→transcribing, attempt_count+1,
clear last_error), `save_transcript` (status→transcribed; previous transcript
appended to `transcript_history`), `save_failure` (status→failed, last_error),
`delete`. Writes atomic (tmp + rename). `updated_at`/`created_at` are RFC3339
UTC strings with millisecond precision (JS `toISOString()` equivalent).

### Settings (localStorage equivalent)

`dirs::config_dir()/starling-gpui/settings.json`:
`{ endpoint, protocol, model, expected_terms }`. Defaults mirroring App.tsx:
endpoint `http://127.0.0.1:8181`, protocol `starling`, model `parakeet`,
expected terms `["auth"]` (input UI is the comma-joined string). Invalid file →
defaults (do not crash).

### Fidelity (mirror fidelity.ts)

`analyze_transcript(text, AnalysisOptions { expected_terms }) -> Vec<FidelityWarning { code, severity, message }>`
with the same codes/messages: non-one-list-start, possible-self-correction
(`er|uh|um` words), negation-present, discourse-word-preserved (`like`),
short-answer (single letter/word incl. agreed|yes|no|okay|ok), expected-term-
missing (case/NFKC-insensitive contains). Only the subset the UI consumes.

### Recorder (replaces useRecorder.ts)

Mono capture via cpal (request f32, 16 kHz if the device allows; else native
rate + resample on stop). While recording, produce two streams for the UI:
cumulative samples (for the final WAV) and recent-window frequency levels for
52 waveform bars. Waveform mapping mirrors the TS analyser loop: magnitude
bins normalized to 0..1, `stride = max(1, bins/52)`, value =
`max(0.045, v.powf(1.45))`, bars 2px wide, idle floor `0.06`.

### Player

`play(wav_bytes)` / `stop()` / `is_playing()` via rodio (WAV only).

## UI contract (gpui crate)

Match `apps/desktop/src/styles.css` + `App.tsx` closely:

- Window: title "Starling", 1180×760, min 920×620, bg `#171813`.
- Tokens: ink `#e8e5dc`, muted `#98998e`, dim `#6e7067`, panel `#20211c`,
  line `rgba(232,229,220,0.105)`, lime `#d9ff6a`, coral `#ff745b`,
  paper `#e8e4d9`/`#efebe0`, status amber `#efc26b`, idle dot `#7d7e75`.
- Display serif for headings/transcript ("Iowan Old Style" → Georgia → serif
  fallback), sans for body, mono for eyebrows/meta (`SFMono-Regular`/monospace).
- Structure: 68px topbar (brand, centered connection status + pulsing dot,
  settings icon-button) / workspace grid (capture pane min 560px + 350px
  history column) / transcript drawer (light paper, min 210px, max 250px) when
  a session is selected. Settings modal centered over blurred dark scrim.
- Capture pane: eyebrow "LOCAL DICTATION", h1 "Say it as you mean it." /
  "Listening closely.", lede copy, 116px round record button (lime disc; coral
  stop square while recording), animated 52-bar waveform behind it (dim bars at
  45% opacity idle, lime at 84% while recording, edge-faded), elapsed m:ss,
  "⌘/Ctrl Shift Space" kbd hint, "Import an audio file" (.wav) via file dialog.
- History: 92px head ("ARCHIVE"/"Recent takes"), rows min-height 78px with
  status dot (lime transcribed / coral failed / spinner transcribing), snippet
  text, `time · duration[ · N attempts]`, active row lime inset bar.
- Transcript drawer: grip, "RAW TRANSCRIPT / No cleanup or silent rewriting",
  actions Retry (when not transcribed/transcribing), WAV (export audio), Copy
  (→ "Copied" 1.4s), Export (.txt download), Delete (danger). Body: audio
  playback control, transcript in serif 18–27px, spinner/failed copy states,
  fidelity warnings strip below (amber `#705a2b` on paper).
- Error banner: bottom of capture pane, `#39241f` bg / `#ffac99` text, with
  unsaved-recording recovery buttons when storage failed.
- Settings modal: endpoint input, protocol select (Starling native / OpenAI
  compatible), model input, "Words to watch" input, connection callout, Test
  connection + Save.
- Keys: in-app `cmd/ctrl-shift-space` toggles recording; system-wide hotkey via
  `global-hotkey` (best effort on Wayland — warn and continue like Electron's
  `console.warn`). Import via `prompt_for_paths`. Clipboard via gpui.
- Diagnostics: with `STARLING_DIAGNOSTICS=1`, print (stdout) on first frame:
  `STARLING_DIAGNOSTICS {"readyToShowMs":…,"rssBytes":…}` (RSS from
  `/proc/self/status` VmRSS; single-process app).

## Parity deviations (intentional, keep this list)

- IndexedDB → on-disk session store (same manifest schema).
- `AnalyserNode` → local FFT; visual mapping kept, not bit-identical.
- Electron multi-process metrics → single process RSS.
- Global hotkey on Wayland is best-effort (portal limitations); Electron has
  the same class of issues via X11-only paths.
