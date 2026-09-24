# Starling monorepo

Starling is a native speech engine, a portable serving API, and a set of clients.
CPU is the baseline. Platform accelerators are optional implementations behind
the same native engine boundary. The NVIDIA-only Python/CUDA service is deprecated.

## Components

```mermaid
flowchart LR
  Desktop["Desktop · Rust gpui<br/>Windows · Linux · macOS"] --> API["Starling native server<br/>OpenAI-compatible batch + Starling streaming"]
  Android["Android · Kotlin<br/>Recorder + voice keyboard"] --> API
  iOS["iOS · SwiftUI<br/>Recorder + share / copy"] --> API
  Shared["Shared wire contract<br/>TypeScript client + fidelity rules"]
  Shared -. protocol .-> Desktop
  Shared -. protocol .-> Android
  Shared -. protocol .-> iOS
  Quants["GGUF recipes + artifact records"] --> Engine["Starling ggml engines<br/>CPU · Metal · Vulkan · HIP · optional CUDA"]
  API --> Engine
```

Each build system owns its natural boundary. Cargo owns the gpui desktop app.
pnpm workspaces link the TypeScript library and the npm launcher. CMake owns
the native server and quantizer.
Gradle owns Android. XcodeGen and Swift Package Manager own iOS. The small Python
quant catalog is independent of the deprecated Python inference environment.

| Path | Ownership |
| --- | --- |
| `apps/desktop-gpui` | Desktop app (Rust, gpui): UI, microphone capture, journaled local history, HTTP client, global shortcut |
| `apps/mobile` | Android activity, recorder, private storage, network client, voice keyboard |
| `apps/ios` | iOS recorder, private history, native HTTP client, share/copy |
| `packages/dictation` | WAV preparation, TypeScript client, immutable analysis, browser session persistence (reference library; the Rust desktop re-implements this contract) |
| `packages/contracts` | Language-independent OpenAPI contract |
| `backends/native` | Native CMake build definition and real HTTP contract tests |
| `quants` | Recipe catalog, calibrated recipes, artifact provenance CLI |
| `backends/python` | Deprecated service entry point and migration notes |
| `cpp`, `src/starling`, `scripts`, `benchmarks` | Existing engine source, reference kernels, converters, evaluation harnesses |

The root `CMakeLists.txt` forwards to `backends/native`. It preserves existing
build commands and binary output locations. Native source stays in `cpp/` to
preserve C API consumers and harness paths. The root `pyproject.toml` and
`uv.lock` remain the reference Python environment. Quant recipes have moved to
`quants/recipes`; historical measurement JSON retains its original provenance.

## Platform targets

| Target | Implementation | Verification needed on hardware |
| --- | --- | --- |
| Windows | gpui desktop + native CPU server | Microphone permissions, shortcut, installer |
| Linux | gpui desktop + native CPU/Vulkan server | X11/Wayland audio and shortcut behavior, packaging |
| macOS | gpui desktop + native CPU/Metal server | Microphone prompt, global shortcut, signing and notarization |
| Android | Native Kotlin recorder and voice keyboard | Microphone/device lifecycle, IME behavior, LAN connectivity |
| iOS / iPadOS | Native SwiftUI recorder | Microphone interruptions, local-network prompt, audio routes, signing |

These are development targets, not five published store releases. The gpui
workflow runs the Rust test suite on Linux, compile-checks the Windows service
host, and packages unsigned macOS (universal DMG) and Windows (portable
archive) builds on every push. `apps.yml` builds Android and builds
iOS for the simulator. Physical-device recording and distribution remain
separate acceptance checks. A CI configuration is not evidence of a successful
run until that workflow executes.

Apps connect to a server you start yourself. They do not bundle models or start
an inference sidecar yet. On a phone, `localhost` means the phone; use your
computer's reachable endpoint or a development tunnel. HTTP on a trusted LAN
requires explicit mobile opt-in. Production remote access belongs behind an
authenticated HTTPS proxy.

Android implements an explicit Insert action in its voice keyboard. iOS ships
a standalone recorder with share/copy, not a keyboard extension with microphone
access. Desktop exposes copy/export; cross-application text insertion is a
separate platform integration.

## Fidelity requirements

Audio is durable before the app relies on a server. Batch uploads start only
after the completed WAV is saved; live streaming captures journal every chunk
to storage as it arrives, offer a chunk to the server only after that chunk is
journaled, and finalize the WAV before the streamed result is accepted as the
transcript. A network failure changes the session status but does not remove
the WAV.

```mermaid
sequenceDiagram
  participant User
  participant App
  participant Store as Private storage
  participant Server
  User->>App: Stop recording
  App->>Store: Save completed WAV
  Store-->>App: Durable session
  App->>Store: Record transcription attempt
  App->>Server: Upload saved WAV
  alt Transcription succeeds
    Server-->>App: Raw transcript
    App->>Store: Save raw text and prior transcripts
  else Request fails
    Server-->>App: Error or timeout
    App->>Store: Save error and keep WAV
    User->>App: Retry
    App->>Store: Record retry attempt
    App->>Server: Upload the same saved WAV
  end
  User->>App: Delete session
  App->>Store: Remove audio and text
```

### Streaming captures

The desktop app streams audio to the server's `WS /stream` endpoint while
the microphone is live and shows partial transcripts. The streaming path keeps
the same guarantees:

1. The capture is converted to 16 kHz mono PCM16. Only samples the durable
   journal has acknowledged are sent to the server.
2. On Stop the journal is assembled into the canonical WAV and the session is
   persisted before the client asks the server to commit. The streamed final
   is accepted only if the socket carried the whole take without errors.
3. If anything fails mid-take — the socket drops, the server's live buffer
   fills, journaling hits a storage error — the recording continues locally,
   the failure is recorded on the session, and Stop falls back to the batch
   upload of the saved WAV. A streaming failure can shorten nothing; at worst
   it costs the live preview.
4. Journals orphaned by a crash mid-recording are recovered on the next start
   as retryable sessions with their audio intact.

1. Keep the raw recognition text. Any future cleanup result is a separate,
   explicitly reviewed edit.
2. Preserve list numbers, negations, domain terms, corrections, short answers,
   and deliberate words such as "like". Do not treat a one-letter answer as noise.
3. Persist completed audio before requesting transcription. Keep it through
   errors, retry, and text export; deleting a recording must be explicit.
4. Show failures. An empty transcript is a result to review, not evidence of silence.
5. Do not claim vocabulary biasing unless the chosen engine implements it.
6. Keep reported audio duration separate from recognized content coverage.
   Segment boundaries often describe chunks, not word-level coverage.

The gpui desktop mirrors every captured take into an append-only journal with
fsynced boundaries while the microphone is live, so a crash mid-recording
leaves recoverable audio up to the last acknowledged boundary; recovery runs
on the next start. Native mobile recording writes audio into app-private
files. None of these paths promises recovery from every OS or storage
failure. Use saved audio exports for recordings that must outlive the app.

The text fixtures cover the six failure modes raised in the original request.
They test preservation and recovery behavior, not real-world recognition
accuracy. Release evaluation still needs real audio containing these cases,
including a ten-minute session with short answers and interrupted uploads.

## Design references

These projects inform the architecture and interaction design. Their code is
not copied into the apps or substituted for Starling's own engine.

| Reference | Design question |
| --- | --- |
| [FluidVoice](https://github.com/altic-dev/FluidVoice), [Handy](https://github.com/cjpais/handy), [Freestyle](https://github.com/sims1253/freestyle/tree/dev), [Hex](https://github.com/anomalyco/hex) | How can desktop capture, shortcuts, preview, and recovery feel immediate? |
| [Android Transcribe](https://github.com/notune/android_transcribe_app), [FUTO Voice Input](https://github.com/futo-org/voice-input) | How should recorder and voice-keyboard workflows differ? |
| [transcribe-rs](https://github.com/cjpais/transcribe-rs), [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) | How should model and hardware implementations share a stable interface? |
| [llamafile](https://github.com/mozilla-ai/llamafile) | How can native serving and model distribution become easier to install? |
| [handy-keys](https://github.com/handy-computer/handy-keys) | How should shortcuts behave across platform constraints? |
| [Wispr Flow](https://wisprflow.ai/), [Monologue](https://www.monologue.to/) | Where do polished dictation workflows help, and where can automatic edits undermine trust? |

## Next product milestones

- Bundle and supervise a local server, with model installation and verified
  artifacts, so desktop setup does not require a terminal.
- Add recording checkpoints and durable native desktop storage for recovery
  during long captures, plus explicit retention limits and disk-space handling.
- Evaluate real audio for the six fidelity cases across models and quants.
- Add capability-backed vocabulary biasing and reviewable cleanup proposals.
- Implement and test platform-specific insertion, overlays, and phone handoff.
- Produce signed installers, Android distribution builds, and iOS device builds.
