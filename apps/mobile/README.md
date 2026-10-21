# Starling Mobile (Android)

Starling Mobile is a small native Android app that transcribes speech on
the phone. It records 16 kHz mono PCM16 microphone audio, writes a WAV file
in app-private storage, and only then transcribes it with the on-device
engine. The exact text the model returns is stored as the raw transcript.
Nothing in the app fixes numbering, removes filler words, or silently
rewrites words such as `not`, `like`, or `auth`. Audio never leaves the
phone; the network is used only to download models.

While recording, the same 16 kHz chunks are also transcribed live: growing
partial transcripts appear in the recorder, become composing text in the voice
keyboard, and are delivered as partial results through the system recognizer,
and Stop finalizes the live session for the final transcript.

The same capture flow is available from the standalone app and the optional
Starling Voice Input keyboard. The keyboard shows a transcript first; it only
calls `InputConnection.commitText` after the user taps **Insert transcript** —
except while a live session is running, where growing partials are shown as
composing text (the Android dictation idiom) and the final text is committed
once, on Stop, into the field the recording started in. The field's declared
attributes (package, id, input type, hint) are compared in memory only, to
recognize the same field after a screen lock or app switch.

Every write follows the insertion-boundary rules
(`packages/contracts/insertion-boundary/`, #341): right before the one
`commitText`, the keyboard reads the text before the cursor
(`getTextBeforeCursor`, 128 characters; no rule needs the text after it, so
that is never read) and
adds a leading space or lowercases the first letter when the text continues
a sentence; live composing text gets the same boundary. Words are never
changed, code-looking first words keep their case, and a field showing only
its hint reads as empty (an `InputConnection` reports content, never the
placeholder). Password and incognito fields
(`IME_FLAG_NO_PERSONALIZED_LEARNING`) are never read, verbatim modes write
the text as recognized, and a field that does not report its text gets it as
dictated (the status says so, and says when the cursor left the live text
instead). The text read is used for this
decision only; it is not stored or sent anywhere. An adjusted insertion is
kept on the recording as a derived revision ("Inserted with boundary fixes"
in Saved recordings); the transcript itself stays as recognized. An editor
target generation and connection identity check prevents a late
transcription from being inserted into a field that has changed.

## Prerequisites

- Android Studio Ladybug or newer, or JDK 17 with the Android SDK.
- Android SDK platform 35 and build-tools 36.0.0 installed.
- A device or emulator with a microphone. Android 8.0 / API 26 is the minimum.

The app embeds the repository's native engine (`libstarling_ggml`). Tap
**Download recommended model** to fetch
`parakeet-tdt-0.6b-v3-q4pw-fullimx.gguf` (364 MB) from
[`scholzmx/parakeet-tdt-0.6b-v3-gguf`](https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf),
or **Download another model…** for the phone quants of the Underdog Bark
3.0 family (Pro, Clean, Plus, 3.0: English Parakeet fine-tunes in the same
364 MB layout), Qwen3-ASR (0.6B, 1.7B), ARK-ASR 0.6B, Granite Speech and
MOSS-Transcribe (see the *Phone recipes* section of `docs/quantization.md`
and [models](../../docs/models.md)). Every download is pinned to a
repository revision, resumes where it stopped after a dropped connection
or app restart, and is checked against its SHA-256 before it replaces the
active model (`engine/ModelCatalog.kt`). Any other Parakeet-TDT GGUF can
be imported through **Import model (.gguf)** instead, and so can the
repository's other speech-model GGUFs: Qwen3-ASR (1.7B, 0.6B), ARK-ASR
(3B, 0.6B), Granite Speech and MOSS-Transcribe. The engine is picked from
the file's own metadata (`engine/ModelKind.kt`); anything else, including
the S1-mini text model, is refused at import. These LLM-decoder models are
several times larger and slower than Parakeet on a phone, and their load
is memory-gated with a larger working-set estimate (a KV cache on top of
the weights); they transcribe long recordings in 30 s windows, and live
previews of a slow model are spaced so they keep the engine busy at most
half of the time. Every model (Parakeet, Qwen3-ASR, MOSS, ARK and Granite)
runs on the GPU (Vulkan fast engines) where the phone has one, and on the
CPU otherwise. GPU inference is both faster and several times more
energy-efficient per transcription than the CPU engine. The model is
loaded on first use. When a transcription fails — missing model, load
failure — the failure is surfaced and the local recording remains
available for retry.
Building the app additionally requires the Android NDK and CMake SDK packages.

Models live in app-private storage until you tap **Keep models across
reinstalls** and allow Android's *All files access* for Starling (Android
11+). From then on they are kept in `Starling/models` on the phone's shared
storage (`ModelStorage.kt`); the models already installed move there. That
folder survives an uninstall, and every Starling build on the phone —
release, experimental, debug — uses the same copy, so a reinstall finds its
models without downloading them again. Which model is active stays per
install. Deleting a model there removes it for every build.

The arm64 engine is built for ARMv8.2 with the dot-product and FP16
extensions (every mainstream phone core since 2018). On older CPUs the app
reports that it cannot transcribe instead of loading the library. The
engine uses the performance cores only, and the loaded model is released
when Android reports memory pressure (it reloads on the next use); a load
that clearly cannot fit in free memory is refused with an explanation.

## Install a release build

For the latest master build on Pixel 10 Pro, use the arm64/i8mm APK from
[Experimental releases](https://github.com/sims1253/starling/releases?q=experimental&expanded=true).
It installs as **Starling Experimental**, beside the regular app, and preserves
its own data across experimental updates. See the
[experimental installation guide](../../docs/experimental-releases.md).

Tagged releases carry two signed APKs, built by
`.github/workflows/release-android.yml`:

- `starling-mobile-<version>.apk` runs on every arm64 phone since about 2018.
- `starling-mobile-<version>-i8mm.apk` is the same app whose on-device engine
  also uses the int8 matrix-multiply instructions (faster Q4_K/Q6_K/Q8_0
  matmuls): Pixel 8 or newer (Tensor G3+), Snapdragon 8 Gen 1 or newer,
  Dimensity 9000 or newer. On other phones it refuses on-device
  transcription with an explanation; use the standard APK there. Both share
  the application id and key, so either installs over the other. Build it
  locally with `-PstarlingArmArch=armv8.2-a+dotprod+fp16+i8mm`.
 Open the release page on the phone,
download the APK, and open it; Android asks once to allow installs from the
browser or file manager. A `v*` tag attaches the APK to the regular release;
an `android-v*` tag (for example `android-v0.1.0`) makes an APK-only
prerelease without building the server binaries. Every CI run of the
*Portable apps and contracts* workflow also uploads a debug APK as the
`starling-mobile-debug-apk` artifact; its native engine is unoptimized, so use
a release APK to judge on-device speed.

Releases are signed with the key in the repository secrets
`STARLING_ANDROID_KEYSTORE_BASE64`, `STARLING_ANDROID_KEYSTORE_PASSWORD`,
`STARLING_ANDROID_KEY_ALIAS`, and `STARLING_ANDROID_KEY_PASSWORD`. Create the
key once and keep it: Android only installs an update over an app signed by
the same key.

```bash
keytool -genkeypair -keystore starling-release.jks -storetype PKCS12 \
  -alias starling -keyalg RSA -keysize 3072 -validity 10000 \
  -dname "CN=Starling Mobile"
gh secret set STARLING_ANDROID_KEYSTORE_BASE64 < <(base64 -w0 starling-release.jks)
gh secret set STARLING_ANDROID_KEYSTORE_PASSWORD
gh secret set STARLING_ANDROID_KEY_ALIAS --body starling
gh secret set STARLING_ANDROID_KEY_PASSWORD   # same as the store password for PKCS12
```

Without these secrets, `v*` releases fail to build, and `android-v*` test
builds are signed with a throwaway key per release, so installing a later one
needs an uninstall first (which deletes the app's recordings, and its models
unless they are kept in shared storage).

Local release builds read the same four values from `STARLING_ANDROID_*`
environment variables (`STARLING_ANDROID_KEYSTORE` is the keystore path);
without them `assembleRelease` produces an unsigned APK.

## Build and run

For the recorder's UI acceptance test and independently verifiable audio
evidence, see [Run the Android acceptance journey](e2e/README.md).

From this directory:

```bash
./gradlew test
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Debug builds install as **Starling Debug** (application id
`dev.starling.mobile.debug`) beside a release or experimental build, so
device tests never touch the app you use.

Open **Starling Mobile**, grant microphone access, and download or import a
model. Recordings are retained in the app's private files directory. Each
item keeps its WAV, state, exact raw transcript, its provenance (streamed
live or transcribed after Stop), and any error. **Retry** is available from
the recording list, with any installed model. Deleting an item requires an
explicit confirmation and removes its WAV and transcript from this app.

To enable the keyboard, tap **Enable Starling keyboard**, enable Starling Voice
Input in Android settings, then select it from the keyboard switcher. The
**⌨** button switches back to the previous keyboard, so Starling also works as
the voice key of typing keyboards that switch to a voice input method
(HeliBoard, FUTO Keyboard). Without the microphone permission, Record opens
Starling to ask for it and closes again.

A keyboard take keeps recording while the keyboard is hidden — screen lock,
app switch, another field — through a microphone foreground service; its
notification carries **Stop** (Android 13+ asks once for notifications for
this). Hiding the keyboard, or leaving the field, removes the take's
composing text while the keyboard can still reach the field; the final is
then committed in one piece. (An app that closes its input connection before
Android tells the keyboard the field is gone can finish the partial as
ordinary text itself — the same outcome as switching fields mid-take always
had.) While the field keeps focus, live text continues. When a field with the same
declared attributes comes back, the take attaches to it but writes nothing
by itself any more (two chats can share one field layout): its live text stays
in the keyboard and the final waits for **Insert transcript**. In any other
field the transcript can only be copied, and it stays in Starling.

Private fields — password input types and fields that set
`IME_FLAG_NO_PERSONALIZED_LEARNING`, such as incognito browser tabs — get an
ephemeral take: it never appears in Saved recordings, its WAV and transcript
are deleted as soon as it settles (a crash leftover is deleted on the next
start), its text is hidden in the keyboard outside its own field, and a
copied transcript is marked sensitive.

### Modes and staged dictation

The keyboard's mode chip (or a long-press on Record) picks the mode for the
next takes. The modes are an ordinary profiles document
(`app/src/main/assets/modes/android-profiles.json`), held to the shared
contract in `packages/contracts/mode-routing/` by the Rust `staging_conformance`
suite (`apps/desktop-gpui/crates/processing/tests/`):

- **Direct** (default): the flow above — live text in the field, the
  verbatim final committed on Stop.
- **Draft**: Stop does not insert. The take waits in an editable draft above
  the keyboard: the live tail while speaking, then the text with spoken
  punctuation and layout commands ("comma", "new line", "bullet") applied as
  a processed proposal. **Show raw** flips to the recognition exactly as it
  arrived; tapping a word selects it for **Delete word**, or for a spoken
  correction (tap Record with the word selected and say what replaces it);
  Record again adds to the draft; only **Insert** writes, once, into the
  take's field — and if the field refuses the text, the draft stays and
  nothing is sent. Typed edits need a typing keyboard: insert, then switch
  with **⌨**.
- **Message**: like Draft, but Insert also presses the field's own action
  (Send, Search, Go) through `performEditorAction` — never a synthetic Enter.

A leading phrase picks a mode for one take ("draft mode …", "message mode
…"; "literal …" keeps the rest verbatim), and a trailing "Starling, …"
instruction is recognized; both follow the contract grammar and are shown
struck through in the draft, never sent to the field, with **That was
literal** to undo the decision. Until the first words are clearly not a
phrase, direct mode holds them back from the field, and from a possible
"Starling" delimiter on, nothing more is composed into it. An instruction needs a
text model, which the phone does not have yet (#295), so it is only set
aside. Private fields never stage, route or process.

Processing is the deterministic rules step only, on the device, in
microseconds; the draft's status shows its time, and `adb logcat -s
StarlingTiming` logs stop→raw and stop→processed per take for device
measurements. A mode that asks for a model step runs rules only (and says so,
also under battery saver). The draft, routing and rules are Kotlin ports
(`processing/`) that replay the same fixtures as the Python oracles and the
desktop's Rust ports; so does the insertion-boundary port
(`processing/InsertionBoundary.kt`). The draft lives in memory: the raw recording is in
Saved recordings as always, but edits are lost if Android kills the keyboard.

### Live transcription and its fallback

The WAV written during capture stays the source of truth: the live session
is an observer of the same chunks, never a gate on them. On Stop the WAV is
finalized and committed first, and only then is the session finished for its
final transcript. If it fails at any point (the model cannot load, a window
fails, or the engine falls too far behind the recording), the session is
dropped, the interruption is shown, and the already-saved WAV is transcribed
after Stop instead. A live failure therefore loses nothing: the recording,
retry, and failure story is indistinguishable from the non-live flow. In the
voice keyboard a failed session removes its composing text and falls back to
the explicit **Insert transcript** flow.

### Voice input and Android settings

Starling also registers as a system speech recognizer
(`StarlingRecognitionService`). Keyboards that use Android's `SpeechRecognizer`
API — for example the open-source HeliBoard, OpenBoard, or AnySoftKeyboard —
then show their own microphone button and transcribe through Starling, without
switching keyboards. Android may offer a speech recognition service picker in
some contexts, but the Pixel **Default digital assistant app** screen is a
different setting and does not list Starling. Enable **Starling Voice Input**
as a keyboard for dictation in text fields. Closed keyboards such as Gboard
or SwiftKey keep their bundled engines and cannot delegate to a custom
recognizer. Growing partials are delivered through the platform's
`partialResults` callback while you speak; the final result is still
delivered once, after the live session finishes or the transcription of the
saved WAV completes. Every dictation stays visible
in Saved recordings, including failed transcriptions, so it can be retried or
deleted there.

Apps that ask the system for speech input with
`RecognizerIntent.ACTION_RECOGNIZE_SPEECH` (a microphone button that opens a
"speak now" popup) can pick Starling too: a small dialog starts listening at
once, shows live partials, and returns the verbatim transcript when you tap
**Done**. **Cancel** keeps the audio in Saved recordings without transcribing
it.

## Scope and limitations

This directory contains the Android app. The sibling standalone iOS recorder is
at `../ios`, and the desktop client is at `../desktop`. The
Android app does not provide an iOS keyboard extension; its system-wide voice
keyboard is Android-specific. Push-to-talk hardware integration remains a
future extension. Live transcription uses the same window geometry as the
native server's `WS /stream` (12 s windows, 3 s overlap, word stitching)
with partials about every second. The app does not ship model weights or
third-party/copyrighted application code.
