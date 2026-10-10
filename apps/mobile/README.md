# Starling Mobile (Android)

Starling Mobile is a small native Android client for a Starling server that
you run yourself. It records 16 kHz mono PCM16 microphone audio, writes a WAV
file in app-private storage, and only then sends a multipart transcription
request. The exact `text` returned by the server is stored as the raw
transcript. Nothing in the client fixes numbering, removes filler words, or
silently rewrites words such as `not`, `like`, or `auth`.

While recording, the same 16 kHz chunks are also transcribed live: growing
partial transcripts appear in the recorder, become composing text in the voice
keyboard, and are delivered as partial results through the system recognizer,
and Stop finalizes the stream for the final transcript. Live transcription
runs either on this device (the on-device engine with an imported model) or
over the native server's `WS /stream` protocol (the Starling protocol on a
remote server, under the same trusted-host rules as the batch client). The
OpenAI-compatible route has no streaming; it records exactly as before and
transcribes after Stop.

The same capture flow is available from the standalone app and the optional
Starling Voice Input keyboard. The keyboard shows a transcript first; it only
calls `InputConnection.commitText` after the user taps **Insert transcript** —
except while a live stream is running, where growing partials are shown as
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
target generation and connection identity check prevents a late network
response from being inserted into a field that has changed.

## Prerequisites

- Android Studio Ladybug or newer, or JDK 17 with the Android SDK.
- Android SDK platform 35 and build-tools 36.0.0 installed.
- A running Starling native server (`starling-serve`). Native serving is the
  supported portable path and requires a 16 kHz WAV. The Python/CUDA server is
  retained for compatibility but is deprecated for new app deployments.
- A device or emulator with a microphone. Android 8.0 / API 26 is the minimum.

No Python or GPU runtime is bundled. An optional **on-device engine**
(experimental) embeds the repository's native Parakeet engine
(`libstarling_ggml`) for offline transcription. Tap **Download recommended
model** to fetch `parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf` (553 MB) from
[`scholzmx/parakeet-tdt-0.6b-v3-gguf`](https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf).
The download is pinned to a repository revision, resumes where it stopped
after a dropped connection or app restart, and is checked against its
SHA-256 before it replaces the active model (`engine/ModelCatalog.kt`).
Any other Parakeet-TDT GGUF can be imported through **Import model
(.gguf)** instead. Then select **This device** under *Transcribe on*. The model stays in app-private storage and is loaded on first
use; no server or network is needed. Server transcription remains the
default. When a transcription fails — unreachable server or missing model —
the failure is surfaced and the local recording remains available for retry.
Building the app additionally requires the Android NDK and CMake SDK packages.

The arm64 engine is built for ARMv8.2 with the dot-product and FP16
extensions (every mainstream phone core since 2018). On older CPUs the
on-device option reports that it cannot run instead of loading the library;
server transcription still works there. The engine uses the performance
cores only, and the loaded model is released when Android reports memory
pressure (it reloads on the next use); a load that clearly cannot fit in free
memory is refused with an explanation.

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
needs an uninstall first (which deletes the app's recordings and model).

Local release builds read the same four values from `STARLING_ANDROID_*`
environment variables (`STARLING_ANDROID_KEYSTORE` is the keystore path);
without them `assembleRelease` produces an unsigned APK.

## Build and run

From this directory:

```bash
./gradlew test
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Open **Starling Mobile**, grant microphone access, enter the server URL and
its model slug, then tap **Save connection**. `starling-serve` binds to
`127.0.0.1` by default; start it with `--host 0.0.0.0` (or the machine's LAN
or Tailscale address) so the phone can reach it. The model slug is shown by
the server's `/v1/models` response. The default transport
is HTTPS. For a local development server using the documented default HTTP
port, explicitly check **Allow HTTP for a trusted private LAN** and use an
address such as:

```text
http://127.0.0.1:8181       # emulator: http://10.0.2.2:8181
http://192.168.1.20:8181   # physical device on the same LAN
http://100.101.42.1:8181   # Tailscale/CGNAT VPN overlay
```

The HTTP setting accepts only loopback, RFC1918 private, link-local, CGNAT
(`100.64.0.0/10`, as assigned by Tailscale-style VPN overlays), or `.local`
hosts. Remote HTTP URLs and credentials embedded in URLs are rejected.
The Android network policy is configured for this explicit private-LAN opt-in;
HTTPS remains the default and is recommended whenever available. Existing
Starling deployments have no authentication layer, so expose a server through
an authenticated reverse proxy when a network boundary matters. This client
does not persist credentials or bearer tokens.

Recordings are retained in the app's private files directory. Each item keeps
its WAV, state, retry count, exact raw transcript, its provenance (streamed
live or batch-uploaded), and any transport error. Automatic bounded retries
cover transient HTTP/server failures; **Retry** is also available from the
recording list. Deleting an item requires an explicit confirmation and removes
its WAV and transcript from this app.

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
contract in `packages/contracts/mode-routing/` by `tests/test_staging.py`:

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

### Live streaming and its fallback

The WAV written during capture stays the source of truth: the stream is an
observer of the same chunks, never a gate on them. The save still completes
before any network use — on Stop the WAV is finalized and committed first,
and only then is the stream committed for its final transcript. If the socket
fails at any point (connect, mid-stream, at commit, or the server's 60 s live
buffer cap fires), the stream is dropped, the interruption is shown, and the
already-saved WAV is transcribed through the ordinary batch upload path with
its normal retries. A streaming failure therefore loses nothing: the
recording, retry, and failure story is indistinguishable from the
non-streaming flow. In the voice keyboard a failed stream removes its
composing text and falls back to the explicit **Insert transcript** flow.

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
recognizer. When the configuration supports streaming, growing partials are
delivered through the platform's `partialResults` callback while you speak;
the final result is still delivered once, after the stream commits or the
batch transcription of the saved WAV completes. Every dictation stays visible
in Saved recordings, including failed transcriptions, so it can be retried or
deleted there.

Apps that ask the system for speech input with
`RecognizerIntent.ACTION_RECOGNIZE_SPEECH` (a microphone button that opens a
"speak now" popup) can pick Starling too: a small dialog starts listening at
once, shows live partials, and returns the verbatim transcript when you tap
**Done**. **Cancel** keeps the audio in Saved recordings without transcribing
it.

## API compatibility

Batch transcription sends the multipart fields `file`, `model`, and
`response_format=json` to `/v1/audio/transcriptions`. The model field is
required. Successful JSON responses must contain a `text` string. Live
dictation uses Starling's `WS /stream` and falls back to batch transcription
if the stream fails. No normalization endpoint is called, so the returned
transcript remains the source of truth.

Set the endpoint to a server root, or enter the exact route. A server root
gets `/v1/audio/transcriptions` appended; a root ending in `/v1` gets
`/audio/transcriptions` appended.

## Scope and limitations

This directory contains the Android app. The sibling standalone iOS recorder is
at `../ios`, and the desktop client is at `../desktop`. The
Android app does not provide an iOS keyboard extension; its system-wide voice
keyboard is Android-specific. Push-to-talk hardware integration remains a
future extension. Server live streaming follows the
`WS /stream` contract in `../../docs/native-serving.md` and requires a native
Starling server; on-device live transcription uses the same window geometry
(12 s windows, 3 s overlap, word stitching) with partials about every second. The app does not ship model weights or
third-party/copyrighted application code.
