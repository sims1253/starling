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
once, on Stop, into the field the recording started in. The keyboard does not
read surrounding editor text, package names, or selection context. An editor
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
(`libstarling_ggml`) for offline transcription: download a Parakeet-TDT GGUF
from [`scholzmx/parakeet-tdt-0.6b-v3-gguf`](https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf)
(`parakeet-tdt-0.6b-v3-q4_k_m.gguf`, 704 MB, is a good default), import it
through **Import model (.gguf)**, then select **This device** under
*Transcribe on*. The model stays in app-private storage and is loaded on first
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
Input in Android settings, then select it from the keyboard switcher. Switching
apps or editor fields while a request is in flight leaves the transcript in
Starling and disables insertion for the old target.

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
keyboard is Android-specific. Background recording and push-to-talk hardware
integration remain future extensions. Server live streaming follows the
`WS /stream` contract in `../../docs/native-serving.md` and requires a native
Starling server; on-device live transcription uses the same window geometry
(12 s windows, 3 s overlap, word stitching) with partials about every second. The app does not ship model weights or
third-party/copyrighted application code.
