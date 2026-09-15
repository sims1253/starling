# Starling Mobile (Android)

Starling Mobile is a small native Android client for a Starling server that
you run yourself. It records 16 kHz mono PCM16 microphone audio, writes a WAV
file in app-private storage, and only then sends a multipart transcription
request. The exact `text` returned by the server is stored as the raw
transcript. Nothing in the client fixes numbering, removes filler words, or
silently rewrites words such as `not`, `like`, or `auth`.

The same capture flow is available from the standalone app and the optional
Starling Voice Input keyboard. The keyboard shows a transcript first; it only
calls `InputConnection.commitText` after the user taps **Insert transcript**.
It does not read surrounding editor text, package names, or selection context.
An editor target generation and connection identity check prevents a late
network response from being inserted into a field that has changed.

## Prerequisites

- Android Studio Ladybug or newer, or JDK 17 with the Android SDK.
- Android SDK platform 35 and build-tools 35.0.0 installed.
- A running Starling native server (`starling-serve`). Native serving is the
  supported portable path and requires a 16 kHz WAV. The Python/CUDA server is
  retained for compatibility but is deprecated for new app deployments.
- A device or emulator with a microphone. Android 8.0 / API 26 is the minimum.

No Python or GPU runtime is bundled. An optional **on-device engine**
(experimental) embeds the repository's native Parakeet engine
(`libstarling_ggml`) for offline transcription: import a Parakeet-TDT GGUF
(for example `models/parakeet-tdt-0.6b-v3-q4_0.gguf`) through
**Import model (.gguf)**, then select **This device** under *Transcribe on*.
The model stays in app-private storage and is loaded on first use; no server
or network is needed. Server transcription remains the default. When a
transcription fails — unreachable server or missing model — the failure is
surfaced and the local recording remains available for retry. Building the
app additionally requires the Android NDK and CMake SDK packages.

## Build and run

From this directory:

```bash
./gradlew test
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Open **Starling Mobile**, grant microphone access, enter the server URL, choose
the API protocol, and tap **Save connection**. The default protocol is the
legacy Starling route. OpenAI-compatible mode requires the model slug served by
the process (the native `/v1/models` response shows it). The default transport
is HTTPS. For a local development server using the documented default HTTP
port, explicitly check **Allow HTTP for a trusted private LAN** and use an
address such as:

```text
http://127.0.0.1:8181       # emulator: http://10.0.2.2:8181
http://192.168.1.20:8181   # physical device on the same LAN
```

The HTTP setting accepts only loopback, RFC1918 private, link-local, or
`.local` hosts. Remote HTTP URLs and credentials embedded in URLs are rejected.
The Android network policy is configured for this explicit private-LAN opt-in;
HTTPS remains the default and is recommended whenever available. Existing
Starling deployments have no authentication layer, so expose a server through
an authenticated reverse proxy when a network boundary matters. This client
does not persist credentials or bearer tokens.

Recordings are retained in the app's private files directory. Each item keeps
its WAV, state, retry count, exact raw transcript, and any transport error.
Automatic bounded retries cover transient HTTP/server failures; **Retry** is
also available from the recording list. Deleting an item requires an explicit
confirmation and removes its WAV and transcript from this app.

To enable the keyboard, tap **Enable Starling keyboard**, enable Starling Voice
Input in Android settings, then select it from the keyboard switcher. Switching
apps or editor fields while a request is in flight leaves the transcript in
Starling and disables insertion for the old target.

### Voice input inside other keyboards

Starling also registers as a system speech recognizer
(`StarlingRecognitionService`). Keyboards that use Android's `SpeechRecognizer`
API — for example the open-source HeliBoard, OpenBoard, or AnySoftKeyboard —
then show their own microphone button and transcribe through Starling, without
switching keyboards. Tap **Set Starling as voice input** and choose Starling
Voice Recognition as the voice input service. Closed keyboards such as Gboard
or SwiftKey keep their bundled engines and cannot delegate to a custom
recognizer. Dictation returns one final result (the server response is
batch), and every dictation stays visible in Saved recordings, including
failed uploads, so it can be retried or deleted there.

## API compatibility

Starling legacy mode sends a multipart field named `file` to `/inference`,
matching the legacy contracts in `../../docs/python-serving.md` and
`../../docs/native-serving.md`. OpenAI-compatible mode sends the standard
multipart fields `file`, `model`, and `response_format=json` to
`/v1/audio/transcriptions`; the model field is required and is never guessed at
request time. Successful JSON responses must contain a `text` string. No
normalization endpoint is called, so the returned transcript remains the
source of truth.

Set the endpoint to a server root, or enter the exact route. For OpenAI mode a
server root gets `/v1/audio/transcriptions` appended; a root ending in `/v1`
gets `/audio/transcriptions` appended. Legacy mode appends `/inference` to a
server root. This supports the native server and the shared app protocol
adapter without sending OpenAI fields to the legacy route.

## Scope and limitations

This directory contains the Android app. The sibling standalone iOS recorder is
at `../ios`, and the desktop client is at `../desktop`. The
Android app does not provide an iOS keyboard extension; its system-wide voice
keyboard is Android-specific. Background recording, streaming `/stream`,
push-to-talk hardware integration, and embedded native inference remain future
extensions. The app does not ship model weights or third-party/copyrighted
application code.
