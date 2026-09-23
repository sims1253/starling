# Starling Voice for iOS

Starling Voice is a standalone SwiftUI recorder for a Starling server. It
records mono 16 kHz, 16-bit PCM WAV, saves the WAV in the app's Application
Support directory, and streams PCM16 frames to `WS /stream` during capture.
The final transcript is saved after the WAV is committed; if streaming fails,
the app retries the saved WAV through the batch endpoint. Raw server
text is saved without cleanup. Recordings remain available for playback,
retry, and export until the user explicitly deletes them.

Capture begins in an app-private `Pending` directory. Promotion removes that
staged file only after the history copy and manifest are durable; pending files
are recovered on the next launch after an interruption, and one unreadable
session directory is skipped and reported instead of hiding the whole history.
Sessions left mid-request by an app exit become retryable failures on the next
launch while keeping their audio, transcripts, and attempt counts. Successful
retries keep earlier raw transcripts in the session manifest. Network requests
use an ephemeral URL session and refuse redirects, so audio cannot silently
follow a server redirect to a destination the user did not configure.

Recording owns the process-wide audio session exclusively: history playback
neither reconfigures nor deactivates the session while capture is active, and
a capture ended by the system — a call, Siri, or a disconnected microphone —
is finalized with the audio gathered so far saved to history instead of
continuing to show a running timer.

Batch retries send `file`, `model`, and `response_format=json` to
`POST /v1/audio/transcriptions` and read the returned raw `text`.

HTTPS is the default. Plain HTTP must be enabled in Settings and is accepted
only for loopback, `.local`, private IPv4, link-local, or private/link-local
IPv6 hosts. `Info.plist` declares local networking through
`NSAllowsLocalNetworking`; it does not enable `NSAllowsArbitraryLoads`.

This is an app, not an iOS keyboard extension. iOS does not grant a third-party
keyboard general microphone access suitable for this recorder. Finished text
can be copied or shared with another app; a future integration can add app
intents or a share extension without misrepresenting system-wide dictation.

## Generate and build on macOS

Install Xcode 16 and [XcodeGen](https://github.com/yonaskolb/XcodeGen), then:

```bash
cd apps/ios
xcodegen generate
xcodebuild -project StarlingVoice.xcodeproj \
  -scheme StarlingVoice \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  CODE_SIGNING_ALLOWED=NO build
swift test
```

Open `StarlingVoice.xcodeproj` to select a signing team and run on a physical
device. Microphone capture is best verified on hardware. The initial HTTPS URL
is a placeholder; configure the server under the gear button before recording.

The portable core tests cover endpoint security, multipart protocol fields,
response fidelity, all six regression patterns described in the repository
request, and durable file-backed retry history. This checkout was created in a
Linux environment without Swift, XcodeGen, or the iOS SDK, so the commands above
are the required macOS validation and have not been claimed as run here.
