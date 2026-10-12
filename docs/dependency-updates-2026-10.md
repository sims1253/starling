# October 2026 dependency updates

This batch covers Dependabot PRs #454–#462. Recommendations below are follow-up
work unless explicitly described as part of the compatibility fixes.

## Versions and compatibility

| PR | Dependency | Upgrade | Compatibility work |
| --- | --- | --- | --- |
| #454 | Gradle wrapper | 9.8.0 → 9.8.1 | Wrapper JAR and distribution updated; Android build and emulator checks validate it. |
| #455 | alibaba/open-code-review action | 1.12.11 → 1.12.13 | Action SHA updated; the separately pinned CLI is still 1.12.5 (see below). |
| #456 | actions/upload-artifact | 7.0.1 → 7.0.2 | SHA updated across packaging and test workflows. |
| #457 | actions/download-artifact | 8.0.1 → 8.0.2 | SHA updated across release workflows. |
| #458 | thiserror, tokio, uuid, libc | 2.0.20 → 2.0.21; 1.53.1 → 1.53.2; 1.26.1 → 1.27.0; 0.2.189 → 0.2.190 | Patch GPUI's transitive xattr 0.2.3 requirement for libc's removed Linux/Android ENOATTR alias. |
| #459 | x11rb | 0.13.2 → 0.14.0 | Direct app and insertion dependencies updated; GPUI retains its own 0.13 dependency. |
| #460 | windows-sys | 0.59.0 → 0.61.2 | Import BOOL from windows_sys::core in the host's named-pipe and console-control code. |
| #461 | tokio-tungstenite | 0.24.0 → 0.30.0 | Convert outgoing binary Vecs to Bytes and fake-server text Strings to Utf8Bytes with .into(). |
| #462 | cpal | 0.15.3 → 0.17.1 | SampleRate is now a u32 alias: remove tuple construction and .0 access while preserving 16 kHz mono preference. |

The Rust group changes locked versions: tokio, thiserror and uuid are direct
dependencies; libc is transitive. The xattr patch is an explicit transitive
override. Cargo regenerates the lockfile; no checksums are edited manually.

Licenses are unchanged between old and new releases, checked against crates.io
metadata and upstream tagged license files: CPAL, Gradle and OpenCodeReview use
Apache-2.0; Tokio, tokio-tungstenite and the artifact actions use MIT;
thiserror, uuid, libc, windows-sys and x11rb offer MIT OR Apache-2.0.
The vendored xattr crate retains both license files.

## Features worth using

### CPAL: durable microphone selection and better capture diagnostics

The [0.17.1 changelog](https://github.com/RustAudio/cpal/blob/v0.17.1/CHANGELOG.md)
includes the intervening 0.16/0.17 changes.

- **First priority: stable device IDs.** `DeviceTrait::id` and
  `HostTrait::device_by_id` can replace name-based selection in
  `crates/dictation/src/microphone.rs`. Persist the host and ID beside a display
  name; migrate legacy name preferences once and preserve the existing visible
  fallback behavior. Verify two microphones with identical names, unplug/replug,
  and default-device changes. This batch retains name-based preferences.
- **Distinguish capture errors.** `StreamInvalidated` identifies a stream that
  needs rebuilding; `BufferUnderrun` identifies an overrun/underrun. The current
  recorder treats callback errors as fatal. Add separate diagnostics and test
  audio-gap reporting before introducing recovery; never silently move a
  running take to another microphone or claim lost samples are complete.
- **Explore system-audio recording separately.** CoreAudio now supports loopback
  capture on macOS newer than 14.6. This could support meeting transcription,
  with explicit source selection, permission handling and a microphone fallback.
- Wider integer sample-format support is available. Extend callback conversion
  only alongside format-specific tests; unsupported formats must remain visible.

### Tungstenite: fewer payload copies and explicit memory budgets

The [0.30.0 changelog](https://github.com/snapview/tungstenite-rs/blob/v0.30.0/CHANGELOG.md)
records the intervening releases. Binary payloads use `Bytes`; text uses
`Utf8Bytes`. The compatibility conversion transfers the existing audio Vec into
`Bytes`. If streaming later adds fan-out or retry buffers, retain `Bytes` through
the queue to make clones cheap rather than repeatedly copying audio.

`WebSocketConfig` now has builder methods and a configurable `read_buffer_size`.
The default increased to 128 KiB; upstream suggests 4–8 KiB to recover the older
memory footprint. Benchmark an 8 KiB buffer for the host's small incoming
transcript events via `connect_async_with_config`; measure latency, allocations
and fragmented/large messages before changing it. Keep write/flush behavior and
message limits explicit. Reading from `WebSocketStream` is documented as cancel
safe, relevant to the host's `tokio::select!` loop.

### UUID: more tolerant parsing; optional time-ordered IDs

[UUID 1.27.0](https://github.com/uuid-rs/uuid/releases/tag/v1.27.0) accepts
case-insensitive `urn:uuid:` prefixes and fixes v7 ordering for earlier seconds
with later subsecond nanoseconds. Use `Uuid` parsing if an import/API starts
accepting UUID URNs. Starling currently generates v4 IDs; v7 is a future option
for new time-ordered records, requiring its feature and storage-order tests.
Keep existing IDs and explicit creation timestamps intact.

### Review automation: upgrade the actual CLI before using new behavior

[OpenCodeReview 1.12.12](https://github.com/alibaba/open-code-review/releases/tag/v1.12.12)
allows OpenRouter model overrides beyond its built-in list;
[1.12.13](https://github.com/alibaba/open-code-review/releases/tag/v1.12.13)
improves root-path scanning and timeout diagnostics. These are useful for custom
review models and debugging slow review calls. `.github/workflows/ocr-review.yml`
still sets `ocr_version: '1.12.5'` and disables self-updates. Updating the action
alone does not enable the newer CLI fixes: update that pin in a follow-up and
exercise a maintainer-triggered review before changing model/timeout settings.

## Benefits that need no feature adoption

- [Gradle 9.8.1](https://docs.gradle.org/9.8.1/release-notes.html) fixes daemon and
  worker deserialization vulnerabilities and unsafe repository handling after
  SSL failures ([worker advisory](https://github.com/gradle/gradle/security/advisories/GHSA-mvvg-497x-hmj8),
  [daemon advisory](https://github.com/gradle/gradle/security/advisories/GHSA-xwqc-3h47-hg64),
  [repository advisory](https://github.com/gradle/gradle/security/advisories/GHSA-j5m7-59rp-24f5)).
  The patch also fixes dependency resolution and packaged-JDK detection.
  The release page includes 9.8-series features already available in 9.8.0;
  do not count those as newly introduced by this patch.
- [Artifact upload 7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2)
  and [download 8.0.2](https://github.com/actions/download-artifact/releases/tag/v8.0.2)
  improve HTTP 429 download retries, including `Retry-After` handling. Release
  artifact transfers benefit automatically; no workflow input changes needed.
- [Tokio 1.53.2](https://github.com/tokio-rs/tokio/releases/tag/tokio-1.53.2)
  fixes synchronization, shutdown and Windows child-handle behavior.
  [thiserror 2.0.21](https://github.com/dtolnay/thiserror/releases/tag/2.0.21)
  fixes generic unit-variant display parsing. Both are maintenance updates.
- [x11rb 0.14](https://github.com/psychon/x11rb/blob/v0.14.0/doc/changelog.md)
  corrects CUT_BUFFER atom names and supports libloading 0.9. It removes abstract
  Unix sockets and moves to raw-window-handle 0.6. Check filesystem X11 socket
  access when diagnosing Linux connection failures. No new UI feature required.
- [windows-sys 0.61.2](https://github.com/microsoft/windows-rs/releases/tag/0.61.2)
  requires the BOOL import migration above; preserve native named-pipe tests.
- [libc 0.2.190](https://github.com/rust-lang/libc/releases/tag/0.2.190) removes
  old deprecated APIs. Its unstable 32-bit time64 configuration is relevant only
  if a future 32-bit desktop target needs it; existing 64-bit targets need no opt-in.

## xattr patch and validation

GPUI 0.2.2 → zed-async-tar 0.5.0-zed requires xattr ^0.2, so xattr 1.x cannot
simply replace it. `apps/desktop-gpui/vendor/xattr` contains the published 0.2.3
sources with one errno compatibility change: Linux/Android use ENODATA, while
macOS/BSD keep ENOATTR. See its [patch record](../apps/desktop-gpui/vendor/xattr/STARLING-PATCH.md)
for provenance and the removal condition. Upstream tests cover missing
attributes plus set/get/list/remove through paths and file descriptors.

Run `cargo test --manifest-path vendor/xattr/Cargo.toml` from the Rust workspace,
then `cargo test --workspace --locked` with the engine contract fixture required.
CI covers Linux tests, native Windows host/storage/insertion/app paths, and the
macOS app check. Dispatch `desktop-gpui-rust.yml` on the integrated branch for
macOS universal and Windows packages. Android wrapper validation is covered by
the Android build/unit and emulator suites.
