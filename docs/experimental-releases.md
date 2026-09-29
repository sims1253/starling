# Try the current master build

Open [Experimental releases](https://github.com/sims1253/starling/releases?q=experimental&expanded=true)
and choose the highest build number with the **Pre-release** label. Download
from the release's **Assets** list. You do not need a GitHub account or a compiler.
The release notes identify the source commit and link to the build log.

Every push to `master` starts a build, including merges and direct commits.
A successful build publishes all nine packages together. A failed build leaves
the previous downloads available. Builds can finish out of order, so compare
build numbers when choosing the newest one. Stable releases keep GitHub's
**Latest** designation. Experimental releases do not update the npm package.

## Choose your downloads

| Machine | Download |
| --- | --- |
| Linux desktop, CPU or Vulkan GPU (AMD, Intel, NVIDIA) | `starling-gpui-linux-x64.tar.gz` |
| Windows desktop, CPU or Vulkan GPU | `starling-gpui-windows-x64.zip` |
| RTX 5090 on Linux or Windows (CUDA) | Desktop archive + `starling-serve-<platform>-cuda` archive (see [Advanced](#advanced-run-your-own-server)) |
| Headless or scripted use | `starling-serve-<platform>-<backend>` archives (CPU, Vulkan, CUDA) |
| Pixel 10 Pro | `starling-mobile-<version>-i8mm.apk` |

The desktop archives are self-contained: the CPU and Vulkan `starling-serve`
engines ship inside them in an `engines/` directory, and the app launches one
as a supervised sidecar. It picks Vulkan when a Vulkan driver is present and
otherwise CPU, and shows which engine it uses. Vulkan supports AMD, Intel,
and NVIDIA GPUs with a suitable driver. The experimental CUDA packages target
[SM 120, including the RTX 5090](https://developer.nvidia.com/cuda/gpus),
to reduce compilation work; CUDA is not bundled — download the CUDA server
archive and connect to it with Manual server mode (Advanced below). Tagged
stable releases retain their wider CUDA targets.

## Linux desktop

The app is built on Ubuntu 24.04. Use Ubuntu 24.04 or a compatible newer
system with a Wayland or X11 session. This is a portable archive, not an
AppImage: system libraries are required. On Ubuntu 24.04:

```bash
sudo apt install libasound2t64 libxcb1 libxkbcommon0 libxkbcommon-x11-0 \
  libwayland-client0 libvulkan1 libfontconfig1
```

Recording also needs a working PipeWire/PulseAudio microphone source. The
bundled engines' own prerequisites are listed in `engines/RUNTIME.md` inside
the archive; on Ubuntu 24.04 the packages above cover them.

1. Extract the desktop archive into a directory.
2. Open `./starling-gpui`.
3. Pick a model in Settings. It downloads and verifies in-app into the app
data dir — `~/.local/share/starling-gpui/models/` on Linux, i.e.
`dirs::data_dir()/starling-gpui/models`. Manually placed catalog `.gguf`
files there are verified and used too.
4. Record a short take, check the transcript, and use **Copy** to paste it
into another app.

## Windows desktop

1. Install the [Microsoft Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   for the desktop app.
2. Extract the desktop zip into a directory.
3. Open `starling-gpui.exe`, pick a model in Settings (downloaded and
verified in-app into `%APPDATA%\starling-gpui\models\`), and dictate.
Manually placed catalog `.gguf` files there are verified and used too.

The app is unsigned, so Windows may show a publisher warning.

## Advanced: run your own server

The bundled engines cover CPU and Vulkan. To compare backends, run headless,
or use CUDA, start a standalone `starling-serve` instead and point the app at
it with a manual endpoint override: download one of the
`starling-serve-<platform>-<backend>` archives, install the prerequisites
listed in its `RUNTIME.md`, download a compatible GGUF model from the
[model guide](https://github.com/sims1253/starling/blob/master/docs/models.md),
and start the server in a terminal. For the Vulkan build on Linux:

```bash
./starling-serve-linux-vulkan --model parakeet --gguf /path/to/model.gguf --port 8181
```

Then open the app, choose **Manual server mode** in Settings, and set its
server address to `http://127.0.0.1:8181`. For CUDA or CPU testing, start the
corresponding executable with the same arguments; keep only one server
listening on port 8181. The server supports Ubuntu 22.04 or newer; the
desktop app needs the newer baseline above. GPU drivers and CUDA runtime
libraries are never bundled.

## Android on Pixel 10 Pro

Download and open the `-i8mm.apk` on the phone. Allow installs from your browser
or file manager when Android asks. This release is arm64-only and optimized
for phones with i8mm instructions, including the Pixel 10 Pro.

The app appears as **Starling Experimental**. It has a separate application ID
(`dev.starling.mobile.experimental`), so it installs alongside Starling Mobile.
It starts with its own settings and storage. Later experimental APKs update
this installation and retain its recordings and model. All use the repository's
persistent Android signing key.

For offline transcription, select **This device** and tap **Download recommended
model**. To dictate into another app, enable **Starling Experimental Voice Input**
in Android's keyboard settings and select it. See the
[Android guide](https://github.com/sims1253/starling/blob/master/apps/mobile/README.md)
for server connections and speech-recognizer support.

## Updates and bug reports

Updates are manual: download the assets from a newer experimental release.
Desktop archives use the app's existing settings and recordings; extract the
new executable after closing the old one. Experimental desktop builds share
data with other desktop builds, so copy your data before testing changes you
may want to roll back. See the desktop README for data locations.

Old releases remain available. Desktop rollback means extracting the older
app archive. Android blocks a lower version code; uninstalling allows an
older APK but deletes that installation's recordings, settings, and model.

Each release includes `SHA256SUMS.txt` and `build-info.json`. The latter records
the commit, version, and build URL. Include the commit, OS, server backend,
model, and steps to reproduce when reporting a problem. The desktop archive
also includes `COMMIT.txt`; the server prints its version with `--version`.

These builds make current behavior easy to test. Desktop delivery currently
uses **Copy**; direct insertion into other apps is still pending. The global
recording shortcut is Ctrl+Shift+Space, with support on Wayland dependent on
the compositor and portal. CI checks builds, unit tests, checksums, and server
startup; it does not prove GPU inference or microphone behavior on your hardware.

## How publication works

`.github/workflows/release-experimental.yml` calls the same server and Android
build workflows used by tagged releases, plus the shared desktop packager.
The desktop job runs after the servers and bundles their CPU and Vulkan
engines into the desktop archives, so one download contains the app and its
engines. All checkouts use the triggering commit. The Android version code uses seconds
since 2020 UTC, independent of workflow run numbers and stable versions.

Each run owns an `experimental-<run-number>-<commit>` tag. There is no moving
tag. The publisher checks for all nine packages, computes checksums, uploads
to a draft, and then publishes it as a prerelease with `--latest=false`.
A failed upload leaves a draft that a rerun can complete. A rerun never replaces
the assets of an already published release. Retrying failed jobs should happen
within three days, while their successful siblings' CI artifacts still exist;
after that, rerun all jobs. Use **Run workflow** on `master` for a fresh build.

The four `STARLING_ANDROID_*` signing secrets documented in the Android guide
must be configured. Experimental builds refuse temporary signing keys because
they would require uninstalling the app on every update.

To limit work, this channel omits macOS and ROCm, compiles CUDA for SM 120 only,
and builds one arm64 Android variant. Desktop dependencies, Gradle downloads,
and the Windows Vulkan SDK are cached. CI artifacts expire after three days;
release assets remain available. The ordinary desktop CI reuses the Windows
packager on PRs and other branches and leaves master packaging to this workflow.

GitHub's standard hosted runners are free for public repositories; larger
runners have separate billing. See
[GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
