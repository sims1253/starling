# starling-serve packaging

How the native server reaches users: prebuilt release archives built by
[the release workflow](../.github/workflows/release-starling-serve.yml), and
the `starling-serve` npm package ([packages/serve](../packages/serve/)) that
installs the right one with a single command.

## The npm launcher

```bash
npx starling-serve --model parakeet --gguf model.gguf --port 8181
```

`npx`/`pnpm dlx` fetch the launcher from npm; the launcher then fetches the
native executable from GitHub Releases **on first run**, verifies it, caches
it, and `exec`s it with all arguments forwarded. `npm i -g starling-serve`
works the same way.

### Why download at first run, not install time

The package has no `postinstall` script, by design:

- **Offline and air-gapped installs stay possible.** An install-time download
  makes `npm install` fail (or silently skip the binary) without network.
- **No surprise network I/O.** Package managers and security policies
  increasingly block lifecycle scripts; depending on one makes installs
  fragile in exactly the locked-down environments that want a local server.
- **CI caching stays clean.** The npm cache caches the launcher (~10 KB); the
  100 MB binary is cached by the launcher in a user cache directory that CI can
  persist explicitly when wanted.
- **Failures surface where they can be acted on.** A first-run download error
  is a clear, actionable message at the moment the user asked for the server —
  not a truncated warning two screens above the prompt.

### Backend selection

| Platform    | Default                                                                      | Reasoning (against [runtime requirements](release-runtime.md))                                                                                                                 |
| ----------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| macOS arm64 | `metal`                                                                      | Metal ships with macOS 14+; no extra runtime to install.                                                                                                                       |
| Linux x64   | `vulkan` if the Vulkan loader (`libvulkan.so.1`) is discoverable, else `cpu` | The vulkan archive is the cross-vendor GPU path but needs a loader plus a vendor driver, which servers and containers often lack; `cpu` is the safe fallback that always runs. |
| Windows x64 | `cpu`                                                                        | Stock Windows has neither the Vulkan loader nor CUDA runtime; `cpu` is the only artifact guaranteed to start. Opt in to `vulkan`/`cuda`.                                       |

Override the default with the `--starling-backend <backend>` flag or the
`STARLING_SERVE_BACKEND` environment variable (the flag wins). The launcher
only accepts backends that have an artifact for the platform, and refuses
unknown `--starling-*` flags instead of silently forwarding them.

| Platform       | Available backends              |
| -------------- | ------------------------------- |
| Linux x86_64   | `cpu`, `vulkan`, `cuda`, `rocm` |
| Windows x86_64 | `cpu`, `vulkan`, `cuda`         |
| macOS arm64    | `cpu`, `metal`                  |

There are no Intel-mac artifacts: GitHub retired its Intel macOS runners
(`macos-13` in December 2025; the `macos-15-intel` stopgap is deprecated), so
an Intel-mac release job could not run reliably. Intel Mac users build from
source ([native serving guide](native-serving.md#build)).

### Environment overrides

| Variable                 | Default              | Effect                                                                  |
| ------------------------ | -------------------- | ----------------------------------------------------------------------- |
| `STARLING_SERVE_BACKEND` | per-platform default | Force the backend (`cpu`, `vulkan`, `cuda`, `rocm`, `metal`).           |
| `STARLING_SERVE_RELEASE` | `v<package version>` | Download from this release tag instead — for custom builds and testing. |
| `STARLING_SERVE_REPO`    | `sims1253/starling`  | Download from another GitHub `owner/repo` — for forks.                  |
| `STARLING_SERVE_CACHE`   | OS cache dir         | Store binaries here instead.                                            |

Repository and tag values become cache path components, so each must match
`[A-Za-z0-9._+-]` (no slashes, backslashes, or other separators); anything
else fails fast with a clear error instead of touching the filesystem.

### Cache

Verified binaries live under the OS cache directory, one folder per
repository and release tag so versions — and alternate `STARLING_SERVE_REPO`
sources — coexist:

- Linux: `$XDG_CACHE_HOME/starling-serve` or `~/.cache/starling-serve`
- macOS: `~/Library/Caches/starling-serve`
- Windows: `%LOCALAPPDATA%\starling-serve\cache`

Layout: `<cache>/releases/<owner>/<repo>/<tag>/starling-serve-<os>-<backend>[.exe]`
plus a `.verified` marker recording the artifact identity — `repo`, `tag`,
`binary`, and the `sha256` the launcher checked. Every launch re-checks that
identity and re-hashes the binary against that marker, so on-disk corruption
or tampering is caught before exec and triggers a re-download, and a binary
cached from one repository is never served for another. Cache entries are
never evicted automatically; delete the directory to reclaim space.

Entries written by launcher versions before provenance was recorded (a
marker holding only a checksum) are never read again; they are safe to
delete, and the first run after upgrading re-downloads each still-used
binary once under its new identity.

During a download, a hidden `.starling-serve-*` staging directory appears
next to the cache entry (staging inside the cache keeps the final move a
same-filesystem rename — a system temp dir can sit on another device, where
that rename would fail). It is removed when the install finishes or fails; a
hard-killed run may leave one behind, but it is never mistaken for a verified
cache entry and the next successful run needs nothing from it.

Offline machines: run once anywhere with network and copy the whole
`releases/<owner>/<repo>/<tag>/` entry (binary and marker), or point
`STARLING_SERVE_CACHE` at a pre-populated directory.

### Security

- Binaries come only from `https://github.com/<repo>/releases/download/…` over
  TLS.
- Two checksum layers must both pass before anything executes: the archive
  hash against the release's consolidated `SHA256SUMS.txt`, and the extracted
  executable's hash against the `.sha256` file shipped inside the archive.
- The launcher extracts only the two named members it needs from the archive
  (never a wildcard) into a staging directory inside the cache, then moves the
  verified binary into place with a same-filesystem rename.
- No code from the release runs before both checks pass.

## Release-side checklist (owner steps)

1. Cut the `v*` tag; the release workflow builds, checks, and uploads nine
   archives plus `SHA256SUMS.txt`.
2. Publish the npm package from the same tag:
   `pnpm --filter starling-serve publish` (runs typecheck, tests, and the
   `tsc` build via `prepublishOnly`). Package version `X.Y.Z` must equal the
   release tag `vX.Y.Z` — that mapping is how the launcher resolves binaries.
3. The first release after this change must ship all nine artifacts before
   `npx starling-serve` works on every platform; until then the launcher fails
   with a `ReleaseAssetError` naming the missing asset.

## Relationship to native-packages

[crmne/native-packages](https://github.com/crmne/native-packages) (referenced
in [#128](https://github.com/sims1253/starling/issues/128)) generates OS-level
packages (DEB/RPM/APK/DMG/MSIX) from built apps via a Ruby toolchain. We took
its invariants — checksums recorded and verified before anything runs,
release-pinned download URLs, one-command installs per platform — and skipped
its machinery: per-OS packaging toolchains, signing infrastructure, and
external recipe maintenance (AUR/Homebrew) are more than this repo needs, and
npm already reaches Linux, Windows, and macOS from the existing pnpm
workspace. The manual archive path (see the
[native serving guide](native-serving.md#release-artifacts)) remains for
environments without Node.
