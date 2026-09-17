# starling-serve (npm launcher)

One command to run the Starling native speech-recognition server, on Linux,
Windows, and macOS (Apple Silicon), with or without a GPU:

```bash
npx starling-serve --model parakeet --gguf model.gguf --port 8181
```

This package contains no native code. On first run it detects your platform,
downloads the matching prebuilt `starling-serve` executable from
[GitHub Releases](https://github.com/sims1253/starling/releases) (the release
whose tag matches this package's version), verifies two layers of SHA-256
checksums, caches it, and runs it. There is no postinstall script and no
install-time download: `npm install` stays offline-safe and the one-time
download happens when you actually start the server.

## Backends

| Platform       | Default backend                                        | Available backends              |
| -------------- | ------------------------------------------------------ | ------------------------------- |
| Linux x86_64   | `vulkan` when the Vulkan loader is present, else `cpu` | `cpu`, `vulkan`, `cuda`, `rocm` |
| Windows x86_64 | `cpu`                                                  | `cpu`, `vulkan`, `cuda`         |
| macOS arm64    | `metal`                                                | `cpu`, `metal`                  |

Force a backend with `--starling-backend <backend>` (e.g.
`npx starling-serve --starling-backend cuda --version`) or with the
`STARLING_SERVE_BACKEND` environment variable. All other arguments are
forwarded verbatim to the server.

## Environment variables

| Variable                 | Default              | Effect                                         |
| ------------------------ | -------------------- | ---------------------------------------------- |
| `STARLING_SERVE_BACKEND` | per-platform default | Force the backend.                             |
| `STARLING_SERVE_RELEASE` | `v<package version>` | Use this release tag (custom builds, testing). |
| `STARLING_SERVE_REPO`    | `sims1253/starling`  | Download from another GitHub `owner/repo`.     |
| `STARLING_SERVE_CACHE`   | OS cache directory   | Cache location override.                       |

## Cache

Verified binaries are cached under `~/.cache/starling-serve` (Linux),
`~/Library/Caches/starling-serve` (macOS), or
`%LOCALAPPDATA%\starling-serve\cache` (Windows), one directory per release
tag. Each launch re-verifies the cached binary's SHA-256; a mismatch triggers
a re-download. Delete the directory to reclaim space.

## Security

Downloads come from `github.com` over TLS. The archive hash must match the
release's `SHA256SUMS.txt` and the extracted executable must match the
`.sha256` checksum shipped inside the archive before anything is executed.
Only the two named archive members needed by the launcher are extracted.

## Requirements

- Node.js 20 or newer.
- `tar` on `PATH` (present by default on Linux, macOS, and Windows 10+).
- GPU backends additionally need the vendor driver and runtime libraries; see
  the release
  [runtime requirements](https://github.com/sims1253/starling/blob/master/docs/release-runtime.md).

MIT licensed. The server itself is part of the
[starling](https://github.com/sims1253/starling) monorepo.
