# starling-serve packaging

The native server reaches users as prebuilt release archives built by
[the release workflow](../.github/workflows/release-starling-serve.yml) and
attached to [GitHub releases](https://github.com/sims1253/starling/releases).
Experimental master builds attach the Linux and Windows CPU, Vulkan, and CUDA
subset (see [experimental releases](experimental-releases.md)).

## Release archives

| Archive | Platform | Backend | Notes |
| --- | --- | --- | --- |
| `starling-serve-linux-cpu.tar.gz` | Linux x86_64 | CPU | No GPU required |
| `starling-serve-linux-vulkan.tar.gz` | Linux x86_64 | Vulkan | Intel / AMD / NVIDIA |
| `starling-serve-linux-cuda.tar.gz` | Linux x86_64 | CUDA | NVIDIA |
| `starling-serve-linux-rocm.tar.gz` | Linux x86_64 | ROCm / HIP | AMD; tagged releases only |
| `starling-serve-windows-cpu.zip` | Windows x86_64 | CPU | No GPU required |
| `starling-serve-windows-vulkan.zip` | Windows x86_64 | Vulkan | AMD / Intel / NVIDIA |
| `starling-serve-windows-cuda.zip` | Windows x86_64 | CUDA | NVIDIA |
| `starling-serve-macos-metal.tar.gz` | macOS arm64 | Metal | Apple Silicon; tagged releases only |
| `starling-serve-macos-cpu.tar.gz` | macOS arm64 | CPU | Apple Silicon; tagged releases only |

Each archive holds the executable (`starling-serve-<platform>-<backend>`, with
`.exe` on Windows), its `.sha256`, and `RUNTIME.md` with the drivers and
runtime libraries that build needs (see the [runtime guide](release-runtime.md)).
Pick `cpu` when no GPU is available, `cuda` for NVIDIA, `rocm` for AMD on
Linux, `vulkan` for AMD/Intel (or as a cross-vendor fallback), and `metal` on
Apple Silicon.

There are no Intel-mac artifacts: GitHub retired its Intel macOS runners
(`macos-13` in December 2025; the `macos-15-intel` stopgap is deprecated), so
an Intel-mac release job could not run reliably. Intel Mac users build from
source ([native serving guide](native-serving.md#build)).

## Download, verify, run

Download the archive and the release's `SHA256SUMS.txt` from the release's
**Assets** list, or with the GitHub CLI:

```bash
gh release download <tag> -R sims1253/starling \
  -p starling-serve-linux-cpu.tar.gz -p SHA256SUMS.txt
sha256sum -c --ignore-missing SHA256SUMS.txt   # archive against the release
tar xzf starling-serve-linux-cpu.tar.gz
sha256sum -c starling-serve-linux-cpu.sha256   # executable against the archive
./starling-serve-linux-cpu --version
./starling-serve-linux-cpu --model parakeet --gguf model.gguf --port 8181
```

On Windows, extract the zip and run
`starling-serve-windows-cpu.exe --model parakeet --gguf model.gguf --port 8181`;
`Get-FileHash -Algorithm SHA256` gives the hash to compare against
`SHA256SUMS.txt`.

## Release-side checklist (owner steps)

1. Cut the `v*` tag; the release workflow builds, checks, and uploads nine
   archives plus `SHA256SUMS.txt`.
2. Confirm every archive in the table above is attached; the release notes
   list them for users.

## Relationship to native-packages

[crmne/native-packages](https://github.com/crmne/native-packages) (referenced
in [#128](https://github.com/sims1253/starling/issues/128)) generates OS-level
packages (DEB/RPM/APK/DMG/MSIX) from built apps via a Ruby toolchain. We took
its invariants — checksums recorded and verified before anything runs and
release-pinned download URLs — and skipped its machinery: per-OS packaging
toolchains, signing infrastructure, and external recipe maintenance
(AUR/Homebrew) are more than this repo needs.
