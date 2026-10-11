# Starling

<img align="right" src="assets/branding/starling-logo.webp" alt="Starling logo: a watercolor starling and sound wave inside a golden hexagon" width="160" height="160">

Speech recognition that you can serve anywhere, with apps for desktop and phone.
Starling owns its inference engine: native ggml backends run GGUF models on CPU,
with optional Metal, Vulkan, HIP, or CUDA acceleration. No Python or NVIDIA GPU
is required for the main server or the apps.

This is a monorepo for the engine, serving API, quantization tools, shared
transcription contracts, and applications. The apps are development foundations;
see the [platform status](docs/monorepo.md) for what works and what still needs
native device testing.

<br clear="right">

## Try the current master build

[Experimental releases](https://github.com/sims1253/starling/releases?q=experimental&expanded=true)
provide Linux and Windows desktop apps, CPU/Vulkan/CUDA servers, and an Android
APK after each successful master build. See the
[installation guide](docs/experimental-releases.md) for the right downloads
and setup steps. No compiler is needed.

## Run the server

Quickest path — a prebuilt binary, no compiler and no GPU required. Download
the `starling-serve-<platform>-<backend>` archive for your machine from
[GitHub releases](https://github.com/sims1253/starling/releases) (every
experimental master build attaches them too). On Linux with the CPU build:

```bash
tar xzf starling-serve-linux-cpu.tar.gz
sha256sum -c starling-serve-linux-cpu.sha256
./starling-serve-linux-cpu --model parakeet --gguf /path/to/model.gguf --port 8181
```

Each archive's `RUNTIME.md` lists the drivers and runtime libraries that build
needs; see [packaging](docs/packaging.md) for the full artifact list.

To build from source instead, you need CMake, a C++17 compiler, Git, and Bash.
Initialize the submodule, build, and supply a compatible GGUF model from the
[model guide](docs/models.md):

```bash
git submodule update --init --recursive
cmake -B build -DSTARLING_SERVE=ON
cmake --build build -j --target starling-serve
./build/starling-serve --model parakeet --gguf /path/to/model.gguf --port 8181
```

The root CMake entry point remains supported. You can also configure the native
component directly with `cmake -S backends/native -B build/native -DSTARLING_SERVE=ON`,
or use `cmake --preset native-cpu` with CMake 3.21+ and Ninja.

Use the OpenAI-compatible batch transcription API:

```bash
curl http://127.0.0.1:8181/v1/audio/transcriptions \
  -F model=parakeet -F file=@recording.wav
```

The initial compatible subset accepts 16 kHz WAV and returns `{"text":"…"}`.
`GET /v1/models` lists the configured model. Unsupported features return errors;
read the [API contract](docs/api.md) before pointing a third-party client at it.
Live dictation uses Starling's `WS /stream` route.

## Run the apps

The desktop app is the native Rust build in `apps/desktop-gpui`:

```bash
cd apps/desktop-gpui
cargo run -p starling-gpui --release
```

Linux needs a Wayland or X11 session plus the usual audio and font
libraries; see [the app's README](apps/desktop-gpui/README.md) for
prerequisites, data locations, and the packaging script CI uses. CI builds
unsigned macOS and Windows artifacts on every push.

- [Desktop: Windows, Linux, macOS](apps/desktop-gpui/README.md)
- [Android recorder and voice keyboard](apps/mobile/README.md)
- [iOS recorder](apps/ios/README.md)

Recognition returns raw text. Apps retain recordings for review and retry;
copying or inserting a transcript is an explicit action. Vocabulary hints and
suggested edits must never silently replace the original. These rules protect
against destructive processing; they cannot guarantee ASR accuracy.

## Workspace

| Component | Location |
| --- | --- |
| Native server and engine build | [`backends/native/`](backends/native/) · engine source in `cpp/` |
| Quant recipes, catalog, artifact tooling | [`quants/`](quants/) |
| Desktop app (Rust, gpui) | [`apps/desktop-gpui/`](apps/desktop-gpui/) |
| Android app and voice keyboard | [`apps/mobile/`](apps/mobile/) |
| iOS app | [`apps/ios/`](apps/ios/) |
| Language-independent API contract and shared test fixtures | [`packages/contracts/`](packages/contracts/) |
| Deprecated Python/CUDA serving | [`backends/python/`](backends/python/) · reference source in `src/starling/` |

The NVIDIA-only Python serving path is deprecated. Its kernels, benchmarks,
and existing commands remain for research and reproducibility. New app and
serving work targets the native engine and the portable HTTP contract.

## Documentation

[Architecture and platform status](docs/monorepo.md) · [API](docs/api.md) ·
[Models](docs/models.md) · [Native serving](docs/native-serving.md) ·
[Packaging and release archives](docs/packaging.md) ·
[Quantization tools](quants/README.md) · [Quantization research](docs/quantization.md) ·
[Benchmarks](docs/benchmarks.md) · [Engine development](docs/ggml-engine.md) ·
[Fast Vulkan engines](docs/fast-engine.md) ·
[Coding-agent dictation (MCP)](docs/mcp-dictation.md) ·
[Model Optimizer and Unsloth research](docs/research/model-optimizer-unsloth-cross-backend.md)
