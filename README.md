# Starling

Speech recognition that you can serve anywhere, with apps for desktop and phone.
Starling owns its inference engine: native ggml backends run GGUF models on CPU,
with optional Metal, Vulkan, HIP, or CUDA acceleration. No Python or NVIDIA GPU
is required for the main server or the apps.

This is a monorepo for the engine, serving API, quantization tools, shared
transcription contracts, and applications. The apps are development foundations;
see the [platform status](docs/monorepo.md) for what works and what still needs
native device testing.

## Run the server

You need CMake, a C++17 compiler, Git, and Bash. Initialize the submodule, build,
and supply a compatible GGUF model from the [model guide](docs/models.md):

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
Existing `/inference`, `/transcribe`, and `/stream` routes remain available.

## Run the apps

For the desktop interface, install Node.js 24.13.1+ and pnpm, then run:

```bash
pnpm install
pnpm run dev
```

This starts a browser preview connected through a development proxy to the
server at `127.0.0.1:8181`. Microphone access requires localhost or HTTPS.
Run `pnpm run desktop` for the Electron desktop app after installing its
[platform prerequisites](apps/desktop/README.md).

- [Desktop: Windows, Linux, macOS](apps/desktop/README.md)
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
| Desktop app | [`apps/desktop/`](apps/desktop/) |
| Android app and voice keyboard | [`apps/mobile/`](apps/mobile/) |
| iOS app | [`apps/ios/`](apps/ios/) |
| Shared TypeScript client and fidelity rules | [`packages/dictation/`](packages/dictation/) |
| Language-independent API contract | [`packages/contracts/`](packages/contracts/) |
| Deprecated Python/CUDA serving | [`backends/python/`](backends/python/) · reference source in `src/starling/` |

The NVIDIA-only Python serving path is deprecated. Its kernels, benchmarks,
and existing commands remain for research and reproducibility. New app and
serving work targets the native engine and the portable HTTP contract.

## Documentation

[Architecture and platform status](docs/monorepo.md) · [API](docs/api.md) ·
[TypeScript development](docs/typescript.md) ·
[Models](docs/models.md) · [Native serving](docs/native-serving.md) ·
[Quantization tools](quants/README.md) · [Quantization research](docs/quantization.md) ·
[Benchmarks](docs/benchmarks.md) · [Engine development](docs/ggml-engine.md)
