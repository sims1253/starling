# Native Serving Layer: starling-serve

A self-contained native binary that wraps `libstarling_ggml` behind the same
endpoint surface as the Python `starling.server` (same routes and message
shapes; client-visible differences are listed under "API contract"):

```typescript
// Before (Python subprocess):
spawn("python", ["-m", "starling.server", "--model", slug, ...])

// After (native binary):
spawn("starling-serve", ["--model", slug, "--gguf", ggufPath, "--port", "8181"])
```

No Python, no torch, no transformers, no triton — just a single static binary
per platform.

## Build

```bash
# CPU-only (development / smoke tests):
cmake -B build -DSTARLING_SERVE=ON
cmake --build build -j --target starling-serve

# CUDA (NVIDIA production path):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON
cmake --build build -j --target starling-serve

# ROCm / HIP (AMD Radeon & Instinct path on Linux):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_HIP=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j --target starling-serve

# Vulkan (universal Intel/AMD/ARM path):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_VULKAN=ON
cmake --build build -j --target starling-serve

# Metal (macOS):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j --target starling-serve
```

On Windows, add `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` to statically link
the CRT (no VC++ redistributable dependency).

## Usage

```bash
# Verify compatibility at startup:
starling-serve --version
starling-serve --abi-version

# Serve a model:
starling-serve --model parakeet --gguf model.gguf --port 8181 [--warmup]
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <slug>` | (required) | Model slug: parakeet, moss, ark, higgs, hojo |
| `--gguf <path>` | (required) | Path to the GGUF model file |
| `--host <addr>` | `127.0.0.1` | Bind address |
| `--port <n>` | `8181` | Bind port |
| `--warmup` | off | Capture CUDA graphs on startup |
| `--no-eager-load` | off | Defer model load to first request |
| `--idle-timeout <s>` | `0` (never) | Shut down after N seconds idle |
| `--stream-chunk-seconds <s>` | `12.0` | Fixed WS stream window |
| `--stream-overlap-seconds <s>` | `3.0` | Overlap between windows |
| `--min-chunk-seconds <s>` | `5.0` | Min audio before first partial |
| `--partial-interval-seconds <s>` | `3.0` | Min gap between partials |
| `--version` | — | Print version + ABI + backend, exit |
| `--abi-version` | — | Print ABI version integer, exit |

## API contract

Mirrors the Python server's endpoint set, but the two are not drop-in
identical. A client written against one needs three adjustments on the other:

- **Inputs**: this server requires 16 kHz audio (below); the Python server
  resamples non-16 kHz WAVs via scipy instead of rejecting them.
- **Phase names**: `unloaded → loading → ready → busy` here; the Python
  server reports `loading_weights` and `warming_up` during startup.
- **Error responses**: the status-code and body details documented below
  differ in places from the Python server's.

### `GET /health`

```json
{"status":"ok","model":"parakeet","loaded":true,"busy":false,"phase":"ready","queue_depth":0}
```

Phase drives the UI: `unloaded → loading → ready → busy`.

### `POST /transcribe` / `POST /inference`

Accepts raw WAV bytes (or multipart/form-data). **Audio must be 16 kHz** —
there is no C++ resampler, so WAVs at other sample rates are rejected with
`400` and a `sample rate mismatch` error (the Python server resamples via
scipy instead). Payloads that aren't valid WAV are treated as raw mono
PCM16 @ 16 kHz little-endian. Returns:

```json
{"text":"hello world","segments":[{"text":"hello world","start_s":0.0,"end_s":2.5}],"duration_s":2.5,"request_id":"..."}
```

Uses `X-Request-Id` header for tracking. Errors map to: `400` malformed
audio / sample-rate mismatch, `409` duplicate active request id, `499`
cancelled, `503` busy or model not loaded, `504` queue timeout, `500` other
engine failures.

### `POST /warmup`

Idempotent silent-clip warmup (CUDA graph capture). Returns `202`.

### `DELETE /inference/<id>`

Cancels a queued or in-flight request by request ID.

### `WS /stream`

Real-time streaming dictation. Send binary frames (raw PCM16 or WAV) and
receive JSON messages:

- `{"type":"partial","text":"...","start_s":0.0,"end_s":12.5}` — growing partial
- `{"type":"final","text":"...","segments":[...],"duration_s":12.5}` — on commit
- `{"type":"error","message":"..."}` — on error
- `{"type":"pong"}` — in response to `{"type":"ping"}`
- `{"type":"reset_ack"}` — in response to `{"type":"reset"}`

Control frames (JSON text):
- `{"type":"commit"}` — finalize all buffered audio (returns final)
- `{"type":"reset"}` — discard buffer without finalizing (returns reset_ack)
- `{"type":"ping"}` — heartbeat (returns pong)

## Architecture

```text
cpp/serve/
├── main.cpp            — CLI parsing, lifecycle, HTTP/WS transport (cpp-httplib)
├── server.hpp/.cpp     — StarlingServer: model lifecycle, serial queue, transcribe
├── stream_session.hpp/.cpp — Rolling buffer + ChunkStreamer (port of Python logic)
└── audio.hpp/.cpp      — WAV/PCM decoding (dr_wav) + multipart extraction
```

The streaming session logic (`ChunkStreamer` + `StreamSession`) is a faithful C++
port of `src/starling/stream_chunk.py` and the streaming portions of
`src/starling/server.py`. The fixed-window overlapping-chunk strategy bounds
work to O(N), keeps per-transcribe prompts bounded, and enables cudagraph
encoder reuse.

## Pre-converted GGUF files

Hosted on HuggingFace, one repo per model:

| Model | HF Repo | Quantizations |
|-------|---------|---------------|
| parakeet-tdt-0.6b-v3 | `starling/parakeet-tdt-0.6b-v3-gguf` | bf16-exact, q8_0 |
| moss-transcribe-preview-2b | `starling/moss-transcribe-preview-2b-gguf` | bf16-exact |
| ark-asr-3b | `starling/ark-asr-3b-gguf` | bf16-exact, q8_0 |
| higgs-audio-v3-stt | `starling/higgs-audio-v3-stt-gguf` | bf16-exact |
| hojo-asr-v1 | `starling/hojo-asr-v1-gguf` | bf16-exact |

Naming convention: `<model-slug>-<quant>.gguf` (e.g., `parakeet-tdt-0.6b-v3-q8_0.gguf`).

### Quantization strategy (recommendation)

- **q8_0** as the default download (halves VRAM with negligible WER delta).
- **bf16-exact** as an opt-in upgrade for users who want byte-exact parity with
  the Python reference and have VRAM to spare.

### GGUF converters

| Model | Converter script |
|-------|-----------------|
| parakeet | `scripts/convert_parakeet_gguf.py` |
| moss | `scripts/convert_moss_gguf.py` |
| ark | `scripts/convert_ark_gguf.py` |
| higgs | `scripts/convert_higgs_gguf.py` |
| hojo | `scripts/convert_hojo_gguf.py` |

## Release artifacts

GitHub release publishes six static binaries + checksums:

| Artifact | Platform | Backend | Notes |
|----------|----------|---------|-------|
| `starling-serve-linux-cuda` | Linux x86_64 | CUDA | NVIDIA |
| `starling-serve-linux-rocm` | Linux x86_64 | ROCm / HIP | AMD Radeon & Instinct |
| `starling-serve-linux-vulkan` | Linux x86_64 | Vulkan | Intel / AMD / ARM |
| `starling-serve-windows-cuda.exe` | Windows x86_64 | CUDA | NVIDIA (no VC++ runtime dep) |
| `starling-serve-windows-vulkan.exe` | Windows x86_64 | Vulkan | AMD / Intel / NVIDIA (no VC++ runtime dep) |
| `starling-serve-macos-metal` | macOS arm64 | Metal | Apple Silicon |

### GPU detection

Each build variant is compiled for a specific backend (CUDA, ROCm/HIP, Metal,
Vulkan, or CPU). There is no runtime backend auto-selection: download the
binary matching the detected platform and GPU. Explicit variant selection is
more predictable than runtime auto-detection.

## Open questions (resolved)

1. **Quantization strategy**: Default download is q8_0 (halves VRAM, negligible
   WER delta); bf16-exact is an opt-in upgrade.

2. **GPU detection**: Resolved in favor of explicit variant selection — no
   runtime backend auto-selection (see above).

3. **Streaming buffer implementation**: Ported to C++ (lower latency, no per-chunk
   HTTP round-trips). The `ChunkStreamer` is a faithful port of the Python
   `stream_chunk.py`.
