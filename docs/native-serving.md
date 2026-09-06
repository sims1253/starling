# Native server: starling-serve

`starling-serve` runs GGUF models behind an HTTP and WebSocket API. It uses
`libstarling_ggml` and does not require Python, PyTorch, Transformers, or Triton.
GPU builds still need the platform's driver and runtime libraries.

Start with [build](#build) and [usage](#usage), or see
[release artifacts](#release-artifacts) for binary prerequisites. The
[API contract](#api-contract) covers differences from `starling.server`.

## Build

Run these commands from a checkout with submodules initialized. You need
CMake 3.18 or later, a C++17 compiler, Git, and Bash (Git Bash on Windows).
CMake validates and applies the ggml patches during configuration.
GPU builds also need the matching development toolkit: CUDA, ROCm, the Vulkan
SDK, or Apple's Metal tools. Choose one backend below and start with a fresh
`build` directory. If you keep several builds, give each a separate directory.

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
the CRT across the build. This does not bundle GPU runtime libraries.

## Usage

```bash
# Verify compatibility at startup:
starling-serve --version
starling-serve --abi-version

# Serve a model:
starling-serve --model parakeet --gguf model.gguf --port 8181 [--warmup]
```

The built executable is `build/starling-serve` (`build/Release/starling-serve.exe`
with a multi-configuration Windows generator). Replace `starling-serve` in the
examples with that path, or add its directory to `PATH`. The optional `--warmup`
flag runs a warmup at startup; omit the square brackets when using it.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <slug>` | (required) | Model slug: parakeet, moss, ark, higgs, hojo, granite, qwen3, s1, audex |
| `--gguf <path>` | (required) | Path to the GGUF model file |
| `--host <addr>` | `127.0.0.1` | Bind address |
| `--port <n>` | `8181` | Bind port |
| `--warmup` | off | Capture CUDA graphs on startup |
| `--no-eager-load` | off | Defer model load to first request |
| `--idle-timeout <s>` | `0` (never) | Shut down after N seconds idle |
| `--request-timeout-seconds <s>` | `600` | Fail queued requests after N s waiting for the engine (`504`); same flag as the Python server |
| `--stream-chunk-seconds <s>` | `12.0` | Fixed WS stream window |
| `--stream-overlap-seconds <s>` | `3.0` | Overlap between windows |
| `--min-chunk-seconds <s>` | `5.0` | Min audio before first partial |
| `--partial-interval-seconds <s>` | `3.0` | Min gap between partials |
| `--max-stream-seconds <s>` | `60.0` | Per-WS-connection LIVE buffer cap in s (0 = unlimited); see `WS /stream` |
| `--version` | n/a | Print version + ABI + backend, exit |
| `--abi-version` | n/a | Print ABI version integer, exit |

## API contract

The servers share audio routes and streaming messages. Clients must account
for these differences:

- **Inputs**: this server requires 16 kHz audio (below); the Python server
  resamples non-16 kHz WAVs via scipy instead of rejecting them. Native HTTP
  uploads also accept raw mono PCM16; Python HTTP uploads require WAV. Both
  WebSocket endpoints accept PCM16 and WAV.
- **Models**: both serve `parakeet`, `moss`, `ark`, `higgs`, `granite`, `qwen3`,
  and `audex`. Native serving also supports `hojo` and `s1`; Python serving
  also supports `parakeet_unified` and `cohere`. Native `s1` exposes the
  additional `POST /normalize` text endpoint.
- **Request ids**: `X-Request-Id` values starting with `#` are rejected
  with `400`: the prefix is reserved for the server's internal queue
  tickets. The Python server accepts them.
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

Accepts raw WAV bytes (or multipart/form-data). **Audio must be 16 kHz**. The native server has no resampler, so WAVs at other
sample rates are rejected with
`400` and a `sample rate mismatch` error (the Python server resamples via
scipy instead). WAV parsing is bounded by the actual payload size: a header
whose claimed frame count exceeds what the payload can hold (crafted or
truncated) is rejected with `400` and a `malformed audio payload` error: it
is never reinterpreted as raw PCM. Payloads without RIFF/WAVE magic are
treated as raw mono PCM16 @ 16 kHz little-endian. Returns:

```json
{"text":"hello world","segments":[{"text":"hello world","start_s":0.0,"end_s":2.5}],"duration_s":2.5,"request_id":"..."}
```

Uses `X-Request-Id` header for tracking. Errors map to: `400` malformed
audio / sample-rate mismatch / invalid request id, `409` duplicate active
request id, `413` request body too large, `499` cancelled, `503` busy or
model not loaded, `504` queue timeout, `500` other engine failures.

### `POST /warmup`

Idempotent warmup (CUDA graph capture): a silent clip for audio models, a
probe transcript for text models (s1). Returns `202`.

### `POST /normalize` (s1 only)

Text-in/text-out path for the normalizer. JSON body:

```json
{"transcript":"so um i need to send the the report by uh friday","styling":"semi-formal","structure":"prose","context":"general"}
```

`transcript` is required; the control fields are optional (defaults
`semi-formal`/`prose`/`general`) and must come from the trained sets. Unknown
values are rejected with `400` (the card warns off-spec controls make
the model hallucinate). Prompts over ~1,000 tokens (the trained input max)
are rejected with `400`; chunk long transcripts at sentence boundaries
first. Returns:

```json
{"text":"So I need to send the report by Friday.","request_id":"..."}
```

Audio models answer `400` ("model has no text path"): use `/transcribe`.

### `DELETE /inference/<id>`

Cancels a queued or in-flight request by request ID.

### `WS /stream`

Real-time streaming dictation. Send binary frames (raw PCM16 or WAV) and
receive JSON messages:

- `{"type":"partial","text":"...","start_s":0.0,"end_s":12.5}`: growing partial
- `{"type":"final","text":"...","segments":[...],"duration_s":12.5}`: on commit
- `{"type":"error","message":"..."}`: on error
- `{"type":"pong"}`: in response to `{"type":"ping"}`
- `{"type":"reset_ack"}`: in response to `{"type":"reset"}`

**Buffer cap** (`--max-stream-seconds`, default 60 s): a binary frame that
would push the session's live audio buffer past the cap is refused. The
server emits one error frame:

```json
{"type":"error","message":"stream buffer limit reached (60 s live buffer); audio ignored until reset"}
```

and then ignores all further audio frames for that session (no more partials,
no crash, memory bounded). The client sends `{"type":"reset"}` to clear the
session and start accepting audio again.

The cap bounds the **live** buffer (memory), not cumulative audio: finalized
windows are trimmed from the buffer as the stream advances, so a long
dictation session without commits keeps memory bounded without tripping the
cap. It only fires when the un-finalized buffer itself grows past the limit
(e.g. streaming faster than the engine finalizes, or a session where
transcription never succeeds).

Control frames (JSON text):

- `{"type":"commit"}`: finalize all buffered audio. Returns `final` on success;
  if bounded retries stay busy, returns `{"type":"error","message":"server busy"}`
  and retains the audio. Retry `commit` after a delay.
- `{"type":"reset"}`: discard buffer without finalizing (returns reset_ack;
  also re-enables audio after a buffer-cap error)
- `{"type":"ping"}`: heartbeat (returns pong)

## Architecture

```text
cpp/serve/
├── main.cpp            — CLI parsing, lifecycle, HTTP/WS transport (cpp-httplib)
├── server.hpp/.cpp     — StarlingServer: model lifecycle, serial queue, transcribe
├── stream_session.hpp/.cpp — Rolling buffer + ChunkStreamer (port of Python logic)
└── audio.hpp/.cpp      — WAV/PCM decoding (dr_wav) + multipart extraction
```

The C++ and Python streaming sessions use fixed overlapping windows, including
during busy retries. Successful windows advance the committed boundary;
incomplete commits preserve the remaining audio for a later retry. Transcript
stitching uses matching words rather than timestamps, so disagreements between
neighboring windows can still omit or duplicate words.

## Pre-converted GGUF files

Download Parakeet weights from
[`scholzmx/parakeet-tdt-0.6b-v3-gguf`](https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf).
The repository includes Q8_0, K-quants, IQ2_XXS, and the importance matrix.
With the Hugging Face CLI installed:

```bash
hf download scholzmx/parakeet-tdt-0.6b-v3-gguf \
  parakeet-tdt-0.6b-v3-q8_0.gguf --local-dir ./models
starling-serve --model parakeet \
  --gguf ./models/parakeet-tdt-0.6b-v3-q8_0.gguf --port 8181
```

The planned `starling/*-gguf` repositories are not public downloads.
For other models, use the converters below with the original model weights.
The [GGUF file guide](hf-gguf-readme.md) describes filenames and metadata.

### Quantization

Choose `q8_0` for smaller weights where it is available, or `bf16-exact` for
reference comparisons. Exact transcript parity depends on the model, backend,
and input; the filename alone does not guarantee it. See the
[engine parity notes](ggml-engine.md#correctness-contract) and
[quantization guide](quantization.md) for measured results.

### GGUF converters

| Model | Converter script |
|-------|-----------------|
| parakeet | `scripts/convert_parakeet_gguf.py` |
| moss | `scripts/convert_moss_gguf.py` |
| ark | `scripts/convert_ark_gguf.py` |
| higgs | `scripts/convert_higgs_gguf.py` |
| hojo | `scripts/convert_hojo_gguf.py` |
| granite | `scripts/convert_granite_gguf.py` |
| qwen3 | `scripts/convert_qwen3_gguf.py` |
| s1 | `scripts/convert_s1_gguf.py` |
| audex | `scripts/convert_audex_gguf.py` |

## Release artifacts

The release workflow packages six executables with SHA-256 checksums in
`.tar.gz` archives on Linux/macOS and `.zip` archives on Windows. It links
Starling and ggml into the executable, but does not bundle accelerator runtime
libraries. These are not fully static binaries.

| Backend | Runtime prerequisites |
| --- | --- |
| CUDA | Compatible NVIDIA driver and CUDA runtime/cuBLAS libraries. The workflow builds with CUDA 13.3. |
| ROCm / HIP | Compatible AMD driver, HIP runtime, hipBLAS, and rocBLAS libraries. The workflow installs ROCm from its `latest` repository. |
| Vulkan | Vulkan loader and a compatible GPU driver. |
| Metal | Apple Silicon macOS with the system Metal frameworks. |

The workflow runs startup checks on its build machines, where the development
toolkits are already installed. Those checks do not establish that the archives
run on a clean machine. Missing-runtime packaging is tracked in
[issue #57](https://github.com/sims1253/starling/issues/57).

Choose the executable for your operating system, CPU architecture, and GPU:

| Artifact | Platform | Backend | Notes |
|----------|----------|---------|-------|
| `starling-serve-linux-cuda` | Linux x86_64 | CUDA | NVIDIA |
| `starling-serve-linux-rocm` | Linux x86_64 | ROCm / HIP | AMD Radeon & Instinct |
| `starling-serve-linux-vulkan` | Linux x86_64 | Vulkan | Intel / AMD / NVIDIA |
| `starling-serve-windows-cuda.exe` | Windows x86_64 | CUDA | NVIDIA (static application CRT) |
| `starling-serve-windows-vulkan.exe` | Windows x86_64 | Vulkan | AMD / Intel / NVIDIA (static application CRT) |
| `starling-serve-macos-metal` | macOS arm64 | Metal | Apple Silicon |

### GPU selection

Each release variant includes a specific GPU backend. Download the variant for
your platform and GPU; the executable cannot add a backend that was not compiled
in. Within the build, the runtime chooses the first GPU or integrated GPU.
Set `STARLING_GGML_DEVICE` to a device name such as `CUDA0`, `Vulkan0`, or `Metal`
to select it, or to `cpu` to force the CPU backend.
