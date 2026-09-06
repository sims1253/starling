# Python serving

Run one speech recognition model as a local HTTP and WebSocket service.
For CPU, AMD GPUs, or Apple Silicon, use [native serving](native-serving.md).


## Requirements

- An Ampere or newer NVIDIA GPU (RTX 30/40/50, A100, H100) with bf16 support.
  The kernels are tuned on an RTX 5090 (sm_120).
- Python 3.10 to 3.12 and [uv](https://github.com/astral-sh/uv).
  `pyproject.toml` selects CUDA 13.0 PyTorch wheels.

Install the server dependencies from the repository:

```bash
uv sync --extra server
```

On Windows, set `STARLING_GPU_LOCK_DISABLE=1` before starting the server.
The process lock uses POSIX flock, which is unavailable on Windows.

## Kernel backends

The Python pipelines run on Linux and native Windows. They require NVIDIA
CUDA graphs; use the native engines on macOS. Model modules select their
kernels through `src/starling/_kernels/`.

Automatic selection tries these backends in order:

| Backend | When it is used | Notes |
| --- | --- | --- |
| Triton | Available on Linux | Autotuned kernels used for the recorded benchmarks. |
| CUDA C++ | CUDA toolkit and compiler available | Compiles on first use, then caches the result under `~/.cache/starling`. |
| Torch | Neither backend above is available | Leave `fp8_weights` off: this backend expands FP8 weights to bf16. |

The recorded CUDA C++ kernel measurements were 1 to 2 microseconds for
elementwise operations and 2.6 to 8 times faster than Torch for FP8.
First-use compilation took about 30 to 60 seconds.
`tests/test_kernel_backends.py` checks elementwise operations for exact
equality and RoPE and quantized operations with numerical tolerances.
Model fixture checks do not establish parity across every backend or GPU.

Select the backend explicitly with the `STARLING_KERNEL_BACKEND` env var
(`auto` | `triton` | `cuda` | `torch`) before importing model modules.
`auto` tries `triton`, then CUDA when a GPU is visible, then `torch`. CUDA
compilation happens during backend resolution; if it fails, automatic selection
uses `torch`. An explicit backend request reports initialization errors.
`get_backend_name()` resolves the backend and reports the one actually in use.

To recreate the environment and check core imports on either platform:

```bash
python scripts/setup_env.py
```

Compare kernel backends on a machine that has all three installed:

```bash
uv run --extra bench python benchmarks/bench_kernels.py
```


## Run the server

`src/starling/server.py` is a long-lived local HTTP/WebSocket sidecar that keeps
one model resident in VRAM. One process runs one model at a time (`--model
granite|parakeet|parakeet_unified|moss|qwen3|ark|cohere|higgs|audex`, default
`granite`); `/health` reports which is loaded. Higgs must be run from the
isolated `.venv-higgs` environment documented in [Higgs environment notes](../src/starling/higgs/UV_NOTES.md).

```bash
uv run python -m starling.server --model granite --port 8181 --max-chunk-seconds 30
uv run python -m starling.server --model parakeet --profile realtime --warmup
uv run python -m starling.server --model moss --profile batch  # SDPA + fused fp8
```

On multi-GPU hosts, set `CUDA_VISIBLE_DEVICES` to a full GPU UUID from
`nvidia-smi -L`, for example `CUDA_VISIBLE_DEVICES=GPU-... python -m starling.server`.
The process lock rejects numeric masks on these hosts because CUDA ordinals
can differ from `nvidia-smi` order. Automatic locking also requires working
NVIDIA discovery. If it fails, inference returns JSON 500 and the server log
contains the configuration error. `STARLING_GPU_LOCK_DISABLE=1` bypasses this
lock when GPU access is serialized externally; it is also needed on Windows,
where POSIX flock is unavailable.

The server uses FastAPI and Uvicorn. `uv sync --extra dev` includes the
server dependencies and development tools.

Endpoints:

| Method + path             | Purpose |
| ------------------------- | ------- |
| `GET  /` `/health`        | liveness + `phase` (`loading_weights`/`warming_up`/`ready`) and `queue_depth` |
| `POST /inference`         | multipart or raw WAV -> `{text, segments, duration_s, request_id}` |
| `POST /transcribe`        | multipart or raw WAV -> same shape as `/inference` |
| `POST /warmup`            | pre-capture CUDA graphs on a silent clip (idempotent; 202, or 409 when the model is not loaded) |
| `DELETE /inference/<id>`  | cancel a queued or running request by its `X-Request-Id` |
| `WS   /stream`            | real-time streaming dictation |

A single GPU worker serves one request at a time; concurrent requests queue
(up to `MAX_WAITERS`) and only get HTTP 503 when full. `X-Request-Id` on a POST
enables `DELETE /inference/<id>` cancellation, which is best-effort once on the GPU
(CUDA-graph replays aren't preemptible; an in-flight request finishes its
current step then returns HTTP 499).

Requests without an `X-Request-Id` receive a generated ID in the response.
Uploads are capped at 256 MiB and requests have a 10-minute wall-clock deadline
by default; tune these with `--max-upload-mb` and `--request-timeout-seconds`
(pass `0` or a negative value to disable the deadline entirely). A single GPU
worker serves one request at a time, so disabling the deadline can occupy it
indefinitely. Only use a non-positive value for trusted local use, or ensure
an upstream proxy enforces its own timeout. The API has no authentication, so
binding a non-loopback `--host` emits a warning and should only be done behind
an authenticated proxy.

A WebSocket `commit` returns a final transcript only after all buffered audio
has been transcribed. If retries remain busy, the server sends
`{"type":"error","message":"server busy"}` and retains the audio. Retry `commit`
after a delay; use `reset` only to discard the buffered session.

Profiles provide supported defaults for the main workloads:

| profile | intended workload | graph/optimization policy |
| ------- | ----------------- | ------------------------- |
| `file` (default) | one-shot files | adaptive graphs, strict flags |
| `realtime` | low-latency dictation | graphed recurring windows + tolerance-mode SDPA |
| `batch` | long-form offline throughput | graphed chunks + tolerance-mode SDPA, plus graph-safe fused fp8 weights on granite/moss |
| `accuracy` | baseline numerical behavior | adaptive graphs, approximate options disabled |

In the [recorded benchmarks](benchmarks.md), Parakeet has the lowest latency
and MOSS has the lowest average leaderboard WER. Compare the results for your
workload. Granite also has a self-speculative decoding path in Python.
The server serves one request at a time. For offline batching, `bench_all.py`
exposes the Granite and Qwen3 batched pipelines.

### Timestamps

`/inference` returns chunk-level segments
(`[{text, start_s, end_s}]`). LLM-decoder models (granite, moss, qwen3, ark,
higgs, audex) have no
per-token audio alignment, so segments are at `--max-chunk-seconds` granularity;
shrink it for finer segments at the cost of more decode passes. Parakeet
and cohere chunk internally and return a single whole-utterance segment.

