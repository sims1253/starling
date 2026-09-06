# Starling

Starling runs speech recognition locally. It has Python pipelines that use
CUDA graphs on NVIDIA GPUs and native ggml engines for CPU, CUDA, Metal,
Vulkan, and HIP. Both paths offer an HTTP and WebSocket server.

Models include Parakeet, Granite Speech, Qwen3-ASR, MOSS, and others.
[S1-mini](docs/models.md) can clean up a transcript after recognition.
See the [model list](docs/models.md) for architectures and model licenses.

## Transcribe a file

For the Python server, you need Linux or Windows, Python 3.10 to 3.12,
[uv](https://github.com/astral-sh/uv), and an Ampere or newer NVIDIA GPU.
The project uses CUDA 13.0 PyTorch wheels. On Windows, first set
`STARLING_GPU_LOCK_DISABLE=1` because the process lock requires POSIX flock.
Run these commands from the repository:

```bash
uv sync --extra server
uv run --extra server starling-python-serve --model parakeet --port 8181
```

In another terminal, send a WAV file:

```bash
curl http://127.0.0.1:8181/inference -F "file=@recording.wav"
```

The response contains `text`, `segments`, `duration_s`, and `request_id`.
The server keeps the model loaded and queues concurrent requests.
Read [Python serving](docs/python-serving.md) for streaming, profiles,
GPU selection, Windows setup, and request limits.

## Run without Python

Build `starling-serve` and load a GGUF model:

```bash
git submodule update --init --recursive
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON
cmake --build build -j --target starling-serve
./build/starling-serve --model parakeet --gguf model.gguf --port 8181
```

Use the same curl request above with a 16 kHz WAV file. The native server
requires 16 kHz audio. See [native serving](docs/native-serving.md) for
GGUF downloads, CPU and other GPU builds, and API differences.

## Documentation

| I want to… | Read |
| --- | --- |
| Choose a model | [Models](docs/models.md) |
| Configure the Python server | [Python serving](docs/python-serving.md) |
| Build or use the native server | [Native serving](docs/native-serving.md) |
| Compare speed and accuracy | [Benchmarks](docs/benchmarks.md) |
| Reduce model memory use | [Quantization](docs/quantization.md) |
| Embed or extend a native engine | [ggml engine guide](docs/ggml-engine.md) |

The Python kernels are tuned on an RTX 5090. Benchmark results include
hardware and workload details; fixture parity alone does not establish
accuracy across a corpus or across backends.
