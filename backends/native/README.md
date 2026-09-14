# Native serving backend

This is the primary Starling backend. `CMakeLists.txt` owns the native targets;
engine and transport sources remain in `../../cpp/` so existing C API bindings,
benchmark scripts, and downstream includes keep their paths.

Configure from this component or use the root compatibility entry point:

```bash
cmake -S backends/native -B build/native -DSTARLING_SERVE=ON
cmake --build build/native -j --target starling-serve starling-quantize
./build/native/starling-serve --model parakeet --gguf /path/to/model.gguf
```

CPU is the default. Metal, Vulkan, HIP, and the optional legacy CUDA accelerator
use the same API. See [native serving](../../docs/native-serving.md) for flags.
Python, PyTorch, and Triton are not build or runtime dependencies.

Run HTTP contract tests without model files:

```bash
cmake --preset native-cpu
cmake --build build/native-cpu --target starling-serve-contract-fixture -j
python -m unittest discover -s backends/native/tests -v
```

The fixture target links a deterministic test engine into the real HTTP server.
It validates the protocol and raw text preservation, not speech recognition.
It is excluded from default builds and release targets.
