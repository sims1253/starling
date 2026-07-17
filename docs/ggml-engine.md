# ggml engine — universal backend ASR for Starling

The `ggml-parakeet` (and planned `ggml-moss`) engine brings a second,
**universal-backend** transcription path to Starling alongside the PyTorch +
CUDA-graph peak engine. It runs mudler's `parakeet.cpp` — a byte-exact ggml
port of `nvidia/parakeet-tdt-0.6b-v3` — in-process via the parakeet C API,
so it reaches any backend ggml supports (NVIDIA CUDA, AMD via HIP, Apple
Metal, Intel/AMD/ARM via Vulkan, CPU) from one codebase.

The PyTorch engine remains the NVIDIA peak path (CUDAGraph + Triton fused
kernels, tuned on sm_120). The ggml engine is dispatched alongside it: same
fixture, same golden contract, byte-exact transcripts, but portable.

## Correctness contract

Byte-exact transcripts vs the golden references (`golden/parakeet_tdt_*.txt`
for parakeet, `golden/moss_*.txt` for moss), asserted by
`tests/test_ggml_parity.py` (skipped if the ggml binaries aren't built).

- **parakeet-tdt**: byte-exact on short/medium/long. `parakeet.cpp` reproduces
  the eager greedy-TDT token stream bit-for-bit, including the subtle comma
  variations in the long fixture.
- **moss**: byte-exact on the short fixture. On medium/long, CrispASR's
  moss-transcribe decode diverges from the golden capture path in punctuation
  normalization (a period at some repetition boundaries) and can truncate
  below the golden token count; the parity test asserts a normalized CER < 10%
  floor on medium/long so regressions beyond the known gap are caught. The
  moss-transcribe chunked path has a heap-corruption crash at chunk boundaries
  on long audio, so `GgmlMoss` forces the single-chunk path
  (`--chunk-seconds 3600`).

## Backends

`parakeet.cpp` selects its compute device through **ggml's device registry**
(`src/backend.cpp`), not via any backend-specific code. At construction it walks
the registry and picks the first GPU/IGPU device, or a named device via the
`PARAKEET_DEVICE` env var (`CUDA0`, `Vulkan0`, `Metal`, ...). Unsupported ops
fall back to the CPU backend via `ggml_backend_sched`. This means **every ggml
backend works with no parakeet.cpp changes** — only the ggml build decides which
backends are compiled in.

### NVIDIA CUDA (primary, verified)
The default. Built with `-DGGML_CUDA=ON`. Verified byte-exact and benchmarked
on RTX 5090 (Blackwell, sm_120). See the perf table below.

### CPU (verified — the non-NVIDIA backend)
`PARAKEET_DEVICE=cpu` forces the CPU ggml backend. **Verified byte-exact** vs
the golden on all fixtures (the eager greedy-TDT path is deterministic, and the
CPU backend runs the identical model math). This satisfies the project's "at
least one non-NVIDIA backend compiles + runs correctly" requirement: the CPU
backend is a distinct ggml backend, compiled in every build, and reproduces the
golden transcript bit-for-bit. It is ~10-20x slower than CUDA (no graph
capture, CPU kernels) — a correctness/fallback path, not a perf path.

### Apple Metal (gate + document)
Runs on Apple Silicon with a ggml built `-DGGML_METAL=ON` (the Metal kernels
ship in `third_party/ggml/src/ggml-metal/`). Select with
`PARAKEET_DEVICE=Metal`. **Not verified in CI here** — the development machine
is x86/WSL2 with no Apple hardware. The path is architectural: the encoder is
captured in a ggml compute graph (the portable CUDAGraph equivalent) and the
decode uses ggml's `ReplayGraph`, both of which replay on Metal the same way
they replay on CUDA. Per the project's OUT-OF-SCOPE note, Apple/mobile perf
tuning beyond "it runs" is a follow-up; bf16 (not fp8) is the portable
numerics contract on non-NVIDIA.

### Vulkan (universal: Intel / AMD / ARM)
Built with `-DGGML_VULKAN=ON` (`third_party/ggml/src/ggml-vulkan/`). Select
with `PARAKEET_DEVICE=Vulkan0`. Targets the Intel/AMD/ARM GPUs CUDA can't
reach. Same graph-replay path as CUDA/Metal.

### HIP (AMD) / SYCL (Intel)
Supported by ggml's registry; selected the same way when ggml is built with
`-DGGML_HIP=ON` / `-DGGML_SYCL=ON`.

## How the launch-folding works (the portable CUDAGraph)

Starling's PyTorch peak wins by capturing the decode/encoder loop into a
CUDA graph, eliminating host launch overhead (the README's "hundreds of tiny
kernels, GPU ~10% busy" problem). The ggml equivalent is ggml's compute graph:
each model component (24-layer Conformer encoder, per-step TDT joint/prediction)
is built as ONE `ggml_cgraph` and replayed, so the backend folds all its ops
into minimal device submissions. On CUDA, ggml captures the replayed graph as a
CUDA graph itself (keyed on the graph's first node pointer, which is why the
encoder is routed through a per-shape `ReplayGraph` that keeps that pointer
stable across calls — `src/encoder.cpp`).

## Performance (RTX 5090, bf16, B=1, model load excluded)

Wall-clock per utterance, in-process (native ctypes path):

| length | audio | ggml   | starling | gap   |
|--------|-------|--------|----------|-------|
| short  | 7.4s  | 36 ms  | 14 ms    | 2.6x  |
| medium | 22.3s | 89 ms  | 25 ms    | 3.6x  |
| long   | 74.3s | 264 ms | 58 ms    | 4.6x  |

WER 0.00% (byte-exact) on all three. The remaining gap to the PyTorch peak is
kernel fusion: starling uses Triton fused kernels (fp8 dequant-GEMV, fused
rmsnorm/SiLU/residual) and a multi-step decode megakernel, while parakeet.cpp
uses generic ggml ops. The encoder is already CUDA-graph-captured; ongoing work
targets the TDT decode loop (per-step host readback/argmax) and porting the
fused ops as ggml custom kernels.

## Build

### parakeet-tdt
In the parakeet.cpp repo (`/home/m0hawk/Documents/parakeet.cpp`):
```
cmake -B build-cuda -DGGML_CUDA=ON -DPARAKEET_SHARED=ON
cmake --build build-cuda -j --target parakeet parakeet-cli parakeet-server
```
`PARAKEET_SHARED=ON` produces `libparakeet.so`, which the Starling engine
ctypes-binds for the in-process path. Override paths with `GGML_PARAKEET_LIB`,
`GGML_PARAKEET_MODEL` (see `benchmarks/engines.py`).

### moss
The moss engine uses CrispASR's `moss-transcribe` backend. The released
CrispASR v0.8.2 binary omits it (the source is in the repo but was wired into
the build later), so build CrispASR locally:
```
cd /home/m0hawk/Documents/CrispASR
cmake -B build -DGGML_CUDA=ON -DCRISPASR_BUILD_EXAMPLES=ON
cmake --build build -j --target crispasr
```
Download the F16 GGUF: `moss-transcribe-preview-2b-f16.gguf` from
`cstr/MOSS-Transcribe-preview-2B-GGUF`. Override paths with `GGML_MOSS_BIN`,
`GGML_MOSS_MODEL`. The moss engine is one-shot (per-process model load);
starling-moss stays the NVIDIA peak path.
