# ggml engine — universal backend ASR for Starling

Starling's in-tree `starling-ggml-parakeet` and `starling-ggml-moss` engines
bring a second, **universal-backend** transcription path alongside the PyTorch
+ CUDA-graph peak engine. Both are model-tagged entry points in the shared
`libstarling_ggml` C API and reach any backend ggml supports (NVIDIA CUDA, AMD
via HIP, Apple Metal, Intel/AMD/ARM via Vulkan, CPU) from one codebase. Legacy
external `GgmlParakeet` and `GgmlMoss` wrappers remain only for temporary A/B
comparison and are deprecated pending Phase-4 removal.

The PyTorch engine remains the NVIDIA peak path (CUDAGraph + Triton fused
kernels, tuned on sm_120). The ggml engines are dispatched alongside it: same
fixtures, same golden contract, portable. **Correctness:** parakeet-tdt and the
in-tree MOSS engine return the exact canonical eager token IDs/text on all
fixtures. MOSS logits retain documented bf16 ULP differences, so its component
gates use max-abs tolerances while the end-to-end contract is token/text exact.
The legacy CrispASR MOSS engine remains only a near-exact comparison path.
**Speed:** parakeet-tdt
short is at parity with the PyTorch peak; medium/long within ~1.5-1.8x
(see `docs/ggml-parakeet-perf-analysis.md`); moss is slower
(one-shot/per-server load path) and not yet at parity.

## Correctness contract

Byte-exact transcripts vs the golden references (`golden/parakeet_tdt_*.txt`
for parakeet, `golden/moss_*.txt` for moss), asserted by
`tests/test_ggml_parity.py` (skipped if the ggml binaries aren't built).

- **parakeet-tdt**: byte-exact on short/medium/long. `parakeet.cpp` reproduces
  the eager greedy-TDT token stream bit-for-bit, including the subtle comma
  variations in the long fixture.
- **moss (in-tree `StarlingGgmlMoss`)**: exact eager greedy token IDs and text
  on short/medium/long. The reference explicitly propagates eager attention to
  nested model configs and uses exact-width `DynamicCache`; padded StaticCache
  reduction-order noise is not a golden contract. Component ULP tolerances are
  specified in `docs/ggml-moss-goldens.md`.
- **moss (legacy external CrispASR)**: short is byte-exact; medium/long retain
  the historical normalized-CER gate and single-chunk workaround. This engine
  is deprecated and does not define Starling's in-tree correctness.

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

## One-shot graph safety

`run_graph` graphs are transient (`cgraph->uid == 0`); only persistent
`ReplayGraph` instances receive stable nonzero UIDs and may use ggml-CUDA graph
capture (patch 0008). This avoids pointer-key collisions across recycled
one-shot metadata contexts. Intermediate captures must be expanded explicitly,
but diagnostic capture branches are not a numerical oracle: changing which
tensors are marked output changes gallocr reuse. In particular, never mark a
graph-input leaf as an output merely to inspect it. MOSS's former
`ggml_set_output(mask_input)` experiment changed the allocation layout and
produced non-deterministic mask/softmax garbage; the durable LLM parity probes
instead select an intermediate as the graph's normal output via
`STARLING_MOSS_L0_STAGE`.

## Performance (RTX 5090, bf16, B=1, model load excluded)

Wall-clock per utterance, harness `bench_all.py` (20 reps, steady-state) and
the in-process ctypes path (median of 7, steady-state):

| length | audio | ggml (harness) | ggml (ctypes) | starling (harness) | gap   |
|--------|-------|----------------|---------------|--------------------|-------|
| short  | 7.4s  | 16 ms          | 16 ms         | 16 ms              | 1.00x |
| medium | 22.3s | 38 ms          | 37 ms         | 26 ms              | 1.46x |
| long   | 74.3s | 108 ms         | 126 ms        | 60 ms              | 1.80x |

WER 0.00% (byte-exact) on all three. **Short is at parity with the PyTorch
peak.** The remaining medium/long gap is the TDT decode loop (data-dependent,
serial — each step's argmax determines the next token), now GPU-compute-bound
after the K-step multistep + double-sync elimination. The encoder graph
itself is ~2.8ms GPU compute (faster than starling's on raw compute). See
`docs/ggml-parakeet-perf-analysis.md` for the full per-phase breakdown.

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
MOSS is now built into Starling's shared `libstarling_ggml` alongside
parakeet. Build the in-tree library from the repository root:
```
flock /tmp/starling-cpp-build.lock bash -c \
  'cmake -B build -DSTARLING_GGML_CUDA=ON -DSTARLING_GGML_SHARED=ON && cmake --build build -j'
```

Place Starling's exact BF16 GGUF at
`models/moss-transcribe-preview-2b-bf16-exact.gguf`, or override it with
`STARLING_GGML_MOSS_MODEL=/path/to/model.gguf`. The benchmark key is
`starling-ggml-moss`; it loads once and calls the in-tree C API directly. For
example, the Python binding is `GgmlModel(MOSS, path)` from
`starling._ggml`.

The legacy external `GgmlMoss` CrispASR engine remains available temporarily
for A/B comparisons, but is **deprecated** and will be removed in Phase 4.
It is not required to build or run the Starling-owned MOSS path.
