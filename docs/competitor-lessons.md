# Lessons from `transcribe-rs` and `transcribe.cpp`

Primary-source review, inspected 2026-07-21.

- [`cjpais/transcribe-rs`](https://github.com/cjpais/transcribe-rs) at commit
  `48ac240a54419e788078640d91c631a436ac4e49`.
- [`handy-computer/transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp)
  at commit `a951f0af73b5d6f6153729039ba32e3017dc65cf`.

The projects overlap Starling's product space, but not its primary optimization
strategy. Starling specializes in exact, CUDA-graph-heavy peak inference and is
already substantially faster on its tuned RTX 5090 workloads. The useful lesson
is therefore not to replace Starling's engine with either project. It is to
adopt their stronger model-family contracts, public API ergonomics, and
portable packaging while retaining Starling's specialized execution paths.

## What to keep

### 1. A uniform model-family porting contract

`transcribe.cpp` gives every family the same ladder:

1. synthetic GGUF loader smoke,
2. real-model structural smoke,
3. end-to-end public-C-ABI transcript smoke,
4. staged numerical dumps compared with per-family tolerance data,
5. reproducible benchmark reports.

Sources:

- [`docs/model-family-testing.md`](https://github.com/handy-computer/transcribe.cpp/blob/a951f0af73b5d6f6153729039ba32e3017dc65cf/docs/model-family-testing.md)
- [`docs/tools/validate.md`](https://github.com/handy-computer/transcribe.cpp/blob/a951f0af73b5d6f6153729039ba32e3017dc65cf/docs/tools/validate.md)
- [`docs/tools/reference-dumps.md`](https://github.com/handy-computer/transcribe.cpp/blob/a951f0af73b5d6f6153729039ba32e3017dc65cf/docs/tools/reference-dumps.md)

Starling already has strong goldens, but its gates have grown model-by-model.
Adopt this ladder for every new `cpp/` family. Keep strict token/text equality
where Starling can provide it; use tolerance files only for staged floating
point tensors, never to weaken transcript gates silently.

### 2. A stable, capability-driven C API

`transcribe.cpp`'s public header documents model/session lifetime, serialized
compute, size-aware structs, result ownership, timestamps, cancellation,
input/output truncation, and feature probes in one place:

- [`include/transcribe.h`](https://github.com/handy-computer/transcribe.cpp/blob/a951f0af73b5d6f6153729039ba32e3017dc65cf/include/transcribe.h)

Starling's current C API is intentionally small and sufficient for internal
ctypes use. Before the single-executable phase, deepen it around a loaded model
plus per-run/session state rather than adding one-off model entry points. Useful
pieces are capability queries, explicit result ownership, structured errors,
timings, and version/ABI metadata. Do not copy the entire broad API prematurely.

### 3. Feature-gated engines and honest capability reporting

`transcribe-rs` exposes a small common `SpeechModel` surface, engine-specific
options, compile-time backend features, and runtime accelerator/capability
queries:

- [`README.md`](https://github.com/cjpais/transcribe-rs/blob/48ac240a54419e788078640d91c631a436ac4e49/README.md)
- [`Cargo.toml`](https://github.com/cjpais/transcribe-rs/blob/48ac240a54419e788078640d91c631a436ac4e49/Cargo.toml)
- [`src/accel.rs`](https://github.com/cjpais/transcribe-rs/blob/48ac240a54419e788078640d91c631a436ac4e49/src/accel.rs)

For Starling, the transferable idea is the capability model—not an ONNX or Rust
rewrite. A future CLI should be able to ask a model whether it supports word or
segment timestamps, streaming, translation, batch, languages, and a selected
backend instead of inferring behavior from the model slug.

### 4. Streaming and VAD as separable interfaces

`transcribe-rs` keeps VAD behind an interface with both energy and Silero
implementations (`src/vad/`), while `transcribe.cpp` treats streaming as a
first-class session operation. Starling already has WebSocket streaming and
chunk logic, but the VAD/chunk policy is less reusable than the model engines.
A model-independent segmentation contract is worth adopting after the in-tree
CLI exists. It should remain outside model inference so accuracy/performance can
be benchmarked with and without segmentation.

### 5. Decode flash attention as a measured MOSS candidate

`transcribe.cpp` uses ggml flash attention, including native grouped-query
attention and valid-KV views, across its LLM-style decoders. Starling's MOSS
decode currently does not call `ggml_flash_attn_ext` even though the pinned ggml
exposes it. This is worth a **default-off, token-gated experiment**, not an
assumed win: MOSS decode is already near its weight-streaming floor and its long
fixture contains a known near-tie, so any reduction-order change can flip a
token. Benchmark only after exact short/medium/long IDs and text pass.

### 6. Release and binding discipline

`transcribe.cpp` supplies a minimal C example, generated bindings, versioned ABI
metadata, model converters, quantization tools, and canonical model artifacts.
Starling's north star already calls for one packaged executable; these are good
acceptance criteria for that phase. In particular, add one tiny pure-C example
as the executable/API stabilizes.

## What not to copy

- **Do not replace the tuned CUDA path with ONNX Runtime.** `transcribe-rs` is a
  broad integration library. Its published Parakeet CPU figures are useful for
  portability, not a performance target for Starling's RTX 5090 path.
- **Do not import another inference engine as a runtime dependency.** Starling's
  self-containment decision remains correct. `transcribe.cpp` is useful as a
  primary-source reference for model structure, converters, test manifests,
  and portable behavior—not as a binary Starling shells out to.
- **Do not claim dozens of models before enforcing parity.** Starling's strict
  staged and end-to-end gates are a differentiator. Breadth should follow the
  common family contract above.
- **Do not prioritize generic quantization solely for smaller artifacts.** The
  current Starling decode paths are often launch- or synchronization-bound;
  quantization must show an end-to-end win under the exactness/WER contract.

## Model-port opportunities exposed by `transcribe.cpp`

At the inspected commit, `transcribe.cpp` has implementations and conversion /
reference-validation material for several models Starling already supports in
PyTorch: Parakeet variants, Qwen3-ASR, Cohere Transcribe, Granite Speech/NAR,
and MOSS Transcribe-Diarize. This materially reduces architecture-research
risk for future Starling-owned ports. It does **not** remove the need to:

- write or adapt a Starling-owned GGUF converter and tensor map,
- reproduce Starling's existing goldens,
- satisfy Starling's stricter exactness contract,
- benchmark Starling's own runtime under the GPU lock,
- review license compatibility per checkpoint.

Good candidates are evaluated in `docs/ggml-roadmap.md`. The shortest path is
Parakeet Unified: `transcribe.cpp` demonstrates that RNN-T is a configuration of
the same Parakeet family (no duration head/classes), so Starling can reuse its
FastConformer, prediction, joint, and K-step infrastructure. Qwen3-ASR follows
because its Qwen3 decoder can reuse MOSS's device-resident runtime.

## Immediate actions

1. Keep the in-tree Parakeet text/content-token parity gate added alongside the
   GPU-isolation work.
2. Convert the next model port to the five-level family contract above rather
   than adding another bespoke collection of tests.
3. Design the future `starling_cli` and C API around capability queries and
   explicit model/session/result lifetimes.
4. Treat competitor source as an implementation/reference oracle and source of
   validation manifests—not as a new external runtime dependency.
