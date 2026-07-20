# Starling ggml engine — architecture & roadmap

Status: living document, owned by the architect. Updated as phases land.

## North star

**Self-contained, portable, hardware-ceiling ASR.** Starling ships its own
ggml-based inference engine (in-tree C++, `cpp/`) that runs parakeet-tdt and
MOSS-Transcribe byte-exact (or token-exact where the reference itself is
bit-unstable) vs the `transformers` reference, as fast as the hardware allows,
on any backend ggml supports (CUDA primary, CPU fallback verified, Metal/
Vulkan/HIP/SYCL by build flag). Endgame: **one packaged executable** — no
Python, no PyTorch, no external engine binaries, no system deps beyond the
GPU driver.

## Hard decisions (made)

1. **Self-containment is non-negotiable.** The external engines
   (`benchmarks/engines.py` `GgmlParakeet` shelling to mudler's parakeet.cpp,
   `GgmlMoss` shelling to CrispASR) were a mistake. They will be **removed**
   once the in-tree engine covers both models (Phase 4). No external repo,
   binary, or GGUF artifact is required to build, test, or run Starling.
2. **Starling-owned GGUF conversions.** We do not depend on third-party GGUFs
   (mudler/cstr). `scripts/convert_*_gguf.py` produce our own GGUFs from the
   HF checkpoints with Starling-owned tensor-name maps and config KVs. The C++
   loaders target *our* format from day one. (Bootstrap exception: the
   in-tree parakeet loader currently reads the mudler-layout GGUF; a
   converter reproducing that layout from the HF checkpoint lands in Phase 4.)
3. **Exactness contract.** Byte-exact transcript text vs the golden (captured
   from the eager `transformers` reference path) on short/medium/long, for
   both models, on CUDA *and* CPU backends. Where the reference itself is
   run-to-run bit-unstable (cuBLAS bf16 nondeterminism over long greedy
   decodes — documented in `tests/test_moss.py`), the contract is
   **token-exact modulo the reference's own instability set**, i.e. token-ids
   equal on short/medium and transcript-agreement on long. Staged component
   goldens (mel → encoder → embeds → prefill logits → ids) localize any
   divergence to a stage.
4. **Perf rule.** Never trade exactness for speed below the contract. Above
   the contract, anything goes: patch ggml (carry patches in
   `third_party/ggml-patches/`), fuse kernels, capture CUDA graphs, keep
   decode state device-resident. The PyTorch+Triton engine remains the
   NVIDIA peak reference point; the ggml engine's job is to *match or beat*
   it while being portable.
5. **Numerics mirroring.** Byte-exactness comes from replicating the HF
   reference's op order and dtype flow *exactly*: f32-accumulated norms with
   the weight-mul in the reference's dtype/order, erf-vs-tanh GELU pinned to
   the reference, ATen-order RoPE (no fused approximation), softmax upcast
   semantics, conv/sub sampling in the reference's precision. Every staged
   golden names its dtype. When in doubt, the reference code wins.

## Current state (verified 2026-07-21)

- **In-tree engine** (`cpp/`, ggml v0.13.0 + patch series,
  `libstarling_ggml.so`) supports Parakeet-TDT and MOSS through one C API.
  Parakeet has shape-cached encoder graphs, fused relative-position attention,
  K-step TDT decode, on-device argmax, and device-resident state for K<=16.
  It is exact at transcript and non-blank content-token level on the three
  canonical fixtures. Current synthetic latency is 14 / 30 / 86 ms versus
  16 / 24 / 58 ms for the PyTorch peak (short / medium / long).
- **MOSS Phase 2 correctness and performance are complete enough to ship**:
  exact canonical greedy token IDs/text, whole-model prefill/decode graphs,
  device-resident KV, K-step decode, captured audio encoder, parallel mel, and
  bounded replay caches. Current synthetic latency is 214 / 535 / 1180 ms
  versus 166 / 397 / 1499 ms for the PyTorch peak. Decode is near its 4.85
  ms/token PyTorch target; remaining short/medium cost is frontend/encoder and
  prefill rather than the old per-layer host graph loop.
- `docs/ggml-parakeet-perf-analysis.md` includes useful history but mixes
  external `parakeet.cpp` commits with in-tree state. Treat current code and the
  maintained README tables as authoritative.
- **Granite ggml study** (comms.md): naive per-op ggml decode is ~4x slower
  than the Triton peak (host dispatch + f32 elementwise traffic); with
  CUDA-graph capture the gap closes to ~parity. Our ReplayGraph + K-step
  multistep design exists precisely to avoid that trap.

## Phases

### Phase 2 — MOSS in-tree (landed; optimize incrementally)

Model: `OpenMOSS-Team/MOSS-Transcribe-preview-2B` = Qwen3-omni dense audio
encoder (32L, d1280, H20, conv2d ÷8 subsample, sinusoidal pos, windowed
non-causal attention, LayerNorm+GELU) → gated adapter → Qwen3 decoder (28L,
d2048, GQA 16Q/8KV, bf16 KV cache). Despite the upstream class name, the ASR
audio path has no routed experts; see `docs/ggml-moss-spec.md`.

Landed: Starling-owned GGUF conversion/loading, staged goldens, exact mel/audio
encoder/adapter/prompt merge, exact greedy IDs/text, whole-model captured
prefill/decode, device-resident KV, K-step decode, captured encoder, parallel
mel, OOB tail protection, and bounded replay caches. Future MOSS work should be
profile-led. The most plausible safe target is frontend/encoder or prefill
latency on short/medium audio; decode is already near the PyTorch per-token
floor and long end-to-end is already faster than the PyTorch path.

### Phase 3 — parakeet perf closure (mostly landed)

Landed: K-step decode, on-device argmax, T-aware K tuning, shape-cached encoder,
attention/norm/depthwise fusions, and device-resident state for K<=16. The
remaining concrete gap is long-audio K>16 state writeback: it falls back to
host state round-trips after a ggml CUDA-graph topology defect. The next safe
experiment is to isolate that writeback in a tiny replay regression, fix or
work around the topology instability, then enable device-resident state for
K=96 behind the existing exact content-token/text gates. Do not re-optimize the
encoder without a profile showing a new bottleneck.

### Phase 4 — self-containment hardening and next family

The recommended next family is **Parakeet Unified RNN-T**: it reuses the
FastConformer frontend/encoder and prediction/joint machinery already in
`cpp/parakeet`, while Starling already owns Python goldens. `transcribe.cpp`
commit `a951f0af73b5d6f6153729039ba32e3017dc65cf` provides primary-source
converter/reference-validation research for this exact checkpoint. Keep the
implementation and GGUF format Starling-owned.

After Unified, Qwen3-ASR is the strongest LLM-family candidate because its
Qwen3 decoder can reuse MOSS's device-resident decode runtime; its Whisper-style
encoder is new work. Cohere has excellent external reference material but a
new seq2seq cross-attention decoder, so it is a larger first port. Granite/NAR,
ARK, Higgs, and Audex require more family-specific encoder/decoder work or have
license/environment constraints.


- `scripts/convert_parakeet_gguf.py` (HF/.nemo → the loader's GGUF layout).
- Delete `GgmlParakeet`/`GgmlMoss` external engines + their docs; rewrite
  `docs/ggml-engine.md` for the in-tree engine only.
- `tests/test_ggml_parity.py` → in-tree engines only.
- Vendor check: no path outside this repo + HF checkpoints required.

### Phase 5 — single executable

- `starling_cli` C++ binary (static libstarling_ggml + ggml backends; model
  GGUF as a sibling file or embedded via linker blob). CUDA builds dynamically
  load only the NVIDIA driver (unavoidable and acceptable); CPU/Metal/Vulkan
  fully static.
- Server mode (OpenAI-compatible `/v1/audio/transcriptions`) behind a flag,
  replacing the Python server for the two ggml models.
- Optional Rust/Go thin frontend later — only if packaging UX demands it;
  the engine stays C++ (FFI overhead is real at 16ms/utterance scale).

## Verification contract (every phase)

1. Staged component goldens pass (stage-localized).
2. End-to-end byte/token-exact per §decision-3, CUDA **and** CPU backends.
3. `tests/test_ggml_parity.py` green, exit 0 (no teardown crashes).
4. Pure-Python install unaffected (C++ optional, discovered at runtime).
5. Benches under the GPU lock (`src/starling/parakeet/gpu_lock.py`),
   reps ≥ 20, median ± stdev reported.
