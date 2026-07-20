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

## Current state (verified 2026-07-18)

- **In-tree engine** (`cpp/`, ggml v0.13.0 + 7 patches, `libstarling_ggml.so`):
  parakeet-tdt Phase 1 complete — loader, CPU mel frontend, 24-layer Conformer
  encoder, TDT greedy + K-step multistep decode. **BYTE-EXACT on
  short/medium/long** (verified via `StarlingGgmlParakeet` ctypes path).
  MOSS Phase 2 pipeline is present (`cpp/moss/`: loader, mel, audio encoder,
  adapter, prompt merge, Qwen3 LLM, tokenizer, and shared-C-API wiring). The
  staged pipeline is byte-exact through merged embeddings; decoder exactness
  remains blocked: CUDA short reaches a layer-0 non-finite and CPU prefill
  differs (max abs 59.1484375), so IDs/text are not yet byte-exact. The
  in-tree parity tests are intentionally an exact gate and expose this until
  the LLM numeric divergence is resolved.
- **External parakeet.cpp analysis** (`docs/ggml-parakeet-perf-analysis.md`)
  showed: encoder ~2.8ms GPU (faster than PyTorch peak), TDT decode is the
  medium/long bottleneck (serial, data-dependent), short at 1.00x parity,
  medium ~1.46x, long ~1.80x. The in-tree engine inherited the architecture;
  its own steady-state baseline is being measured (Phase 3 entry).
- **MOSS PyTorch peak** (`src/starling/moss/`): fused-Triton decode
  4.85ms/tok (206 tok/s), encoder eager ~25-40ms, end-to-end short 248ms /
  medium 618ms / long 1151ms. This is the bar for ggml-moss.
- **Granite ggml study** (comms.md): naive per-op ggml decode is ~4x slower
  than the Triton peak (host dispatch + f32 elementwise traffic); with
  CUDA-graph capture the gap closes to ~parity. Our ReplayGraph + K-step
  multistep design exists precisely to avoid that trap.

## Phases

### Phase 2 — MOSS in-tree (current focus)

Model: `OpenMOSS-Team/MOSS-Transcribe-preview-2B` = Qwen3-omni MoE audio
encoder (32L, d1280, H20, conv2d ÷8 subsample, sinusoidal pos, windowed
non-causal attention via cu_seqlens, LayerNorm+GELU) → gated-MLP adapter
(2048→8192→2048 SiLU) → Qwen3 LLM decoder (28L, d2048, GQA 16Q/8KV, hd128,
SwiGLU 6144, RMSNorm eps 1e-6, QK-norm before RoPE, RoPE θ=1e6, tied
embeddings, vocab 151936, bf16 KV cache).

- **2a — spec + goldens** (running): `docs/ggml-moss-spec.md` (op-by-op
  contract) + staged component goldens `golden/moss_{short,medium,long}_*
  {mel,audio_embeds,inputs_embeds,prefill_logits}.pt` alongside the existing
  ids/text goldens.
- **2b — GGUF converter + loader + mel**: `scripts/convert_moss_gguf.py`
  (HF checkpoint → starling GGUF, HF-native tensor names, F16 weights /
  F32 norms); `cpp/moss/{config,loader,mel}` byte-exact mel vs golden.
  Resolve the n_fft 400-vs-640 question against the golden mel first.
- **2c — audio encoder + adapter**: conv2d stack, sinusoidal pos, 32×
  (LN → windowed attn → LN → GELU MLP), ln_post, proj, adapter.
  Gate: audio_embeds golden.
- **2d — prompt merge + Qwen3 prefill + greedy decode**: embed merge
  (masked_scatter semantics), 4D-mask causal attention, bf16 KV cache,
  QK-norm→RoPE order, tied-embedding lm_head, BPE detok. Gates: prefill
  logits golden, then ids/text goldens (short/medium exact).
- **2e — perf**: per-shape ReplayGraph for encoder (the PyTorch encoder is
  *eager* 25-40ms — a captured ggml encoder can beat it), K-step multistep
  decode with on-device argmax chaining (target ≤4.85ms/tok, the starling
  bar), single-sync-per-K. Then bench vs starling-moss.
- Wiring: `_native.py` MOSS binding, `StarlingGgmlMoss` engine, parity tests
  extended to moss.

### Phase 3 — parakeet perf closure

Baseline the in-tree engine (steady state), then close medium/long:
1. Device-resident decode state between K-step replays (kill per-replay
   H2D/D2H round-trips of LSTM h/c + frame + token).
2. K tuning + termination-check readback minimization.
3. Encoder: verify the in-tree encoder matches the 2.8ms external
   measurement; port any missing wins (depthwise-direct, NORM fusion — both
   already in patches).
Stretch: fully on-device TDT loop (megakernel-style) — evaluate against
exactness risk.

### Phase 4 — self-containment hardening

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
