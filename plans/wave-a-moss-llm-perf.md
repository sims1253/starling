# Wave A — MOSS LLM decode/prefill rearchitecture (Phase 2e, decode track)

## Why

`cpp/moss/llm.cpp` is a correctness-first implementation and is catastrophically
slow by design:

- **One one-shot graph per layer per token** (28 graph builds + allocs + H2D +
  D2H per decode step), via `run_graph`.
- **Host-resident KV cache** (`std::vector<ggml_bf16_t>` per layer), fully
  re-uploaded to the device every layer every step, and re-materialized on host
  (`append_kv` copies the whole cache) after every step.
- **Per-head attention loop** (16 iterations of mul_mat + concat per layer).
- **Host argmax over 151936 logits per token**; host RoPE table + host mask
  rebuilt per layer per step; `tobf`/f32 conversions of the full hidden state
  between layers; a finiteness scan over every layer output in the hot loop.

The PyTorch peak (`src/starling/moss/`) does 4.85 ms/token (206 tok/s),
end-to-end short 248 ms / medium 618 ms / long 1151 ms. That is the bar.

The correctness result to preserve: **token-exact greedy ids/text vs the eager
reference on short/medium** and transcript-agreement on long (the reference
itself is bit-unstable on long; a known token-21 near-tie exists on
medium/long — see `docs/ggml-moss-goldens.md` and `tests/test_ggml_parity.py`).

## Design (decided — follow this shape)

### A1. Whole-model graphs

Replace the per-layer `run_graph` loop with:
- **one prefill graph**: embeds -> all 28 layers -> final RMSNorm -> lm_head
  (last token only), built once per utterance (prefill length varies; one-shot
  is fine for now — it runs once).
- **one decode-step graph**: same stack at S=1, captured as a persistent
  `ReplayGraph` (see `cpp/runtime/backend.hpp` — read its invariant list
  before wiring; note patch 0008: only persistent ReplayGraphs get CUDA-graph
  capture, one-shot graphs never do).

The bf16 op-order/dtype discipline in the current `layer()` function (`bf`/`ff`
casts, f32 elementwise, RMSNorm in f32, rotate-half RoPE in f32, softmax via
`ggml_soft_max_ext` with f32 mask) is the **numerics contract** — keep the
math identical while restructuring. `docs/ggml-moss-spec.md` is the op-by-op
authority.

### A2. Device-resident KV cache

Per layer, fixed-capacity zero-initialized device buffers `[D, max_cache, KV]`
bf16 (max_cache = 2048 as today). The step's k/v are written in-graph into the
current slot via `ggml_view` + `ggml_cpy` (the llama.cpp static-cache pattern).
No host KV, no `append_kv`, no per-step cache uploads. The cache tensors live
outside the graph contexts (allocated once on the backend buffer, referenced
as leaves the way loader weights are — see `clone_weight`'s ->data-set
convention).

Attention reads a **view of exactly the first `past+1` slots** where possible.
For the captured decode graph the KV length is baked, so:

- First try: attention over the **full capacity** with an additive f32 mask
  (0 for valid slots, -3.3895313892515355e38 beyond — the exact constant used
  today), zero-initialized cache so masked slots are finite. Gate on the parity
  tests: if short/medium ids stay exact, this is the design (simplest, one
  captured graph serves the whole decode).
- If padded softmax flips a token: fall back to **bucketed capture** (KV length
  rounded up to a bucket, e.g. multiples of 128, one ReplayGraph per bucket,
  mask covers the pad) and re-gate. Document which design survived in the
  final report.

### A3. Batched-head attention

Replace the 16-iteration per-head loop with 3D batched matmuls (GQA: view K/V
per kv-head group, or `ggml_mul_mat` broadcast over the head dim — the standard
llama.cpp formulation). Math per head is unchanged; gate on goldens.

### A4. K-step multistep decode

After A1-A3 are token-exact: chain K decode steps (default K=8, tune later) in
one captured ReplayGraph — in-graph `ggml_argmax` over the lm_head logits,
`ggml_get_rows` on the tied embedding for the next step's input, position/mask
advance in-graph — one readback of K token ids per replay, host checks EOS and
stops. This mirrors the proven K-step designs in the external parakeet.cpp
(`~/Documents/parakeet.cpp/src/tdt_multistep.cpp`) and starling's
`decode_mega.py`. Register the decode cache with
`register_decode_cache_clearer` (teardown must stay exit-0 clean).

### A5. Hot-loop hygiene

- RoPE cos/sin: precompute for all 2048 positions ONCE as a device tensor;
  select the step's row in-graph (`ggml_get_rows` on a position index input).
  Keep the exact f32 `std::pow`-based table math at build time.
- Kill the per-step finiteness scan and the `STARLING_MOSS_DUMP_LAYERS` /
  L0-stage machinery from the hot path — keep them working behind their env
  flags on a slow/debug path (the staged-probe workflow must survive; it is
  the divergence-localization tool).
- No `tobf` host round-trip between layers (hidden state stays on device).

## Requirements

1. **Exactness gate:** `uv run python -m pytest tests/test_ggml_parity.py -x
   -q` — all 9 pass, exit 0, after EVERY milestone (A1, A2/A3, A4). Never
   proceed past a milestone with a red gate.
2. **CPU backend keeps working** (fallback path; the one-shot non-captured
   code path must still run on CPU — `Backend::is_gpu()` gates capture).
3. No edits under `third_party/ggml/` (new ggml kernels/fusions are NOT in
   scope for this wave; if you hit a missing-op wall, stop and report instead
   of patching).
4. **No git state-changing commands. Ever.** The user commits.
5. Keep the C API surface (`cpp/moss/capi_moss.cpp`, `starling_ggml.h`)
   unchanged for callers; persistent state (graphs, KV, scratch) lives on the
   moss context like parakeet's encoder/decode caches do.

## Verification

```bash
cmake --build build -j
uv run python -m pytest tests/test_ggml_parity.py -x -q   # 9/9, exit 0
uv run python benchmarks/bench_all.py --models moss \
  --engines starling,starling-ggml --lengths short,medium,long --batches 1 \
  --reps 10 --warmup 2
```

Success bar for this wave: decode steady-state **<= 10 ms/token** (stretch:
4.85 ms/token = peak parity), end-to-end short **<= 400 ms**, and exact gates
green. (The encoder is NOT this wave — it stays eager/one-shot; a follow-up
wave captures it. Don't spend time there beyond keeping it working.)

Report at the end: files changed, which A2 design survived (full-capacity mask
vs bucketed), parity result, bench table, measured ms/token.
