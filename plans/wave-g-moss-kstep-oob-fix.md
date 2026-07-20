# Wave G — fix MOSS K-step decode out-of-bounds KV-cache access (CORRECTNESS BUG)

## Bug report

Found by running `benchmarks/bench_leaderboard.py --models moss --engines
starling-ggml --num-samples 50` (real diverse-length audio, not the synthetic
short/medium/long tiered fixtures). Crash:

```
CUDA error in ggml_backend_cuda_get_tensor_async (ggml-cuda.cu:3152)
  cudaMemcpyAsync(... cudaMemcpyDeviceToHost ...)
starling::ggml::ReplayGraph::readback_async_then_sync
starling::ggml::ReplayGraph::compute
starling::ggml::moss::greedy_generate
starling_ggml_moss_decode -> starling_ggml_transcribe_pcm
```

This is a **sticky CUDA error** — the driver reports it on a later, unrelated
call once an illegal memory access has occurred. The actual fault happens
earlier, inside the K-step decode graph.

### Root cause (confirmed by code inspection, `cpp/moss/llm.cpp`)

`run_kstep` (lines 976-1007) computes a block of K decode steps in ONE
captured graph (`MossKStepGraph`, built in `get_or_build_kstep`,
lines 887-971) before truncating to the actual token budget on the host:

```cpp
for (int j = 0; j < K; ++j) {
    ...
    kg->host_pos[(size_t)j] = (int32_t)(past + j);   // <-- NOT clamped to max_cache-1
    ...
}
...compute_with_captures(out)...              // <-- the graph ALREADY ran all K steps
for (int j = 0; j < K; ++j) {
    if ((int)ids.size() >= max_new_tokens) break;    // <-- truncation happens AFTER
    ...
}
```

Inside the graph (lines 944-957), position `pos_t[j]` (= `past+j`, unclamped)
is used for TWO device-side accesses sized `[.., max_cache, ..]`:
1. `ggml_get_rows(c, dc->rope_cos, pos_t[j])` / `dc->rope_sin` — **OOB read**
   if `pos_t[j] >= max_cache`.
2. `append_layer_new(..., kv_mode=2, pos_t[j])` — writes the layer's k/v into
   the device KV cache via `SET_ROWS` at slot `pos_t[j]` — **OOB write** into
   `dc->k[li]`/`dc->v[li]` (each `[D, max_cache, KV]` bf16) if
   `pos_t[j] >= max_cache`.

`greedy_generate`'s outer bound check (line 1035:
`i.n_tokens + op.max_new_tokens > op.max_cache_len`) only bounds the *last
needed* token position (`n_tokens + max_new_tokens - 1 <= max_cache - 1`). It
does NOT bound the *wasted tail steps* of the last K-step block: if the
remaining token budget when a block starts is less than K, the graph still
computes all K steps (including ones beyond `max_new_tokens`), and those
extra steps' positions can run up to `K-1` slots past `max_cache-1`. Given
`max_cache=2048` and `K` up to 8 (`moss_kstep_K()`), any utterance whose
`n_tokens + max_new_tokens` lands within `K-1` of `max_cache` reliably
triggers an out-of-bounds device write. Long real-world audio (AMI,
earnings22 in the Open ASR Leaderboard corpus) produces exactly this prompt
length range; the synthetic short/medium/long fixtures (max prompt 977
tokens) never got close.

**This is a heap-buffer-overflow class bug.** It sometimes crashes (as
observed) and could otherwise silently corrupt device memory without
crashing (worse). Fix it correctly — do not just narrow the crash window.

## Task — fix (pick the option that stays simplest and provably correct)

Preferred: **clamp the wasted tail steps to be no-ops that read/write a safe
scratch slot, not real cache slots.**

- Compute, per block, `remaining = max_new_tokens - (int)ids.size()` (or
  equivalently pass the remaining budget into `run_kstep`) BEFORE building
  the position inputs.
- For steps `j >= remaining` in the block (the "wasted" tail), do NOT let
  `pos_t[j]` reach `max_cache` or beyond. Two viable approaches — pick one and
  justify the choice in the report:
  1. **Clamp position to `max_cache - 1`** for wasted steps (repeatedly
     re-reading/re-writing the same last valid slot is harmless: the host
     loop already discards these steps' output tokens via the existing
     `if (ids.size() >= max_new_tokens) break`). Simplest, smallest diff.
  2. **Reduce K dynamically** for the tail block (build/fetch a smaller-K
     graph, e.g. `K' = min(K, remaining)`, from the existing per-K graph
     cache — `get_or_build_kstep` already keys on K, so this reuses the
     existing multi-K infrastructure if the small-K graphs are already built
     elsewhere, or build them on demand). More graphs cached, more capture
     variety, but no wasted compute.
  Prefer (1) unless it conflicts with byte-exactness (verify: does reading
  the same slot repeatedly change any *committed* KV entry? It must not —
  the clamped steps still WRITE to slot `max_cache-1` every wasted step,
  which would clobber that real slot's k/v with garbage. **So plan (1) is
  only safe if the wasted steps' writes are also suppressed or redirected to
  a scratch slot, not the real last valid slot.** Consider a dedicated
  scratch row (`max_cache` sized cache is `[D, max_cache, KV]` — either
  allocate `max_cache+1` and dedicate row `max_cache` as scratch that never
  becomes a real position, or mask the SET_ROWS write itself for wasted
  steps). Get this exactly right — silent KV corruption of a REAL slot is
  worse than a crash. Reason through it carefully and pick whichever design
  is provably safe; document the choice.
- The mask/position for RoPE lookup (`dc->rope_cos`/`rope_sin`, sized
  `[D, max_cache]`) needs the same treatment — clamp to a valid row, and
  since the wasted step's OUTPUT is discarded anyway, correctness only
  requires the read stays in-bounds (no exactness constraint on wasted
  steps' logits).

Whichever design you pick, the load-bearing invariant is: **no step, wasted
or not, ever generates a device index (`SET_ROWS` target or `get_rows`
source) `>= max_cache`.**

### Also check while you're in there

- `moss_kstep_K()` clamps K to 8 given `kGraphSize=32768`. Confirm the fix
  doesn't need a second graph per (K, is-tail-block) — prefer the single
  scratch-row design if it avoids that.
- Verify the exact-width/single-step decode path (`forward_decode_new`, the
  `!use_kstep` branch) does NOT have the same class of bug — its position is
  `state.length` incremented one at a time by the generation loop itself
  (`for n in 1..max_new_tokens`), so it should be naturally bounded by the
  existing `i.n_tokens + op.max_new_tokens <= op.max_cache_len` check with no
  "wasted lookahead." Confirm this in the report; do not change it unless you
  find a real problem.
- Quickly sanity-check parakeet's Wave-D device-resident decode state
  (`cpp/parakeet/tdt_multistep.cpp`, `DecodeDevCache`) for an analogous
  "wasted step past a fixed-size buffer" pattern. Parakeet's loop terminates
  on `t < T` (mel frame count) rather than a separate max-cache concept, so
  it is probably not exposed to this bug class, but confirm by reading the
  code rather than assuming — report a one-paragraph finding either way.

## Required regression test (this bug MUST get a permanent test)

The existing parity fixtures cannot catch this class of bug (prompt lengths
too short). Add a **new, minimal, in-tree regression test** that exercises
the exact boundary condition:

- A synthetic case where `i.n_tokens + max_new_tokens` lands within `K-1`
  tokens of `max_cache` (e.g. construct `inputs_embeds` with
  `n_tokens = max_cache - max_new_tokens` exactly, or as close as the model's
  real constraints allow), run `greedy_generate` via the C API or a direct
  C++ unit test (follow the pattern of the existing
  `moss_encoder_test`/`moss_mel_test`/`moss_llm_test` targets in `build/`),
  and assert it **completes without a CUDA error** and produces
  `max_new_tokens` (or fewer, on EOS) tokens with no out-of-range values.
- If a lighter-weight repro is easier (e.g. a small unit test at the
  `run_kstep`/`MossKStepGraph` level with a fake short model, or a targeted
  test that just checks no generated position index ever reaches
  `max_cache`), that is acceptable too — the bar is "this bug class cannot
  regress silently," not "reproduce the exact leaderboard clip."
- Wire it into whatever runs `tests/test_ggml_parity.py` today, or as a new
  `build/`-registered C++ test target (`CMakeLists.txt` already has a
  pattern for `moss_llm_test` etc. — follow it) that CI/the verification
  command below also runs.

## Requirements

1. **This is a correctness fix — treat the exactness contract as absolute.**
   `uv run python -m pytest tests/test_ggml_parity.py -x -q` must stay 9/9
   exit 0 (CrispASR external moss-long flake: retry in isolation /
   `GGML_MOSS_SERVER=0`) AND the new regression test must pass.
2. No `third_party/ggml/` edits; no git state-changing commands.
3. Do not regress Wave A-F's perf numbers (moss short/medium/long e2e and
   ms/token) — re-run the `bench_all.py` moss cell after the fix and report
   the delta (expect ~0, this is a boundary-only fix).

## Verification

```bash
cmake --build build -j
uv run python -m pytest tests/test_ggml_parity.py -x -q
# + whatever new regression test you add — run it explicitly and show output
uv run python benchmarks/bench_all.py --models moss \
  --engines starling-ggml --lengths short,medium,long --batches 1 \
  --reps 10 --warmup 2
```

Then, as the FINAL check (do this yourself if time allows, otherwise report
that it still needs to run): a real stress case, e.g. build a long synthetic
`inputs_embeds`/PCM input whose audio-token count lands near the
`max_cache - max_new_tokens` boundary and confirm `starling_ggml_transcribe_pcm`
completes cleanly (no CUDA error, exit 0).

Report: exact fix (which of options 1/2 above, and the scratch-slot design if
applicable), regression test added + its result, parakeet sanity-check
finding, parity result, bench deltas.
