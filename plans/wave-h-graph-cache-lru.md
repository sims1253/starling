# Wave H — bound the per-shape ReplayGraph caches with real LRU eviction (CORRECTNESS/STABILITY BUG)

## Bug report

Found running `benchmarks/bench_leaderboard.py --models parakeet,moss
--engines starling-ggml --num-samples 50` (post Wave-G fix — the K-step OOB
bug is fixed and confirmed: voxpopuli's 50 MOSS clips completed cleanly,
WER 3.81% matching the known reference exactly). The run then crashed on the
SECOND dataset:

```
CUDA error: out of memory
  in function ggml_backend_cuda_get_tensor_async (ggml-cuda.cu:3152)
starling::ggml::ReplayGraph::readback_async_then_sync
starling::ggml::ReplayGraph::compute
starling::ggml::moss::greedy_generate
starling_ggml_moss_decode -> starling_ggml_transcribe_pcm
```

By the time of the crash, CUDA graph IDs had climbed past 800 (`CUDA Graph id
813 reused`, etc.) — evidence of runaway graph accumulation, not a single bad
allocation.

### Root cause (confirmed by code inspection)

Three per-shape/per-length `ReplayGraph` caches exist, ALL are process-global
`unordered_map`s with **no eviction policy** — every distinct shape/length
seen permanently allocates a new `ReplayGraph` (its own captured CUDA graph +
**private gallocr**, i.e. its own device buffer) that is NEVER freed until
process exit (`register_decode_cache_clearer` only clears at
`shutdown_backend()`, i.e. process teardown, not during normal operation):

1. `cpp/moss/audio_encoder.cpp` — `g_encoder_cache` (`std::unordered_map<
   ShapeKey, ...>`, keyed on the encoder's `(C, tail)` mel shape). Comment at
   line ~324 claims "LRU-bounded... including its bound" — **this claim is
   false**; there is no LRU, no bound, no eviction anywhere in the file.
2. `cpp/moss/llm.cpp` — `g_prefill_cache` (`std::unordered_map<int64_t, ...>`,
   keyed on prompt length `S`). Same problem: unbounded.
3. `cpp/parakeet/encoder.hpp`/`.cpp` — `Encoder::ReplayCache`
   (`std::unordered_map<int, unique_ptr<ReplayEntry>>`, keyed on mel length
   `T`, member `replay_cache_`). This is the pattern the moss caches above
   claim to follow — **it is ALSO unbounded**, no eviction, despite
   `docs/ggml-roadmap.md`/plan text elsewhere implying it has a bound. It is
   scoped to one `Encoder` instance (freed when the model context closes), so
   it leaks only for the lifetime of one loaded model, but that is still
   unbounded across however many distinct audio lengths that model processes
   in one session — exactly the leaderboard's 50-clips-per-dataset x 7
   real-audio-length distribution.
4. `docs/review.md` §5 explicitly (and incorrectly) states "the current model
   graph caches are bounded, so there is no active unbounded-OOM bug" — this
   benchmark run proves that claim false. Correct it as part of this wave.

The MOSS K-step decode cache (`g_moss_kstep_cache` in `cpp/moss/llm.cpp`,
keyed on `K`) is NOT in scope — `K` only takes values in `[1,8]`
(`moss_kstep_K()`), an inherently tiny bounded domain; confirm this in the
report but no fix needed there.

Real audio (unlike the 3 synthetic tiered fixtures) has near-continuous
length variation, so every one of the ~50-per-dataset x 7-dataset x 2-model
leaderboard run produces dozens to hundreds of distinct shapes/lengths per
cache, each pinning its own device buffer forever. This is what exhausted 32GB
VRAM on dataset #2.

## Task

Add genuine bounded LRU eviction to all three caches (parakeet
`Encoder::ReplayCache`, moss `g_encoder_cache`, moss `g_prefill_cache`):

- On a cache miss when the cache is already at capacity, evict the
  least-recently-used entry (its `ReplayGraph`/`GraphInputPool` destructor
  frees the private gallocr + captured graph) before inserting the new one.
- A simple `std::unordered_map` + intrusive doubly-linked list (or a
  `std::list` of keys in LRU order + the map storing list iterators) is
  sufficient — no need for anything fancier. Consider a small shared helper
  (`cpp/runtime/` — mirrors `review.md`'s "shared graph-cache and
  replay-policy helper" idea) IF it doesn't cost extra risk/time; otherwise
  three independent small LRU maps following the same pattern are fine and
  lower-risk. Prefer whichever is less invasive to land correctly.
- **Capacity**: pick a bound generous enough that realistic within-run reuse
  (e.g. repeated similar-length utterances, or the 3 synthetic tiers) still
  hits warm captured graphs, but small enough to cap VRAM. A reasonable
  starting point is 16 entries per cache; make it overridable via an env var
  (e.g. `STARLING_REPLAY_CACHE_SIZE`) so it can be tuned without a rebuild.
  Justify your chosen default with a rough per-entry VRAM estimate (gallocr
  size) if you can get one cheaply (e.g. from existing `STARLING_REPLAY_TIMING`
  instrumentation or a quick log).
- Eviction must be safe mid-run: no in-flight replay may be evicted out from
  under itself (the call pattern is synchronous — build/fetch, then
  set_input+compute — so this should be naturally safe, but confirm no
  reentrancy hazard, e.g. nothing recursively calls back into the same cache
  during a compute).
- Correctness is unaffected by eviction+rebuild: a re-captured graph for a
  previously-evicted shape must produce byte-identical output (same graph
  construction code, same weights) — this follows from the existing capture
  code being deterministic, but confirm with the parity suite.

## Requirements

1. `uv run python -m pytest tests/test_ggml_parity.py -x -q` — 10/10, exit 0
   (CrispASR moss-long external flake: retry isolated / `GGML_MOSS_SERVER=0`).
2. **New regression test / stress check**: exercise enough distinct
   shapes/lengths in one process to prove the cache is actually bounded —
   e.g. a small C++ test or a script that runs the moss encoder+adapter (or
   prefill, or parakeet encoder) across N > capacity distinct lengths and
   asserts device memory usage plateaus (query free/used VRAM via
   `cudaMemGetInfo`-style backend query, or via `nvidia-smi` from the test
   harness) rather than growing linearly with N. This is the test that would
   have caught this bug — make it permanent.
3. No `third_party/ggml/` edits; no git state-changing commands.
4. Correct `docs/review.md` §5's false "bounded" claim (either remove the
   stale claim now that a real bound exists, or update it to describe the new
   bound — whichever is accurate post-fix).
5. Re-verify perf did not regress: bounded LRU adds O(1) bookkeeping per
   cache hit and eviction only on capacity overflow — should be free in the
   steady 3-tier bench, but confirm.

## Verification

```bash
cmake --build build -j
uv run python -m pytest tests/test_ggml_parity.py -x -q
# + the new bounded-cache stress test — run it explicitly, show VRAM-plateau evidence
uv run python benchmarks/bench_all.py --models parakeet,moss \
  --engines starling-ggml --lengths short,medium,long --batches 1 \
  --reps 10 --warmup 2
```

Then, if time allows, actually re-run a slice of the real leaderboard bench
that triggered this (e.g. `bench_leaderboard.py --models moss --engines
starling-ggml --num-samples 50 --datasets voxpopuli,ami` — two datasets is
enough to reproduce the original OOM point) and confirm it completes without
a CUDA error. This is the strongest possible verification since it's the
exact scenario that found the bug.

Report: which caches got LRU, chosen capacity + justification, the stress
test + its result (VRAM plateau evidence), parity result, bench deltas
(expect ~0 on the 3-tier fixtures), and whether the two-dataset leaderboard
slice now completes cleanly.
