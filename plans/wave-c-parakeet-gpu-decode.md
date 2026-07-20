# Wave C — port the GPU K-step TDT decode into the in-tree parakeet engine

## Why

The in-tree engine (`cpp/parakeet/`) is byte-exact on short/medium/long but
6-9x off the PyTorch peak:

| length | in-tree (median) | peak (starling) | external parakeet.cpp (dev) |
|--------|------------------|-----------------|------------------------------|
| short  | 141 ms           | 17 ms           | 16 ms                        |
| medium | 253 ms           | 28 ms           | 38 ms                        |
| long   | 388 ms           | 62 ms           | 108 ms                       |

The gap is the decode: `cpp/parakeet/tdt.cpp` runs the serial greedy loop with
`PredictionNet::step` and `Joint::step_argmax` as **one-shot CPU graphs per
step** (`cpp/parakeet/prediction.cpp`, `cpp/parakeet/joint.cpp`). The encoder
already uses per-shape `ReplayGraph` CUDA-graph capture and is fine.

The external reference at `~/Documents/parakeet.cpp` (branch `dev`) contains
the **proven byte-exact** GPU decode this wave ports:

- `src/tdt_multistep.{cpp,hpp}` — K-step multistep decode: K consecutive
  TDT steps (prediction LSTM step + joint + argmax + duration-advance +
  blank-skip chaining) captured in ONE CUDA graph via `ReplayGraph`; sync once
  per K steps; T-aware K (16 for T<=512, 64 for long); termination check on
  host after each replay from a single readback.
- `src/prediction.cpp`, `src/joint.cpp` — the GPU step formulation.
- `src/tdt.cpp` — the serial fallback + how multistep is gated/dispatched.
- `src/ggml_graph.{cpp,hpp}` — their ReplayGraph (ours,
  `cpp/runtime/backend.{hpp,cpp}`, mirrors it; API differences are minor).

## Task

Port the GPU K-step multistep TDT decode into `cpp/parakeet/`, adapted to the
Starling runtime (`cpp/runtime/backend.hpp` ReplayGraph +
`register_decode_cache_clearer` in `cpp/runtime/graph.hpp`).

Requirements:

1. **Byte-exactness is the gate.** `uv run python -m pytest
   tests/test_ggml_parity.py -x -q` must stay green (9/9, exit 0 — the exit
   code matters, teardown crashes count as failures). The emitted id stream
   must keep including blanks (see the header comment in `cpp/parakeet/tdt.cpp`
   — our goldens differ from parakeet.cpp's `hyp` in exactly this way; the
   multistep port must preserve it).
2. **CPU fallback stays.** The current serial CPU path remains the
   byte-identical reference and the non-GPU path (`Backend::is_gpu()` gates
   the multistep path, mirroring how the external repo gates it). Keep an env
   override to force the serial path (`STARLING_GGML_TDT_SERIAL=1`).
3. **Decode graph cache** persists across utterances on the parakeet context
   (like the encoder's `ReplayCache`), keyed the way the external repo keys it,
   and registers a cache clearer via `register_decode_cache_clearer` so
   process teardown stays clean (exit 0, no SIGABRT).
4. **Weights on device once.** The prediction-net LSTM weights and joint
   weights must be realized to the GPU backend once (loader weights are
   already device-resident via `clone_weight`); no per-step weight uploads.
5. **T-aware K** (16 short/medium, 64 long) as in the external repo's final
   state, including the multistep termination fix (external commits `a4ca1fa`,
   `ade52bf`, `23f0958` — read them).
6. **Single sync per replay** — use the ReplayGraph async-readback path
   (`readback_async_then_sync`); do not add extra `ggml_backend_sched`/sync
   calls per step.
7. **No edits under `third_party/ggml/`** unless a new numbered patch in
   `third_party/ggml-patches/` + `scripts/apply_ggml_patches.sh` is the only
   way (not expected for this wave — the fattn/uid/NORM-fusion patches are
   already applied in-tree).
8. **Do not run `git add`/`git commit`/any git state-changing command.** Leave
   the tree dirty; the user commits.

## Verification (run all, in this order)

```bash
cmake --build build -j
uv run python -m pytest tests/test_ggml_parity.py -x -q   # 9/9, exit 0
uv run python benchmarks/bench_all.py --models parakeet \
  --engines starling-ggml --lengths short,medium,long --batches 1 \
  --reps 20 --warmup 3
```

Success bar: short <= 25 ms, medium <= 45 ms, long <= 120 ms (the external
repo's proven numbers + slack), byte-exact everywhere (WER 0.00 in the bench
table), and the parity suite exits 0.

If a timed run needs the GPU exclusively, respect the `.gpu.lock` protocol
(`src/starling/parakeet/gpu_lock.py` documents it; benches take the lock
themselves).

## Notes / gotchas from the codebase

- `ReplayGraph` invariants are documented at the top of
  `cpp/runtime/backend.hpp` — read them before wiring (private gallocr per
  graph, inputs pushed after alloc, stable uid, single sync).
- ggml-cuda only captures CUDA graphs for **persistent** cgraphs with stable
  nonzero uid (patch 0008): one-shot graphs never capture. The multistep graph
  MUST go through a persistent `ReplayGraph` to get capture.
- The encoder output layout is row-major `[T, H]` per `cpp/parakeet/tdt.cpp`'s
  header — the external repo uses the same layout; verify rather than assume.
- `cpp/parakeet/capi_parakeet.cpp` holds the per-context persistent objects
  (encoder, prediction, joint); the decode cache belongs there too.
- Read `docs/ggml-parakeet-perf-analysis.md` for the full optimization history
  and where the remaining time goes (mel ~1.5-10 ms, encoder ~2.8 ms GPU).
