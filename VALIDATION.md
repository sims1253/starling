# GPU validation runbook — program/gpu-validation

Combined engine branch: S01 (on-device MOSS KV clearing) + S02 (TDT graph
budget) + S03 (EOS/budget termination), merged for hardware validation.
Branch `program/gpu-validation` on origin. No merges/pushes needed from you —
run, collect, and hand results back; the coordinator integrates them.

## One-time setup (RTX 5090 PC)

```bash
git fetch origin && git checkout program/gpu-validation
git submodule update --init third_party/ggml
cmake -B build-cuda -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON
CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build build-cuda --target \
  starling-serve device_cache_clear_test tdt_graph_budget_test \
  greedy_termination_test stream_session_test
```

Models (see docs/models.md for exact artifacts): MOSS transcribe preview 2B
GGUF and Parakeet TDT 0.6B v3 GGUF. Record each file's name + SHA256.

## A. Correctness gates (run first, all must pass)

```bash
./build-cuda/device_cache_clear_test     # now exercises the CUDA memset path
./build-cuda/greedy_termination_test
./build-cuda/tdt_graph_budget_test       # graphs on the CUDA device
./build-cuda/stream_session_test
```

Paste full stdout of each.

## B. S01 timing evidence (the acceptance items the notebook cannot produce)

Question: does backend-side KV clearing measurably remove the host upload,
without changing outputs?

1. Start the server twice — once from this branch, once from plain `master`
   (two checkouts; same model, same flags):
   `./build-cuda/starling-serve --model moss --gguf <moss.gguf> --port 8181`
   plus the structured timing trace flags (see docs/native-serving.md /
   the #180 trace options).
2. Drive the same fixed WAV set (≥3 durations: short/medium/long; reuse
   benchmarks/ fixtures if present, else any fixed recordings — record their
   SHA256s) with ≥5 repeats each, cold start + warm.
3. Capture per-run: the structured trace (enqueue / device / readback /
   total), and `nvidia-smi` sampled during load.

Success shape: identical transcripts master-vs-branch (paste any diff!), and
the branch's per-request trace shows the KV-clear stage's host bytes ≈ 0 with
request latency not worse. A no-gain or negative result is a valid outcome —
report it as-is.

## C. S02 budget evidence

1. Run Parakeet with varied-length audio (many distinct durations — the
   point is many distinct (T,K) shapes) under
   `STARLING_TDT_GRAPH_BUDGET_MB=64 / 128 / 512`.
2. Capture: VRAM plateau (nvidia-smi during steady state), rebuild/thrash
   behavior if logged, and that transcripts are identical across budgets.

## D. S03 real-model A/B (MOSS)

Fixed audio set, master vs branch, identical transcripts expected (the
predicate only changes *when* generation stops for EOS-bearing outputs;
report any text diff verbatim). If you can craft/note an input whose
transcription previously over-generated after an early EOS, call it out.

## Results template

Create `docs/program/evidence/gpu-validation-5090-<YYYY-MM-DD>.md`:

```
## Environment
GPU/driver/CUDA versions, model files + SHA256, audio set + SHA256, flags
## A. Correctness
<stdout pastes>
## B. S01 timing
<tables from traces + nvidia-smi>
## C. S02 budget
<VRAM plateaus, transcript equality>
## D. S03 A/B
<transcript diffs (expect none), notes>
```

Anything ambiguous: paste raw logs rather than summarizing. Thank you —
this closes the recorded GPU validation gaps for S01/S02/S03.
