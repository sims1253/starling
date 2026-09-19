# Shared experiment records (issue #168)

Make each claimed improvement reproducible and prevent invalid benchmark
comparisons. This directory adds the missing layer ON TOP of the existing
tools — SONAR stays the quality harness, `bench_all.py` stays the fixture
benchmark — and contributes an **experiment record** plus the **comparison
command** that decides whether a candidate may be called a win.

Stdlib only (the runner speaks HTTP through `urllib`/`requests-free` code):
production server installations gain no Python dependency from this code.

## The one-command reproduction

```bash
git submodule update --init --recursive
cmake -B build-exp -DSTARLING_SERVE=ON -DSTARLING_GGML_TESTS=ON \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF
cmake --build build-exp -j --target starling-serve-contract-fixture
python benchmarks/experiments/run_experiment.py demo \
    --binary build-exp/starling-serve-contract-fixture \
    --out-dir build-exp/experiments/demo
```

The demo runs the SAME binary as both arms of a sealed spec — fresh
processes per repeat, interleaved seeded order, cold/warm split — and
compares them under preregistered rules. Identical arms compare
**inconclusive**; that is the point. On a noisy box the point estimate can
look like several percent "improvement", and the comparator still refuses
it unless the CI clears the preregistered bar. That refusal is the product.

## Real experiments

1. Pin the workload (ordered files, sizes, hashes → one digest):

   ```bash
   python benchmarks/experiments/run_experiment.py pin-workload --audio clips/
   # paste the printed block into the spec's "workload"
   ```

2. Write the spec — everything is preregistered BEFORE any arm runs:
   `objective` (the hypothesis), `metric` + `direction` (registry in
   `record.py`; adding a metric requires a comparator test), `arms`
   (baseline/candidate: binary path, model, env overrides), `workload`
   (the pin), `protocol` (repeats, requests, warmup, seeded order,
   timeouts), `acceptance` (min improvement, max CI halfwidth, max
   regression). The spec is hashed; both records embed the hash and the
   comparator rejects pairs whose seals disagree — post-hoc rule edits
   cannot pass unnoticed.

3. Run and compare:

   ```bash
   python benchmarks/experiments/run_experiment.py run --spec spec.json --run-dir runs/exp1
   python benchmarks/experiments/run_experiment.py compare --spec spec.json --run-dir runs/exp1
   ```

## What the comparator rejects (hard errors, all listed)

- different workload manifests (the corpus changed between arms),
- different metric identities — SONAR WER vs quantization-driver WER vs
  raw HTTP wall time are never averaged or compared,
- different normalizers, model/config claims, or hardware/runtime claims
  (cross-machine comparisons need an explicit `hardware_claim` in the spec),
- a spec-seal mismatch, a failed run, zero usable warm samples, missing or
  malformed records.

## Verdicts

- `pass` — the CI clears the preregistered improvement bar entirely.
- `fail` — the CI sits beyond the regression bound.
- `inconclusive` — CI spans the boundary or is too wide for the declared
  power requirement. **Cannot be promoted as a win.**
- `unavailable` — a required run failed or is missing; diagnostics
  accompany the verdict instead of a number.

## Protocol guarantees

- Arms never run concurrently; repeats interleave in a seeded order so
  systematic drift hits both arms equally. Fresh server process per
  (repeat, arm).
- The first request per process is the **cold** sample (model load + graph
  capture): recorded and reported separately, excluded from the gated
  estimate; `warmup_requests` more untimed requests follow before the
  measured window.
- Timeouts kill the whole process group; failures are recorded with their
  diagnostics and flip the run to `status=failed`.
- GPU work follows the SONAR serialization contract: all `STARLING_*` GPU
  lock variables pass through to the server processes, so externally held
  locks are honored.

## Files

- `record.py` — spec/record schemas, validation, workload manifest, seal.
- `runner.py` — fresh-process arm execution, provenance collection.
- `stats.py` — deterministic paired bootstrap (seeded; identical records →
  identical interval).
- `compare.py` — compatibility gates + preregistered verdicts.
- `run_experiment.py` — CLI (`pin-workload` / `run` / `compare` / `demo`).
- `stub_serve.py` — HTTP double for the runner TESTS only (never evidence).
- `test_experiment.py` — the acceptance suite (comparator negatives,
  verdict semantics, determinism, runner mechanics over the stub).

Tests: `python -m unittest discover -s benchmarks/experiments -p 'test_*.py'`
(wired into `.github/workflows/test.yml`).
