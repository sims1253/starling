# Benchmarks

Install benchmark and corpus-generation dependencies before running these tools:

```sh
uv sync --locked --extra bench
uv run --extra bench python benchmarks/bench_all.py --help
```

Use `--extra dev --extra bench` when running tests that also download corpora or
score transcripts. The `dev` extra alone covers the CPU CI suite. Model execution
still requires the [model files](../docs/models.md) and
[Python setup](../docs/python-serving.md).
`librosa` remains a runtime dependency because the Higgs model imports it.

Maintained entry points:

- `bench_all.py`: unified latency/RTFx regression grid and [benchmark table](../docs/benchmarks.md) source.
- `bench_leaderboard.py`: real-corpus Open ASR Leaderboard WER/RTFx evaluation.
- `s1/bench_normalize.py`: s1-mini normalization suite: latency/throughput
  per engine, byte-exact parity vs stock, curated quality cases, and the
  16-cell control matrix (styling × structure × context). [Benchmark](../docs/benchmarks.md) `BENCH:S1`
  table source.
- `bench_ablate.py`: optimization-flag ablation harness.
- `wer.py` / `wer_leaderboard.py`: fixture drift gate and real-corpus scoring.
- `run_leaderboard_all.sh`: orchestrates the maintained per-model leaderboard runs.

The remaining `bench_*.py` files are exploratory model/kernel microbenchmarks.
They are useful for reproducing individual optimization investigations, but are
not release gates and may require model-specific environments or hardware.

External installs are configurable: set `ASR_BENCH_ROOT` for CrispASR and
`STARLING_FIXTURES_DIR` when fixtures live outside this checkout. All GPU-timed
maintained entry points must hold `starling.parakeet.gpu_lock.with_gpu_lock`.
