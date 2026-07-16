# Benchmarks

Maintained entry points:

- `bench_all.py` — unified latency/RTFx regression grid and README table source.
- `bench_leaderboard.py` — real-corpus Open ASR Leaderboard WER/RTFx evaluation.
- `bench_ablate.py` — optimization-flag ablation harness.
- `wer.py` / `wer_leaderboard.py` — fixture drift gate and real-corpus scoring.
- `run_leaderboard_all.sh` — orchestrates the maintained per-model leaderboard runs.

The remaining `bench_*.py` files are exploratory model/kernel microbenchmarks.
They are useful for reproducing individual optimization investigations, but are
not release gates and may require model-specific environments or hardware.

External installs are configurable: set `ASR_BENCH_ROOT` for CrispASR and
`STARLING_FIXTURES_DIR` when fixtures live outside this checkout. All GPU-timed
maintained entry points must hold `starling.parakeet.gpu_lock.with_gpu_lock`.
