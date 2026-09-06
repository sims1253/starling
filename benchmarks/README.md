# Benchmarks

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

For a backend without NVIDIA discovery, give the runner and every nested lock
consumer the same stable physical device key:

```bash
uv run starling-gpu-run --session native-bench --uuid vulkan:device-serial -- python bench.py
```

Inside `bench.py`, pass the same key to the lock adapter:

```python
from starling.parakeet.gpu_lock import with_gpu_lock

with with_gpu_lock(session="native-bench", model="parakeet", uuid="vulkan:device-serial"):
    # Run inference on the device identified by this key.
    ...
```

The key identifies the physical device for locking; it does not select a GPU.
Replace the example key with a stable identity shared by all callers of that
device. If the benchmark selects another device, use its key instead. An
omitted key still requires NVIDIA discovery, even under an explicit-key runner.
The lock requires POSIX flock; on Windows, serialize GPU work externally and
set `STARLING_GPU_LOCK_DISABLE=1`.
