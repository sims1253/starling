# Running Qwen3-ASR

Run these commands from the repository root. The required
`transformers>=5.14.1` includes Qwen3-ASR; no bootstrap script is needed.

```bash
uv sync

# Capture the eager reference (gitignored under golden/qwen3/).
uv run python -m starling.qwen3.golden

# Compare against the reference, then run an end-to-end smoke test.
uv run python -m pytest tests/test_qwen3_pipeline.py -q
uv run python -m starling.qwen3.pipeline

# Benchmarks.
uv run python scripts/bench_qwen3_rtf.py
uv run python scripts/bench_qwen3_batched.py
uv run python scripts/bench_qwen3_crispasr.py
```

Use `uv run python -m pytest` so pytest uses the project interpreter.
Give each worktree its own `.venv`; sharing an editable installation can
resolve imports to another checkout.

## Existing environments

If you previously ran `scripts/setup_qwen3_tf.py`, remove its
`zzz_starling_qwen3_tf_bootstrap.pth` file from your environment's
`site-packages` directory, or recreate the environment. That old startup
hook imports the deleted bootstrap module and can pin imports to the wrong
checkout when environments are shared.
