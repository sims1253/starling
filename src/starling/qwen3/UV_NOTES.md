# Running the Qwen3-ASR megakernel

All commands run via **`uv`** (the project's environment manager). From the
worktree root (`starling-qwen3/`):

```bash
# 1. One-time: ensure the shared transformers exposes qwen3_asr.
#    The pinned transformers (git main) ships qwen3_asr source but, at this
#    commit, find_packages drops 4 model dirs from the wheel and they aren't
#    registered in the auto-mappings. Any `uv sync`/reinstall can wipe them.
#    This installs a .pth startup hook that restores them on every interpreter
#    launch + does an immediate restore:
uv run python scripts/setup_qwen3_tf.py

# 2. (Re)capture the golden reference (eager transformers, gitignored under
#    golden/qwen3/):
uv run python -m starling.qwen3.golden

# 3. Correctness tests (byte-identical vs golden):
uv run python -m pytest tests/test_qwen3_pipeline.py -q

# 4. End-to-end smoke + self-check:
uv run python -m starling.qwen3.pipeline

# 5. Benchmarks:
uv run python scripts/bench_qwen3_rtf.py        # RTFx vs stock + CrispASR tiers
uv run python scripts/bench_qwen3_batched.py    # batched B-size sweep
uv run python scripts/bench_qwen3_crispasr.py   # CrispASR cross-engine
```

## Important: always use `uv run python -m …`

Run pytest and scripts as **`uv run python -m pytest`** / **`uv run python script.py``,
not bare `uv run pytest`. The host has a user-space (`~/.local`) transformers
that shadows the venv; `uv run python -m` uses the venv's `sys.path`
consistently, while bare `uv run pytest` can pick up the user-space one.

## Why the setup script / bootstrap

`Qwen3ASRForConditionalGeneration` is not in an official transformers release
at the pinned commit. Three layers keep it importable:

1. `scripts/setup_qwen3_tf.py` — copies the 4 missing model dirs
   (`minicpm3`, `nemotron3_5_asr`, `qwen3_asr`, `xcodec2`) + the additive
   `audio_utils.py` from the stable uv git checkout into site-packages, and
   writes a `.pth` startup hook so the restore re-runs on every interpreter
   launch. Run once after a fresh `uv sync` / venv rebuild.
2. The `.pth` hook (`zzz_starling_qwen3_tf_bootstrap.pth`) — runs
   `_tf_bootstrap.ensure_qwen3_asr()` at startup, before any user import, so
   `from transformers import Qwen3ASRForConditionalGeneration` works cold.
3. `starling/qwen3/__init__.py` imports `_tf_bootstrap` too — so any
   `import starling.qwen3` self-heals even if the `.pth` is absent.
