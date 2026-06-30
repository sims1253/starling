# Running the Higgs-Audio-v3 megakernel

All commands run via **`uv`** (the project's environment manager). higgs-audio-v3-stt
runs in its **own isolated venv** (`.venv-higgs`, `transformers==4.51.3`) — NOT the
repo's shared `.venv` (transformers 5.13) — because the model's `trust_remote_code`
modeling breaks across the 4.51→5.x boundary (Whisper mask plumbing,
`Qwen3DecoderLayer` return shape, `GenerationConfig.generation_kwargs`). The shared
`.venv` is untouched; granite/parakeet/qwen3/moss/ark keep using it.

Because `.venv-higgs` is a separate environment (not the `uv run` project default),
**every uv invocation must use `--no-project --python .venv-higgs/bin/python`**.
`--no-project` is essential: plain `uv run` syncs to the project's pinned deps
(transformers 5.13) and would either rebuild a wrong env or risk clobbering the
shared `.venv`. From the worktree root (`starling-higgs/`):

```bash
# 0. One-time: create the isolated venv + install the higgs deps.
uv venv .venv-higgs --python 3.10
uv pip install --python .venv-higgs/bin/python \
    "transformers==4.51.3" accelerate soundfile librosa pytest \
    torch==2.12.1   # cu130 wheel; matches the repo's torch for sm_120
# (the model + tokenizer download from HF on first run)

# 1. (Re)capture the golden reference (upstream transcribe(), gitignored under
#    golden/higgs_golden.json):
uv run --no-project --python .venv-higgs/bin/python python scripts/capture_golden_ref.py

# 2. Correctness tests (byte-identical vs golden):
uv run --no-project --python .venv-higgs/bin/python python -m pytest tests/test_higgs_mega.py -q

# 3. End-to-end smoke + byte-exact check:
uv run --no-project --python .venv-higgs/bin/python python scripts/verify_megakernel.py

# 4. RTF benchmark (starling vs stock vs CrispASR tiers):
uv run --no-project --python .venv-higgs/bin/python python benchmarks/bench_higgs_rtf.py
```

## Important: always `uv run --no-project --python .venv-higgs/bin/python python -m …`

- Use `uv run ... python -m pytest`, **not** bare `uv run pytest` (the host has a
  user-space `~/.local` transformers that can shadow the venv; `python -m` uses
  the venv's `sys.path` consistently).
- Never drop `--no-project`: plain `uv run` syncs to the project env
  (`pyproject.toml`, transformers 5.13) and ignores `.venv-higgs`.
- `.venv-higgs/` is gitignored — it is rebuildable from step 0 above.
