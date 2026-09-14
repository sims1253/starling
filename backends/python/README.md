# Deprecated Python/CUDA serving backend

The NVIDIA-only Python serving path is deprecated in favor of
[`backends/native`](../native/). It remains available for kernel research,
benchmark reproduction, and existing integrations. New app features and API
compatibility work target the native server.

The root `pyproject.toml` and `uv.lock` still own this environment, with source
in `src/starling/`. Keeping these paths preserves Python imports, recorded
benchmark commands, and existing editable installs.

To run the legacy server deliberately:

```bash
uv sync --extra server
uv run --extra server python backends/python/serve.py --model parakeet --port 8181
# Existing entry point remains available:
uv run --extra server starling-python-serve --model parakeet --port 8181
```

This environment requires the original CUDA/PyTorch dependencies. It does not
implement the new `/v1/audio/transcriptions` route. Use `/inference` or switch to
the native backend. See [legacy setup](../../docs/python-serving.md).
