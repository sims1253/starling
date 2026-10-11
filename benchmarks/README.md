# Benchmarks

Measurement and calibration tools for the native engine. They drive
`libstarling_ggml` in-process (through [`tools/starling_ggml.py`](../tools/starling_ggml.py))
or `starling-serve` over HTTP; none of them needs PyTorch to run a model.

```sh
uv sync --locked --group eval   # corpus export, FLEURS calibration clips, WER
```

Entry points:

- [`experiments/`](experiments/README.md): sealed paired A/B experiments
  against `starling-serve` (`run_experiment.py`), the serve contract smoke and
  the streaming replay workloads. Stdlib only.
- [`fast_engine/`](fast_engine/AUTORESEARCH.md): Vulkan fast-engine and phone
  measurement. `wer_engines.py` scores engine variants of one library on a
  corpus exported by `export_fleurs.py`; `wer.py` holds the shared
  normalizer and WER/CER scorer; `score_phone_quant.py` scores phone
  quantization runs.
- `imatrix_collect.py`: collects a parakeet activation importance matrix for
  `starling-quantize --imatrix` (fixtures, `--wavs` directories, or FLEURS
  clips via `fleurs_util.py`; `fleurs_download.py` writes FLEURS clips to disk).
- `s1/`: S1-mini quantization tools: `collect_imatrix.py` (decoder imatrix
  over the text fixtures) and `quant_spans.py` (protected-span audit of a
  quantized model against BF16, cases in `tests/fixtures/s1_quant_spans.json`).

Model files are described in [docs/models.md](../docs/models.md). The former
Python/CUDA benchmark suite (`bench_all.py`, the leaderboard and quantization
WER drivers, SONAR, contention and speculative pilots) was removed; it is at
tag `python-research-final`.
