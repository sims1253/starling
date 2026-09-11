# SONAR-OSS harness for Starling

Evaluate Starling models, native backends, and quantized GGUFs with
[SONAR-OSS](https://github.com/PSDN-AI/SONAR-OSS) (`psdn-sonar`) instead of the
in-repo `benchmarks/bench_leaderboard.py`.

Each **variant** — one `starling-serve` model × ggml backend × GGUF — is launched
as its own server on a free port and scored over a SONAR-prepared dataset. SONAR
produces per-utterance CSV, a `scores_<model>.json` artifact, and its own
leaderboard aggregation. Nothing here derives a number SONAR did not measure.

## Why a separate environment

SONAR pins Python 3.10–3.12 plus its own torch/transformers, and the native ggml
engines exist precisely to avoid the Python runtime. The harness therefore lives
in its own `uv` project (`benchmarks/sonar/pyproject.toml`) and talks to Starling
over HTTP (`POST /inference`), the same contract both `starling-serve` and
`python -m starling.server` expose. The adapter registers itself into SONAR's
model registry at run time. The optional quality bypass patches the pinned
SONAR version's quality pass and model loaders.

## Setup

```bash
# 1. isolated SONAR venv (CPU torch — this box has no NVIDIA GPU)
uv sync --project benchmarks/sonar

# 2. native serve binaries the presets expect (run from the repository root)
git submodule update --init --recursive
cmake -B build-ar-vk -DSTARLING_SERVE=ON -DSTARLING_GGML_VULKAN=ON
cmake --build build-ar-vk -j --target starling-serve
cmake -B build-ar-cpu -DSTARLING_SERVE=ON
cmake --build build-ar-cpu -j --target starling-serve
```

See [native serving](../../docs/native-serving.md) for backend prerequisites.
`--binary NAME=PATH` overrides either binary path. Dataset preparation does not
require these binaries or GGUF files. Scoring requires the GGUFs listed in
`variants.py` under `models/`, or in the directory passed to `--gguf-dir`.

## Prepare the dataset (once)

Uses SONAR's own `discover`. FLEURS English is the native English corpus;
`--max-samples` bounds each split (the full HF split is still downloaded the
first time, per SONAR's documented behavior). Use `--language bn|hi|ko` for the
other languages SONAR ships normalizers for.

```bash
uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
    --prepare-only --language en --max-samples 100
# -> benchmarks/sonar/data/en/fleurs/{train,validation,test}.tsv
```

Prepared splits retain their original sample cap. To enlarge a cached split,
add `--force-prepare --max-samples N`; use `N=0` for all samples. `--tsv` uses
the supplied file without preparing data.

## Run a sweep

GPU sweeps use Starling's shared lock and pass its descriptor to the server.
For Vulkan, HIP, or Metal, pass `--gpu-uuid` with a stable physical-device key
shared by every process using that GPU. Replace `vulkan:device-serial` below
with that key. It identifies the lock; `STARLING_GGML_DEVICE` selects the
runtime device. A backend mismatch or CPU fallback fails the variant before
scoring. CPU variants explicitly select CPU. On Windows, serialize GPU work
externally and set `STARLING_GPU_LOCK_DISABLE=1`.

```bash
# quick plumbing check: parakeet bf16 + q4_0 on Vulkan
uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
    --preset smoke --language en --max-samples 100 --gpu-uuid vulkan:device-serial

# the full parakeet quant ladder on Vulkan (accuracy vs. bits)
uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
    --preset quants --language en --max-samples 100 --gpu-uuid vulkan:device-serial

# same model+quant on CPU vs Vulkan (does the implementation change output?)
uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
    --preset backends --language en --max-samples 100 --skip-audio-quality \
    --gpu-uuid vulkan:device-serial

# custom grid / an alternative build
uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
    --models parakeet moss --quants bf16 q8_0 --backends vulkan \
    --binary vulkan=build-ar-vk/starling-serve \
    --tag my-run --skip-audio-quality --gpu-uuid vulkan:device-serial
```

`--dry-run` prints the resolved grid without launching anything.
`--skip-audio-quality` drops SONAR's per-clip SNR/MOS pass (recomputed for every
variant, hours of CPU here) and disables UTMOS/DNSMOS/SQUIM model loading,
including background prewarming. WER/CER/POSEIDON and latency remain enabled.

Each run needs a new `--tag`; existing result directories are rejected. The
runner finishes the sweep and writes available results, then exits nonzero
if any variant failed or no leaderboard rows were produced.

## Presets

| Preset | Grid |
| --- | --- |
| `smoke` | parakeet `bf16`, `q4_0` on Vulkan |
| `quants` | parakeet `bf16`, `q8_0`, `q6k_imx`, `q5_0`, `q4_0`, `q4_imx`, `q2_imx` on Vulkan |
| `backends` | parakeet `bf16`, `q4_0` on Vulkan + CPU |
| `moss_quants` | moss `bf16`, `q8_0`, `q4e8_imx`, `q4e4_imx`, `q2e4_imx` on Vulkan |

The ladder and binaries live in [`variants.py`](variants.py); add a model by
adding a `QUANT_LADDERS` entry, or a backend by adding to `DEFAULT_BINARIES`
(e.g. a CUDA `starling-serve`). The orchestrator verifies the native server's
startup log; Python-server use requires a separate lifecycle integration.

## Output

```
benchmarks/sonar/results/<tag>/
  <variant>/scores_<variant>.json     SONAR artifact (WER/CER/POSEIDON/latency/lineage)
  <variant>/asr_detailed_<variant>.csv per-utterance rows
  <variant>/variant.json              engine/quant provenance for this cell
  logs/<variant>.log                  server stdout/stderr
  leaderboard.json / .md              SONAR aggregation over the run
  manifest.json                       leaderboard rows joined with variant provenance
```

## Does this replace `bench_leaderboard.py`?

Not directly, and the harness is built to make the difference explicit:

- **Corpus.** `bench_leaderboard.py` scores the 7-dataset English Open ASR
  Leaderboard corpus; this harness scores SONAR's native FLEURS/CommonVoice/
  Zeroth splits. Bridging the OAL corpus into a SONAR TSV is the follow-up if the
  native run looks worthwhile.
- **Metric.** `bench_leaderboard.py` uses Whisper's `EnglishTextNormalizer` +
  kaldialign WER; SONAR uses its own per-language normalization contract. The
  same audio will not produce byte-identical WER. SONAR additionally reports
  CER, semantic similarity, POSEIDON, latency percentiles, and provenance
  (`scores_*.json` records device/normalization/revision).
- **Latency.** `bench_leaderboard.py` reports RTFx with in-process timing; here
  latency is measured by the adapter over the HTTP hop, so it includes transport.
  Treat it as a backend-vs-backend comparison, not an absolute RTFx number.

The natural end state is: keep `bench_all.py` for kernel/fixture drift, use this
for public quality numbers, and upstream an OAL→SONAR dataset bridge to SONAR-OSS
if the comparison holds up.
