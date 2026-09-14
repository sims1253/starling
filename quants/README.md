# Starling quants

This component owns the GGUF recipe catalog and artifact records. Recipes moved
from `benchmarks/recipes/` to `quants/recipes/`; historical benchmark result JSON
keeps the commands used at measurement time. Converter scripts and calibration
harnesses remain in `scripts/` and `benchmarks/`.

The catalog distinguishes baseline, measured, and experimental profiles. Those
labels refer to the checked-in research, not a new accuracy certification.
A newly built artifact always starts with `evaluation: not_evaluated`.

## Inspect and build

The catalog CLI needs only Python 3.10+:

```bash
python -m quants.starling_quants list
python -m quants.starling_quants check
python -m quants.starling_quants plan parakeet-q8 \
  --input models/parakeet-f32.gguf --output models/parakeet-q8.gguf
```

You can also install this component independently with
`uv run --project quants starling-quants check`. It has no Torch or CUDA dependency.

Build the native quantizer, then execute the plan:

```bash
cmake --preset native-cpu
cmake --build build/native-cpu --target starling-quantize -j
python -m quants.starling_quants build parakeet-q8 \
  --input models/parakeet-f32.gguf --output models/parakeet-q8.gguf
python -m quants.starling_quants verify models/parakeet-q8.gguf
```

Use a source GGUF that matches the selected profile's model. The wrapper checks
file magic, not the model architecture; engine compatibility and accuracy still
need validation. Calibrated profiles require `--imatrix path/to/calibration.imx`.
The catalog carries recipe flags such as `--f32-1d` and `--shrink-f16` so the plan
cannot silently omit them.

`build` refuses existing outputs. It runs into a temporary file and publishes a
completed artifact using an exclusive hard link, then a `.gguf.json` record.
Use a local filesystem with hard-link support. If the process crashes between
those two publications, a complete GGUF can remain without a record; `verify`
will fail until the artifact is rebuilt at a new destination. The record includes
source, recipe, calibration, quantizer, and output SHA-256 hashes where applicable.
Hashes identify files; they do not establish model quality or publisher trust.

Keep source artifacts and calibration files unchanged during a build. Large
models, generated quants, recordings, and calibration corpora stay out of Git.

## Evaluation

Run `python -m unittest discover -s quants/tests -v` for catalog, failure recovery,
and provenance checks. Use the existing [quantization research](../docs/quantization.md)
and [benchmark harnesses](../benchmarks/README.md) for real audio evaluation.

Include negation, uncommon vocabulary, resumed lists, spoken corrections,
intentional filler, short answers, and long recordings in release evaluation.
Text-only regression fixtures cannot measure omissions made by the ASR model.
