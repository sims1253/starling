# IQ2_XXS second-order pilot for Parakeet (#49)

The current [imatrix collector](../cpp/runtime/imatrix.cpp) stores only the
diagonal of the input covariance: each entry is a sum of squared activations
for one input channel. The [GPTQ paper](https://arxiv.org/pdf/2210.17323)
uses the inverse of the full input covariance to compensate rounding errors
in later columns (Algorithm 1). [AWQ](https://arxiv.org/html/2306.00978)
instead rescales salient input channels before weight quantization and applies
the inverse scale to activations (Section 3.2). These are distinct algorithms;
neither follows from increasing the amount of diagonal imatrix data.

This pilot adds an opt-in trace to the CPU imatrix callback. Set
`STARLING_IMATRIX_TRACE_TENSOR` to one named weight and
`STARLING_IMATRIX_TRACE_PATH` to a binary output path while also setting the
usual `STARLING_IMATRIX`. It retains up to 4096 activation vectors and writes
`STLGACT1`, little-endian `u32` width, `u64` vector count, and contiguous F32
vectors. Normal collection and serving behavior are unchanged when the trace
variables are absent. The trace is for a bounded offline study, not serving.

## Experiment

- Source: public Parakeet TDT 0.6B v3 revision
  `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, F32 GGUF SHA256
  `f3337ef2cd24b458942e05f420a0f8fbf6466d25a07e9b97f7db7e2775845ffa`.
- Existing 325,124,224-byte IQ2_XXS + shrink16 baseline SHA256
  `bfee29b2b3419b387c8bc1b28a1772b506dd57aa9cfd879b2530bc287d93b0d3`.
  Its original imatrix SHA256 is
  `de9f635dc79a149d3389d41d7e8f460c4362427d68c2b0199697baefe97c69e8`.
- Tensor: `encoder.layers.0.self_attn.linear_q.weight`, shape 1024 × 1024.
  The 256-weight IQ2_XXS block occupies 66 bytes, or **2.0625 bits per
  weight**. The issue's “sub-2-bit” title does not describe this format.
- Calibration: FLEURS `en_us_train_0..3` and `de_de_train_0..3`, 1136
  activation vectors; trace SHA256
  `279198fb5c0d7ab2c6eb4440ca31577f00b5475ff8ce8e5ab51445f620dba4f1`.
  Held out: `en_us_validation_0..3` and `de_de_validation_0..3`, 1012 vectors;
  trace SHA256
  `9a159197cddddf013b64fdc44f17527578052e2b7da849b7cd2417332e7d1305`.
  The [16-file WAV manifest](quantization-second-order-wavs.json) pins the
  exact decoded audio bytes; the original downloader did not pin the dataset
  repository revision.

The [pilot script](../benchmarks/quant_iq2_block_pilot.py) calls the vendored
ggml IQ2_XXS encoder and uses the same imatrix as the baseline. Its direct
quantization reproduces the baseline tensor **byte for byte**. The blockwise
GPTQ variant estimates the full 1024 × 1024 input covariance from calibration
activations, adds 1% of its mean diagonal for damping, and uses its inverse to
send each finished 256-channel block's quantization error into the remaining
blocks. ggml still rounds each 256-channel vector block as a unit, so this is
a blockwise GPTQ-style update, not GPTQ's within-block column-by-column
rounding. The repaired tensor has exactly the same 270,336-byte packed shape;
the candidate GGUF differs from the baseline only in that tensor.

The AWQ arm uses mean absolute input activation as salience. It searches
`alpha` in `{0.25, 0.5, 0.75, 1.0}` on calibration output error, quantizes
`W * s` with transformed imatrix weights, and evaluates with `1/s` applied to
the input channels. It chose `alpha = 0.5`. This arm is a layer calculation:
the inverse activation scale has not been folded into the native model, so it
does not produce a runnable GGUF.

| Relative layer-output RMS error | Calibration | Held out |
| --- | ---: | ---: |
| Existing IQ2_XXS baseline / direct encoder | 0.31421 | 0.31341 |
| Block-compensated IQ2_XXS | 0.19380 | 0.20856 |
| AWQ-scaled IQ2_XXS (layer-only) | 0.29249 | 0.29072 |

The block-compensated tensor reduced held-out relative output error by 33.5%
against the exact existing IQ2_XXS encoding. The AWQ layer transform reduced
it by 7.2%. These are local reconstruction results for one layer and eight
held-out clips. The blockwise method does not save model bytes.

The runnable GGUF with this one patched tensor (SHA256
`89d492571936a55f7610d34ec8fcf2371a0068a9c50f0aaf42e56d51691b0e89`)
was then transcribed on the same 300 English and 300 German FLEURS validation
clips as the baseline, in a fresh CPU process with the same native library,
source, imatrix, scorer, and clip hashes. The
[paired comparison](quantization-second-order-wer.json) checks those
identities and uses 10,000 paired bootstrap resamples.

| CPU cohort | Baseline mean WER | One-tensor candidate | Candidate minus baseline, pp [95% CI] |
| --- | ---: | ---: | ---: |
| English | 8.12% | 8.22% | +0.102 [-0.082, +0.299] |
| German | 10.23% | 10.11% | -0.116 [-0.346, +0.113] |

The English interval reaches beyond the +0.2 pp margin used in #50, so this
one-tensor candidate is inconclusive on that quality rule. German satisfies
that margin, but the interval includes zero. The file saves **zero bytes**,
as expected when replacing one tensor within the same fixed-size IQ2_XXS
format. This is not a rejection of #49's quality objective; it only fails
#50's separate 15 MB storage gate. The local layer gain did not establish a
whole-model WER improvement. Tail languages and Pixel
latency/energy remain unmeasured. This candidate is an experimental artifact,
not a replacement model.

The numeric record is in [quantization-second-order-pilot.json](quantization-second-order-pilot.json).
To reproduce, build the CPU shared library and place the manifest's WAVs in
separate `calibration-wavs` and `validation-wavs` directories. For each
directory, run:

```bash
STARLING_GGML_LIB=build/libstarling_ggml.so \
STARLING_GGML_DEVICE=cpu \
STARLING_IMATRIX_TRACE_TENSOR=encoder.layers.0.self_attn.linear_q.weight \
STARLING_IMATRIX_TRACE_PATH=calibration-activations.bin \
python benchmarks/imatrix_collect.py --model parakeet-f32.gguf \
  --output calibration-trace.imx --tiers '' --wavs calibration-wavs
```

Use distinct trace and imatrix outputs for validation. Then run:

```bash
python benchmarks/quant_iq2_block_pilot.py \
  --source parakeet-f32.gguf --baseline parakeet-iq2-baseline.gguf \
  --tensor encoder.layers.0.self_attn.linear_q.weight \
  --imatrix parakeet-original.imx \
  --calibration calibration-activations.bin \
  --validation validation-activations.bin \
  --ggml-base build/ggml/src/libggml-base.so \
  --gguf-out parakeet-iq2-one-tensor-pilot.gguf \
  --json iq2-pilot.json
```

`gguf==0.19.0` and NumPy are required for this Python script. The
`--gguf-out` option copies the baseline and patches only the named tensor.
