# Calibrated GGUF quantization

Starling ships its own quantization pipeline for the native ggml engines —
the ASR analogue of the "dynamic quants" the text-LLM world has: block
quantization (Q8_0 / K-quants) weighted by an **importance matrix** collected
on real audio, plus per-tensor recipes for sensitivity sweeps, verified by
WER. Parakeet-tdt-0.6b-v3 is the proof of concept; the same tooling is
model-agnostic and intended for the larger engines (moss-transcribe-2b,
qwen3-asr-1.7b) where halving VRAM actually changes what
fits on a local machine.

## Community-GGUF compat

The parakeet engine also **loads community GGUFs directly** — there is no
single "standard" parakeet GGUF, so the loader (via
`cpp/parakeet/compat.cpp`) recognizes and normalizes the two dialects found
in the wild, zero-copy:

- **parakeet.cpp / CrispASR naming** (e.g. `cstr/parakeet-tdt-0.6b-v3-GGUF`):
  `decoder.lstm.N.w_ih` / `encoder.layers.N.attn.q.weight` /
  `encoder.pre.*` tensors and flat `parakeet.*` KV, with embedded mel
  tensors. Pure name-alias + KV remap.
- **transcribe.cpp naming** (e.g. `handy-computer/parakeet-tdt-0.6b-v3-gguf`):
  `enc.blocks.N.*` / `pred.lstm.N.{Wx,Wh,bias}` tensors with a FUSED LSTM
  bias, `stt.*` KV, and no filterbank tensor. The compat layer synthesizes
  the slaney mel filterbank + hann window from the `stt.frontend.*`
  metadata (bit-identical, max-abs 0.0, to this repo's converter-embedded tensor;
the synthesis reproduces the librosa-fallback numerics, which matched the
cached-filterbank path here) and splits the
  fused bias (bias_ih = fused, bias_hh = 0 — the two only ever sum).

The vocab/blank/durations conventions are reconciled against the joint
head's geometry (`[vocab | blank | durations]`), and the prediction
embedding is fetched dtype-agnostically (F32/F16/BF16 or any quant type via
ggml's type-traits dequant — community files store it F16 or quantized).
Accepted: cstr F16, handy Q8_0 and Q4_K_M all transcribe the fixtures
verbatim (0.00% WER). Quantized community files run through the same
`mul_mat` dequant paths as starling's own quants.

## Pipeline

```
nvidia/parakeet-tdt-0.6b-v3 ──convert_parakeet_gguf.py──> f32.gguf
                                                            │
              calibration audio (fixtures + real wavs)      │
                            │                               │
              STARLING_IMATRIX=... engine transcribe        ▼
                            │                      starling-quantize
                            └── imatrix.bin ──────────────► │ (--imatrix)
                                                    quantized ggufs
                                                            │
                                    benchmarks/fast_engine/wer_engines.py
```

1. **Convert** to a numeric-exact base GGUF (F32 is the quantization input,
   matching llama.cpp practice — quantize once from exact weights, never from
   an already-rounded dtype):

   ```bash
   uv run --group convert python scripts/convert_parakeet_gguf.py \
       --model-id nvidia/parakeet-tdt-0.6b-v3 --dtype f32 \
       --output models/parakeet-tdt-0.6b-v3-f32.gguf
   ```

2. **Collect** the importance matrix (optional but recommended below Q6_K):
   run the engine over calibration audio with `STARLING_IMATRIX` set. Every
   `MUL_MAT` whose src[0] is a named weight contributes the per-input-channel
   sum of squared activations (`cpp/runtime/imatrix.cpp`), accumulated across
   the encoder pass and every greedy-TDT decode step. Collection rides
   `ggml_backend_sched`'s eval callback and forces observed nodes onto the
   CPU backend, so it is an offline pass, not a serving mode:

   ```bash
   uv run --group eval python benchmarks/imatrix_collect.py \
       --model models/parakeet-tdt-0.6b-v3-f32.gguf \
       --output models/parakeet-tdt-0.6b-v3.imx.bin \
       --tiers short,medium,long --repeats 2 --wavs path/to/calibration_wavs/
   ```

   The fixtures are one LibriSpeech speaker; for real calibration add an hour
   or two of diverse 16 kHz mono audio (`--wavs`). Parakeet v3 was trained on
   Granary (manifests CC-BY at `nvidia/Granary`; audio via the YODAS/MOSEL
   upstream), which is distribution-matched calibration data by construction.

3. **Quantize** (`build/starling-quantize`):

   ```bash
   ./build/starling-quantize \
       --input  models/parakeet-tdt-0.6b-v3-f32.gguf \
       --output models/parakeet-tdt-0.6b-v3-q4_k_m.gguf \
       --quant  q4_k_m \
       --imatrix models/parakeet-tdt-0.6b-v3.imx.bin
   ```

4. **Verify** by corpus WER against the f32 baseline: export a fixed clip set
   once (from a local `google/fleurs` parquet cache in the HF hub cache), then
   score each GGUF on it with the same native library (one run per file;
   `--variants ggml` keeps the ggml engine):

   ```bash
   uv run --group eval python benchmarks/fast_engine/export_fleurs.py \
       --cfg en_us --n 100 --out models/fleurs_en
   for m in f32 q8_0 q4_k_m; do
     uv run --group eval python benchmarks/fast_engine/wer_engines.py \
         --lib build/libstarling_ggml.so --model parakeet --variants ggml \
         --gguf models/parakeet-tdt-0.6b-v3-$m.gguf --clips models/fleurs_en
   done
   ```

   For a promotion decision use the sealed paired protocol below.

## What quantizes and what must stay exact

The engine consumes linears through `ggml_mul_mat`, which dequantizes any
block-quantized src[0] on the fly (CPU vec-dot, CUDA dmmv, Vulkan dmmv) —
those tensors are fair game:

- attention projections `self_attn.linear_{q,k,v,out,pos}.weight`
- FFN `feed_forward{1,2}.linear{1,2}.weight`
- subsampling output projection `encoder.pre_encode.out.weight`
- joint projections `joint.enc.weight`, `joint.pred.weight`,
  `joint.joint_net.2.weight`
- prediction LSTM `decoder.prediction.dec_rnn.lstm.weight_{ih,hh}_l*`

Kept at the source dtype on purpose:

- **conv weights** (`conv` in the name): the depthwise/subsampling convs go
  through `ggml_conv_*` — no dequant path. The conformer pointwise convs
  (`encoder.layers.N.conv.pointwise_conv{1,2}.weight`) are opt-in, see below.
- **the prediction embedding** `decoder.prediction.embed.weight`: the engine
  expands it into a host F32 table (`prediction.cpp`). An explicit recipe
  rule can change its storage dtype for `general.architecture=parakeet_tdt`;
  named levels and recipe defaults keep its source dtype.
- **batch-norm statistics, norms, biases, pos_bias_u/v**: 1-D host-folded or
  broadcast operands.
- **mel constants** (`preprocessor.*`): the filterbank/window must stay exact.

Opt-in tensors keep their source dtype under named levels and `default`
lines, and change only when an explicit recipe rule matches them. The table
(`opt_ins()` in `cpp/tools/starling_quantize.cpp`) is keyed by
`general.architecture` and lists the types each engine reads the tensor
with; a matching rule with any other type fails:

| architecture | tensor | accepted types |
|--------------|--------|----------------|
| parakeet_tdt | `decoder.prediction.embed.weight` | q8_0, f32 |
| parakeet_tdt | `encoder.layers.N.conv.pointwise_conv{1,2}.weight` | f16, q8_0, q6_k, q4_k, q4_0 |
| moss_transcribe, qwen3, granite, ark, voxtral | `llm.embed.weight` | f32, f16, q8_0, q6_k, q5_k, q4_k, q4_0 |
| moss_transcribe | `enc.conv_out.weight` | f32, f16, q8_0, q6_k, q5_k, q4_k, q4_0, iq4_nl |
| granite | `enc.blk.N.conv_{up,down}.weight` | f32, f16, q8_0, q6_k, q5_k, q4_k, q4_0, iq4_nl |

The pointwise convs are kernel-size-1 convs stored `[1, in, out]`; a
quantized copy is written flattened as `[in, out]` (same element order) so
the row sits on ne[0]. The conformer graph and the fast engine both read
them as 2-D matmul weights. The importance matrix records them under ggml's
derived name (`<name> (copy) (reshaped)`); the quantizer falls back to that
entry when the plain name is missing. A tied `llm.embed.weight` is also the
lm_head: its full table is read on every decode step and never repacked by
the CPU backend, so its type sets a large share of the decode bandwidth.

Row width decides the floor: the joint/LSTM linears have 640-wide rows, which
is not a multiple of the K-quant block size (256), so they fall back to
Q8_0 (block 32) automatically — handy, since those are the most
duration-sensitive tensors anyway. The big encoder tensors (1024/4096-wide)
take any K-quant.

## Levels and recipes

`starling-quantize --quant` understands llama.cpp-style mixes:

| level   | base | bump group |
|---------|------|------------|
| q8_0    | Q8_0 | — |
| q6_k    | Q6_K | — |
| q5_k_s  | Q5_K | — |
| q5_k_m  | Q5_K | Q6_K |
| q4_k_s  | Q4_K | — |
| q4_k_m  | Q4_K | Q6_K |

The bump group (attention value/out projections, FFN down-projections,
`joint.enc`) gets one notch more precision, mirroring llama.cpp's empirical
`_M` mixes.

`--recipe file` overrides the level with first-match regex rules for
sensitivity sweeps (this is the "dynamic" hook — recipes are how
per-tensor-group experiments are expressed):

```
# parakeet-q4-enc.recipe: encoder at Q4_K, joint+LSTM pinned to Q8_0,
# attention a notch higher
default q4_k
self_attn\.linear_q\.weight$   q5_k
self_attn\.linear_k\.weight$   q5_k
^joint\.                       q8_0
^decoder\.prediction\.dec_rnn\. q8_0
```

`--shrink-f16` additionally stores the kept CONV WEIGHTS (~300 MB for
parakeet) as F16. The pointwise convolutions use F16 matmuls; the CUDA
depthwise kernel requires F32, so the graph widens those stored F16 values
before that operation. Everything else stays F32: 1-D biases/norms/BN
statistics feed `ggml_add`/`ggml_mul` broadcasts which reject mixed dtypes.

### Parakeet embedding and compact recipes

Two opt-in recipes retain the IQ2_XXS encoder and its importance matrix:

- `quants/recipes/parakeet-iq2-embedding-q8.recipe` stores the prediction
  embedding as Q8_0 and preserves every other tensor's existing policy.
- `quants/recipes/parakeet-iq2-compact.recipe` also stores six 640-wide
  joint/LSTM matrices as IQ4_NL (32-element blocks, 4.5 bits per weight).
  This option remains **experimental**: it increased observed word errors
  in the controlled study below.

A third opt-in recipe targets the speed/memory sweet spot rather than
minimum size:

- `quants/recipes/parakeet-q4-fullimx.recipe` is uniform Q4_0 with the
  full-corpus importance matrix (fixtures x2 + 32 real utterances, ~26k
  observations). On the Vulkan iGPU this beat Q8_0 on every axis — medium
  fixture 770.7 -> 603.5 ms, peak RSS 972 -> 710 MB — with real-corpus WER
  3.55 vs the Q8_0 baseline's 3.94. The calibration mix matters: a
  short-tier-only imatrix measured strictly worse than Q8_0, and uniform
  Q4_0 beat attention-only and hybrid FFN variants. Requires `--imatrix`;
  the collection command is in the recipe header.

Named levels and their fallback rules stay unchanged. Other model
architectures retain the embedding keep-list.

```bash
cmake -B build-cpu -DSTARLING_GGML_SHARED=ON
cmake --build build-cpu --target starling-quantize starling_ggml -j
build-cpu/starling-quantize \
  --input models/parakeet-tdt-0.6b-v3-f32.gguf \
  --output models/parakeet-iq2-compact.gguf \
  --recipe quants/recipes/parakeet-iq2-compact.recipe \
  --imatrix models/parakeet-imx-prod-25x48.bin --shrink-f16
```

For embedding-only storage, substitute `parakeet-iq2-embedding-q8.recipe`
and a separate output filename. To reproduce the baseline, replace
`--recipe ...` with `--quant iq2_xxs`; keep the source, matrix, and
`--shrink-f16` identical.

Start from the original floating-point model. The quantizer cannot decode
an already quantized source. Recipe rules use first-match precedence:
place any F32 or Q8_0 precision overrides before the compact rules.
Any regex rule that matches the embedding opts it in, including broad rules
such as `.* q8_0` or `^decoder\. q8_0`. This changes older custom recipes
whose broad rules previously left embeddings untouched. A first matching
Q6_K or other unsupported type now fails with an error; only Q8_0 and F32
are accepted. `default q8_0` alone still keeps the source embedding dtype.

To preserve an F32 source embedding while applying a broad rule, put its
exact F32 override first. An override after the broad rule has no effect:

```text
default q8_0
^decoder\.prediction\.embed\.weight$ f32
^decoder\. q6_k
```

This override stores F32; it does not preserve the storage dtype of an F16
or BF16 source embedding.
The embedding has no collected importance entry because the engine reads
it on the host. Q8_0 does not require one.

The 2026-09-06 CPU study used 300 English and 300 German FLEURS test clips,
the F32 source, and `parakeet-imx-prod-25x48.bin`. Each model ran in a fresh
process. Tensor payload comparisons confirmed that only the intended one
or seven tensors changed. WER is the mean of per-clip WER percentages;
brackets give 95% bootstrap intervals. MB means decimal megabytes.

| variant | MB | EN WER | DE WER |
|---------|----|--------|--------|
| released IQ2 + shrink16 | 325.1 | 8.54 [7.59, 9.54] | 9.35 [8.22, 10.51] |
| Q8 embedding | 309.7 | 8.53 [7.56, 9.51] | 9.35 [8.22, 10.51] |
| Q8 embedding + IQ4_NL linears | 303.6 | 8.74 [7.67, 9.84] | 9.75 [8.54, 10.93] |

Paired WER differences from the released IQ2 baseline, in percentage points:

| variant | EN delta [95% CI] | DE delta [95% CI] |
|---------|-------------------|------------------|
| Q8 embedding | -0.013 [-0.040, 0.000] | 0.000 [0.000, 0.000] |
| Q8 embedding + IQ4_NL linears | +0.197 [-0.220, +0.638] | +0.394 [-0.020, +0.823] |

Q8 embedding alone saved 15.40 MB. It changed one English hypothesis,
removing one word error; all German hypotheses matched the baseline.
The zero German interval reflects this sample's identical outputs, not
proof of losslessness on other inputs. IQ4_NL saved another 6.10 MB but
added 18 English and 24 German word errors relative to the baseline.
Its paired intervals include zero and extend to +0.64 EN / +0.82 DE
percentage points. Noninferiority is not established. The measured combined
size is 303.6 MB; [issue #50](https://github.com/sims1253/starling/issues/50)
had proposed 285 MB.

Reproduce the evaluation with the exact cached corpus identified in the
[aggregate study record](quantization-compact-eval.json). To populate a
new cache, the downloader selects the first 300 valid test clips per config;
check its manifest against the record before treating it as the same sample.

```bash
uv run --group eval python benchmarks/fleurs_download.py \
  --out models/fleurs_test_big_ende --split test --fleurs en_us:300 --fleurs de_de:300
```

The study's scorer (`benchmarks/wer_quant.py`, protocol-sealed per-clip
records) and paired comparator (`benchmarks/compare_quant_wer.py`) were part
of the removed Python stack; reproduce it exactly from
`git checkout python-research-final`.
It scored baseline, embedding-only and combined files separately, retaining
clip IDs, decoded-audio hashes, references,
hypotheses, and unrounded WER. The paired analysis matched IDs, audio hashes,
and references, then sampled 300 per-clip WER differences with replacement
10,000 times per language using NumPy `default_rng(0)`. Its interval is the
2.5th–97.5th percentile of the resampled means. The aggregate record contains
input/output hashes, corpus-manifest construction, counts, timings, and the
incremental combined-minus-embedding comparison; raw transcripts stay local.

### Held-out compact-recipe promotion gate

The first 300 EN/DE **test** clips above were used to inspect the recipes, so
the next decision uses 300 EN/DE **validation** clips. The fixed acceptance
protocol is [`quants/protocols/parakeet-compact-validation-v1.json`](../quants/protocols/parakeet-compact-validation-v1.json):
each language must have exactly 300 paired clips, the candidate must save at
least 15,000,000 stored bytes, and the upper 95% paired-bootstrap bound on
candidate-minus-baseline mean per-clip WER must be **below +0.2 percentage
points** for both languages. The same rule applies to the embedding-only and
combined candidates. Passing is evidence for this sample and backend; a
release default still needs the intended accelerator and workload checks.

Run each arm in a fresh process with the same original F32 GGUF, imatrix,
native library, and device. The scorer, model, source, imatrix, engine, device,
and protocol hashes are recorded with the clip IDs, decoded-audio hashes,
references, hypotheses, and unrounded WER. The comparison refuses mismatched
inputs, missing clips, duplicate IDs, changed scoring code, a native-library
fallback from `STARLING_GGML_LIB`, or a protocol edited after evaluation.

```bash
uv run --group eval python benchmarks/fleurs_download.py \
  --out models/fleurs_validation_ende --split validation \
  --fleurs en_us:300 --fleurs de_de:300
```

The study's scorer (`benchmarks/wer_quant.py`, protocol-sealed per-clip
records) and paired comparator (`benchmarks/compare_quant_wer.py`) were part
of the removed Python stack; reproduce it exactly from
`git checkout python-research-final`.
At the tag, `wer_quant.py --protocol quants/protocols/parakeet-compact-validation-v1.json
--source-model ... --imatrix ... --include-clips --json ...` scored each arm
(baseline, embedding, combined) in its own process, and
`compare_quant_wer.py --protocol ... --baseline ... --candidate ...` applied
the gate. The comparator reports `pass`, `fail`, or `inconclusive`; a CI
touching the margin remains inconclusive. Keep opt-in recipes when a gate is inconclusive. For CUDA or
Vulkan runs, build the corresponding native library and set `--device CUDA0`
or `--device Vulkan0` in **both** arms; the result is specific to that backend.

The fresh CPU study used the public Parakeet checkpoint at revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, converted once to F32
(2,356,818,944 bytes; SHA256 `f3337ef2cd24b458942e05f420a0f8fbf6466d25a07e9b97f7db7e2775845ffa`).
The imatrix (SHA256 `de9f635dc79a149d3389d41d7e8f460c4362427d68c2b0199697baefe97c69e8`)
used six fixture repetitions and 84 FLEURS **train** clips across eight
languages. The held-out corpus used 300 validation clips each from `en_us`
and `de_de`. Its decoded WAV/TXT manifest SHA256 was
`50d9b85b633a536ec8b97d97c8f6aa602ec23d1b37c36ad33ab437704ac0e1aa`;
the dataset repository revision observed was
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, although the downloader's
`hf://` URL was not revision-pinned. The exact clip and decoded-audio hashes
are in the local per-arm JSON records. All three arms used the same CPU native
library (SHA256 `c13894a1e1972fc1e3afd5151f108764bd0ee6a3af1524d23f6bcbc4c60c728a`).

| CPU arm | Stored bytes | EN WER | DE WER | Paired EN delta [95% CI], pp | Paired DE delta [95% CI], pp | Gate |
|---------|-------------:|-------:|-------:|------------------------------:|------------------------------:|------|
| IQ2_XXS baseline | 325,124,224 | 8.12% | 10.23% | — | — | reference |
| Q8 embedding | 309,721,472 | 8.12% | 10.23% | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | pass |
| Q8 embedding + IQ4_NL linears | 303,616,512 | 8.45% | 10.79% | +0.328 [-0.090, +0.722] | +0.563 [+0.060, +1.100] | inconclusive |

The embedding-only arm saved 15,402,752 bytes and produced identical
hypotheses on these 600 CPU clips. That is a sample-specific result, not a
losslessness guarantee. The combined arm saved 21,507,712 bytes, but its
paired intervals cross the declared +0.2-point margin.

The same baseline and embedding-only files were also evaluated in fresh
CUDA0 processes on the same 600 validation clips. Both runs used the CUDA
native library (SHA256
`bb51327172bd90d05a474946bfe1a8e63e79acc7dff23962831dacbfb68a6528`)
with the depthwise convolution compatibility fix from #331. The model, source,
imatrix, corpus, scorer, and protocol hashes matched the CPU study; the
comparator verified CUDA0 as the actual backend in both runs.

| CUDA0 arm | Stored bytes | EN WER | DE WER | Paired EN delta [95% CI], pp | Paired DE delta [95% CI], pp | Gate |
|-----------|-------------:|-------:|-------:|------------------------------:|------------------------------:|------|
| IQ2_XXS baseline | 325,124,224 | 8.30% | 9.99% | — | — | reference |
| Q8 embedding | 309,721,472 | 8.28% | 9.99% | -0.018 [-0.053, 0.000] | 0.000 [0.000, 0.000] | pass |

CUDA0 Q8 embedding saved 15,402,752 bytes and passed the same predeclared
gate. The CPU and CUDA WER values differ, so the paired comparison is made
within each backend. The combined arm has no CUDA verdict. Both recipes stay
opt-in pending broader deployment and workload evidence, including Vulkan and
resident-memory/latency measurements.

### Pixel 10 Pro fast-Vulkan validation (2026-09-28)

The same two IQ2_XXS files passed the fixed validation gate on a Pixel 10 Pro
(Android 17, build `CP3A.260905.009`). Each arm decoded the same 300 EN and
300 DE validation clips through the fast Vulkan engine on the PowerVR D-Series
DXT-48-1536 MC1. The source F32 and imatrix hashes above, the
[`protocol`](../quants/protocols/parakeet-compact-validation-v1.json), and the
decoded audio/reference hashes matched the CPU and CUDA studies. The Android
`starling-bench` binary SHA256 was
`e1a6b3b0f7ff04b3e5a03a62aa5c55deeb2bf0d8bbd4ab4ee80b202755044e4a`
from commit `2a3bda4`. Its generic `backend=cpu` header describes the GGML
backend, while the fast engine reports its selected Vulkan device separately.
The [phone scorer](../benchmarks/fast_engine/score_phone_quant.py) requires
that actual fast-engine device log for every process.

| Fast-Vulkan arm | Stored bytes | EN WER | DE WER | Paired EN delta [95% CI], pp | Paired DE delta [95% CI], pp | Gate |
|-----------------|-------------:|-------:|-------:|------------------------------:|------------------------------:|------|
| IQ2_XXS baseline | 325,124,224 | 8.48% | 9.93% | — | — | reference |
| Q8 embedding | 309,721,472 | 8.48% | 9.98% | 0.000 [0.000, 0.000] | +0.050 [0.000, +0.150] | pass |

The candidate saved 15,402,752 stored bytes. Its upper paired-bootstrap
bounds were below +0.2 percentage points in both languages. Thirteen of the
600 hypotheses changed (five EN, eight DE). EN has the **same mean WER**, not
identical transcripts or lossless conversion; one DE clip increased by 15
per-clip percentage points, yielding the +0.050-point mean DE difference.
The [13 exact changes](https://github.com/sims1253/starling/blob/python-research-final/docs/evidence/pixel-compact-2026-09-28/changed-hypotheses.json),
[per-clip records and verdict](https://github.com/sims1253/starling/blob/python-research-final/docs/evidence/pixel-compact-2026-09-28/README.md),
and raw benchmark logs (all at tag `python-research-final`) make the result
auditable.

The fast engine repacked both IQ2 files into the same 718.6 MiB Vulkan weight
allocation. A matched, sequential baseline/candidate probe after the same
short-WAV warmup found identical GL mtrack (783,232 kB) and total process PSS
of 838,895 versus 839,229 kB. The 334 kB difference is too small to support
a resident-memory benefit; this is a warmed short-fixture snapshot, not a
peak or co-residency budget. Stored-byte savings therefore did **not** reduce
the observed Vulkan weight or graphics allocation in this build. The corpus
ran across rising thermal status, screen wakeups, and a charge counter that
reversed direction as the phone began charging. Its timings and energy samples
do not establish a controlled latency or energy result. The paired WER gate
compares quality only. This phone result does not promote the recipe to a
release default.

## Phone recipes (Pixel 10 Pro, 2026-10)

Goal: every model usable on the phone, with faster transcription, lower
energy, and smaller files. Quality is measured on the host CPU; speed,
memory and energy on the phone.

**Quality gate.** FLEURS test, 100 en + 100 de clips. Each GGUF is
decoded in a fresh process (`benchmarks/quant_phone/wer_models.py`).
Against the source (or an all-Q8_0 reference when the source is too large
for CPU), a recipe passes when, in both languages:

- the corpus WER delta is at most +0.3 pt, and
- the upper bound of the paired per-clip bootstrap 95 % CI is at most
  +1.0 pt (`compare_wer.py`: 10k resamples, seed 0).

Importance matrices use 160 FLEURS **validation** clips (100 en, 60 de).
Collect them on an all-Q8_0 file, not on the BF16 source. The collector
only records matmuls with F32 activations, and on a BF16 source the qwen
trunks feed BF16 activations to every LLM linear, so the matrix stays
nearly empty.

```bash
# 160 validation clips: export both languages, then merge the WAVs and the
# two refs.json maps into one directory (calib_en_de)
python benchmarks/fast_engine/export_fleurs.py --split validation --cfg en_us --n 100 --out calib_en
python benchmarks/fast_engine/export_fleurs.py --split validation --cfg de_de --n 60 --out calib_de
# all-Q8_0 file: every linear plus the embedding at Q8_0
printf 'default q8_0\n^llm\\.embed\\.weight$ q8_0\n' > /tmp/q8.recipe
build/starling-quantize --input models/qwen3-asr-0.6b-bf16-exact.gguf \
    --output models/qwen3-asr-0.6b-q8_0.gguf --recipe /tmp/q8.recipe
STARLING_IMATRIX=models/qwen3_06-calib160.imx STARLING_GGML_DEVICE=cpu \
    python benchmarks/quant_phone/wer_models.py --lib build/libstarling_ggml.so \
    --model qwen3_06 --clips calib_en_de --json /dev/null q8=models/qwen3-asr-0.6b-q8_0.gguf
```

The matrix is written when the process exits.

**Phone measurement.** `benchmarks/quant_phone/phone_quant_bench.sh` runs
on a host with adb. It starts each process only once the skin temperature
is back near idle. Variants alternate ABBA across rounds. Energy per
process is read from the coulomb counter on battery, minus the idle drain.

What makes a recipe fast on arm64:

- Q4_0 repacks to the i8mm 4x8 GEMM, Q4_K/Q5_K/Q6_K to 8x8, and Q8_0 to
  4x8. IQ4_NL gets only a dotprod 4x4 kernel. Q2_K/Q3_K/IQ2/IQ3 are not
  repacked.
- K-quants need a row width that is a multiple of 256; narrower rows fall
  back to Q8_0. Use Q4_0 for 640- or 896-wide rows.
- The phone's Vulkan fast engine maps Q4_0/Q8_0 onto its W4/W8 layouts
  losslessly and re-rounds K-quants.

### Parakeet TDT 0.6B v3 — `parakeet-q4pw-fullimx.recipe`

All linears are Q4_0, including the pointwise convs (opt-in, flattened).
The joint, the LSTM and the prediction embedding are Q8_0. Matrix:
`parakeet-full.imx`. Built with `--shrink-f16` from the F32 source.

| variant | MB | en WER | de WER | paired en Δ [95 % CI] | paired de Δ [95 % CI] |
|---------|---:|-------:|-------:|----------------------:|----------------------:|
| F32 source | 2357 | 5.61 | 4.68 | — | — |
| q4_k_m-shrink16 (today's download) | 553 | 5.42 | 4.72 | | |
| **q4pw** | **364** | **5.33** | **4.46** | −0.31 [−0.80, +0.13] | −0.20 [−0.60, +0.19] |
| q4pw, joint/LSTM also Q4_0 | 358 | 5.80 | 5.06 | | fails |

Pixel 10 Pro, q4pw vs q4_k_m-shrink16 (medians over ABBA rounds):

| engine | metric | q4_k_m-shrink16 | q4pw |
|--------|--------|----------------:|-----:|
| fast Vulkan | load | 2.30 s | 1.88 s |
| fast Vulkan | medium / long fixture | 2.26 / 7.69 s | 2.06 / 6.91 s |
| fast Vulkan | peak RSS | 1180 MB | 778 MB |
| fast Vulkan | energy per process | 25.8 mWh | 21.9 mWh |
| ggml CPU | peak RSS | 728 MB | 548 MB |
| ggml CPU | CPU time | 192 s | 173 s |

The fixture transcripts are identical. On the notebook's Radeon (RADV),
transcripts are also identical and latency is within 8 %. The catalog
entry is `parakeet-q4pw-phone`.

### The LLM-decoder models

Every finalist below passes the gate against its reference. The catalog
id is the recipe name plus `-phone`, except for the two ARK-3B entries. The
phone-sized layout differs per model:

- **Qwen3-ASR:** the decoder needs Q5_K/Q6_K. Every decoder at about
  4.5 bpw (Q4_K, Q4_0, or Q4_K with Q5_K/Q6_K bumps) failed German by
  +0.5 to +1.5 pt.
- **ARK-0.6B and Granite:** they hold up at 4 bit. ARK-0.6B needs its
  decoder attention v/o and ffn.down at Q8_0.

"MB/token" estimates the bytes read per decoded token: the decoder
linears plus the head.

| model | recipe | MB | MB/token | reference | en WER (paired Δ [95 % CI]) | de WER (paired Δ [95 % CI]) |
|-------|--------|---:|---------:|-----------|-----------------------------|-----------------------------|
| Qwen3-ASR-0.6B | `qwen3-0.6b-k56x8e6` | 663 | 454 | BF16 5.23 / 6.59 | 5.04 (−0.02 [−0.61, +0.57]) | 6.59 (−0.01 [−0.74, +0.64]) |
| Qwen3-ASR-0.6B | `qwen3-0.6b-k6x8e8` | 736 | 527 | BF16 | 5.04 (−0.04 [−0.58, +0.50]) | 6.72 (+0.09 [−0.30, +0.52]) |
| Qwen3-ASR-1.7B | `qwen3-1.7b-k56x6e6` | 1567 | 1294 | all-Q8_0 3.91 / 3.91 | 3.77 (−0.13 [−0.43, +0.17]) | 3.91 (+0.04 [−0.42, +0.50]) |
| Qwen3-ASR-1.7B | `qwen3-1.7b-k6x8e8` | 1835 | 1487 | all-Q8_0 | 3.77 (−0.10 [−0.52, +0.24]) | 3.61 (−0.20 [−0.55, +0.13]) |
| ARK-ASR-0.6B | `ark06-kds8e4` | 732 | 347 | BF16 4.81 / 6.97 | 4.90 (+0.08 [−0.29, +0.40]) | 6.85 (−0.36 [−1.44, +0.40]) |
| ARK-ASR-0.6B | `ark06-kds8` | 806 | 421 | BF16 | 4.86 (+0.06 [−0.11, +0.26]) | 6.63 (−0.46 [−1.54, +0.31]) |
| ARK-ASR-3B | `ark-3b-q4k` | 2125 | 1736 | BF16 6.13 / — | 6.36 (+0.32 [−0.11, +0.82]) | not gateable |
| ARK-ASR-3B | `ark-3b-q5k` | 2635 | 2163 | BF16 | 6.08 (−0.05 [−0.30, +0.16]) | not gateable |
| Granite-Speech-4.1-2B | `granite-q40-h4` | 1472 | 918 | all-Q8_0 6.18 / 5.78 | 5.52 (−0.43 [−1.16, +0.24]) | 5.82 (+0.01 [−0.54, +0.62]) |
| Granite-Speech-4.1-2B | `granite-q4k-h4` | 1472 | 918 | all-Q8_0 | 6.22 (+0.11 [−0.45, +0.76]) | 5.82 (−0.28 [−1.03, +0.37]) |
| MOSS-Transcribe-2B | `moss-q4e4-cal` | 1397 | 968 | BF16 8.11 / — | 8.02 (−0.18 [−1.11, +0.75]) | not gateable |
| MOSS-Transcribe-2B | `moss-q4e8-fullimx` (existing) | 1553 | 1123 | BF16 | 7.59 (−0.68 [−1.75, +0.31]) | not gateable |

- **German is not gateable for MOSS and ARK-3B.** Both models answer many
  German clips in English, even at BF16. MOSS scores about 100 % WER on
  German. ARK-3B answers about 46 % of German clips in English (de WER
  ≈ 49 %).
  - On the clips where both files decode German, `ark-3b-q5k` stays as
    close to BF16 as Q8_0 does.
  - On the same clips, `ark-3b-q4k` loses +1.27 pt [+0.09, +2.66].
- **Narrow rows.** The Qwen3-0.6B encoder and the ARK-0.6B decoder are
  896 wide, so K-quants do not apply there. Q4_0 on the Qwen3-0.6B encoder
  cost +0.9 pt in German.
- **Tied heads.** Q4_0 heads are cheap only with a matrix that covers the
  head.
  - With the FLEURS matrix, the MOSS Q4_0 head is +0.05 pt against Q8_0.
  - `moss-full.imx` has no entry for the head.
  - On ARK-0.6B a Q4_0 head cost 0.1–0.6 pt in every screened pair, and
    `ark06-kds8e4` still passes.
- **Granite.** The two recipes have identical bytes and both pass. Pick the
  one whose kernel is faster on the phone (Q4_0 4x8 or Q4_K 8x8). On a
  30-clip German BF16 spot check, Q4_K sat closer to BF16.
- **Loops.** One ARK-3B Q4_K_M file looped on one English clip and scored
  15 % WER. Screen for repetition loops, not only the mean.
- **Vulkan.** On the notebook's Radeon (RADV), the Qwen3, Granite and MOSS
  finalists give the same fixture transcripts as on its CPU.
- **Fast engine.** The MOSS finalists use only Q4_0 and Q8_0, which the
  fast engine maps losslessly. It stores `enc.conv_out` as F16 whatever
  its GGUF type, so quantizing that tensor shrinks only the download.
- **Reproducibility.** Rebuilding from these recipes with the recorded
  matrices reproduces the evaluated tensors bit for bit.

## Results (parakeet-tdt-0.6b-v3, LibriSpeech fixtures)

The historical numbers below use the CPU path. The dated Vulkan
retest below records the failures in that build; Vulkan corpus parity remains
unvalidated. The fixtures repeat one utterance, so read the deltas against the
f32 row, not as leaderboard WERs. The imatrix was collected over the same three fixtures (single speaker,
~1.5 min audio — deliberately minimal; real calibration would use an hour of
diverse audio, e.g. Granary-derived clips).

### Vulkan fast-path retest (post-#47, 2026-09-05)

This historical retest used the merged-#39 build, including #47's galloc
INPUT-reuse patch. F32 showed premature termination with a correct prefix.

Later changes gave each model ownership of its replay caches and freed graphs
before weights. The current [engine validation record](ggml-engine.md)
also reports two consecutive Q8_0 short-fixture calls on CPU and Vulkan.
That limited smoke check does not establish GPU parity for these corpus sweeps.

- Fixtures, f32, same binary, one model per process: CPU
  `0.00/0.00/0.00` vs `STARLING_GGML_DEVICE=Vulkan0`
  `60.87/86.96/0.00` (short/medium/long). Single-model runs sidestep the
  replay-cache hazard present in that build.
- Repetition sweep (base utterance 7.4 s, k× no gap): k=1–5 emit only the
  opening phrase ("Well, I don't wish to see it any more,") and stop — the
  WERs reconcile exactly as deletions (short 13/21 words → 60.87, medium
  55/63 → 86.96). k=6 (44.6 s) through 10: complete, correct transcripts
  (long = WER 0.00).
- Real clips (fleurs_test bg_bg, 8–19 s, 8/8 truncated): correct-prefix
  truncation after ~5–8 words, WER 72–89 — not a repeated-utterance
  artifact.
- Prefix sweep (1–7.4 s of the base): partial audio yields correct partial
  transcripts — encoder/weights compute fine; only the stop decision is
  wrong.

In that build, single-window decode (input below ~40 s, bracketed 37.2–44.6 s)
stopped after the first K-step batch; longer audio used the segmented/composite
path and completed. GPU sweeps and ladder timings still need validation on the
current build. The benchmark drivers retain CPU as the measured default.

Clean audio — every level down to q2_k is lossless here:

| model   |   MB | wer_short | wer_medium | wer_long |
|---------|------|-----------|------------|----------|
| f32     | 2508 | 0.00      | 0.00       | 0.00     |
| q8_0    |  906 | 0.00      | 0.00       | 0.00     |
| q6_k    |  777 | 0.00      | 0.00       | 0.00     |
| q5_k_m  |  740 | 0.00      | 0.00       | 0.00     |
| q4_k_m  |  704 | 0.00      | 0.00       | 0.00     |
| q4_k_s  |  639 | 0.00      | 0.00       | 0.00     |

5 dB gaussian noise (`--snr-db 5`) — where calibration earns its keep:

| model         |   MB | wer_short | wer_medium |
|---------------|------|-----------|------------|
| f32           | 2508 | 8.70      | 0.00       |
| q8_0          |  906 | 8.70      | 0.00       |
| q4_k_m        |  704 | 8.70      | 0.00       |
| q3_k_m (unif) |  634 | 0.00*     | 8.70       |
| q3_k_m +imx   |  634 | 8.70      | 2.90       |
| q2_k (unif)   |  574 | 30.43     | 17.39      |
| q2_k +imx     |  574 | **8.70**  | **0.00**   |
| recipe attn3  |  522 | 8.70      | 8.70       |

(*) a different word than the f32 row errs — sample noise at this fixture
size, not a level effect.

The size floor (`--shrink-f16` stores the ~300 MB of conv weights at F16;
everything below ~2.1 bpw is IQ territory and REQUIRES the imatrix):

| model                  |   MB | % of F32 | clean s/m | 5 dB s/m |
|------------------------|------|----------|-----------|----------|
| q2_k +imx              |  574 |    23%   | 0.0 / 0.0 | 8.7 / 8.7|
| iq2_xs +imx            |  494 |    20%   | —         | 0.0 / 8.7|
| iq2_xxs +imx           |  477 |    19%   | —         | 0.0 / 8.7|
| iq1_m +imx             |  456 |    18%   | 0.0 / 10.1| 13.0 / 17.4|
| **iq2_xxs +imx +shrink** | **325** | **13%** | **0.0 / 0.0** | 0.0 / 8.7 |
| iq1_s +imx +shrink     |  292 |    12%   | —         | 26.1 (cliff) |

The usable floor for this model is **iq2_xxs + imatrix + conv-shrink at
325 MB (13% of F32)**, which stays within single-word noise of F32 on both
clean and 5 dB-noised fixtures. IQ1 territory (≈1.6–1.75 bpw encoder) starts
dropping words even on clean audio — the cliff, measured.

## German (MLS de test, 40 clips / ~10 min, human transcripts)

v3 is multilingual, so a quant level must hold German too. f32's baseline on
these audiobook clips is 13.0% WER; the table shows the calibration-mix
effect at the low end (imx = English fixtures only; imx2 = fixtures + the
same 40 German clips):

| model                  | wer_mls_de | note |
|------------------------|------------|------|
| f32                    | 13.02      | baseline |
| q4_k_m                 | 12.72      | holds German |
| q2_k +imx (EN-only)    | 15.30      | +2.3 over f32 |
| q2_k +imx2 (EN+DE)     | 15.78      | mix barely matters here |
| iq2_xxs +imx +shrink   | 29.38      | EN-only calibration breaks DE |
| iq2_xxs +imx2 +shrink  | **20.38**  | +9 pts recovered by DE clips |

Two lessons: (1) at 2 bits, **calibration-data diversity is part of quality**
— English-only importance matrices silently cost 16 German WER points, and
adding ten minutes of German audio recovered nine of them (English fixtures
stayed perfect throughout); (2) the remaining gap to f32 at iq2_xxs is
NOT calibration-hungry — the volume study below (and the kitchen-sink run)
show more audio does not close it; Granary-scale audio buys language
COVERAGE, not volume. This is the miniaturized version
of exactly the multilingual-calibration story Granary exists for.

## Language-coverage dynamics (FLEURS, controlled experiment)

How much does the calibration language mix matter, and does covering all 25
languages cost English? Three imatrices with EQUAL per-language budgets
(12 FLEURS train clips each: English-only, English+German, all 25 languages
— 197,830 observations for the 25-language matrix), evaluated on FLEURS test
clips the calibration never touched (en/de: 50 clips, others: 8). All
absolute numbers are FLEURS-domain with jiwer normalization — the f32 row is
the reference, not a leaderboard.

25-language mean WER (Δ vs the f32 mean of 13.90):

| calibration → | EN only | EN + DE | all 25 |
|---------------|---------|---------|--------|
| q4_k_m        | 13.79 (−0.11) — flat regardless |
| q2_k          | 16.96 (+3.06) | 16.70 (+2.80) | **16.47 (+2.57)** |
| iq2_xxs+shrink| 26.40 (+12.50) | 25.61 (+11.71) | **25.18 (+11.28)** |

English and German specifically:

| model        | calib | wer_en | Δen  | wer_de | Δde  |
|--------------|-------|--------|------|--------|------|
| iq2_xxs+shrk | EN    | 6.94   | +0.51| 8.79   | **+3.74** |
| iq2_xxs+shrk | EN+DE | 7.12   | +0.69| 7.72   | +2.67 |
| iq2_xxs+shrk | 25    | 7.46   | +1.03| 7.93   | +2.88 |
| q2_k         | 25    | 6.37   | −0.06| 6.08   | +1.03 |

Takeaways:

- **Coverage ordering is real and monotone**: the 25-language mean improves
  EN-only → EN+DE → all-25 at both bit levels. Small per step (~0.3–0.8),
  consistent everywhere.
- **English pays essentially nothing for full coverage** (≤ +1.0 WER at the
  worst level, within 50-clip noise). There is no English-vs-coverage
  tradeoff here: ship the 25-language-calibrated quant.
- **German wants to be in the calibration set** at iq2_xxs (+3.74 EN-only
  vs +2.7–2.9 included) per these 50-clip reads — superseded by the
  300-clip CI re-run below, which found the calibration mix not measurable.
- **2 bits are an EN/major-language trade**: the iq2_xxs mean gap is carried
  by the tail languages — Lithuanian/Latvian/Slovenian/Romanian/Finnish/
  Swedish degrade +14–29 points over f32 with ANY calibration (the 25-lang
  matrix trims only 2–4 of those points). Those languages need q2_k (+2.6
  mean, evenly spread) or q4_k_m (free). For an EN+DE deployment, iq2_xxs
  at 325 MB stays within ~+1/+3 points of f32 on exactly those languages.

Caveats: 8 test clips per non-EN/DE language (per-language σ ≈ ±5 points —
read the means and the en/de columns, not single cells), and the imatrix
budgets are minutes-per-language, far below what Granary-scale audio would
give the 25-language matrix.

## Does more calibration data help? (volume saturation)

No — not at fixed coverage, for this model. Scaling the EN+DE calibration
from 12 to 60 clips per language (15k → 76k observations, same split and
balance) moved nothing beyond eval noise:

| model   | calib volume | wer_de | wer_en |
|---------|--------------|--------|--------|
| q2_k    | 12/lang      | 6.78   | 6.11   |
| q2_k    | 60/lang      | 7.13   | 5.70   |
| iq2_xxs | 12/lang      | 7.72   | 7.12   |
| iq2_xxs | 60/lang      | 8.43   | 7.25   |

(f32 references: DE 5.05, EN 6.43; 50-clip evals, σ ≈ ±0.9.)

The mechanism is visible in the matrices themselves: the per-channel
importance vectors of the 12-clip and 60-clip imatrices agree at **0.993
mean cosine across all 275 tensors** — the activation statistics saturate at
roughly a dozen clips per language, so the quantizer makes the same block-
scale decisions either way. A kitchen-sink matrix over EVERY training clip
on disk (1,224 FLEURS wavs across three
corpora, plus LibriSpeech fixtures and MLS German audiobooks as extra
domains; 838k observations) confirms
the same: q2_k 6.06/6.50 and iq2_xxs 9.20/8.41 DE/EN vs the production
matrix's 6.03/6.36 and 9.38/8.39, all within noise. The remaining low-bit gaps (DE ≈ +2.7 at
iq2_xxs, ≈ +1.7–2.1 at q2_k per this table's rows) are the inherent error of that bit width for those
languages, not a calibration-data deficit. What does move them: more bits
(q4_k_m is free), or *different* data — calibration audio matched to the
deployment domain, since FLEURS's read-speech distribution is itself part of
the residual gap. By the same saturation argument, the tail languages'
+14–29-point iq2_xxs degradation is bit-width-driven too; the remedy is the
quant level, not more calibration clips.

## Confidence-interval re-run (300 EN/DE clips, 48 per tail language)

The tables above use 50-clip EN/DE and 8-clip-per-language cells, so
sub-point differences there are noise (σ ≈ ±0.9 and ±5 respectively). This
re-run on 300/48 clips with bootstrap CIs confirms the big claims and
corrects two small ones. (The removed `wer_quant.py` reported mean [95% CI]
whenever a column had ≥5 clips.)

EN/DE at 300 clips (f32: DE 5.30 [4.45–6.14], EN 6.50 [5.59–7.44]):

| model        | wer_de                  | wer_en                  |
|--------------|-------------------------|-------------------------|
| q4_k_m       | 5.32 [4.51–6.15]        | 6.57 [5.70–7.54]        |
| q2_k +imx25  | 5.99 [5.04–6.90]        | 6.39 [5.56–7.31]        |
| q2_k +imxENDE| 6.55 [5.65–7.55]        | 6.15 [5.33–7.03]        |
| iq2xxs +imxEN| 9.45 [8.29–10.66]       | 8.11 [7.15–9.24]        |
| iq2xxs +imxENDE | 9.20 [8.08–10.34]    | 7.77 [6.83–8.81]        |
| iq2xxs +imx25| 9.30 [8.14–10.49]       | 8.38 [7.36–9.57]        |

Results from the larger evaluation:

- **q4_k_m differs by at most 0.07 WER percentage points** in these two
  language samples. This does not establish lossless quantization for other
  inputs or languages.
- **q2_k: English free, German +0.7–1.25**, tail languages +4–7 on the worst
  (lt 22.5→29.1 [26.1–32.2], sl →30.2, ro →17.5 — CIs separate from f32).
  25-language mean 16.90 vs f32's 14.00 (+2.90 at 48 clips/lang, matching
  the +2.57 the 8-clip sample estimated).
- **iq2_xxs is decisively an EN/DE trade**: German +4.0 (bigger than the
  50-clip sample suggested — small samples biased toward easy clips), and
  the tail is catastrophic with CIs fully separated from f32: sl 24.9→53.4
  [47.0–59.9], lv →47.1, lt →45.5, mt →46.1, hu →35.6, sk →27.6 (25-lang
  mean 26.39 vs 14.00).

What gets **corrected**: the 50-clip read that "German wants to be in the
calibration set" (+1 pt at iq2_xxs) does not survive — EN-only / EN+DE /
all-25 land at 9.45 / 9.20 / 9.30 German with fully overlapping CIs, and
English likewise. At these bit widths the calibration LANGUAGE MIX for
EN/DE is not a measurable lever once the matrix exists at all; the effects
that matter are calibrated-vs-uniform (huge) and the bit width itself.

## Extending to the other engines

The quantizer is name-agnostic (2-D linears minus the keep-list). What a new
model family needs:

1. its loader must tolerate quantized dtypes on the tensors it `mul_mat`s
   (parakeet needed nothing — `ModelLoader` is dtype-agnostic and the graphs
   never cast those weights), and
2. its host-read tensors (embeddings, folded norms) added to the keep rules
   in `cpp/tools/starling_quantize.cpp` if their names do not already match.

Two engine-side rules came out of the first non-parakeet port (an engine
that has since been removed) and apply to every quantized family:

- **Loader allowlist + profile stamp**: the loader's header guard must accept
  `"quantized"`, and the quantizer stamps `starling.numeric_profile=quantized`
  when it actually quantized at least one tensor, instead of inheriting the
  source's exact-profile string (a degenerate all-kept run is numerically
  identical to the source, so it keeps the inherited profile).
- **F32 activations into quantized weights**: ggml's quantized matmul asserts
  F32 activations (`GGML_ASSERT(src1->type == GGML_TYPE_F32)` in the quantized
  branch of `ggml_compute_forward_mul_mat`,
  `third_party/ggml/src/ggml-cpu/ggml-cpu.c`), and the qwen-trunk graphs feed
  **bf16** activations into every weight GEMM. `gemm_act()`
  (graph_helpers.hpp) upcasts the activation to F32 only when the weight is
  block-quantized — bf16 is exact in f32, so the F32-accumulated GEMM and
  trailing bf16 round preserve the numerical boundary and only the weight
  carries quantization noise; unquantized weights keep the identical bf16
  path. `lin()`/`linear_bf16()` and the lm_head GEMMs (`lm_head_gemm()`)
  route through it. Conv weights stay kept: an im2col GEMM that takes its
  activation straight from an F32 buffer with no `gemm_act` would fail at
  eval time (the im2col/quantized matmul assert), not at load.

### Moss-transcribe-2b port (2026-09-10) — quants + Vulkan/CPU engine work

Executed against the bf16_exact base
(`models/moss-transcribe-preview-2b-bf16-exact.gguf`, 4845 MB). All big
linears have row widths divisible by 32 and 256, so no block-size fallbacks
(unlike parakeet's 640-row linears). Three engine fixes were load-bearing:

1. Loader allowlist: `cpp/moss/loader.cpp` gains `"quantized"`.
2. Adapter `mul_mat` passed BF16 activations straight into (now quantized)
   weights, aborting in ggml-vulkan's `q_f16` path. Fix: route both adapter
   linears (`cpp/moss/adapter.cpp`, fused `build_adapter`) through the
   existing `gemm_act()` rule (F32 activations against quantized weights;
   exact upcast, weight-only noise). Same latent pattern to audit in other
   engines when they quantize.
3. `starling-quantize` grew `--f32-1d` (kept 1-D norms/biases stored as F32:
   bit-identical values, but no CAST graph nodes, so Vulkan's
   `{NORM,MUL,ADD}` / `{RMS_NORM,MUL}` fusions see consecutive patterns)
   and a moss-scoped explicit embed rule (`^llm.embed.weight$`, q8_0 or —
   experimental — q4_0; every other family keeps the parakeet-only gate).

Imatrix collection rides the stock collector (`STARLING_IMATRIX` +
`STARLING_GGML_DEVICE=cpu`), but on the **Q8_0 model**: the collector only
observes F32 activations and the BF16 model's GEMMs take BF16, while Q8's
`gemm_act` routes F32 everywhere. Activation statistics are
weight-precision agnostic. One pass over fixtures + 32 real-corpus clips:
394 tensors, ~196k observations. The one-off driver, `.auto/collect_moss_imx.py`,
was a local session artifact and is not included in this repository. These
collection counts record that experiment; the committed recipes require the
resulting imatrix file or a new collection pass over the same Q8_0 model.

Recipes (`quants/recipes/moss-*.recipe`, all with `--f32-1d` where noted):

| recipe | linears | embed (tied head) | size | note |
|--------|---------|-------------------|------|------|
| `moss-q4-fullimx` | q4_0 + imx | BF16 exact | 1838 MB | best single-model speed pre-head-quant |
| `moss-q4e8-fullimx` | q4_0 + imx | q8_0 | 1546 MB | recommended if Q2 tail unacceptable |
| `moss-q4e4-fullimx` | q4_0 + imx | q4_0 | 1391 MB | safe pick: WER-identical, 28/32 fidelity |
| `moss-q2e4-fullimx` | q2_k + imx | q4_0 | 900 MB | fastest (-15% vs Q4): short/med WER 0.00, noisier tail |

(Q8_0 uniform, no imatrix, was the stepping stone: 2883 MB. Sub-Q2 linears
(IQ2_XXS: speed ties, CER 0.11 word errors) and sub-Q4 heads (Q3: tie with
worse tail; Q2: faster but CER 0.06 with real errors) were tried and
rejected — Q2_K is the linear floor, Q4 the head floor.)

Fixture WER vs ground truth (`benchmarks/fast_engine/wer.py` LibriSpeech refs), Vulkan:

| model | short | medium | long |
|-------|-------|--------|------|
| bf16_exact | 0.00% | 0.00% | 40.00% |
| q8_0 | 0.00% | 0.00% | 40.00% |
| q4_0 + imx | 0.00% | 0.00% | 40.00% |
| q4_0 + q8 head | 0.00% | 0.00% | 40.00% |
| q4_0 + q4 head | 0.00% | 0.00% | 40.00% |
| q2_k + q4 head | 0.00% | 0.00% | 90.00%* |

The long-tier 40% is a tiled-audio repetition artifact (the model loses the
10x repetition count), identical across all five artifacts — no quant
regression. (*) The Q2 model early-stops after 1 repetition on exact-repeat
tiles (long WER 90): repetition-specific, verified deterministic and
order-independent (a stale-script false alarm about nondeterminism was
chased down to a `sed` chain drift). A 31 s diverse-concatenation clip is
covered fully (346 vs 350 chars, CER 0.04), and short/med are exact — but
the tail is noisier generally (see fidelity below), so Q2 needs leaderboard
WER before any release claim. 32-clip real-corpus fidelity q4-head vs q8-head: 28/32
byte-identical transcripts, mean normalized CER 0.0035 (the corpus's own
`reference.json` is index-mismatched — even bf16_exact "hallucinates"
against it — so fidelity, not accuracy, is the gate there; full Open-ASR
WER before any release claim). Q2-head fidelity vs q8-head: 16/32 identical,
mean CER 0.03, max 0.20 — single-word flips on hard audio; acceptable as an
experimental trade for -15%/-31% but the reason Q4 stays the safe pick.

Speed interleaves quant + engine work (this box: RADV RENOIR uma iGPU +
6C/12T Zen3; short fixture, steady-state median, RTF in parens):

| step | Vulkan ms | CPU ms |
|------|-----------|--------|
| bf16_exact | 5328 (1.40x) | 37857 (0.20x) |
| + q8_0 | 3205 (2.32x) | 4285 (1.74x) |
| + q4_0 + imx | 2817 (2.64x) | 3882 (1.92x) |
| + embed q8_0 | 2614 (2.85x) | 3689 (2.02x) |
| + F32 K-step decode + rope_ext | 2418 (3.08x) | — |
| + bucketed K-step attention | 1985 (3.75x) | — |
| + F32 prefill / encoder | 1874 (3.97x) | — |
| + q4 head | 1793 (4.15x) | 3165 (2.35x) |
| + physical-core thread default | — (neutral) | 3381→3150* |
| + q2_k linears (Q4 head kept) | 1532 (4.85x) | 2507 (2.97x) |

(*) thread default measured on the q4e8 model; q4-head CPU re-measured after.
Peak RSS (Vulkan): 4765 -> 2932 -> 1930 -> 1657 -> 1644 MB (q4-head: 1504).
Medium/long Vulkan: ~19 s -> 5.3 s / ~55 s -> 16.2 s (long varies run to
run; short is the stable gate).

Engine notes (all auto-gated: GPU + quantized linears, CPU/BF16 keep the
exact discipline; CER-gated per run): `f32_acts` decode (skipped bf16
round-trips unlock 452 Vulkan `RMS_NORM_MUL[_ROPE]` fusions per K-step
replay; attention core stays BF16 to avoid full-cache F32 traffic),
bucketed exact-width K-step attention (128/256/512/1024 prefix views +
runtime masks; `STARLING_MOSS_NOBUCKET` forces full-cap), `rope_ext`
(NEOX = half-rotation, matches the engine formula), F32 prefill/encoder,
physical-core CPU thread default (`STARLING_GGML_THREADS` still wins).
Dead ends kept for the record: KSTEP=8 (noise, then +7.6% slower),
`GGML_VK_FLOPS_PER_SUBMIT` one-submit (neutral), `GGML_VK_NO_BARRIERS`
(4x slower + garbage — never), IQ4_NL (ties Q4_0), F32-on-CPU and CPU
K-step (correct, zero gain — CPU is GEMV-compute-bound at ~82 ms/tok).

## Ours vs the community quants (matched levels, 300-clip EN/DE with CIs)

Head-to-head against handy-computer's transcribe.cpp-dialect ladder through
the compat layer — same engine, same eval, so it compares the pipelines
(converter + calibration + tensor policy) end to end:

| level | handy MB | ours MB | EN / DE verdict |
|-------|----------|---------|-----------------|
| baseline | 1256 (F16) | 2508 (F32) | identical (6.48/5.30 vs 6.50/5.30) — converter + compat parity |
| q8_0   | 740 | 906 | identical |
| q6_k   | 610 | 777 | identical |
| q5_k_m | 549 | 740 | identical |
| q4_k_m | 485 | 704 | identical (ours 6.41/5.31, handy 6.62/5.13 — CIs overlap fully) |
| q4_k_m + shrink16 | — | 553 | identical (EN 6.46 [5.63–7.35] / DE 5.31 [4.55–6.15]) |

(The `ours` column measures the released artifacts, quantized with the
production imatrix; the CI re-run table's q4_k_m row measured the earlier
uniform-build research file — both f32-equivalent, hence the small EN/DE
wiggles between the tables.)

Their files are smaller at the same level name because their policy also
quantizes the embedding and keeps convs F16; `--shrink-f16` closes most of
that gap; the remainder is mostly the `_M` bump mixes (Q6_K attention /
down-projection groups vs their uniform Q4) plus the exactness-motivated
F32 embedding (~18 MB). Tail
languages at q4_k_m (48 clips/lang): ours 14.15 vs handy 14.62 vs f32 14.00
mean — a small consistent edge (sk 12.2→10.4, cs 13.4→11.5), borderline
individual noise.

**Verdict**: at ≥ Q4 both pipelines sit at the level's quality ceiling —
parity, not improvement. Our ladder's value is everything BELOW Q4, which
the community repos don't ship: q3_k_m/q2_k/iq2_xxs (634/574/325 MB) exist
only calibrated (uniform q2_k collapses to 30%/17% where the calibrated
build matches f32), plus the published imatrix for custom recipes.

Tooling note: install the corpus dependencies with
`uv sync --locked --group eval` before using `benchmarks/fleurs_download.py`.
Use `uv run --group eval python benchmarks/fleurs_download.py --help` for options.
It fetches FLEURS corpora over HTTP range reads on the parquet shards (~100 MB per config instead of the
2 GB shard, resumable per config) and `imatrix_collect.py --wavs` feeds the local
clips to collection — the datasets-library streaming path accumulates
multi-GB per config and stalled repeatedly next to a loaded model.

## Speed (CPU, medium fixture, 3-run mean after warmup)

| build | RTFx | note |
|-------|------|------|
| f32 (2508 MB) | 12.9× | |
| q4_k_m (704 MB) | 11.9× | −8% |
| q2_k (574 MB) | 14.2× | +10% |
| q3_k_m (634 MB) | 11.1× | between q4 and iq2, as expected |
| iq2_xxs+shrink16 (325 MB) | 7.5× | −42% (IQ vec-dot kernels are CPU-compute-bound) |

The quants' value is size/VRAM, not CPU speed: k-quants are roughly
speed-neutral on this encoder, and the IQ formats trade inference speed for
bytes. When both matter, q3_k_m is the measured middle (11.1×). GPU behavior may
differ (bandwidth-bound, mmvq paths) — unmeasurable on this hardware while the
Vulkan fast-path truncation bug stands (see the post-#47 retest note above).
