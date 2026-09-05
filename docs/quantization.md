# Calibrated GGUF quantization

Starling ships its own quantization pipeline for the native ggml engines —
the ASR analogue of the "dynamic quants" the text-LLM world has: block
quantization (Q8_0 / K-quants) weighted by an **importance matrix** collected
on real audio, plus per-tensor recipes for sensitivity sweeps, verified by
WER. Parakeet-tdt-0.6b-v3 is the proof of concept; the same tooling is
model-agnostic and intended for the larger engines (audex-2b,
moss-transcribe-2b, qwen3-asr-1.7b) where halving VRAM actually changes what
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
                                                   benchmarks/wer_quant.py
```

1. **Convert** to a numeric-exact base GGUF (F32 is the quantization input,
   matching llama.cpp practice — quantize once from exact weights, never from
   an already-rounded dtype):

   ```bash
   uv run --with gguf python scripts/convert_parakeet_gguf.py \
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
   uv run python benchmarks/imatrix_collect.py \
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

4. **Verify** by WER against the f32 baseline on the fixtures (and, for
   release gates, the full Open-ASR-Leaderboard sets via
   `benchmarks/wer_leaderboard.py` with `STARLING_GGML_PARAKEET_MODEL`
   pointing at the quantized file):

   ```bash
   uv run python benchmarks/wer_quant.py --tiers short,medium,long \
       --models f32=models/parakeet-tdt-0.6b-v3-f32.gguf \
                q8_0=models/parakeet-tdt-0.6b-v3-q8_0.gguf \
                q4_k_m+imx=models/parakeet-tdt-0.6b-v3-q4_k_m.gguf
   ```

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

- **conv weights** (`conv` in the name): the conformer pointwise convs are
  `ggml_cast` to F16 and reshaped before their matmul, the depthwise/subsampling
  convs go through `ggml_conv_*` — no dequant path.
- **the prediction embedding** `decoder.prediction.embed.weight`: read as a
  raw F32 host table (`prediction.cpp`), not a matmul.
- **batch-norm statistics, norms, biases, pos_bias_u/v**: 1-D host-folded or
  broadcast operands.
- **mel constants** (`preprocessor.*`): the filterbank/window must stay exact.

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
parakeet) as F16 — they are consumed through F16 paths anyway (the pointwise
convs are `ggml_cast` to F16, the depthwise/subsampling convs take F16
kernels). Everything else stays F32: 1-D biases/norms/BN statistics feed
`ggml_add`/`ggml_mul` broadcasts which reject mixed dtypes.

## Results (parakeet-tdt-0.6b-v3, LibriSpeech fixtures)

All numbers from `benchmarks/wer_quant.py` on the CPU path (the Vulkan fast
path has a known output discrepancy on this RADV iGPU independent of
quantization — F32 degrades there too; see the PR notes). The fixtures repeat
one utterance, so read the deltas against the f32 row, not as leaderboard
WERs. The imatrix was collected over the same three fixtures (single speaker,
~1.5 min audio — deliberately minimal; real calibration would use an hour of
diverse audio, e.g. Granary-derived clips).

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
corrects two small ones. (`wer_quant.py` now reports mean [95% CI] whenever
a column has ≥5 clips.)

EN/DE at 300 clips (f32: DE 5.30 [4.45–6.14], EN 6.50 [5.59–7.44]):

| model        | wer_de                  | wer_en                  |
|--------------|-------------------------|-------------------------|
| q4_k_m       | 5.32 [4.51–6.15]        | 6.57 [5.70–7.54]        |
| q2_k +imx25  | 5.99 [5.04–6.90]        | 6.39 [5.56–7.31]        |
| q2_k +imxENDE| 6.55 [5.65–7.55]        | 6.15 [5.33–7.03]        |
| iq2xxs +imxEN| 9.45 [8.29–10.66]       | 8.11 [7.15–9.24]        |
| iq2xxs +imxENDE | 9.20 [8.08–10.34]    | 7.77 [6.83–8.81]        |
| iq2xxs +imx25| 9.30 [8.14–10.49]       | 8.38 [7.36–9.57]        |

What survives, now with statistical teeth:

- **q4_k_m is free** for both languages (Δ ≤ 0.07) — bulletproof.
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

Takeaways:

- **q4_k_m is free**: 704 MB (28% of F32) with zero measurable loss on noisy
  English AND German. This matches the community GGUF results for parakeet.
- **At 2 bits, calibration is decisive**: uniform q2_k breaks down
  (30%/17%) while the imatrix-weighted q2_k matches f32 exactly at the same
  574 MB, and the calibrated floor reaches 325 MB (13% of F32) before the
  IQ1 cliff.
- **At the floor, calibration data must be multilingual** (see the German
  table): an English-only imatrix cost 16 German WER points at iq2_xxs.

## Extending to the other engines

The quantizer is name-agnostic (2-D linears minus the keep-list). What a new
model family needs:

1. its loader must tolerate quantized dtypes on the tensors it `mul_mat`s
   (parakeet needed nothing — `ModelLoader` is dtype-agnostic and the graphs
   never cast those weights), and
2. its host-read tensors (embeddings, folded norms) added to the keep rules
   in `cpp/tools/starling_quantize.cpp` if their names do not already match.

Concrete next target — **audex-2b** (where 2-bit savings are worth ~4 GB):
the weights are `nvidia/Nemotron-Labs-Audex-2B` and the converter exists, but
the audex loader's header guard (`cpp/audex/loader.cpp`,
`check_gguf_header(..., {"bf16_exact"}, ...)`) rejects every other numeric
profile, so the steps are: convert an f32 base, extend the guard's allowlist
to the quantized profiles, audit `cpp/audex/` for conv/cast/host-read
tensors (same analysis parakeet got), then run this same sweep. moss-2b and
qwen3-asr-1.7b follow the same recipe.

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

Tooling note: `benchmarks/fleurs_download.py` fetches FLEURS corpora over
HTTP range reads on the parquet shards (~100 MB per config instead of the
2 GB shard, resumable per config) and `--wavs`/`--corpus` feed the local
clips to collection/eval — the datasets-library streaming path accumulates
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
differ (bandwidth-bound, mmvq paths) — untested on this hardware.
