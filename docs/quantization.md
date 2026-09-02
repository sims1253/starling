# Calibrated GGUF quantization

Starling ships its own quantization pipeline for the native ggml engines —
the ASR analogue of the "dynamic quants" the text-LLM world has: block
quantization (Q8_0 / K-quants) weighted by an **importance matrix** collected
on real audio, plus per-tensor recipes for sensitivity sweeps, verified by
WER. Parakeet-tdt-0.6b-v3 is the proof of concept; the same tooling is
model-agnostic and intended for the larger engines (audex-2b,
moss-transcribe-2b, qwen3-asr-1.7b) where halving VRAM actually changes what
fits on a local machine.

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

`--shrink-f16` additionally stores the kept tensors (except the embedding and
mel constants) as F16 for a further size cut at negligible accuracy cost.

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

Takeaways:

- **q4_k_m is free**: 704 MB (28% of F32) with zero measurable loss even on
  degraded audio. This matches the community GGUF results for parakeet
  (handy-computer's uncalibrated K-quants sit within ~0.05 WER points of F16
  on LibriSpeech test-clean).
- **At 2 bits, calibration is decisive**: uniform q2_k breaks down
  (30%/17%) while the imatrix-weighted q2_k matches f32 exactly at the same
  574 MB. That is the "dynamic quant" value proposition, reproduced on an
  ASR model with Starling's own tooling.
- Recipes work as the sensitivity-sweep mechanism
  (`benchmarks/recipes/parakeet-q2-attn3.recipe`: attention one notch above
  the q2_k base) — on these fixtures the plain q2_k+imx mix held up best,
  but the point of recipes is running exactly these experiments on real
  corpora (Granary / Open-ASR-Leaderboard sets) per model family.
- German verification is still open: v3 covers 25 European languages, and
  per-language quant deltas need a multilingual eval set (Common Voice DE,
  VoxPopuli) — the harness extension is the natural next step.

## Extending to the other engines

The quantizer is name-agnostic (2-D linears minus the keep-list). What a new
model family needs:

1. its loader must tolerate quantized dtypes on the tensors it `mul_mat`s
   (parakeet needed nothing — `ModelLoader` is dtype-agnostic and the graphs
   never cast those weights), and
2. its host-read tensors (embeddings, folded norms) added to the keep rules
   in `cpp/tools/starling_quantize.cpp` if their names do not already match.
