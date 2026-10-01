# Encoder K/V rank study (#59)

`bench_kv_spectral.py` captures the projections used by the Granite and
Qwen3-ASR encoders, fits per-head PCA on even-indexed audio clips, and measures
reconstruction on odd-indexed clips. It reports the rank required to explain
95%, 99%, and 99.9% of *training* variance alongside held-out relative squared
error at those ranks and at fixed quarter-, half-, and full-width ranks.
Separating clips matters: a small rank fitted and scored on the same audio
can hide a component that appears in another utterance.

For each head, held-out error is
`sum((K - reconstructed_K)^2) / sum((K - training_mean)^2)` (and likewise for
V). This ratio lies in [0, 1], apart from numerical roundoff, because PCA
projects onto an orthonormal basis. A distribution shift can move it toward
one. Full width should have
near-zero reconstruction error; it acts as a numerical control. Model loading,
projection hook locations, and audio preprocessing are in the
[`bench_kv_spectral.py`](bench_kv_spectral.py) source.

Run with at least two independent clips:

```sh
uv run python benchmarks/bench_kv_spectral.py --models granite --audio clip1.wav clip2.wav
```

## Granite run on public speech (2026-09-28)

The first real-model run used the [Granite Speech 4.1 2B source revision
`de575db6`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/tree/de575db64086f84fdc79da4932d1076e965bc546)
in BF16, with eight English [FLEURS train
clips](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd/parquet-data/en_us).
The clips were the first eight eligible rows in parquet order, exported as
16-bit PCM WAV; the export URL was unpinned, so the eight file hashes in
[`results/kv_spectral_granite_fleurs8.json`](results/kv_spectral_granite_fleurs8.json)
are the definitive input identity. Even-indexed clips (2,032 encoder frames)
fit the per-head PCA; odd-indexed clips (2,253 frames) were held out. These
are independent read-speech utterances, but one language and one dataset,
so they do not represent dictation or a cross-domain generalization test.

| Quantity, averaged over 16 layers and eight heads | K | V |
| --- | ---: | ---: |
| Rank / 128 at 95% **training** variance | 16.6% | 53.4% |
| Rank / 128 at 99% **training** variance | 24.6% | 73.4% |
| Rank / 128 at 99.9% **training** variance | 36.3% | 89.6% |
| Held-out relative squared error at fixed rank 32 | 0.0346 | 0.3399 |
| Held-out relative squared error at fixed rank 64 | 0.00244 | 0.14485 |
| Held-out relative squared error at full rank 128 | ~1e-9 | ~1e-9 |

At the training-derived 99% rank, layer-mean held-out error ranges from
0.007 to 0.017 for K and 0.015 to 0.050 for V. K is often compact in this
sample, but V is much less so: a shared rank-64 projection loses 14.5% of
held-out V energy on average. The full-rank control is near zero, as expected.
The detailed per-layer/per-head ranks, fixed-rank errors, input hashes and
train/held-out split are in the JSON artifact. This measures reconstruction
only; it does not measure attention-output error, WER or runtime performance.

Reproduce with a pinned Granite snapshot and the eight hash-matched WAVs:

```sh
PYTHONPATH=src python benchmarks/bench_kv_spectral.py --models granite \
  --granite-snapshot /path/to/de575db64086f84fdc79da4932d1076e965bc546 \
  --audio /path/to/en_us_train_{0,1,2,3,4,5,6,7}.wav
```

The original synthetic test still checks that a new held-out direction raises
the reported error while training rank stays unchanged. Qwen3-ASR has not
been measured here.

Even a low reconstruction error would only justify a native experiment. The
encoder path in [`cpp/granite/encoder.cpp`](../cpp/granite/encoder.cpp) builds
K and V as graph intermediates for block-local attention; it has no persistent
encoder K/V cache to shrink. A follow-up must measure actual peak memory,
latency, and WER for a changed encoder against the unchanged one before a
runtime flag or benchmark ablation row is added. The low-rank draft-head
question was measured separately with CTC token acceptance and complete
Python decoder latency in [`ctc_draft_rank.md`](ctc_draft_rank.md).

## Native follow-up: low-rank K/V in the engine (2026-10-01, issue #59)

The follow-up the paragraph above demanded now exists. The granite encoder
gained an opt-in factor path (`STARLING_GRANITE_KVFACT=<file>`,
`cpp/granite/kv_factors.hpp`): per attention layer, per head, the K and/or
V half of the fused `attn_kv` GEMM is replaced by two GEMMs through a rank-r
factorization, with the original bf16 rounding boundaries. Three factor
provenances were measured on the notebook (Ryzen 5650U, CPU backend,
granite-2b-dynq4 GGUF, alternating A/B vs a frozen baseline binary,
median of 3x3 on medium.wav; exact-transcript contract on all three
fixtures; FLEURS en_us test 100-clip WER gate at 0.2 points):

| Provenance | rank | enc | wall | transcripts | WER |
| --- | ---: | ---: | ---: | --- | ---: |
| weight-space SVD | 32 | +2.22% | +0.74% | identical | (not gated) |
| weight-space SVD | 24 | +2.56% | +0.71% | identical | +0.28 FAIL |
| weight-space SVD | 16 | +2.79% | +1.15% | identical | +0.19 pass |
| weight-space SVD | 8 | — | — | differ | — |
| K+V both r=32 | 32 | — | — | destroyed | — |
| bf16-activation PCA | 32 | +2.20% | +0.82% | identical | +0.42 FAIL |
| runtime-activation PCA | 32 | +2.12% | +1.14% | identical | +0.28 FAIL |

Memory does not improve: the factor file is additional resident data
(f32 r=16 factors ≈ 2x the Q4_K K-half bytes; peak RSS 1900 → 1929 MB).
The encoder K/V are graph intermediates only — there is no persistent cache
to shrink, so low-rank buys a small latency cut at a memory and WER cost.
Counter to the calibration's promise (K held-out rel-MSE 0.002–0.035 at
rank 32–64), activation-fitted bases transfer *worse* to WER than the
weight-space SVD: both the bf16-reference fit and a fit on the engine's own
K dumps (new `STARLING_GRANITE_DUMP_K` probe) show the same per-layer
projection error and the same WER outcome, so provenance is not the cause;
PCA-style bases zero out-of-basis directions that held-out test audio
excites, while weight-space SVD spreads a uniform (larger) error over all
directions. WER deltas are deterministic (greedy decode) but not monotone
in rank — every useful rank costs 0.19–0.42 points and rank 16 passes the
gate with roughly one flipped word of margin.

Verdict for #59 on granite: **not worth enabling**. The speed ceiling is
structural (the whole attention block is 16.6% of encoder MACs; the K path
4.5%), the only gate-passing config sits on the WER gate edge, and memory
regresses. The factor path stays env-gated OFF; the exporters
(`benchmarks/export_kv_lowrank*.py`), the dump probe and the numbers above
are the reproducible record. Qwen3-ASR spectral calibration remains
unmeasured.

## Qwen3-ASR run on public speech (2026-10-01)

The missing measurement noted above. Method: the same per-head PCA with
even/odd clip split over the same eight FLEURS en_us train clips as the
granite run (hashes in `results/kv_spectral_granite_fleurs8.json`; the
qwen3 artifact is `results/kv_spectral_qwen3_fleurs8.json`). The capture
classes, `pca_layer` and `effective_dim` are imported from
`bench_kv_spectral.py` unmodified; the driver ran on CPU (notebook, bf16)
because the script's qwen3 forward path is CUDA-hardcoded — a session-side
CPU driver (issue #59 notebook campaign) reproduced its device plumbing
only.

| Quantity, averaged over 24 layers and sixteen heads | K | V |
| --- | ---: | ---: |
| Rank / 64 at 95% training variance | 58.3% | 72.0% |
| Rank / 64 at 99% training variance | 81.4% | 91.1% |
| Rank / 64 at 99.9% training variance | 93.0% | 99.0% |
| Held-out relative squared error at fixed rank 16 | 0.268 | 0.428 |
| Held-out relative squared error at fixed rank 32 | 0.126 | 0.231 |
| Held-out relative squared error at full rank 64 | ~1e-13 | ~1e-13 |

Unlike granite's K (25% of 128 at 99% variance), Qwen3-ASR's K and V are
both close to full rank on held-out speech — only layer 0's K is compact
(16%). Halving either projection's rank (32 of 64) costs 13–23% of
held-out K/V energy. The calibration-level answer for Qwen3-ASR is
therefore **no-go**: there is no rank with both useful compression and
small reconstruction error, so no native follow-up experiment is justified
on this evidence.

**Margin stress test (same day):** re-gating the draw-1 "passing" rank-16
config on a second, disjoint 100-clip FLEURS draw (clips 100–199) moved the
delta to **+0.29 (base 5.38 → 5.67) — FAIL**. The draw-1 pass was clip-draw
luck; the real WER cost of K rank-16 compression straddles the 0.2-point
gate. Final verdict for #59: **encoder KV compression is a no-go on granite
in every tested configuration** (weight-space r16/r24, activation-basis r32,
runtime-basis r32: 0.19–0.42 points across draws, none reliably within 0.2),
on top of a structural latency ceiling (attention is 16.6% of encoder MACs,
the K path 4.5%) and a memory regression. The env-gated factor path,
exporters and K-dump probe remain committed as reproducible research
tooling; no runtime default changes.

**Selective map follow-up (same campaign):** compressing only the six
runtime-low-rank layers (per-layer held-out error <= 0.0045, others full
rank; v3 factor format) keeps transcripts identical, wins enc +0.83% /
wall +0.50%, and moves WER by **-0.19 on both disjoint draws** (5.85→5.66,
5.38→5.19) — deterministic and inside the gate with margin. The sign is
consistent with the rank projection denoising the Q4_K weight error that
lies orthogonal to the speech subspace on near-exactly-low-rank layers.
So the final, nuanced verdict: uniform-rank K/V compression fails the WER
gate at every useful rank, but a calibration-driven selective map —
roughly a third of the K path's compute — passes every gate with a small
latency win and neutral-to-positive WER. Tooling: v3 format
(`cpp/granite/kv_factors.hpp`), `benchmarks/export_kv_lowrank_selective.py`.
