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
