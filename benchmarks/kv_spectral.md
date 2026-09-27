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

The script has not yet been run on a representative ASR set in this change.
The local workspace lacks the Granite and Qwen3-ASR model snapshots and a
multi-clip committed audio set. The synthetic test checks that a new held-out
direction raises the reported error while training rank stays unchanged.

Even a low reconstruction error would only justify a native experiment. The
encoder path in [`cpp/granite/encoder.cpp`](../cpp/granite/encoder.cpp) builds
K and V as graph intermediates for block-local attention; it has no persistent
encoder K/V cache to shrink. A follow-up must measure actual peak memory,
latency, and WER for a changed encoder against the unchanged one before a
runtime flag or benchmark ablation row is added. The low-rank draft-head
question is separate: it also needs CTC token acceptance and complete
decoder latency, not projection time alone.
