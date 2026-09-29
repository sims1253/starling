# Granite low-rank CTC draft-head study (#59)

The pinned [Granite Speech 4.1 2B source
revision](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/tree/de575db64086f84fdc79da4932d1076e965bc546)
provides a BF16 `out_llm` CTC head of shape 100353 × 1024. Its source
`out_llm.safetensors` SHA-256 is
`6cc10d68fe05aec359aceffd597617c875b23f27211ee6dcdb7510d9e90fc64e`.
The baseline draft follows the repository's
[`CTCBPEDraft`](../src/starling/granite/speculative.py): blank-weighted
four-frame pooling, BF16 head projection, CTC collapse, and label-to-token
mapping. [`bench_ctc_draft_rank.py`](bench_ctc_draft_rank.py) takes the top
eigenvectors of `W.T @ W` and projects through two BF16 factors. It uses no
audio to fit the factors. The eight hash-pinned public FLEURS English clips
from the [K/V study](kv_spectral.md) are evaluation inputs, not factor-training
examples. Detailed frame labels, collapsed drafts, hashes and timings are in
[`results/ctc_draft_rank_granite_fleurs8.json`](results/ctc_draft_rank_granite_fleurs8.json).

On the RTX 5090, a single projection over all 1,074 pooled frames took 2.26
ms for the full BF16 head. Rank 256 took 0.79 ms with 25.3% of the head's
weight storage, but its collapsed drafts differed from the full head on all
eight clips (136 total token edits). Rank 512 took 1.36 ms with 50.5% storage
and still changed all eight drafts (113 edits). Rank 1024 is a *full-width
factorization control*, not compression: it uses 101% of the original weight
storage and changed one of eight drafts due to BF16 factor rounding.

[`bench_ctc_draft_e2e.py`](bench_ctc_draft_e2e.py) repeated each WAV twice,
after warming the decoder. Each full-wall trial starts at the decoded waveform
and includes processor, encoder, draft head, projector, prompt, prefill and
verification. Factor preparation is excluded as a one-time export cost.
Method order alternates between repeats. The Python
[`SpeculativeDecoder`](../src/starling/granite/speculative.py) verifies every
emitted token; all 80 speculative trials produced the same greedy IDs as the
target-only path. Complete per-trial stage times, draft counts and acceptance
are in
[`results/ctc_draft_e2e_granite_fleurs8.json`](results/ctc_draft_e2e_granite_fleurs8.json).
The saved trials warmed only the target and full-head methods on the first
clip before timing. The current script prepares rank slices once and warms all
methods for each clip shape. The saved stage times are historical measurements,
so a fresh run is needed for timing under the corrected method.

| Method | Head weight / full | Head (ms) | LLM prefill + generation (ms) | Full wall (ms) | Accepted / proposed | Clips faster than direct full CTC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target only | — | — | 178 | 232 | — | 1/8 |
| Direct full CTC | 100% | 1.6 | 98 | 148 | 115/206 | — |
| Rank 64 | 6.3% | 1.0 | 163 | 200 | 0/15 | 1/8 |
| Rank 256 | 25.3% | 1.5 | 169 | 211 | 57/100 | 0/8 |
| Rank 512 | 50.5% | 2.4 | 160 | 210 | 91/208 | 1/8 |
| Rank 1024 control | 101.0% | 1.4 | 141 | 184 | 115/206 | 2/8 |

Accepted/proposed counts are per one pass over eight clips (each clip's two
repeats produced the same counts). They include the Python verifier's
re-alignment matches. Full-wall values average each clip's two-trial median.
The full-wall column also includes waveform preparation (about 2 ms), encoder
(30–47 ms), and projector/prompt (4–5 ms). LLM generation includes prefill
and verified decode. Each number is the mean of per-clip two-trial medians,
rounded independently; columns therefore need not sum exactly.
With only eight English clips and two repeats, these are directional desktop
measurements, not a device-wide confidence interval. Rank 256 saved about
1.5 ms in the batched head microbenchmark, then lost about 63 ms per take
against direct full-head speculation because weaker drafts needed more verify
passes. Rank 512 had the same outcome. Independently re-factorizing the head
also changed one rank-64 and one rank-256 draft between the rank and end-to-end
runs, so any future factorized artifact needs a fixed exported weight file.

The current result is **no-go for a low-rank CTC head**: the useful ranks did
not beat the direct full head on this workload. We did not add a runtime flag
or ablation row. The direct full-head Python verifier is promising on these
desktop clips, but native #311 verification, Pixel latency/energy and a
broader workload must decide native enablement. This study measures no phone
energy and does not claim a native speedup.

Reproduce with the pinned snapshot and eight hash-matched WAVs:

```sh
PYTHONPATH=src uv run python benchmarks/bench_ctc_draft_rank.py \
  --snapshot /path/to/de575db64086f84fdc79da4932d1076e965bc546 \
  --audio /path/to/en_us_train_{0,1,2,3,4,5,6,7}.wav \
  --output benchmarks/results/ctc_draft_rank_granite_fleurs8.json
PYTHONPATH=src uv run python benchmarks/bench_ctc_draft_e2e.py \
  --snapshot /path/to/de575db64086f84fdc79da4932d1076e965bc546 \
  --rank-result benchmarks/results/ctc_draft_rank_granite_fleurs8.json \
  --audio-dir /path/to/wavs \
  --output benchmarks/results/ctc_draft_e2e_granite_fleurs8.json
```
