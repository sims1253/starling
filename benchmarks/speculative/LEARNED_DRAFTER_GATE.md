# Learned drafter gate for #314 (2026-09-28)

This is a decision record, not a runtime benchmark. The current decision is
**defer training and deployment**. Issue #314 permits at most one prototype
after the free drafts and #311 verifier are priced on the target device and
workload. The generic native verifier [#338](https://github.com/sims1253/starling/pull/338)
and an S1 CPU copy-draft pilot [#345](https://github.com/sims1253/starling/pull/345)
now exist. They do not provide a Pixel/fast-engine multi-token cost curve,
per-take energy, or the representative #310 dictation workload. A learned
model would add memory and draft work before those gates establish whether
its cost can be repaid.

## Measured inputs and break-even test

The Pixel fast-engine log records MOSS target-only decode at about 69–70
ms/token in one thermal window. Its two-row W4 GEMV microbenchmark processes
both token rows at 1.76–2.11 times the single-row *per-token* rate, but this
is a kernel measurement, not the cost of a complete verification pass. The
same log records standalone MOSS's Parakeet text draft at only 1.03 output
tokens/pass (K=2) or 1.05 (K=3) over 12 FLEURS clips; online n-gram yielded
1.000. Those streams have no substantial free-draft benefit in that
standalone setup. See [the measured log](../fast_engine/RESEARCH_LOG.md),
P1-6 through P1-8. In the separate S1-mini cleanup replay, aligned copy at
K=2 gives 1.375, 1.491, and 1.495 output tokens/pass on the short, medium,
and long goldens. Those are exact-token oracle replays of only three fixtures,
not device timing or a representative workload; see
[the copy-draft study in PR #327](https://github.com/sims1253/starling/pull/327).
The native S1 follow-up measured the entire CPU C API call, including source
tokenization and proposal work, after #338 corrected the CPU verifier's
attention width. At K=2, copy matched greedy and captured stock IDs/text on
all three goldens; a K=4 synthetic protected fixture kept both protected
spans. Across two warmed alternating pairs per case, copy/greedy latency was
0.58–0.94×, with all eight pairs favoring copy. These are small CPU samples,
not Pixel, fast-engine, energy, or #310 results. #345 preserves the earlier
full-capacity CPU run separately because it had different verifier behavior.

For a fixed workload, let `A` be actual output tokens per verify pass,
`V(K)` the measured full target verify-pass time, `D(K)` the full cost of
proposing K tokens, and `T1` target-only decode time per output token. A
candidate wins on decode latency only if `(V(K) + D(K)) / A < T1` on the same
device and prompt distribution. It must also beat the *best free draft* with
`(V_free + D_free) / A_free`, with prefill, residency, load cost, memory and
energy accounted per take. For S1 cleanup, even a zero-cost copy drafter at
K=2 requires `V(2)/T1 < 1.375–1.495`; the replay alone cannot establish
this, while the current native CPU call measurement does show a bounded
S1-specific win. For standalone MOSS, a zero-cost Parakeet draft would
require `V(2–3)/T1 < 1.03–1.05`. No target-device `V(K)`/`D(K)` curve or
representative #310 result exists yet, so the CPU S1 result cannot justify a
learned-drafter speed, energy, or deployment claim.

## Candidate comparison

| Family | Benefit to test | Added work / artifact | Decision now |
| --- | --- | --- | --- |
| Small independent model, e.g. Qwen3-0.6B for an exact compatible Qwen3 target | No draft training if tokenizer and target IDs match | A second decoder, weights and KV; K sequential draft steps. Even raw 0.6B 4-bit weights have a 300 MB lower-bound before scales, embeddings and KV. | First *possible* prototype because it is reversible and needs no training. Do it only after the verifier gate, exact tokenizer/template check, and per-take co-residency budget. |
| EAGLE-3 | Target-feature-conditioned draft; its released method has trained weights and training code. | Target-specific feature capture, training, sequential draft steps, new export/runtime path. The official checkpoint table does not list Starling's MOSS, S1-mini or Granite revision. | Skip until a target-specific training set and verifier economics justify that cost. |
| DFlash | Its block diffusion drafter proposes a block in one forward, which could reduce phone dispatch cost. | Target-feature capture, trained target-specific model, block runtime and export. Public code/checkpoints list other exact targets, not Starling's current revisions. | Skip until a measured fast-engine verify pass and draft memory/latency budget exist. |
| DSpark | Semi-autoregressive drafting and confidence-scheduled verify length could reduce wasted wide passes. | Target-specific training and more runtime state/logic. SpecForge now has DSpark training recipes, so the issue body's “no public code found” is outdated. | Skip: neither a verify-width cost curve nor a trained compatible artifact exists here. Reuse scheduling only after a functioning drafter exists. |
| MTP / Medusa heads | Parallel next-token guesses can avoid a separate decoder. | Trained heads tied to the target hidden state; Medusa's tree verify needs a different verifier shape. | Skip until a fast-engine verifier and representative workload are measured. |
| LayerSkip | Draft from earlier layers of one model. | Training with layer dropout and early-exit loss changes the target; not an add-on to the pinned target revision. | Reject for current fixed targets. |

The 300 MB figure is arithmetic (`0.6e9 * 4 / 8`), not a measured model
footprint. The authors' published speedups use other targets and hardware and
are **not** inserted into the Starling cost curve. All candidates must preserve
the target's greedy IDs through #311 verification; draft precision is a
measured acceptance/memory trade-off, not an assumption.

## Reopen gate and bounded experiment

1. Extend the narrow CPU S1 pilot to the #310 buckets, then on one pinned
   Pixel and desktop fast-engine build measure complete `V(K)` for K=1, 2,
   4, 8, with target-only greedy parity and actual peak memory. Capture
   per-take energy by the declared protocol.
2. Replay free drafts on the *same* emitted target IDs. For cleanup, include
   aligned copy and revision reuse; for ASR, distinguish live-preview Parakeet
   (already paid) from batch Parakeet (new cost). Select the best free baseline
   for each workload and device.
3. Only if a workload has poor free acceptance *and* the verifier inequality
   leaves enough room for positive `D(K)`, try one exact-target compatible
   small independent model first. Measure its draft precision, acceptance,
   memory, load time, stop-to-final latency, energy and greedy parity against
   the best free baseline. If that fails, stop this issue's prototype track
   unless new evidence changes the economics.

No draft artifact was trained or exported for this record, and no runtime
default or ablation row changes are justified.

## Primary sources checked

- [EAGLE-3 official repository, pinned commit](https://github.com/SafeAILab/EAGLE/tree/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b), including its checkpoint table and training path.
- [DFlash paper](https://arxiv.org/abs/2602.06036) and [official code/checkpoint list, pinned commit](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a).
- [DSpark paper](https://arxiv.org/abs/2607.05147) and [SpecForge training catalog, pinned commit](https://github.com/sgl-project/SpecForge/blob/c8c636f844d0016b1151be333d3bc27f8f7e91f6/examples/configs/README.md).
- [Medusa paper](https://arxiv.org/abs/2401.10774) and [official repository, pinned commit](https://github.com/FasterDecoding/Medusa/tree/e2a5d20c048a9b0a4092e6933c34313687422518).
- [LayerSkip paper](https://arxiv.org/abs/2404.16710) and [official repository, pinned commit](https://github.com/facebookresearch/LayerSkip/tree/494752e5fbb0a82989f6cb384841684b1c2ef5c3).
