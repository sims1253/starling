# #313 Parakeet text → MOSS copy: desktop preregistration

Recorded before scoring the paired trial. This is a bounded feasibility test,
not a product rollout or a Pixel acceptance test. The source is the **saved
final** Parakeet TDT output for each public LibriSpeech fixture, reused as if a
streaming preview were already available in memory. Actual last partial text
may be shorter or wrong, so these results are optimistic for live use. A
Parakeet inference is neither run nor charged in these timings; this is valid
only when the application has already paid for that preview. Its cost for
batch files remains unmeasured here.

## Pinned inputs and design

- Base: `b967ef23c734c5e16f4a52865af96e61e86c8214` (#338 CPU exact-width
  native MOSS verifier). Model: MOSS bf16 exact GGUF, SHA-256
  `b96dae2bcadc9e89f61a3ac1e103915e0abb11da521ca6d8db91e3244129e71a`.
- Public fixture provenance: `tests/fixtures/short.wav` derives from LibriSpeech
  `2086-149220-0033`; medium tiles the same utterance. Their SHA-256 values
  are `5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150`
  and `4a62a4b4c9ca8d669fff179fed8a303e474e63ca8724025d3c1c70e5b44787ab`.
  The local golden paths point to the repository's canonical saved outputs;
  source text SHA-256 values are
  `41f9e18f92ad8dfa94861d37ccb1262f07a1660c7b42f50bf51252cf5b5d0383`
  (short) and
  `cdda8e3b110231be27454106494a9c217518549bf514075db37e4346983c2e08`
  (medium). MOSS saved text hashes are
  `4b7b1073246846994578dce6910ba0fa85cf07918053dd0df4223ccbaef10ddc`
  and `9ed7421e4851d167b5cd53cb283802f0cad41e6a9fc9f8c940a4109bf1dc6b8d`.
- CPU only, affinity 16–23, four ggml threads, Release, ggml CPU AVX2, no
  CUDA. Executable SHA-256 before scoring:
  `51c842888ceb4282a53e85aa04d594b995d8a4c03c5cf10160992728bf870528`.
- For each fixture, warm greedy, K=2 and K=4; then run two paired repeats per K,
  reversing greedy/draft order on the second repeat. Each invocation redoes
  the actual MOSS mel, audio encoder, prompt, and full greedy or speculative
  decode to EOS (model token budget 200). Source lowercasing, whitespace
  normalization, MOSS BPE tokenization and copy-proposer construction are
  charged to the draft arm. File I/O and model load are excluded from both.
  Source copy aligns only with verified MOSS prefix IDs; mismatches fall back
  to the native verifier. No oracle target IDs or logits enter the proposer.
- The measured outcome is full MOSS call time and exact ID/EOS parity. Report
  proposed/accepted counts and all paired times. The narrow desktop go signal
  requires at least one K to save **at least 5%** of full-call time in **both**
  repeats for **both** fixtures, while every draft produces identical IDs and
  EOS. Otherwise this pilot does not support enabling the path. A go signal
  would only justify broader corpus and Pixel tests, not enable it by default.

Run with `STARLING_GGML_DEVICE=cpu STARLING_GGML_THREADS=4` and
`taskset -c 16-23` against `moss-preview-copy-pilot`, model path, golden root,
short WAV, medium WAV, and `2` repeats, in that order.

## Amendment after the first scored failure, before the opt-in paired trial

The original binary exited at the first medium K=2 pair. Its saved record is
`moss_preview_copy_initial_failed.jsonl`: short K=2/K=4 pairs (four total)
matched, but medium K=2 diverged despite ending on EOS. A diagnostic build
found first disagreement at zero-based output ID 21 (greedy 13, verifier 432):
greedy 89 IDs versus speculative 91. K=1 reproduced that same first mismatch;
an empty proposer that routes every token through the one-step fallback matched
all 89 greedy IDs. These controls are not scored repetitions.

An isolated, **opt-in** native verifier experiment now computes each batched
attention row over `past + row + 1` keys, while retaining batched projections
and cache writes. The default remains unchanged. One diagnostic medium K=1
trial with `STARLING_MOSS_VERIFY_ROW_ATTN=1` matched all 89 IDs and EOS. The
next scored trial keeps the same two fixtures, warmups, K=2/K=4, alternating
two repeats, CPU affinity, parity rule and ≥5% full-call rule above; it adds
only `STARLING_MOSS_VERIFY_ROW_ATTN=1`. Its rebuilt executable SHA-256 before
scoring is
`79efffb670aa22a1a777906bae42742ea09fa0ede3074f7dd4f5fc38c964b6ba`.
The first failure remains part of the result rather than being erased by this
amendment. Passing the two fixtures would be bounded evidence for this fix,
not a proof that all CPU models or future inputs are identical.

After scoring, #338 adopted the same row-attention graph as the CPU default in
`419723df29f2990f2a22efb58e256d8b848bee24`. The historical result and
binary hash above refer to the opt-in experiment, not a fresh timing of that
later default commit. The final default can be reproduced without the old
opt-in flag; `STARLING_MOSS_VERIFY_BATCH_ATTN=1` selects the earlier batch
attention for diagnosis.
