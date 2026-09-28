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
