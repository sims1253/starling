# Ideas backlog (#317 loop, ranked)

- [0] **Side measurement for #311**: skinny-GEMM cost on the Pixel for W4/W8
  at M = 1, 2, 4, 8, 16 (speculative-decoding verify shapes). Log only; no
  code change. Decides whether speculative decoding pays.
- [1] **int8 activations × current W4/W8** (kernel-only; the int8 baseline):
  gemv variants quantizing x to int8 per K-group in-register, dotPacked4x8EXT
  against W8 bytes directly; W4 needs nibble→int8 expansion (price it first —
  if expansion eats the win, W4 keeps unpackUnorm4x8 and only W8/lm_head goes
  idot). WER gate (int8 activations change numerics).
- [2] **Symmetric W4 with pre-permuted nibbles** (`w4g64sym-p1`-style): lean
  scale store (1 f16 per group) + byte-wise unpack with no activation
  reorder; kernel + gemm/dequant_row variants needed (G5).
- [3] **Group 64/128** with 8-bit group scales under an f16 per-row
  super-scale (`w4g128symu8s` — quantizer + eval already exist). Phase 0:
  g64 costs ~+1 pt en WER; only worth it if the ALU cut is big.
- [4] **W8 with group 32** (scale per 32, not the Q6_K-inherited 16):
  `w8g32sym` → Q8_0-shaped; −5.5 % scale bytes, needs gemv variant.
- [5] **Scales interleaved with the weight tile** vs separate arrays
  (locality for the GEMV's row walk).
- [6] **Hadamard rotation** folded offline (QuaRot-style) to tame activation
  outliers for int8 activations; pairs with [1].
- [7] **One data point**: quality per resident byte of IQ*_KT (ik_llama.cpp,
  CPU) vs the best affine candidate; one STARLING_FAST_MICRO probe pricing an
  integer-trellis GEMV on PowerVR. EXL3 = reference, not target.
- [8] **W8 lm_head via unpackSnorm4x8** (the RESEARCH_LOG #33 follow-up;
  needs a proof that no -128 codes occur in Q8_0-repacked embeds — trivial to
  check: Q8_0 codes are clamped to ±127 by quantization... verify in
  pack_cpu_q8/pack_gpu_matrix).
- [9] **Packed-load speed**: the STARLING_FAST_PACKED path doubles load
  memory (pack + HostMatrix copies). mmap the pack + slice views if load
  time on the phone matters for experiments.
