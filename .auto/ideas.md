# Ideas backlog (#317 loop, ranked; revised 2026-09-26 after the idot experiments)

Data-driven model of the phone (all measured this session):
- ALL GEMV variants (W4 scalar, W4U, W8 scalar, W8-idot) plateau at
  37-46 G weights/s regardless of bytes; dotPacked4x8EXT costs ~2 issue
  slots on DXT (same as the f32 dot(vec4) it replaces) -> int8 activations
  via OpSDot buy NOTHING on this GPU. Dead: W8/W4 idot, any OpSDot GEMV.
- F16 GEMV: 46.5 G w/s = 93 GB/s at the lm_head shape -> memory has >=2x
  headroom over the W4/W8 rates; the ALU (unpack+dot) is the limiter.
- W4U is near the vec4-dot op floor (~20 ops/32w); no ISA trick found below
  it (unsigned-dot on masked nibbles prices worse; f16 dots need f16 unpacks
  that cancel the win).

- [0] **(#311) Skinny-GEMM pricing at M = 1,2,4,8,16** for speculative
  decoding: if the GEMM at M=4 costs < 2.5x the GEMV, token verification
  halves effective decode — the only lever left that beats the op floor.
  Probe: time the existing gemm_w4 at M=1..16 (real llm shapes) on the
  phone. Needs a tiny GEMM micro (Kernels::micro gemm mode) + phone session.
- [1] **lm_head as F16** (+350 MB resident, memory-sensitive): measured
  6.7 vs 8.0 ms per token for the lm_head GEMV (93 GB/s path). ~-2%
  decode. Cheap to test (repack choice at load, no new kernel).
- [2] **Fewer dispatches per token** (~146): attention 3.9-4.2 ms/token
  (RESEARCH_LOG #22), norms ~1-2%. Fuse RMSNorm into the NEXT GEMV (the
  gemv already fuses its OWN norm via NORM=1; the intermediate norms
  between attention and FFN may double-dispatch).
- [3] **Hadamard rotation** for int8/f16 activation robustness — only
  interesting if an int8/f16 activation path ever pays; deprioritized.
- [4] Layout table quality data (from Phase 0): w4g64sym +1.0 pt en WER,
  w4g128symu8s +0.8 pt, w8g32sym ~baseline — group-size/scale-dtype
  variants only matter for MEMORY (#316), not speed (op floor).
- [5] One data point: IQ*_KT quality/byte vs the best affine candidate
  (desktop only; cheap to log, informs #316).
