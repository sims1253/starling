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

- [0] ~~(#311) CLOSED FOR THIS LOOP~~: gemv_w4um (kept) delivers 1.76-2.11x
  per token at M=2, but NO deployable drafter exists for standalone MOSS
  decode: online n-gram 1.000 tokens/pass, Parakeet copy-draft 1.03-1.05
  (BPE-level divergence without prompt conditioning), self-lookahead
  zero-gain by construction. Needs a learned drafter (#292 gated follow-up)
  or the product cleanup flow (different measurement than moss_decode).
- ~~lm_head as F16~~ REJECTED (P1-9): +9.4% — doubles per-token table bytes;
  in-context bandwidth ~43 GB/s (not the micro's 90). The lm_head is at its
  floor in every format. BACKLOG EXHAUSTED: decode budget fully accounted
  (linears op-bound ~41 ms + lm_head ~8 ms + dispatch 4-6 ms + attention
  ~4 ms = the measured 69-70 ms/token); every remaining idea is either
  measured-dead, gated on #311's learned drafter, or below the noise floor.
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
