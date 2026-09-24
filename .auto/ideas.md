# Ideas backlog (fast engines, Pixel 10 Pro)

- **MOSS decode dispatch fusion**: 5 sequential kernels × 28 layers per token (72 dispatches/token); effective 9.8 GB/s vs 13-24 GB/s isolated GEMV = per-dispatch/barrier fixed cost. Candidates: fuse dec_o into dec_attn via per-head atomic accumulation of W_o partials; fuse residual+RMSNorm chains further. Biggest remaining structural item (~2-4% total each).
- **Coopmat**: driver 0x00696bd0 crashes the shader compiler on any coopmat op (probe: coop_probe.comp). Revisit after driver updates — SINT8 coopmat would be the 3-5x encoder path. Kernel exists (gemm_coop.comp, STARLING_FAST_COOPMAT=1).
- **MOSS mel**: shared whisper_mel frontend is thread-noisy (124-270ms regardless of threads); a PkMel-style specialized path (or float FFT where numerics allow) would save ~100-150ms/run.
- **GEMM shared-traffic**: 155-180 GFLOPS vs ~500 ALU peak; per-thread bigger TM×TN hits register pressure at wg=128. A layout with f16 accumulators held in registers across BK (F16MATH) was slower; int8-activation dot (VK_KHR_shader_integer_dot_product works at feature level, but glslang has no GLSL front-end for OpSDot — needs hand-assembled SPIR-V or a newer glslang).
- **q4e4 model file**: −7% MOSS decode for +0.14 WER (8.06 vs 7.92) — an ops-level option, WER-verified this session.
- **Speculative decoding for MOSS** (draft K tokens, verify in one pass) — the 100ms/token chain is sequential-dependency bound; large potential, significant work.
- **PkMel frame parallelism** (cpp/fast/pk_mel.cpp, in fast-engine scope): frames are independent → disjoint-range threading is determinism-safe like whisper_mel's; PK mel 31→~8ms on the phone (~0.35% composite, below noise floor — verify with repeated A/B before keeping).
- **dec_attn barrier count**: 4 block-reductions × 7 barriers per WG; estimated ~3-4ms/token real cost — a one-barrier shared layout could save ~1% composite (norm-probe instrument can be extended to attn shapes).
- **lm_head W4 + exact top-K GPU rescore**: ~7% decode potential; two-pass, needs a new kernel + FLEURS re-gate (near-tie argmax flips).
- **Speculative decoding**: only pays with a real draft model (self-n-gram acceptance < breakeven ~50% for dictation); needs M=2 prefill-style batching machinery.
