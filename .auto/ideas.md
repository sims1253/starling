# Ideas backlog (fast engines, Pixel 10 Pro)

- **Coopmat**: revisit after any PowerVR driver update (probe: `cpp/fast/shaders/coop_probe.comp`, run with STARLING_FAST_COOPMAT=1). Kernel exists: gemm_coop.comp. Driver 0x00696bd0 crashes the compiler on any coopmat op.
- **SINT8 coopmat** (A/B=SINT8, C=SINT32) exists on the device per properties — same crash today; int8 activations × int8 weights with per-row scales would be the bandwidth/ALU dream if a driver fixes coopmat.
- **Subgroup arithmetic/shuffle ops** are supported (ops=0x6ff, subgroup 128): could replace shared-memory B broadcast in GEMV/GEMM with shuffles.
- **VK_KHR_shader_integer_dot_product** exposed: dp4a-style int8 math without coopmat — needs int8 activation quantization (WER gate) + kernel work. Medium effort, uncertain payoff vs f16.
- **MOSS decode**: 100ms/token ≈ 11GB/s. Q8 lm_head = 28% of bytes → try q4e4 model variant or smaller lm_head precision; speculative decoding (draft from Parakeet transcript / n-grams).
- **Norm fusion into GEMM A-load**: 97 norm dispatches in PK encoder; fold row stats (2 scalars/row) into the next GEMM's A-tile load.
- **Mel on GPU** for MOSS (250ms!) — MOSS mel=250ms vs PK mel=25ms; check pk_mel vs moss mel path.
- **Load-time repack cache on disk** (mmap) to skip repacking on later loads.
- **FLEURS WER gate** must run at next numerics-affecting milestone.
