# Ideas backlog (fast engines, Pixel 10 Pro) — post-exhaustion state

Everything above the composite noise floor (~1%) has been tried; remaining items:

- **lm_head W4 requant**: machinery exists (`convert_w8_w4`, `STARLING_FAST_LM4=1`), transcripts verified identical, −149MB, load −0.3s, lm_head 8% faster isolated — but only 0.9% of decode (W8 already at 36.6 GB/s). Flip on only with a FLEURS run and if memory/load matter for the app.
- **Coopmat**: driver 0x00696bd0 crashes the compiler on any coopmat op (probe: `coop_probe.comp`). The 3-5× encoder path if a driver update fixes it; kernel exists (`gemm_coop.comp`).
- **Speculative decoding for MOSS decode** (88.7ms/tok, sequential-dependency bound): needs a real draft model (self-n-gram acceptance < the ~50% breakeven) and M>1 batching machinery.
- **Flash-style fused attention for the encoders**: removes S/BD materialization (~1.4GB traffic at T=743); numerics change (softmax order) requires joint fixture+WER re-baseline.
- **Load-time repack cache (mmap)**: PK 1.7s/MOSS 4.6s loads → skip repack on warm loads. Not in phone_ms; measure via the load= line if pursued.
