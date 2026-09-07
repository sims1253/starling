# Ideas backlog

- **F16 activations through the encoder** (~40-70ms est): halve elementwise+GEMM-operand bandwidth; needs casts at op boundaries or f16-native graph; numerics change (WER gate; sha will drift). The conv module already uses f16 weights.
- **Dependency-aware barriers in ggml-vulkan** (~80-130ms est): emit pipelineBarriers only between producer→consumer node pairs (track last-writer per tensor), let independent branches overlap; replaces the global barrier-per-op. Measured ceiling: no-barrier run was 17% faster (racy); exec-only was 13% (racy).
- **Device-resident decode state**: LSTM h/c as persistent device buffers (TDT only commits state on emit → no per-step state upload/readback); enc_proj_t via GET_ROWS from a device-resident encoder output + frame index input. Est 5-15ms.
- **Fused LSTM-cell custom op** (CPU+Vulkan): collapse ~14 gate/cell nodes per layer into one kernel; ~20-30ms est on medium.
- **Fold 1/sqrt(dk) into cached ph** (kills 24 SCALE nodes on [T,T,H] masks; changes rounding order slightly).
- **REPEAT=160 + CPY=72 in decode window**: investigate what emits them (pos-bias adds? state cpy?) and remove.
- **mel GPU graph** is 8 nodes/~3-9ms incl readback — check if the DFT matmuls ([2231×257], [257×2231] f32 GEMMs, 2.8ms) could fold preemph/window into one pass or run f16.
- **CPU-side**: CPU build hasn't been tuned at all (LLAMAFILE ON already). Bench GGML_CPU_ALL_VARIANTS, llamafile sgemm vs AVX2 paths for our skinny GEMMs; the encoder GEMMs on CPU use q8_0 vec-dot — try q8_1/q8_K? Also CPU suffers from the same node-count issue → CPU flash attention is used? Check.
- **starling-serve streaming**: chunked decode reuses replay graphs — the pos-cache LRU (16 T values) could hold ph for the common chunk lengths.
- **Quant recipes**: q8_0 wins on Vulkan because dequant cost dominates; a CUSTOM q8 format with smaller blocks (16 instead of 32) or per-tensor scales could trade a bit of density for faster dequant in shaders. Requires converter + shader variant + loader — medium effort.
- **Peak-memory**: ph cache adds device-resident [dk,P,H] per cached T (up to 16×183MB worst case for very long audio); consider capping ph-bearing entries in the LRU or storing ph as f16 (halves bytes, small numerics risk).
