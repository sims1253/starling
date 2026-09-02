# Vulkan-backend diagnostics (ggml bf16-activation work)

Debug harnesses used to bisect the Vulkan backend's bf16-activation gap on
the AMD Ryzen 5 5650U / RADV RENOIR iGPU notebook (2026-09-02). Build from
the repo root against the built ggml shared libraries:

```bash
GXX=g++
INC="-I third_party/ggml/include -I third_party/ggml/src -I cpp -I cpp/include"
LIBS="build/third_party/ggml/src/libggml.so build/third_party/ggml/src/libggml-base.so \
      build/third_party/ggml/src/ggml-vulkan/libggml-vulkan.so.0 build/third_party/ggml/src/libggml-cpu.so.0"
RPATH="-Wl,-rpath,$PWD/build/third_party/ggml/src -Wl,-rpath,$PWD/build/third_party/ggml/src/ggml-vulkan"

$GXX -O2 -std=c++17 $INC -o /tmp/test_bf16_gemv test_bf16_gemv.cpp $LIBS $RPATH
$GXX -O2 -std=c++17 $INC -o /tmp/test_ops      test_ops.cpp       $LIBS $RPATH
$GXX -O2 -std=c++17 $INC -o /tmp/stage_cmp     stage_cmp.cpp      build/libstarling_ggml.so $LIBS $RPATH
```

Run with `STARLING_GGML_DEVICE=Vulkan0` (or `cpu`) and `env -u LD_LIBRARY_PATH`.

## Findings (status of the Vulkan unlock)

1. **Patch 0009 (bf16-y GEMV) is exact.** `test_bf16_gemv` compares
   `mul_mat(BF16 W, BF16 x)` CPU-vs-Vulkan: maxdiff 3.6e-07 (float epsilon;
   tighter than the f32-y control's f16-storage noise 4e-03). Before the
   patch this asserted `b_type ∈ {F32, F16, Q8_1}` and killed every
   qwen_decode-family engine at the first decode GEMV. Wide (prefill-shaped)
   matmuls already had a bf16xbf16 pipeline and were unaffected.

2. **Every elementary op is numerically exact on Vulkan.** `test_ops`
   verifies CPU-vs-Vulkan: rms_norm, add, mul, soft_max (64/640-wide),
   soft_max_ext with mask, rope, argmax, casts f32<->bf16, get_rows(bf16),
   set_rows into a bf16 cache, cache view+cpy write-back, GEMV T=1/7,
   wide matmul T=64/512, 3-D KV-cache-shaped batched GEMV — all maxdiff at
   fp-epsilon, argmax-equal.

3. **Composite prefill still diverges.** `stage_cmp` runs the moss engine
   stage-by-stage (mel -> encoder -> adapter -> inputs_embeds -> prefill
   logits -> ids) and prints per-stage FNV/sums. With patch 0009: mel is
   byte-identical; the encoder drifts slightly (f16-accumulation class);
   and even when the CPU run's inputs_embeds are force-fed into the Vulkan
   run (`STAGE_CMP_EMBEDS=/tmp/embeds_dump.f32`, byte-equal FNV), prefill
   logits are completely different and decode degenerates to repetition.
   With every constituent op verified exact, the divergence lives in the
   engines' composite prefill execution on Vulkan (ReplayGraph
   orchestration / persistent-state interaction / an op combination the
   probes do not replicate). Reproduce:

   ```bash
   STARLING_GGML_DEVICE=cpu      /tmp/stage_cmp              # writes /tmp/embeds_dump.f32
   STARLING_GGML_DEVICE=Vulkan0  /tmp/stage_cmp              # encoder drift visible
   STARLING_GGML_DEVICE=Vulkan0 STAGE_CMP_EMBEDS=/tmp/embeds_dump.f32 /tmp/stage_cmp
   ```

## Next steps

- Instrument the per-layer prefill logits inside `lib::forward_prefill`
  (dump after layer 1, 2, ...) to find the first diverging layer on
  Vulkan; compare against CPU with identical merged embeds.
- Check whether the first prefill ReplayGraph's captured state (persistent
  KV buffers + `add_graph_root` write-backs) interacts badly with the
  Vulkan submission model (one-shot re-run vs replay).
- Parakeet (no qwen_decode stack) runs on Vulkan but terminates TDT decode
  early — suspect f16-accumulating pipelines on matrix-core-less devices;
  try forcing f32acc pipelines.
