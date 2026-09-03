# Vulkan-backend diagnostics (ggml bf16-activation work)

Debug harnesses used to bisect the Vulkan backend's bf16-activation gap on
the AMD Ryzen 5 5650U / RADV RENOIR iGPU notebook (2026-09-02/03). Build from
the repo root against the built ggml shared libraries:

```bash
GXX=g++
INC="-I third_party/ggml/include -I third_party/ggml/src -I cpp -I cpp/include"
LIBS="build/third_party/ggml/src/libggml.so build/third_party/ggml/src/libggml-base.so \
      build/third_party/ggml/src/ggml-vulkan/libggml-vulkan.so.0 build/third_party/ggml/src/libggml-cpu.so.0"
RPATH="-Wl,-rpath,$PWD/build/third_party/ggml/src -Wl,-rpath,$PWD/build/third_party/ggml/src/ggml-vulkan -Wl,-rpath,$PWD/build"

$GXX -O2 -std=c++20 $INC -o /tmp/test_bf16_gemv      scripts/diagnostics/vulkan/test_bf16_gemv.cpp      $LIBS $RPATH
$GXX -O2 -std=c++20 $INC -o /tmp/test_ops            scripts/diagnostics/vulkan/test_ops.cpp            $LIBS $RPATH
$GXX -O2 -std=c++20 $INC -o /tmp/test_bmm            scripts/diagnostics/vulkan/test_batched_mulmat.cpp $LIBS $RPATH
$GXX -O2 -std=c++20 $INC -o /tmp/stage_cmp           scripts/diagnostics/vulkan/stage_cmp.cpp           build/libstarling_ggml.so $LIBS $RPATH
$GXX -O2 -std=c++20 $INC -o /tmp/stage_cmp_granite   scripts/diagnostics/vulkan/stage_cmp_granite.cpp   build/libstarling_ggml.so $LIBS $RPATH
```

(zsh does not word-split unquoted variables — use `${=INC}` / `${=LIBS}`
`${=RPATH}` or spell the flags inline. Compile with
`env -u LD_LIBRARY_PATH g++ ...`; the ZCode AppImage exports an
LD_LIBRARY_PATH that breaks the compiler's own internals.)

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
   fp-epsilon, argmax-equal. Notes: the leaf-seed pattern is injective so
   the argmax probe compares whole 32-bit indices without backend
   tie-break differences (tie order differs between backends by design —
   the engines' argmax_low_ties exists for that); rope uses the pinned
   ggml_rope_ext signature (mode, not a rope_type arg — a shifted call
   silently NaNs every frequency past index 0 on BOTH backends and
   vacuously "passes").

3. **RESOLVED — the composite prefill divergence was the wide batched
   mul_mat on a batch-strided (KV-cache view) operand.** `stage_cmp` /
   `stage_cmp_granite` run the moss/granite engines stage-by-stage (mel ->
   encoder -> merged embeds -> prefill logits -> ids) and print per-stage
   FNV/sums. Even with byte-equal force-fed embeds (`STAGE_CMP_EMBEDS=
   /tmp/embeds_dump.f32`; the CPU run writes the dump) prefill logits were
   garbage and decode degenerated to repetition, while the SAME graph with
   per-head attention (`STARLING_MOSS_PERHEAD=1` — needs `_NOKSTEP=1`, the
   per-head K-step graph overflows kGraphSize) and the legacy per-layer
   path (`STARLING_MOSS_DUMP_LAYERS=`, which routes llm_prefill through
   forward_legacy) were exact. That split exonerated the ReplayGraph
   orchestration and pinned the default batched attention
   (`mul_mat(kall[D,K,KV] bf16, q[D,S,H] bf16)` with GQA broadcast) —
   specifically its `kall`: the kv_mode=0 prefill assembles it as
   `ggml_cpy(k, view_3d(cache_k[D,max_cache,KV], ...))`, whose RESULT is a
   [D,S,KV] view with batch stride nb[2] = D*max_cache != D*S.
   ggml-vulkan passed the shader `stride_batch_x = ne00*ne01` (the
   contiguous assumption), so every kv batch except 0 read the wrong
   (zeroed) cache slots: heads mapped to kv 0 were exact, everything else
   attended to all-zero keys. `test_batched_mulmat` reproduces this in
   isolation (CACHE-VIEW probes: maxdiff 15.9 bf16 / 161.5 f32, only
   batches >= kv 1 wrong) and verifies the fix (patch 0010: pass the real
   nb[2]-derived stride for in-place operands, fold-guard nb[3] ==
   ne[2]*nb[2] via the contiguous-copy path, descriptor ranges sized to
   ggml_nbytes, ALIGNED gated on 8-divisible strides) — cache-view bf16
   back to fp-epsilon, moss + granite stage ids byte-identical CPU-vs-
   Vulkan on the default hot path. The two f16 `test_batched_mulmat`
   probes carry the documented f16-STORAGE-noise class (maxdiff 0.04,
   identical with and without striding and present in the plain f16
   control) and gate at 5e-2; bf16/f32 probes gate at 2e-2 — see 4.

4. **Known numeric classes that remain (not correctness bugs).** f16-stored
   operands show ~4e-2 storage noise vs CPU (the encoder-stage drift in the
   non-pinned stage_cmp runs); bf16 operands are exact. The Vulkan
   prefill logits sit within the same band as CPU's own
   batched-vs-per-head attention difference (fp-order), ids identical.

5. **Parakeet TDT early-termination: ggml-vulkan breaks the K-step
   multistep graph's SECOND+ replay of the same cgraph; mechanism not yet
   identified; worked around engine-side.** Parakeet ran on Vulkan but
   stopped transcribing early (WER 60.9-87.0% on the tiled fixtures vs
   0.0% CPU; bf16 and q8_0 alike). Bisect: `GGML_VK_DISABLE_F16` (f32acc
   pipelines — hypothesis falsified), `GGML_VK_DISABLE_FUSION` /
   `GGML_VK_DISABLE_GRAPH_OPTIMIZE`, `GGML_VK_SERIALIZE_SUBMISSIONS` /
   `GGML_VK_DISABLE_ASYNC`, `GGML_VK_MAX_NODES_PER_SUBMIT` both
   directions: no effect. The serial TDT loop
   (`STARLING_GGML_TDT_SERIAL=1`), the serial+fused path
   (`STARLING_GGML_TDT_KSTEP_MAX_T=1`), and a single-replay K-step graph
   (`STARLING_GGML_TDT_KSTEP=96` on the short fixture) are EXACT on
   Vulkan — every replay AFTER the first of the SAME captured K-step
   cgraph corrupts (both the device-resident add_graph_root sub-path at
   K<=16 and the host round-trip baseline at K>16).
   `STARLING_GGML_TDT_KSTEP_DEBUG=1` (enriched with the full token/frame
   ring + post-replay device-cache dumps) traces it: replay 1's ring is
   sane and the device-cache leaves read back correct (frame=26,
   write-back cpys fine), then replay 2's ring frames jump to
   f32-bit-pattern garbage (1073741824.0 = float(0x40000000), dur_final
   = 1065353216.0 = float(0x3F800000)) — the in-graph chain consumed
   wrong bytes even though the host readback of the same leaves is
   correct. The mechanism inside ggml-vulkan is UNIDENTIFIED (an earlier
   "argmax reads stale zeros" minimal-repro claim was RETRACTED — its
   input fill underflowed size_t and the "divergence" was legitimate
   argmax tie-breaking over ~2^64-scale floats; see PR #45 discussion).
   **Workaround (this repo):** tdt.cpp gates the K-step multistep
   capture off on Vulkan* devices and decodes via the serial loop +
   per-step fused graph (exact; one host<-device sync per step).
   `STARLING_GGML_TDT_KSTEP_FORCE=1` re-enables it for re-validation
   after a ggml-vulkan fix. Validation: short/medium/long fixtures now
   produce the full tiled transcripts on Vulkan (bf16 + q8_0); the long
   tier differs from CPU by one punctuation token (~0.7% WER; fp-tie
   class, same family as the encoder drift), vs 79.6% before.

## Reproduce the moss stage comparison

```bash
STARLING_GGML_DEVICE=cpu      /tmp/stage_cmp              # writes /tmp/embeds_dump.f32
STARLING_GGML_DEVICE=Vulkan0 STAGE_CMP_EMBEDS=/tmp/embeds_dump.f32 /tmp/stage_cmp
```

The plain (non-OVR) run OVERWRITES the dump — run CPU last before any OVR
comparison. granite: same with `/tmp/stage_cmp_granite` and
`/tmp/embeds_dump_granite.f32`.

## Next steps

- **ggml-vulkan second-replay corruption (finding 5):** identify the
  mechanism — the entry point is the engine itself with
  STARLING_GGML_TDT_KSTEP=2 + STARLING_GGML_TDT_KSTEP_DEBUG=1 on the short
  fixture (corrupts at the very first replay boundary; VK_LAYER lunarg
  validation + a bisect of the K-step graph's op set are the next tools).
  Then patch ggml (patch 0011 candidate) and drop the tdt.cpp Vulkan gate
  (validate with STARLING_GGML_TDT_KSTEP_FORCE=1 + the fixtures). The
  moss/qwen K-step graphs (bf16 activations, no tiny i32 get_rows chains)
  do NOT trigger it and stay on the multistep path.
- higgs wrong output on CPU (byte-exact on CUDA per repo docs) — same
  dtype-discipline family; the repo's staged golden-component probes are
  the tool (golden files are dev-machine artifacts, not in git).
- Optional upstream follow-up: ggml_vk_mul_mat_id_q_f16 /
  ggml_vk_mul_mat_vec_id_q_f16 (MoE expert matmuls) still derive batch
  strides from the contiguous assumption; no starling engine feeds them a
  batch-strided operand, but the 0010 fix pattern applies.
