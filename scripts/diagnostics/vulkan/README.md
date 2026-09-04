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
$GXX -O2 -std=c++20 $INC -o /tmp/stage_cmp_higgs     scripts/diagnostics/vulkan/stage_cmp_higgs.cpp     build/libstarling_ggml.so $LIBS $RPATH
$GXX -O2 -std=c++20 $INC -o /tmp/higgs_fe            scripts/diagnostics/higgs/higgs_front_end_cmp.cpp  build/libstarling_ggml.so $LIBS $RPATH
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

5. **RESOLVED — parakeet TDT early-termination was ggml's gallocr
   recycling INPUT tensor storage (patch 0011), not a Vulkan bug.** The
   K-step multistep graph seeds its constant inputs (enc_proj, duration
   table, masks) ONCE per utterance and replays the same cgraph per K
   decode steps — a contract ggml's allocator silently broke:
   GGML_TENSOR_FLAG_INPUT only affects allocation ORDER, and the
   live-range reuse (free_node when n_children/n_views hit zero)
   recycled an input's chunk after its LAST IN-GRAPH READ, handing it
   to late intermediates. Compute #1's tail then wrote f32 values over
   the once-seeded i32 tables (dispatch log: dur_final's add lands at
   byte 270880 inside dur_tbl's chunk), and every replay #2+ re-read
   the poison: the gathered i32 = raw f32 bytes (0x40000000 = 2.0f)
   inflated by the i32->f32 VALUE cast to 1073741824.0 — the exact
   ring-garbage signature. Four-way confirmation: (a) the dispatch
   streams of replay #1 vs #2 are byte-identical while the device
   bytes differ (no dispatch-state bug); (b) the aliased offsets are
   visible in the -DGGML_VULKAN_DEBUG dispatch dump; (c) patch 0011
   (ggml-alloc.c: never free/reuse-in-place INPUT-flagged nodes,
   mirroring the OUTPUT exemption) cures it with the once-per-utterance
   seeding intact; (d) the discriminating control — WITHOUT the patch,
   re-uploading every once-seeded input before each replay also cures
   it. Every earlier symptom follows: both multistep sub-paths corrupt
   identically (both seed once), removing the tiny debug captures
   worsened it (one_t was INPUT+OUTPUT and thus protected — dropping
   its capture un-protected it), single-replay graphs are exact, all
   GGML_VK_* knobs were irrelevant, and the moss/qwen K-step survived
   because its engine re-uploads its constants every replay (an
   accidental defense). Upstream master has the same INPUT/OUTPUT
   asymmetry (not filed upstream — repo policy). The PR #45 Vulkan
   gate on the multistep path is reverted by the patch-0011 PR; the
   K=48/K=128 "CUDA-graph topology defect" inexactness documented at
   tdt_multistep.cpp's kstep notes is plausibly the same gallocr bug
   at other shapes — worth re-checking on the CUDA machine with 0011
   applied. Validation after 0011 + un-gate: K=2/K=16/default
   multistep produce the exact CPU transcript on short/medium/long
   (bf16 + q8_0); moss/granite stage ids, test_ops, test_bmm all
   unchanged.


6. **RESOLVED — higgs CPU garbage output was a dtype-blind readback in the
   engine's own Backend::compute/ReplayGraph (plus two follow-ons).** higgs
   on Vulkan was CORRECT all along (the fused encoder+projector ReplayGraph
   never round-trips its BF16 pooled tensor through the host), while CPU
   (and the one-shot debug path on any backend) produced "inaudience"
   repetition (WER 100%). Bisect: stage comparators (stage_cmp_higgs, new)
   localized the corruption between the encoder layers and the projector;
   per-layer LAYER_CUT probes showed all 32 layers + avg_pool agree between
   backends; a trusted output-tensor probe of the LN stage matched — and
   exposed the mechanism: run_graph read `n * sizeof(float)` bytes out of
   the BF16 `pooled` graph output, copying HALF the tensor and leaving the
   rest of the caller's vector uninitialized (the higgs code even documents
   a "run_graph converts the BF16 graph output to f32 on readback"
   conversion that never existed). Three-part fix: (a) Backend::compute and
   ReplayGraph now convert non-F32 outputs/captures elementwise (F16/BF16
   convert; I32/F32 stay bit-compatible — the K-step token rings rely on
   that); (b) the higgs host conv front-end rounds at the two BF16 oracle
   boundaries (conv1->conv2, conv2->layers) like ark's host path and the
   graph path's gelu_erf_bf16 stores — without these the 32-layer encoder
   amplifies the sub-ulp deltas chaotically; (c) higgs's frozen ~600-line
   LLM port is replaced by the shared lib/qwen_decode stack (spec: untied
   lm_head + im_end as eos2) — the port's CPU decode fell into a 4-token
   repetition loop the shared stack does not have. Validation: higgs
   produces full correct transcripts on short/medium/long on BOTH cpu and
   Vulkan0 (WER 0% vs 100% before); moss/granite stage ids unchanged;
   parakeet fixtures (incl. the K-step i32-token-ring captures through the
   new readback) unchanged; test_ops/test_bmm green. The qwen3/s1/audex
   llm.cpp files are the same era of ports — worth the same migration.

## Reproduce the moss stage comparison

```bash
STARLING_GGML_DEVICE=cpu      /tmp/stage_cmp              # writes /tmp/embeds_dump.f32
STARLING_GGML_DEVICE=Vulkan0 STAGE_CMP_EMBEDS=/tmp/embeds_dump.f32 /tmp/stage_cmp
```

The plain (non-OVR) run OVERWRITES the dump — run CPU last before any OVR
comparison. granite: same with `/tmp/stage_cmp_granite` and
`/tmp/embeds_dump_granite.f32`.

## Next steps

- **CUDA machine follow-ups:** run tests/test_ggml_parity.py over PRs
  #43/#44/#45/#46 (the gate), and re-check the K-step K-sweep with patch
  0011 applied — the K=48/K=128 "topology-defect" inexactness may have
  been the same gallocr input-reuse bug (finding 5).
- higgs wrong output on CPU (byte-exact on CUDA per repo docs) — same
  dtype-discipline family; the repo's staged golden-component probes are
  the tool (golden files are dev-machine artifacts, not in git).
- Optional upstream follow-up: ggml_vk_mul_mat_id_q_f16 /
  ggml_vk_mul_mat_vec_id_q_f16 (MoE expert matmuls) still derive batch
  strides from the contiguous assumption; no starling engine feeds them a
  batch-strided operand, but the 0010 fix pattern applies.
