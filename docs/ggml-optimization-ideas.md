# ggml ASR engine — optimization-idea digest (mined from personal wiki)

**Scope.** Ideas mined from `/mnt/z/concepts/*.md` (and the one relevant
`/mnt/z/research` file) that have a plausible application to Starling's
in-tree ggml engine (`cpp/`): **parakeet-tdt-0.6b** (FastConformer encoder as
one CUDA graph ~2.8 ms + serial data-dependent TDT decode loop, K-step
multistep, sync once per K) and **MOSS-Transcribe-2B** (32-layer windowed
audio-transformer encoder + Qwen3 LLM decoder, port starting). Target:
byte-exact greedy transcripts vs the PyTorch reference on RTX 5090 (sm_120).

**Read-only mining output.** No code touched, no commits. Each idea below names
its wiki source, the mechanism, the concrete application (stage:
`mel` / `encoder` / `decode` / `mel-cache`), the expected gain class
(launch-overhead / memory-BW / compute / sync-elimination), the risk to the
exactness contract (none / token-level / byte-level), and rough effort (S/M/L).

**Hard constraint honored throughout.** The exactness contract
(`docs/ggml-roadmap.md` decision 3–5) forbids trading speed for exactness
*below* the contract. Ideas that cannot meet it are either flagged conditional
or moved to the appendix.

**Known negative results (not re-proposed without a new angle):** weight
quant (int8/fp8) loses to cuBLAS bf16 for tiny decode matmuls on this GPU;
graphing eager encoders that carry host-dependent ops fails; fused-Triton RoPE
diverges from ATen.

---

## 0. Frame every decision with the three-regime diagnostic (do this first)

- **Source:** `gpu-performance-regimes.md`.
- **Mechanism:** Horace He's compute / memory-bandwidth / overhead framework.
  Measure achieved-FLOPS%-of-peak to identify which regime each stage is in;
  only then pick a lever (overhead → tracing/fusion/graph; bandwidth →
  fusion/quant/rematerialization; compute → you win or change algorithm).
- **Application:** The two decoders sit in *different* regimes and the levers
  differ:
  - **parakeet `decode`** is **GPU-compute-bound at the serial step floor**
    (`ggml-parakeet-perf-analysis.md`: the ~6.5 ms K=64 readback is dominated by
    the graph *executing*, not the sync). Levers: **fewer serial steps**
    (spec-decode-with-rollback) and **eliminate residual host interaction
    across replays** (on-device loop). BW/launch fusion alone will *not* move
    it much.
  - **moss `decode`** is **memory-bandwidth-bound** on weight streaming
    (small cuBLAS GEMVs replayed under a graph; the negative-result note
    confirms quant can't beat bf16 here precisely because the matmuls are
    BW-bound). Levers: **fusion / megakernel** (keep intermediates on-chip,
    stream weights once) and **spec decoding** (fewer steps). This is the
    decode regime where the megakernel wiki material pays off most.
  - **moss `encoder`** is currently **overhead-bound** (eager, 25–40 ms) →
    graph capture is the dominant lever (see §1).
- **Gain class:** meta (determines which other idea applies).
- **Exactness risk:** none (profiling only).
- **Effort:** S.

---

## 1. Capture the MOSS audio encoder as one CUDA graph  ★ highest gain/effort

- **Source:** `inference-optimization-levers.md` (lever 4, "Kernel Fusion +
  CUDA Graph Capture"); `cuda-kernel-execution-stack.md` (why graphs win: one
  driver submission, one doorbell, no per-launch ioctl/QMD stream);
  `tts-inference-optimization.md` (multi-stage graph capture).
- **Mechanism:** Record the 32-layer encoder's op sequence into one captured
  graph and replay; collapses hundreds of per-op launches + host dispatch into
  a single driver submission. The parakeet encoder already does this and runs
  *faster than the PyTorch peak* on raw compute (2.8 ms).
- **Application:** `moss/encoder` — Phase 2e in the roadmap. The PyTorch
  encoder is *eager* 25–40 ms; a captured ggml `ReplayGraph` (the per-shape
  infrastructure already exists from parakeet) should beat it the same way
  parakeet's did. The encoder is conv2d ÷8 subsample + sinusoidal pos + 32×
  (LN → windowed attn → LN → GELU MLP) + adapter — a fixed-shape graph per
  audio length, ideal for capture.
- **Gain class:** launch-overhead + sync-elimination (the dominant moss win).
- **Exactness risk:** none — graph capture removes launch overhead only; op
  math is unchanged. Parakeet proved this byte-exact.
- **Effort:** M (ReplayGraph infra is reusable; main work is wiring the moss
  encoder graph + per-shape keying).

---

## 2. Port parakeet's proven encoder fusions to the MOSS encoder

- **Source:** `ggml-parakeet-perf-analysis.md` (the documented wins) +
  `moe-decode-kernel-fusion.md` (the fusion *pattern*, applied to the moss
  adapter's gated-MLP) + `hybrid-sliding-window-attention.md` (for the
  encoder's windowed attention).
- **Mechanism:** Parakeet already paid the byte-exactness cost for several
  encoder fusions; the same patches (`third_party/ggml-patches/`) apply
  directly to moss's structurally similar encoder. Concretely:
  - **NORM+MUL+ADD fusion** → moss uses LayerNorm ~128×; same `layer_norm_f32`
    + `ggml_cuda_can_fuse` patch collapses 3 nodes → 1 each.
  - **Conv depthwise-direct** (avoid `[k,T,C]` im2col materialization) → moss's
    conv2d ÷8 subsample is the same shape class as parakeet's conv module.
  - **f16 conv pointwise weights** → route conv onto the fp16 tensor-core
    cuBLAS path (parakeet's GGUF stored these as f32).
  - **Sliding-window flash-attn** for the encoder's *windowed non-causal
    attention via cu_seqlens* → express the window as a mask in
    `ggml_flash_attn_ext` (parakeet already added per-head-mask support) rather
    than materializing a full `[T,T]` mask. This is the moss-encoder instance
    of the SWA-kernel idea (`hybrid-sliding-window-attention.md`).
  - **Adapter SwiGLU fusion** (gate-up + SiLU + down in one kernel) → the moss
    adapter (2048→8192→2048 SiLU) is exactly the `moe-decode-kernel-fusion`
    dense-FFN pattern (gate-up GEMM + act + down GEMM, intermediate never
    leaves registers).
- **Application:** `moss/encoder` (all five sub-items).
- **Gain class:** memory-BW + launch-overhead (parakeet saw −16% to −24% on
  long encoder from the conv + NORM work alone).
- **Exactness risk:** none *if* the patch numerics are mirrored to the
  reference op order (parakeet already proved each of these byte-exact; the
  same fp32-reduction / reference-dtype mirroring applies).
- **Effort:** M (patches exist; integration + golden gating per stage).

---

## 3. Device-resident decode state between K-step replays (parakeet, incremental)

- **Source:** `cuda-kernel-execution-stack.md` (graph launch + return path,
  semaphore/DMA readback); roadmap Phase 3 item 1; `ggml-parakeet-perf-analysis.md`.
- **Mechanism:** Each K-step replay currently round-trips the LSTM h/c + frame
  counter + last-token through host (small H2D uploads + D2H readbacks per
  replay). Keep that state device-resident across replays and read back *only*
  the token/frame needed for the host-side termination check.
- **Application:** `parakeet/decode`.
- **Gain class:** sync-elimination + launch-overhead (kills per-replay
  H2D/D2H). Bounded gain because parakeet decode is compute-bound within a
  replay — this removes the *cross-replay* overhead, not the in-replay compute.
- **Exactness risk:** none (state is the same bits, just not round-tripped).
- **Effort:** M (rewires the KStepGraph input/output model; invasive but
  localized). This is the natural incremental precursor to §4.

---

## 4. Fully on-device TDT loop — device-terminated graph / on-device argmax megakernel (parakeet)  ★ flagged: serial decode loop

- **Source:** `megakernel.md` (the "keep the whole data-dependent loop
  on-device within one captured graph" idea — this is exactly how the starling
  PyTorch peak engine hits 60 ms on long); `tts-inference-optimization.md`
  (the **non-delayed inner-loop** analogy: TDT emits a *variable* number of
  tokens per step, which is the same graph-capture obstacle as non-delayed TTS
  inner loops — solved by looping *inside* the captured graph, not by host
  orchestration); `cuda-kernel-execution-stack.md` (one driver submission).
- **Mechanism:** Instead of "capture K steps, sync, host checks termination,
  replay next K," capture the *single-step* decode graph once and drive the
  repetition from the device: on-device argmax chains the next-step input,
  on-device predicate (next-token == EOS / frame-limit reached) exits the loop.
  Two implementation shapes, in rising risk:
  - **(a) Device-terminated graph replay** — the K3/MegaQwen-winning shape
    (`megakernel.md` counterexample): *refuse* the cooperative megakernel,
    keep a graph-captured sequence of bandwidth-saturating cuBLAS kernels, and
    eliminate the inter-K host sync by making termination device-side. Lower
    risk, reuses existing kernels.
  - **(b) Cooperative on-device megakernel** — Fable shape: one persistent
    cooperative kernel runs the whole loop with grid-wide barriers /
    producer-consumer handoffs. Higher ceiling, higher risk.
- **Application:** `parakeet/decode`. This is the roadmap's Phase 3 stretch and
  the documented mechanism by which starling's peak engine is 1.8× faster on
  long audio (108 ms → ~60 ms target).
- **Gain class:** sync-elimination + launch-overhead (removes *all* host
  interaction from the hot loop); shape (b) also memory-BW if it fuses ops.
- **Exactness risk:** **none** for shape (a) at the token level — the starling
  peak engine is byte-exact with exactly this approach, so it is proven
  byte-exact-safe. Shape (b) carries byte-level risk (in-kernel reduction
  order must mirror cuBLAS); the Fable lesson is that the dequant/reduction
  path must match the reference bit-for-bit or the TDT argmax flips a token.
- **Effort:** L.

---

## 5. Decode-step kernel fusion for the Qwen3 decoder (precursor to the megakernel)

- **Source:** `moe-decode-kernel-fusion.md` (the dense-FFN analog: fuse
  gate-up + act + down, fold nothing into HBM); `inference-optimization-levers.md`
  (lever 4, GEMM-epilogue reparameterization — fuse RMSNorm + residual + RoPE
  as epilogues onto the QKV GEMM).
- **Mechanism:** At moss's batch-1 decode, each step is a chain of tiny cuBLAS
  GEMVs (Q/K/V/O proj, SwiGLU gate-up, SwiGLU down, tied lm_head) separated by
  RMSNorm / residual / RoPE / softmax. The launch tax + HBM round-trips
  between them dominate because the matmuls are BW-bound. Fuse:
  - **SwiGLU FFN**: gate-up GEMM + SiLU + down-GEMM in one kernel (intermediate
    activation stays in registers) — the `moe-decode-kernel-fusion` pattern
    verbatim.
  - **GEMM-epilogue fusion**: absorb RMSNorm + residual add + QK-norm + RoPE
    into the QKV-projection GEMM epilogue so the hidden state never
    materializes to DRAM between ops.
- **Application:** `moss/decode` (Qwen3 decoder, 28 layers).
- **Gain class:** memory-BW + launch-overhead (directly attacks the BW-bound
  weight-streaming floor — the regime where fusion helps most).
- **Exactness risk:** **byte-level** — the fused reductions must mirror the
  reference op order (RoPE ATen-order, RMSNorm fp32-accumulate, SiLU order).
  Fused-Triton RoPE is a known divergence; a custom kernel must replicate
  ATen's RoPE bit-for-bit. This is the same mirroring discipline parakeet's
  NORM fusion already follows.
- **Effort:** M (each fusion is one vendored kernel + a `can_fuse` branch,
  mirroring the parakeet NORM-fusion precedent). This is the incremental path
  toward §6.

---

## 6. Cooperative decode megakernel for the Qwen3 decoder (moss)  ★ flagged: fully on-device serial decode loop

- **Source:** `megakernel.md` (Fable's single-fused per-token forward; the
  K3-vs-Fable sync-design-beats-instructions lesson; the MegaQwen
  counterexample as the caution).
- **Mechanism:** Fuse the *entire* Qwen3 decode step (all 28 layers' QKV+O
  proj, RoPE, RMSNorm, residual, SwiGLU FFN, attention over the bf16 KV cache,
  tied lm_head, argmax, KV-cache append) into one cooperative kernel launch,
  with intermediates in registers/SMEM and grid-wide barriers staging the
  forward. The win condition is the bandwidth-bound decode regime
  (`gpu-performance-regimes.md`) — exactly moss's regime — where eliminating
  DRAM round-trips and launch overhead dominates.
- **Application:** `moss/decode`.
- **Gain class:** memory-BW + launch-overhead + sync-elimination (all three).
- **Exactness risk:** **byte-level** — every in-kernel reduction (GEMV
  accumulator order, softmax upcast, RoPE) must mirror the reference or greedy
  argmax flips at low-confidence boundaries. The roadmap's moss contract
  already tolerates "token-exact modulo the reference's own instability" on
  long, so byte-level divergence at the *boundary* may be admissible; short
  must stay byte-exact. Manageable but real.
- **Effort:** L. **Caution from MegaQwen:** a cooperative megakernel can
  *lose* to a graph-captured sequence of bandwidth-saturating cuBLAS kernels
  if grid-wide barriers serialize stage tails across all SMs. Validate shape
  choice against the §5 fusion baseline before committing — the safer path is
  §5-fusion → graph-replay, escalating to a cooperative kernel only if the
  fusion baseline leaves BW headroom unclaimed.
- **Composes with §8** (speculative decoding): the megakernel verifies K
  drafted tokens for nearly the cost of one.

---

## 7. Speculative decoding with exact verification  ★ flagged: byte-exact-safe by construction

- **Source:** `speculative-decoding-mtp.md` (MTP/drafter architecture, exact
  verification, acceptance-rate sensitivity, DFlash/Laguna drafter patterns).
- **Mechanism:** A drafter proposes K tokens; the target verifies all K in one
  forward pass (the expensive part is already happening). The target always
  has final say on acceptance, so **the verified output is bit-identical to
  greedy decode** — speed-only, never correctness. This is the only
  high-impact lever that is *provably* byte-exact-safe.
- **Application:**
  - **`moss/decode` (Qwen3 decoder) — primary.** ASR transcript text is highly
    locally predictable given strong encoder audio context, so acceptance
    should sit in the high regime (>80% needed for ~1.6×; clear speech likely
    higher). Drafter options, cheapest first:
    1. A small external LM / n-gram over the transcript vocab (no retraining).
    2. An **MTP head** on the Qwen3 decoder's hidden states (Gemma-4 / GLM-5.2
       pattern) — shares the target's KV cache and embeddings, so the drafter
       is cheap and avoids KV-cache mismatch.
    3. A block-diffusion / hidden-state drafter (DFlash pattern) proposing a
       block of tokens per step.
  - **`parakeet/decode` (TDT) — secondary, riskier.** TDT already emits
    multiple tokens per step via T/D heads, so classical spec decoding is less
    natural. The applicable variant is **speculative multi-step decode with
    rollback** (roadmap item 2): batch a fixed lookahead, roll back on a blank
    token. Rollback must restore LSTM h/c + frame state *bit-exactly* or
    byte-exactness breaks — token-level risk.
- **Gain class:** compute (amortize one forward pass over K verified tokens →
  effectively K× fewer serial steps). For moss this multiplies with §6.
- **Exactness risk:** **none** (moss, exact verification by construction);
  **token-level** (parakeet rollback, if state restoration isn't bit-exact).
- **Effort:** L (drafter asset + verification wiring into the ggml graph;
  rollback state-save/restore for parakeet).

---

## 8. Pipeline host-mel with GPU + mel-feature cache reuse (streaming long audio)

- **Source:** `tts-inference-optimization.md` (the headline lesson: the biggest
  TTS wins are *outside* the model — host-to-device transfers, streaming stitch
  points); `embedding-inference-optimization.md` (padding tax / process
  one-at-a-time); `kvflash-paging.md` (graph-replayed prefill in fixed batches).
- **Mechanism:** Parakeet computes mel on the **host in double precision**
  (GPU float breaks byte-exactness) and it costs ~1.5 ms (short) → ~10 ms
  (long). Two composable moves:
  - **Pipeline mel(next chunk) on host with encode+decode(current chunk) on
    GPU** — hides the 10 ms behind GPU work once audio is chunked. Mel math is
    untouched, so byte-exactness holds; only scheduling changes.
  - **Mel-feature cache reuse across chunk boundaries** (`mel-cache` stage) —
    streaming/chunked ASR recomputes mel for overlap windows; cache the
    boundary frames and avoid recompute.
- **Application:** `mel` + `mel-cache` (both models; largest absolute win on
  long audio where mel is ~10 ms of the 108 ms).
- **Gain class:** sync-elimination + compute (hides host mel behind GPU;
  avoids recompute).
- **Exactness risk:** none (mel numerics unchanged; chunking must preserve the
  reference's framing/windowing exactly at boundaries — a staging concern, not
  a numerics one).
- **Effort:** M (requires a chunked/streaming decode architecture; the
  one-shot path can't overlap mel with anything before it).

---

## 9. KV-cache L2 residency + coalesced layout (moss decode attention)

- **Source:** `gpu-memory-hierarchy.md` (L2 write-back, coalescing, the
  balance-point math); `cutedsl-attention.md` + `gluon-attention.md`
  (decode-specialized attention: stream over KV, split-K for batch-1
  parallelism).
- **Mechanism:** At moss's short context (~200 tokens) the entire bf16 KV
  cache is tiny and should live in L2 (write-back means it may never reach
  VRAM). Ensure (a) coalesced KV layout (no scatter; vectorized `LDG.E.128`
  loads) so attention is L2-resident not DRAM, and (b) for batch-1 attention
  use a **decode-specialized kernel that streams over KV and splits the KV
  reduction across SMs** (split-K) rather than tiling the trivially-short
  query — the CuteDSL/Gluon shape.
- **Application:** `moss/decode` (attention); marginally `parakeet/decode`
  (TDT cross-attention to encoder states).
- **Gain class:** memory-BW + compute (better SM utilization at batch-1).
- **Exactness risk:** byte-level for a custom attention kernel (softmax/reduction
  order); none for the layout/coalescing-only part.
- **Effort:** S (layout/coalescing) → M (custom split-K decode-attention
  kernel). Bounded absolute gain at ~200-token context (attention is a small
  fraction of the 4.85 ms/tok, which is GEMV-dominated) — prioritize after §5/§6.

---

## 10. Minor knobs

- **Source:** `cuda-kernel-execution-stack.md`, `embedding-inference-optimization.md`,
  `gpu-warp-scheduling.md`.
- **Lazy-module pre-warm** (`cuda-kernel-execution-stack.md`: CUDA 12.2+
  lazy-loads SASS cubins on first launch, ~948 ioctls). Pre-touch every kernel
  path at startup so first-utterance latency matches steady state.
  `mel`/`encoder`/`decode`. Gain: launch-overhead (cold start only). Risk:
  none. Effort: S.
- **FTZ / flush-to-zero denormals** (`embedding-inference-optimization.md`:
  "kill denormals, especially on attention softmax"). Gain: small compute.
  Risk: **token-level** (denormal flushing changes results slightly; can flip
  a low-confidence argmax). Only enable above the exactness contract.
  Effort: S.
- **No-pad utterance batching** (`embedding-inference-optimization.md`: the
  padding tax — batching variable-length audio pads to the longest). If
  Starling ever batches utterances, process one-at-a-time or length-sort rather
  than pad. `mel`/`encoder`. Gain: compute. Risk: none. Effort: S.
- **ILP-over-occupancy kernel design** (`gpu-warp-scheduling.md`: well-tuned
  kernels deliberately run low occupancy with high in-loop ILP). Applies when
  writing the §6 megakernel or §5 fused kernels — design for hand-scheduled
  ILP, not max warps, and avoid warp divergence in any per-step branching.
  Gain: compute. Risk: none. Effort: folded into §5/§6.

---

## 11. Conditional — encoder-only MXFP4/NVFP4 weight quant (flagged, likely inadmissible)

- **Source:** `mxfp4-quantization.md`, `inference-optimization-levers.md`
  (lever 1).
- **Mechanism:** 4-bit microscaling block-quant with shared exponent; ~3.8×
  weight/BW reduction. The wiki evidence (GLM-5.2, Inkling) shows it
  *benchmark*-lossless vs FP8.
- **The new angle (vs the known negative result):** the negative result is for
  *tiny decode matmuls*. The **encoder** runs *large, prefill-style* matmuls
  where 4-bit has a real BW case. So quantizing only the moss/parakeet
  *encoder* weights is the untested angle.
- **Application:** `encoder` (both models).
- **Gain class:** memory-BW + (capacity).
- **Exactness risk:** **byte-level — almost certainly disqualifying.**
  "Benchmark-lossless" is *within-eval-noise*, not bit-exact; greedy argmax
  will flip on near-ties and break the byte-exact transcript contract. The
  roadmap (decision 4) only permits speed-for-exactness trades *above* the
  contract. **Verdict: set aside unless the contract is ever relaxed for the
  encoder stage, or as a non-default "fast" mode.** Do not pursue for the
  byte-exact path.

---

## Top 5 by gain-per-effort (risk-adjusted)

| # | Idea | Stage | Gain class | Exactness risk | Effort | Why it ranks here |
|---|------|-------|------------|----------------|--------|-------------------|
| 1 | **§1 Capture MOSS encoder as a CUDA graph** | moss `encoder` | launch-overhead + sync-elimination | none | M | Single biggest moss win; eager 25–40 ms → graph. ReplayGraph infra reused from parakeet. Byte-exact by construction. |
| 2 | **§2 Port parakeet's proven encoder fusions to MOSS** | moss `encoder` | memory-BW + launch-overhead | none | M | Proven byte-exact on parakeet (−16–24%); moss encoder is bigger (32 L). Patches exist; mostly integration + goldens. |
| 3 | **§4 Fully on-device TDT loop (device-terminated graph)** | parakeet `decode` | sync-elimination + launch-overhead | none (shape a) | L | The documented mechanism by which starling's peak hits 60 ms on long (we're at 108 ms). Proven byte-exact in the starling engine. The flagged "fully on-device serial decode loop." |
| 4 | **§7 Speculative decoding, exact verification** | moss `decode` (+parakeet rollback) | compute | none (moss) / token (parakeet) | L | The only high-impact lever that is *provably* byte-exact-safe (exact verification). ASR text predictability should give high acceptance. Composes multiplicatively with §6. |
| 5 | **§6 Cooperative Qwen3 decode megakernel** (via §5 fusion) | moss `decode` | memory-BW + launch-overhead + sync-elimination | byte-level | L | Attacks moss's BW-bound weight-streaming floor directly. MegaQwen caution: validate §5-fusion → graph-replay first, escalate to cooperative kernel only if BW headroom remains. |

**Honorable mentions (just outside the top 5):** §3 (device-resident decode
state — the M-effort incremental precursor to §4), §8 (mel pipelining +
mel-cache — big on long audio, needs streaming architecture), §9 (KV-cache L2
+ coalescing — S effort, guaranteed small gain).

**Sequencing note:** §1+§2 are the moss *encoder* track (do first; biggest,
lowest-risk gains). §3→§4 is the parakeet *decode* track (incremental to
full). §5→§6 and §7 are the moss *decode* track (fusion→megakernel, and
spec-decode, which compose). §7's exact-verification property makes it the
safest large moss-decode lever and the natural first move on that track.

---

## Considered and set aside (for transparency)

- **MLA** (`inference-optimization-levers.md` lever 2): neither model uses MLA
  (parakeet = FastConformer; moss = standard GQA). Not applicable.
- **MoE kernel fusion** (`moe-decode-kernel-fusion.md`, `inference-optimization-levers.md`
  lever 3): neither model is MoE. Only the *dense-FFN fusion pattern* transfers
  (folded into §2/§5).
- **TP topology** (`inference-optimization-levers.md` lever 6): single-GPU
  target (RTX 5090). Not applicable.
- **Subquadratic / linear / SWA-decoder attention** (`subquadratic-attention.md`,
  `hybrid-sliding-window-attention.md` decoder side): changing the attention
  mechanism requires retraining → breaks byte-exactness. SWA *kernel* ideas
  apply only where the reference already uses windowed attention (moss encoder
  — folded into §2).
- **CPU-GPU heterogeneous inference** (`cpu-gpu-heterogeneous-inference.md`):
  both models fit in VRAM (0.6 B / 2 B on 32 GB). The hot/cold-expert idea is
  MoE-specific. The only transferable thread — overlap host bookkeeping with
  GPU — is covered by §3/§4/§8.
- **KVFlash paging** (`kvflash-paging.md`): moss context is ~200 tokens; KV
  cache is tiny, no paging needed. Relevant only if long-form (>5 min) ASR with
  growing decoder context becomes a target.
- **Random-Hadamard / asymmetric quantization** (`random-hadamard-quantization.md`,
  `asymmetric-quantization.md`): RHT enables quant (ruled out by the
  byte-exact contract, see §11); asymmetric-quant is retrieval-specific
  (ColBERT). Not applicable.
- **Jenga memory layout** (`jenga-memory-layout.md`): multi-state KV cache
  decoupling; both models have single-state KV (no conv/SWA decoder state).
  Not needed now; revisit if moss decoder ever adds state types.
- **Flow-attention experiments** (`/mnt/z/research/flow-attention-experiments.md`):
  from-scratch retraining of novel attention/memory architectures.
  Fundamentally incompatible with byte-exact reproduction of fixed weights.
- **`/mnt/z/projects/*`**: no project name relates to ASR / ggml / CUDA graphs
  / kernel fusion; skipped per the selective-scan instruction.

---

*Sources mined: 20 concept pages under `/mnt/z/concepts/` + 1 research file.
Digest generated as a read-only analysis; no code modified, no commits made.*
