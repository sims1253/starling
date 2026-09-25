# Fast-engine research log (Pixel 10 Pro)

Baseline: branch head `94230cf`. Workload: `starling-bench` on the phone,
`STARLING_ENGINE=fast`, median of 3 runs after warm-up (`pixel_measure.sh`;
`phone_ab.sh` for alternating A/B).
Gates: fixture transcripts identical (G1), FLEURS WER within 0.2 pts (G2, at
milestones), `fast_weights_test` (G3), desktop RADV ≤ 10 % regression (G4).

| # | Hypothesis | Change | Before → after (median) | Gates | Verdict |
|---|---|---|---|---|---|
| 0 | Baseline | — | PK medium 2886 ms (mel 26, enc 2612, dec 248); MOSS short 7183 ms (enc+prefill 3675, decode 3280 @102.5 ms/tok); phone_ms 10069 | G1 captured | — |
| 1 | Autotuner mis-ranks tiles on PowerVR (synthetic shapes unrepresentative) | env sweep first: TILE 32,64,4,4 → PK 2907→2526 (−13 %); GEMV rows 8 already best (4/16/32/64/128 worse) | — | — | measured, see #2 |
| 2 | Same, structural | Added candidates {32,64,4,4},{48,64,4,4}; tune recording now barriers after every kernel + norm interleave; M 256→293; **vendor 0x1010 default tile 32,64,4,4 + gemv rows 8** (tuner kept for other vendors) | phone_ms 10069 → **9363** (PK 2886→2529, enc 2612→2288; MOSS 7183→6834, decode 99.2 ms/tok) | G1 ✓ | **keep** (`91dd41f`) |

Notes:
- Tuner on-device picks by isolated synthetic recordings: 3 attempts to make
  it representative (barriers, norms, odd M) all still rated 64,32,4,4 ≥
  32,64,4,4 while reality differs ~2×. Per-dispatch tuning on PowerVR is
  dominated by fixed overheads; end-to-end wall time is the only trustworthy
  signal → vendor-keyed measured defaults.
| 6 | GEMV workgroups with lanes=K/32<128 threads waste PowerVR issue slots (subgroup=128) | RSPLIT row slots pad the workgroup to a full subgroup (wg=lanes×RSPLIT); two-stage tree reduction kept | W4 K=1024 GEMV 7.0→13.6 GB/s isolated (CPU-ref checked, rel err 0); MOSS decode 3203→3060ms (95.6ms/tok); phone_ms 9285→9151 | G1 ✓ | **keep** (`4388153`) |
| 7 | Milestone gates | desktop RADV: PK medium 429ms (was 522), MOSS short 1759ms (was 1820) — no regression, faster; fast_weights_test all pass; STARLING_FAST=OFF builds; FLEURS en_us 100: PK fast 5.33% vs ggml 5.47%, MOSS q4e8 fast 7.87% vs 7.92%, MOSS q4e4 fast 8.06% (+0.14, inside gate, −7% decode) | — | G1-G4 ✓ | milestone |
| 8 | GEMM tile landscape after ALU probe (peak ~500 GFLOPS vs 155 achieved) | swept wg=128-compatible configs | 32,128,4,8 best: 1992-2033ms enc (vs 2242) — full subgroup + BN=128 halves weight re-reads; vendor default updated | G1 ✓ | **keep** (`4fd9045`) |
| 9 | Shared-tile double buffering hides load latency | two-buffer A/B tiles | PK enc 2029→3118ms — **54% worse**: doubled shared halves occupancy; reverted | — | discard (occupancy > latency hiding on PowerVR) |
| 10 | PK decoder single-threaded (0.9ms/step, ~30× above ALU floor) | NEON/AVX2 quantize; `cpu::GemvHelper` persistent spin worker, row-split GEMVs (>1M MACs); decode-stage timing env | decode split: joint 106→57-67ms, pred 138→124ms → decode 252→201ms; phone_ms 8812→8615 | G1 ✓ (row split is order-identical); desktop decode 61→36ms | **keep** (`4be985e`) |
| 11 | Milestone: long fixture + desktop re-gates | — | PK long 9000→7579ms; desktop PK medium 421ms (was 522), MOSS 1759ms (was 1820); fast_weights_test all pass | G1-G4 ✓ | milestone |
| 12 | Energy per transcription (batterystats power model, 40/20-run averages) | — | PK medium: fast ≈1.4 mWh vs ggml ≈5.0 (3.6×); MOSS short: fast ≈2.2 vs ≈5.2 (2.4×). MOSS/fast: GPU 10.2 mAh + CPU small; ggml: CPU 24.5 mAh | — | measured |
| 13 | MOSS mel thread count (shared frontend uses all 9 cores) | STARLING_MEL_THREADS sweep | 1: 124ms, 2: 269(!), 3: 124, 9: 216 — scheduling noise dominates, no reliable win; real fix is a faster FFT path | — | skip (noise) |
| 14 | KSTEP (decode tokens per submission) | env sweep 4-64 | 4-8 ≈ 3098ms vs 16 ≈ 3141 — ~1% at best, within noise | — | skip |
| 15 | x re-reads dominate GEMV for large N (ff_up: 12.6MB x vs 15.7MB weights at rows=8; lm_head reads x 19k×) | adaptive gemv_rows: grow while N/rows≥384, cap 32 (48/64 regress on registers); env pins rows exactly | MOSS decode 3050→2821ms (88.2ms/tok); phone_ms 8615→8358.8; desktop unchanged. NOTE: old global rows sweep predated RSPLIT — landscape flipped | G1 ✓ (reduction order row-independent) | **keep** (`d9eb75b`) |
| 16 | MOSS mel: thread spawn per parallel_for call + hypot | persistent parked pool; hypot→double sqrt | pool: no win (big.LITTLE join-gating); sqrt: **flipped fixture transcript** (gate 1, reverted) — mel is numerics-pinned | G1 ✗ | discard ×2 |
| 17 | GEMM inner loop shared-transaction bound → uvec2 wide loads (6→3 transactions/k) | f32 product loop rewrite | PK enc 2082-2095 vs 2058-2066 — neutral; glslc already coalesces | — | discard |
| 18 | Norm kernels ~10× off memory speed (0.29ms/dispatch for 1.75MB r/w; new norm micro probe) | subgroup-shuffle reduction (1 barrier instead of 8-per-statistic tree) | **4× SLOWER** on PowerVR (1.23 vs 0.29ms) — subgroup shuffles are not cheap here; reverted. Norm cost totals only ~1-2% per model anyway | — | discard |
| 19 | Mel filterbank multiplies all 201 bins per (mel,frame); filters are ~2/3 zeros | per-mel nonzero [lo,hi) ranges (PkMel-style); bit-exact zero skipping; STARLING_MEL_TIMING phase probe | filterbank 82-100→12-14ms (7×); MOSS mel 200→131ms; phone_ms 8358.8→8319.1; desktop mel 6.5→3.2ms | G1 ✓ (bit-exact: products ≥ +0.0, x+0.0==x) | **keep** (`1e15bee`) |
| 20 | qkv rows check (N=3072 keeps rows=8 under the ≥384-WG rule) | micro rows 8/16/24 | 16.2 / 14.9 / 15.2 GB/s — rows=8 already optimal; heuristic stands | — | no change |
| 21 | Milestone: certify bit-exact keeps at corpus scale + refresh energy | FLEURS en_us 100 (fast) | PK 5.33 %, MOSS 7.87 % — identical to pre-change values; energy PK 1.33 mWh (3.8× vs ggml), MOSS 2.05 mWh (2.5×) | G2 ✓ | milestone |
| 22 | dec_attn: 3.9-4.2 ms/token (new attn micro probe: `STARLING_FAST_MICRO=attn[,pos[,reps[,maxpos]]]`); only ~16µs/token is FLOPs, 16 head-WGs = latency-dominated | (a) vec2 q+k stats + max-scan (barriers 30→16); (b) depth-4 rolling V-load prefetch (add order preserved) | (a) 0.151→0.158 ms, worse at pos=1000 (scan adds O(pos) reads; barriers already overlap across WGs); (b) 0.157 ms neutral — loads already pipelined. Both falsified, reverted | — | discard ×2 |
| 23 | Verification of kept state after reverts (cool phone) | — | **phone_ms 8143** (session best): PK 2184 (enc 1974), MOSS 5959 (mel 83, decode 88.5 ms/tok) | G1 ✓ | verification |
| 24 | lm_head W4 requant (brief: "28% of decode bytes") | convert_w8_w4 + STARLING_FAST_LM4=1; embed table W8→W4 at load (paths already exist for q4e4) | isolated lm_head: W8 already 36.6 GB/s (9.57ms), W4 22.1 GB/s (8.78ms) — W4's ALU-per-byte caps it; net 0.79ms/token = 0.9% decode = sub-noise at composite. Transcripts identical, −149MB, load −0.3s. Machinery works; not worth the FLEURS gate cost | G1 ✓ (kept code discarded) | discard, documented |
| 25 | Verification after reverts | — | phone_ms 8140.9 (session best 8143 within 0.03%) | G1 ✓ | verification |
| 26 | Brief's load-time idea: cache repacked blobs on disk (mmap) to skip repack | full PackCache built+verified (content fingerprint, parallel mmap carve, offset-reserved parallel pwrite, atomic rename; transcripts identical) | **net loss on this phone**: PK cold+write 3320ms (vs 455 repack), warm 320-345ms; MOSS cold 6197ms (vs 1324), warm 2134ms — 1.6GB doesn't fit page cache. Phone flash: ~200MB/s write, ~0.8-1.5GB/s read; the parallel repack (~1.2GB/s) is already storage-speed. Reverted; design documented (attractive on NVMe/desktop) | — | discard, data-closed |
| 27 | Verification after revert | — | phone_ms 8331 (thermal band vs best 8143, identical code); baseline loads restored (repack 1347ms) | G1 ✓ | verification |
| 28 | Reviewer round: GEMV immediate partial stores | RPP>=16 variant captured ff_up (+10%)/lm_head (+6%) isolated | **in-context decode 4-7% WORSE** (alternating A/B, same thermal window) — isolated probes do not predict interleaved decode; reverted | G1 ✓ | discard |
| 29 | Reviewer round: narrow-N GEMM tiles | N≤64 ops (attention PV dk=64, subsampling) use 32,64,4,4 instead of BN=128 | PK encoder **−2.7%** (alternating A/B 3/3 rounds: 2010-2035 → 1957-1980); bit-exact (tile partitions outputs only); desktop unchanged; MOSS unaffected (head_dim 128) | G1 ✓ | **keep** (`c604898`) |
| 30 | Reviewer round: OpSDot probe (glslang has no GLSL front-end for it) | hand-assembled SPIR-V via spirv-as (`shaders/idot_probe.asm`), prebuilt-.spv embed support in fast.cmake, `STARLING_FAST_MICRO=idot` probe | **OpSDot WORKS on this driver** (exact correctness; coopmat's compiler crash does not extend to integer dot). ILP probe: ~154 G i8-MAC/s vs 77.5 G f32 MAC/s for the scalar GEMM ≈ **2× headroom** for an int8-activation GEMM | — | infrastructure keep |
| 31 | Review round (PR #287): bounds / CI / idle-worker fixes | coopmat stub removed, GEMV rows>WGS, MOSS hgroup + token buffer, tuner OOB, Android configure without spirv-as; decoder worker parks when idle | alternating A/B vs `283a21e`: PK 2163-2200 vs 2185-2226, MOSS 5975-6008 vs 5932-6019 (noise); NDK glslc = host glslc | G1 ✓, weights test ✓ | keep (`be39f3e`) |
| 32 | Parked decoder worker wakes cold at decode start | first version parked between jobs: PK decode median 204 vs 173 ms (+17 %, 4×12 runs), the woken thread lands on a little core; fix: the engine holds the worker spinning for one transcription | decode 197.7 vs 196.6 ms (parity), no idle burn between transcriptions | G1 ✓, helper test bit-exact | keep (`be39f3e`) |
| 33 | Decode GEMVs are issue-bound, not bandwidth-bound: W8 36.6 GB/s ≈ W4 22 GB/s ≈ 33 G weights/s (#24) → cut ALU per weight | `gemv_w4u`: mask one nibble per byte, `unpackUnorm4x8` → 4 floats (~5 ops / 8 weights vs ~16), x permuted to even/odd order, ×255 folded into the scale | isolated +12..60 %; MOSS decode 2950 → 2575 ms (alternating, 3/3), 88.5 → 76.8 ms/tok; **phone_ms 8143 → 7823**; RADV slower (38 → 31 GB/s) → PowerVR default only | G1 ✓ (short/medium/long identical), G2 ✓ MOSS FLEURS 7.87 % both, 0/100 differ | **keep** |
| 34 | The same A/B exposed an init bug in the first W4U build: PowerVR re-ran the cached tuner result over the vendor tile | control flow fixed before commit | PK encoder 2330 → 1980 ms (the stale cached tile is 17 % slower) | — | fix |

Next levers (not pursued; the PR is ready to merge): the same issue-bound
argument applies to the W8 lm_head (28 % of decode bytes; `unpackSnorm4x8`
needs a proof that no -128 codes occur) and, larger, to int8-activation
GEMVs with `OpSDot` (works on this driver; needs a glslc newer than the
NDK's and a WER run).

## #319 Phase 0: layout descriptors, source quantizer, packed-file override (2026-09-25)

Infrastructure (branch `feat/pixel-layout-quantizer`, on top of master
`4276631`): `cpp/fast/layout.*` (parameterized descriptor + RTN/imatrix
scale-search quantizer + bit-exact reference dequant), `cpp/fast/packed_file.*`
(SFPK file, `STARLING_FAST_PACKED` override in the MOSS engine),
`cpp/tools/starling_layout_quant` (pack / eval / verify / cmp),
`benchmarks/fast_engine/layout_eval.py` (desktop weight-only WER, cached),
`fast_weights_test` descriptor checks, glslc `GL_EXT_integer_dot_product`
capability probe. Source pinned: MOSS snapshot `c98175cb`, sha256
`6dce3c8c…`, imatrix `models/moss-full.imx`.

Desktop numbers (this machine, RADV, FLEURS 100 clips/lang; "eval" =
weight-only protocol: eval GGUF through the ggml engine; baseline =
ggml on `q4e8-fullimx` raw):

| layout | en | de | ta | note |
| --- | --- | --- | --- | --- |
| q4e8 GGUF raw (baseline) | 7.50 | 106.04 | 155.82 | ggml's Q4_0/Q8_0 draw |
| bf16-exact, unquantized | 8.13* | — | — | *10 clips; q4e8 on the same 10 = 9.09 |
| rebuild w4g32sym-a + w8g16sym embed | **7.31** | 105.91 | 174.73 | **sanity ✓**; fast engine + pack = 7.31 too |
| w4g64sym | 8.49 | 104.63 | 129.33 | +1.0 en for −6 % bytes |
| w4g128symu8s | 8.30 | 102.55 | 153.52 | u8-super works end-to-end |
| w8g32sym (native Q8_0 blocks) | 7.78 | 101.83 | 146.12 | bf16 store costs W8 ~0.4 pt — use native blocks |

Findings:

- **Free-offset asymmetric W4 loses.** `w4g32asym` beats ggml Q4_0 on the
  imatrix-weighted MSE (8.0 vs 10.3 on synthetic, −25 % on real tensors) yet
  costs **+1.9 pt en WER** (9.76 vs 7.87 fast engine, 9.95 vs 7.50 eval).
  Symmetric with the same search lands at −0.2 to −0.6 pt *better*. The
  offset freedom overfits the importance-weighted objective; the bulk of the
  weights (near zero) loses grid resolution to the weighted tails. Q4_1-style
  layouts are dead for MOSS — symmetric it is.
- **The rebuild sanity gate passes with the symmetric scheme** (`w4g32sym-a`:
  Q4_0 values in the legacy W4 pair bytes, engine-identical layout):
  fast-packed 7.31 vs GGUF-repacked 7.87, transcripts differ on ~60/100
  clips (expected: a different quantization draw), de 105.91 vs 106.04 ✓.
  Tamil (ta_in, English-only model, > 100 % WER hallucination regime) moved
  +19 pt — out-of-distribution hallucination is extremely weight-draw
  sensitive; treat tail-language gates at this WER level as qualitative.
- **±0.5 pt en deltas are numerics-draw noise, not quality.** The
  unquantized bf16 model scores *worse* than the Q4_0-quantized one (8.13 vs
  9.09 on the same 10 clips; ggml-vulkan, greedy decode). Candidate
  comparisons should rank by eval deltas ≥ ~1 pt and confirm winners with the
  fast engine on the full 100.
- Protocol traps documented in the code: F16/F32 eval weights trip ggml
  asserts (the MOSS bf16-oracle graph feeds bf16 activations into
  unquantized-weight muls; quantized weights get F32 — use bf16 or native
  block types); the eval GGUF must inherit the q4e8 model's F32 1-D tensors
  (graph-fusion parity — measured no WER effect, kept anyway); a bf16 store
  costs ~0.4 % per weight, material for W8 candidates only.

## #317 Phase 1 loop: int8/OpSDot is dead on PowerVR; the GEMV plateau is structural (2026-09-26)

Branch `autoresearch/pixel-layout-2026-09-25`. Phone protocol learned the
hard way: screen OFF (the compositor shares the GPU: ±100 % swings), order-
balanced A/B rounds (the second side of a round runs +3-8 % warm), ±2.5 %
same-window noise floor on identical binaries, absolute numbers drift per
boot (68-77 ms/token) — only same-window deltas count. GPU driver health
degrades after ~8 model loads per boot (vkWaitForFences VkResult 2, then
startup hangs); reboot between sessions.

| # | Hypothesis | Result |
| --- | --- | --- |
| P1-1 | int8 activations × W8 lm_head GEMV via `dotPacked4x8EXT` cut the issue-bound op count ~2x | correct (micro rel err 1e-3, transcripts identical) but NO speed change: 42.5 vs 43.4 GB/s isolated at the real lm_head shape; end-to-end +0.65 % (noise). **Root cause: `OpSDot` costs ~2 issue slots on DXT — the same slot budget as the f32 `dot(vec4)` it replaces.** The #30 probe's "2x headroom" was MACs, not slots. |
| P1-2 | packSnorm4x8 x-quant (1 op per 4 values) + 4-chain ILP unlock it | v2 flat (39.0 vs 38.7 GB/s), v3 (4 chains) WORSE (34.1). Idea stopped after three variants: **dead**. |
| P1-3 | skinny-GEMM pricing for #311 | tiled GEMM W4 6144x2048: flat 7.1-7.3 ms at M=1,2,4,8,16 — the M rows are free, but M=1 already costs 6.8x a GEMV pass (1.04 ms). Batch verification must use a **dedicated M-token GEMV kernel** (weights unpacked once, M x-vectors: ~1.6-1.8x fewer ops/token at M=4-8, and the unpacks amortize exactly where the plateau hurts). That kernel is #311's enabling work. |
| P1-4 | f16 `dot(f16vec4)` rate | 377 G f16-MAC/s = 4 MAC/issue-slot (2.6x the f32 FMA's MAC/slot) — the same per-slot efficiency as OpSDot. F16-layout GEMV measured 46.5 G w/s / 93 GB/s: memory has ≥2x headroom over the W4/W8 rates. No f16-dot GEMV rewrite either: the surrounding ops, not the dots, set the ~40 G w/s plateau all GEMV variants share (latency/occupancy class limit). |

Baseline this boot: MOSS decode 73-77 ms/token (matches the #33-era
per-operand mix; the historical 76.8 sits inside the boot-to-boot band).

Conclusion for the layout question (#317): within affine weight layouts
(W4/W8/F16 × group sizes × scale dtypes × nibble orders) the decode GEMV on
this GPU is at a structural plateau — layout changes move quality and
bytes, not decode speed. The speed lever that survives measurement is
**token batching in a GEMV-shaped kernel** (M-token GEMV for speculative
decoding, #311): the M rows are provably ~free at the op level, unlike the
tiled GEMM. Quality side (Phase 0 table): symmetric-only W4, rebuild-from-
source ≥ GGUF draw, g64 costs ~1 pt.

### #317 acceptance table (every layout tried)

Quality = desktop weight-only eval (ggml engine, FLEURS 100/lang; q4e8 GGUF
raw baseline = en 7.50 / de 106.04 / ta 155.82). Bytes = per linear weight.
Decode cost = phone GEMV weight rate; every quantized layout lands on the
same ~37-46 G w/s plateau — on this GPU the layout moves quality and bytes,
not decode speed.

| layout | en | de | ta | bits/w | decode |
| --- | --- | --- | --- | --- | --- |
| w4g32asym (free offset, imx search) | 9.95 | 101.4* | 146* | 4.5 | plateau |
| w4g32sym-a (Q4_0-shaped, imx search) | **7.31** | 105.91 | 174.73 | 4.5 | plateau (= today's W4 bytes) |
| w4g32sym lean store (+ -p1 nibble perm) | = sym (decode-transparent, tested) | — | — | 4.25 | kernel variant unwarranted |
| w4g64sym | 8.49 | 104.63 | 129.33 | 4.25 | plateau |
| w4g128symu8s | 8.30 | 102.55 | 153.52 | 4.125 | plateau |
| w8g32sym (Q8_0-shaped) | 7.78 | 101.83 | 146.12 | 8.25 | plateau |
| w8g16sym embed (= today) | 7.31 (with w4g32sym-a linears) | — | — | 8.5 | plateau |
| F16 (probe only) | — | — | — | 16 | 46.5 G w/s (+26 %, 2.6x bytes) |

\* bf16-stored eval (pre-protocol-fix), same direction.

**Recommendation for #316 (recipes) and #318 (layout ABI to cache):** keep
today's byte layouts — linears W4-as-Q4_0-shape (`w4g32sym-a`), embed W8
scale-per-16 (`w8g16sym`) — but quantize from source with the symmetric
scheme + imatrix search: it measured strictly better than the GGUF/Q4_0 draw
(7.31 vs 7.50 en) with identical kernel bytes. Free-offset asymmetric W4 is
measured harmful (~+2 pt); group sizes >32 trade ~1 pt for ≤6 % bytes and
no speed. The decode-speed lever is not the layout: it is token batching
(M-token GEMV, #311).

| P1-5 | W4U GEMV plateau comes from the second (scale) load stream | speed-only `W4_NOSCALE` probe: saturated lm_head shape unchanged (25.7 vs 26.5 GB/s) — the scale stream is free at saturation; small N=6144 +28% (unsaturated = load-latency dominated). Scale-interleave layout (#5) not supported. discard |
| P1-6 | **M-token GEMV**: share the weight unpacks across M tokens — per 32 weights at M=2, ~12 ops/token vs 20 at M=1 | **confirmed, kept (`bb7db62`)**: `gemv_w4um` (GEMV_M) processes two x-vectors per weight pass. Phone per-token rate **1.76–2.11× M=1** (N=151936: 69.7 vs 34.0 G w/s; N=6144: 34.2 vs 16.2; N=12288: 48.7 vs 27.7), second token nearly free; both tokens check rel err ~1e-3; end-to-end unchanged (+0.25 %, noise) with identical transcripts — the GemvArgs push-constant extension breaks nothing. Desktop: second token literally free (bandwidth-bound). This is the enabling kernel for #311 batch verification; at acceptance ≥ 0.5 the GEMV time per output token halves. **keep** |

| P1-7 | A deployable drafter makes the M-token GEMV exploitable for standalone MOSS decode (online n-gram, or copy-draft from Parakeet) | **both dead** (12 FLEURS clips, greedy id streams — the stream encodes every argmax, so acceptance is simulable offline): online n-gram (n=2..4) **1.000 tokens/pass** (no exploitable repeats in ~30-token ASR streams); Parakeet copy-draft **1.03 (K=2) / 1.05 (K=3)** — Parakeet and standalone-MOSS transcripts diverge constantly at the BPE level without prompt conditioning; casing normalization changes nothing. Also proven by construction: single-draft self-lookahead gains zero (the pass re-derives the pending token — identical context, identical logits — and advances exactly one token; K≥2 real drafts are required). Speculative decoding for the standalone metric needs a **learned drafter** (#292's gated EAGLE-3-class follow-up) or the product cleanup flow (Parakeet text in the MOSS prompt), which is a different measurement. `gemv_w4um` stays as the verify primitive; `STARLING_FAST_DUMP_TOKENS` lands as the study hook. discard |
