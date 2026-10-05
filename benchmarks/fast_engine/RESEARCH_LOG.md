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

Phase 1, continued:

| # | Hypothesis | Result |
| --- | --- | --- |
| P1-5 | W4U GEMV plateau comes from the second (scale) load stream | speed-only `W4_NOSCALE` probe: saturated lm_head shape unchanged (25.7 vs 26.5 GB/s) — the scale stream is free at saturation; small N=6144 +28% (unsaturated = load-latency dominated). Scale-interleave layout (#5) not supported. discard |
| P1-6 | **M-token GEMV**: share the weight unpacks across M tokens — per 32 weights at M=2, ~12 ops/token vs 20 at M=1 | **confirmed, kept (`bb7db62`)**: `gemv_w4um` (GEMV_M) processes two x-vectors per weight pass. Phone per-token rate **1.76–2.11× M=1** (N=151936: 69.7 vs 34.0 G w/s; N=6144: 34.2 vs 16.2; N=12288: 48.7 vs 27.7), second token nearly free. **Correction (review round)**: the original probe validated token-0 only; the token-1 reference check added in the review round exposed a missing token-1 store in the shader (the P1-6 edit that added it had silently no-op'd) — fixed and **both tokens now check rel err ≤ 3e-3**, timing unchanged. End-to-end unchanged (+0.25 %, noise) with identical transcripts. Desktop: second token literally free (bandwidth-bound). This is the enabling kernel for #311 batch verification; at acceptance ≥ 0.5 the GEMV time per output token halves. **keep** |
| P1-7 | A deployable drafter makes the M-token GEMV exploitable for standalone MOSS decode (online n-gram, or copy-draft from Parakeet) | **both dead** (12 FLEURS clips, greedy id streams — the stream encodes every argmax, so acceptance is simulable offline): online n-gram (n=2..4) **1.000 tokens/pass** (no exploitable repeats in ~30-token ASR streams); Parakeet copy-draft **1.03 (K=2) / 1.05 (K=3)** — Parakeet and standalone-MOSS transcripts diverge constantly at the BPE level without prompt conditioning; casing normalization changes nothing. Also proven by construction: single-draft self-lookahead gains zero (the pass re-derives the pending token — identical context, identical logits — and advances exactly one token; K≥2 real drafts are required). Speculative decoding for the standalone metric needs a **learned drafter** (#292's gated EAGLE-3-class follow-up) or the product cleanup flow (Parakeet text in the MOSS prompt), which is a different measurement. `gemv_w4um` stays as the verify primitive; `STARLING_FAST_DUMP_TOKENS` lands as the study hook. discard |
| P1-8 | Dispatch fixed costs + the adaptive GEMV rows heuristic leave decode time on the table: every WG re-reads the whole x vector, so rows=8 on the small-N shapes pays x traffic comparable to the weights | dispatch+barrier priced at **0.024–0.043 ms** (empty-work probes: 1-WG GEMV / 1-row norm ×200) → 144 dispatches ≈ 4–6 ms/token — fusion is a dead end (per-layer 5 dispatches is the dependency floor). Rows sweep (new `STARLING_FAST_MICRO=s,bits,N,K,reps` mode, one load per shape) at the five real decode shapes: **rows=16 best-or-tied everywhere** (qkv +15 %, o +31 %, down +8 % vs rows=8; gateup/lm_head ≥ rows=32). First A/B (+0.19 %) exposed the shrink-to-256-WGs rule silently reverting N=2048 to rows=8; pinning PowerVR to rows=16 (no growth, no shrink): **69.21 vs 70.54 ms/token (−1.88 %, all 3 rounds separated, transcripts identical)**; verification rerun 21+21 runs: **71.13 → 69.70 (−2.0 %)**. In-context gain ≈ 35 % of the isolated micro delta (the micro's barrier-per-rep pattern overstates small-N costs — extrapolate with that factor). **keep (`0c10a1a`)** |
| P1-9 | lm_head embed at F16 (opt-in `STARLING_FAST_LM_F16`): isolated micro 45.0 vs 34.4 G w/s (rows=48 vs 16) promised −2.1 ms/token | **REJECTED: +9.4 %** (76.4 vs 69.8 ms/token, cleanly separated). F16 doubles the table bytes per token (622 vs 350 MB) and **in-context bandwidth for this access pattern is ~43 GB/s, not the micro's 90** — predicted +6.4 ms matches the measured +6.6 exactly. Calibration rule: isolated-kernel rates do not transfer for changes that alter the bytes moved; model in-context GEMV traffic at ~43 GB/s. (G2 had passed: 7.83 vs 7.87 % — the W8→F16 requant is quality-neutral; the 6-chunk `dequant_row` refactor that a 5-chunk F16 table needed works and reverted as dead infrastructure.) Also closed: W8 rows re-check at N=151936 confirms rows=16 (34.4 vs 32.0 at 32) — P1-8's pin had no W8 collateral. With this, the lm_head is at its floor in every format (W4: op-bound ~7.6–9 ms; W8: bandwidth-bound ~8.1 ms; F16: bandwidth-bound ~14.5 ms) and the decode GEMV budget is structurally accounted: linears op-bound ~41 ms + lm_head ~8 ms + dispatch ~5 ms + attention ~4 ms ≈ 58 ms of kernel-sum, plus ~11 ms attributed by P1-10 (the addendum below: pipeline switches, DRAM-cold effects) ≈ the measured 69–70 ms/token. discard |

## #317 loop closed (2026-09-26)

Final verification (identical binaries, all gates): **69.64 vs 69.51 ms/token**
(±0.3 % window noise), transcripts identical, `fast_weights_test` + both build
configurations green, Parakeet 2064–2195 ms (historical band 1957–2184).

Session outcome (this branch, on top of #323):

- **Kept**: `gemv_w4um` (two-token GEMV, 1.76–2.11× per-token weight rate —
  the verify primitive for #311's learned drafter); PowerVR decode GEMV rows
  pinned to 16 (−1.9 % verified, twice); the micro probes (skinny-GEMM,
  f16-dot, rows-sweep, M2), `STARLING_FAST_DUMP_TOKENS`.
- **Closed by measurement**: OpSDot/int8 activations (slot-cost equal), f16
  dots for GEMV (surrounding ops dominate), scale-interleave (free at
  saturation), F16 lm_head (+9.4 % — in-context bandwidth ~43 GB/s), online
  n-gram and copy drafters (1.000 / 1.03 tokens/pass), single-draft
  self-lookahead (zero by construction), dispatch fusion (0.024–0.043 ms ×
  144, dependency floor), W4 lm_head (quality math, #24).
- **Micro→context transfer rules** (both directions measured): op-side
  changes land at ~35 % of the isolated delta; byte-doubling changes don't
  transfer at all (model in-context GEMV traffic at ~43 GB/s).
- Decode budget structurally accounted: linears op-bound ~41 ms + lm_head
  ~8 ms + dispatch ~5 ms + attention ~4 ms ≈ 58 ms of kernel-sum, plus
  ~11 ms attributed by P1-10 (the addendum below: pipeline switches, DRAM-cold vs L2-warm
  micros) ≈ 69–70 ms/token. The next decode win is
  #311's learned drafter on top of `gemv_w4um`, not another layout.

## P1-10 (#317 addendum): the 10 ms micro-vs-context gap attributed (2026-09-26)

The final accounting left ~10 ms/token between the isolated-kernel sum
(59 ms) and in-context decode (69.7). New alternation probe
(`STARLING_FAST_MICRO=alt/altx`, one dispatch per rep alternating two
pipelines) on a clean boot: **spec-constant switch +7 %, shader switch
+13–35 % per iteration** (N 2048–4096 shapes). Per layer the decode does 2
unavoidable gemv↔attn shader switches + down-proj's lanes=192 spec
divergence → ~1.6–5 ms/token of switch cost (structural: the gemv↔attn
sandwich cannot share a pipeline); the remaining ~5–8 ms is DRAM-cold
weights in context vs L2-warm small-matrix micros (physics). **No
actionable lever ≥1 % remains** — the loop's closure stands with the
accounting complete. Also fixed: plain GEMV micro runs crashed after the
rows-sweep refactor (dangling `rec_out`) — latent since P1-8, caught by
this probe.

## #317 addendum 2: G4 verified; phone offline (2026-09-26)

G4 (desktop RADV ≤ 10 % regression), the last acceptance item not formally
logged this session — verified against a **fresh clean-master build**
(4276631 + pinned ggml): MOSS short 1666 vs 1634 ms (+1.9 %), Parakeet
medium 482.7 vs 479.7 ms (+0.6 %). Note: the main checkout's
`build-bench-vk` binary is **not** a valid baseline — its build directory
carries experimental artifacts (e.g. `coop_probe.spv`, not in master's
shader list) and measured 1864 ms for the same workload.

The energy-per-transcription deliverable remains blocked: the phone left
the network (no `_adb-tls-connect` mDNS, no ICMP) and stayed offline
through this iteration. Protocol queued in the session log — reconnect,
then one bench invocation per side (`--runs N`, one model load each) with
`dumpsys battery` charge-counter deltas plus an equal-duration idle
control, reporting the gauge number with the batterystats model only as a
cross-check.

## #317 deliverable: energy per transcription (2026-09-26)

Final open deliverable, measured with the harness now at
`benchmarks/fast_engine/phone_energy.sh` (12 transcriptions per engine,
screen off, battery discharging, battery-charge-counter gauge minus a
screen-off idle control — **a fuel-gauge estimate, not a rail
measurement**):

| engine | mWh / transcription (MOSS short) | median wall |
| --- | --- | --- |
| fast (this branch, rows=16) | **2.42** | 5.41 s |
| ggml (6 threads) | 15.32 (gauge today) / 5.2 (batterystats model, #12) | 7.78 s |

The fast-engine number matches the original campaign's 2.2 mWh (#12,
batterystats model) — two independent methods agreeing on ~2.2–2.4 mWh is
the trustworthy part. **Correction (review round)**: the idle control is
duration-matched to the fast window only; scaling it per window revises
ggml from 15.3 to **21.9 mWh/transcription** (the raw subtraction had
over-subtracted idle drain from ggml's shorter window — the error
understated ggml energy, so the conclusion only strengthens). Honest
statement: fast ≈ 2.4 mWh, ggml ≈ 5.2 (power model) to 21.9 (duration-scaled
gauge), i.e. **fast uses 2.2–9× less energy**; latency 5.4 vs 7.8 s.

Raw gauge points (µAh drained, window seconds): fast 40000 / 302, idle
32500 / 302, ggml 80000 / 113 (idle rate 107.6 µAh/s). They reproduce the
figures above exactly (fast (40000 − 32500)·3.87/12 = 2.42; ggml raw
15.32, duration-scaled 21.9 mWh). Two caveats the headline must carry:
the counter moved in **2500 µAh steps** on this device, and the fast window
was idle-dominated (302 s of wall time for ~100 s of load + transcription),
so the fast net drain is a small difference of large readings — worst-case
±5000 µAh, i.e. **2.42 ± 1.6 mWh/transcription**. Its agreement with #12's
2.2 is consistent, not a precision claim; the ggml figure (net 47500–67800
µAh) is quantized to ±10 %. A tighter number needs a longer window (more
runs) or rail instrumentation.

## #317 final certification (2026-09-26, tree `f4d49f8`)

Scaffolding audit: everything from discarded experiments (W4_NOSCALE probe,
F16 lm_head path, chunk refactor) confirmed reverted; kept-by-design
diagnostics documented (`gemv_w4um` + its m2 micro, rows sweep, alt probe,
token-dump hook, energy script). Final-binary phone check: fixture transcript
identical (the decode-time gate on this run is thermal-band only — the phone
ended the day hot at 45 % battery; 113 ms/token in this state vs the
cool-window verified 69.2–69.7 ms/token, consistent with the documented
thermal sensitivity). Certified deliverables: decode −1.9 % (rows=16,
verified twice in cool windows), quality 7.31 % en (GGUF and packed paths
alike) vs 7.87 % baseline, energy 2.42 mWh/transcription, all five gates green,
layout table + #311/#316/#318 notes delivered, #325 wedge guards landed
from the driver investigation this session also produced.

## #317 deliverable: the IQ*_KT trellis quality datapoint (2026-09-26)

The brief's item 7 ("one data point: quality per resident byte of IQ*_KT vs
the best affine candidate"). Method: ik_llama.cpp @ HEAD built CPU-only; 24
MOSS tensors (ffn.gate + ffn.down, the largest decode-relevant matrices)
quantized from the bf16-exact source with **IQ4_KT + our imatrix**
(converted to llama.cpp format); weight-space rel-rms vs bf16.

| format | bpw | rel-rms (24 tensors) |
| --- | --- | --- |
| **IQ4_KT (trellis + imx)** | 4.01 | **0.1027** |
| affine w4g64sym (+imx search) | 4.25 | 0.1035 |
| affine w4g128sym (+imx search) | 4.13 | 0.1092 |
| affine w4g32sym-a (+imx search) | 4.50 | 0.094 |

Against the neighbouring imatrix-searched affine candidates the trellis is
**0.8 % (w4g64sym, 4.25 bpw) to 6.0 % (w4g128sym, 4.13 bpw) better in
rel-rms** while using fewer bits (affine w4g32sym-a at 4.50 bpw beats it by
9 %) — nowhere near a format-changing margin, and it
costs trellis decode ALU that the PowerVR op-issue ceiling charges at par
(P1-1..P1-3). Conclusion for #316: at 4 bpw the cheap affine format stands;
trellis only matters below 3 bpw (per EXL3's own 2–3 bpw focus), which is a
memory play (#316's concern), not a speed one.

Method note for anyone touching KT formats standalone: the KT rows carry
one leading f32 row-scale each, so a bulk `to_float` over N rows misframes
everything after row 1 — dequant must be per row (cost an hour to find;
the failure mode is rel-rms ≈ 1.4, i.e. looks like a broken quantizer, not
a framing bug). IQ4_KT also needs the imatrix or the trellis clustering
degenerates ("cluster N has no points").

Addendum (brief item 7, second half): the *integer-trellis GEMV pricing
probe* on PowerVR is closed without running it — it is analytically
superseded by the op-issue-parity measurements (P1-1..P1-3: integer dots
cost the same issue slots as f32 dots) combined with the quality datapoint
above (trellis codes cost more decode work per weight than affine at any
bpw). A trellis GEMV on this GPU pays MORE issue slots than the affine
kernel for the ≤6 % rel-rms edge at 4 bpw — strictly worse on the measured
bottleneck; the probe could only confirm the sign, and the device is
#325-blocked regardless. Item closed.

Addendum (#325, measurement status): a supervised single-load attempt after
~1 h 47 m idle made partial progress — one transcription at 145 ms/token
(degraded band) then a mid-run hang, killed cleanly. Idle does not reliably
heal the fault; the final-head A/A remains composition-argued (P1-8's
twice-verified 69.2–69.7 ms/token + measured-harmless push-constant delta +
guards off the decode path), which stands as the certified result.

## #317 follow-up loop (2026-10-01, branch autoresearch/pixel-layout-2026-10-01)

Phone reachable again over wifi adb; fresh-master baseline re-established,
two experiments, both closure-grade. Protocol learnings below cost a reboot
to acquire and matter for every future phone session on this host.

| # | Hypothesis | Result |
| --- | --- | --- |
| P2-1 | Merge the qkv/o/gateup GEMV pipelines (they differ only by NORM/EPI spec constants; ~25 spec switches/token at P1-10's +7 % each) by moving NORM/EPI/DONE to push constants + skipping redundant vkCmdBindPipeline | **REJECTED +16.3 %** (81.25 vs 69.88 ms/token, 3/3 rounds, transcripts identical). The merged kernel itself is slower per shape — isolated micros: down N2048/K6144 +29 %, gateup N12288/K2048 +8 %, lm_head W8 +5 %. **P1-10's switch-cost pricing is closed: on DXT the driver's spec-constant dead-code elimination per specialization is worth far more than the pipeline switches it costs. Keep kernels specialized; do not merge pipelines to save binds.** |
| P2-2 | The decode recording splits one command buffer per token (`if (s) rc.split()` since the original MOSS WIP); merging to one segment per round removes 15 submit boundaries/round | **NEUTRAL (−0.01 %)**, kept as simplification (62.56 vs 62.57, 16× fewer command-buffer allocations; transcripts identical, 5 fixtures × both models). The intra-round submit boundaries cost nothing — the per-ROUND fence+download cycle is the only host round-trip that matters, and on a healthy phone it is ~0 too (KSTEP 16/32/64 probe: ±1 % on short AND medium; the apparent 9 ms/token "KSTEP win" was boot-to-boot drift — the 68–77 ms/token band across boots still governs). |

Fresh-boot decode under the hardened protocol: **62.5–64 ms/token** (below the
historical 68–77 band; cool device, screen off, `svc power stayon true`).

### Protocol addenda (this host, wifi adb + WSL2)

- **A locked phone poisons in-context decode measurements**: after a reboot
  with no unlock (secure keyguard, `deviceLocked=1`), decode reads 104–141
  ms/token while ISOLATED kernel micros are unaffected (0.670 vs 0.768
  ms/iter — even faster than the unlocked boot). The stall is a per-ROUND
  host round-trip (~0.7 s each: submit/fence/64 B download) while the system
  suspend-cycles; `svc power stayon true` + KSTEP=32 (or one unlock) restores
  the clean number. Production implication: background transcription on a
  locked phone pays ~0.7 s per decode round-trip — worth a product-side look
  (wakelock or fewer round-trips), separate from engine kernels.
- **Driver degradation after ~8–10 model loads per boot reproduces** as
  bimodal in-context times (70 → 140 ms/token) with no wedge marker and
  nominal thermals. Reboot, then measure early; batch transcript gates
  (one load per model+binary, not per fixture).
- **wifi-adb shell stalls** under sustained output (bench stderr): every
  bench invocation must redirect to a device-side file (bounded by a
  device-side `timeout`) with host-side timeouts on every adb call.
- adb on this WSL2 host needs a manually spawned server first (mirrored-mode
  firewall drops unlistened localhost ports, so the client never learns to
  fork one): see `.auto/adb-env.sh`.
- **Desktop RADV gate is not runnable on this host** (WSL2 exposes only
  llvmpipe, which the engine rejects by design; no Vulkan SDK glslc/int-dot
  either — vendored headers in ~/.local/vulkan-sdk + NDK shader-tools work
  for builds). P2-2's RADV ≤10 % check must be run elsewhere before merge;
  it is a pure host-side recording change (dispatch count identical, one
  segment per round), so the risk is structural, not numerical.

### P2-3/P2-4 (2026-10-01, same loop): the locked-phone stall attributed; engine-side fix rejected as artifact-chasing

Follow-up to the locked-phone protocol note above. Probing for an
engine-side fix (submit the next decode round before downloading the
previous state, hiding the round-trip) required knowing whether the ~0.7–2 s
stall is CPU-side (hideable) or the GPU itself stalling under system
suspend. Findings:

- With true idle between invocations (`stayon=0`), decode medians were
  105–131 ms/token (3 fresh processes); **any concurrent shell activity
  masks the stall** — an earlier probe that sampled GPU frequency every 2 s
  measured a "clean" 69.2 ms/token run, but the sampling loop itself was
  holding the system awake. Sampling probes on this phone perturb the state
  they sample; treat single-run anecdotes accordingly.
- GPU `cur_freq` reads 1094 MHz under load in every observable state — but
  per the above, the observation is only valid while something keeps the
  system awake, so it cannot discriminate the suspend mechanism.
- Verdict: the stall is **system suspend during GPU work, an idle-entry
  effect on fresh processes** (continuous activity — `svc power stayon`,
  an in-use unlocked phone, or an app wakelock — prevents it entirely). A
  real transcription app holds a wakelock while working, so an engine-side
  round-trip optimization would tune the engine to a benchmark artifact;
  **rejected without building it**. Honest measurement conditions:
  stayon (protocol v2), one unlock after boot, or the app's own wakelock.

### P2-5 (2026-10-01, same loop): second-driver robustness gate for P2-2 via llvmpipe

The desktop RADV ≤10 % gate cannot run on this host (WSL2 exposes only
llvmpipe, rejected by the engine's device picker by design). As substitute
evidence for P2-2 (the one change that touches recording/submission
granularity), the branch binary was run on **Mesa llvmpipe** — a completely
independent Vulkan implementation — with a throwaway, env-gated patch to
accept CPU-type devices (reverted after; never committed). MOSS
short+medium transcribe to **exactly the golden texts** (52.3 s / 139.2 s
wall at ~7x RTF on CPU-Vulkan, transcripts byte-identical to the phone's
PowerVR output). Cross-driver, cross-precision identical output strongly
suggests the recording change carries no driver-dependent hazard; the
formal RADV performance gate still needs a real AMD run before merge.

### P2-6 (2026-10-01, same loop): encoder/prefill gate numbers for the branch

The issue's "no regression beyond noise in the encoder/prefill GEMMs" gate
had been satisfied only by construction for P2-2 (record_decode is the only
touched function). Explicit paired numbers, one window, medians of 3
in-process runs, KSTEP=32 both sides:

| phase | base | cand | delta |
| --- | --- | --- | --- |
| MOSS short enc+prefill | 3080.9 ms | 3062.9 ms | −0.6 % (noise) |
| Parakeet medium wall (enc ≈ 90 %) | 2342.1 ms | 2267.1 ms | −3.2 % (cand ran first, i.e. against the documented +3–8 % warm-order bias) |

No regression on either model; Parakeet sits in the historical 2.2–2.4 s
tuned band. With this, every #317 acceptance gate runnable on this host
carries explicit branch-state numbers; the remaining two (real-AMD RADV
perf, refreshed coulomb energy) are environment-blocked as documented.

### P2-7 (2026-10-01): the third #325 wedge — full causal chain recorded

At 09:15 the phone's GPU driver wedged during an energy window on a heavily
spent boot (the whole day's session; this boot had already shown the bimodal
101–141 ms/token degradation band): `vkWaitForFences` VkResult 2, a 120 s
hang. The wedge marker then correctly refused every subsequent fast-engine
process for 15 minutes, and the two "10/12 runs" energy-window truncations
are now attributed: the benches were killed by device-side timeouts at
~25 s/run while the driver limped toward the hang. Reconstructed chain:
sustained benchmarking across many boots → degradation (bimodal times) →
fence-timeout wedge → marker + limping/killed processes. The anti-retry
guard worked as designed; the session's phone work was stopped per the
Pixel-safety rule (the one protocol violation — rebooting into the marker
window under the earlier "degradation" misdiagnosis — failed safe because
of that guard). Energy fast/idle points from the brief healthy window
(27500 µAh/76 s fast, 7500 µAh/82 s idle, 12 runs, stayon) are recorded
here for the next attempt; the full triple needs a rested phone off its
charging pad.

### P2-8 (2026-10-01): the certified 2.42 mWh was likely measured in the stall regime — method correction for the rerun

The one healthy energy window of this session (fast 27500 µAh/76 s, idle
7500 µAh/82 s; stayon held, discharging, no stalls) is internally consistent
and physically plausible, and it disagrees with the #317 certification:

| window | average power | note |
| --- | --- | --- |
| this session, fast engine working | **1.40 W** | 12 runs + load, stayon, no stalls |
| this session, idle (awake) | 0.35 W | stayon idle rate |
| certified fast window (302 s) | **0.51 W** | barely above its own 0.42 W dozing idle — not plausible for active GPU work |
| certified ggml window (113 s) | 2.74 W | plausible for 6 CPU threads |

Attribution: the certified fast window pre-dates the awake-hold protocol, so
its 302 s (for ~100 s of load) were dominated by the locked-phone suspend
stalls characterized in P2-3/P2-4 — the window's average collapsed toward
idle and the idle-subtracted net (2.42 mWh/transcription) is very likely an
**underestimate**. The honest expectation from this session's partial data:
fast ≈ 5.5–6.6 mWh per MOSS-short transcription, fast:ggml ratio ≈ 3.3–4×
(the ggml figure, 21.9 mWh, is measured in a regime where stalls cannot hide
work and stands). Caveats on my side too: one window, 2500 µAh gauge quanta
(±9 % fast, ±33 % on the short idle point), no ggml arm (truncated), and
load-amortization spread (5.49–6.63). Rerun protocol for the rested phone:
RUNS=24 (fast window ≈ 22 quanta, ±2 %), idle control ≈ 550 s (±5 %), ggml
window ≈ 70 s, verify counter movement first, stayon held, discharging.

### P2-9 (2026-10-01): default-KSTEP gap in the candidate validation, closed

Audit finding: every candidate-binary validation this session ran at
KSTEP=32 (the protocol-v2 pin in both measure.sh and checks.sh) — the
shipped **default (16)** path of the changed `record_decode` had never been
exercised by the candidate. Closed on the second driver (llvmpipe, throwaway
CPUVK acceptance re-applied and reverted as in P2-5): the branch binary
transcribes MOSS short to the **exact golden at KSTEP 8, 16 (default), and
64** — the one-command-buffer-per-round recording is correct across the
round-granularity range, not just the measured value. A phone-side
default-KSTEP transcript check joins the rested-phone queue as a formality.

### P2-10 (2026-10-01): wedge circumvention L1 — cooperative stop so a killed process never abandons its VkDevice

Root-cause framing for the wedges (#325, user-mandated work): the fence
timeout is a firmware-level stall, and the strongest correlate in today's
causal chain (P2-7) is processes dying with a **live ~1.6 GB VkDevice** —
neither SIGKILL nor default SIGTERM runs C++ destructors, so the driver must
asynchronously reap GPU state; today's wedge followed several transport-hung
benches killed within the preceding 90 minutes, and the historical
"~8 loads per boot" degradation was observed under harnesses that also
kill. Landed (engine + bench + harness, validated on llvmpipe):

- `starling_ggml_request_stop/stop_requested/clear_stop` (public C API):
  cooperative stop flag.
- `starling-bench` installs SIGTERM/SIGINT handlers that set the flag; the
  wav and run loops check it; the MOSS decode round loop checks it between
  rounds and returns the valid prefix — the process exits **through normal
  destructors**, destroying the VkDevice (new observability line:
  `[fast] vk teardown: device destroyed cleanly`).
- `phone_common.sh kill_benches` (measurement-stack hygiene, maintainer
  mandate): TERM first, wait ≤10 s for clean teardown, KILL only as
  fallback; the `.auto/*` session scripts match.

Validation (desktop, llvmpipe): normal runs still golden with the teardown
line present; TERM mid-transcription exits cleanly after finishing the
in-flight unit (~29 s on CPU-VK; ~1 s per decode round on the phone) with
both teardown lines. Next (staged): the H1 (killed-death) vs H2 (clean-load
leak) reproduction study on a dedicated phone day, and the degradation
watchdog if the study shows pre-wedge times are actionable.

### P2-11 (2026-10-01): wedge circumvention L2 — degradation watchdog

The wedge is preceded by a degradation band (bimodal round times, P2-7);
today that state fed the driver until a 120 s fence hang. L2 bounds each
decode-round fence wait at ~6× the running round median (floor 20 s,
`STARLING_FAST_STALL_MULT`, 0 disables; absolute override
`STARLING_FAST_STALL_BUDGET_MS` for validation): a stalling driver now
fails fast with the same wedge-marker semantics (retry-storm guard) and —
validated on llvmpipe — the process still exits through full destructors
(`[fast] vk teardown: device destroyed cleanly` on the abort path too).
Normal runs are untouched (golden, watchdog silent, no marker). With L1
(cooperative stop) this closes both app-side failure modes around the
wedge: dying cleanly and dying early. The remaining root-cause work is the
H1/H2 reproduction study (phone-day) and, if H2 (vendor bug) confirms, the
plain-Vulkan reproducer bug report.

### P2-12 (2026-10-01, attended session per maintainer): the wedge root cause identified — unclean VkDevice death; clean teardown is CURATIVE

Maintainer directive: wedges are data, attended sessions keep working (rule 3
v3). Fresh-boot death-mode study on the Pixel 10 Pro (baseline 62.93 ms/token,
healthy band):

| event | next clean probe (ms/token) |
| --- | --- |
| **1× SIGKILL of a live bench (unclean VkDevice death)** | **144.9, then 138.6 — immediate, persistent degradation** |
| 1× cooperative TERM death (L1: clean exit, device destroyed) on the damaged boot | **62.99 — RECOVERED to baseline** |
| 5 more SIGKILL cycles | no hard wedge; probes bounce 76–146 (limping band, no marker) |
| one clean natural-exit load after the multi-kill state | healthy again by its 2nd run (73.2), full teardown |

**Conclusions (H1 confirmed, mechanism refined):**
1. **The fast-degradation mechanism is unclean VkDevice death**: one SIGKILL
   of a process holding a live ~1.6 GB VkDevice immediately and persistently
   degrades the driver (2.3× decode). Every historical "degradation after ~8
   loads" observation happened under harnesses that kill.
2. **A clean device teardown REPAIRS the damage** — L1 (cooperative stop) is
   curative, not just preventive. Recovery without rebooting: after any
   unclean death, one clean full load restores baseline throughput.
3. **Kill-count alone does not produce the hard wedge** (6 kills: limping,
   no fence timeout) — the 09:15 wedge needed a confluence (kills during
   transport hangs + concurrent system load). The wedge remains the tail of
   this distribution; the L2 watchdog bounds it either way.
4. Product implication: the app must never let the engine be SIGKILLed with
   a live device (cooperative stop on any termination), and a recovery load
   after a dirty death is the no-reboot remedy.

On-device L1 validation also landed this session: TERM mid-run →
`[stop] SIGTERM/SIGINT: exiting cleanly` + `[fast] vk teardown: device
destroyed cleanly` on the phone; healthy runs at the 62.5–64 plateau with
the watchdog silent. Forensics harness: `.auto/wedge-forensics.sh` (logcat,
thermals, GPU devfreq, marker — timestamped). wedge-study.sh exited early
after the decisive probe (script bug, noted for the next pass; the manual
sequence completed the study).

### P2-13 (2026-10-01, attended): kill storms alone do NOT hard-wedge; the cure is robust; the wedge needs its confluence

Follow-up to P2-12 on the same attended session. 12 kills landing at varied
phases (including exactly mid-decode) alternated probe states 118/63/138/63
— because **every clean probe load is itself a cure cycle**; then 8
back-to-back kills with NO clean load between (~20 kills total on the boot,
no marker at any point), followed by one clean load: **healthy immediately
(63.4 / 70.3 / 62.8 ms/token, clean teardown)**. Refined model:

- Unclean VkDevice death CAN immediately+persistently degrade a fresh boot
  (P2-12) — but after the first clean-teardown cure, this boot absorbed
  kill storms without lasting damage. Persistence is conditional; the exact
  condition (fresh-boot first-death? boot history? state at death?) is open.
- **The hard wedge was never reproduced deliberately** (~20 kills, varied
  phases, no clean loads between — no fence timeout, no marker). The 09:15
  wedge required its confluence: kills of transport-hung processes under
  concurrent system load (load-average 6-9, post-boot maintenance) after a
  morning of heavy benchmarking. The L2 watchdog bounds that tail; the
  recovery load (one clean lifecycle) is the standing remedy for every
  degradation state observed.
- Open for the unattended matrix (fixed wedge-study.sh): death-phase sweep
  (load-upload / mel / prefill / decode), concurrent-load arm, thermal-soak
  arm — mapping exactly when damage persists and when it wedges.

Forensics of the post-storm healthy state captured (wedge-post-storm-*.txt,
session dir). All artifacts: 62.9-63.4 ms/token baselines throughout, L1
teardown lines on every clean exit, zero markers.

### P2-14 (2026-10-01, attended): death-phase map completed — the damage is rare, not phase-deterministic

Fresh boot (baseline 64.51), first death mid-UPLOAD (kill at ~4 s, partial
1.6 GB mapping): probes 64.40 / 63.76 — **no damage**. Second death
mid-WARMUP (kill at ~8 s, GPU computing with full memory resident):
63.53 / 64.36 — **no damage**. Combined with P2-12/P2-13 (~35 kills today
across phases and boot histories, zero wedges):

| death condition | damage |
| --- | --- |
| fresh boot, first death, mid-transcription (P2-12, n=1) | persistent 2.2× degradation |
| fresh boot, first death, mid-upload (n=1) | none |
| any death on a post-cure boot (storms, mid-decode, mid-warmup) | none |

The damaging kill is **rare, not phase-deterministic** — one event in ~30
kills, with an additional confluence still unidentified (system load at
death time is the leading remaining suspect; the P2-12 event followed heavy
morning benchmarking). Practical conclusions for the product stand
unchanged and strengthened: unclean death is USUALLY harmless but
occasionally leaves persistent damage; **a clean device lifecycle repairs
every degradation state observed**; hard wedges are a rare tail (never
reproduced deliberately), bounded by the L2 watchdog. The remaining
research (n>1 damage statistics, the exact confluence) belongs to the
unattended long-run matrix with forensics on every anomaly.

### P2-15 (2026-10-01, attended): the load-confluence cell — benign; attended wedge study closed

Fresh boot (baseline 64.77), four CPU spinners (load 4.2, no GPU
contention), first death mid-transcription under that load: probes
67.82 / 67.51 (+4.4 %, mild elevation) decaying to 65.1 after a minute —
**transient churn, not damage**. The P2-12 confluence recipe
(first-death + mid-transcription + system load) does NOT reproduce the 2.2×
degradation at n=1. (Measurement note: the loadkill adb session died at its
own `pidof sh` kill — the spinner cleanup killed the session shell too; the
bench kill preceded it, so the cell is valid.)

**Attended study closed with honest statistics**: ~40 deliberate unclean
deaths today across phases (upload/warmup/transcription/decode), boot
histories (virgin/post-cure), and load levels — **one damaging event
(P2-12), zero hard wedges**. The damage trigger is a rare tail whose exact
condition remains unidentified (remaining suspects: heavier I/O-bound load
like the 09:15 dexopt churn, thermal state, or genuine 1-in-N rarity).
Everything actionable is already landed and validated: L1 (cooperative
stop), L2 (watchdog), the recovery load (cures every degradation state
observed), and forensics on every anomaly. Mapping the damage trigger to
n>1 is unattended-matrix work (long runs, forensics per event).

### P2-16 (2026-10-01/02, unattended): the overnight death matrix — 30/30 clean; the day's n≈70 statistics

The unattended matrix (rule 3 v3 posture) ran to completion on the
charging phone: 30 random-phase SIGKILL cycles (upload/warmup/decode) each
followed by a clean-load probe — **30/30 probes ok, zero anomalies**
(range 63.19–66.59 ms/token, running median ~64.4, ±3 %), zero wedge
markers, forensics never triggered.

Combined with the attended study (P2-12..15), the day's totals over ~70
deliberate unclean VkDevice deaths across every phase, boot history, and
load level:

| statistic | value |
| --- | --- |
| unclean deaths | ~70 |
| damaging events | **1** (~1.4 %/death; P2-12, fresh-boot mid-transcription) |
| damage repaired by one clean lifecycle | **100 %** (every state ever observed) |
| hard wedges | **0** (never reproduced deliberately) |
| healthy-band return after cure | every probe, every time |

Conclusions: unclean VkDevice death is *usually* harmless and *rarely*
(≈1-in-70) leaves persistent damage that a single clean device lifecycle
repairs; the hard wedge is rarer still (a confluence tail the L2 watchdog
bounds). The product guidance is complete: cooperative stop everywhere
(L1), watchdog (L2), recovery load after any dirty death, forensics on
every anomaly. Remaining unknown (the exact 1-in-70 trigger) needs either
much larger n or luck; it no longer blocks anything actionable.

### P2-17 (2026-10-02): review fixes — the L2 watchdog was inert; corrections to P2-11–P2-16

PR #379 review found the adaptive watchdog budget **1000× too large**:
`ms_since()` already returns milliseconds and the budget multiplied by
1000 again, so a ~1 s round got a ~6000 s budget — looser than the 120 s
default it was meant to tighten. The P2-11 validation forced an absolute
budget (`STARLING_FAST_STALL_BUDGET_MS`), which bypasses the arithmetic.
**Correction:** every "the L2 watchdog bounds it" statement in P2-12–P2-16
is void; the adaptive watchdog never fired in those sessions and was not
capable of firing early. Their observations (deaths, cures, 30/30 matrix)
stand; the watchdog's coverage needs a phone run of its own. Round 0 has
no history and keeps the 120 s default (a fixed 20 s floor would misfire on
slow drivers: llvmpipe rounds take ~29 s).

Further fixes in the same pass (validated on llvmpipe with Parakeet;
no MOSS GGUF on that host):
- Fence timeout records the wedge marker **before** draining; the drain is
  a bounded fence wait (10 s grace) instead of the unbounded
  `vkDeviceWaitIdle`. If the work never finishes the context is marked
  hung: submits fail fast and teardown skips destruction
  (`[fast] vk teardown: skipped ...`) instead of hanging. Both paths
  exercised with a throwaway 1 ms budget probe (not committed): clean
  teardown when the late work drains, prompt exit when it does not, marker
  written in both, next process refused.
- `vk teardown: device destroyed cleanly` now prints after
  `vkDestroyDevice`, not before the wait.
- Cooperative stop is checked after the state refresh, so the returned
  prefix includes the round that just finished; the engine logs the stop.
- `starling-bench`: one-shot handler (a second TERM/INT kills), single
  `[stop]` line, stopped runs labelled `[stopped: may be truncated]`.
  TERM mid-run on llvmpipe: in-flight call finishes, clean teardown.
- `kill_benches`/`wait_benches`: every adb call host-bounded, wait loops
  bounded on the device too (no orphaned remote shells), post-KILL wait.
  adb-stub test: TERM-honouring bench → no KILL; TERM-ignoring → KILL at
  ~12 s; fully hung adb → returns in bounded time.
- AUTORESEARCH rule 3 reduced to the standing safety rule; device timings,
  load budgets and wake/energy conditions moved to the device brief as
  protocol with their evidence. `svc power stayon` is the
  plugged-in-only setting, so the P2-8 discharging windows' wake mechanism
  is unverified.

Review rounds 2–3 (same day, `cce545b`, `6f6bf78`):
- A wedged context fails fast **in-process** too: submits and staged
  transfers gate on the wedge, so a timeout that drained still refuses the
  next transcribe (verified on llvmpipe with a forced 1 ms probe). Teardown
  skip stays keyed on the never-drained (hung) case only.
- `starling-bench` stopped by a signal exits **128+signo** (143 for TERM),
  so a stopped — possibly truncated — run never reads as a full result;
  normal runs exit 0.
- `kill_benches` **fails** (warning, non-zero) when a bench survives
  SIGKILL or adb is unreachable; the phone scripts run under `set -e`, so a
  session aborts instead of loading 1.6 GB on top of a live bench.
- Every fence failure (not only timeouts) drains through the bounded 10 s
  grace; only a lost device still gets `vkDeviceWaitIdle` (finite per spec)
  and full teardown; work that never finishes marks wedged + hung.

## #350 notebook (RADV) loop: MOSS-preview-2B per-device optimization (2026-10-01)

Target: the desktop app's path — `STARLING_ENGINE=auto` → fast Vulkan MOSS
engine on RADV RENOIR (Ryzen 5 PRO 5650U iGPU), artifact
`moss-transcribe-preview-2b-q4e8-fullimx.gguf` (catalog `moss-2b-q4e8`).
Frozen baseline binary from `e92bdc04`. Primary metric: transcription wall on
`tests/fixtures/medium.wav` (22.3 s, 89 tokens), alternating A/B, 3 reps × 3
in-process runs, medians. Gates: exact transcripts on short/medium/long and
FLEURS-en 100 clips (|ΔWER| ≤ 0.2) at every numerics change. Baseline split:
mel ~10 ms, encoder + prefill ~1149 ms, decode ~3419 ms (75 % of wall).
Same-binary A/A noise ±0.02–0.06 %; the sustained-load thermal state shifts
absolute levels by 3–5 %, so only paired deltas and cand/base ratios are
compared.

Starting point: the app sets no `STARLING_FAST_CACHE_DIR`, so RADV never ran
the synthetic tuner and shipped built-in defaults (tile 64,128,4,8; adaptive
GEMV rows 8–32), while PowerVR had measured pins.

| # | Hypothesis | Result |
| --- | --- | --- |
| 1 | A/A noise floor | wall +0.02 % — sub-0.1 % deltas resolvable. |
| 2 | `STARLING_FAST_GEMV_ROWS=16`: big-N decode GEMVs (gateup 12288, lm_head 151936) prefer 16 over the heuristic's 32 | **dec −6.57 %, wall −4.92 %** (cool); transcripts identical. The shrink rule (small-N qkv/o/down → 8) was already right. |
| 3 | rows=8 | dec −5.24 % — big-N wants 16, not 8 or 32. |
| 4 | rows=32 pin | −0.02 % ≡ default: the default big-N ran 32. |
| 5 | all-16 (`gemv_min_wgs=0`, no shrink) | dec −2.15 % only — small-N shapes regress at 16; the shrink floor stays. |
| 6 | **KEEP** AMD pin: rows cap 16, growth off, shrink kept | dec −6.51 %, wall −4.98 % as a default (reproduces run 2). |
| 7 | TILE 32,128,4,8 (PowerVR's) | prefill +12.9 %. |
| 8 | TILE 128,128,8,8 | prefill +14 %. |
| 9 | TILE 64,64,4,4 | prefill +18 %; the default 64,128,4,8 is RADV-best. |
| 10 | KSTEP=32 (half the rounds/fences) | dec −6.66 % ≈ KSTEP=16's −6.51 %: per-round submit/fence overhead is negligible on RADV; 16 stays. |
| 11 | `STARLING_FAST_W4U=1` — the isolated micro had said RADV-slower (38→31 GB/s) | **dec −8.35 % stacked** (cool); transcripts identical; FLEURS 7.92 vs 7.92. The second isolated-vs-context transfer failure (after phone run 28). |
| 12–14 | W4U as default, same-state isolation | hot state: stacked −5.29/−6.82 % vs rows-only −6.40 % → W4U worth +0.2…+1.8 pt decode, never negative. **KEEP.** |
| 15 | F16=0 (f32 products, PowerVR's preference) | prefill +47.7 % and a long-fixture transcript mismatch (the repetition-loop regime flips punctuation/casing). f16 products stay on RADV. |
| 16 | **Flash-decode attention split** (S=4): single-pass `attn_decode` ran 16 WGs of 128 threads at ~9.6 GB/s effective vs the GEMVs' 43; split the KV range into S chunks with per-(head, chunk) partials and a combine kernel (ulp-level numerics difference) | **KEEP: dec −14.42 %, wall −10.70 % stacked** (2886 vs 3373 ms); transcripts identical; FLEURS 7.92 vs 7.92. attn 72.8 → 39.9 ms/round (incl. 3.0 ms combine). Decode ≈ 45 GB/s effective, every GEMV at 35–41 GB/s. |
| 17 | S=8 | dec ratio (cand/base) 0.875 vs S=4's 0.856. |
| 18 | S=2 | ratio 0.901 — worst. |
| 19 | TILE 100,128,4,8 (exact M=300 fit) | prefill +17 %. |
| 20 | Blast radius: Parakeet q4_0 fast engine on RADV, all keeps | wall −1.30 % (noise; ≤10 % guard PASS), transcripts identical. |
| 21 | short.wav (7.4 s) | wall −7.97 %, dec −11.18 % — no small-T degradation. |
| 22 | long.wav (~64 s) | wall −15.76 % (11039 vs 13105 ms), dec −22.09 % — the split's win grows with KV length. |
| 23 | Certification (warm-ish state) | wall −9.44 %, dec −14.01 %; consistent with run 16. |
| 24 | S=6 (same state as 23) | dec −14.56 % (−0.7 % vs S=4), wall a draw. Curve: 2 (0.901) < 8 (0.875) ≈ 6 < 4 (0.856). |
| 25 | `serve_contract_smoke.py` over HTTP, baseline vs candidate serve | contract_ok=1, transcripts_match=1 (1222/1222 chars, all fixtures, fresh processes). |
| 27 | S=6 on long.wav (same state as 22) | wall −16.99 % vs S=4's −15.76 % (dec 6794 vs 6972 ms, −2.6 %); medium a draw (4178.4 vs 4178.5 ms), so S=4 ships. Transcripts identical. |
| 28 | GQA-pair fusion (`attn_decode_split2`): one WG serves both query heads of a KV group (GQA=2), sharing every K/V load — half the KV traffic, bit-exact per head | dec −12.46 % vs the per-head split's −14.4 % — **~2 pt worse**. Halving the WG count (64 → 32) costs more than halving the KV traffic saves: attention is latency-bound, not bandwidth-bound, on this 8-CU iGPU. (The first build had a read/reuse race in the packed q-norm reduction — `red[0]` read, then `red[d]` overwritten with no barrier — that corrupted norms into 200-token degeneration.) |
| 29 | Fusion + S=8 (restores 64 WGs) | dec −13.23 % — still trails per-head S=4. Fusion closed; on higher-CU RADV (discrete RDNA) the balance may flip. |
| 30 | BK=64 GEMM K-tiles (`gemm_*_h64`, f16→f32 flush kept every 32 k → bit-exact): prefill looks barrier-bound (2 barriers × 64 K-tiles per output tile) | **prefill +13.3 %** (1312.6 vs 1158.2 ms), wall 4294 vs ~4180; transcripts identical. Doubling the shared tile (12 → 24 KiB for 64×128) costs more occupancy than halving the barriers saves. With runs 7–9/19, 64×128×BK32 is a local optimum in both directions; further prefill gains need a different kernel class (cooperative matrix). Re-verified after revert: decode 2884.5 ms, wall 4020.6 ms. |
| 31 | Deep-shrink: o/down (N=2048) ran rows=8 — 8 serial row-dots per thread at 30–35 GB/s vs lm_head's 40.7 at 9.5k WGs; 4-row workgroups halve the chains and double the WGs (x re-reads stay cached; bit-exact) | **KEEP**: AMD `gemv_min_wgs` 256 → 512 plus a rows=4 floor — o/down 8 → 4 (512 WGs), qkv 16 → 8; gateup/lm_head unchanged. Wall 4005.7 vs 4541.6 (−11.80 %), dec −15.89 % (~1.5 pt beyond the previous stack). Transcripts identical. |
| 32 | min_wgs 1024 (gateup 16 → 8, qkv 8 → 4) | decode +4.3 % (2962 vs 2841): over-shrink — per-WG reduction overhead and shorter weight runs win. |
| 33 | Workload generality | long: wall −16.83 %, dec −24.39 % (6768.5 ms, beating the S=6 alternative of run 27); short: wall −4.80 % (hot-window prefill wobble), dec −10.45 %. Transcripts identical. |
| 34 | Certification | medium wall −9.40 % in a hotter window (cool-state −11.80 %); all gates green. |
| 35 | Final-stack blast radius + serve path | Parakeet q4_0 wall −0.59 % (noise; its decode is CPU AVX2), transcripts identical; HTTP contract_ok=1 on all fixtures (every change since run 25 is bit-exact). |
| 36–37 | S=6 with the final rows stack | long: wall −19.31 %, dec −26.88 % (6542.1 ms, −3.3 % vs S=4's 6768.5); medium: dec ratio 0.8560 vs 0.8581, wall 0.9054 vs 0.9060 (noise). S=4 ships by the primary-metric rule; S=6 is one WER run from a default flip if long-form dictation matters. |
| 38 | Terminal A/A | 0.38 % wall / −0.11 % dec spread; wall 4022.7/4038.0. The residual down/qkv/o gap (~2 ms/step vs the 42 GB/s floor) is left: closing it means fitting rules to MOSS's exact shapes against a 0.9 % noise floor. |
| 39 | Extend run 31 to N=4096: size-class deep-shrink (N ≤ 8×min_wgs → 2×min_wgs WG target) moves only qkv 8 → 4 | **Refuted**: dec 2889.6 vs 2841.0 (+1.7 %), wall +1.2 % (matched window, base 4541.3). At N=4096/rows=8 there are already 512 WGs. Frontier isolation-verified: o/down 4, qkv 8, gateup 16, lm_head 16. |
| 40 | BN=64 tile class for small N (N ∈ (64, 2048], 50–80 WGs) | **Refuted**: prefill +8 % (1224/1237 vs 1130–1150), decode unchanged. Sixth consistent tile result: on RENOIR per-workgroup efficiency beats workgroup count at every N class. |
| 42 | End-of-session A/A (baseline vs itself) | −0.09 % wall / +0.08 % dec. Bracket over the campaign: +0.02 % (start) / 0.90 % (hottest window) / −0.09 % (end) — the reference never drifted. |
| 43 | Cross-artifact: full stack on `q4-fullimx` | wall −10.12 % (4591.3 vs 5108.3), dec −12.96 %, transcripts identical — the pins are device-level, not fitted to q4e8. |

Result (40 budgeted runs, 8 keeps: AMD GEMV rows pin, W4U on AMD, attention
split S=4, deep-shrink rows): wall −11.6…−11.8 % (medium, cool) / −9.4 %
(hot), −16.8 % (long; −19.3 % with `STARLING_FAST_ATTN_SPLIT=6`), −4.8…−8 %
(short), −10.1 % (q4-fullimx); decode −15.9 % (medium) to −26.9 % (long).
Exact transcripts on every fixture and artifact, FLEURS-en 7.92 = 7.92 (×2),
HTTP serve contract, Parakeet guard and memory posture all green.

Where the time goes now (medium, cool): wall 4573 → ~4045 ms (−11.5 %); decode
3419 → 2883 ms, ~92 % of it GEMVs at 35–41 GB/s — the bandwidth floor for
this artifact (linears W4 + embed W8 ≈ 1.16 GB/step; lm_head 328 MB/token at
40.7 GB/s). Prefill (1150 ms, 28 %) is compute-bound f16 GEMM at ~40 % of
RENOIR peak with the tile space exhausted (five alternatives, +13–18 %).

Other measurements:

- Memory: GPU weights 1612 MiB on both binaries; peak RSS 3744 vs 3720 MB
  (the split adds a 66 KB partials buffer). Energy is not measurable here
  without root (AMD powercap empty, as in the #59 notebook session). Cold load: repack 837–1379 ms + upload
  442–529 ms.
- CPU fallback: the ggml CPU engine takes 12047–12117 ms on medium (RTF
  0.54), ~3× the fast engine's 4021–4122 ms; nothing here touches it.
- Vendor gate: the split first shipped on every vendor, including PowerVR,
  whose phone numbers predate it. It now defaults on only for AMD;
  `STARLING_FAST_ATTN_SPLIT` opts other devices in and `=0` forces the
  single pass. RADV verified unchanged.
- CI-parity tests: `fast_weights_test`, `moss_mel_test`, `moss_encoder_test`
  pass; `moss_llm_test` fails identically on the baseline binary (maxabs
  0.8838425, a pre-existing stale golden on the ggml LLM path).

Open follow-ups: a W4/W8g8 lm_head recipe (−3 ms/step, ~−8 % decode;
quality-gated artifact change, #316 class); prefill GEMM work below the tile
level (F16MATH issue efficiency, cooperative matrix); a disk cache of the
repacked arena (SFPK) for load time; attention score-phase coalescing (attn
still 18.7 GB/s inside the split); the S=6 default flip for long-form;
validating the AMD pins on other AMD GPUs (measured on RENOIR only); and a
Pixel A/B of `STARLING_FAST_ATTN_SPLIT=4` under the phone gates — the
16-workgroup latency argument likely applies to the DXT too.

## #325 prevention work (2026-10-03, branch `fix/325-device-lifetime`)

### P3-1: a wedge on in-process device re-creation — GPU rail ~0 mW while the fence hung

Run: `starling-bench --cycles 3 --runs 2` (Parakeet q4_k_m-shrink16,
medium.wav, `STARLING_ENGINE=fast`, KSTEP default) — the new refcounted
context destroys the VkDevice when the engine is freed and creates a fresh
one on the next load, all in one process. Phone: uptime 15 h (user reboot
the morning before), battery 49 %, **discharging** (no stay-awake hold is
possible unplugged), 30 °C skin, ~01:53 at night; the same test had passed
3/3 cycles on the previous boot.

- Cycle 1: load 2.07 s, runs 2.60 / 2.30 s, `vk teardown: device destroyed
  cleanly`.
- Cycle 2: load 3.18 s (slow), first transcribe: fence timeout at the
  120 s default, drain grace (10 s) expired → context marked hung; the
  process still exited by itself (cycle 3 refused by the wedge; no stuck
  process) — the #379 bounded-teardown paths held on real hardware.
- Forensics (`wedge_forensics.sh event`, `~/starling-forensics/
  20261003-015557-event-cycles-reload-wedge`): **no kernel log line at all**
  between 01:41 and the capture — the PowerVR driver reported no hardware
  recovery, lockup or fault for a 120 s hang. pixel-thermal's per-minute
  power rails during the hang: **`S2S_VDD_GPU` 0.80 / 0.73 mW** (GPU idle,
  effectively power-gated) with the CPU rails at 50–67 mW: the GPU was not
  busy with a long job — the submitted work never completed while the GPU
  sat idle (lost submission or lost completion). Power state at capture:
  `mWakefulness=Dozing`, deep-idle `IDLE`, `mStayOn=false`, no suspend
  blockers held. The main logcat buffer had already rotated past the hang
  (pixel-thermal floods it in < 1 min) → `wedge_forensics.sh watch start`
  added: a rotating device-side logcat for the whole session.
- Boot log of this unit (any boot): `gpu-secure-trusty:
  RGXValidateFWHeaderVersion2: KM and FW version mismatch (expected: 24.2,
  found: 25.3)` — recorded for the upstream report; meaning unknown.

Open: in-process re-creation vs. doze/suspend as the trigger (P3-2). The
GPU-idle signature fits the earlier locked-phone stalls (P2-3/P2-4: system
suspend between rounds) and the night-time, unplugged wedges of #325. The
app held **no** wake lock at all (the old note in `docs/fast-engine.md`
was wrong); it now holds a partial wake lock around on-device GPU work.

### P3-2: in-process re-creation is clean when the phone is awake — doze is the suspect

Same binary and command as P3-1, after the marker expired (02:11), phone
held awake (`input keyevent KEYCODE_WAKEUP` re-sent every ~10 s; still
unplugged): **3/3 and then 5/5 cycles clean** — every free printed
`device destroyed cleanly`, loads 1.2–2.6 s, Parakeet medium 2.26–3.17 s
(in band). With the previous boot's 3/3, that is 11/11 awake in-process
device re-creations against 1/1 wedged while dozing. In-process
re-creation is not the trigger on this evidence; screen-off doze/suspend
during GPU work is (n=1 on the failing side — a deliberate dozing repro is
the next experiment, and risks a phone restart).

Consequences landed: the app holds a partial wake lock for loads,
transcriptions and the idle release (device teardown); bench sessions on
an unplugged phone must keep it awake (see the protocol).


### P3-3: deliberate doze repro — 3/3 dozing trials wedged, 2/2 awake clean

`doze_repro.sh` (02:35–03:22): alternating trials of the P3-1 workload
(one process, `--cycles 3 --runs 2`, Parakeet q4_k_m-shrink16, medium.wav),
phone unplugged, battery 47 → 43 %, rotating logcat throughout. D = screen
off + `dumpsys deviceidle force-idle`, no wake signals; A = Doze lifted,
screen woken every poll. Planned D A D A D A; stopped after trial 5 on the
3-wedge rule. Results: `~/starling-forensics/doze-repro-20261003-023500`.

| trial | arm | outcome |
|---|---|---|
| 1 | D | wedge on cycle 3's first transcription (2 clean cycles before) |
| 2 | A | clean, runs 2.33–4.03 s |
| 3 | D | wedge on cycle 1's first transcription |
| 4 | A | clean, runs 2.23–2.46 s |
| 5 | D | cycle 1 run 0 stalled 48.0 s then recovered (the P2-3 suspend-stall pattern); wedge on cycle 2 run 1 |

- Every wedge has the P3-1 signature: `S2S_VDD_GPU` 133–174 mW while
  transcribing, then 0.6–0.8 mW for the whole 120 s fence wait (GPU idle
  with work outstanding); no PowerVR kernel line; `mWakefulness=Dozing`,
  no suspend blockers held. The #379 bounded teardown held each time (no
  stuck process; marker written; later cycles refused).
- No restart: device uptime ran on (15:56 → 16:38 h across the wedges; 1 d
  54 min the next morning), boot reasons unchanged. Three wedges in an hour
  did not escalate.
- The wedge point varies (first transcription of a fresh device in 1 and 3,
  second run on a device in 5): not tied to device re-creation.
- Deep-idle `mState` read IDLE in trial 2 and ACTIVE in trial 4, both clean:
  the variable that separates the arms is wakefulness (Dozing vs. Awake —
  whether the kernel may suspend), not the deviceidle state itself.

Tally with P3-1/P3-2: **dozing 4/4 wedged, awake 0/13** (Fisher exact,
one-sided p = 1/C(17,4) ≈ 4·10⁻⁴). Trigger confirmed: GPU work submitted
while the phone is allowed to suspend. The likely mechanism is a suspend/
resume of the PowerVR stack losing an in-flight job or its completion —
driver/firmware code we cannot fix. Prevention is to never let the phone
suspend with GPU work outstanding: the app's partial wake lock (b340ad8)
and the bench protocol's awake rule. `doze_repro.sh` is the upstream
reproducer.

Open: the A arm kept the screen on; the app's guard is a screen-off partial
wake lock, which blocks kernel suspend but is ignored by Doze for apps that
are not exempt. Untested whether it prevents the wedge — the next arm is
screen off + Doze + a held partial wake lock (needs the app, since shell
cannot take a kernel wake lock on a user build).

### P3-4: a shell-uid wake lock prevents the wedge in deep Doze — the trigger is suspend, not Doze

Shell cannot write `/sys/power/wake_lock` on a user build, but it holds
`android.permission.WAKE_LOCK`. `wakehold/WakeHold.java` (run via
`app_process`) takes a `PARTIAL_WAKE_LOCK` through `IPowerManager` as uid
2000. Checked in `dumpsys power` with the screen off and deep Doze forced
(`mWakefulness=Dozing`, `mState=IDLE`): the lock stays active (not
`DISABLED` — Doze only disables app-uid locks) and holds the suspend
blocker (`mHoldingWakeLockSuspendBlocker=true`); killing the holder releases
it.

`doze_repro.sh "W D W D W"` (2026-10-05 15:12–15:35), phone on the charger
with `FAKE_UNPLUG=1` (`dumpsys battery unplug`, so Doze engages); W = the D
setup plus the wake lock. Results:
`~/starling-forensics/doze-repro-20261005-151249`.

| trial | arm | outcome |
|---|---|---|
| 1 | W | clean, runs 2.25–2.51 s, lock held at the end |
| 2 | D | wedge on cycle 1's first transcription; GPU rail 0.74–0.98 mW during the fence wait |
| 3 | W | clean, runs 2.17–2.49 s, lock held at the end |
| 4 | D | clean, runs 2.19–2.47 s |
| 5 | W | clean, runs 2.19–2.45 s, lock held at the end |

- The D control wedged with the P3-1 signature while the charger was
  connected: physical discharge is not part of the trigger.
- The W arms ran in forced deep Doze (`mState=IDLE`) with only suspend
  blocked, and were clean in the normal band. Doze itself is not the
  trigger; kernel suspend with GPU work outstanding is. A plugged-in phone
  with the screen off can suspend too.
- Tally (P3-1–P3-4): screen off without a lock 5/6 wedged, awake 0/13,
  screen off in deep Doze under the lock 0/3 (W vs. unlocked: Fisher exact,
  one-sided p = 6/126 ≈ 0.05). Small n on the W side; strong mechanism
  evidence.

Consequences landed: `phone_gates.sh` and `phone_energy.sh` hold the lock
for the whole session (`phone_common.sh` `wake_hold` / `wake_release` /
`wake_held`), and the protocol rule now names suspend. Open: the app's own
wake lock is an app-uid lock, which deep Doze disables. App GPU work in
deep idle needs a long stationary screen-off period first, so the window is
small, but not closed.
