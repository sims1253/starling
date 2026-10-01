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

## #59 notebook campaign: granite encoder K/V low-rank follow-up (2026-10-01)

Device: notebook (Ryzen 5 PRO 5650U, 12 threads, CPU backend, reduced power
profile for part of the session). Target: `cpp/granite/encoder.cpp`
attention K/V projections via the opt-in `STARLING_GRANITE_KVFACT` factor
path (off by default). Baseline: frozen `starling-bench` built at branch
point. A/B: alternating arms, median of 3 reps x 3 in-process runs,
medium.wav (enc stage 5.62-5.65 s under the reduced profile). Contract:
short/medium/long transcripts identical for every kept row. WER gate
(FLEURS en_us test 100 clips, |delta| <= 0.2 vs baseline ggml):

| Experiment | enc delta | wall delta | transcripts | WER delta | verdict |
| --- | ---: | ---: | --- | ---: | --- |
| weight-space SVD K r=32 (f32 factors) | +2.22% | +0.74% | identical | not gated | kept (tooling) |
| same, Q8_0 factors | +0.99% | -0.62% | identical | - | discarded (slower: narrow quantized dots lose to f32 FMA) |
| weight-space K r=16 | +2.79% | +1.15% | identical | +0.19 (5.85->6.03) PASS | only gate-clearing config |
| weight-space K r=24 | +2.56% | +0.71% | identical | +0.28 FAIL | discarded |
| weight-space K r=8 | - | - | DIFFER | - | rejected at probe |
| K+V r=32 | - | - | destroyed | - | rejected at probe (V not compressible) |
| activation basis r=32 (bf16 ref K) | +2.20% | +0.82% | identical | +0.42 FAIL | discarded |
| runtime-fitted basis r=32 (engine K dumps) | +2.12% | +1.14% | identical | +0.28 FAIL | discarded |

Memory: k16 factors raise peak RSS 1900->1929 MB (+1.5%) — with Q4_K
production weights, low-rank factors cannot save memory (f32 r=16 factors
are ~2x the original K-half bytes; even q8 factors exceed Q4_K below the
quality cliff). This is a latency-only trade.

Findings: (1) basis provenance does not explain the WER cost — bf16-fit and
runtime-fit bases have identical per-layer projection error (~0.04 mean,
layers 3/7 ~0.11) and both fail; PCA-style bases zero out-of-basis
directions that OOD test audio excites, while weight-space SVD keeps
uniform spectral error. (2) WER deltas are deterministic (greedy decode)
but do not scale monotonically with rank — every rank 16-32 costs
0.19-0.42 points; k16's pass has ~1 flipped word of margin. (3) The
encoder attention path is 16.6% of encoder MACs (K path 4.5%); the measured
+2.2-2.8% enc / +0.7-1.15% wall ceiling matches the compute model — no
factorization can beat it materially. Conclusion for #59 on granite:
K-only rank-16 weight-space is a marginal, gate-edge latency win; V and
basis approaches are measured no-gos. Runtime default unchanged (path is
env-gated off); tooling committed (exporters + STARLING_GRANITE_DUMP_K).

Margin stress test: k16 re-gated on a second disjoint FLEURS draw
(clips 100-199): +0.29 FAIL (base 5.38 -> 5.67). Draw-1's +0.19 pass was
clip-draw luck. Verdict amended to a clean no-go: no tested configuration
reliably clears the 0.2-point WER gate; runtime default stays unchanged
(factor path env-gated off).

Selective per-layer map (idea backlog → measured): compress ONLY the six
layers whose runtime-K is genuinely low-rank (map {0:r32, 6:r32, 9:r8,
10:r24, 11:r32, 13:r32}, chosen from calibration dumps — per-layer held-out
projection error <= 0.0045; all other layers full rank). Latency enc
+0.83% / wall +0.50% (matches the MAC model 0.83%). WER: -0.19 on draw 1
(5.85->5.66) AND -0.19 on draw 2 (5.38->5.19) — deterministic, passes the
gate on both disjoint draws with margin to spare; consistent with the
projection denoising Q4_K error orthogonal to the speech subspace on
near-exactly-low-rank layers. Transcripts identical on all fixtures.
KEEP: v3 per-layer-rank factor format (STLGKVF3) + selective exporter.
Verdict nuance: uniform ranks fail the WER gate; selective per-layer maps
pass it — encoder KV "compression" on granite is viable only in this
selective form (+0.83% enc ceiling for K).

Selective-map boundary probe (same campaign): V-side per-layer dumps show V
is full-rank in EVERY layer (held-out rel-MSE 0.12-0.51 at r=32, 0.05-0.39
at r=48; even layer 0, whose K is near rank-8, has V err 0.40 at r=8) —
the K/V asymmetry is per-layer-intrinsic, closing the V question at zero
WER risk. K-map expanded with the next-best calibration layers
{4:r32 err 0.016, 12:r32 err 0.011} -> 8-of-16-layer map: enc +1.27%,
wall +0.54%, transcripts identical, WER -0.19 / -0.14 on the two disjoint
draws (both PASS). KEEP. The frontier boundary is now measured: remaining
uncompressed K layers sit at r32 error 0.038-0.13, the regime where uniform
compression failed the gate (+0.28) — further expansion is not pursued.

Per-head packed map probe (v4, same campaign): per-head dump analysis shows
compression structure is per-HEAD (L8's 0.038 layer mean = head 6 alone at
0.162, other seven heads <= 0.0049; L14 has five clean heads). A v4 format
(packed per-head widths; skipped heads as full-rank identity blocks) added
L8(7 heads)+L14(5 heads): enc +1.35% — only +0.08% over the kept sel2 (the
wide 544-column factor GEMMs eat the partial-layer gain; increment within
cross-run drift) — and FAILED the two-sided WER neutrality gate on the
improvement side: draw1 -0.28 (5.85->5.56) while draw2 -0.10. The identity
blocks route untouched heads through f32-GEMM(dequant W) instead of the
Q4KxQ8 dot, materially moving transcripts (the same effect as the CTC
study's rank-1024 control). Reverted; the kept final config remains the
v3 8-layer map (sel2, enc +1.27%, WER -0.19/-0.14). Per-head granularity is
real structure but not profitably exploitable within the quality gate.

Per-head piece-wise exact routing (v5, same campaign): second variant of the
per-head idea, fixing v4's mechanism (residual heads now sliced from the
real attn_kv weight — exact Q4_K dot, no shipped dequantized weights).
WER -0.19 / +0.05 on the two draws: BOTH PASS, confirming identity blocks
were v4's gate violation and per-head compression itself is gate-safe. But
enc +1.26% is identical to the kept sel2 (+1.27%): the multi-piece overhead
(extra GEMM dispatches + concats per partial layer per transcription)
cancels the theoretical +0.25% MAC gain. Equal latency at higher complexity
=> discarded (v5 loader/encoder/exporter reverted; spec preserved here).
Per-head idea closed after two variants: the structure is real and
gate-compatible, but not profitably exploitable on this engine/CPU. Final
config remains the v3 8-layer selective map.

Post-discard audit note: the v5 variant additionally carried a latent
DEFAULT-PATH segfault (has_pieces() treated pieces_k.size()==n_layers==0 as
"v5 loaded" and indexed pieces_k[li] out of bounds with no factor file set).
It never affected run 12's measurements (every gate ran with the factor file
present) and the code was reverted by the discard before it could reach any
commit; the rebuilt clean binary re-verified transcript-identical on the
default path. Any future factor-format work must gate on an explicit
"loaded" flag, not size==n_layers.

In-r-space score folding (run 13): the old "blocked by WER" rollback reason
died with sel2's gate pass, so the idea was retried with that assumption
changed. K stays in the rank basis (z) and the expand basis folds into q
(qtilde = f2T @ q, an equal-MAC swap for the dropped expand GEMM); content
scores then contract over rank instead of head_dim for the 8 compressed
layers. Same v3 factor file (f2T is a load-time transpose); the intermediate
bf16 round moves from the K side to the Q side. Measured on top of sel2:
enc +1.67% (a +0.40pt increment over sel2's +1.27; predicted +0.35),
transcripts identical, WER -0.19 / -0.05 on the two disjoint draws (draw1
byte-identical to sel2's transcripts outcome). KEPT, env-gated behind
STARLING_GRANITE_KVINR (default off = expand path). Final best config:
sel2 + KVINR: enc +1.67%.

Final verification (run 14): sel2+KVINR re-measured in a fresh session —
enc +1.68% / wall +0.94% (run 13: +1.67% / +0.15%; the enc delta
reproduces to 0.01pt, the wall delta wanders in a ~0.15-0.94% band as
noted throughout). Transcripts identical; per-arm spreads ~±0.1%. This is
the campaign's certified final number for the granite encoder K path:
SELECTIVE 8-layer rank map + in-r scores = enc +1.67-1.68% vs the frozen
baseline, exact-transcript contract intact, FLEURS WER neutral-to-positive
on two disjoint draws, runtime default unchanged (both knobs env-gated).

Cross-domain validation (run 15): (a) long-workload A/B (74 s fixture, 10
chunks, chunk-policy path): enc +1.65% — matches medium's +1.67/+1.68, no
shape tuning. (b) DOMAIN PROBE: sel2+KVINR on the LibriSpeech-style
real_corpus (8 clips, held from all calibration): base 4.49 -> cand 5.13 =
+0.64, OUTSIDE the 0.2 gate. The FLEURS-fit bases do not generalize across
domain — the OOD mechanism the campaign documented, confirmed at quality
level. (c) Mixed-domain refit (same map, bases refit on FLEURS 0-4 +
real_corpus utts 0-3; utts 4-7 and FLEURS draws held out): FLEURS draw1
-0.28 (violation, improvement side), draw2 -0.10 pass, real holdout +0.78
FAIL — no better in-domain than the FLEURS fit on 4 clips / ~200 words.
Conclusion: the selective map's WER neutrality is DOMAIN-CONDITIONED
(FLEURS-like speech only); basis composition cannot satisfy both domains
inside the +-0.2 band on these small sets. Deliverable claim downgraded:
"WER neutral on the calibration domain; +0.6-0.8 observed on a small
read-speech set" — the env-gated OFF default is the correct production
posture; any enablement needs domain-matched re-gating.

Functional-surface survey + hygiene (run 16): the granite CTC speculative
path (ctc_max_k > 0, extract_ctc_draft -> shaw_attention with factors)
has NO live consumer today (serve/stream/tools never pass it; only the
deprecated Python research backend used it) — exercising the factor path
there would require a starling-bench flag, i.e. a measurement-stack change
(out of loop scope per AUTORESEARCH hard constraint 1; noted as future
issue material if the spec path is ever wired up). Hygiene: f2tk (the
in-r transpose, ~15 MB) is now materialized only when
STARLING_GRANITE_KVINR is set. Confirmation A/B after the change: enc
+1.55% — within the config's four-session band (1.55/1.65/1.67/1.68,
mean ~1.64); transcripts identical; default path unchanged.

Repo-test backpressure (run 17): fresh clean configure+build of the
committed branch with STARLING_GGML_TESTS=ON; all CI-relevant granite
tests PASS — granite_stage_test, granite_trace_test, granite_ctc_argmax_
test, granite_ctc_proposer_test, granite_ctc_entry_test,
granite_fairness_test. (granite_ctc_chunk/fused_parity need external CTC
GGUF fixtures and are not in CI.) The #59 research path changes
(capi/encoder/loader/kv_factors) are regression-clean against the repo's
own suite; no code change in this run.

Mechanism test on the unquantized model (run 18): the same selective
map+in-r config refit from granite-speech-4.1-2b-bf16-exact.gguf (K dumps
re-taken; factors bit-exact vs HF verified after fixing a BF16 reinterp
bug in dequant_2d). Results: enc +1.61% (same band as dynq4's
1.55-1.68% — the win is structural, not quant-dependent); transcripts
identical; FLEURS draw1 WER +0.05 (vs dynq4's -0.19) and the
LibriSpeech-style domain probe +0.00 (vs dynq4's +0.64 FAIL).
Mechanism conclusions, now measured rather than inferred: (a) the WER
"improvements" on dynq4 were removal of Q4_K quantization noise by the
speech-subspace projection (pure truncation costs only ~+0.05 in-domain);
(b) the cross-domain failure on dynq4 is an interaction between the
projection and weight quantization on OOD speech — on exact weights the
same truncation costs ZERO cross-domain. Deliverable refinement: on the
bf16-exact GGUF the config passes every gate on BOTH domains; the
domain-conditional caveat is specific to the aggressively quantized
dynq4 file. Side observation: the bf16 encoder is faster than dynq4's on
this CPU (bf16 row-dots beat Q4_K block-dots for these shapes).

bf16 workload generalization (run 19): the run-18 bf16-exact result
generalizes — long fixture (74 s, 10 chunks): enc +1.56% (medium was
+1.61%; dynq4 long was +1.65%), transcripts identical on short/medium/
long under bf16+factors (the long fixture gap from run 18 closed).
Follow-up filed for a future issue (NOT this loop's scope): on this CPU
the bf16-exact encoder is ~6% FASTER than the Q4_K one (bf16 row-dots
beat Q4_K block-dots for these shapes) — an engine-level load-time
repack of quantized GEMM rows to f32/bf16 could recover that for quantized
model files generally; transcript-contract risk is the dot-method change
(same class as the full-rank control effect), so it needs its own gated
campaign. Also re-filed: a starling-bench flag to exercise the (currently
unconsumed) granite CTC speculative path under factors.

STARLING_GGML_CPU_REPACK on the notebook (run 20): the engine's existing
repack knob (cpp/runtime/cpu_repack, ggml CPU_REPACK interleaved kernels;
default ON only on Android) was never measured in this campaign — the
frozen baseline ran with it off. Env-only A/B on the SAME frozen binary
(granite dynq4, medium): enc +18.99% (5746->4655 ms), wall +9.25%
(13.97->12.67 s), transcripts identical, FLEURS WER -0.09 / +0.00 — every
gate green, zero code change. Stacked with the #59 factor path
(sel2+KVINR): enc +19.62%, wall +8.69% (wall within band), WER -0.14 /
+0.00 — gates green; the factor path keeps a ~+0.6pt edge on top of
repack (its f32 GEMMs still beat repacked-Q4_K on the compressed K
layers). This reframes the notebook recommendation: the big granite
encoder win is the existing repack knob (12x the factor path), with the
selective map adding a further ~0.6%. Whether x86 should default
STARLING_GGML_CPU_REPACK on (platform_default in cpu_repack.cpp) is a
one-line production decision for humans — this loop records the measured
numbers, it does not change the default. Also note repack's
transcript-exact behavior here contrasts with the bf16/dot-method concern
that killed v4: ggml's repacked kernels preserve the fixtures and WER
neutrality on both draws.

Domain-probe reinterpretation + long-workload seal (run 21): the 8-clip
LibriSpeech-style set is a boundary-word hair-trigger, not a domain gate.
Repack-ONLY (factors off) produces the SAME +0.64 (4.49->5.13) the factor
path produced in run 15, and the stacked config lands on the same
transcript set (one punctuation-level difference in clip 8) — the same ~3
near-boundary words flip under ANY Q8-dot -> f32-class accumulation change.
The bf16 model (+0.00, run 18) is the control: its baseline already uses
f32-class dots. Unified mechanism: the flips are an accumulation-method
phenomenon visible only on this tiny set; on the campaign's actual quality
bar (FLEURS draws, 100 clips) repack, factors and stacked are all neutral
(-0.19..+0.05). Run 15's "domain-conditional caveat" is therefore
reinterpretED: the factor path shares the dot-method sensitivity of the
SHIPPED repack knob (Android default) and has no unique domain fragility.
Long-workload seal for the recommended config (repack+sel2+KVINR, 74 s
fixture): enc +19.00% / wall +8.84% — matches medium's +19.62/+8.69 band.

Repack blast-radius sweep (run 22): STARLING_GGML_CPU_REPACK=1 on the
frozen binary across the other ggml-engine models (transcript contract,
short+medium fixtures): parakeet-tdt q4, MOSS q4, qwen3-1.7b dynq4 —
transcripts IDENTICAL on every model (granite already validated in run
20). Indicative single-run wall times on medium.wav (not the paired A/B
protocol): parakeet 2128->1641 ms (-23%), moss 13564->11728 ms (-14%),
qwen3 13327->12500 ms (-6%). The x86 repack recommendation is therefore
engine-wide: every measured model benefits and every transcript contract
holds. A rigorous per-model A/B (alternating, >=3 reps) belongs to the
repack-default issue if humans take it.

Memory posture (run 23): peak RSS on granite dynq4 short.wav — default
1900 MB, repack-only 1904 MB (+4 MB: the q4_K_8x8 interleaved layout is
size-neutral), repack+sel2+KVINR 1935 MB (+35 MB: factor file + f2tk +
misc). The repack log also confirms the designed interaction: with factors
on, the compressed layers' enc.blk.{0,4,6,9,10,11,12,13}.attn_kv are
absent from the repack list (their view-use marks them never-repack;
everything else repacks). Recommendation due diligence complete:
performance, quality, blast radius, and memory are all measured for the
x86 repack decision.

Thread-count characterization (run 25): the engine already defaults to
physical cores (graph.cpp: "SMT siblings measured slower at every stage";
STARLING_GGML_THREADS overrides). Sweep on this 6C/12T notebook with the
recommended config (repack+sel2+KVINR, short.wav, in-process medians):
threads 4 / 6 / 8 / 12 -> enc stage 1849 / 1404 / 1439 / 1852 ms. The
default (6 physical) is optimal on this device; the SMT penalty (~32% at
12 threads) reconfirmed. No change warranted.

Per-model rigorous repack A/Bs (run 26): the run-22 indicative numbers
upgraded to the paired alternating protocol (3x3, same frozen binary both
arms, env-only). Wall medians on medium.wav: parakeet-tdt q4 2119.7 ->
1639.2 ms (-22.67%, spreads +-0.3%); MOSS q4 13712.3 -> 11859.6 ms
(-13.51%); qwen3-1.7b dynq4 13404.8 -> 12585.6 ms (-6.11%). All three
match the indicative sweep (-23/-14/-6). Transcript contracts were
validated in run 22. Session-tooling note: measure.sh's arm dispatch had
a latent bug for same-binary env-only A/Bs (both arms landed in the base
array); granite runs were unaffected (distinct binaries). Fixed before
any per-model number was logged — the first parakeet outputs that hit
the bug were discarded un-logged.

Repack scope + workload matrix completion (run 27): (a) the CPU_REPACK
mechanism covers quantized types only — on granite-speech bf16-exact ZERO
tensors repack (enc unchanged at ~5.35 s), so an x86 default flip's blast
radius is exactly the quantized model files; unquantized ones are
untouched. (b) Short-fixture cell (single chunk, T~400, narrowest GEMMs):
repack enc +21.65% / wall +8.77% — the workload matrix is now complete
across short/medium/long (enc +21.65 / +18.99 / +19.00 repack-alone) and
the effect does not degrade at small T.

Energy + cold-start closure (run 28): (a) energy per transcription is NOT
measurable on this device — the AMD powercap interface exists but the
RAPL energy counters read empty without root/kernel support; noted for any
future energy campaign. (b) Cold-start cell (first request after process
start, no warmup — includes the one-time lazy repack transformation of
every touched weight): median of 3 alternating fresh-process runs on
short.wav, base 5010 ms vs repack 4659 ms (-7.0%). The repack
transformation pays for itself within the very first transcription; no
warm-up penalty exists for serving. All cells of the repack dossier that
this device can measure are now closed.

Serve-level contract closure (run 29): the exact-transcript contract run
through AUTORESEARCH's named serve harness (benchmarks/experiments/
serve_contract_smoke.py over HTTP, fresh server processes, all three
fixtures) for the recommended configuration: baseline serve vs the same
binary with STARLING_GGML_CPU_REPACK=1 (expressed via an exec wrapper so
the script's own two-binary protocol stays unmodified) —
transcripts_match=1, 1390/1390 chars. Every harness element the standing
file names for notebook work has now been used by this campaign: the
paired-alternation discipline throughout, starling-bench for timing, and
the serve contract smoke for the HTTP path.

Artifact reproducibility (run 30): regenerating the pinned sel2 factor
file from the documented command (export_kv_lowrank_selective + the
granite dynq4 GGUF + the kdumps) byte-reproduces the pinned artifact
(sha256 prefix fb6fee5c930c3725 both). The chain GGUF -> K dumps ->
factors is bit-deterministic on this machine (greedy engine, fixed
inputs). Caveat for cross-machine reproduction: LAPACK SVD sign
conventions may differ across numpy builds — signs cancel in the subspace
math but change file bytes, so other machines should verify against the
recorded hashes rather than expect byte equality. The ctc study's
"fixed exported weight file" lesson is thus satisfied and documented.

Mel-threads knob closure (run 32): STARLING_MEL_THREADS sweep on the
recommended config (enc stage, medium): threads=1 -> 4724.8 ms, default
(=hw concurrency) -> 4690.1 ms. The mel front-end's entire parallelizable
cost is ~35 ms (~0.7% of the stage; total mel share ~1-1.5%), and the
default already sits at the fast end — the knob is closed with a number.
Every engine tunable that touches the granite enc metric has now been
measured: GGML threads (physical-core default optimal), CPU repack (the
+19-21.7% recommendation), MEL threads (sub-1%, default optimal), and the
#59 factor knobs (+1.6% stacked).
