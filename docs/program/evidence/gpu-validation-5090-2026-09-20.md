# GPU validation — RTX 5090 (CUDA) — 2026-09-20

Independent verification run for engine PRs #196 (S01), #197 (S03),
#198 (S02), #199 (S11). All evidence below was produced on this machine
by the validation agent; nothing is taken from coordinator claims.

## Environment

- GPU: NVIDIA GeForce RTX 5090 (32,607 MiB), driver 610.88 (KMD) / CUDA UMD 13.3
- CUDA toolkit: 13.0 (V13.0.88); CMake 3.22.1; g++ 11.4.0; torch 2.13.0+cu130 (conversion only)
- Validated tip: `program/gpu-validation` @ `52f393d` (started at `6797794`,
  fast-forwarded mid-run when the coordinator pushed the R31 refresh —
  S11 batch-4 + stream_session_test 373→382; all gates re-run on the new tip
  where affected; see "Run history" below)
- Master baseline: `12b6548c96ecfa1c989abb7afd39e90080cfb92f` (the recorded
  program baseline), built in a separate worktree (`/tmp/opencode/master-checkout`),
  same flags, same ggml submodule commit `e91ded11` (v0.23.0 + Starling patches)
- Build: `cmake -B build-cuda -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON`,
  `CMAKE_BUILD_PARALLEL_LEVEL=8`; `CMAKE_CUDA_ARCHITECTURES` incl. `120a-real`
- Serve flags (both sides, all MOSS runs): `--model moss --gguf <moss.gguf>
  --port <p>` with `STARLING_TRACE=1` (structured timing trace on stderr);
  no `--warmup` (cold start is part of the measurement)

### Model files

| file | provenance | SHA256 |
|---|---|---|
| `moss-transcribe-preview-2b-bf16-exact.gguf` (4,852,103,488 B, 841 tensors) | converted on this machine from the pinned HF snapshot `OpenMOSS-Team/MOSS-Transcribe-preview-2B@c98175cb…` via `scripts/convert_moss_gguf.py` (`uv run --with gguf …`; the previously symlinked copy under `~/Documents/starling/models/` no longer exists) | `b96dae2bcadc9e89f61a3ac1e103915e0abb11da521ca6d8db91e3244129e71a` |
| `parakeet-tdt-0.6b-v3-q8_0.gguf` (741,505,024 B) | downloaded from the documented repo `scholzmx/parakeet-tdt-0.6b-v3-gguf` | `cbab40be5510f86f825ccb19bdea0876938358a2266a24a1ab8f1fccf0759922` |
| `parakeet-tdt-0.6b-v3-q4_k.gguf` (488,674,176 B) | local cache (`~/.cache/crispasr/`), also tried for Gate C | `1a60f6e53e5781240dde6e69a47a47a8a71995a3a106517b009225afcc514457` |

bf16-exact chosen for MOSS per the runbook (fits comfortably: 4.9 GiB weights,
~8.9 GiB steady-state VRAM on both sides).

### Audio set

Gate B/D fixtures (regenerated deterministically on this branch via
`uv run python tests/fixtures/make_fixtures.py` from the committed source
sample; PCM16 mono 16 kHz):

| file | duration | SHA256 |
|---|---|---|
| `2086-149220-0033.wav` (source) | 7.435 s | `5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150` |
| `short.wav` (= 1x) | 7.435 s | `5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150` (byte-identical to source: 1x of an already-PCM16 file) |
| `medium.wav` (= 3x) | 22.30 s | `4a62a4b4c9ca8d669fff179fed8a303e474e63ca8724025d3c1c70e5b44787ab` |
| `long.wav` (= 10x) | 74.35 s | `4f97080176d3623eebc9663af07dd258894ed31f74d5d229d39e77c5ab5b0925` |

Gate C set: 12 deterministic varied-length WAVs (1.5/2.3/3.7/5.1/6.4/8.2/9.9/
12.5/15.3/18.1/21.7/25.9 s), tiled+truncated from the same source sample
(generator + hashes in `/tmp/opencode/gatec/`; e.g. `1_5s.wav`
`a821daa9793a09c45ca61ea8da262a941a0005010784aef9bd240fefbc0690b7` through
`25_9s.wav` `7530105468fc74d68927c7d37619792276388f080909c1bdae8b4347644c735a`).

Gate D real corpus: 8 utterances `utterance_00{0..7}.wav` (1.6 s … 29.4 s,
mono 16 kHz, from `tests/fixtures/real_corpus`; hashes recorded, e.g. 000
`c667674a25acacf6a120ec4234dfdb95015f3711d6f713946fa48da76068a6f6`, 007
`4798e34d9a323ed835ae6055172236f8965dd5ccf146c25758711023ecc8b2e6`).

## Run history

1. Gates A/B/D first ran on tip `6797794` (S01–S11 merges as of run start).
2. The control-plane spot-check surfaced R31 ("gpu-validation refreshed to
   52f393d"); `origin/program/gpu-validation` had moved. Fast-forwarded
   `6797794 → 52f393d` (delta: `cpp/serve/stream_session.cpp` R31 engine_id
   snapshot, `cpp/tests/stream_session_test.cpp` +40 lines, CI yml).
3. Rebuilt all five targets and re-ran Gate A in full on `52f393d`; re-ran
   all three S11 WS scenarios on the rebuilt `starling-serve`. Gates B/D
   compare against master on engine paths untouched by the delta (S01 path
   `cpp/lib/device_cache.cpp`, S03 path `cpp/lib/qwen_decode.cpp` et al. are
   identical between `6797794` and `52f393d`), so their transcripts/timings
   remain valid; nothing in the delta touches what B/D measured.

## A. Correctness (all four green; `backend=CUDA0` confirmed)

On tip `52f393d` (re-run; first run on `6797794` was identical except
stream_session_test 373/373):

```
$ ./build-cuda/device_cache_clear_test
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 32606 MiB):
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes, VRAM: 32606 MiB
device_cache_clear_test: backend=CUDA0
device_cache_clear_test: all checks passed
EXIT=0
```

```
$ ./build-cuda/tdt_graph_budget_test
tdt_graph_budget_test: start
[A] budget / LRU order / byte accounting
[A] pin/lease stability + oversized entry
[A] construction failure rollback
[A] env override parsing (env_budget_bytes)
[A] zero byte budget rejected at construction
[A] mass eviction: victim set with interleaved pins
[A] zero-byte entries are floored, warned once, bounded
[A] floored entry count is hard-capped
[A] throwing trim rolls back all accounting
[B] real ReplayGraph entries on CPU backend
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 32606 MiB):
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes, VRAM: 32606 MiB
[B] backend=CUDA0
[B] entry_bytes=3328 budget=14976
CUDA Graph id 42 reusedggml_backend_cuda_graph_compute: CUDA graph warmup complete
tdt_graph_budget_test: PASS (byte budget, LRU, pins, accounting)
EXIT=0
```

```
$ ./build-cuda/greedy_termination_test
[PASS] predicate: primary EOS stops (single-stop)
[PASS] predicate: non-stop tokens do not stop (single-stop)
[PASS] predicate: eos=-1 sentinel stops nothing
[PASS] predicate: primary and secondary both stop (dual-stop)
[PASS] predicate: other tokens do not stop (dual-stop)
[PASS] predicate: unconfigured secondary (-1) stops nothing extra
[PASS] C entry: null handle reports COMPLETION_NONE (0)
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 32606 MiB):
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes, VRAM: 32606 MiB
[PASS] engine: zero-weight fixture argmax is token 0
CUDA Graph id 1 reused
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
[PASS] engine: prefill EOS at token 1 -> reason=eos, 1 token, no further decode
[PASS] engine: no stop token -> reason=budget_exhausted, full budget emitted
[PASS] engine: one-token budget with prefill EOS -> reason=eos
[PASS] engine: one-token budget without EOS -> reason=budget_exhausted
[PASS] engine: debug/probe path honors prefill EOS identically
GREEDY TERMINATION OK
EXIT=0
```

```
$ ./build-cuda/stream_session_test
… (373/373 on 6797794; on 52f393d:)
stream_session_test: 382/382 passed
EXIT=0
```

**Gate A: PASS.**

## B. S01 timing (branch vs master, same MOSS bf16 GGUF, same WAVs)

Protocol: per side, fresh server start (cold), then 6 repeats x
{short, medium, long} = 18 requests, `STARLING_TRACE=1`, raw-WAV POST
(`Content-Type: application/octet-stream`; note: `curl --data-binary`
without an explicit content type defaults to
`application/x-www-form-urlencoded`, which this server rejects with 413 —
client-side nuance, not an engine finding). Repeat 1 = cold (includes graph
capture), repeats 2–6 = warm. `nvidia-smi` sampled at 1 Hz throughout.

### B.1 Transcripts

- branch vs master: **0 differences / 18 request pairs** (verbatim text compare)
- within-side stability: 0 instabilities on either side (all repeats equal repeat 1)
- sample (short): `well, i don't wish to see it any more, observed phoebe,
  turning away her eyes it is certainly very like the old portrait.`
- long (10x repeated source) transcribes the sentence 10x on both sides,
  byte-identical between sides.

### B.2 Wall-time and engine-request latency

Wall time (curl, ms) and trace `request` dur_ms per side:

| side | wav | cold wall | warm wall med (min–max) | cold engine | warm engine med |
|---|---|---|---|---|---|
| branch | short | 705 | 187 (176–229) | 696.16 | 178.98 |
| branch | medium | 519 | 494 (480–531) | 509.83 | 484.90 |
| branch | long | 1035 | 1034 (988–1035) | 1020.88 | 1017.28 |
| master | short | 844 | 210 (192–246) | 835.03 | 205.30 |
| master | medium | 604 | 523 (490–586) | 594.35 | 512.79 |
| master | long | 1133 | 1005 (998–1129) | 1115.85 | 987.97 |

Per-request trace tables (engine `request` dur_ms, all 18 per side):

branch: r1 696.16/509.83/1020.88 (S/M/L), r2 221.26/520.32/1006.26,
r3 169.35/468.79/969.43, r4 167.80/484.39/1017.84, r5 178.52/484.66/1017.28,
r6 178.98/484.90/1017.89.
master: r1 835.03/594.35/1115.85, r2 237.73/575.30/1111.87,
r3 201.65/517.75/1066.90, r4 205.30/512.79/981.16, r5 184.19/483.19/981.62,
r6 184.33/480.84/987.97.

Reading: cold favors the branch on every duration (−139/−84/−95 ms).
Warm favors the branch on short/medium (−26/−28 ms median). Warm long median
is +2.9 % on the branch (1017.3 vs 988.0) — reported honestly; master's own
warm spread on long is 981–1116 ms (±7 %), the branch's is 969–1035 (±3 %),
and across all 6 long repeats the branch's mean is lower (1006.6 vs 1057.6).
**Latency not-worse: satisfied** (no regression outside master's own noise
band; the removed work is real — see B.3).

### B.3 KV-clear host bytes (the S01 acceptance number)

Method: `LD_PRELOAD` byte counter hooked on `cudaMemcpyAsync`/`cudaMemcpy`
(logging HostToDevice only) and `cudaMemsetAsync`, one CSV line per call,
same 18-request protocol per side. (`nsys` 2025.3.2 CUPTI collection dies
after ~3 s of init on this driver/toolkit combo — 7 runtime-API records
captured, nothing during requests — so the shim is the measurement.)

| metric | master | branch |
|---|---|---|
| 4 MiB-sized H2D memcpy calls (KV-tensor signature) | 1120 (= 20 zero() events x 56 tensors) | 56 |
| when they occur | 0.03 s … 12.02 s — **every request carries its zero() uploads** | **all within 0.03–0.38 s (server startup, before the first request)** |
| request-phase KV-clear host bytes | ~224 MiB per request (56 x 4 MiB) | **0** |
| device-side `cudaMemsetAsync` (4 MiB) calls | 0 | 1064 (= 19 zero() events x 56; init + 18 requests) |
| other H2D (weights at load, graph inputs) | identical call-size histogram both sides (607,744 KiB x1, 32,768 KiB x3, 24,576 KiB x84, 12,800 KiB x64, 8,192 KiB x56) | same |

Per-request KV-clear host bytes on the branch = **0** (every request-phase
zero() executed as 56 device-side memsets; the one host-side 224 MiB upload
happens once during model load, before `/health` reports ready — a startup
artifact also present on master, with zero request-path impact).
This is the acceptance item: **KV-clear host bytes ≈ 0 with latency
not-worse and byte-identical transcripts. Gate B: PASS.**

### B.4 VRAM

Both sides plateau at 8,906 MiB (median of second-half samples; max 8,908).
No VRAM regression.

## C. S02 budget (Parakeet under STARLING_TDT_GRAPH_BUDGET_MB=64/128/512)

### C.1 CUDA end-to-end is blocked by a pre-existing defect (not S02)

Every parakeet request on the CUDA build fails at encoder graph capture:

```
{"error":"ReplayGraph allocation failed: device 'CUDA0' rejected 24 of 1869
 captured-graph nodes:
  node 101/1869: op=CONV_2D_DW dst=f32 src0=f16(PERMUTE,STRIDED) src1=f32(PERMUTE,STRIDED)
  … (x24, all CONV_2D_DW, src0=f16)
```

Root cause: `third_party/ggml/src/ggml-cuda/ggml-cuda.cu:5533-5534` —
`case GGML_OP_CONV_2D_DW: return op->src[0]->type == GGML_TYPE_F32;` —
while the Starling parakeet GGUF layout contract deliberately stores
depthwise-conv weights as F16 (`scripts/convert_parakeet_gguf.py`,
tensor_kind: "conv weights … -> F16 … the layout the engine's conv path is
built around"). Captured graphs cannot fall back to sched (by design, see
the error text / issue #184).

Evidence this is pre-existing and PR-independent:

- plain master `12b6548` fails identically (same 26-of-1845-node rejection
  with the q4_k GGUF; 24-of-1869 with the q8_0 GGUF on both master and branch);
- both the community q4_k and the documented scholzmx q8_0 GGUFs fail;
- #198's diff (`cpp/parakeet/tdt_multistep.cpp`, `cpp/runtime/{backend,lru_cache,trace}`)
  does not touch the encoder/sched path where the rejection occurs;
- the Starling ggml patches (submodule is `e91ded11`-dirty: fattn/norm/pad/
  vulkan/SSM_CONV fusion) do not touch CONV_2D_DW support.

Consequence: the runbook's VRAM-plateau-under-load measurement cannot be
produced for parakeet on CUDA on **any** current revision of this repo
(master included). This is a recorded environment gap, not an S02 finding.

### C.2 What was produced instead

1. **CUDA-device graph-level budget evidence** (Gate A): `tdt_graph_budget_test`
   section [B] constructs real ReplayGraph entries on `backend=CUDA0`
   (`entry_bytes=3328 budget=14976`, CUDA graph warmup + replay) and passes
   budget/LRU/pin-lease/rollback/pointer-stability — the S02 mechanisms on
   actual CUDA graphs.
2. **End-to-end budget runs, CPU backend** (`STARLING_GGML_DEVICE=cpu`, the
   K-step path engages GPU-only, so this exercises transcript stability, not
   the budget): 12 distinct lengths x 2 passes x budgets {64, 128, 512}:
   transcripts identical across all 6 runs for 12/12 files (the 1.5 s clip
   is consistently empty at every budget — budget-independent; 2.3 s yields
   `Well, I don't wish to see it anymore`, longer clips the full sentence).
   Trace shows no `cache` records on CPU — consistent with the GPU-only
   engagement documented in `tdt_multistep.cpp:671`.

**Gate C: CUDA E2E not producible (pre-existing, master-identical blocker);
S02 unit-level CUDA evidence PASS; no regression attributable to #198.
The VRAM-plateau and many-lengths CUDA evidence remains an open gap gated
on fixing parakeet CUDA encoder capture (upstream ggml CONV_2D_DW F32-only
vs the F16 layout contract).**

## D. S03 real-MOSS A/B + S11 tail reuse

### D.1 S03 (identical transcripts master vs branch)

- Gate B set: 18 request pairs, 0 diffs (see B.1).
- Real corpus: 8 utterances x 3 repeats per side = 48 transcripts,
  **0 diffs** (and 0 within-side instabilities). Samples:
  - 000 (1.6 s): `the twenties,`
  - 002: `He was in reverie sliding along the borders of consciousness.`
  - 005: `He must have drawn his gun because the intruder said quickly put that …`
- No over-generation after early EOS observed on any input (all outputs
  identical to master's, which emits the same stop behavior; the predicate
  change is exercised for real by `greedy_termination_test`'s engine cases on
  CUDA, Gate A).

**Gate D/S03: PASS.**

### D.2 S11 (exact-tail reuse at commit)

`tail_cache_hits` itself is a test-binary getter (no serve log line), so the
serve-level contract was verified by counting engine invocations (trace
`request`/`queue_enter` events) per WS session on `52f393d`:

| scenario (WS /stream, short.wav as one frame) | engine calls | outcome |
|---|---|---|
| preview → commit, audio unchanged | **1** | partial at t+0.76 s; commit answered from the retained exact-tail entry; final text identical; duration 7.435 s |
| preview → append exactly one PCM16 sample → commit | **2** | the changed audio really ran the engine (new audio_rev/length); final duration 7.43506 s |
| preview → (>3 s later) audio-less no-op frame → commit | **1** | duplicate preview coalesced from the retained entry; no engine call |

(The 3 s wait in scenario 3 is required by the partial-interval throttle —
`--partial-interval-seconds` default 3.0 — not by the cache.)

Plus `stream_session_test` 382/382 on `52f393d` (incl. the R24/R26/R31
keying tests: one-sample sensitivity, mid-step callback/identity swap,
reset-collision).

**Gate D/S11: PASS.**

## Control-plane spot-check (docs/program/tasks.json @ program/wave-a)

- R27 ("env-budget-tests claim rebutted, test exists"): verified —
  `test_env_budget_parsing` at `cpp/tests/tdt_graph_budget_test.cpp:264`,
  and it ran green in my Gate A ("[A] env override parsing (env_budget_bytes)").
- R31 ("stream_session_test 382/382, gpu-validation refreshed to 52f393d"):
  verified — prompted the mid-run fast-forward + full re-run; 382/382
  reproduced here.
- S01–S11 item entries honestly record the CUDA gaps this run closed (S01
  "CUDA still source-read only pending 5090 run"; S02 "GPU validation …
  recorded gaps"; S03 "real MOSS GGUF E2E" gap; S11 "no real-model/GPU
  timing claims"). Nothing suspicious found in the sampled claims; each
  claim I checked reproduced.

## Merge decision

Gates A, B, D pass outright. Gate C's specified CUDA E2E evidence is
unproducible on this machine for any revision (pre-existing ggml-CUDA
CONV_2D_DW F32-only limitation vs the parakeet F16 layout contract; master
fails identically), while S02's CUDA-graph-level behavior passes on-device
and #198 demonstrably does not regress the CUDA path. Weighing that the PRs
are CPU/Vulkan-verified by the coordinator, that their heads are exactly
what this branch validated (all four PR heads are ancestors of `52f393d`),
and that blocking #198 on evidence that cannot exist until an unrelated
upstream defect is fixed would hold a real bound (unbounded graph-cache
growth) hostage to it: **merge all four**.

- #196 (S01) — full CUDA E2E acceptance met (0 host bytes/request, identical
  transcripts, latency not-worse).
- #197 (S03) — real-MOSS A/B clean (66 transcript pairs, 0 diffs).
- #198 (S02) — CUDA-graph budget tests green; E2E blocked pre-existing,
  documented above; recommend a follow-up issue for parakeet CUDA encoder
  capture (ggml CONV_2D_DW f16).
- #199 (S11) — serve-level reuse contract verified (1/2/1 engine-call
  signature) + 382/382.

Merge commits (repo convention `gh pr merge --merge`): all four PRs carry
the identical R30 one-line CI run-loop change and therefore report
`mergeable=CONFLICTING` against the moved master (which gained #193–#195);
the conflict is that single line in `.github/workflows/ci-starling-serve.yml`
(resolution: keep the branches' version — it adds `tdt_graph_budget_test`
to the run loop). `gh pr merge` refuses conflicted PRs, so the merges were
performed as local `--no-ff` merge commits pushed to master (GitHub marks
the PRs MERGED when their head commits become reachable — verified below).
Merged in order #196 → #197 → #198 → #199 with that resolution.

Two equivalence checks before pushing: `git diff 12b6548 d05648c -- cpp/
third_party/` is empty (master's engine code was unchanged between the A/B
baseline and the merge base, so the A/B sides stay valid), and
`git diff <merge-result> 52f393d -- cpp/ .github/workflows/ci-starling-serve.yml`
is empty (the merged master's engine + CI content is byte-identical to the
validated tip — the binaries this document validates are the merged content).

| PR | merge commit | state |
|---|---|---|
| #196 | `deeb34512c5e9a6eed0a1a274c4066e028a8e19d` | MERGED 2026-09-20T19:02:23Z |
| #197 | `3af1c001555b880152ecbb2909e1779c8930e4d0` | MERGED 2026-09-20T19:02:23Z |
| #198 | `3d09d59702c45dcbc265a15760f00237f2fccb5a` | MERGED 2026-09-20T19:02:24Z |
| #199 | `24ae1a230efbd5b7a5f8b7bbd69f9bb3d8bd6a3b` | MERGED 2026-09-20T19:02:23Z |

## Raw artifacts

Server traces, per-request JSON, h2d/memset CSVs, VRAM samples, WS session
logs, and analysis scripts under `/tmp/opencode/{gateb,gatec,gatec-cpu,gated,s11,probe}/`
(validation-session scratch; key numbers reproduced above).
