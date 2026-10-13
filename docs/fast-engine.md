# Fast engines (model-specialized Vulkan)

Starling ships two kinds of engines behind the same C API:

* the **ggml engines** — general, every model, every backend
  ([engine notes](ggml-engine.md));
* the **fast engines** in `cpp/fast/` — hand-written Vulkan compute kernels
  for two model families, **Parakeet-TDT** and the audio-LLMs
  (**MOSS-Transcribe**, **Qwen3-ASR**, **ARK-ASR** and **Granite-speech**),
  built around their shapes the way
  [Splash](https://inco.ai/blog/splash/) builds an engine around one model.
  MOSS and Qwen3-ASR share one engine (`audio_llm_engine.cpp`): the same
  Qwen3-Omni-style audio tower and Qwen3 decoder, with MOSS's SwiGLU adapter
  as the one structural difference.

A fast engine loads the same GGUF files. It is chosen per model load and
falls back to ggml whenever it cannot run, so enabling it never removes a
working path.

## Selecting the engine

| `STARLING_ENGINE` | Behavior |
| --- | --- |
| `auto` (default) | fast engine when the build has it, a Vulkan GPU is present and the model's tensors are supported; otherwise ggml |
| `fast` | fast engine or a load error (no silent fallback; for benchmarking) |
| `ggml` | always the ggml engine |

Build with `-DSTARLING_FAST=ON` (needs `glslc` and the Vulkan headers; the
Android build takes both from the NDK and enables the fast engine for
`arm64-v8a`). `libvulkan` is opened at run time, so a fast-enabled binary
still runs, on ggml, on machines without a Vulkan driver. CPU/software Vulkan
devices (SwiftShader, llvmpipe) are ignored.

| Variable | Effect |
| --- | --- |
| `STARLING_FAST_F16=0` | f32 products in the GEMMs (default: packed-f16 products with f32 accumulation per 32-deep slice where the device has `shaderFloat16`) |
| `STARLING_FAST_CACHE_DIR` | where the Vulkan pipeline cache and autotuning results persist (the Android app sets it to its private model directory, never the model folder it may share with other builds) |
| `STARLING_FAST_TUNE=1` | re-run the tile autotuner now (also on PowerVR and AMD, which otherwise ship measured defaults) |
| `STARLING_FAST_W4U=0/1` | W4 decode GEMVs through `unpackUnorm4x8` (default on for PowerVR and AMD) |
| `STARLING_FAST_W8U=0/1` | W8 decode GEMVs through xor 0x80 + `unpackUnorm4x8` (default on for PowerVR) |
| `STARLING_FAST_MICRO=...` | run one isolated kernel probe at load (`bits,N,K[,reps[,rows]]` GEMV, `norm,...`, `alu,...`, `idot,...`; see `Kernels::micro`) |
| `STARLING_FAST_SG=0/1` | encoder/prefill GEMMs through the subgroup-broadcast kernel `gemm_sg.comp` (default on for PowerVR where compute shaders have subgroup broadcast and ≥ 16-lane subgroups; off elsewhere) |
| `STARLING_FAST_TILE=BM,BN,TM,TN` / `STARLING_FAST_GEMV_ROWS=n` | override the tuned GEMM tile / GEMV rows |
| `STARLING_FAST_KSTEP=n` | audio-LLM decode tokens per submission (default 16) |
| `STARLING_FAST_{ARK,GRANITE}_HOST_ENCODER=1` | keep that model's audio encoder on ggml; the fast engine takes its rows (decoder only) |
| `STARLING_FAST_ATTN_SPLIT=S` | audio-LLM decode attention split into `S` KV chunks, 1–8; `0` = single pass (default 4 on AMD, 0 elsewhere) |
| `STARLING_FAST_DEVICE=n` | Vulkan physical device index |
| `STARLING_FAST_TIMING=1` / `STARLING_FAST_PROFILE=1` | per-stage wall times (load: GGUF map, engine create, plan, arena allocate/fill) plus the audio-LLM decode round times (ms per step) / per-kernel GPU times |
| `STARLING_FAST_MAPPED_WEIGHTS=0/1` | repack straight into mapped device memory (default where it is CPU-cached: UMA phones) / fill one host staging buffer per device buffer and copy |
| `STARLING_FAST_SKIP=label[,prefix*]` | diagnostic: drop every dispatch recorded under those labels (`enc_qkv`, `llm_ff_up`, `dec_qkv`, `conv_pw2`, `enc_*`, …; results are garbage) to price work by difference where GPU timestamps are unreliable (PowerVR) |
| `STARLING_FAST_VERBOSE=1` | device, weights and tuning choices at load |
| `STARLING_NPU_DIR=dir[:dir…]` | where TPU artifacts are looked up and `status-*.txt` written (default: `STARLING_FAST_CACHE_DIR`); see *Pixel TPU encoder* below |
| `STARLING_NPU=0` | never use a TPU artifact (GPU encoder) |

## Design

**Weights are repacked at load.** GGUF stays the distribution format; every
matrix is rewritten once into one of three device layouts: *W4* (4-bit, a
16-bit scale and offset per 32 weights, nibbles stored apart from scales so a
tile is a few contiguous 16-byte loads — lossless for Q4_0, and Q4_K keeps
its sub-block scales), *W8* (int8 with a 16-bit scale per 16 weights —
lossless for Q8_0 and Q6_K), and *F16*. Other types are dequantized and
requantized to W8. Rows of paired projections are interleaved so one
workgroup owns both halves of a gated unit (Conformer GLU, SwiGLU). A
format like EXL3 would buy quality per bit at 2–3 bits, but its trellis
decode costs ALU work that mobile GPUs do not have to spare; at 4 bits the
cheap affine format is the right trade.

**Loading streams the GGUF into device memory once.** The runtime loader
maps the GGUF privately instead of reading it (`ModelLoader::weight_data`),
so nothing is copied before the engine starts and pages load on first
touch, from all fill threads at once. The engines then *plan* every weight: a matrix is a
`MatSpec` — source rows plus a destination row mapping (gate/up and GLU
interleaves, q/k/v concatenation, embedding chunks) and a scale folded
into the group scales (Granite's multipliers).
Its format and size follow from the source types alone, so
`Arena::finalize` places and allocates every device buffer first and then
repacks ~1 MiB row units on all cores straight into the mapped device
memory (UMA phones; through one staging buffer at a time elsewhere), the
allocations overlapping the fill. Conversions too irregular for a row
mapping (conv weight reorders, Granite's relative-position tables) run as
deferred producers in the same pass. The host never holds a repacked copy,
and each unit drops the GGUF pages it consumed; since the mapping is
file-backed they refault intact if read again, so the ggml fallback still
finds its weights after a fast engine has loaded.

**Whole passes are recorded once and replayed.** Each forward pass is a
recorded Vulkan command buffer, cached per input shape; replaying it costs the
host nothing per operation. Recordings are split into one submission per
layer, which keeps each GPU job short (driver watchdogs, compositor
interleaving).

**Fusion.** GEMM epilogues apply bias, activation (SiLU / ReLU / exact
GELU), residual accumulation, GLU / SwiGLU pairs, and Parakeet's two
positional-bias query projections. A layer's closing LayerNorm is fused with
the next layer's first. Decode attention fuses q/k RMSNorm, RoPE, the
KV-cache append and the attention itself into one dispatch per head (on
AMD, one per head and KV chunk plus a merge). Decode RMSNorms are their own
one-workgroup dispatch: `gemv.comp` can fuse the norm, but every workgroup
then repeats the statistic and the norm-weight loads, which measured slower
than the extra dispatch once barriers were scoped (below).

**Barriers** are recorded lazily and scoped to what they order:
compute → compute unless a copy/fill sits on either side. On the Pixel's
PowerVR the wide compute+transfer form cost ~35 µs per dispatch+barrier
pair, the scoped one ~10 µs — a decode step is ~140 such pairs.

**No host round trips.** The Parakeet mel runs on the CPU (bit-identical to
the reference, 4–6× faster than the ggml CPU frontend). Everything from mel
to the joint encoder projection is one replay. MOSS writes the adapter
output (Qwen3-ASR: the projection) straight into the LLM's input rows, and
the greedy loop runs on the GPU: argmax, token history, EOS/budget test and
next-token embedding live in a device state buffer, so the host waits on one
fence per 16 tokens and the KV cache never leaves GPU memory. The budget is
per call (state slot 3): Qwen3-ASR scales it to the clip length like its
ggml engine. Qwen3-ASR's KV cache holds 1024 positions instead of the GGUF's
4096 (its C API cuts audio into ≤ 30 s pieces), 117 MiB instead of 470 MiB of
unified memory on a phone.

**ARK and Granite.** ARK's Whisper encoder (conv stem, 32 RoPE
global-attention layers, ln_post, merge-by-4 GELU adapter) runs on the
device too: the host uploads the log-mel; the two k3 Conv1d layers are
plain GEMMs over time-major rows padded by one zero row (output row t reads
the contiguous run t·s .. t·s+2), the interleaved partial RoPE is
`enc_rope.comp`, global attention runs in head groups under a 64 MiB score
budget, and the adapter reads merge·d_model-wide rows straight from the
ln_post output and writes into the prompt's audio slots.
`STARLING_FAST_ARK_HOST_ENCODER=1` keeps the encoder on ggml. Granite's
conformer (200-frame attention windows with a Shaw relative-position bias,
`shaw_bias.comp`; the depthwise conv reuses Parakeet's kernel with BatchNorm
folded at load) and its windowed Q-Former run on the device the same way,
through the shared `generate_from_mel()`; the projector writes the audio
slots with the embedding multiplier applied.
`STARLING_FAST_GRANITE_HOST_ENCODER=1` keeps both on ggml, and the engine
then takes their output rows. Granite's CTC speculative decode needs the
ggml encoder's weights and runs only under `STARLING_ENGINE=ggml`. The trunk variations are spec flags of the
Qwen3 decoder — Qwen2.5's q/k/v biases ride the projection epilogues and
the attention kernels skip the q/k norm; Granite's residual multiplier is
folded into the o/down projections' group scales, its embedding multiplier
into the (untied) input table, its attention multiplier is the softmax
scale and its logit divisor is dropped (it cannot change a greedy argmax).
ARK 0.6B's suppressed ids (`bad_words_ids`) are a −3e38 lm_head bias. The
host pages of the loader tensors the engine now owns (`llm.*`, and ARK's
`enc.*`/`adapter.*`) are returned to the OS (`release_host_weights`).

**Parakeet's transducer decoder stays on the CPU** — sequential, a few small
matrix-vector products per step — with int8 dot-product kernels (ARMv8.2
SDOT / x86 AVX2, chosen at run time). The prediction network runs only after
an emission, and its first-layer input projection is memoized per token.

**Autotuning.** Tile sizes that suit one GPU starve another, so with a cache
directory set the engine times candidate GEMM tiles and GEMV row counts on
synthetic weights at the real shapes the first time it meets a device, and
stores the choice.

**Pixel TPU encoder (Qwen3-ASR, `cpp/npu`).** On Tensor phones Qwen3-ASR's
audio encoder can run on the TPU instead of the GPU. Nothing from Google's
Tensor SDK is involved: the phone's own firmware library
`libedgetpu_litert.so` (a public vendor library, opened with `dlopen` at run
time; the app declares it with `<uses-native-library android:required="false">`)
both compiles a `.tflite` on the device and runs the result. Compiling only
works from the adb shell (the firmware's compile service serves just the apps
on Google's EdgeTPU allowlist), so the user compiles on their own phone over
adb and nothing precompiled ships: load the Qwen3-ASR model in the app once,
then run `scripts/npu/npu_compile_all.sh`. It finds the phone
(`ANDROID_SERIAL`, the one `adb devices` entry, or wireless debugging via
mDNS; `--wait S`), the installed Starling packages (release, `.debug`,
`.experimental`), reads what each loaded model reported (below), and compiles
every missing or stale Qwen3-ASR encoder with `npu_compile.sh MODEL.gguf
[--hash 0x…] [--app PKG,…]`: export the graph from the GGUF the app loads
(`export_encoder.py`), compile it with the phone's compiler (only the
Qwen3-ASR 0.6B/1.7B encoder window; the phone-side helper checks the graph's
key, its inputs and output and its operator signature — operator and tensor
counts and operators per builtin, as `export_encoder.py` builds that window —
itself), install it into `files/npu` of a debuggable package (run-as) or the
external files dir `Android/data/<pkg>/files/npu` of any other. The artifact
is one encoder attention window (800 mel frames → 104 projector rows, an
additive key mask for the partial last window),
`qwen3-asr-enc-<model hash>.stnpu`. The engine loads an artifact only if its
header matches the GGUF, the graph shape and the firmware's API version; a
worker thread hashes the GGUF's encoder tensors, then opens and warms the
artifact up while the GPU weights load; the engine then runs the
encoder window by window on the TPU, uploads the projector rows into the
LLM's audio rows and records a prefill without the GPU encoder pass. Short
clip (7.4 s, Qwen3-ASR-0.6B, after the subgroup-broadcast GEMM): encoder +
prefill 558 → 396 ms, wall 1576 → 1422 ms (TPU encoder 54 ms).

*The GGUF's own weights.* The graph is built from the GGUF file the app
loads and nothing else: its metadata gives the shapes, its `enc.*` and
`proj.*` tensors — dequantized from whatever they are stored as (Q4_0, Q8_0,
K-quants, BF16, F16) — the weights, stored as fp16 in the `.tflite` (1.7B:
f32, see the size caps below; the TPU computes in fp16 either way). So the
TPU encoder runs the GPU path's encoder weights, quantization included, and
its accuracy is the GPU path's. (The first version exported the HF
checkpoint's fp16 weights instead: on FLEURS-en, 100 clips, Qwen3-ASR-0.6B
scored 5.23 % WER with the TPU encoder against 5.85 % on the GPU. That small
gain is gone with the GGUF's weights.) On the host the export matches the
GPU engine: on `short.wav` the fast engine's (RADV) projector rows are within
rel-L2 3.6e-3 (0.6B q4e8 and 1.7B q4e8-imx) of the f32 torch graph over the
dequantized weights, the exported `.tflite` (LiteRT on the CPU) within
2.9e-4 (0.6B, fp16 weights) and 2.7e-6 (1.7B, f32 weights), and the LLM
turns the `.tflite`'s rows into the GPU path's transcripts (`short.wav`,
`medium.wav`).

*The model hash* (`cpp/npu/encoder_contract.hpp`) binds an artifact to one
GGUF: a digest of every audio-encoder and projector tensor (`enc.*`,
`proj.*`) exactly as the GGUF stores it — name, type, shape, and an XXH64 of
the raw bytes — and of the encoder metadata the graph is built from (every
`qwen3.enc.*` and `qwen3.proj.*` key: shapes, window sizes, LayerNorm
epsilon, as the engine's loader reads them), folded with FNV-1a in name
order. So a GGUF whose encoder differs in any byte or hyperparameter, a
weights-only fine-tune or another quantization included, does not match an
artifact compiled for another. The export computes it from the file it read
(`export_encoder.py --model-hash GGUF` prints it; plain Python, faster with
the `xxhash` package) and the artifact is named by it, so an artifact
carries the hash of the encoder it was built from; the engine computes the
same digest from the GGUF it loads (the Python and C++ sides are pinned to
the same test vectors, `tests/test_npu_fingerprint.py`,
`cpp/tests/npu_fingerprint_test.cpp`). The contract key's
version (`qwen3-asr-enc/2`) moves whenever what an artifact computes for a
GGUF can change, so artifacts from an older exporter report `stale`. The
engine computes it only where a TPU can be used (a Qwen3-ASR model, an
artifact directory, firmware that lists the TPU library), on that worker,
and the load waits for it only if the GPU weights are in place first;
elsewhere nothing is hashed. On the host it takes 26 ms (0.6B q4e8, 112 MB
of encoder tensors) and 43 ms (1.7B, 186 MB) with the GGUF in the page
cache, 97 and 168 ms reading it cold. `starling-npu-tool fingerprint GGUF`
prints the same value on the phone.

*Which GGUF.* The host needs the very file the app loads, and
`npu_compile_all.sh` finds it by the model hash in the status file:
`--gguf HOST_GGUF` if its hash is one the app asked for, else the first GGUF
with that hash under `--models DIR` (default `$STARLING_MODELS_DIR`, else the
repository's `models/`, else the main checkout's when run from a git
worktree; hashes are cached by path, size and mtime in
`~/.cache/starling-npu`), else the file the status names, pulled from the
phone (run-as for a debuggable package). Otherwise it refuses that model and
asks for `--gguf`. With `--hash`, `npu_compile.sh` refuses a GGUF whose hash
is another.

*Fallback, said out loud.* Without a matching artifact, with `STARLING_NPU=0`,
or after any TPU error the engine runs the GPU encoder as before. Each piece's
state (`ok`, `missing`, `stale` — compiled for older firmware or by an older
exporter — or `error`), with the model hash and the GGUF file it loaded,
goes to `status-<piece>-<model hash>.txt` in every existing
`STARLING_NPU_DIR` directory (what `npu_compile_all.sh` reads;
`missing` only where the firmware lists the TPU library, and without loading
it), stale and error states are logged once per process (logcat tag
`StarlingNpu`: "TPU … not used (stale), the GPU runs it: … — recompile via
adb"), and `describe()` carries the stale note.

*Compiler limits, and why only Qwen3-ASR* (firmware 0.21.0, Pixel 10 Pro).
The on-device compiler accepts graphs whose tensors stay window-sized; the
other audio encoders were exported (exact against HF on the host) but do not
compile or run correctly:
* a batched matmul with more than 256 keys fails ("Reshape input/output slice
  mismatch"), so 30 s of global (ARK-ASR) or long sliding-window attention
  does not compile as is;
* slicing an activation fails — the last dimension of a 64-wide head (RoPE's
  rotate-half; a constant `[hd, hd]` matmul works instead), K/V or GLU halves
  ("Cannot remove elements due to shape mismatch"), blocks along the sequence;
* whole-sequence tensors (1500–1600 rows × 1024–1280) overflow on-chip memory
  ("TENSOR_MOVEMENT layer … overflows memory", "Model is too large to fit in
  on-chip memory") or exceed the compiler's 60 s overflow budget, which
  rules out blocking a 30 s encoder into ≤ 256-row pieces (ARK, Granite);
* graphs of ~0.7 GB of weights exhaust the compile service's memory
  (`DEAD_OBJECT`, Android killing background apps). `FAILED_TRANSACTION` /
  "Binder buffer full" is only the compiler's oversized error text: the real
  error is in logcat (`edgetpu_vendor_service`).

MOSS-Transcribe's encoder (the Qwen3-Omni audio tower, at 32 × 1280, in
8-layer segments per window) compiles and runs (~110 ms per 8 s window; the GPU
encoder took ~1.1 s on that clip before the subgroup GEMM), but its last
segment returns wrong rows (rel-L2 0.98: ln_post and the projection are
exact, the gated-MLP adapter is not — its SwiGLU products are ~1e-4, half
below fp16's normal range, and scaling them by 1024 did not help), so it
stays on the GPU too.

**Compiling can reboot the phone.** Twice the compiler took the whole phone
down — `kernel_panic,system_is_deadloomld_on_oomory`, with 11.7 GB available
before the compile: once on a 4 MB probe graph (four 256-row slices of a
`[1, 1024, 1024]` input, each through a linear layer, concatenated:
"pslice-1024"), once on MOSS's adapter alone (`[1, 104, 2048]` → 8192 →
2048, 100 MB). Neither size nor free memory predicts it. The compile scripts
therefore build only the graph with a clean record (the Qwen3-ASR encoder),
with fixed shapes, refuse graphs larger than the window has been
(640 MB for 0.6B, 1300 MB for 1.7B), and refuse to compile with less than 4 GB available
(`NPU_MIN_MEM_MB`); save your work on the phone before running them.

## Kernels

| Shader | Role |
| --- | --- |
| `gemm.comp` | tiled GEMM, B = W4/W8/F16/F16ᵀ, f16-packed shared tiles 32 or 16 deep, batched heads/windows, causal tile skipping, implicit-GEMM 3×3/s2 convs (MOSS), fused epilogues |
| `gemm_sg.comp` | the same GEMMs (all but F16ᵀ) without shared memory, for PowerVR: one row of A per lane, 16 columns per subgroup-sized workgroup, weights dequantized by 16 lanes and shared with `subgroupBroadcast`, A through texel-buffer views (row-blocked under `maxTexelBufferElements`), same epilogues |
| `gemv.comp` | decode matrix-vector, x kept in registers per K slice, optional fused RMSNorm, residual / SwiGLU / argmax-partial epilogues |
| `attn_decode.comp` | fused qk-norm + RoPE + KV append + attention (one query head per workgroup; 16-byte key loads, value mix in four position groups) |
| `attn_decode_split.comp`, `attn_split_combine.comp` | the same, split over KV chunks (one workgroup per head and chunk), then a per-head merge of the partial softmaxes |
| `rope_kv.comp` | prefill q/k norm + RoPE + KV-cache write |
| `enc_rope.comp` | encoder RoPE (GPT-J pairs over the first rope_dim of each head; ARK), in place on q and k |
| `softmax.comp` | attention softmax with Parakeet's relative-position skew folded into the read index, causal and key-length masks |
| `norm.comp` | LayerNorm / RMSNorm, including the fused layer-boundary pair |
| `pk_conv.comp` | subsampling convolutions, Conformer depthwise conv + batch-norm + SiLU, MOSS stage-1 conv |
| `embed_rows.comp`, `dec_next.comp` | embedding lookup; on-device greedy step (argmax, EOS, next embedding) |

Shaders use only Vulkan 1.1 core features (16-bit values are packed in 32-bit
words); the packed-f16 variants additionally need `shaderFloat16`, and
`gemm_sg.comp` the subgroup ballot (broadcast) operations in compute shaders.

## Validation

* `fast_weights_test` (CPU-only, runs in CI): every repacked format against
  ggml's dequantizers, and the CPU GEMV against an f64 reference.
* `benchmarks/fast_engine/parity_parakeet.py`: ggml vs fast encoder output,
  token streams and transcripts on the same library build.
* `benchmarks/fast_engine/wer_engines.py` with `export_fleurs.py`: corpus WER
  and latency for several engine variants on the same clips.
* `benchmarks/fast_engine/android_bench.sh`: cross-builds `starling-bench`,
  pushes it to a phone with the models and compares engines there.
* `benchmarks/fast_engine/pixel_measure.sh`: the tuning campaign's phone
  metric (Parakeet medium + MOSS short) with a fixture-transcript gate;
  `phone_ab.sh` interleaves several binaries in one thermal window.
* `benchmarks/fast_engine/phone_gates.sh`: A/B of the current Android build
  against a sha256-pinned base binary (alternating rounds, screen off) with
  the G1 transcript gate and the optional Parakeet canary; prints `METRIC`
  lines. `phone_energy.sh` measures energy per MOSS-short transcription
  from the battery charge counter with a duration-scaled idle control (an
  estimate, not a rail measurement).

The fast engines are **not** byte-exact with ggml: activations enter the
matrix products as f16 (ggml's CPU path quantizes them to 8 bits), so
near-tie tokens occasionally differ. They are held to corpus quality, not to
the ggml engines' exact-text gates.

## Measured results

Development machine: AMD Ryzen 5 PRO 5650U with its Radeon (Vega 7, RADV)
iGPU. **These are not Pixel numbers** — measure the phone with
`benchmarks/fast_engine/android_bench.sh`, where the first load also runs the
autotuner. Wall time per transcription after one warm-up, same build and
model files.

| Model / audio | ggml CPU | ggml Vulkan | fast (Vulkan, f16) |
| --- | --- | --- | --- |
| Parakeet q4_k_m-shrink16, 7.4 s | 485 ms | — | 230 ms |
| Parakeet, 22.3 s | — | 640 ms | 522 ms |
| Parakeet, 74.4 s | — | 2 331 ms | 1 743 ms |
| MOSS q4e8, 7.4 s | 3 129 ms | — | 1 820 ms |
| MOSS, 22.3 s | 9 596 ms | — | 5 150 ms |

Encoder-only (Parakeet, 22.3 s): 345 ms fast vs 535 ms ggml Vulkan. MOSS
decode runs at ~33 ms/token, close to this iGPU's memory-bandwidth floor
(~1.1 GB of weights per token, 28 % of it the Q8 tied lm_head).

FLEURS en_us test, first 100 clips (`wer_engines.py`):

| Model | ggml CPU WER | fast f32 WER | fast f16 WER | time ggml / fast f16 |
| --- | --- | --- | --- | --- |
| Parakeet q4_k_m-shrink16 | 5.47 % | 5.33 % | 5.33 % | 72 s / 25 s |
| MOSS q4e8 | 7.92 % | 7.87 % | 7.92 % | 441 s / 186 s |

10 (Parakeet) and 16 (MOSS) of 100 transcripts differ from ggml, all in
near-tie words and punctuation, in both directions.

### Pixel 10 Pro (Tensor G5, PowerVR DXT-48-1536)

Tuned on-device (see `benchmarks/fast_engine/RESEARCH_LOG.md` for the
experiment log and `AUTORESEARCH.md` for the method). Vendor-keyed defaults
for PowerVR: GEMM tile 32,128,4,8 (a full 128-thread subgroup with BN = 128)
with 16-deep k-tiles (RESEARCH_LOG P5), f32 GEMM products, GEMV rows 16 pinned (measured best-or-tied at every MOSS
decode shape — rows=8 cost 15–31 % on the small-N GEMVs whose per-workgroup
x re-read rivals the weights; RESEARCH_LOG P1-8) with RSPLIT row slots
padding workgroups to a whole subgroup, raised to at least 16 rows per row
slot while 64 workgroups remain (rows 64 at K = 1024, 32 at K = 2048;
RESEARCH_LOG P4-2); the CPU transducer decoder splits its
GEMV rows across a second thread (held spinning for one transcription,
parked in between); W4 decode GEMVs unpack nibbles with `unpackUnorm4x8`,
W8 ones bytes with xor 0x80 + `unpackUnorm4x8`.
ggml runs with the app's 6 threads. Transcripts match ggml on the fixtures;
FLEURS en_us 100: Parakeet 5.33 % (ggml 5.47 %), MOSS q4e8 7.87 %
(ggml 7.92 %).

Two further PowerVR notes from the layout research (#317, same log):
`gemv_w4um` processes two tokens' GEMVs in one weight pass (1.76–2.11× the
per-token rate; the verify primitive for #311's speculative decoding — not
dispatched by the engine yet), and the GPU driver on the Pixel 10 Pro can
enter a degraded state under sustained load where fences time out (#325):
the engine marks itself wedged, leaves a 15-minute marker file
(`starling-fast-gpu-wedged`, in `STARLING_FAST_CACHE_DIR`) so fresh
processes refuse with a restart hint instead of joining a retry storm, and
checks `VK_EXT_memory_budget` before the big allocations where the driver
exposes it.

The #317 follow-up loop (2026-10-01, RESEARCH_LOG P2-*) landed one more
simplification — decode records one command buffer per round instead of
per token (neutral, 16× fewer allocations) — and pinned down how strongly
the decode number depends on the measurement protocol: cool fresh boots
with an awake-hold measure **62.5–64 ms/token**; the historical 68–77
band spans boots without the hold; and a locked phone without any hold
reads 105–141 ms/token because system suspend stalls each decode
round-trip by ~0.7 s, and suspend with GPU work outstanding wedges the
driver. The app holds a partial wake lock around on-device GPU work; the
bench scripts hold a shell-uid one, which Doze does not disable, for their
screen-off windows (phone measurement protocol in
`benchmarks/fast_engine/AUTORESEARCH.md`). MOSS short wall ≈ 5.25 s
median in a 12-run window; Parakeet medium 2.27–2.34 s (in band).

| Model / audio | ggml CPU | fast (start of tuning) | fast (tuned) |
| --- | --- | --- | --- |
| Parakeet, 22.3 s | 7.2 s (enc 1.30 s, dec 5.9 s) | 2.79 s (enc 2.56 s, dec 0.20 s) | **2.22 s** (enc 1.98 s, dec 0.14–0.20 s) |
| Parakeet, 74.4 s | — | 9.0 s | **7.6 s** |
| MOSS, 7.4 s | 7.3 s | 7.0 s (decode 100 ms/tok) | **5.6 s** (mel 0.13 s, enc+prefill 2.95 s, decode 76 ms/tok) |

**Audio-LLM decode (2026-10-12, RESEARCH_LOG P4).** Where a step went,
priced by ablation (each kernel class dropped from the recording) on Qwen3-ASR-0.6B (39.8
ms/step): GEMVs ~33 ms (the K = 1024 ones with the fused RMSNorm at only
15–17 G weights/s), attention 5.9 ms, and ~4 ms in 141 dispatch+barrier
pairs at ~30 µs each. Landed, each an interleaved A/B with identical
transcripts: 16-byte key loads and a value mix in four position groups in
the attention; a rows-per-row-slot floor for the GEMVs; bit-exact trims of
the GEMV reduction tail; barriers scoped to compute → compute (the wide
compute+transfer form was most of the ~30 µs; now ~10 µs per pair);
RMSNorm as its own dispatch instead of fused into every GEMV workgroup;
the W8 unpack (Granite, whose Q5_K linears are W8). Measured
not to pay: int8 activations with `OpSDot` (+17 % on 0.6B, +5 % on MOSS
despite neutral FLEURS WER — on this GPU the packed dot costs the issue
slots of the f32 dot it replaces), the flash-decode split, 2–4× threads
per attention head, 128-row lm_head workgroups, KSTEP 32.

| Model (decode ms/token, short / medium) | before | after | wall short / medium after |
| --- | --- | --- | --- |
| Qwen3-ASR-0.6B q4e8 | 41.9 / 48.4 | **24.3 / 26.2** | 2.15 / 5.78 s (was 2.74 / 7.92) |
| Qwen3-ASR-1.7B q4e8-imx | 77.6 / 83.6 | **62.3 / 64.5** | 4.84 / 12.78 s (was 5.35 / 14.72) |
| MOSS q4e8 | 71.1 / 81.0 | **60.6 / 64.1** | 5.11 / 13.14 s (was 5.45 / 14.66) |
| ARK-0.6B q4k | 45.1 / 49.4 | **32.2 / 33.8** | 4.56 / 13.56 s (was 4.99 / 15.10) |
| Granite dynq4 | 92.2 / 101.5 | **74.6 / 78.8** | 6.24 / 15.13 s (was 6.80 / 16.63) |

Before = `t3/optimize-phone-model-serving` 84f53c03, after = this branch;
interleaved in one session per model (screen off, wake hold), medians of
4. Transcripts identical before/after on the phone and the
host. Every GEMV now streams 21–30 G weights/s; what remains is the GEMV
plateau below, not dispatch overhead. The ggml CPU engine decodes
Qwen3-0.6B at ~22 ms/token on 6 threads.

**Encoder and prefill (2026-10-12, RESEARCH_LOG P5).** Priced by
`STARLING_FAST_SKIP` ablation on Qwen3-ASR-0.6B, enc+prefill is GEMMs
end to end: the LLM linears ~50 %, the encoder linears ~20 %, the two
3×3 conv GEMMs ~17 %, attention ~12 %, norms ~0; on Parakeet the feed
forwards are 58 %. Every one of them ran at ~200 GFLOPS whatever the
tile shape, kernel variant or operand kind, against a measured f32 peak
of ~1.4 TFLOPS (`alu` probe, ≥ 280 workgroups). The probes located the
limit: the register-tiled product loop itself, with its operands read
from shared memory as the GEMM does, caps at ~400 GFLOPS (packed f16
operands; ~300 with f32 words, ~800–1000 with the same FMAs on register
operands), and the tile loads and barriers take the
rest. What paid is occupancy: 16-deep k-tiles halve the shared tiles and
each thread's tile-load registers, +20–40 % per GEMM isolated (BK = 8
pays less, more barriers), and the causal prefill attention now skips
the score tiles above the diagonal and the P·V products past it. Both
are bit-exact: transcripts identical before and after on the phone and
the host for all six models.

| Model (enc+prefill short / medium; wall) | before | after | wall short / medium after |
| --- | --- | --- | --- |
| Qwen3-ASR-0.6B q4e8 | 1.14 / 2.86 s | **0.98 / 2.39 s** | 1.99 / 5.27 s (was 2.16 / 5.75) |
| Qwen3-ASR-1.7B q4e8-imx | 2.40 / 5.95 s | **2.30 / 5.55 s** | 4.66 / 12.31 s (was 4.79 / 12.60) |
| MOSS q4e8 | 3.01 / 7.03 s | **2.74 / 6.34 s** | 4.83 / 12.33 s (was 5.09 / 13.01) |
| ARK-0.6B q4k | 3.39 / 9.95 s | **2.84 / 8.67 s** | 3.86 / 11.83 s (was 4.44 / 13.13) |
| Granite dynq4 | 3.73 / 9.62 s | **3.27 / 7.86 s** | 5.79 / 13.24 s (was 6.25 / 15.04) |
| Parakeet (encoder) | 0.74 / 2.00 s | **0.63 / 1.62 s** | 0.80 / 1.86 s (was 0.88 / 2.22) |

Before = `t3/optimize-phone-model-serving` 46cb1a38; interleaved, two
rounds with the order reversed in the second, screen off, wake hold,
medians of 4. RADV unchanged (BK stays
32 there; ±3 %). Measured not to pay on this GPU: f16 products (packed
FMAs −30 %, `dot(f16vec4)` with k-major tiles ±), f32 shared tiles
(−30 %), register prefetch of the next tile (−15 %), a broadcast kernel
that keeps whole weight columns in registers (± 10 %), cheaper unorm
dequant in the tile loads (±), larger or smaller register tiles, 256-
thread workgroups, and hoisting the conv im2col divisions (+1 %).

**Subgroup-broadcast GEMM (2026-10-13, RESEARCH_LOG P6).** LiteRT's GPU
accelerator ran the Qwen3-ASR encoder 3.4× faster than P5's GEMMs, and
its kernels showed why: no shared memory at all. `gemm_sg.comp` now takes
every encoder/prefill GEMM on PowerVR — W4 / W8 / F16 weights, the 3×3
conv GEMMs, the attention scores; P·V stays on gemm.comp — with one row
of A per lane, 16 output columns per
subgroup-sized workgroup, each 8-deep slice of the 16 columns' weights
loaded and dequantized to f32 by 16 lanes and handed to all 128 with
`subgroupBroadcast`, f32 dot products, and A read through texel-buffer
views (the texture path; row-blocked under the driver's 65536-texel
limit, the conv by input window). Per class on Qwen3-0.6B short: LLM
linears 489 → 246 ms, encoder linears 195 → 106, conv 137 → 69,
attention 33 → 28. Not bit-exact with gemm.comp, but more exact against
an f64 reference (W4/W8 weights stay f32 after dequantization) and
transcript-equal: identical transcripts on the phone for all six
models on short and medium, and identical FLEURS en_us WER on 100 clips
per model with 0 of 100 transcripts differing. Decode is unchanged; on
RADV gemm.comp's packed-f16 tiles stay 2–2.5× faster, so the kernel is
PowerVR-only (`STARLING_FAST_SG`).

| Model (enc+prefill short / medium) | gemm.comp | gemm_sg.comp | wall short / medium |
| --- | --- | --- | --- |
| Qwen3-ASR-0.6B q4e8 | 0.99 / 2.41 s | **0.57 / 1.54 s** | 1.64 / 4.54 s (was 2.03 / 5.40) |
| Qwen3-ASR-1.7B q4e8-imx | 2.32 / 5.26 s | **1.22 / 3.58 s** | 3.62 / 10.46 s (was 4.70 / 12.04) |
| MOSS q4e8 | 2.76 / 6.30 s | **1.49 / 4.80 s** | 3.66 / 11.33 s (was 4.87 / 12.35) |
| ARK-0.6B q4k | 2.88 / 9.00 s | **1.82 / 7.15 s** | 2.75 / 10.49 s (was 3.81 / 11.98) |
| Granite dynq4 | 3.28 / 8.23 s | **2.05 / 5.73 s** | 4.51 / 11.83 s (was 5.65 / 13.42) |
| Parakeet (encoder) | 0.71 / 1.98 s | **0.43 / 1.51 s** | 0.65 / 1.87 s (was 0.93 / 2.38) |

One binary, `STARLING_FAST_SG` 0 / 1 interleaved (rounds 0 1 1 0, screen
off, wake hold), medians of 6 per arm. The long session heated the
phone and the big models' decode drifted up in the middle rounds, which
understates the wall gains; with a cool-down before every invocation
decode measured equal.
Measured not to pay: f16 products (no 2× f16 rate through Vulkan on this
driver — LiteRT's fp16 margin is an OpenCL-compiler effect), vec4
accumulators, two rows per lane, 8 or 32 columns per workgroup. The
next lever is a ML-Drift-style pre-packed weight layout (~15 % more in
the probe), which needs a second resident layout next to decode's.

**Model load (2026-10-12).** Load used to read the whole GGUF into
anonymous memory, repack every matrix into host vectors, allocate the
device buffers, copy, and free the host vectors — from eight threads at
once, which on Android serialized on `munmap` and turned a 40 ms copy of
Parakeet's 585 MiB into 0.9 s. Granite peaked at 4.6 GB resident. Now the
GGUF is mapped, every weight is planned first and repacked straight into
the mapped device memory (see *Design*), the device allocations overlap
the fill, the KV caches are created alongside it, and the BPE encoder
tables (only s1 encodes) are built on first use. `starling-bench` prints
the load wall time with peak RSS (VmHWM) and the largest system
MemAvailable drop during the load; `--cold` evicts the GGUF from the page
cache first (`POSIX_FADV_DONTNEED`; no root needed).

| Model (GPU weights) | warm load before → after | cold load before → after | peak RSS before → after | MemAvailable drop before → after |
| --- | --- | --- | --- | --- |
| Parakeet q4_k_m-shrink16 (585 MiB) | 2.04 s → **0.33–0.35 s** | 2.17 s → **0.58–0.73 s** | 1182 → **111–113 MiB** | 1.7 → 0.6 GB |
| Qwen3-ASR-0.6B q4e8 (557 MiB) | 2.18 s → **0.42–0.76 s** | 1.85 s → **0.71–0.75 s** | 1319 → **147–151 MiB** | 1.6 → 0.7 GB |
| ARK-0.6B q4k (1037 MiB) | 3.75 s → **0.73–0.85 s** | 3.05 s → **1.40–2.22 s** | 2145 → **130–136 MiB** | 2.9 → 1.1 GB |
| MOSS q4e8 (1612 MiB) | 7.39 s → **0.78–0.88 s** | 5.44 s → **2.20–2.54 s** | 3508 → **186–190 MiB** | 4.8 → 1.9 GB |
| Granite dynq4 (2546 MiB) | 11.1 s → **1.38–1.60 s** | 12.6 s → **2.63–2.72 s** | 4615 → **321–324 MiB** | 6.6 → 2.7 GB |

Pixel 10 Pro, screen off with the wake hold, `starling-bench` load line
(the whole `starling_ggml_load`: GGUF, tokenizer, engine). Before = the
first of two interleaved rounds (base, then this branch, cold then warm
per binary; the second, warmer round was 20–60 % slower for both); after
= the range over two sweeps of the final code (cold-load spread is mostly
storage). Transcripts of
short and medium are identical before and after on all five models, on
the phone and on the RADV host. Cold = the GGUF evicted from the page
cache first: storage reads ~1 GB/s here, so a cold load is now mostly
the read itself. The MemAvailable drop is dominated by the device
allocation, which the weights need anyway. On the host (RADV, staged
fill): Granite 5.7 → 2.7 s and 5.1 GB → 0.7 GB peak.

Where a warm Granite load goes now: GGUF header parse 0.11 s, Vulkan
device 0.08 s, two threads allocating 2.5 GB (the driver zeroes it) for
0.5 s overlapped with the fill, which ends at 1.2 s; most of the fill's
CPU time past the allocation is the Q5_K → W8 requantization. Two
allocator threads instead of one: Granite ~0.4 s faster.

Pipeline creation is not part of the load: pipelines are built while the
first pass is recorded. With the pipeline cache in `STARLING_FAST_CACHE_DIR`
populated, the first transcription costs ~0.3 s extra on Parakeet and
nothing measurable on Granite; with an empty cache directory (first
launch, or a cleared app cache) it costs ~7 s on Parakeet and ~6.6 s on
Granite — worth keeping the cache directory out of anything the OS
trims. The ggml fallback is unaffected by the fast engine having
consumed (and dropped) the mapped GGUF pages: with the fast engine forced
to fail after a full load, `STARLING_ENGINE=auto` gave the ggml
transcripts on the phone (Parakeet, Qwen3) and on the host (all six
models).

Measured and not kept: a CPU-cached (non-coherent) memory type for the
weights (no gain once the copy was gone; the coherent write-combined
type takes plain sequential stores at ~13 GB/s); `MADV_POPULATE_WRITE`
on the mapped buffers (the driver's mapping refuses it, and its pages are
already present); staging uploads on PowerVR (585 MiB took 9.8 s — the
mapped path stays the default there); a background readahead of the
GGUF (`posix_fadvise` WILLNEED from a thread: on cold Pixel loads it
never beat the fill's own parallel page faults and often queued the
header parse behind it — Qwen3 0.94–1.0 s without, 1.0–1.7 s with;
Granite 1.95–2.16 s vs 2.26–2.86 s — and over a cached file walking its
pages contends with the faults: host warm Qwen3 0.82 → 0.52 s without);
an inlined f32→f16 conversion (5.0 → 3.3 ns per value, too small
to matter); lazily reading GGUF string arrays (2 ms). A repacked-weight
cache on disk was not built: on a warm Granite load it could save at
most the ~0.7 s the fill runs past the allocation (the fill would become
a 2.5 GB copy from the page cache), a cold load would read 2.5 GB instead
of 1.86 GB at ~1 GB/s (slower), and it would cost 0.6–3.8 GB of storage
per model.

Energy per transcription (batterystats power model, 40/20-run averages,
measured before the W4 unpack change, i.e. MOSS at 88.7 ms/token):
Parakeet medium fast ≈ 1.33 mWh vs ggml ≈ 5.0 mWh (3.8×), MOSS short fast
≈ 2.05 mWh vs ggml ≈ 5.2 mWh (2.5×).

The desktop-tuned GPU kernels now reach ~180 GFLOPS on the encoder GEMMs
(peak f32 ≈ 500 GFLOPS); the remaining gap is shared-memory traffic, not
barriers or tail waves. Decode GEMVs, in contrast, are instruction-issue
bound on this GPU, not bandwidth bound: W4 and W8 both stream ~33 G
weights/s, so saving ALU per weight (the `unpackUnorm4x8` path, −13 % MOSS
decode) pays and saving bytes alone does not. `VK_KHR_cooperative_matrix`
is advertised (f16 and int8 shapes) but the driver's shader compiler faults
on any coopmat instruction; an unfinished coopmat GEMM was removed (commit
b1182f2 has it). Integer dot products (`OpSDot`) do work
(`STARLING_FAST_MICRO=idot`, needs a glslc with `GL_EXT_integer_dot_product`;
the NDK's does not have it), but int8 activations through them measured
slower than the f32 GEMVs on both the W8 lm_head (P1-1) and the W4 linears
(P4-3): the packed dot takes the issue slots of the `dot(vec4)` it
replaces. The q4e4 model file saves a further ~7 % decode time at +0.14 % WER (8.06 vs 7.92, inside the
0.2-pt gate) if maximum battery life matters more than the last word error.
