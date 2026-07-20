# Starling fused-op spec for the ggml (parakeet.cpp) port

Purpose: extract the EXACT math, dtypes, layouts, and constants of every fused
kernel Starling uses, so byte-exact ggml custom CUDA kernels can be written for
parakeet.cpp. Every claim cites a `file:line` in the Starling source.

Repository paths below are relative to `/home/m0hawk/Documents/starling/`.

---

## 0. CRITICAL finding (read this first)

The premise in the task brief is **partially incorrect** and it changes the
porting strategy:

1. **Starling does NOT have hand-written Triton/CUDA kernels for the parakeet
   Conformer encoder.** There is no `src/starling/parakeet/_kernels/` directory.
   The only file under `src/starling/parakeet/` that touches `_kernels` is
   `encoder_graph.py:48`, and it imports only `torch_compile` (a no-op wrapper
   when triton is absent) — see `src/starling/_kernels/_compile.py:28`.
   Confirmed by grep: the `src/starling/parakeet/` tree has ZERO references to
   `fp8`, `OptFlags`, `residual_multiplier`, `fused_rmsnorm`, or `quantize`.

2. **All hand-written fused kernels in `src/starling/_kernels/` are LLM-decoder
   ops.** They are consumed only by `granite/llm_mega.py`, `qwen3/llm_mega.py`,
   `moss/fused_decode.py`, `higgs/fused_decode.py`, `ark/llm_mega.py`,
   `audex/llm_mega.py`, `nar/fused_llm.py` (see grep in task notes). The parakeet
   Conformer encoder and the parakeet TDT decoder never call them.

3. **Parakeet's speed comes from CUDAGraph capture + `torch.compile`, not from
   kernel fusion.** The encoder (`src/starling/parakeet/encoder_graph.py`) runs
   the stock HF `model.get_audio_features(...)` (nvidia/parakeet-tdt-0.6b-v3's
   native 24-layer Conformer) and either (a) captures it into ONE
   `torch.cuda.CUDAGraph` (`GraphedEncoder`, byte-exact, default), or (b) wraps it
   in `torch.compile(mode="reduce-overhead")` + a BatchNorm fold
   (`CompiledEncoder`, NOT byte-exact). See `encoder_graph.py:213` (graph capture),
   `encoder_graph.py:320-432` (compiled path).

   This means the parakeet Conformer is fusing its elementwise/memop glue via
   **PyTorch Inductor** (auto-generated Triton), NOT via the Starling hand-written
   kernels. Starling's hand-written kernels give no recipe for *what* a fused
   Conformer block should compute — Inductor decides that at trace time.

4. **parakeet-tdt is bf16 throughout. No fp8, no fp4.** `fp8_weights`,
   `nvfp4_weights`, `fp8_attention` are all `OptFlags` defaults = `False`
   (`src/starling/flags.py:123,144,94`), and parakeet never sets them. The fp8/fp4
   GEMV kernels are reachable only by the granite/moss LLM-decoder path. **For the
   parakeet Conformer port, fp8/fp4 is OUT OF SCOPE** (flagged per-op below).

Practical consequence for parakeet.cpp: the reason Starling's encoder leaves the
GPU busy while ggml's leaves it 93% idle is **launch overhead, not kernel
fusion** — Starling collapses ~hundreds of per-layer launches into one
`graph.replay()` (or one Inductor cudagraph tree). The Starling source does NOT
provide a hand-fused Conformer kernel to copy. The correct ggml port strategy is
documented in Section 8 (priorities): the highest-value work is (a) a CUDA-graph
or persistent-kernel mechanism to remove launch overhead, and (b) porting the
hand-written **LLM-decoder** elementwise kernels (rmsnorm/silu-mul/residual) only
if parakeet.cpp's decoder uses the same SwiGLU+RMSNorm structure (the parakeet
TDT decoder does NOT — it's LSTM + linear joint, see `ALGORITHM.md`).

The hand-written kernel specs ARE still documented in full below (Sections 1-6),
because they are the exact numerical recipes the brief asks for, and they're the
template for any custom ggml CUDA op of the same shape.

---

## Architectural reference (parakeet-tdt-0.6b-v3)

From `src/starling/parakeet/ALGORITHM.md` and
`src/starling/parakeet_unified/config.py:60-79`:

- Encoder: 24 Conformer layers, d_model 1024, 8 heads, head_dim 128, macaron-style
  (FFN1 + FFN2), relative-pos attention, conv module = depthwise conv kernel 9 +
  BatchNorm1d. Conv2DSubsampling (x8 vs mel frames).
- Encoder output projected 1024 -> 640 (`encoder_projector`).
- TDT decoder: `Embedding(8193,640)` + 2-layer `LSTM(640,640)` + `Linear(640,640)`
  + device-side blank-skip freeze. NO RMSNorm, NO SwiGLU, NO RoPE in the decoder.
- Joint: `Linear(640,8198)` over `ReLU(enc_frame + decoder_out)`; 8198 = 8193
  tokens + 5 durations.

The macaron Conformer block order (from the granite sibling encoder, same
structure as parakeet's, `src/starling/granite/encoder_mega.py:232-244`):

```
x = x + 0.5 * ff1(x)          # FFN half-step (macaron)
x = x + attn(x)               # block-local relative-pos attention
x = x + conv(x)               # conv module (norm -> up_conv -> glu -> depthwise -> silu -> bn -> down_conv)
x = x + 0.5 * ff2(x)          # FFN half-step
x = post_norm(x)
```

Note the residual scaling here is `0.5 * ff(...)` per macaron block — a
per-architecture constant, NOT the granite LLM `residual_multiplier=0.22`.

---

## 1. Fused RMSNorm

**Files / function:**
- Triton: `src/starling/_kernels/triton_backend.py:126` (`_rmsnorm_kernel`) and
  launcher `fused_rmsnorm` at `triton_backend.py:157`.
- CUDA C++: `src/starling/_kernels/cuda/backend.cu:54` (`rmsnorm_kernel`),
  entrypoint `fused_rmsnorm` at `backend.cu:183`.
- Torch fallback: `src/starling/_kernels/torch_backend.py:97`.

**Exact formula** (`triton_backend.py:138-149`):
```
var   = sum(x.to(f32)^2) / N          # mean of squares, fp32
rstd  = 1.0 / sqrt(var + eps)
x_normed = (x.to(f32) * rstd).to(bf16)   # truncate to bf16 BEFORE the weight mul
y     = x_normed * weight                 # bf16 * bf16 (Triton fp32 internal -> bf16)
```

- **eps**: passed as a parameter; **not** hardcoded in the kernel. Per-model
  values:
  - Granite: `LLM_RMS_NORM_EPS = 1e-5` (`src/starling/config.py:55`), read as
    `self._rms_eps = float(cfg.rms_norm_eps)` (`granite/llm_mega.py:561`).
  - Qwen3: `1e-6` (`qwen3/llm_mega.py:29`).
- **Mean computed in fp32, then the multiply in bf16**: YES. This is the
  load-bearing detail — the normalized hidden is truncated to bf16 BEFORE
  multiplying by the bf16 weight. The comment at `triton_backend.py:144-148`
  explicitly states computing the weight product in fp32 and truncating once gives
  a different (0.125-magnitude) result. Same recipe in CUDA
  (`backend.cu:74-78`: `__float2bfloat16(xi * rstd)` then `__hmul(xn, weight[i])`).
- **Input/output dtype**: bf16 in, bf16 out. Weight is bf16. Internal variance in
  fp32. (`triton_backend.py:136` `dtype = Y_ptr.dtype.element_ty  # bf16`.)
- **Residual add folded in**: NO. The residual add is a separate kernel
  (`residual_add`, Section 3). RMSNorm is invoked AFTER the residual in the decode
  loop (e.g. `granite/llm_mega.py:918` residual, then `:921` rmsnorm).
- **Layout**: row-major `(*, N)`, one program/block per leading row, reduction
  over the last dim `N` (= hidden_size). `N = weight.numel()`; the launcher
  reshapes to `(M, N)` (`triton_backend.py:163-172`). For decode `M = 1`.
  `BLOCK_N = next_power_of_2(N)` (`triton_backend.py:169`).

**ggml port notes:** Replaces `ggml_rms_norm` + a separate `ggml_mul`.
Existing `ggml_rms_norm` computes the whole thing in fp32 and truncates once,
which does NOT match this recipe — to be byte-exact you need a custom op that
truncates the normalized vector to bf16 before the elementwise weight multiply,
or fuse `rmsnorm -> mul(weight)` into one custom op that reproduces the two-step
truncation. Must be a **new custom CUDA op** (ggml's built-in rms_norm has the
wrong rounding for Granite/Qwen3 byte-exactness).

**parakeet-tdt uses it:** NO (parakeet's Conformer uses LayerNorm, not RMSNorm,
and runs it via HF/Inductor, not this kernel). Relevant only to the LLM-decoder
models (granite/qwen3/moss/higgs/ark).

---

## 2. Fused SiLU·mul (SwiGLU gate)

**Files / function:**
- Triton: `src/starling/_kernels/triton_backend.py:274` (`_silu_mul_kernel`),
  launcher `fused_silu_mul` at `triton_backend.py:300`.
- CUDA: `src/starling/_kernels/cuda/backend.cu:87` (`silu_mul_kernel`),
  entrypoint `backend.cu:197`.
- Torch: `torch_backend.py:116`.

**Exact formula** (`triton_backend.py:285-293`):
```
g      = gate.to(f32)
silu_g = g * (1.0 / (1.0 + exp(-g)))        # SiLU in fp32
silu_g_bf = silu_g.to(bf16)                  # truncate to bf16 BEFORE the mul
u      = up                                 # bf16, no cast
out    = silu_g_bf * u                       # bf16 * bf16
```

- **dtype**: gate/up bf16, out bf16. SiLU sigmoid computed in fp32, **truncated
  to bf16 before multiplying by up** — matches PyTorch ATen's intermediate
  truncation (`triton_backend.py:287-289` comment; CUDA `backend.cu:100-102`).
- **One fused kernel**: YES — silu and mul are fused into one launch. But it is
  **never fused with the preceding matmul or the following residual**. The
  precedes-matmul is a plain cuBLAS GEMM (`granite/llm_mega.py:892`
  `F.linear(x3, f["gu_w"])`); the silu·mul result feeds a separate down-proj GEMM
  (`:913`) and then a separate `residual_add` (`:918`). Confirmed identically in
  `qwen3/llm_mega.py:561` and `moss/fused_decode.py:238`.
- **Layout**: `(*, N)` row-major, `N` = intermediate_size (4096 for granite).
  One program/row, `BLOCK_N = next_power_of_2(N)`.

**ggml port notes:** Replaces `ggml_silu` + `ggml_mul` (two ggml nodes → one).
Can be a **new custom op** `ggml_silu_mul` (ggml has `ggml_silu_inplace` and
`ggml_mul` separately; the fusions exist in some backends but not as a single
byte-exact op with the bf16-truncation-between-steps rule). Worth fusing for
kernel-count reduction in any LLM-decoder port.

**parakeet-tdt uses it:** NO (parakeet TDT decoder has no SwiGLU; the Conformer
FFN uses SwiGLU but it's run by HF/Inductor, not this kernel). Relevant only to
LLM-decoder models.

---

## 3. Fused residual scale-add (`residual_add` / `fused_residual_scale`)

**Files / function:**
- Triton: `src/starling/_kernels/triton_backend.py:333` (`_residual_scale_kernel`),
  launcher `residual_add` at `triton_backend.py:358` (unified name; the
  per-model shim `granite/llm_kernels.py` re-exports it as
  `fused_residual_scale`).
- CUDA: `src/starling/_kernels/cuda/backend.cu:112` (`residual_add_kernel`),
  entrypoint `backend.cu:211`.
- Torch: `torch_backend.py` (residual_add).

**Exact formula** (`triton_backend.py:346-351`):
```
y       = Y_ptr.load().to(f32)
scaled  = (alpha * y).to(bf16)       # fp32 product, truncate to bf16
x       = X_ptr.load()              # bf16 residual
z       = x + scaled                # bf16 + bf16
```
The `alpha == 1.0` fast path in CUDA is plain `__hadd(x, y)` (`backend.cu:121-124`).

- **Scale**: scalar `alpha`. **Granite** passes
  `LLM_RESIDUAL_MULTIPLIER = 0.22` (`src/starling/config.py:59`), read as
  `self._res_mult = float(cfg.residual_multiplier)` (`granite/llm_mega.py:560`)
  and passed at `granite/llm_mega.py:846` and `:918`. **Qwen3 / Moss** pass
  `alpha=1.0` (plain `x + y`) — `qwen3/llm_mega.py:545,563`,
  `moss/fused_decode.py` (no `residual_multiplier` in its config).
- **parakeet Conformer**: does NOT use this kernel and does NOT use the granite
  `0.22` multiplier. The macaron Conformer residuals are plain `x + branch` (or
  `x + 0.5 * ff(x)` for the FFN half-steps — a hardcoded `0.5`, not a learned
  multiplier), from `granite/encoder_mega.py:235-241`.
- **Order**: scale the delta in fp32 → truncate to bf16 → add the residual (bf16).
  Matches the Granite model's `residual + delta * multiplier` recipe
  (`triton_backend.py:344-345`).
- **Fused with the following norm?** NO. Always a standalone kernel.

**ggml port notes:** Replaces `ggml_mul` (constant) + `ggml_add`, or a single
`ggml_add` when `alpha=1`. For the `alpha=1` case (parakeet/Qwen3/moss, and the
Conformer macaron residual), ggml's existing `ggml_add` IS byte-exact (bf16 add
of two bf16 tensors). For granite's `alpha=0.22`, you need a custom
`ggml_axpy`-style op that scales in fp32 then truncates to bf16 before the add.

**parakeet-tdt uses it:** NO (Conformer residuals are native HF adds). The
concept (plain `x + branch`) applies if parakeet.cpp emits the macaron residuals
as separate nodes — but there's no scaling to fuse, just an add.

---

## 4. fp8 weight-only dequant-GEMV  (LLM decoder only; NOT parakeet)

**Files / function:**
- Triton: `src/starling/_kernels/triton_backend.py:507` (`_fp8_gemv_kernel`),
  launcher `fp8_linear` at `triton_backend.py:537`; quantizer
  `quantize_weight_e4m3` at `triton_backend.py:483`.
- CUDA: `src/starling/_kernels/cuda/backend.cu:145` (`fp8_gemv_kernel`),
  entrypoint `fp8_linear` at `backend.cu:224`.
- Constants: `FP8_DTYPE = torch.float8_e4m3fn`, `FP8_MAX = 448.0`
  (`src/starling/_kernels/base.py:56,59`).

**fp8 format:** E4M3 (`float8_e4m3fn`). NOT E5M2.

**Weight-only:** YES. Weights are stored fp8e4m3 and dequantized on the fly in
the GEMV; the activation `x` stays bf16 (`triton_backend.py:480-482`).

**Quantization scheme** (`quantize_weight_e4m3`, `triton_backend.py:483-493`):
per-output-channel symmetric absmax.
```
amax  = weight.abs().amax(dim=1).clamp(min=1e-8)   # (N,) per output row
scale = amax / 448.0                               # (N,) fp32, per output channel
w_fp8 = (weight / scale[:, None]).clamp(-448, 448).to(float8_e4m3fn).contiguous()
```
Layout: `w_fp8` is `(N, K)` row-major (the layout the GEMV streams). `scale` is
`(N,)` fp32.

**Exact GEMV math** (`_fp8_gemv_kernel`, `triton_backend.py:517-534`):
```
for each output row o (BLOCK_M rows per program):
  scale[o] = load(SCALE_ptr + o)               # fp32 per-channel
  acc = zeros(BLOCK_M, fp32)
  for k0 in range(0, K, BLOCK_K):
    w = load(W[o, k:k+BLOCK_K]).to(f32)        # hardware fp8 -> fp32 cast
    x = load(X[k:k+BLOCK_K]).to(f32)           # bf16 -> fp32
    acc += sum(w * x, axis=K)                  # accumulate in fp32
  acc = acc * scale                            # apply per-channel scale (fp32)
  store(out[o], acc.to(bf16))                  # truncate once at the end
```
CUDA mirrors this: `(float)wr[k]` hardware cast, `__bfloat162float(x[k])`, dot in
fp32, block-reduce, `__float2bfloat16(sum * scale[row])` (`backend.cu:158-168`).

- **Accumulation in fp32**: YES.
- **Activation dtype**: bf16. No per-token activation quant.
- **Output**: `(1, N)` bf16.

**Autotune** (`triton_backend.py:496-502`): sweeps BLOCK_M ∈ {16,32,64,128},
BLOCK_K ∈ {128,256,512,1024}, num_warps ∈ {4,8}, num_stages ∈ {1,2,3}. CUDA uses
a fixed `BLOCK_K=256`, one block per output row (`backend.cu:233-234`).

**ggml port notes:** Replaces a bf16 `ggml_mul_mat` for the per-step
projections. ggml already has `GGML_TYPE_F8_E4M3` and dequant-GEMV support in
`vecdotq`/`mul_mat_vec_q`; whether it is byte-exact with THIS per-channel scale
application (scale multiplied into the fp32 accumulator AFTER the full K-reduce,
not per-element) must be verified. If ggml applies the scale per-element during
the dot product, the fp32 accumulation rounding will differ. Likely needs a
**custom CUDA op** to guarantee the `acc *= scale` post-reduce ordering.

**parakeet-tdt uses it:** **NO — explicitly out of scope.** parakeet-tdt is
bf16 throughout. `OptFlags.fp8_weights` defaults `False` (`flags.py:123`) and
parakeet never enables it. Only the granite/moss LLM-decoder path can reach
`fp8_linear` (`granite/llm_mega.py:770-778,834-837`). Do NOT port fp8 for
parakeet.

---

## 5. Conformer-layer fusion structure (the key question for kernel count)

**Answer: Starling does NOT hand-fuse the Conformer layer. It relies entirely on
PyTorch Inductor (`torch.compile`) + CUDAGraph capture.**

Evidence:
- `src/starling/parakeet/encoder_graph.py:320` (`CompiledEncoder`) wraps
  `model.get_audio_features` in `torch_compile(_encode, mode="reduce-overhead")`
  (`encoder_graph.py:382`). Inductor traces the HF-native Conformer and emits its
  OWN auto-fused Triton kernels for the elementwise chains.
- `src/starling/parakeet/encoder_graph.py:182` (`GraphedEncoder`, the DEFAULT
  encoder mode per `pipeline.py:211`) captures the eager HF forward into one
  `torch.cuda.CUDAGraph` with NO fusion at all — it just collapses launches.
- The granite Conformer encoder (same architecture) is likewise stock PyTorch:
  `src/starling/granite/encoder_mega.py:220-244` (`_block_eager`,
  `_forward_impl`) — a pure-Python macaron block loop using HF modules. The only
  "fusion" is an optional `fold_conformer_batchnorm` that bakes the BN affine
  into the depthwise conv weights (`encoder_mega.py:324`, mirroring parakeet's
  `fold_conformer_batchnorm` at `encoder_graph.py:68`).

**What this means for the ggml port's ~1200-node problem:**

Starling's lower kernel count on the Conformer comes from TWO layers of
launch-overhead removal, NEITHER of which is a hand-written fused Conformer
kernel:

1. **CUDAGraph replay** (the `graphed` default, byte-exact): one
   `graph.replay()` replaces ~hundreds of per-layer host launches. The kernel
   count on-GPU is unchanged; only the host-side launch tax is removed. This is
   the dominant win and is a **scheduling** change, not a fusion change.
2. **Inductor fusion** (`compiled` mode, not byte-exact): Inductor fuses the
   elementwise glue (residual adds, the `0.5 *` macaron scale, activations, the
   norm/mul chains) into auto-generated Triton kernels. This is what reduces
   on-GPU kernel count, but the fusion decisions are Inductor's, not encoded in
   any Starling source file you can copy.

There is NO Starling source that says "the Conformer block is N kernels." To
match Starling's on-GPU kernel count, parakeet.cpp must either (a) use Inductor's
fusion rules as a guide (out of scope here — read Inductor, not Starling), or
(b) hand-fuse the obvious chains listed in Section 8.

---

## 6. Depthwise conv module (Conformer)

**Answer: Starling does NOT write a custom conv kernel. It uses stock
`nn.Conv1d` (groups=channels) and folds the BatchNorm1d into the conv weights at
load time.**

Evidence (`src/starling/parakeet/encoder_graph.py:68-157`,
`fold_conformer_batchnorm`):
- The conv module is HF's `ParakeetEncoderConvolutionModule` with
  `depthwise_conv` (`nn.Conv1d`, groups == channels) + `norm`
  (`nn.BatchNorm1d`).
- For inference, BN is a deterministic per-channel affine:
  `y = (x - running_mean)/sqrt(running_var+eps) * weight + bias`
  (`encoder_graph.py:73-74`).
- Folded into the preceding depthwise conv (`encoder_graph.py:120-128`):
  ```
  scale = bn_weight / sqrt(running_var + bn_eps)            # (C,) fp32
  W'    = W * scale[:, None, None]                          # (C,1,K) fp32
  b'    = (b - running_mean) * scale + bn_bias              # (C,) fp32
  ```
  then re-cast to the conv dtype. `conv.norm = nn.Identity()` removes the BN
  entirely (`encoder_graph.py:145`).
- `bn.eps` is read from the loaded BN module (`encoder_graph.py:118`). The fold
  is exact in fp32; in bf16 there can be sub-ULP differences because the scale is
  baked into the weight rather than applied to the conv output
  (`encoder_graph.py:82-87`).

**Kernelization:** ONE op — the depthwise `nn.Conv1d` (fused with the former BN
via the weight/bias rewrite). Run by cuDNN under HF, captured into the graph /
compiled by Inductor. No Starling custom kernel.

**ggml port notes:** Replaces `ggml_conv_transpose_1d` / depthwise conv +
`ggml_mul` + `ggml_add` (for the BN). Do the BN fold offline at weight-load time
(rewrite depthwise weight & bias once), then ggml needs only a depthwise conv-1d
op. ggml does NOT have a built-in depthwise Conv1d with `groups=channels` and a
generic kernel (the conv kernel is 9 for parakeet, `config.py:65`); this likely
needs a **custom CUDA op** (a small `__shared__`-buffer depthwise conv, kernel=9,
which is trivially fused with the surrounding `silu` and the up/down pointwise
convs — see Section 8).

**parakeet-tdt uses it:** YES (every Conformer layer has one). The BN fold is
already applied by Starling's `compiled` mode and SHOULD be replicated in the
ggml port's weight loader.

---

## 7. Secondary kernels (documented for completeness; not on the parakeet path)

These are reachable only by the LLM-decoder models. Included because the brief
asks for the full `_kernels/` inventory.

### 7a. Fused RoPE (Q + K in one launch)
- Triton: `triton_backend.py:187` (`_rope_kernel`), launcher `fused_rope:229`.
- CUDA: `backend.cu:253` (`rope_kernel`), entrypoint `backend.cu:289`.
- Formula: `rotate_half(x)[i] = -x[i+hd/2] for i<hd/2; x[i-hd/2] for i>=hd/2`.
  `out = bf16(x*cos) + bf16(rotate_half(x)*sin)` — each product fp32→bf16
  truncated BEFORE the add (`triton_backend.py:223-226`, `backend.cu:283-285`).
  Q and K share one launch (grid = n_q + n_kv).
- **parakeet uses it:** NO (Conformer uses relative-pos attention; TDT decoder
  has no RoPE). Note: the granite LLM port actually keeps RoPE in PyTorch, NOT
  this kernel, because Triton's bf16 rounding on large Q values (>0.05 logit
  divergence) breaks the byte-exact gate (`granite/llm_mega.py:713-716`).

### 7b. GEMM-epilogue fusion (`compute_rstd` + `fused_gemv_normscale`)
- Triton: `triton_backend.py:407` (`_rstd_kernel`), `triton_backend.py:426`
  (`_gemv_normscale_kernel`); launchers at `:417` and `:454`.
- CUDA: `backend.cu:316` (`rstd_kernel`), `backend.cu:349` (`gemv_normscale_kernel`).
- Folds `rstd = rsqrt(mean(x²)+eps)` into the M=1 GEMV epilogue:
  `out[i] = rstd * sum_k (gamma·W)[i,k] * x[k]`, where `W` is pre-prescaled by
  gamma at load time. fp32 accumulate, bf16 out (`triton_backend.py:442-451`).
- **EXPERIMENTAL and OFF by default** — gated by `OptFlags.gemm_epilogue_fusion`
  (`flags.py:139`, default `False`). Not byte-exact (~1-3 bf16 ULP,
  `triton_backend.py:404-406`). Only granite's LLM decoder.
- **parakeet uses it:** NO.

### 7c. NVFP4 dequant-GEMV
- Triton: `triton_backend.py:592` (`_fp4_gemv_kernel`), launcher `fp4_gemv_fused:671`.
- CUDA: `backend.cu:401` (`fp4_gemv_kernel`), entrypoint `backend.cu:426`.
- Storage: nibble-packed e2m1 codes `(OUT, K//2) uint8` + per-16-block fp8e4m3
  scales `(OUT, K//16)`. `BLOCK_SIZE = 16` (`triton_backend.py:579`).
- Dequant: `w ~= scale_fp8 * e2m1_level(code) / 6.0`, where e2m1 levels are
  `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` indexed by `code & 0x7`, sign = `(code>>3)&1`
  (`triton_backend.py:629-647`, CUDA `backend.cu:385-398`). fp32 accumulate.
- **EXPERIMENTAL, OFF** — `OptFlags.nvfp4_weights` (`flags.py:144`, default
  `False`); also requires `tolerance_mode=True`. Granite only.
- **parakeet uses it:** NO.

---

## 8. Prioritized fusions for closing the parakeet Conformer gap

The brief asks which 2-3 fusions would most cut parakeet's kernel count. Given
the Section 0/Section 5 finding that **Starling has no hand-fused Conformer
kernel to copy**, the honest ranking is:

**Priority 1 — Eliminate launch overhead (NOT a fusion; the actual bottleneck).**
Starling's `graphed` encoder (the default, byte-exact) does ZERO fusion — it
wins purely by collapsing ~hundreds of per-layer launches into one
`graph.replay()` (`encoder_graph.py:213-271`). parakeet.cpp's "93% GPU idle" is
the same launch-overhead symptom Starling solved with CUDA graphs. The single
highest-value ggml work is a graph-capture / persistent-kernel / mega-kernel
launch mechanism for the encoder, NOT hand-fusing ops. Source:
`encoder_graph.py:1-39` (module docstring: "it is launch-overhead bound (hundreds
of tiny per-layer kernels with sequential dependencies)").

**Priority 2 — Fuse the conv-module chain into one kernel.** Per Conformer layer
the conv module is the most fragmented chain: `LayerNorm → up_conv(pointwise) →
GLU → depthwise_conv(kernel 9) → SiLU → (BN folded) → down_conv(pointwise)`. With
the BN pre-folded into the depthwise weights (Section 6), ggml currently emits
this as ~5-6 separate nodes. A single fused depthwise-conv + SiLU + pointwise
CUDA op (with the two pointwise Linear/GEMMs kept separate, or fused as
`GEMM+activation` epilogues) would cut the most nodes/layer here. d_model=1024,
conv kernel=9, 24 layers. Reference structure: `granite/encoder_mega.py:306-320`
(`_conv_forward`).

**Priority 3 — Fuse the macaron FFN + residual + norm chains.** Each FFN
half-step is `LayerNorm → GEMM(1024→4096) → SwiGLU → GEMM(4096→1024) →
scale 0.5 → residual add`. Fuse `SwiGLU → scale(0.5) → add(residual)` into one
custom op (mirrors the LLM-decoder `fused_silu_mul` + `residual_add` pattern in
Sections 2-3, but with the macaron `0.5` constant and a LayerNorm instead of
RMSNorm). Two FFN blocks per layer × 24 layers = 48 residual-scale-add nodes
collapsible.

**Not worth porting for parakeet:** fp8/fp4 GEMV (parakeet is bf16 — Section 4),
RMSNorm (parakeet Conformer uses LayerNorm — Section 1), RoPE (parakeet uses
relative-pos attention — Section 7a). All of these are LLM-decoder-only.

**Bottom line:** Starling's parakeet Conformer speed is a launch-overhead
solution (CUDA graphs) plus Inductor's automatic fusion — there is no hand-written
fused Conformer kernel in this repo to port byte-for-byte. The hand-written
kernels in `src/starling/_kernels/` are the right *numerical template* (fp32
internal, bf16 truncation ordering) for any custom ggml op of the same shape, but
they apply to the LLM-decoder (granite/moss/qwen3), not to parakeet's encoder.
