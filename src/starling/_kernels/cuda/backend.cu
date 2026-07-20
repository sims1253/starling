// =========================================================================== //
// Starling fused decode kernels -- CUDA C++ source.
//
// Cross-platform (compiles on Linux gcc + Windows MSVC via nvcc) implementations
// of the same fused ops as the Triton backend.  Selected automatically where
// triton is unavailable (Windows) and CUDA is present, giving Windows the same
// fused-kernel performance as Linux's Triton path.  Verified byte-exact with the
// Triton and torch backends (see tests/test_kernel_backends.py).
//
// All kernels use bf16 inputs/outputs with fp32 internal accumulation, matching
// the model's eager reference ops.  bf16 math uses __nv_bfloat16 / __nv_bfloat162;
// the fp8 e4m3 weight uses __nv_fp8_e4m3 with the hardware cast to fp32.
//
// Reduction note: rmsnorm uses a two-pass warp+block reduction (sum of squares).
// For the decode shapes here (N=2048 or 128) the block is sized to cover the
// whole row so this is a single-block, single-reduction-pass warp shuffle.
// =========================================================================== //
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ---- block / warp helpers ---------------------------------------------------
template <int BLOCK>
__device__ __forceinline__ float block_sum(float val) {
    // warp shuffle reduction
    for (int off = 16; off > 0; off >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, off);
    }
    __shared__ float s[BLOCK / 32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) s[wid] = val;
    __syncthreads();
    int n_warps = BLOCK / 32;
    val = (threadIdx.x < n_warps) ? s[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
    }
    __shared__ float bcast;
    if (threadIdx.x == 0) bcast = val;
    __syncthreads();
    return bcast;
}

// =========================================================================== //
// fused_rmsnorm:  y[i] = weight[i] * (bf16)((bf16->f32)x[i] * rstd)
//   rstd = rsqrt(mean(x^2) + eps), mean computed in fp32.
//   ONE block per row, BLOCK threads covering the (power-of-two) feature dim.
//   Matches GraniteRMSNorm/Qwen3RMSNorm: normalize in fp32, truncate to bf16,
//   THEN multiply by weight in bf16.
//   x,y: (*, N) bf16 row-major; weight: (N,) bf16.  eps: fp32 scalar.
// =========================================================================== //
template <int BLOCK>
__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    float eps,
    int N)
{
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * N;
    __nv_bfloat16* yr = y + (int64_t)row * N;

    float v = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xi = __bfloat162float(xr[i]);
        v += xi * xi;
    }
    float sumsq = block_sum<BLOCK>(v);
    float rstd = rsqrtf(sumsq / (float)N + eps);

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xi = __bfloat162float(xr[i]);
        __nv_bfloat16 xn = __float2bfloat16(xi * rstd);   // truncate to bf16 (matches model)
        yr[i] = __hmul(xn, weight[i]);                     // bf16 * bf16
    }
}

// =========================================================================== //
// fused_silu_mul:  out[i] = (bf16)silu(gate[i]) * up[i]
//   silu(g) = g * sigmoid(g), computed in fp32, truncated to bf16 BEFORE the
//   multiply by up (matches PyTorch ATen intermediate truncation).
// =========================================================================== //
template <int BLOCK>
__global__ void silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int N)
{
    int row = blockIdx.x;
    const __nv_bfloat16* gr = gate + (int64_t)row * N;
    const __nv_bfloat16* ur = up   + (int64_t)row * N;
    __nv_bfloat16* or_ = out + (int64_t)row * N;

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float g = __bfloat162float(gr[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        __nv_bfloat16 silu_g = __float2bfloat16(g * sig);   // fp32 silu -> bf16
        or_[i] = __hmul(silu_g, ur[i]);                      // bf16 * bf16
    }
}

// =========================================================================== //
// residual_add:  z[i] = x[i] + (bf16)(alpha * (f32)y[i])
//   alpha=1.0 fast path is plain x+y (the Qwen3/Moss case).  General path
//   computes the scaled delta in fp32, truncates to bf16, THEN adds the
//   residual (matches the Granite residual_multiplier recipe).
// =========================================================================== //
__global__ void residual_add_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ z,
    float alpha,
    int N)
{
    int row = blockIdx.x;
    int64_t base = (int64_t)row * N;
    if (alpha == 1.0f) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            z[base + i] = __hadd(x[base + i], y[base + i]);   // plain x + y
        }
    } else {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float yf = __bfloat162float(y[base + i]);
            __nv_bfloat16 scaled = __float2bfloat16(alpha * yf);
            z[base + i] = __hadd(x[base + i], scaled);
        }
    }
}

// =========================================================================== //
// fp8 dequant-GEMV (M=1):  out[o] = scale[o] * sum_k (f32)(w_fp8[o,k]) * (f32)x[k]
//   w_fp8: (OUT, K) fp8e4m3 row-major.  x: (K,) bf16.  scale: (OUT,) fp32.
//
// One BLOCK (BLOCK_K threads) computes one output row: each thread strides
// across K accumulating partial dot products, then a block-reduction sums them.
// This is the standard bandwidth-bound GEMV pattern (each weight byte read once,
// x reused from L1).  Mirrors the Triton _fp8_gemv_kernel accumulation: fp8
// weight -> fp32 hardware cast, bf16 x -> fp32, dot in fp32, *= per-row scale.
// =========================================================================== //
template <int BLOCK_K>
__global__ void fp8_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,       // (K,) bf16
    const __nv_fp8_e4m3* __restrict__ w,       // (OUT, K) fp8e4m3 row-major
    const float* __restrict__ scale,           // (OUT,) fp32
    __nv_bfloat16* __restrict__ out,           // (OUT,) bf16
    int K,
    int OUT)
{
    int row = blockIdx.x;
    if (row >= OUT) return;
    const __nv_fp8_e4m3* wr = w + (int64_t)row * K;

    // Each thread accumulates over its strided set of columns.
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += BLOCK_K) {
        float wv = (float)wr[k];                 // hardware fp8 -> fp32 cast
        float xv = __bfloat162float(x[k]);       // bf16 -> fp32
        acc += wv * xv;
    }
    // Block-wide reduction to one scalar.
    float sum = block_sum<BLOCK_K>(acc);
    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(sum * scale[row]);
    }
}

// =========================================================================== //
// torch C++ entrypoints
// =========================================================================== //
static int pick_block(int N) {
    if (N <= 32)    return 32;
    if (N <= 64)    return 64;
    if (N <= 128)   return 128;
    if (N <= 256)   return 256;
    if (N <= 512)   return 512;
    return 1024;
}

torch::Tensor fused_rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
    int N = weight.size(0);
    int M = x.numel() / N;
    auto y = torch::empty_like(x);
    int B = pick_block(N);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (B == 128)       rmsnorm_kernel<128><<<M, 128, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)weight.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (float)eps, N);
    else if (B == 256)  rmsnorm_kernel<256><<<M, 256, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)weight.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (float)eps, N);
    else if (B == 512)  rmsnorm_kernel<512><<<M, 512, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)weight.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (float)eps, N);
    else if (B == 1024) rmsnorm_kernel<1024><<<M, 1024, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)weight.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (float)eps, N);
    else                rmsnorm_kernel<64><<<M, 64, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)weight.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (float)eps, N);
    return y;
}

torch::Tensor fused_silu_mul(torch::Tensor gate, torch::Tensor up) {
    int N = gate.size(-1);
    int M = gate.numel() / N;
    auto out = torch::empty_like(gate);
    int B = pick_block(N);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (B == 128)       silu_mul_kernel<128><<<M, 128, 0, stream>>>((const __nv_bfloat16*)gate.data_ptr(), (const __nv_bfloat16*)up.data_ptr(), (__nv_bfloat16*)out.data_ptr(), N);
    else if (B == 256)  silu_mul_kernel<256><<<M, 256, 0, stream>>>((const __nv_bfloat16*)gate.data_ptr(), (const __nv_bfloat16*)up.data_ptr(), (__nv_bfloat16*)out.data_ptr(), N);
    else if (B == 512)  silu_mul_kernel<512><<<M, 512, 0, stream>>>((const __nv_bfloat16*)gate.data_ptr(), (const __nv_bfloat16*)up.data_ptr(), (__nv_bfloat16*)out.data_ptr(), N);
    else if (B == 1024) silu_mul_kernel<1024><<<M, 1024, 0, stream>>>((const __nv_bfloat16*)gate.data_ptr(), (const __nv_bfloat16*)up.data_ptr(), (__nv_bfloat16*)out.data_ptr(), N);
    else                silu_mul_kernel<64><<<M, 64, 0, stream>>>((const __nv_bfloat16*)gate.data_ptr(), (const __nv_bfloat16*)up.data_ptr(), (__nv_bfloat16*)out.data_ptr(), N);
    return out;
}

torch::Tensor residual_add(torch::Tensor x, torch::Tensor y, c10::optional<double> alpha) {
    int N = x.size(-1);
    int M = x.numel() / N;
    auto z = torch::empty_like(x);
    float a = (alpha && *alpha) ? (float)*alpha : 1.0f;
    int B = pick_block(N);
    auto stream = at::cuda::getCurrentCUDAStream();
    residual_add_kernel<<<M, B, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)y.data_ptr(),
        (__nv_bfloat16*)z.data_ptr(), a, N);
    return z;
}

torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor w_fp8, torch::Tensor w_scale) {
    int K = w_fp8.size(1);
    int OUT = w_fp8.size(0);
    auto out = torch::empty({OUT}, x.options());
    auto x1 = x.reshape({-1}).slice(0, 0, K).contiguous();
    auto scale1d = w_scale.reshape({-1});
    auto stream = at::cuda::getCurrentCUDAStream();
    // One block per output row, 256 threads striding over K. BLOCK_K=256 keeps
    // the warp-reduction cheap and gives good memory coalescing on the fp8 read.
    constexpr int BK = 256;
    fp8_gemv_kernel<BK><<<OUT, BK, 0, stream>>>(
        (const __nv_bfloat16*)x1.data_ptr(),
        (const __nv_fp8_e4m3*)w_fp8.data_ptr(),
        scale1d.data_ptr<float>(),
        (__nv_bfloat16*)out.data_ptr(),
        K, OUT);
    return out.view({1, OUT});
}

// =========================================================================== //
// fused RoPE (rotary position embedding) applied to Q and K simultaneously.
//   Byte-exact match of the Triton _rope_kernel.
//   q_out = q * cos + rotate_half(q) * sin ; k_out likewise.
//   rotate_half(x)[i] = -x[i+hd/2] for i<hd/2 ; x[i-hd/2] for i>=hd/2.
//   Grid: one block per head across BOTH Q and K (grid = n_q + n_kv).
//   Block: HEAD_DIM threads (one per element; head_dim is always 128 here).
//   Load-bearing detail: each product is fp32->bf16-truncated BEFORE the bf16
//   add, matching Triton's (x*cos).to(bf16) + (x_rot*sin).to(bf16).
// =========================================================================== //
template <int HEAD_DIM>
__global__ void rope_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ QO,
    __nv_bfloat16* __restrict__ KO,
    const float* __restrict__ COS,
    const float* __restrict__ SIN,
    int n_q)
{
    constexpr int HALF = HEAD_DIM / 2;
    int pid = blockIdx.x;
    int i   = threadIdx.x;
    const __nv_bfloat16* src;
    __nv_bfloat16* dst;
    if (pid < n_q) {
        src = Q  + (int64_t)pid * HEAD_DIM;
        dst = QO + (int64_t)pid * HEAD_DIM;
    } else {
        int kid = pid - n_q;
        src = K  + (int64_t)kid * HEAD_DIM;
        dst = KO + (int64_t)kid * HEAD_DIM;
    }
    if (i >= HEAD_DIM) return;
    float cos_v = COS[i];
    float sin_v = SIN[i];
    float x = __bfloat162float(src[i]);
    int   rot_idx = (i < HALF) ? (i + HALF) : (i - HALF);
    float x_rot   = __bfloat162float(src[rot_idx]);
    if (i < HALF) x_rot = -x_rot;
    __nv_bfloat16 prod1 = __float2bfloat16(x     * cos_v);
    __nv_bfloat16 prod2 = __float2bfloat16(x_rot * sin_v);
    dst[i] = __hadd(prod1, prod2);
}

// q_flat: (n_q, hd) bf16, k_flat: (n_kv, hd) bf16, cos/sin: (hd,) fp32.
std::vector<torch::Tensor> fused_rope(
    torch::Tensor q_flat, torch::Tensor k_flat,
    torch::Tensor cos_flat, torch::Tensor sin_flat)
{
    int n_q  = q_flat.size(0);
    int n_kv = k_flat.size(0);
    int hd   = q_flat.size(1);
    auto q_out = torch::empty_like(q_flat);
    auto k_out = torch::empty_like(k_flat);
    int total_heads = n_q + n_kv;
    auto stream = at::cuda::getCurrentCUDAStream();
    rope_kernel<128><<<total_heads, 128, 0, stream>>>(
        (const __nv_bfloat16*)q_flat.data_ptr(),
        (const __nv_bfloat16*)k_flat.data_ptr(),
        (__nv_bfloat16*)q_out.data_ptr(),
        (__nv_bfloat16*)k_out.data_ptr(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        n_q);
    return {q_out, k_out};
}

// =========================================================================== //
// rstd_kernel:  rstd = rsqrt(mean(x^2) + eps)  [single-block scalar producer]
//   x: (N,) bf16 flattened.  rstd: (1,) fp32.  CODA Pattern 1 fusion (granite).
// =========================================================================== //
template <int BLOCK>
__global__ void rstd_kernel(
    const __nv_bfloat16* __restrict__ x, float* __restrict__ rstd, float eps, int N)
{
    float v = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xi = __bfloat162float(x[i]);
        v += xi * xi;
    }
    float sumsq = block_sum<BLOCK>(v);
    if (threadIdx.x == 0) rstd[0] = rsqrtf(sumsq / (float)N + eps);
}

torch::Tensor compute_rstd(torch::Tensor x, double eps) {
    int N = x.size(-1);
    auto x1 = x.reshape({-1}).contiguous();
    auto rstd = torch::empty({1}, x.options().dtype(torch::kFloat32));
    int B = pick_block(N);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (B == 32)       rstd_kernel<32><<<1, 32, 0, stream>>>((const __nv_bfloat16*)x1.data_ptr(), rstd.data_ptr<float>(), (float)eps, N);
    else if (B == 64)  rstd_kernel<64><<<1, 64, 0, stream>>>((const __nv_bfloat16*)x1.data_ptr(), rstd.data_ptr<float>(), (float)eps, N);
    else if (B == 128) rstd_kernel<128><<<1, 128, 0, stream>>>((const __nv_bfloat16*)x1.data_ptr(), rstd.data_ptr<float>(), (float)eps, N);
    else if (B == 256) rstd_kernel<256><<<1, 256, 0, stream>>>((const __nv_bfloat16*)x1.data_ptr(), rstd.data_ptr<float>(), (float)eps, N);
    else if (B == 512) rstd_kernel<512><<<1, 512, 0, stream>>>((const __nv_bfloat16*)x1.data_ptr(), rstd.data_ptr<float>(), (float)eps, N);
    else               rstd_kernel<1024><<<1, 1024, 0, stream>>>((const __nv_bfloat16*)x1.data_ptr(), rstd.data_ptr<float>(), (float)eps, N);
    return rstd;
}

// =========================================================================== //
// gemv_normscale_kernel (M=1): out[i] = rstd * sum_k w_scaled[i,k]*x[k]
//   w_scaled: (OUT, K) bf16 prescaled by gamma.  x: (K,) bf16.  rstd: (1,) fp32.
//   Same row-per-block GEMV pattern as fp8_gemv_kernel; W is bf16 here.
// =========================================================================== //
template <int BLOCK_K>
__global__ void gemv_normscale_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ rstd, __nv_bfloat16* __restrict__ out, int K, int OUT)
{
    int row = blockIdx.x;
    if (row >= OUT) return;
    const __nv_bfloat16* wr = w + (int64_t)row * K;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += BLOCK_K) {
        acc += __bfloat162float(wr[k]) * __bfloat162float(x[k]);
    }
    float sum = block_sum<BLOCK_K>(acc);
    if (threadIdx.x == 0) out[row] = __float2bfloat16(sum * rstd[0]);
}

torch::Tensor fused_gemv_normscale(torch::Tensor x, torch::Tensor w_scaled, torch::Tensor rstd) {
    int K   = w_scaled.size(1);
    int OUT = w_scaled.size(0);
    auto x1 = x.reshape({-1}).slice(0, 0, K).contiguous();
    auto out = torch::empty({OUT}, w_scaled.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BK = 256;
    gemv_normscale_kernel<BK><<<OUT, BK, 0, stream>>>(
        (const __nv_bfloat16*)x1.data_ptr(),
        (const __nv_bfloat16*)w_scaled.data_ptr(),
        rstd.reshape({-1}).data_ptr<float>(),
        (__nv_bfloat16*)out.data_ptr(),
        K, OUT);
    return out;
}

// =========================================================================== //
// NVFP4 dequant-GEMV (M=1): nibble-packed e2m1 codes + fp8 block scales.
//   codes: (OUT, K//2) uint8 ; scales: (OUT, K//16) fp8e4m3 ; x: (K,) bf16.
//   w[k] ~= scale_fp8 * e2m1_level(code) / 6.0.  One block per output row.
// =========================================================================== //
__device__ __forceinline__ float e2m1_level(uint8_t code) {
    float mag;
    switch (code & 0x7) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        default: mag = 6.0f; break;
    }
    return (code >> 3) & 1 ? -mag : mag;
}

template <int BLOCK_K>
__global__ void fp4_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ codes,
    const __nv_fp8_e4m3* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int K_BYTES, int K_BLOCKS, int OUT)
{
    int row = blockIdx.x;
    if (row >= OUT) return;
    const uint8_t* cr = codes + (int64_t)row * K_BYTES;
    const __nv_fp8_e4m3* sr = scales + (int64_t)row * K_BLOCKS;
    const float inv6 = 1.0f / 6.0f;
    float acc = 0.0f;
    for (int b = threadIdx.x; b < K_BYTES; b += BLOCK_K) {
        uint8_t raw = cr[b];
        int elem_lo = 2 * b;
        int blk = elem_lo / 16;                 // == (2b+1)/16 (consecutive elems)
        float scale = (float)sr[blk];
        float w_lo = e2m1_level(raw & 0xF)      * scale * inv6;
        float w_hi = e2m1_level((raw >> 4) & 0xF) * scale * inv6;
        acc += w_lo * __bfloat162float(x[elem_lo])
             + w_hi * __bfloat162float(x[elem_lo + 1]);
    }
    float sum = block_sum<BLOCK_K>(acc);
    if (threadIdx.x == 0) out[row] = __float2bfloat16(sum);
}

torch::Tensor fp4_gemv_fused(torch::Tensor x, torch::Tensor codes, torch::Tensor scales) {
    int OUT = codes.size(0);
    int K_BYTES = codes.size(1);
    int K = K_BYTES * 2;
    int K_BLOCKS = K / 16;
    auto out = torch::empty({OUT}, x.options());
    auto x1 = x.reshape({-1}).slice(0, 0, K).contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BK = 256;
    fp4_gemv_kernel<BK><<<OUT, BK, 0, stream>>>(
        (const __nv_bfloat16*)x1.data_ptr(),
        (const uint8_t*)codes.data_ptr(),
        (const __nv_fp8_e4m3*)scales.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(),
        K_BYTES, K_BLOCKS, OUT);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &fused_rmsnorm, "fused_rmsnorm (CUDA)");
    m.def("fused_silu_mul", &fused_silu_mul, "fused_silu_mul (CUDA)");
    m.def("residual_add", &residual_add, "residual_add (CUDA)");
    m.def("fp8_linear", &fp8_linear, "fp8_linear (CUDA)");
    m.def("fused_rope", &fused_rope, "fused RoPE Q+K (CUDA)");
    m.def("compute_rstd", &compute_rstd, "compute_rstd (CUDA)");
    m.def("fused_gemv_normscale", &fused_gemv_normscale, "fused_gemv_normscale (CUDA)");
    m.def("fp4_gemv_fused", &fp4_gemv_fused, "fp4_gemv_fused (CUDA)");
}
