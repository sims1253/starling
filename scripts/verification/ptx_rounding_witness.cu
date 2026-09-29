// Two source-derived PTX kernels with the same real sum and different bf16 bytes.
// Compile with nvcc -ptx -arch=sm_120; WITNESS_RUN adds a GPU check.
#include <cuda_bf16.h>

extern "C" __global__ void staged_sum(const __nv_bfloat16* x,
                                        const __nv_bfloat16* y,
                                        const __nv_bfloat16* z,
                                        __nv_bfloat16* out) {
    out[0] = __hadd(__hadd(x[0], y[0]), z[0]);
}

extern "C" __global__ void direct_sum(const __nv_bfloat16* x,
                                        const __nv_bfloat16* y,
                                        const __nv_bfloat16* z,
                                        __nv_bfloat16* out) {
    float sum = __bfloat162float(x[0]) + __bfloat162float(y[0]);
    sum += __bfloat162float(z[0]);
    out[0] = __float2bfloat16(sum);
}

#ifdef WITNESS_RUN
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    __nv_bfloat16 *x, *y, *z, *out;
    if (cudaMallocManaged(&x, 2) != cudaSuccess ||
        cudaMallocManaged(&y, 2) != cudaSuccess ||
        cudaMallocManaged(&z, 2) != cudaSuccess ||
        cudaMallocManaged(&out, 2) != cudaSuccess) return 2;
    *x = __float2bfloat16(1.0f);
    *y = __float2bfloat16(1.0f / 256.0f);
    *z = *y;
    staged_sum<<<1, 1>>>(x, y, z, out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    const unsigned staged = *reinterpret_cast<unsigned short*>(out);
    direct_sum<<<1, 1>>>(x, y, z, out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    const unsigned direct = *reinterpret_cast<unsigned short*>(out);
    std::printf("staged=0x%04x direct=0x%04x\n", staged, direct);
    cudaFree(out);
    cudaFree(z);
    cudaFree(y);
    cudaFree(x);
    return staged == 0x3f80 && direct == 0x3f81 ? 0 : 1;
}
#endif
