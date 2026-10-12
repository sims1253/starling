// OpenCL ALU probes on the phone GPU, to separate the fp16 ALU rate from
// kernel structure when explaining ML Drift's speed:
//   f32   8 independent float4 a*x+y chains, register operands
//   f16   the same with half4
//   f16fma / f32fma  the same written with fma() (OpenCL fma() must be
//         correctly rounded; for half the driver may emulate it)
//   bc32  ML Drift's FC inner loop shape in float: a weight float4 per lane,
//         sub_group_broadcast to all lanes, times a register activation
//   bc16  the same with half4 (as_long(half4) broadcast, as ML Drift does)
// cl_alu_probe [groups=512] [iters=4096] [build options]
// Default options are "-cl-std=CL2.0" (no -cl-fast-relaxed-math: with it the
// compiler reassociates the a*x+y chains and the mad rows exceed peak).
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef void *cl_platform_id, *cl_device_id, *cl_context, *cl_command_queue, *cl_program, *cl_kernel, *cl_mem,
    *cl_event;

#define F(ret, name, args) static ret(*name) args;
F(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))
F(cl_int, clGetDeviceIDs, (cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*))
F(cl_int, clGetDeviceInfo, (cl_device_id, cl_uint, size_t, void*, size_t*))
F(cl_context, clCreateContext, (const intptr_t*, cl_uint, const cl_device_id*, void*, void*, cl_int*))
F(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_ulong, cl_int*))
F(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
F(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void*, void*))
F(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_uint, size_t, void*, size_t*))
F(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))
F(cl_mem, clCreateBuffer, (cl_context, cl_ulong, size_t, void*, cl_int*))
F(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))
F(cl_int, clEnqueueNDRangeKernel,
  (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*,
   cl_event*))
F(cl_int, clFinish, (cl_command_queue))
F(cl_int, clWaitForEvents, (cl_uint, const cl_event*))
F(cl_int, clGetEventProfilingInfo, (cl_event, cl_uint, size_t, void*, size_t*))
F(cl_int, clReleaseEvent, (cl_event))

static const char* SRC =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
    "#pragma OPENCL EXTENSION cl_khr_subgroup_extended_types : enable\n"
    "#define BODY(T, CV, OP)                                                        \\\n"
    "  T a0 = CV(gid), a1 = a0 + (T)1, a2 = a0 + (T)2, a3 = a0 + (T)3;           \\\n"
    "  T a4 = a0 + (T)4, a5 = a0 + (T)5, a6 = a0 + (T)6, a7 = a0 + (T)7;          \\\n"
    "  T x = CV(seed), y = CV(seed * 0.5f);                                       \\\n"
    "  for (int i = 0; i < iters; ++i) {                                          \\\n"
    "    a0 = OP(a0, x, y); a1 = OP(a1, x, y); a2 = OP(a2, x, y); a3 = OP(a3, x, y); \\\n"
    "    a4 = OP(a4, x, y); a5 = OP(a5, x, y); a6 = OP(a6, x, y); a7 = OP(a7, x, y); \\\n"
    "  }                                                                          \\\n"
    "  T s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;                               \\\n"
    "  float t = (float)(s.x + s.y + s.z + s.w); if (t == -12345.f) out[gid] = t;\n"
    "#define MAD(a, x, y) ((a) * (x) + (y))\n"
    "__kernel void f32(__global float* out, float seed, int iters) {\n"
    "  int gid = get_global_id(0); BODY(float4, (float4), MAD) }\n"
    "__kernel void f16(__global float* out, float seed, int iters) {\n"
    "  int gid = get_global_id(0); BODY(half4, (half4), MAD) }\n"
    "__kernel void f32fma(__global float* out, float seed, int iters) {\n"
    "  int gid = get_global_id(0); BODY(float4, (float4), fma) }\n"
    "__kernel void f16fma(__global float* out, float seed, int iters) {\n"
    "  int gid = get_global_id(0); BODY(half4, (half4), fma) }\n"
    // Broadcast loop: per iteration 16 lanes' weight vectors are broadcast and
    // multiplied by 4 components of 2 activation vectors (32 vec4 FMAs).
    "#define BC(T, CV, B)                                                       \\\n"
    "  int gid = get_global_id(0); int lane = get_sub_group_local_id();          \\\n"
    "  T w = CV(lane * seed); T s0 = CV(gid * 0.001f), s1 = s0 + (T)1;             \\\n"
    "  T r0 = (T)0, r1 = (T)0, r2 = (T)0, r3 = (T)0, r4 = (T)0, r5 = (T)0, r6 = (T)0, r7 = (T)0; \\\n"
    "  for (int i = 0; i < iters; i += 4) {                                      \\\n"
    "    r0 += B(w, 0) * s0.x; r1 += B(w, 0) * s1.x; r0 += B(w, 1) * s0.y; r1 += B(w, 1) * s1.y; \\\n"
    "    r0 += B(w, 2) * s0.z; r1 += B(w, 2) * s1.z; r0 += B(w, 3) * s0.w; r1 += B(w, 3) * s1.w; \\\n"
    "    r2 += B(w, 4) * s0.x; r3 += B(w, 4) * s1.x; r2 += B(w, 5) * s0.y; r3 += B(w, 5) * s1.y; \\\n"
    "    r2 += B(w, 6) * s0.z; r3 += B(w, 6) * s1.z; r2 += B(w, 7) * s0.w; r3 += B(w, 7) * s1.w; \\\n"
    "    r4 += B(w, 8) * s0.x; r5 += B(w, 8) * s1.x; r4 += B(w, 9) * s0.y; r5 += B(w, 9) * s1.y; \\\n"
    "    r4 += B(w, 10) * s0.z; r5 += B(w, 10) * s1.z; r4 += B(w, 11) * s0.w; r5 += B(w, 11) * s1.w; \\\n"
    "    r6 += B(w, 12) * s0.x; r7 += B(w, 12) * s1.x; r6 += B(w, 13) * s0.y; r7 += B(w, 13) * s1.y; \\\n"
    "    r6 += B(w, 14) * s0.z; r7 += B(w, 14) * s1.z; r6 += B(w, 15) * s0.w; r7 += B(w, 15) * s1.w; \\\n"
    "    w += (T)0.001f; s0 += (T)0.001f; s1 -= (T)0.001f;                       \\\n"
    "  }                                                                          \\\n"
    "  T s = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;                               \\\n"
    "  float t = (float)(s.x + s.y + s.z + s.w); if (t == -12345.f) out[gid] = t;\n"
    "#define B32(w, i) sub_group_broadcast(w, i##u)\n"
    "#define B16(w, i) as_half4(sub_group_broadcast(as_long(w), i##u))\n"
    "__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void bc32(__global float* out, float seed, int iters) { BC(float4, (float4), B32) }\n"
    "__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void bc16(__global float* out, float seed, int iters) { BC(half4, (half4), B16) }\n";

int main(int argc, char** argv) {
  int groups = argc > 1 ? atoi(argv[1]) : 512, iters = argc > 2 ? atoi(argv[2]) : 4096;
  void* h = dlopen("libOpenCL.so", RTLD_NOW);
  if (!h) h = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
  if (!h) { fprintf(stderr, "no OpenCL: %s\n", dlerror()); return 1; }
#define L(n) n = dlsym(h, #n); if (!n) { fprintf(stderr, "missing %s\n", #n); return 1; }
  L(clGetPlatformIDs) L(clGetDeviceIDs) L(clGetDeviceInfo) L(clCreateContext) L(clCreateCommandQueue)
  L(clCreateProgramWithSource) L(clBuildProgram) L(clGetProgramBuildInfo) L(clCreateKernel) L(clCreateBuffer)
  L(clSetKernelArg) L(clEnqueueNDRangeKernel) L(clFinish) L(clWaitForEvents) L(clGetEventProfilingInfo)
  L(clReleaseEvent)
  cl_platform_id p; cl_device_id d; cl_int e;
  clGetPlatformIDs(1, &p, NULL);
  clGetDeviceIDs(p, 0xFFFFFFFF, 1, &d, NULL);
  char name[256] = {0}, ver[256] = {0};
  cl_uint cu = 0, mhz = 0;
  clGetDeviceInfo(d, 0x102B, sizeof name, name, NULL);   // CL_DEVICE_NAME
  clGetDeviceInfo(d, 0x102F, sizeof ver, ver, NULL);     // CL_DRIVER_VERSION
  clGetDeviceInfo(d, 0x1002, sizeof cu, &cu, NULL);      // CL_DEVICE_MAX_COMPUTE_UNITS
  clGetDeviceInfo(d, 0x100C, sizeof mhz, &mhz, NULL);    // CL_DEVICE_MAX_CLOCK_FREQUENCY
  printf("device %s, driver %s, %u CUs, %u MHz\n", name, ver, cu, mhz);
  cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
  cl_command_queue q = clCreateCommandQueue(c, d, 2 /* profiling */, &e);
  cl_program pr = clCreateProgramWithSource(c, 1, &SRC, NULL, &e);
  const char* opts = argc > 3 ? argv[3] : "-cl-std=CL2.0";
  printf("build options: %s\n", opts);
  e = clBuildProgram(pr, 1, &d, opts, NULL, NULL);
  if (e) {
    static char log[1 << 16];
    clGetProgramBuildInfo(pr, d, 0x1183, sizeof log, log, NULL);
    fprintf(stderr, "build failed %d:\n%s\n", e, log);
    return 1;
  }
  cl_mem out = clCreateBuffer(c, 1, (size_t)groups * 128 * 4, NULL, &e);
  const char* ks[] = {"f32", "f16", "f32fma", "f16fma", "bc32", "bc16"};
  for (int k = 0; k < 6; ++k) {
    cl_kernel kr = clCreateKernel(pr, ks[k], &e);
    float seed = 0.999f;
    clSetKernelArg(kr, 0, sizeof out, &out);
    clSetKernelArg(kr, 1, sizeof seed, &seed);
    clSetKernelArg(kr, 2, sizeof iters, &iters);
    size_t g = (size_t)groups * 128, l = 128;
    double best = 1e30;
    for (int r = 0; r < 6; ++r) {   // first runs warm the clock
      cl_event ev;
      clEnqueueNDRangeKernel(q, kr, 1, NULL, &g, &l, 0, NULL, &ev);
      clWaitForEvents(1, &ev);
      cl_ulong a, b;
      clGetEventProfilingInfo(ev, 0x1282, 8, &a, NULL);
      clGetEventProfilingInfo(ev, 0x1283, 8, &b, NULL);
      clReleaseEvent(ev);
      if (r >= 2 && (b - a) * 1e-6 < best) best = (b - a) * 1e-6;
    }
    // 8 vec4 FMAs per iteration (f32/f16); 32 vec4 FMAs per 4 iterations (bc*) = 8 per iteration.
    double flops = 2.0 * 4 * 8 * (double)iters * g;
    printf("%-5s %8.3f ms  %7.1f GFLOPS\n", ks[k], best, flops / best / 1e6);
  }
  return 0;
}
