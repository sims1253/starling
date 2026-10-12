// LiteRT CompiledModel bench for the GPU accelerator (ML Drift), derived from
// benchmarks/npu/npu_bench.cc: load a .tflite, compile it for the requested
// accelerator (gpu|cpu), report whether it is fully accelerated, run it N
// times on the given raw inputs (zeros when an input file is missing or "-")
// and dump the outputs.
//
// litert_gpu_bench MODEL ACCEL LIBDIR RUNS IN0 [IN1 ...] -- OUT_PREFIX
//
// LITERT_GPU_TOML  GPU accelerator options as TOML, ';' separates lines, e.g.
//                  "precision = 1; backend = 1" (precision 1 = fp16, 2 = fp32;
//                  backend 1 = OpenCL, 2 = WebGPU, 3 = OpenGL; see litert_common.h)
// LITERT_LOG       min logger severity (0 = verbose)
// When the OpenCL interposer (clshim) is the libOpenCL.so in LD_LIBRARY_PATH,
// its per-kernel GPU timing is printed for the warm-up and the steady runs.
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

#define CK(x)                                                             \
  do {                                                                    \
    LiteRtStatus s_ = (x);                                                \
    if (s_ != kLiteRtStatusOk) {                                          \
      std::fprintf(stderr, "FAIL %s -> status %d (line %d)\n", #x, (int)s_, \
                   __LINE__);                                             \
      return 1;                                                           \
    }                                                                     \
  } while (0)

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s MODEL npu|cpu|gpu LIBDIR RUNS IN.f32... -- OUTPREFIX\n", argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  std::string accel = argv[2];
  const char* libdir = argv[3];
  int runs = std::atoi(argv[4]);
  std::vector<std::string> ins;
  std::string outprefix;
  for (int i = 5; i < argc; ++i) {
    if (std::string(argv[i]) == "--") { if (i + 1 < argc) outprefix = argv[i + 1]; break; }
    ins.push_back(argv[i]);
  }

  std::vector<LiteRtEnvOption> eo;
  auto add_str = [&](LiteRtEnvOptionTag t, const char* v) {
    LiteRtEnvOption o; o.tag = t; o.value.type = kLiteRtAnyTypeString; o.value.str_value = v; eo.push_back(o);
  };
  add_str(kLiteRtEnvOptionTagDispatchLibraryDir, libdir);
  add_str(kLiteRtEnvOptionTagCompilerPluginLibraryDir, libdir);
  add_str(kLiteRtEnvOptionTagRuntimeLibraryDir, libdir);
  {
    LiteRtEnvOption o; o.tag = kLiteRtEnvOptionTagMinLoggerSeverity; o.value.type = kLiteRtAnyTypeInt;
    o.value.int_value = std::getenv("LITERT_LOG") ? std::atoi(std::getenv("LITERT_LOG")) : 1;
    eo.push_back(o);
  }
  double t0 = now_ms();
  LiteRtEnvironment env;
  CK(LiteRtCreateEnvironment((int)eo.size(), eo.data(), &env));
  LiteRtModel model;
  CK(LiteRtCreateModelFromFile(env, model_path, &model));
  double t_load = now_ms() - t0;

  LiteRtOptions opts;
  CK(LiteRtCreateOptions(&opts));
  LiteRtHwAcceleratorSet hw = kLiteRtHwAcceleratorCpu;
  if (accel == "npu") hw = kLiteRtHwAcceleratorNpu;
  else if (accel == "npu+cpu") hw = kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu;
  else if (accel == "gpu") hw = kLiteRtHwAcceleratorGpu;
  else if (accel == "gpu+cpu") hw = kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu;
  CK(LiteRtSetOptionsHardwareAccelerators(opts, hw));
  if (const char* toml = std::getenv("LITERT_GPU_TOML")) {
    std::string t = toml;
    for (char& c : t) if (c == ';') c = '\n';
    t += "\n";
    char* payload = strdup(t.c_str());
    LiteRtOpaqueOptions oo;
    CK(LiteRtCreateOpaqueOptions("gpu_options", payload, [](void* p) { free(p); }, &oo));
    CK(LiteRtAddOpaqueOptions(opts, oo));
    std::printf("gpu_options: %s\n", toml);
  }
  // clshim hook (only present when the interposer is loaded).
  typedef void (*mark_fn)(const char*, int);
  mark_fn mark = nullptr;
  if (void* h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_NOLOAD)) mark = (mark_fn)dlsym(h, "clshim_mark");
  t0 = now_ms();
  LiteRtCompiledModel cm;
  CK(LiteRtCreateCompiledModel(env, model, opts, &cm));
  double t_compile = now_ms() - t0;
  bool full = false, noncpu = false;
  LiteRtCompiledModelIsFullyAccelerated(cm, &full);
  LiteRtCompiledModelIsNonCpuFullyAccelerated(cm, &noncpu);
  std::printf("load %.1f ms, compile(%s) %.1f ms, fully_accelerated=%d non_cpu_fully_accelerated=%d\n",
              t_load, accel.c_str(), t_compile, (int)full, (int)noncpu);

  LiteRtSignature sig;
  CK(LiteRtGetModelSignature(model, 0, &sig));
  LiteRtParamIndex nin, nout;
  CK(LiteRtGetNumSignatureInputs(sig, &nin));
  CK(LiteRtGetNumSignatureOutputs(sig, &nout));
  std::vector<LiteRtTensorBuffer> ib(nin), ob(nout);
  for (LiteRtParamIndex i = 0; i < nin; ++i) {
    LiteRtTensor t; CK(LiteRtGetSignatureInputTensorByIndex(sig, i, &t));
    LiteRtRankedTensorType tt; CK(LiteRtGetRankedTensorType(t, &tt));
    LiteRtTensorBufferRequirements req;
    CK(LiteRtGetCompiledModelInputBufferRequirements(cm, 0, i, &req));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, req, &ib[i]));
    size_t sz; CK(LiteRtGetTensorBufferPackedSize(ib[i], &sz));
    LiteRtTensorBufferType bt; CK(LiteRtGetTensorBufferType(ib[i], &bt));
    std::printf("input %u: elemtype %d rank %u bytes %zu buftype %d\n", (unsigned)i, (int)tt.element_type,
                (unsigned)tt.layout.rank, sz, (int)bt);
    void* p; CK(LiteRtLockTensorBuffer(ib[i], &p, kLiteRtTensorBufferLockModeWrite));
    if (i < ins.size() && ins[i] != "-") {
      FILE* f = std::fopen(ins[i].c_str(), "rb");
      if (!f) { std::fprintf(stderr, "cannot open %s\n", ins[i].c_str()); return 1; }
      size_t got = std::fread(p, 1, sz, f);
      std::fclose(f);
      if (got != sz) { std::fprintf(stderr, "%s: read %zu of %zu bytes\n", ins[i].c_str(), got, sz); return 1; }
    } else {
      std::memset(p, 0, sz);
    }
    CK(LiteRtUnlockTensorBuffer(ib[i]));
  }
  for (LiteRtParamIndex i = 0; i < nout; ++i) {
    LiteRtTensor t; CK(LiteRtGetSignatureOutputTensorByIndex(sig, i, &t));
    LiteRtRankedTensorType tt; CK(LiteRtGetRankedTensorType(t, &tt));
    LiteRtTensorBufferRequirements req;
    CK(LiteRtGetCompiledModelOutputBufferRequirements(cm, 0, i, &req));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, req, &ob[i]));
    size_t sz; CK(LiteRtGetTensorBufferPackedSize(ob[i], &sz));
    LiteRtTensorBufferType bt; CK(LiteRtGetTensorBufferType(ob[i], &bt));
    std::printf("output %u: elemtype %d bytes %zu buftype %d\n", (unsigned)i, (int)tt.element_type, sz, (int)bt);
  }

  std::vector<double> ts;
  auto epoch = [] { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + t.tv_nsec * 1e-9; };
  double e0 = 0;
  for (int r = 0; r < runs + 1; ++r) {
    if (r == 1) e0 = epoch();
    double a = now_ms();
    CK(LiteRtRunCompiledModel(cm, 0, nin, ib.data(), nout, ob.data()));
    // Force completion: read the output back like a consumer would.
    void* p; CK(LiteRtLockTensorBuffer(ob[0], &p, kLiteRtTensorBufferLockModeRead));
    CK(LiteRtUnlockTensorBuffer(ob[0]));
    double b = now_ms() - a;
    if (r == 0) {
      std::printf("first run %.2f ms\n", b);
      if (!mark) {
        if (void* h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_NOLOAD)) mark = (mark_fn)dlsym(h, "clshim_mark");
      }
      if (mark) mark("compile+first run", 1);
    } else {
      ts.push_back(b);
    }
    if (const char* gap = std::getenv("NPU_GAP_MS")) {
      double g = std::atof(gap), e = now_ms() + g;
      while (now_ms() < e) {}
    }
  }
  std::printf("loop_epoch %.3f %.3f\n", e0, epoch());
  if (mark && runs > 0) mark("steady runs", runs);
  if (!ts.empty()) {
    std::vector<double> s = ts;
    std::sort(s.begin(), s.end());
    double sum = 0; for (double v : ts) sum += v;
    std::printf("runs %zu: median %.2f ms, min %.2f, max %.2f, mean %.2f\n", s.size(), s[s.size() / 2], s[0],
                s.back(), sum / s.size());
  }
  if (!outprefix.empty()) {
    for (LiteRtParamIndex i = 0; i < nout; ++i) {
      size_t sz; CK(LiteRtGetTensorBufferPackedSize(ob[i], &sz));
      void* p; CK(LiteRtLockTensorBuffer(ob[i], &p, kLiteRtTensorBufferLockModeRead));
      std::string fn = outprefix + "_out" + std::to_string(i) + ".f32";
      FILE* f = std::fopen(fn.c_str(), "wb"); std::fwrite(p, 1, sz, f); std::fclose(f);
      CK(LiteRtUnlockTensorBuffer(ob[i]));
    }
  }
  for (auto b : ib) LiteRtDestroyTensorBuffer(b);
  for (auto b : ob) LiteRtDestroyTensorBuffer(b);
  LiteRtDestroyCompiledModel(cm);
  LiteRtDestroyOptions(opts);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env);
  return 0;
}
