// graph_helpers.hpp — ggml graph-builder micro-helpers. Two numeric
// disciplines share this header via explicit suffixes:
//   *_bf16 — the bf16 oracle (moss/ark/higgs): activations live in bf16,
//            elementwise math in f32, rounding at the bf16 boundary.
//   *_f32  — the f32 discipline (hojo audio tower / conformer).
// The short names (bf/ff/addb/mulb/...) exist for the llm.cpp call sites;
// all loader-coupled helpers take ModelLoader& so every engine can use them.
#pragma once

#include "runtime/backend.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace starling::ggml::lib {

// ---- casts --------------------------------------------------------------- //
inline ggml_tensor* bf16(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_BF16 ? x : ggml_cast(c, x, GGML_TYPE_BF16);
}
inline ggml_tensor* f32(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
// llm.cpp-family short names.
inline ggml_tensor* bf(ggml_context* c, ggml_tensor* x) { return bf16(c, x); }
inline ggml_tensor* ff(ggml_context* c, ggml_tensor* x) { return f32(c, x); }

// ---- weight lookup ------------------------------------------------------- //
inline ggml_tensor* weight(ggml_context* c, const ModelLoader& ml, const std::string& n) {
    return clone_weight(c, ml, n.c_str());
}
// llm.cpp-family short name (full tensor name, no suffix appended).
inline ggml_tensor* wb(ggml_context* c, const ModelLoader& ml, const std::string& n) {
    return weight(c, ml, n);
}

// ---- linears -------------------------------------------------------------- //
// Activation-side cast for a weight GEMM. The bf16 oracle keeps activations
// bf16 into the GEMM, but ggml's quantized matmul (Q8 intermediates) only
// accepts F32 operands — and bf16 values are exact in f32, so the upcast
// preserves the F32-accumulated GEMM + trailing bf16 round; only the weight
// carries quantization noise. No behavior change for unquantized weights.
inline ggml_tensor* gemm_act(ggml_context* c, const ggml_tensor* w, ggml_tensor* x) {
    return ggml_is_quantized(w->type) ? f32(c, x) : bf16(c, x);
}
// bf16 linear without bias (Qwen3 attention projections; full weight name).
inline ggml_tensor* lin(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n) {
    ggml_tensor* w = weight(c, ml, n);
    return bf16(c, ggml_mul_mat(c, w, gemm_act(c, w, x)));
}
// bf16 linear with optional bias (n is the BASE name: n + ".weight"/".bias").
// nn.Linear in the BF16 oracle: GEMM (+ bias) exposes F32, rounds at the
// bf16 boundary.
inline ggml_tensor* linear_bf16(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                                const std::string& n, bool bias) {
    ggml_tensor* w = weight(c, ml, n + ".weight");
    ggml_tensor* y = ggml_mul_mat(c, w, gemm_act(c, w, x));
    if (bias) y = ggml_add(c, f32(c, y), f32(c, weight(c, ml, n + ".bias")));
    return bf16(c, y);
}
// f32 linear with optional bias (hojo audio discipline).
inline ggml_tensor* linear_f32(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                               const std::string& n, bool bias) {
    ggml_tensor* y = ggml_mul_mat(c, weight(c, ml, n + ".weight"), f32(c, x));
    if (bias) y = ggml_add(c, f32(c, y), f32(c, weight(c, ml, n + ".bias")));
    return f32(c, y);
}

// ---- elementwise (f32 math, bf16 result) --------------------------------- //
inline ggml_tensor* addb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf16(c, ggml_add(c, f32(c, a), f32(c, b)));
}
inline ggml_tensor* mulb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf16(c, ggml_mul(c, f32(c, a), f32(c, b)));
}

// ---- norms / activations -------------------------------------------------- //
// RMSNorm in f32, then scale by the bf16 weight (Qwen3: no RMSNorm bias).
inline ggml_tensor* rms(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    ggml_tensor* y = ggml_rms_norm(c, f32(c, x), eps);
    y = bf16(c, y);
    return mulb(c, y, bf16(c, weight(c, ml, n)));
}
// F.rms_norm semantics (Nemotron): normalize AND affine in f32, ONE bf16
// round at the end — vs rms's Llama-style round after the rsqrt.
inline ggml_tensor* rms_single(ggml_context* c, const ModelLoader& ml,
                               ggml_tensor* x, const std::string& n, float eps) {
    ggml_tensor* y = ggml_rms_norm(c, f32(c, x), eps);
    y = ggml_mul(c, y, f32(c, weight(c, ml, n)));
    return bf16(c, y);
}
// Exact (erf) GELU under both disciplines.
inline ggml_tensor* gelu_erf_bf16(ggml_context* c, ggml_tensor* x) {
    return bf16(c, ggml_gelu_erf(c, f32(c, x)));
}
inline ggml_tensor* gelu_erf_f32(ggml_context* c, ggml_tensor* x) {
    return f32(c, ggml_gelu_erf(c, f32(c, x)));
}
// PyTorch LayerNorm: F32 reduction + affine. bf16 or f32 result.
inline ggml_tensor* layer_norm_bf16(ggml_context* c, const ModelLoader& ml,
                                    ggml_tensor* x, const std::string& n, float eps) {
    ggml_tensor* y = ggml_norm(c, f32(c, x), eps);
    y = ggml_mul(c, y, f32(c, weight(c, ml, n + ".weight")));
    y = ggml_add(c, y, f32(c, weight(c, ml, n + ".bias")));
    return bf16(c, y);
}
inline ggml_tensor* layer_norm_f32(ggml_context* c, const ModelLoader& ml,
                                   ggml_tensor* x, const std::string& n, float eps) {
    ggml_tensor* y = ggml_norm(c, f32(c, x), eps);
    y = ggml_mul(c, y, f32(c, weight(c, ml, n + ".weight")));
    y = ggml_add(c, y, f32(c, weight(c, ml, n + ".bias")));
    return f32(c, y);
}

// ---- host helpers --------------------------------------------------------- //
inline std::vector<ggml_bf16_t> tobf(const std::vector<float>& x) {
    std::vector<ggml_bf16_t> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = ggml_fp32_to_bf16(x[i]);
    return r;
}
// Argmax over a host logits vector (first index on ties).
inline int32_t argmax_low(const std::vector<float>& x) {
    int32_t best = 0;
    for (int32_t i = 1; i < (int32_t) x.size(); ++i)
        if (x[i] > x[best]) best = i;
    return best;
}
// Read a weight's f32 contents to host (realizes weights if needed; returns
// {} when the tensor is absent). F32 fast path + BF16 convert.
inline std::vector<float> read_f32(const ModelLoader& ml, const char* name) {
    ggml_tensor* t = ml.tensor(name);
    if (!t) return {};
    ensure_weights_realized(ml);
    size_t n = (size_t) ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_bf16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_bf16_to_fp32(raw[i]);
    }
    return out;
}
// Env-gated debug flag (STARLING_<MODEL>_DEBUG == "1").
inline bool debug_enabled(const char* env) {
    const char* p = std::getenv(env);
    return p && std::strcmp(p, "1") == 0;
}

} // namespace starling::ggml::lib
