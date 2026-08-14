// loader_kit.hpp — shared GGUF-metadata helpers for the model loaders. The
// u32 guard rejects kv_int values before casting when they would wrap in the
// uint32_t field or in the int/int32_t narrowing every consumer does.
// check_gguf_header folds the per-loader architecture / numeric-profile /
// format-version / non-empty guard block; only arch string, message label,
// and the accepted profiles differ between models.
#pragma once

#include "runtime/model_loader.hpp"
#include <cstdint>
#include <initializer_list>
#include <string>

namespace starling::ggml::lib {

// kv_int returns int64_t but the config fields are uint32_t and consumers
// (device caches, loop bounds, ggml dims) narrow them to int/int32_t. Reject
// values that would wrap in either narrowing (negative or > INT32_MAX)
// BEFORE casting; the per-loader POS() checks still enforce positivity of
// accepted values.
inline bool u32(const ModelLoader& m, const char* k, uint32_t d, uint32_t& out,
                std::string& err) {
    int64_t v;
    if (!m.kv_int(k, v)) { out = d; return true; }
    if (v < 0 || v > (int64_t) INT32_MAX) {
        err = std::string("GGUF ") + k + " out of int32 range: " + std::to_string(v);
        return false;
    }
    out = (uint32_t) v;
    return true;
}

inline float f32(const ModelLoader& m, const char* k, float d) {
    double v;
    return m.kv_float(k, v) ? (float) v : d;
}

inline double f64(const ModelLoader& m, const char* k, double d) {
    double v;
    return m.kv_float(k, v) ? v : d;
}

inline std::string str(const ModelLoader& m, const char* k, const char* d) {
    std::string v;
    return m.kv_str(k, v) ? v : d;
}

// label == nullptr keeps the moss-style message; else "<label> GGUF missing
// required tensor: n".
inline bool require(const ModelLoader& m, const std::string& n, const char* label,
                    std::string& err) {
    if (m.tensor(n.c_str())) return true;
    err = label ? std::string(label) + " GGUF missing required tensor: " + n
                : "GGUF missing required tensor: " + n;
    return false;
}

// Shared untrusted-GGUF header validation. `profiles` lists the accepted
// starling.numeric_profile values; format_version must be 1 (the only
// supported Starling GGUF format).
inline bool check_gguf_header(const ModelLoader& m, const char* arch, const char* label,
                              std::initializer_list<const char*> profiles,
                              std::string& err) {
    if (std::string a; m.kv_str("general.architecture", a) && a != arch) {
        err = std::string("unsupported ") + label + " GGUF architecture: " + a;
        return false;
    }
    if (std::string p; m.kv_str("starling.numeric_profile", p)) {
        bool ok = false;
        for (const char* want : profiles)
            if (p == want) { ok = true; break; }
        if (!ok) {
            err = std::string("unsupported ") + label + " numeric profile: " + p;
            return false;
        }
    }
    if (int64_t fv; m.kv_int("starling.format_version", fv) && fv != 1) {
        err = "unsupported Starling GGUF format version: " + std::to_string(fv);
        return false;
    }
    if (m.tensor_names().empty()) {
        err = std::string(label) + " GGUF contains no tensors";
        return false;
    }
    return true;
}

} // namespace starling::ggml::lib
