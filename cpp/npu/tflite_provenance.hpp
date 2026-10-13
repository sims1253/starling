// Copyright 2026 The Starling Authors. Apache-2.0.
//
// The provenance export_encoder.py stamps into a .tflite (encoder_contract.hpp,
// kProvenanceName): which GGUF's model hash and which contract key the graph
// was exported for. starling-npu-tool compile reads it and refuses a graph
// stamped for another model than the --hash it is told to build. Plain C++ over
// the flatbuffer bytes, bounds-checked (host-tested in npu_provenance_test).
//
// TFLite schema fields read: Model.{buffers = 4, metadata = 6},
// Metadata.{name = 0, buffer = 1}, Buffer.{data = 0, offset = 1, size = 2}.
#pragma once

#include "encoder_contract.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace starling::npu {

struct TfliteProvenance {
    uint64_t model_hash = 0;
    std::string key;
};

namespace tflite_detail {

struct Reader {
    const uint8_t* p;
    size_t n;
    bool ok = true;

    uint32_t u32(uint64_t o) {
        if (o > n || n - o < 4) return ok = false, 0;
        uint32_t v;
        std::memcpy(&v, p + o, 4);
        return v;
    }
    uint16_t u16(uint64_t o) {
        if (o > n || n - o < 2) return ok = false, 0;
        uint16_t v;
        std::memcpy(&v, p + o, 2);
        return v;
    }
    // Position of field i of the table at t, or 0 when absent.
    uint64_t field(uint64_t t, unsigned i) {
        const int64_t vt = (int64_t)t - (int32_t)u32(t);
        if (!ok || vt < 0) return ok = false, 0;
        if (4 + 2 * i >= u16((uint64_t)vt)) return 0;
        const uint16_t o = u16((uint64_t)vt + 4 + 2 * i);
        return o ? t + o : 0;
    }
    // Target of the uoffset stored at p.
    uint64_t deref(uint64_t at) { return at + u32(at); }
    // (count, first element) of the vector the field at `at` points to.
    uint64_t vec(uint64_t at, uint64_t& first) {
        const uint64_t v = deref(at);
        const uint64_t count = u32(v);
        first = v + 4;
        if (!ok || first > n || count > (n - first) / 4) return ok = false, 0;
        return count;
    }
};

}  // namespace tflite_detail

// Reads the provenance stamp of `tflite` (n bytes). False with `err` when the
// file has none, more than one, or a malformed one.
inline bool read_tflite_provenance(const void* tflite, size_t n, TfliteProvenance& out, std::string& err) {
    tflite_detail::Reader r{static_cast<const uint8_t*>(tflite), n};
    if (n < 8 || std::memcmp(r.p + 4, "TFL3", 4) != 0) {
        err = "not a .tflite";
        return false;
    }
    const uint64_t root = r.deref(0);
    const uint64_t md = r.field(root, 6), bf = r.field(root, 4);
    uint64_t md0 = 0, bf0 = 0;
    const uint64_t n_md = md ? r.vec(md, md0) : 0, n_bf = bf ? r.vec(bf, bf0) : 0;
    std::string text;
    int found = 0;
    for (uint64_t k = 0; r.ok && k < n_md; ++k) {
        const uint64_t m = r.deref(md0 + 4 * k);
        const uint64_t nm = r.field(m, 0);
        if (!r.ok || !nm) continue;
        const uint64_t s = r.deref(nm);
        const uint32_t len = r.u32(s);
        if (!r.ok || s + 4 > n || len > n - s - 4) return err = "malformed .tflite", false;
        if (std::string(reinterpret_cast<const char*>(r.p + s + 4), len) != kProvenanceName) continue;
        if (++found > 1) break;
        const uint64_t bi = r.field(m, 1);
        const uint32_t idx = bi ? r.u32(bi) : 0;
        if (!r.ok || idx >= n_bf) return err = "provenance names a missing buffer", false;
        const uint64_t b = r.deref(bf0 + 4ull * idx);
        const uint64_t data = r.field(b, 0);
        uint64_t d0 = 0, dn = 0;
        if (data) {
            const uint64_t v = r.deref(data);
            dn = r.u32(v);
            d0 = v + 4;
        }
        if (!r.ok || d0 > n || dn > n - d0) return err = "malformed .tflite", false;
        text.assign(reinterpret_cast<const char*>(r.p + d0), dn);
    }
    if (!r.ok) return err = "malformed .tflite", false;
    if (found != 1) {
        err = found ? "more than one provenance stamp" : "no provenance stamp (not exported by this export_encoder.py)";
        return false;
    }
    // "model_hash=0x<hex>\nkey=<key>\n"
    const std::string h = "model_hash=", k = "\nkey=";
    const size_t kp = text.find(k);
    if (text.rfind(h, 0) != 0 || kp == std::string::npos || text.empty() || text.back() != '\n') {
        err = "malformed provenance stamp";
        return false;
    }
    const std::string hex = text.substr(h.size(), kp - h.size());
    char* end = nullptr;
    out.model_hash = std::strtoull(hex.c_str(), &end, 16);
    if (hex.size() != 18 || hex.compare(0, 2, "0x") != 0 || !end || *end) {
        err = "malformed provenance stamp";
        return false;
    }
    out.key = text.substr(kp + k.size(), text.size() - kp - k.size() - 1);
    return true;
}

}  // namespace starling::npu
