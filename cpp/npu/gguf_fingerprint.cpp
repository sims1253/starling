// Copyright 2026 The Starling Authors. Apache-2.0.
//
// Minimal GGUF reader for npu_tool: the model fingerprint of
// encoder_contract.hpp, computed from the file (the engine computes the same
// from its loaded tensors). Only F32 tensors are accepted.
#include "gguf_fingerprint.hpp"

#include "encoder_contract.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

namespace starling::npu {

namespace {

struct Reader {
    FILE* f;
    bool ok = true;
    template <typename T>
    T get() {
        T v{};
        if (std::fread(&v, sizeof v, 1, f) != 1) ok = false;
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (!ok || n > (1u << 20)) { ok = false; return ""; }
        std::string s(n, '\0');
        if (n && std::fread(&s[0], 1, n, f) != n) ok = false;
        return s;
    }
    void skip(long n) { if (std::fseek(f, n, SEEK_CUR) != 0) ok = false; }
};

// GGUF value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool,
// 8 string, 9 array, 10 u64, 11 i64, 12 f64.
long scalar_size(uint32_t t) {
    switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return -1;
    }
}

bool skip_value(Reader& r, uint32_t t, uint32_t* u32_out) {
    if (t == 8) { r.str(); return r.ok; }
    if (t == 9) {
        uint32_t et = r.get<uint32_t>();
        uint64_t n = r.get<uint64_t>();
        for (uint64_t i = 0; i < n && r.ok; ++i)
            if (!skip_value(r, et, nullptr)) return false;
        return r.ok;
    }
    long s = scalar_size(t);
    if (s < 0) return false;
    if (t == 4 && u32_out) { *u32_out = r.get<uint32_t>(); return r.ok; }
    r.skip(s);
    return r.ok;
}

}  // namespace

bool gguf_fingerprint(const std::string& path, uint64_t& hash, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    Reader r{f};
    char magic[4];
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "GGUF", 4) == 0;
    uint32_t version = r.get<uint32_t>();
    uint64_t n_tensors = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
    if (!ok || !r.ok || version < 2) { std::fclose(f); err = "not a GGUF v2+ file: " + path; return false; }
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv && r.ok; ++i) {
        std::string k = r.str();
        uint32_t t = r.get<uint32_t>();
        uint32_t v = 0;
        if (!skip_value(r, t, k == "general.alignment" ? &v : nullptr)) { r.ok = false; break; }
        if (k == "general.alignment" && v) alignment = v;
    }
    struct Info { uint32_t type; uint64_t n; uint64_t off; };
    std::map<std::string, Info> infos;
    for (uint64_t i = 0; i < n_tensors && r.ok; ++i) {
        std::string name = r.str();
        uint32_t nd = r.get<uint32_t>();
        uint64_t n = 1;
        for (uint32_t d = 0; d < nd && r.ok; ++d) n *= r.get<uint64_t>();
        uint32_t type = r.get<uint32_t>();
        uint64_t off = r.get<uint64_t>();
        infos[name] = {type, n, off};
    }
    if (!r.ok) { std::fclose(f); err = "truncated GGUF header: " + path; return false; }
    long pos = std::ftell(f);
    const uint64_t data0 = ((uint64_t)pos + alignment - 1) / alignment * alignment;
    hash = kFnvBasis;
    for (const char* name : kQwen3EncFingerprintTensors) {
        auto it = infos.find(name);
        if (it == infos.end()) { std::fclose(f); err = std::string("GGUF lacks ") + name; return false; }
        if (it->second.type != 0) { std::fclose(f); err = std::string(name) + " is not F32"; return false; }
        std::vector<uint8_t> b(it->second.n * 4);
        if (std::fseek(f, (long)(data0 + it->second.off), SEEK_SET) != 0 ||
            std::fread(b.data(), 1, b.size(), f) != b.size()) {
            std::fclose(f);
            err = std::string("cannot read ") + name;
            return false;
        }
        hash = fnv1a_update(hash, b.data(), b.size());
    }
    std::fclose(f);
    return true;
}

}  // namespace starling::npu
