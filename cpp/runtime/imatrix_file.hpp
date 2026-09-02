// imatrix_file.hpp — the on-disk importance-matrix format shared by the
// runtime collector (cpp/runtime/imatrix.cpp, written) and the quantizer CLI
// (cpp/tools/starling_quantize.cpp, read).
//
// Format (little-endian):
//   char magic[8]  = "STLGIMX1"
//   u32  version   = 1
//   u32  n_entries
//   per entry:
//     u32  name_len
//     char name[name_len]
//     u32  n_vals           (== n_per_row of the collected weight)
//     u64  n_obs            (mul_mat observations accumulated into this entry)
//     f32  values[n_vals]   (raw sum of squared activations per input channel)
//
// The values are RAW sums (not averaged): quantization only consumes their
// relative magnitudes per channel, and raw sums accumulate across any number
// of observed graphs without needing the observation count up front.

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml {

inline constexpr char kImatrixMagic[8] = {'S', 'T', 'L', 'G', 'I', 'M', 'X', '1'};
inline constexpr uint32_t kImatrixVersion = 1;

struct ImatrixEntry {
    std::vector<float> values;  // per-input-channel sum of squared activations
    uint64_t n_obs = 0;         // number of mul_mat nodes accumulated
};

using ImatrixMap = std::unordered_map<std::string, ImatrixEntry>;

inline bool imatrix_write(const std::string& path, const ImatrixMap& map) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(kImatrixMagic, sizeof(kImatrixMagic));
    uint32_t ver = kImatrixVersion, n = (uint32_t)map.size();
    f.write((const char*)&ver, sizeof(ver));
    f.write((const char*)&n, sizeof(n));
    for (const auto& kv : map) {
        uint32_t name_len = (uint32_t)kv.first.size();
        f.write((const char*)&name_len, sizeof(name_len));
        f.write(kv.first.data(), name_len);
        uint32_t n_vals = (uint32_t)kv.second.values.size();
        f.write((const char*)&n_vals, sizeof(n_vals));
        f.write((const char*)&kv.second.n_obs, sizeof(kv.second.n_obs));
        f.write((const char*)kv.second.values.data(),
                (std::streamsize)n_vals * sizeof(float));
    }
    return (bool)f;
}

inline ImatrixMap imatrix_read(const std::string& path) {
    ImatrixMap map;
    std::ifstream f(path, std::ios::binary);
    if (!f) return map;
    char magic[8];
    f.read(magic, sizeof(magic));
    if (f.gcount() != 8 || std::string(magic, 8) != std::string(kImatrixMagic, 8)) {
        std::fprintf(stderr, "imatrix_read: bad magic in %s\n", path.c_str());
        return {};
    }
    uint32_t ver = 0, n = 0;
    f.read((char*)&ver, sizeof(ver));
    f.read((char*)&n, sizeof(n));
    if (ver != kImatrixVersion) {
        std::fprintf(stderr, "imatrix_read: unsupported version %u in %s\n", ver, path.c_str());
        return {};
    }
    for (uint32_t i = 0; i < n && f; ++i) {
        uint32_t name_len = 0;
        f.read((char*)&name_len, sizeof(name_len));
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);
        uint32_t n_vals = 0;
        f.read((char*)&n_vals, sizeof(n_vals));
        ImatrixEntry e;
        e.values.resize(n_vals);
        f.read((char*)&e.n_obs, sizeof(e.n_obs));
        f.read((char*)e.values.data(), (std::streamsize)n_vals * sizeof(float));
        if (f) map.emplace(std::move(name), std::move(e));
    }
    return map;
}

} // namespace starling::ggml
