// packed_file.hpp — the minimal packed-weight file the fast engine loads
// through STARLING_FAST_PACKED (#319).
//
// File format ("SFPK", little-endian):
//   char     magic[4]      "SFPK"
//   u32      version       1
//   char     source[65]    sha256 hex of the source weights, NUL-terminated
//   char     rounding[16]  "rtn" | "imatrix", NUL-terminated
//   u32      n_tensors
//   per tensor:
//     u32  name_len, char name[]           (GGUF tensor name)
//     u32  spec_len,  char spec[]          (canonical layout spec)
//     u32  N, K
//     u64  code_bytes,  code bytes   (padded to a multiple of 4)
//     u64  scale_bytes, scale bytes  (padded to a multiple of 4)
//     u64  super_words, super data   (N * u16 as words; 0 unless u8super)
//     the three blobs, each padded to 4 bytes
//
// This is deliberately the smallest thing that lets a candidate layout run
// (#317): the proper manifest, provenance records and on-disk cache are
// #315/#318. Rows are stored in GGUF row order — the engine's interleave /
// qkv-concat / chunking transforms run on the loaded HostMatrix exactly as
// they do on repacked GGUF data.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "layout.hpp"
#include "weights.hpp"

namespace starling::fast {

struct PackedTensor {
    std::string name;
    LayoutDesc desc;
    uint32_t N = 0, K = 0;
    std::vector<uint8_t> codes, scales;
    std::vector<uint16_t> super;
};

class PackedWeights {
public:
    // Loads and validates the whole file (structure, descriptor validity,
    // blob sizes). Returns nullptr with a reason in err on failure.
    static std::unique_ptr<PackedWeights> load(const std::string& path, std::string& err);

    const std::string& source_hash() const { return source_; }
    const std::string& rounding() const { return rounding_; }

    // True when the file contains a tensor with this name.
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    size_t size() const { return tensors_.size(); }
    const PackedTensor* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (const auto& kv : tensors_) fn(kv.second);
    }

    // Materialize a tensor as a HostMatrix ready for the arena, i.e. the
    // same bytes pack_gpu_matrix would have produced for a legacy-shaped
    // descriptor (w4g32asym / w8g16sym map to the existing kernels;
    // anything else fails with "no kernel support" until #317 adds the
    // variant). The matrix carries the descriptor in `layout` so later
    // kernels can dispatch on it.
    bool matrix(const std::string& name, HostMatrix& out, std::string& err) const;

    // Sum of packed bytes (for load reporting).
    uint64_t bytes() const;

private:
    std::string source_, rounding_;
    std::unordered_map<std::string, PackedTensor> tensors_;
};

} // namespace starling::fast
