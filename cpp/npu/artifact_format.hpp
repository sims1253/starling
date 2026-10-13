// Copyright 2026 The Starling Authors. Apache-2.0.
//
// The NPU artifact file (.stnpu): a 4 KiB header followed by the TPU bytecode
// at `bytecode_offset`. Header parsing and bounds checks are plain C++ (no
// Android API) so they are unit-tested on the host (npu_artifact_test).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace starling::npu {

struct ArtifactInfo {
    std::string function;                  // signature key the bytecode exports
    std::string key;                       // what the graph computes (model family + shapes), checked by the loader
    std::vector<uint64_t> in_bytes, out_bytes;
    std::string vendor;                    // thrGetVendorApiVersion() at compile time
    uint64_t model_hash = 0;               // fingerprint of the model it belongs to (see gguf_fingerprint)
    uint64_t tflite_hash = 0;              // FNV-1a of the compiled .tflite (provenance)
    uint64_t bytecode_offset = 0, bytecode_size = 0;
};

inline constexpr char kArtifactMagic[8] = {'S', 'T', 'N', 'P', 'U', 'v', '3', '\0'};
inline constexpr uint64_t kArtifactHeaderBytes = 4096;   // the bytecode starts here
inline constexpr uint32_t kArtifactMaxIo = 64;
// I/O buffers are AHardwareBuffer BLOBs, whose width is 32-bit.
inline constexpr uint64_t kArtifactMaxIoBytes = 0xffffffffull;

struct ArtifactHeader {
    char magic[8];
    uint32_t n_in, n_out;
    uint64_t in_bytes[kArtifactMaxIo], out_bytes[kArtifactMaxIo];
    char function[64];
    char vendor[64];
    char key[256];
    uint64_t model_hash, tflite_hash;
    uint64_t bytecode_offset, bytecode_size;
};
static_assert(sizeof(ArtifactHeader) < kArtifactHeaderBytes, "header fits");

// Parses the first `head_len` bytes of an artifact whose file is `file_size`
// bytes long. Rejects anything the loader cannot use safely: a bad magic, too
// many I/Os, empty or oversized I/O buffers, and bytecode that is empty, starts
// inside the header or does not lie entirely within the file (a truncated
// artifact would otherwise be mapped and read past its end by the firmware).
inline bool parse_artifact_header(const void* head, size_t head_len, uint64_t file_size, ArtifactInfo& info,
                                  std::string& err) {
    ArtifactHeader h{};
    if (head_len < sizeof h) {
        err = "artifact header truncated";
        return false;
    }
    std::memcpy(&h, head, sizeof h);
    if (std::memcmp(h.magic, kArtifactMagic, sizeof h.magic) != 0) {
        err = "not a Starling NPU artifact (bad magic)";
        return false;
    }
    if (h.n_in > kArtifactMaxIo || h.n_out > kArtifactMaxIo) {
        err = "artifact header: too many inputs/outputs";
        return false;
    }
    for (uint32_t i = 0; i < h.n_in + h.n_out; ++i) {
        const uint64_t b = i < h.n_in ? h.in_bytes[i] : h.out_bytes[i - h.n_in];
        if (b == 0 || b > kArtifactMaxIoBytes) {
            err = "artifact header: bad I/O buffer size " + std::to_string(b);
            return false;
        }
    }
    // Overflow-free: offset + size <= file_size  <=>  size <= file_size - offset.
    if (h.bytecode_offset < kArtifactHeaderBytes || h.bytecode_size == 0 || h.bytecode_offset > file_size ||
        h.bytecode_size > file_size - h.bytecode_offset) {
        err = "artifact truncated or corrupt: bytecode " + std::to_string(h.bytecode_size) + " bytes at " +
              std::to_string(h.bytecode_offset) + ", file " + std::to_string(file_size) + " bytes";
        return false;
    }
    // The loader maps [0, offset + size) with a size_t length.
    if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
        if (h.bytecode_offset + h.bytecode_size > (uint64_t)SIZE_MAX) {
            err = "artifact too large for this address space";
            return false;
        }
    }
    info = ArtifactInfo{};
    h.function[sizeof h.function - 1] = h.vendor[sizeof h.vendor - 1] = h.key[sizeof h.key - 1] = 0;
    info.function = h.function;
    info.vendor = h.vendor;
    info.key = h.key;
    info.in_bytes.assign(h.in_bytes, h.in_bytes + h.n_in);
    info.out_bytes.assign(h.out_bytes, h.out_bytes + h.n_out);
    info.model_hash = h.model_hash;
    info.tflite_hash = h.tflite_hash;
    info.bytecode_offset = h.bytecode_offset;
    info.bytecode_size = h.bytecode_size;
    return true;
}

}  // namespace starling::npu
