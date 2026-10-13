// npu_artifact_test.cpp — header parsing and bounds checks of the Pixel TPU
// encoder artifact (cpp/npu/artifact_format.hpp). A header that points the
// bytecode past the end of the file (truncated download or copy) must be
// rejected before the loader maps it. No model or device needed.
//
// Usage: npu_artifact_test     Exit 0 = pass, 1 = failure.
#include "npu/artifact_format.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using namespace starling::npu;

namespace {

int g_fail = 0;

ArtifactHeader valid_header(uint64_t bytecode_size) {
    ArtifactHeader h{};
    std::memcpy(h.magic, kArtifactMagic, sizeof h.magic);
    h.n_in = 2;
    h.n_out = 1;
    h.in_bytes[0] = 128 * 800 * 4;
    h.in_bytes[1] = 104 * 4;
    h.out_bytes[0] = 104 * 1024 * 4;
    std::snprintf(h.function, sizeof h.function, "serving_default");
    std::snprintf(h.vendor, sizeof h.vendor, "vendor-1");
    std::snprintf(h.key, sizeof h.key, "qwen3-asr-enc/2 test");
    h.model_hash = 0x1234;
    h.tflite_hash = 0x5678;
    h.bytecode_offset = kArtifactHeaderBytes;
    h.bytecode_size = bytecode_size;
    return h;
}

void expect(bool want_ok, const char* what, const ArtifactHeader& h, uint64_t file_size,
            size_t head_len = sizeof(ArtifactHeader)) {
    ArtifactInfo info;
    std::string err;
    const bool ok = parse_artifact_header(&h, head_len, file_size, info, err);
    if (ok != want_ok) {
        std::printf("FAIL %s: expected %s, got %s (%s)\n", what, want_ok ? "accept" : "reject",
                    ok ? "accept" : "reject", err.c_str());
        ++g_fail;
    }
    if (!ok && err.empty()) {
        std::printf("FAIL %s: rejected without a message\n", what);
        ++g_fail;
    }
}

}  // namespace

int main() {
    const uint64_t bc = 1000;
    const uint64_t full = kArtifactHeaderBytes + bc;

    {
        ArtifactHeader h = valid_header(bc);
        ArtifactInfo info;
        std::string err;
        if (!parse_artifact_header(&h, sizeof h, full, info, err) || info.function != "serving_default" ||
            info.vendor != "vendor-1" || info.key != "qwen3-asr-enc/2 test" || info.in_bytes.size() != 2 ||
            info.in_bytes[1] != 104 * 4 || info.out_bytes.size() != 1 || info.model_hash != 0x1234 ||
            info.tflite_hash != 0x5678 || info.bytecode_offset != kArtifactHeaderBytes || info.bytecode_size != bc) {
            std::printf("FAIL valid header did not round-trip (%s)\n", err.c_str());
            ++g_fail;
        }
    }
    expect(true, "trailing bytes after the bytecode", valid_header(bc), full + 4096);

    // Truncation: the case that used to SIGBUS in the firmware.
    expect(false, "file one byte short", valid_header(bc), full - 1);
    expect(false, "file ends inside the header", valid_header(bc), 100);
    expect(false, "empty file", valid_header(bc), 0);
    expect(false, "short header read", valid_header(bc), full, sizeof(ArtifactHeader) - 1);

    // offset + size wraps around 2^64.
    {
        ArtifactHeader h = valid_header(UINT64_MAX - kArtifactHeaderBytes + 2);
        expect(false, "offset + size overflows", h, full);
        h = valid_header(bc);
        h.bytecode_offset = UINT64_MAX - 10;
        expect(false, "offset near 2^64", h, full);
        h.bytecode_offset = UINT64_MAX;
        h.bytecode_size = 1;
        expect(false, "offset 2^64-1, size 1", h, UINT64_MAX);
    }
    {
        ArtifactHeader h = valid_header(0);
        expect(false, "empty bytecode", h, full);
        h = valid_header(bc);
        h.bytecode_offset = 16;
        expect(false, "bytecode inside the header", h, full);
    }
    {
        ArtifactHeader h = valid_header(bc);
        h.magic[0] = 'X';
        expect(false, "bad magic", h, full);
        h = valid_header(bc);
        h.n_in = kArtifactMaxIo + 1;
        expect(false, "too many inputs", h, full);
        h = valid_header(bc);
        h.n_out = UINT32_MAX;
        expect(false, "n_out overflow", h, full);
        h = valid_header(bc);
        h.in_bytes[1] = 0;
        expect(false, "empty input buffer", h, full);
        h = valid_header(bc);
        h.out_bytes[0] = kArtifactMaxIoBytes + 1;
        expect(false, "output buffer over 4 GiB", h, full);
    }
    {
        // Unterminated strings stay within their fields.
        ArtifactHeader h = valid_header(bc);
        std::memset(h.function, 'f', sizeof h.function);
        std::memset(h.key, 'k', sizeof h.key);
        ArtifactInfo info;
        std::string err;
        if (!parse_artifact_header(&h, sizeof h, full, info, err) || info.function.size() != sizeof h.function - 1 ||
            info.key.size() != sizeof h.key - 1) {
            std::printf("FAIL unterminated strings\n");
            ++g_fail;
        }
    }

    if (g_fail) return 1;
    std::printf("npu_artifact_test: ok\n");
    return 0;
}
