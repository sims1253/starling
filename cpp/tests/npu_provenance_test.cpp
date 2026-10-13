// npu_provenance_test.cpp — the provenance stamp scripts/npu/export_encoder.py
// writes into a .tflite and starling-npu-tool compile checks
// (cpp/npu/tflite_provenance.hpp).
//
// kStampedTflite is embed_provenance(kBaseTflite, "0x0123456789abcdef",
// "qwen3-asr-enc/2 test") as export_encoder.py writes it
// (tests/test_npu_fingerprint.py builds the same base and pins the stamped
// bytes' SHA-256), so the Python writer and this reader agree. Also: a file
// without a stamp, a truncated one, a damaged stamp, and a damaged offset are
// refused, never read out of bounds.
//
// Exit 0 = pass, 1 = failure.
#include "npu/tflite_provenance.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace starling::npu;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

std::vector<uint8_t> hex(const char* s) {
    std::vector<uint8_t> out;
    for (; s[0] && s[1]; s += 2) out.push_back((uint8_t)std::stoul(std::string(s, 2), nullptr, 16));
    return out;
}

// A minimal TFLite Model: version 3, buffers [empty, "1.14.0"], metadata
// [min_runtime_version -> buffer 1] (_minimal_tflite in the Python test).
const char* const kBaseTflite =
    "1c00000054464c33120010000400000000000000080000000c0000001400000003000000080000001000000002000000"
    "140000001c000000010000002400000004000400040000000600080004000000080000003800000008000c0004000800"
    "080000000800000001000000130000006d696e5f72756e74696d655f76657273696f6e00000000000000000006000000"
    "312e31342e300000";

const char* const kStampedTflite =
    "1c00000054464c33140024000400000000000000140000001c0000001400000003000000000000000000000000000000"
    "100000000000000018000000000000000300000000010000080100002c000000020000000c0100000c00000008000c00"
    "040008000800000018000000020000000600080004000000080000002000000017000000737461726c696e675f6e7075"
    "5f70726f76656e616e636500370000006d6f64656c5f686173683d3078303132333435363738396162636465660a6b65"
    "793d7177656e332d6173722d656e632f3220746573740a00000000000000000000000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000120010000400000000000000080000000c00000014000000"
    "03000000080000001000000002000000140000001c000000010000002400000004000400040000000600080004000000"
    "080000003800000008000c0004000800080000000800000001000000130000006d696e5f72756e74696d655f76657273"
    "696f6e00000000000000000006000000312e31342e300000";

}  // namespace

int main() {
    const std::vector<uint8_t> base = hex(kBaseTflite), stamped = hex(kStampedTflite);
    TfliteProvenance p;
    std::string err;

    const bool ok = read_tflite_provenance(stamped.data(), stamped.size(), p, err);
    check(ok, "the stamp export_encoder.py writes is read " + err);
    check(p.model_hash == 0x0123456789abcdefull, "... its model hash");
    check(p.key == "qwen3-asr-enc/2 test", "... its key '" + p.key + "'");

    err.clear();
    TfliteProvenance scratch;
    bool refused = !read_tflite_provenance(base.data(), base.size(), scratch, err) &&
                   err.find("no provenance") != std::string::npos;
    check(refused, "a .tflite without a stamp is refused: " + err);

    // A truncated copy is refused while the cut reaches into the stamp (the
    // stamp sits in front of the original flatbuffer), and otherwise reads
    // the same stamp; never past the end (each copy is its own allocation).
    const size_t stamp_end = std::string(stamped.begin(), stamped.end()).find("test\n") + 5;
    bool all = true;
    for (size_t n = 0; n < stamped.size(); ++n) {
        std::vector<uint8_t> cut(stamped.begin(), stamped.begin() + (long)n);
        TfliteProvenance q;
        std::string e;
        const bool read = read_tflite_provenance(cut.data(), cut.size(), q, e);
        if (n < stamp_end ? read : (read && (q.model_hash != p.model_hash || q.key != p.key))) all = false;
    }
    check(all, "truncated copies: refused up to the stamp's end, the same stamp after it");

    // A stamp whose hash text was altered (one hex digit -> 'g') is malformed.
    std::vector<uint8_t> bad = stamped;
    const std::string text = "model_hash=0x0123456789abcdef";
    std::string s(bad.begin(), bad.end());
    const size_t at = s.find(text);
    check(at != std::string::npos, "the stamp text is in the file");
    if (at != std::string::npos) {
        bad[at + text.size() - 1] = 'g';
        err.clear();
        refused = !read_tflite_provenance(bad.data(), bad.size(), scratch, err);
        check(refused, "a damaged stamp is refused: " + err);
    }

    // A uoffset pointing far out of the file (the root's metadata vector) is refused.
    std::vector<uint8_t> wild = stamped;
    wild[28 + 4 + 4 * 6 + 3] = 0x7f;   // root table at 28, slot of field 6
    err.clear();
    refused = !read_tflite_provenance(wild.data(), wild.size(), scratch, err);
    check(refused, "an out-of-range offset is refused: " + err);

    if (g_fail) {
        std::printf("npu_provenance_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("npu_provenance_test: ok\n");
    return 0;
}
