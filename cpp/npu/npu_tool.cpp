// Copyright 2026 The Starling Authors. Apache-2.0.
//
// starling-npu-tool: the adb-shell half of scripts/npu/npu_compile.sh.
// Compiling must run as the shell user: the firmware compile service refuses
// apps not on Google's EdgeTPU allowlist (loading and running an artifact
// works from any app, see tpu_runtime.hpp).
//   starling-npu-tool fingerprint MODEL.gguf    prints "model_hash 0x.."
//   starling-npu-tool compile MODEL.tflite ART --in B,.. --out B,.. --hash 0x.. [--fn F] [--key K]
//   starling-npu-tool info ART
// The artifact's model hash (--hash) is the one export_encoder.py printed for
// the GGUF the graph's weights came from; compile refuses a .tflite whose
// provenance stamp (tflite_provenance.hpp) names another model hash or key.
#include "gguf_fingerprint.hpp"
#include "tflite_provenance.hpp"
#include "tpu_runtime.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

using namespace starling::npu;

namespace {

std::vector<uint64_t> sizes(const char* s) {
    std::vector<uint64_t> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::strtoull(t.c_str(), nullptr, 10));
    return v;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? (size_t)n : 0);
    bool ok = n >= 0 && std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

// Compiles a .tflite on the device and writes the artifact around its bytecode.
bool compile(const std::string& tflite, const std::string& artifact, ArtifactInfo info, std::string& err) {
    if (!load_southbound(err)) return false;
    std::vector<uint8_t> model;
    if (!read_file(tflite, model)) {
        err = "cannot read " + tflite;
        return false;
    }
    TfliteProvenance prov;
    if (!read_tflite_provenance(model.data(), model.size(), prov, err)) {
        err = tflite + ": " + err;
        return false;
    }
    if (prov.model_hash != info.model_hash || (!info.key.empty() && prov.key != info.key)) {
        char b[64];
        std::snprintf(b, sizeof b, "0x%016llx", (unsigned long long)prov.model_hash);
        err = tflite + " was exported for model hash " + b + " ('" + prov.key + "'), not the one to build";
        return false;
    }
    std::vector<uint8_t> bc;
    const double t0 = now_ms();
    if (!compile_tflite(model.data(), model.size(), bc, err)) return false;
    std::printf("on-device compile %.1f ms, %zu-byte model -> %zu-byte bytecode\n", now_ms() - t0, model.size(),
                bc.size());
    info.vendor = vendor_version();
    info.tflite_hash = fnv1a_update(kFnvBasis, model.data(), model.size());
    if (!write_artifact(artifact, info, bc, err)) return false;
    std::printf("wrote %s (TPU firmware %s)\n", artifact.c_str(), info.vendor.c_str());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: starling-npu-tool fingerprint MODEL.gguf\n"
                     "       starling-npu-tool compile MODEL.tflite ART --in B,.. --out B,.. --hash 0x.. [--fn F]\n"
                     "                                 [--key K]\n"
                     "       starling-npu-tool info ART\n");
        return 2;
    }
    const std::string cmd = argv[1];
    std::string err;
    if (cmd == "fingerprint") {
        uint64_t hash = 0;
        const double t0 = now_ms();
        if (!gguf_model_hash(argv[2], hash, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::printf("model_hash 0x%016llx\n", (unsigned long long)hash);
        std::fprintf(stderr, "fingerprinted in %.1f ms\n", now_ms() - t0);
        return 0;
    }
    if (cmd == "compile" && argc >= 4) {
        ArtifactInfo info;
        info.function = "serving_default";
        for (int i = 4; i + 1 < argc; i += 2) {
            if (!std::strcmp(argv[i], "--fn")) info.function = argv[i + 1];
            else if (!std::strcmp(argv[i], "--in")) info.in_bytes = sizes(argv[i + 1]);
            else if (!std::strcmp(argv[i], "--out")) info.out_bytes = sizes(argv[i + 1]);
            else if (!std::strcmp(argv[i], "--key")) info.key = argv[i + 1];
            else if (!std::strcmp(argv[i], "--hash")) info.model_hash = std::strtoull(argv[i + 1], nullptr, 0);
        }
        if (info.model_hash == 0) {
            std::fprintf(stderr, "compile: --hash 0x.. (the model hash export_encoder.py printed) is required\n");
            return 2;
        }
        if (!compile(argv[2], argv[3], info, err)) {
            std::fprintf(stderr, "compile: %s\n", err.c_str());
            return 1;
        }
        return 0;
    }
    if (cmd == "info") {
        ArtifactInfo info;
        if (!read_artifact_info(argv[2], info, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::printf("fn '%s'\nkey '%s'\nvendor %s\nmodel_hash 0x%016llx\ntflite_hash 0x%016llx\n"
                    "bytecode %llu bytes at %llu\n",
                    info.function.c_str(), info.key.c_str(), info.vendor.c_str(),
                    (unsigned long long)info.model_hash, (unsigned long long)info.tflite_hash,
                    (unsigned long long)info.bytecode_size, (unsigned long long)info.bytecode_offset);
        return 0;
    }
    std::fprintf(stderr, "unknown command\n");
    return 2;
}
