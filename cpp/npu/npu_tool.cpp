// Copyright 2026 The Starling Authors. Apache-2.0.
//
// adb-shell tool for the Pixel TPU path (compiling needs the shell user: the
// firmware compile service refuses apps not on Google's EdgeTPU allowlist;
// loading and running an artifact works from any app).
//   npu_tool fingerprint MODEL.gguf
//   npu_tool compile MODEL.tflite ART --in B,B --out B [--fn F] [--key K]
//                    [--gguf MODEL.gguf | --hash 0x...] [--pad N]
//   npu_tool run ART [--mode mmap|fdoff|fd] [--runs N] [--out PREFIX] [IN0 IN1 ...]
//   npu_tool repack ART OUT [--pad N]
//   npu_tool info ART
#include "gguf_fingerprint.hpp"
#include "npu_probe.hpp"
#include "tpu_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

using namespace starling::npu;

static std::vector<uint64_t> sizes(const char* s) {
    std::vector<uint64_t> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::strtoull(t.c_str(), nullptr, 10));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: npu_tool fingerprint MODEL.gguf\n"
                     "       npu_tool compile MODEL.tflite ART --in B,.. --out B,.. [--fn F] [--key K]\n"
                     "                        [--gguf MODEL.gguf | --hash 0x..] [--pad N]\n"
                     "       npu_tool run ART [--mode mmap|fdoff|fd] [--runs N] [--out PREFIX] [IN...]\n"
                     "       npu_tool repack ART OUT [--pad N]\n"
                     "       npu_tool info ART\n");
        return 2;
    }
    const std::string cmd = argv[1];
    bool ok = false;
    std::string rep, err;
    if (cmd == "fingerprint") {
        uint64_t h = 0;
        ok = gguf_fingerprint(argv[2], h, err);
        char b[64];
        std::snprintf(b, sizeof b, "0x%016llx\n", (unsigned long long)h);
        rep = ok ? std::string(b) : err + "\n";
    } else if (cmd == "compile" && argc >= 4) {
        std::string fn = "serving_default", key;
        std::vector<uint64_t> in, out;
        uint64_t pad = 0, hash = 0;
        for (int i = 4; i + 1 < argc; i += 2) {
            if (!std::strcmp(argv[i], "--fn")) fn = argv[i + 1];
            else if (!std::strcmp(argv[i], "--in")) in = sizes(argv[i + 1]);
            else if (!std::strcmp(argv[i], "--out")) out = sizes(argv[i + 1]);
            else if (!std::strcmp(argv[i], "--pad")) pad = std::strtoull(argv[i + 1], nullptr, 10);
            else if (!std::strcmp(argv[i], "--key")) key = argv[i + 1];
            else if (!std::strcmp(argv[i], "--hash")) hash = std::strtoull(argv[i + 1], nullptr, 0);
            else if (!std::strcmp(argv[i], "--gguf") && !gguf_fingerprint(argv[i + 1], hash, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
        }
        rep = probe_compile(argv[2], argv[3], fn, in, out, pad, ok, key, hash);
    } else if (cmd == "repack" && argc >= 4) {
        // Rewrite an artifact with its bytecode at offset 4096 + pad (pad != 0
        // reproduces the unaligned bytecode offset of a LiteRT-serialized model).
        ArtifactInfo info;
        uint64_t pad = argc >= 6 && !std::strcmp(argv[4], "--pad") ? std::strtoull(argv[5], nullptr, 10) : 0;
        FILE* f = read_artifact_info(argv[2], info, err) ? std::fopen(argv[2], "rb") : nullptr;
        std::vector<uint8_t> bc(info.bytecode_size);
        ok = f && std::fseek(f, (long)info.bytecode_offset, SEEK_SET) == 0 &&
             std::fread(bc.data(), 1, bc.size(), f) == bc.size();
        if (f) std::fclose(f);
        ok = ok && write_artifact(argv[3], info, bc, pad, err);
        rep = ok ? "repacked with bytecode at offset " + std::to_string(4096 + pad) + "\n" : "repack failed: " + err + "\n";
    } else if (cmd == "info") {
        ArtifactInfo info;
        ok = read_artifact_info(argv[2], info, err);
        if (ok) {
            char b[512];
            std::snprintf(b, sizeof b, "fn '%s'\nkey '%s'\nvendor %s\nmodel_hash 0x%016llx\ntflite_hash 0x%016llx\n"
                                       "bytecode %llu bytes at %llu\n",
                          info.function.c_str(), info.key.c_str(), info.vendor.c_str(),
                          (unsigned long long)info.model_hash, (unsigned long long)info.tflite_hash,
                          (unsigned long long)info.bytecode_size, (unsigned long long)info.bytecode_offset);
            rep = b;
        } else {
            rep = err + "\n";
        }
    } else if (cmd == "run") {
        LoadMode mode = LoadMode::Mmap;
        int runs = 20;
        std::string out;
        std::vector<std::string> ins;
        for (int i = 3; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
                const std::string m = argv[++i];
                mode = m == "fdoff" ? LoadMode::FdOffset : m == "fd" ? LoadMode::Fd : LoadMode::Mmap;
            } else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) {
                runs = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) {
                out = argv[++i];
            } else {
                ins.push_back(argv[i]);
            }
        }
        rep = probe_run(argv[2], mode, ins, out, runs, ok);
    } else {
        std::fprintf(stderr, "unknown command\n");
        return 2;
    }
    std::fputs(rep.c_str(), stdout);
    return ok ? 0 : 1;
}
