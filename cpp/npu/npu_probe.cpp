// Copyright 2026 The Starling Authors. Apache-2.0.
//
// The two steps of the NPU spike, shared by the adb test binary (npu_tool) and
// the JNI probe running inside an untrusted app: compile a .tflite into an
// artifact, and load + run an artifact. Both return a human-readable report.
#include "npu_probe.hpp"

#include "tpu_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace starling::npu {

namespace {

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

std::string fmt(const char* f, double a) {
    char b[128];
    std::snprintf(b, sizeof b, f, a);
    return b;
}

}  // namespace

std::string probe_compile(const std::string& tflite, const std::string& artifact, const std::string& function,
                          const std::vector<uint64_t>& in_bytes, const std::vector<uint64_t>& out_bytes,
                          uint64_t pad, bool& ok, const std::string& key, uint64_t model_hash) {
    ok = false;
    std::string err, r;
    double t0 = now_ms();
    if (!load_southbound(err)) return "load: " + err + "\n";
    r += "vendor api " + vendor_version() + fmt(", firmware load %.1f ms\n", now_ms() - t0);
    std::vector<uint8_t> model;
    t0 = now_ms();
    if (!read_file(tflite, model)) return r + "cannot read " + tflite + "\n";
    r += fmt("read model %.1f ms", now_ms() - t0) + ", " + std::to_string(model.size()) + " bytes\n";
    std::vector<uint8_t> bc;
    t0 = now_ms();
    if (!compile_tflite(model.data(), model.size(), bc, err)) return r + "compile: " + err + "\n";
    r += fmt("on-device compile %.1f ms", now_ms() - t0) + ", bytecode " + std::to_string(bc.size()) + " bytes\n";
    ArtifactInfo info;
    info.function = function;
    info.in_bytes = in_bytes;
    info.out_bytes = out_bytes;
    info.vendor = vendor_version();
    info.key = key;
    info.model_hash = model_hash;
    info.tflite_hash = fnv1a(model.data(), model.size());
    t0 = now_ms();
    if (!write_artifact(artifact, info, bc, pad, err)) return r + "write: " + err + "\n";
    r += fmt("write artifact %.1f ms", now_ms() - t0) + " -> " + artifact + "\n";
    ok = true;
    return r;
}

std::string probe_run(const std::string& artifact, LoadMode mode, const std::vector<std::string>& inputs,
                      const std::string& out_prefix, int runs, bool& ok) {
    ok = false;
    std::string err, r;
    double t0 = now_ms();
    if (!load_southbound(err)) return "load: " + err + "\n";
    r += "vendor api " + vendor_version() + fmt(", firmware load %.1f ms\n", now_ms() - t0);
    TpuModel m;
    t0 = now_ms();
    if (!m.open(artifact, mode, err)) return r + "open: " + err + "\n";
    r += fmt("artifact load (no compile) %.1f ms", now_ms() - t0) + ", fn '" + m.info().function +
         "', compiled on vendor " + m.info().vendor + "\n";
    if (!m.info().key.empty()) r += "key '" + m.info().key + "'\n";
    for (size_t i = 0; i < m.info().in_bytes.size(); ++i) {
        std::vector<uint8_t> d(m.info().in_bytes[i], 0);
        if (i < inputs.size()) {
            std::vector<uint8_t> f;
            if (!read_file(inputs[i], f) || f.size() != d.size())
                return r + "input " + inputs[i] + " missing or wrong size\n";
            d = f;
        }
        if (!m.write_input(i, d.data(), d.size(), err)) return r + err + "\n";
    }
    std::vector<double> ts;
    for (int k = 0; k <= runs; ++k) {
        t0 = now_ms();
        if (!m.run(err)) return r + "run " + std::to_string(k) + ": " + err + "\n";
        double dt = now_ms() - t0;
        if (k == 0)
            r += fmt("first run %.2f ms\n", dt);
        else
            ts.push_back(dt);
    }
    if (!ts.empty()) {
        std::vector<double> s = ts;
        std::sort(s.begin(), s.end());
        double sum = 0;
        for (double v : ts) sum += v;
        char b[160];
        std::snprintf(b, sizeof b, "runs %zu: median %.2f ms, min %.2f, max %.2f, mean %.2f\n", s.size(),
                      s[s.size() / 2], s[0], s.back(), sum / s.size());
        r += b;
    }
    if (!out_prefix.empty()) {
        for (size_t i = 0; i < m.info().out_bytes.size(); ++i) {
            std::vector<uint8_t> o(m.info().out_bytes[i]);
            t0 = now_ms();
            if (!m.read_output(i, o.data(), o.size(), err)) return r + err + "\n";
            double dt = now_ms() - t0;
            const std::string fn = out_prefix + "_out" + std::to_string(i) + ".f32";
            FILE* f = std::fopen(fn.c_str(), "wb");
            if (f) {
                std::fwrite(o.data(), 1, o.size(), f);
                std::fclose(f);
            }
            r += fmt("output read-back %.2f ms", dt) + " -> " + fn + "\n";
        }
    }
    ok = true;
    return r;
}

}  // namespace starling::npu
