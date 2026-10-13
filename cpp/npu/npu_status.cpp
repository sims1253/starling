// Copyright 2026 The Starling Authors. Apache-2.0.
#include "npu_status.hpp"

#include "tpu_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <sys/stat.h>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace starling::npu {

bool tpu_present() {
    static const bool listed = [] {
        FILE* f = std::fopen("/vendor/etc/public.libraries.txt", "r");
        if (!f) return false;
        char line[256];
        bool found = false;
        while (!found && std::fgets(line, sizeof line, f)) found = std::strstr(line, "libedgetpu_litert.so") != nullptr;
        std::fclose(f);
        return found;
    }();
    return listed;
}

std::string npu_dirs() {
    if (const char* d = std::getenv("STARLING_NPU_DIR"); d && *d) return d;
    if (const char* c = std::getenv("STARLING_FAST_CACHE_DIR"); c && *c) return c;
    return {};
}

void report_piece(const std::string& piece, uint64_t model_hash, const std::string& gguf, const std::string& model,
                  PieceState state, const std::string& message) {
    static const char* const kNames[] = {"ok", "missing", "stale", "error"};
    const char* st = kNames[(int)state];
    // "missing" is reported on every load of a supported model: only where a
    // TPU exists, and without loading its runtime (the engine loads it only
    // once an artifact is there).
    if (state == PieceState::Missing && !tpu_present()) return;
    const std::string firmware = state == PieceState::Missing ? std::string() : vendor_version();
    char hash[24];
    std::snprintf(hash, sizeof hash, "%016llx", (unsigned long long)model_hash);
    if (state == PieceState::Stale || state == PieceState::Error) {
        static std::mutex mu;
        static std::set<std::string> logged;
        std::lock_guard<std::mutex> lock(mu);
        if (logged.insert(piece + hash + st).second) {
            const std::string line = "TPU " + piece + " not used (" + st + "), the GPU runs it: " + message +
                                     (state == PieceState::Stale ? " -- recompile via adb (scripts/npu/npu_compile_all.sh)"
                                                                 : "");
            std::fprintf(stderr, "[npu] %s\n", line.c_str());
#if defined(__ANDROID__)
            __android_log_print(ANDROID_LOG_WARN, "StarlingNpu", "%s", line.c_str());
#endif
        }
    }
    // One line per key: a path with a line break in it is not written, and
    // line breaks in free text become spaces.
    const std::string path_line = gguf.find_first_of("\r\n") == std::string::npos ? gguf : std::string();
    auto one_line = [](std::string s) {
        for (char& c : s)
            if (c == '\n' || c == '\r') c = ' ';
        return s;
    };
    const std::string body = std::string("piece=") + piece + "\nhash=0x" + hash + "\ngguf=" + path_line +
                             "\nmodel=" + one_line(model) + "\nstate=" + st + "\nfirmware=" + one_line(firmware) +
                             "\nmessage=" + one_line(message) + "\n";
    const std::string dirs = npu_dirs();
    for (size_t b = 0; b <= dirs.size();) {
        size_t e = dirs.find(':', b);
        if (e == std::string::npos) e = dirs.size();
        const std::string dir = dirs.substr(b, e - b);
        b = e + 1;
        struct stat sb {};
        if (dir.empty() || stat(dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) continue;
        const std::string path = dir + "/status-" + piece + "-" + hash + ".txt";
        const std::string tmp = path + ".tmp";
        if (FILE* f = std::fopen(tmp.c_str(), "wb")) {
            const bool ok = std::fwrite(body.data(), 1, body.size(), f) == body.size();
            if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), path.c_str());
            else std::remove(tmp.c_str());
        }
    }
}

}  // namespace starling::npu
