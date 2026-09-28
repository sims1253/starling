// Opt-in C entry point validation with the deterministic model that omits
// the optional CTC head. A failed candidate request must leave greedy usable.
#include "include/starling_ggml.h"
#include "tiny_granite_fixture.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

int main() {
    int failures = 0;
    const auto check = [&](bool pass, const char* label) {
        std::printf("[%s] %s\n", pass ? "PASS" : "FAIL", label);
        failures += !pass;
    };
    TinyGraniteFixture fixture(std::filesystem::temp_directory_path() /
                               "granite_ctc_entry_test.gguf");
    if (!fixture.wrote()) return 2;
    const char* err = nullptr;
    void* handle = starling_ggml_granite_load(fixture.path.string().c_str(), &err);
    if (!handle) {
        std::fprintf(stderr, "load: %s\n", err ? err : "unknown error");
        return 2;
    }
    std::vector<float> pcm(8000, 0.0f);
    char* bad = starling_ggml_granite_decode_ctc(handle, pcm.data(), pcm.size(), 0, &err);
    check(!bad && err && std::string(err).find("1..16") != std::string::npos,
          "invalid K rejected before decoding");
    starling_ggml_free_string(bad);
    err = nullptr;
    char* missing = starling_ggml_granite_decode_ctc(
        handle, pcm.data(), pcm.size(), 2, &err);
    check(!missing && err && std::string(err).find("no optional CTC draft head") !=
                          std::string::npos,
          "missing optional CTC head returns a clear error");
    starling_ggml_free_string(missing);
    err = nullptr;
    char* greedy1 = starling_ggml_granite_decode(handle, pcm.data(), pcm.size(), &err);
    check(greedy1 != nullptr, "default greedy still decodes after failed CTC request");
    err = nullptr;
    char* greedy2 = starling_ggml_granite_decode(handle, pcm.data(), pcm.size(), &err);
    check(greedy1 && greedy2 && std::string(greedy1) == greedy2,
          "consecutive default greedy requests keep stable state");
    starling_ggml_free_string(greedy1);
    starling_ggml_free_string(greedy2);
    starling_ggml_granite_free(handle);
    return failures ? 1 : 0;
}
