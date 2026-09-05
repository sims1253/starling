// voxtral_loader_guard_test.cpp — CPU-only loader smoke check for the voxtral
// GGML metadata guards. Loading parses GGUF metadata + tensor names only (no
// backend/GPU is touched), so this runs even while the GPU is busy. Exits 0
// with SKIP when the model file is absent (it is gitignored).
//
// Checks: metadata round-trip (arch, profile, key dims), the mel/pad/token
// arithmetic against the settled fixture numbers (short 1136/142, medium
// 2624/328, long 7832/979), the tokenizer decoding a hand-computed id
// sequence ([22177, 4304] -> "Hello world"; CONTROL ids skipped), and the
// baked 39-id prompt prefix ([1] + [32]*38).
//
// Usage: ./voxtral_loader_guard_test [voxtral.gguf]
#include "voxtral/loader.hpp"
#include "voxtral/tokenizer.hpp"
#include "voxtral/prompt.hpp"
#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what, const std::string& err = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what,
                (!ok && !err.empty()) ? " -- " : "", ok ? "" : err.c_str());
    if (!ok) failures++;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "models/voxtral-mini-4b-realtime-bf16-exact.gguf";

    starling::ggml::voxtral::VoxtralModel m;
    std::string err;
    if (!m.load(path, err)) {
        if (err.find("failed to open") != std::string::npos ||
            err.find("cannot open") != std::string::npos ||
            err.find("open ") != std::string::npos) {
            std::printf("[SKIP] voxtral GGUF absent (%s)\n", path);
            std::printf("LOADER GUARD OK\n");
            return 0;
        }
        check(false, "voxtral GGUF loads", err);
        std::printf("LOADER GUARD FAILED\n");
        return 1;
    }
    check(true, "voxtral GGUF loads (metadata guards accept)");
    const auto& c = m.config;

    // Metadata round-trip: the settled dims.
    check(c.encoder.n_layers == 32 && c.encoder.d_model == 1280 &&
          c.encoder.n_heads == 32 && c.encoder.head_dim == 64,
          "encoder dims round-trip (32 layers, d1280, 32x64)");
    check(c.llm.n_layers == 26 && c.llm.hidden == 3072 &&
          c.llm.n_heads == 32 && c.llm.n_kv_heads == 8 &&
          c.llm.head_dim == 128 && c.llm.intermediate == 9216 &&
          c.llm.vocab == 131072,
          "llm dims round-trip (26 layers, h3072, 32q/8kv x128, i9216)");
    check(c.projector.input_size == 5120 && c.projector.output_size == 3072 &&
          c.projector.downsample == 4,
          "projector dims round-trip (5120->3072, downsample 4)");
    check(c.bos_token_id == 1 && c.eos_token_id == 2 &&
          c.pad_token_id == 11 && c.streaming_pad_id == 32,
          "token ids round-trip (bos 1, eos 2, pad 11, spad 32)");

    // Settled mel arithmetic on the three fixture sample counts
    // (short 118960, medium 356880, long 1189600 @16kHz).
    using starling::ggml::voxtral::mel_frames;
    using starling::ggml::voxtral::audio_token_count;
    check(mel_frames(118960) == 1136 && audio_token_count(1136) == 142,
          "short fixture mel/tokens (1136/142)");
    check(mel_frames(356880) == 2624 && audio_token_count(2624) == 328,
          "medium fixture mel/tokens (2624/328)");
    check(mel_frames(1189600) == 7832 && audio_token_count(7832) == 979,
          "long fixture mel/tokens (7832/979)");

    // Baked prompt prefix: [1] + [32]*38.
    {
        using starling::ggml::voxtral::build_transcribe_prompt;
        const std::vector<int32_t> p = build_transcribe_prompt(c);
        bool ok = p.size() == 39 && p[0] == 1;
        for (size_t i = 1; ok && i < p.size(); ++i) ok = p[i] == 32;
        check(ok, "prompt prefix is [1] + [32]*38");
    }

    // Tokenizer: hand-computed ids decode, CONTROL ids are skipped.
    {
        starling::ggml::voxtral::Tokenizer tok;
        std::string terr;
        if (!tok.load(m.loader, c, terr)) {
            check(false, "voxtral tokenizer loads", terr);
        } else {
            check(tok.decode({22177, 4304}) == "Hello world",
                  "tokenizer decodes [22177, 4304] to 'Hello world'");
            check(tok.decode({1, 22177, 32, 4304, 2}).empty() == false &&
                  tok.decode({1, 22177, 32, 4304, 2}) == "Hello world",
                  "tokenizer skips CONTROL ids (1, 32, 2)");
        }
    }

    std::printf("%s\n", failures ? "LOADER GUARD FAILED" : "LOADER GUARD OK");
    return failures ? 1 : 0;
}
