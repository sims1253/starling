// voxtral_engine_test.cpp — CPU-only end-to-end check against the tiny GGUF.
//
// Loads models/tiny/voxtral-tiny.gguf, runs the FULL capi path (offline pad ->
// mel -> Phase-2a encoder+projector -> prompt/embeds with additive rows ->
// offline greedy loop -> raw-byte tokenize) on the reference PCM from
// models/tiny/voxtral-tiny-ref.json, and asserts the EXACT greedy token-id
// sequence the whole-model torch oracle recorded ("e2e_ids"). Any divergence
// in any stage (frontend, encoder, ada, rope, attention, injection, argmax)
// flips ids, so the exact match is the strictest gate. Also asserts the cap
// ("e2e_cap" == mel_T//8) and that the decoded text matches the tokenizer's
// own decode of the oracle ids.
//
// Usage: ./voxtral_engine_test [repo-root]
#include "voxtral/loader.hpp"
#include "voxtral/mel.hpp"
#include "voxtral/encoder.hpp"
#include "voxtral/prompt.hpp"
#include "voxtral/llm.hpp"
#include "voxtral/tokenizer.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what,
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) failures++;
}

// Minimal JSON reader (same shape as the encoder test's): string keys to a
// scalar, a float array, or an int array ("e2e_ids").
struct JsonRef {
    std::string text;
    const char* seek(const char* key) const {
        std::string q = std::string("\"") + key + "\"";
        const char* p = std::strstr(text.c_str(), q.c_str());
        if (!p) return nullptr;
        p = std::strchr(p + q.size(), ':');
        return p ? p + 1 : nullptr;
    }
    double number(const char* key, bool& ok) const {
        const char* p = seek(key);
        if (!p) { ok = false; return 0; }
        char* end = nullptr;
        double v = std::strtod(p, &end);
        ok = (end != p);
        return v;
    }
    static const char* parse_array(const char* p, std::vector<float>& out) {
        const char* q = p;
        if (*q != '[') return nullptr;
        ++q;
        for (;;) {
            while (*q == ' ' || *q == '\n' || *q == '\t' || *q == '\r') ++q;
            if (*q == ']') return q + 1;
            if (*q == ',') { ++q; continue; }
            char* end = nullptr;
            double v = std::strtod(q, &end);
            if (end == q) return nullptr;
            out.push_back((float) v);
            q = end;
        }
    }
    bool array(const char* key, std::vector<float>& out) const {
        const char* p = seek(key);
        if (!p) return false;
        while (*p == ' ' || *p == '\n') ++p;
        return parse_array(p, out) != nullptr;
    }
};

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string root = argc > 1 ? argv[1] : ".";
    using namespace starling::ggml;
    using namespace starling::ggml::voxtral;

    VoxtralModel model;
    std::string err;
    if (!model.load((root + "/models/tiny/voxtral-tiny.gguf").c_str(), err)) {
        std::printf("[SKIP] tiny GGUF absent or invalid: %s\n", err.c_str());
        std::printf("ENGINE TEST FAILED\n");
        return 1;
    }
    check(true, "tiny GGUF loads (decoder tensors validate)");

    Tokenizer tok;
    if (!tok.load(model.loader, model.config, err)) {
        check(false, "tokenizer loads", err);
        std::printf("ENGINE TEST FAILED\n");
        return 1;
    }
    check(true, "tokenizer loads");

    // Reference JSON: pcm + oracle ids/cap.
    JsonRef ref;
    {
        FILE* f = std::fopen((root + "/models/tiny/voxtral-tiny-ref.json").c_str(), "rb");
        if (!f) {
            check(false, "reference JSON opens");
            std::printf("ENGINE TEST FAILED\n");
            return 1;
        }
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        ref.text.resize((size_t) n);
        if (std::fread(ref.text.data(), 1, (size_t) n, f) != (size_t) n) {
            std::fclose(f);
            check(false, "reference JSON reads");
            std::printf("ENGINE TEST FAILED\n");
            return 1;
        }
        std::fclose(f);
    }
    std::vector<float> pcm, e2e_f;
    bool ok = ref.array("pcm", pcm) && ref.array("e2e_ids", e2e_f);
    bool ok_n = false;
    const double e2e_cap = ref.number("e2e_cap", ok_n);
    ok = ok && ok_n;
    check(ok, "reference JSON parses (pcm/e2e_ids/e2e_cap)");
    if (!ok) {
        std::printf("ENGINE TEST FAILED\n");
        return 1;
    }
    std::vector<int32_t> want_ids;
    for (float v : e2e_f) want_ids.push_back((int32_t) v);

    // Full capi path, stage by stage (same calls capi_voxtral_decode makes).
    MelFeatures mel;
    if (!compute_log_mel(model.config, model.loader, pcm.data(), pcm.size(), mel, err)) {
        check(false, "voxtral mel computes", err);
        std::printf("ENGINE TEST FAILED\n");
        return 1;
    }
    check(true, "voxtral mel computes");
    AudioEncoding audio;
    if (!encode_audio_and_project(model, mel, audio, err)) {
        check(false, "encoder+projector runs", err);
        std::printf("ENGINE TEST FAILED\n");
        return 1;
    }
    check(true, "encoder+projector runs");
    const std::vector<int32_t> prompt = build_transcribe_prompt(model.config);
    {
        char detail[128];
        std::snprintf(detail, sizeof detail, "P=%zu cap=%.0f mel_T=%lld", prompt.size(),
                      e2e_cap, (long long) mel.n_frames);
        check(prompt.size() == 39 && generation_cap((int64_t) prompt.size(), mel.n_frames) ==
              (int64_t) e2e_cap, "prompt is 39 ids; cap == oracle cap (mel_T//8)", detail);
    }
    InputsEmbeds inputs;
    if (!build_inputs_embeds(model, prompt, audio, inputs, err)) {
        check(false, "inputs embeds build (embed + rows 0..P-1)", err);
        std::printf("ENGINE TEST FAILED\n");
        return 1;
    }
    check(true, "inputs embeds build (embed + rows 0..P-1)");
    GenerateOptions options;
    options.max_new_tokens = model.config.max_new_tokens;
    options.max_cache_len = model.config.llm.max_cache;
    options.eos_token_id = model.config.eos_token_id;
    GenerateResult got;
    if (!greedy_generate(model, inputs, audio, mel.n_frames, options, got, err)) {
        check(false, "offline greedy loop runs", err);
        std::printf("ENGINE TEST FAILED\n");
        return 1;
    }
    check(true, "offline greedy loop runs");

    // EXACT id-sequence match against the torch oracle.
    {
        std::string detail;
        if (got.ids.size() != want_ids.size())
            detail = "got " + std::to_string(got.ids.size()) + " ids, want " +
                     std::to_string(want_ids.size());
        else {
            for (size_t i = 0; i < want_ids.size(); ++i)
                if (got.ids[i] != want_ids[i]) {
                    detail = "first diff at " + std::to_string(i) + ": got " +
                             std::to_string(got.ids[i]) + " want " +
                             std::to_string(want_ids[i]);
                    break;
                }
        }
        if (detail.empty()) {
            std::string ids;
            for (size_t i = 0; i < got.ids.size(); ++i) {
                if (i) ids += ",";
                ids += std::to_string(got.ids[i]);
            }
            std::printf("    ids=[%s]\n", ids.c_str());
        }
        check(detail.empty(), "EXACT greedy id match vs torch oracle", detail);
    }

    // Decoded text matches the tokenizer's own decode of the oracle ids.
    {
        const std::string got_text = tok.decode(got.ids);
        const std::string want_text = tok.decode(want_ids);
        check(got_text == want_text, "decoded text matches oracle decode",
              "got " + got_text + " want " + want_text);
        std::printf("    text=%s\n", got_text.c_str());
    }

    starling::ggml::shutdown_backend();
    std::printf("%s\n", failures ? "ENGINE TEST FAILED" : "ENGINE TEST OK");
    return failures ? 1 : 0;
}
