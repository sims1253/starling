// tokenizer.hpp — Voxtral decode-only raw-byte tokenizer.
//
// The GGUF carries a gpt2-model table with NO merges: ids 0..999 are the tekken
// specials (CONTROL type; skipped on decode) and ids 1000..131071 are raw token
// bytes stored latin-1 (exact byte round-trip; GGUF strings carry the bytes
// through untouched). Decoding concats the bytes of non-CONTROL ids. There is
// no BPE/merge step, which is why the shared BpeTokenizer (an encode+decode
// engine needing merges) does not apply here.
#pragma once

#include "runtime/model_loader.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::voxtral {

class Tokenizer {
public:
    bool load(const ModelLoader& m, std::string& e);

    // Overload for the per-model engine shells, which call
    // tokenizer.load(model.loader, model.config, err) for symmetry with the
    // other component loaders. The config is not consulted.
    template <typename ConfigT>
    bool load(const ModelLoader& m, const ConfigT&, std::string& e) {
        return load(m, e);
    }

    std::string decode(const std::vector<int32_t>& ids) const;

private:
    std::vector<std::string> bytes_;  // id -> raw bytes
    std::vector<char> control_;       // id -> nonzero when CONTROL (skipped)
};

inline bool Tokenizer::load(const ModelLoader& m, std::string& e) {
    std::vector<std::string> toks;
    if (!m.kv_arr_str("tokenizer.ggml.tokens", toks) || toks.empty()) {
        e = "VOXTRAL GGUF missing tokenizer.ggml.tokens";
        return false;
    }
    std::vector<int64_t> types;
    m.kv_arr_int("tokenizer.ggml.token_type", types);  // optional; absent = keep all
    bytes_ = toks;
    control_.assign(toks.size(), 0);
    for (size_t i = 0; i < toks.size() && i < types.size(); ++i)
        if (types[i] == 3) control_[i] = 1;  // CONTROL
    return true;
}

inline std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= bytes_.size()) continue;
        if (control_[(size_t) id]) continue;
        out += bytes_[(size_t) id];
    }
    return out;
}

} // namespace starling::ggml::voxtral
