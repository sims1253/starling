// tokenizer.hpp — Voxtral decode-only raw-byte tokenizer.
//
// The GGUF carries a gpt2-model table with NO merges: ids 0..num_special-1 are
// the tekken specials and ids num_special.. are byte pieces. GGUF strings are
// UTF-8, so a byte >= 0x80 cannot ride through a piece string verbatim (it
// would double-encode); the converter therefore stores pieces with non-ASCII
// bytes in llama.cpp's <0xXX> escaped form and marks them TokenType.BYTE, and
// load() unescapes them back to raw bytes. Decoding concats the bytes of ids
// >= num_special (ALL specials are skipped, matching stock
// skip_special_tokens=True, not just CONTROL-typed ones). There is no
// BPE/merge step, which is why the shared BpeTokenizer (an encode+decode
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

    // Test/diagnostic accessors.
    size_t vocab_size() const { return bytes_.size(); }
    const std::string& piece(int32_t id) const { return bytes_[(size_t) id]; }

private:
    std::vector<std::string> bytes_;  // id -> raw bytes (BYTE pieces unescaped)
    int64_t num_special_ = 1000;      // ids below this are skipped on decode
};

// <0xXX> -> single raw byte, for BYTE-typed pieces.
inline std::string unescape_byte_pieces(const std::string& s) {
    auto hexv = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '<' && i + 6 <= s.size() && s[i + 1] == '0' && s[i + 2] == 'x' &&
            hexv(s[i + 3]) >= 0 && hexv(s[i + 4]) >= 0 && s[i + 5] == '>') {
            out += (char) (hexv(s[i + 3]) * 16 + hexv(s[i + 4]));
            i += 6;
        } else {
            out += s[i++];
        }
    }
    return out;
}

inline bool Tokenizer::load(const ModelLoader& m, std::string& e) {
    std::vector<std::string> toks;
    if (!m.kv_arr_str("tokenizer.ggml.tokens", toks) || toks.empty()) {
        e = "VOXTRAL GGUF missing tokenizer.ggml.tokens";
        return false;
    }
    std::vector<int64_t> types;
    m.kv_arr_int("tokenizer.ggml.token_type", types);  // optional; absent = keep all
    m.kv_int("voxtral.num_special", num_special_);     // optional; default 1000
    bytes_ = std::move(toks);
    for (size_t i = 0; i < bytes_.size() && i < types.size(); ++i)
        if (types[i] == 6)  // BYTE: <0xXX>-escaped raw bytes
            bytes_[i] = unescape_byte_pieces(bytes_[i]);
    return true;
}

inline std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id < num_special_ || (size_t) id >= bytes_.size()) continue;
        out += bytes_[(size_t) id];
    }
    return out;
}

} // namespace starling::ggml::voxtral
