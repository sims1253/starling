// tokenizer.cpp — Qwen2.5 BPE byte-decoder (identical scheme to moss; ARK's
// tokenizer is a standard Qwen2.5 tokenizer stored in the GGUF).
#include "tokenizer.hpp"
#include <algorithm>
namespace starling::ggml::ark {
namespace {
std::vector<uint32_t> cps(const std::string& s) {
    std::vector<uint32_t> o;
    for (size_t i = 0; i < s.size();) {
        uint8_t c = s[i];
        uint32_t v;
        size_t n;
        if (c < 128) { v = c; n = 1; }
        else if ((c & 0xe0) == 0xc0) { v = c & 31; n = 2; }
        else if ((c & 0xf0) == 0xe0) { v = c & 15; n = 3; }
        else if ((c & 0xf8) == 0xf0) { v = c & 7; n = 4; }
        else { ++i; continue; }
        if (i + n > s.size()) break;
        bool ok = true;
        for (size_t j = 1; j < n; ++j) {
            uint8_t q = s[i + j];
            if ((q & 0xc0) != 0x80) { ok = false; break; }
            v = (v << 6) | (q & 63);
        }
        if (!ok) { ++i; continue; }
        o.push_back(v);
        i += n;
    }
    return o;
}
void put(std::string& o, uint32_t c) {
    if (c < 128) o.push_back((char) c);
    else if (c < 2048) { o.push_back((char)(0xc0 | (c >> 6))); o.push_back((char)(0x80 | (c & 63))); }
    else if (c < 65536) {
        o.push_back((char)(0xe0 | (c >> 12)));
        o.push_back((char)(0x80 | ((c >> 6) & 63)));
        o.push_back((char)(0x80 | (c & 63)));
    } else {
        o.push_back((char)(0xf0 | (c >> 18)));
        o.push_back((char)(0x80 | ((c >> 12) & 63)));
        o.push_back((char)(0x80 | ((c >> 6) & 63)));
        o.push_back((char)(0x80 | (c & 63)));
    }
}
std::string valid_utf8(const std::string& b) {
    std::string o;
    for (size_t i = 0; i < b.size();) {
        uint8_t c = b[i];
        size_t n = c < 128 ? 1 : (c >= 0xc2 && c <= 0xdf ? 2 : (c >= 0xe0 && c <= 0xef ? 3 : (c >= 0xf0 && c <= 0xf4 ? 4 : 0)));
        bool ok = n && i + n <= b.size();
        uint32_t v = n ? (c & ((1 << (8 - n - 1)) - 1)) : 0;
        for (size_t j = 1; ok && j < n; ++j) {
            uint8_t q = b[i + j];
            ok = (q & 0xc0) == 0x80;
            v = (v << 6) | (q & 63);
        }
        if (ok) ok = !(v >= 0xd800 && v <= 0xdfff) && v <= 0x10ffff && !(n == 2 && v < 128) && !(n == 3 && v < 2048) && !(n == 4 && v < 65536);
        if (ok) { o.append(b, i, n); i += n; }
        else { put(o, 0xfffd); ++i; }
    }
    return o;
}
} // namespace
bool Tokenizer::load(const ModelLoader& m, const Config&, std::string& e) {
    std::string model;
    if (!m.kv_str("tokenizer.ggml.model", model) || model != "gpt2" ||
        !m.kv_arr_str("tokenizer.ggml.tokens", tokens_)) {
        e = "missing/unsupported GGUF GPT-2 tokenizer";
        return false;
    }
    std::vector<int64_t> types_i;
    m.kv_arr_int("tokenizer.ggml.token_type", types_i);
    types_.assign(types_i.begin(), types_i.end());
    if (types_.size() < tokens_.size()) types_.resize(tokens_.size(), 1);
    // GPT-2 byte-to-unicode inverse map.
    std::vector<int> b;
    for (int i = 33; i <= 126; ++i) b.push_back(i);
    for (int i = 161; i <= 172; ++i) b.push_back(i);
    for (int i = 174; i <= 255; ++i) b.push_back(i);
    std::vector<int> cs = b;
    int n = 0;
    for (int i = 0; i < 256; ++i)
        if (std::find(b.begin(), b.end(), i) == b.end()) { b.push_back(i); cs.push_back(256 + n++); }
    for (size_t i = 0; i < b.size(); ++i) byte_decoder_[(uint32_t) cs[i]] = (uint8_t) b[i];
    return true;
}
std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip) const {
    std::string bytes;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tokens_.size()) continue;
        if (skip && types_[id] == 3) continue;
        for (uint32_t c : cps(tokens_[id])) {
            auto it = byte_decoder_.find(c);
            if (it != byte_decoder_.end()) bytes.push_back((char) it->second);
            else put(bytes, c);
        }
    }
    return valid_utf8(bytes);
}
} // namespace starling::ggml::ark
