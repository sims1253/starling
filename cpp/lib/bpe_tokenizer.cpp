// bpe_tokenizer.cpp — GPT-2/Qwen BPE byte-decoder + text encoder (see
// bpe_tokenizer.hpp).
#include "bpe_tokenizer.hpp"
#include <algorithm>
#include <map>
namespace starling::ggml::lib {
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

// ---------------------------------------------------------------------------
// Encoder-side: the Qwen pre-tokenizer regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// hand-rolled over codepoints (leftmost-alternative order preserved).
//
// Codepoint classes: ASCII is exact. Non-ASCII is classified as \p{L} except
// a small explicit punctuation/symbol set (the em/en dashes, curly quotes,
// ellipsis and a few currency/degree signs that realistically appear in ASR
// output); the tokenizer's NFC normalizer is likewise a no-op for ASCII.
// English ASR transcripts are ASCII, so encoder output is byte-exact there;
// exotic non-ASCII scripts may pre-token-bound differently than HF.
// ---------------------------------------------------------------------------
enum CpClass { CP_L, CP_N, CP_SPACE, CP_NEWLINE, CP_OTHER };

bool cp_is_space(uint32_t c) {
    if (c < 128) return c == ' ' || (c >= 0x09 && c <= 0x0d);
    switch (c) {
        case 0x85: case 0xa0: case 0x1680: case 0x2028: case 0x2029:
        case 0x202f: case 0x205f: case 0x3000: return true;
        default: return c >= 0x2000 && c <= 0x200a;
    }
}

bool cp_is_non_ascii_other(uint32_t c) {
    switch (c) {  // common non-ASCII punctuation/symbols in English text
        case 0x2010: case 0x2011: case 0x2012: case 0x2013: case 0x2014:  // hyphens/dashes
        case 0x2018: case 0x2019: case 0x201a:  // single quotes
        case 0x201c: case 0x201d: case 0x201e:  // double quotes
        case 0x2026:                            // ellipsis
        case 0x00a9: case 0x00ae: case 0x00b0: case 0x00b1: case 0x00d7: case 0x00f7:
        case 0x20ac: case 0x00a3: case 0x00a5:  // (c)(r) degree +- x/ EUR GBP JPY
            return true;
        default: return false;
    }
}

CpClass cp_class(uint32_t c) {
    if (c < 128) {
        if (c == '\r' || c == '\n') return CP_NEWLINE;
        if (c == ' ' || (c >= 0x09 && c <= 0x0d)) return CP_SPACE;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return CP_L;
        if (c >= '0' && c <= '9') return CP_N;
        return CP_OTHER;
    }
    if (cp_is_space(c)) return CP_SPACE;
    if (cp_is_non_ascii_other(c)) return CP_OTHER;
    return CP_L;
}

bool lower_eq_ascii(uint32_t c, char want) {
    if (c >= 'A' && c <= 'Z') c += 32;
    return c == (uint32_t) want;
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

bool BpeTokenizer::load(const ModelLoader& m, std::string& e) {
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
    // Encoder-side tables: byte -> byte-level char (UTF-8), piece -> id,
    // special string -> id. First occurrence wins for duplicate pieces
    // (matches HF: the base vocab is unique; duplicates only shadow reused
    // added-token ids, which the special path below resolves instead).
    for (size_t i = 0; i < b.size(); ++i) {
        put(byte_encoder_utf8_[(uint8_t) b[i]], (uint32_t) cs[i]);
    }
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (!types_.empty() && types_[i] == 3) special_ids_[tokens_[i]] = (int32_t) i;
        else if (!token_ids_.count(tokens_[i])) token_ids_[tokens_[i]] = (int32_t) i;
    }
    // Merge ranks ("left right" pairs). Absent merges disable encode().
    std::vector<std::string> merges;
    if (m.kv_arr_str("tokenizer.ggml.merges", merges)) {
        for (size_t r = 0; r < merges.size(); ++r) {
            const std::string& s = merges[r];
            size_t sp = s.find(' ');
            if (sp == std::string::npos || s.find(' ', sp + 1) != std::string::npos)
                continue;
            merge_ranks_[{s.substr(0, sp), s.substr(sp + 1)}] = (int32_t) r;
        }
    }
    return true;
}

std::string BpeTokenizer::decode(const std::vector<int32_t>& ids, bool skip) const {
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

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
namespace {

// The Qwen pre-tokenizer regex, one alternative at a time (order matters).
size_t pretoken_len(const std::vector<uint32_t>& v, size_t i) {
    const size_t n = v.size();
    auto cls = [&](size_t k) { return cp_class(v[k]); };

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (v[i] == '\'' && i + 1 < n) {
        uint32_t a = v[i + 1];
        auto one = [&](char w) { return lower_eq_ascii(a, w); };
        if (i + 2 < n) {
            uint32_t b2 = v[i + 2];
            auto two = [&](char w1, char w2) {
                return lower_eq_ascii(a, w1) && lower_eq_ascii(b2, w2);
            };
            if (two('r', 'e') || two('v', 'e') || two('l', 'l')) return 3;
        }
        if (one('s') || one('t') || one('m') || one('d')) return 2;
    }

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+  — the optional prefix is ANY non-CR/LF
    // letter/digit char (space, tab, punctuation); it only merges when
    // letters follow (else the ? backtracks to empty).
    if (cls(i) == CP_L) {
        size_t j = i;
        while (j < n && cls(j) == CP_L) ++j;
        return j - i;
    }
    if ((cls(i) == CP_OTHER || cls(i) == CP_SPACE) && i + 1 < n &&
        cls(i + 1) == CP_L) {
        size_t j = i + 1;
        while (j < n && cls(j) == CP_L) ++j;
        return j - i;
    }

    // 3. \p{N}  (single digit — Qwen splits digit runs)
    if (cls(i) == CP_N) return 1;

    // 4. " ?"[^\s\p{L}\p{N}]+[\r\n]*
    {
        size_t j = i;
        if (v[j] == ' ' && j + 1 < n && cls(j + 1) == CP_OTHER) ++j;
        if (cls(j) == CP_OTHER) {
            while (j < n && cls(j) == CP_OTHER) ++j;
            while (j < n && cls(j) == CP_NEWLINE) ++j;
            return j - i;
        }
    }

    // 5. \s*[\r\n]+  — whitespace up to and including the last newline run.
    if (cls(i) == CP_SPACE || cls(i) == CP_NEWLINE) {
        size_t j = i;
        while (j < n && (cls(j) == CP_SPACE || cls(j) == CP_NEWLINE)) ++j;
        size_t last_nl = 0;
        for (size_t k = i; k < j; ++k)
            if (cls(k) == CP_NEWLINE) last_nl = k + 1;
        if (last_nl) return last_nl - i;
        // 6. \s+(?!\S) — whitespace not followed by non-space: give one back.
        if (j < n) return (j - i > 1) ? j - i - 1 : 0;
        // 7. \s+ — trailing whitespace at end of text.
        return j - i;
    }
    return 0;  // unreachable for well-formed classes; caller falls back
}

std::string cps_to_utf8(const std::vector<uint32_t>& v, size_t from, size_t to) {
    std::string s;
    for (size_t k = from; k < to; ++k) put(s, v[k]);
    return s;
}

} // namespace

bool BpeTokenizer::encode(const std::string& text, std::vector<int32_t>& ids,
                          std::string& e) const {
    if (merge_ranks_.empty() || token_ids_.empty()) {
        e = "tokenizer has no merges/pieces (GGUF lacks tokenizer.ggml.merges)";
        return false;
    }
    ids.clear();
    if (text.empty()) return true;
    const std::vector<uint32_t> v = cps(text);

    auto bpe_piece = [&](const std::string& piece, std::vector<int32_t>& out) -> bool {
        // Byte-level: each byte becomes its byte-level unicode char, then
        // adjacent lowest-rank pairs merge until none left (GPT-2 loop).
        std::vector<std::string> sym;
        sym.reserve(piece.size());
        for (unsigned char b : piece) sym.push_back(byte_encoder_utf8_[b]);
        while (sym.size() > 1) {
            size_t best_i = sym.size();
            int32_t best_rank = -1;
            for (size_t k = 0; k + 1 < sym.size(); ++k) {
                auto it = merge_ranks_.find({sym[k], sym[k + 1]});
                if (it != merge_ranks_.end() &&
                    (best_rank < 0 || it->second < best_rank)) {
                    best_rank = it->second;
                    best_i = k;
                }
            }
            if (best_i >= sym.size()) break;
            sym[best_i] += sym[best_i + 1];
            sym.erase(sym.begin() + (std::ptrdiff_t) best_i + 1);
        }
        for (const std::string& s : sym) {
            auto it = token_ids_.find(s);
            if (it == token_ids_.end()) {
                e = "BPE piece not in vocabulary: " + s;
                return false;
            }
            out.push_back(it->second);
        }
        return true;
    };

    size_t i = 0;
    while (i < v.size()) {
        // Special-token LONGEST match first (all-ASCII template tokens).
        int32_t special_id = -1;
        size_t special_len = 0;
        for (const auto& [s, id] : special_ids_) {
            const size_t len = s.size();
            if (len <= special_len || i + len > v.size()) continue;
            bool ok = true;
            for (size_t j = 0; j < len; ++j)
                if (v[i + j] != (uint32_t) (uint8_t) s[j]) { ok = false; break; }
            if (ok) { special_id = id; special_len = len; }
        }
        if (special_id >= 0) {
            ids.push_back(special_id);
            i += special_len;
            continue;
        }

        size_t len = pretoken_len(v, i);
        if (len == 0) len = 1;  // fallback: consume one codepoint
        if (!bpe_piece(cps_to_utf8(v, i, i + len), ids)) return false;
        i += len;
    }
    return true;
}
} // namespace starling::ggml::lib
