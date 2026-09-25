// starling_layout_quant.cpp — quantize MOSS source weights straight into a
// candidate fast-engine layout (#319), and build the desktop weight-only
// eval GGUF.
//
//   pack    source safetensors/GGUF -> .pack file (STARLING_FAST_PACKED)
//   eval    .pack + reference GGUF  -> GGUF with those tensors at F16
//           (the dequantized candidate; run the ggml engine on it for WER)
//   verify  .pack                   -> reload, dequant, compare to source
//
// The source is pinned by content: its sha256 is stored in the .pack header
// and printed, so eval caches key on (model, source hash, descriptor,
// rounding). Layout specs, the packing order and the scale search live in
// cpp/fast/layout.*; this file only orchestrates I/O.
//
// Usage:
//   starling-layout-quant pack --source model.safetensors --out w4.pack \
//       --layout w4g32asym [--rules rules.txt] [--imatrix moss.imx] \
//       [--include regex] [--threads N]
//   starling-layout-quant eval --pack w4.pack --gguf-in src.gguf \
//       --gguf-out eval.gguf
//   starling-layout-quant verify --pack w4.pack [--source model.safetensors]
//
// Rules file (first regex wins, like starling-quantize):
//   # comment
//   default w4g32asym
//   ^llm\.embed\.weight$ w8g16sym

#include "ggml.h"
#include "gguf.h"
#include "imatrix_file.hpp"
#include "layout.hpp"
#include "packed_file.hpp"

#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using starling::fast::LayoutDesc;

// ---------------------------------------------------------------------------
// sha256 (FIPS 180-4), streamed, with a sidecar cache
// ---------------------------------------------------------------------------

struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t data_bits = 0;
    uint8_t buf[64];
    size_t fill = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
                   (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void* data, size_t n) {
        const uint8_t* p = (const uint8_t*)data;
        data_bits += (uint64_t)n * 8;
        while (n) {
            const size_t take = std::min(n, 64 - fill);
            std::memcpy(buf + fill, p, take);
            fill += take; p += take; n -= take;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    std::string hex() {
        Sha256 tmp = *this;
        const uint8_t one = 0x80;
        tmp.update(&one, 1);
        const uint8_t zero = 0;
        while (tmp.fill != 56) tmp.update(&zero, 1);
        // length in bits of the *message* (before this padding)
        uint8_t l[8];
        const uint64_t bits = data_bits;
        for (int i = 0; i < 8; ++i) l[i] = (uint8_t)(bits >> (56 - 8 * i));
        tmp.fill = 56;
        tmp.block(l);   // l is exactly 8 bytes -> one final block
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", tmp.h[i]);
        out[64] = 0;
        return out;
    }
};

std::string file_sha256(const std::string& path) {
    const std::string sidecar = path + ".sha256";
    if (FILE* f = std::fopen(sidecar.c_str(), "r")) {
        char buf[80];
        const char* got = std::fgets(buf, sizeof buf, f);
        std::fclose(f);
        if (got) {
            std::string s(got);
            while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
            if (s.size() == 64) return s;
        }
    }
    Sha256 s;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    std::vector<char> buf(1 << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) s.update(buf.data(), n);
    std::fclose(f);
    const std::string hex = s.hex();
    if (FILE* o = std::fopen(sidecar.c_str(), "w")) {
        std::fputs(hex.c_str(), o);
        std::fclose(o);
    }
    return hex;
}

// ---------------------------------------------------------------------------
// Sources: safetensors or GGUF, read as f32 rows
// ---------------------------------------------------------------------------

struct SrcTensor {
    std::string name;   // GGUF name ("" = not mappable / not wanted)
    std::string src_name;
    uint32_t N = 0, K = 0;
    bool rank1 = false;
    int dtype = 0;      // 0 bf16, 1 f16, 2 f32
    // safetensors
    uint64_t off = 0, end_off = 0;
    // gguf
    const uint8_t* gdata = nullptr;
};

class RowReader {
public:
    explicit RowReader(const std::string& st_path) : st_(st_path) {
        if (!st_.empty()) f_ = std::fopen(st_.c_str(), "rb");
    }
    ~RowReader() { if (f_) std::fclose(f_); }
    // Read rows [r0, r0+n) of tensor t into out (n*K floats).
    bool read(const SrcTensor& t, uint32_t r0, uint32_t n, std::vector<float>* out,
              std::string* err) {
        out->resize((size_t)n * t.K);
        if (t.gdata) {   // GGUF source: dequantize rows
            const size_t row_bytes = t.dtype == 0 ? (size_t)t.K * 2
                                  : t.dtype == 1 ? (size_t)t.K * 2
                                                 : (size_t)t.K * 4;
            for (uint32_t r = 0; r < n; ++r)
                if (!cvt(t, t.gdata + (size_t)(r0 + r) * row_bytes, out->data() + (size_t)r * t.K, err))
                    return false;
            return true;
        }
        if (!f_) { *err = t.src_name + ": no source file"; return false; }
        const size_t row_bytes = t.dtype == 2 ? (size_t)t.K * 4 : (size_t)t.K * 2;
        std::vector<uint8_t> raw((size_t)n * row_bytes);
        if (std::fseek(f_, (long long)t.off + (long long)r0 * row_bytes, SEEK_SET) != 0 ||
            std::fread(raw.data(), 1, raw.size(), f_) != raw.size()) {
            *err = t.src_name + ": short read";
            return false;
        }
        for (uint32_t r = 0; r < n; ++r)
            if (!cvt(t, raw.data() + (size_t)r * row_bytes, out->data() + (size_t)r * t.K, err))
                return false;
        return true;
    }

private:
    static bool cvt(const SrcTensor& t, const uint8_t* row, float* dst, std::string* err) {
        if (t.dtype == 2) {
            std::memcpy(dst, row, (size_t)t.K * 4);
        } else if (t.dtype == 0) {  // bf16
            for (uint32_t k = 0; k < t.K; ++k) {
                const uint32_t bits = (uint32_t)(row[2 * k] | (row[2 * k + 1] << 8)) << 16;
                std::memcpy(&dst[k], &bits, 4);
            }
        } else if (t.dtype == 1) {  // f16
            for (uint32_t k = 0; k < t.K; ++k) {
                const uint16_t h = (uint16_t)(row[2 * k] | (row[2 * k + 1] << 8));
                const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
                const uint32_t exp = (h >> 10) & 31, mant = h & 0x3ff;
                uint32_t bits;
                if (exp == 0) {
                    if (mant == 0) bits = sign;
                    else {
                        int e = -1;
                        uint32_t m = mant;
                        while (!(m & 0x400)) { m <<= 1; --e; }
                        bits = sign | ((uint32_t)(e - 1 + 127 - 10 + 1) << 23) | ((m & 0x3ff) << 13);
                    }
                } else if (exp == 31) {
                    bits = sign | 0x7f800000 | (mant << 13);
                } else {
                    bits = sign | ((exp + 112) << 23) | (mant << 13);
                }
                std::memcpy(&dst[k], &bits, 4);
            }
        } else {
            *err = "bad dtype";
            return false;
        }
        return true;
    }
    std::string st_;
    FILE* f_ = nullptr;
};

// MOSS HF checkpoint name -> GGUF name (mirrors scripts/convert_moss_gguf.py).
std::string gguf_name(const std::string& name, bool* ok) {
    *ok = true;
    auto starts = [&](const char* s) { return name.rfind(s, 0) == 0; };
    static const std::pair<const char*, const char*> enc_repl[] = {
        {"self_attn_layer_norm.", "attn_norm."}, {"final_layer_norm.", "ffn_norm."},
        {"self_attn.q_proj.", "attn.q."}, {"self_attn.k_proj.", "attn.k."},
        {"self_attn.v_proj.", "attn.v."}, {"self_attn.out_proj.", "attn.o."},
        {"fc1.", "ffn.fc1."}, {"fc2.", "ffn.fc2."}};
    static const std::pair<const char*, const char*> llm_repl[] = {
        {"input_layernorm.", "attn_norm."}, {"post_attention_layernorm.", "ffn_norm."},
        {"self_attn.q_proj.", "attn.q."}, {"self_attn.k_proj.", "attn.k."},
        {"self_attn.v_proj.", "attn.v."}, {"self_attn.o_proj.", "attn.o."},
        {"self_attn.q_norm.", "attn.q_norm."}, {"self_attn.k_norm.", "attn.k_norm."},
        {"mlp.gate_proj.", "ffn.gate."}, {"mlp.up_proj.", "ffn.up."},
        {"mlp.down_proj.", "ffn.down."}};
    auto layer = [&](const char* prefix, const char* out_prefix,
                     const std::pair<const char*, const char*>* repl, size_t nrepl) -> std::string {
        std::string rest = name.substr(std::strlen(prefix));
        const size_t dot = rest.find('.');
        if (dot == std::string::npos) return "";
        const std::string i = rest.substr(0, dot);
        if (i.empty() || !std::isdigit((unsigned char)i[0])) return "";
        rest = rest.substr(dot + 1);
        for (size_t j = 0; j < nrepl; ++j)
            if (rest.rfind(repl[j].first, 0) == 0)
                return std::string(out_prefix) + i + "." + repl[j].second + rest.substr(std::strlen(repl[j].first));
        return "";
    };
    if (name == "model.audio_model.conv_out.weight") return "enc.conv_out.weight";
    if (starts("model.audio_model.conv2d")) {
        std::string tail = name.substr(std::strlen("model.audio_model."));
        const size_t at = tail.find("conv2d");
        if (at == std::string::npos) { *ok = false; return name; }
        return tail.replace(at, 6, "enc.conv");
    }
    if (starts("model.audio_model.layers."))
        return layer("model.audio_model.layers.", "enc.blk.", enc_repl, 8);
    if (starts("model.audio_model.ln_post."))
        return "enc.ln_post." + name.substr(std::strlen("model.audio_model.ln_post."));
    if (starts("model.audio_model.proj1."))
        return "enc.proj1." + name.substr(std::strlen("model.audio_model.proj1."));
    if (starts("model.audio_model.proj2."))
        return "enc.proj2." + name.substr(std::strlen("model.audio_model.proj2."));
    if (starts("model.audio_adapter.")) {
        std::string tail = name.substr(std::strlen("model.audio_adapter."));
        const size_t at = tail.find("_proj.");
        if (at != std::string::npos) tail.replace(at, 6, ".");   // gate_proj. -> gate.
        return "adapter." + tail;
    }
    if (name == "model.language_model.embed_tokens.weight") return "llm.embed.weight";
    if (starts("model.language_model.layers."))
        return layer("model.language_model.layers.", "llm.blk.", llm_repl, 11);
    if (starts("model.language_model.norm.")) return "llm.final_norm.weight";
    *ok = false;
    return name;
}

// The tensors the fast engine loads as matrices (moss_engine.cpp): the
// default include set for `pack`.
bool engine_matrix_tensor(const std::string& g) {
    static const std::regex re(
        "^(enc\\.blk\\.\\d+\\.(attn\\.[qkvo]|ffn\\.fc[12])\\.weight"
        "|enc\\.proj[12]\\.weight"
        "|adapter\\.(gate|up|down)\\.weight"
        "|llm\\.blk\\.\\d+\\.(attn\\.[qkvo]|ffn\\.(gate|up|down))\\.weight"
        "|llm\\.embed\\.weight)$");
    return std::regex_search(g, re);
}

// Minimal reader for the safetensors header: a flat JSON object of
// "name": {"dtype": "...", "shape": [a, b], "data_offsets": [x, y]}.
bool read_safetensors(const std::string& path, std::vector<SrcTensor>* out, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { *err = "cannot open " + path; return false; }
    uint64_t hlen;
    if (std::fread(&hlen, 8, 1, f) != 1) { *err = "short header length"; std::fclose(f); return false; }
    if (hlen > (64u << 20)) { *err = "implausible header length"; std::fclose(f); return false; }
    std::vector<char> hdr((size_t)hlen);
    if (std::fread(hdr.data(), 1, hdr.size(), f) != hdr.size()) { *err = "short header"; std::fclose(f); return false; }
    std::fclose(f);

    const char* p = hdr.data();
    const char* const end = p + hdr.size();
    auto skip_ws = [&] { while (p < end && std::isspace((unsigned char)*p)) ++p; };
    auto read_str = [&](std::string* s) {
        skip_ws();
        if (p >= end || *p != '"') return false;
        ++p;
        s->clear();
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) ++p;   // tensor names carry no escapes
            s->push_back(*p++);
        }
        if (p >= end) return false;
        ++p;
        return true;
    };
    auto skip_value = [&] {   // positioned at a value; skip it (nested ok)
        int depth = 0;
        bool in_str = false;
        while (p < end) {
            const char c = *p;
            if (in_str) {
                if (c == '\\') ++p;
                else if (c == '"') in_str = false;
            } else if (c == '"') in_str = true;
            else if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') { if (depth == 0) return; --depth; }
            else if ((c == ',' || c == ';') && depth == 0) return;
            ++p;
        }
    };
    skip_ws();
    if (p >= end || *p != '{') { *err = "not a JSON object"; return false; }
    ++p;
    while (true) {
        skip_ws();
        if (p >= end) { *err = "unterminated object"; return false; }
        if (*p == '}') break;
        std::string key;
        if (!read_str(&key)) { *err = "bad key"; return false; }
        skip_ws();
        if (p >= end || *p != ':') { *err = "expected ':'"; return false; }
        ++p;
        skip_ws();
        if (key != "__metadata__") {
            if (p >= end || *p != '{') { *err = key + ": expected object"; return false; }
            ++p;
            SrcTensor t;
            t.src_name = key;
            while (true) {
                skip_ws();
                if (p < end && *p == '}') { ++p; break; }
                std::string field;
                if (!read_str(&field)) { *err = key + ": bad field"; return false; }
                skip_ws();
                if (p >= end || *p != ':') { *err = key + ": expected ':'"; return false; }
                ++p;
                skip_ws();
                if (field == "dtype") {
                    std::string dt;
                    if (!read_str(&dt)) { *err = key + ": bad dtype"; return false; }
                    if (dt == "BF16") t.dtype = 0;
                    else if (dt == "F16") t.dtype = 1;
                    else if (dt == "F32") t.dtype = 2;
                    else { *err = key + ": unsupported dtype " + dt; return false; }
                } else if (field == "shape" || field == "data_offsets") {
                    if (p >= end || *p != '[') { *err = key + ": expected array"; return false; }
                    ++p;
                    uint64_t v[4] = {0, 0, 0, 0};
                    int i = 0;
                    while (p < end && *p != ']') {
                        if (std::isdigit((unsigned char)*p)) v[i < 4 ? i++ : 3] = std::strtoull(p, (char**)&p, 10);
                        else ++p;
                    }
                    ++p;
                    if (field == "shape") {
                        if (i < 1 || i > 4) { *err = key + ": unsupported rank"; return false; }
                        if (i == 2) { t.N = (uint32_t)v[0]; t.K = (uint32_t)v[1]; }
                        else { t.N = (uint32_t)v[0]; t.K = 1; t.rank1 = true; }  // 1/3/4-D: skipped
                    } else {
                        if (i != 2) { *err = key + ": expected 2 offsets"; return false; }
                        t.off = v[0]; t.end_off = v[1];
                    }
                } else {
                    skip_value();
                }
                skip_ws();
                if (p < end && *p == ',') ++p;
            }
            if (t.end_off <= t.off || t.N == 0 || t.K == 0) { *err = key + ": bad entry"; return false; }
            t.off += 8 + hlen;   // data region follows the header
            if (!t.rank1) out->push_back(t);   // biases/norms: not matrix material
        } else {
            skip_value();
        }
        skip_ws();
        if (p < end && *p == ',') ++p;
    }
    return true;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s pack --source F --out F.pack --layout SPEC [--rules F]\n"
        "             [--imatrix F] [--include regex] [--threads N]\n"
        "       %s eval --pack F.pack --gguf-in F.gguf --gguf-out F.gguf [--dtype bf16|f16|f32]\n"
        "       %s verify --pack F.pack [--source F] [--rows N]\n"
        "       %s cmp --source F --imatrix F --tensor NAME --ggml-type T --layout SPEC [--rows N]\n"
        "             (quantizes the first N rows both ways; prints the errors)\n",
        argv0, argv0, argv0, argv0);
}

struct Args {
    std::string source, out, layout = "w4g32asym", rules, imatrix, include, pack, gguf_in,
                gguf_out, dtype = "auto", tensor, ggml_type = "q4_0";
    uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
    uint32_t verify_rows = 8;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", s.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (s == "--source") a.source = next();
        else if (s == "--out") a.out = next();
        else if (s == "--layout") a.layout = next();
        else if (s == "--rules") a.rules = next();
        else if (s == "--imatrix") a.imatrix = next();
        else if (s == "--include") a.include = next();
        else if (s == "--pack") a.pack = next();
        else if (s == "--gguf-in") a.gguf_in = next();
        else if (s == "--gguf-out") a.gguf_out = next();
        else if (s == "--threads") a.threads = (uint32_t)std::atoi(next().c_str());
        else if (s == "--rows") a.verify_rows = (uint32_t)std::atoi(next().c_str());
        else if (s == "--dtype") a.dtype = next();
        else if (s == "--tensor") a.tensor = next();
        else if (s == "--ggml-type") a.ggml_type = next();
        else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); std::exit(1); }
    }
    return a;
}

void parallel_for(size_t n, uint32_t nthreads, const std::function<void(size_t)>& fn) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (uint32_t t = 0; t < std::min<size_t>(nthreads, std::max<size_t>(n, 1)); ++t)
        th.emplace_back([&] {
            size_t i;
            while ((i = next.fetch_add(1)) < n) fn(i);
        });
    for (auto& t : th) t.join();
}

// ---------------------------------------------------------------------------
// pack
// ---------------------------------------------------------------------------

int cmd_pack(const Args& a) {
    if (a.source.empty() || a.out.empty()) { usage("starling-layout-quant"); return 1; }
    LayoutDesc def;
    std::string err;
    if (!starling::fast::layout_from_string(a.layout, &def, &err)) {
        std::fprintf(stderr, "error: --layout %s\n", err.c_str());
        return 1;
    }
    std::vector<std::pair<std::regex, LayoutDesc>> rules;
    if (!a.rules.empty()) {
        FILE* f = std::fopen(a.rules.c_str(), "r");
        if (!f) { std::fprintf(stderr, "error: cannot open rules %s\n", a.rules.c_str()); return 1; }
        char line[512];
        while (std::fgets(line, sizeof line, f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            const auto first = s.find_first_not_of(" \t");
            if (first == std::string::npos || s[first] == '#') continue;
            const auto last = s.find_last_of(" \t");
            if (last == std::string::npos) {
                std::fprintf(stderr, "error: rule line not `<pattern> <spec>`: %s\n", s.c_str());
                return 1;
            }
            const std::string pat = s.substr(first, last - first);
            const std::string spec = s.substr(s.find_first_not_of(" \t", last));
            LayoutDesc d;
            if (!starling::fast::layout_from_string(spec, &d, &err)) {
                std::fprintf(stderr, "error: rule %s: %s\n", pat.c_str(), err.c_str());
                return 1;
            }
            if (pat == "default") def = d;
            else rules.emplace_back(std::regex(pat), d);
        }
        std::fclose(f);
    }
    const std::regex include_re(a.include.empty() ? ".*" : a.include);

    std::vector<SrcTensor> srcs;
    std::string st_path;
    ggml_context* gguf_ctx = nullptr;
    gguf_context* gguf_in = nullptr;
    if (a.source.size() > 5 && a.source.substr(a.source.size() - 5) == ".gguf") {
        gguf_init_params ip = {/*.no_alloc =*/ false, /*.ctx =*/ &gguf_ctx};
        gguf_in = gguf_init_from_file(a.source.c_str(), ip);
        if (!gguf_in) { std::fprintf(stderr, "error: cannot open %s\n", a.source.c_str()); return 1; }
        const int64_t n = gguf_get_n_tensors(gguf_in);
        for (int64_t id = 0; id < n; ++id) {
            const char* name = gguf_get_tensor_name(gguf_in, id);
            ggml_tensor* t = name ? ggml_get_tensor(gguf_ctx, name) : nullptr;
            if (!t || ggml_n_dims(t) != 2) continue;
            SrcTensor st;
            st.name = name;
            st.src_name = name;
            st.N = (uint32_t)(ggml_nelements(t) / t->ne[0]);
            st.K = (uint32_t)t->ne[0];
            if (t->type == GGML_TYPE_BF16) st.dtype = 0;
            else if (t->type == GGML_TYPE_F16) st.dtype = 1;
            else if (t->type == GGML_TYPE_F32) st.dtype = 2;
            else {
                std::fprintf(stderr, "error: %s: source GGUF must hold bf16/f16/f32 tensors\n", name);
                return 1;
            }
            st.gdata = (const uint8_t*)t->data;
            srcs.push_back(st);
        }
    } else {
        st_path = a.source;
        if (!read_safetensors(a.source, &srcs, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        for (SrcTensor& t : srcs) {
            bool ok = false;
            const std::string g = gguf_name(t.src_name, &ok);
            t.name = ok ? g : "";
        }
    }

    starling::ggml::ImatrixMap imap;
    if (!a.imatrix.empty()) {
        imap = starling::ggml::imatrix_read(a.imatrix);
        if (imap.empty()) { std::fprintf(stderr, "error: imatrix %s unreadable\n", a.imatrix.c_str()); return 1; }
        std::fprintf(stderr, "imatrix: %zu tensors from %s\n", imap.size(), a.imatrix.c_str());
    }

    const std::string sha = file_sha256(a.source);
    if (sha.empty()) { std::fprintf(stderr, "error: cannot hash %s\n", a.source.c_str()); return 1; }
    std::fprintf(stderr, "source %s sha256 %s\n", a.source.c_str(), sha.c_str());

    FILE* out = std::fopen(a.out.c_str(), "wb");
    if (!out) { std::fprintf(stderr, "error: cannot write %s\n", a.out.c_str()); return 1; }
    std::vector<SrcTensor> want;
    for (const SrcTensor& t : srcs)
        if (!t.name.empty() && engine_matrix_tensor(t.name) && std::regex_search(t.name, include_re))
            want.push_back(t);
    if (want.empty()) { std::fprintf(stderr, "error: no tensors matched\n"); return 1; }

    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, out); };
    auto w64 = [&](uint64_t v) { std::fwrite(&v, 8, 1, out); };
    auto wblob = [&](const std::vector<uint8_t>& v) {
        std::fwrite(v.data(), 1, v.size(), out);
        static const uint8_t pad[3] = {0, 0, 0};
        std::fwrite(pad, 1, (4 - (v.size() & 3)) & 3, out);
    };
    std::fprintf(stderr, "magic+header\n");
    std::fwrite("SFPK", 1, 4, out);
    w32(1);
    std::vector<char> src_field(65, 0);
    std::memcpy(src_field.data(), sha.c_str(), std::min<size_t>(sha.size(), 64));
    std::fwrite(src_field.data(), 1, 65, out);
    std::vector<char> rnd_field(16, 0);
    std::memcpy(rnd_field.data(), (a.imatrix.empty() ? "rtn" : "imatrix"), 15);
    std::fwrite(rnd_field.data(), 1, 16, out);
    w32((uint32_t)want.size());

    RowReader reader(st_path);
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    uint64_t total_in = 0, total_out = 0;
    for (const SrcTensor& t : want) {
        LayoutDesc d = def;
        for (const auto& r : rules)
            if (std::regex_search(t.name, r.first)) { d = r.second; break; }
        if (!d.valid() || t.K % d.group != 0) {
            std::fprintf(stderr, "error: %s: layout %s does not fit K=%u\n", t.name.c_str(),
                         starling::fast::layout_to_string(d).c_str(), t.K);
            return 1;
        }
        const std::string spec = starling::fast::layout_to_string(d);
        const uint64_t cb = starling::fast::layout_code_bytes(d, t.K);
        const uint64_t sb = starling::fast::layout_scale_bytes(d, t.K);
        const bool has_super = d.scale_dtype == starling::fast::ScaleDtype::U8Super;

        const float* im = nullptr;
        auto it = imap.find(t.name);
        if (it != imap.end() && it->second.values.size() == (size_t)t.K)
            im = it->second.values.data();

        std::vector<uint8_t> codes((size_t)cb * t.N), scales((size_t)sb * t.N);
        std::vector<uint16_t> supers(has_super ? t.N : 0);
        std::vector<double> rel_errs;
        std::mutex io_mu;
        std::string first_err;

        const uint32_t block_rows = 4096;
        for (uint32_t r0 = 0; r0 < t.N; r0 += block_rows) {
            const uint32_t n = std::min(block_rows, t.N - r0);
            std::vector<float> block;
            {
                std::lock_guard<std::mutex> lk(io_mu);
                if (!first_err.empty()) break;
                if (!reader.read(t, r0, n, &block, &err)) { first_err = err; break; }
            }
            rel_errs.resize(rel_errs.size() + n);
            double* errs = rel_errs.data() + rel_errs.size() - n;
            parallel_for(n, a.threads, [&](uint32_t i) {
                if (!first_err.empty()) return;
                const uint32_t r = r0 + i;
                uint8_t* cs = codes.data() + (size_t)r * cb;
                uint8_t* sc = scales.data() + (size_t)r * sb;
                uint16_t* sp = has_super ? &supers[r] : nullptr;
                uint8_t* u8 = has_super ? sc : nullptr;   // u8 scales live in `scales`
                errs[i] = starling::fast::layout_quant_row(d, block.data() + (size_t)i * t.K,
                                                           t.K, im, cs, sc, sp, u8);
            });
        }
        if (!first_err.empty()) { std::fprintf(stderr, "error: %s\n", first_err.c_str()); return 1; }
        double rel_err_sum = 0;
        for (double e : rel_errs) rel_err_sum += e;

        const uint32_t name_len = (uint32_t)t.name.size();
        w32(name_len);
        std::fwrite(t.name.data(), 1, name_len, out);
        const uint32_t spec_len = (uint32_t)spec.size();
        w32(spec_len);
        std::fwrite(spec.data(), 1, spec_len, out);
        w32(t.N);
        w32(t.K);
        w64((uint64_t)cb * t.N);
        w64((uint64_t)sb * t.N);
        w64(has_super ? (uint64_t)t.N * 4 : 0);
        wblob(codes);
        wblob(scales);
        if (has_super) {
            std::vector<uint8_t> raw((size_t)t.N * 4, 0);
            for (uint32_t r = 0; r < t.N; ++r) { raw[4 * r] = supers[r] & 0xff; raw[4 * r + 1] = supers[r] >> 8; }
            std::fwrite(raw.data(), 1, raw.size(), out);   // already word aligned
        }
        total_in += (uint64_t)t.N * t.K * 4;
        total_out += (uint64_t)cb * t.N + (uint64_t)sb * t.N + (has_super ? (uint64_t)t.N * 4 : 0);
        std::fprintf(stderr, "  %-44s %6ux%-5u %-14s rel-rms %.5f\n", t.name.c_str(), t.N, t.K,
                     spec.c_str(), rel_err_sum / t.N);
    }
    std::fclose(out);
    if (gguf_in) { gguf_free(gguf_in); ggml_free(gguf_ctx); }
    const double dt = (double)clock() / CLOCKS_PER_SEC - t0;
    std::printf("%s: %zu tensors, %.2f GB f32 -> %.2f GB packed (%.2f%%), %.1f s\n",
                a.out.c_str(), want.size(), total_in / 1e9, total_out / 1e9,
                100.0 * total_out / std::max<uint64_t>(total_in, 1), dt);
    return 0;
}

// ---------------------------------------------------------------------------
// eval: .pack + reference GGUF -> GGUF with those tensors dequantized to F16
// ---------------------------------------------------------------------------

// The ggml block type that holds this layout's values exactly (same f16
// scales, same codes), when one exists — an eval GGUF built from it measures
// the packed numerics with zero store rounding.
ggml_type layout_ggml_type(const starling::fast::LayoutDesc& d) {
    using namespace starling::fast;
    if (d.scale_dtype != ScaleDtype::F16 || d.order != 0) return GGML_TYPE_COUNT;
    if (d.bits == 4 && d.group == 32) {
        if (d.symmetric) return GGML_TYPE_Q4_0;   // w = d*(q-8)
        return GGML_TYPE_Q4_1;                    // w = d*q + m
    }
    if (d.bits == 8 && d.group == 32 && d.symmetric) return GGML_TYPE_Q8_0;
    return GGML_TYPE_COUNT;
}

int cmd_eval(const Args& a) {
    if (a.pack.empty() || a.gguf_in.empty() || a.gguf_out.empty()) {
        usage("starling-layout-quant");
        return 1;
    }
    std::string err;
    auto packed = starling::fast::PackedWeights::load(a.pack, err);
    if (!packed) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    ggml_context* ctx_in = nullptr;
    gguf_init_params ip = {/*.no_alloc =*/ false, /*.ctx =*/ &ctx_in};
    gguf_context* in = gguf_init_from_file(a.gguf_in.c_str(), ip);
    if (!in) { std::fprintf(stderr, "error: cannot open %s\n", a.gguf_in.c_str()); return 1; }
    const int64_t n_tensors = gguf_get_n_tensors(in);
    ggml_context* ctx_out = ggml_init({ggml_tensor_overhead() * (size_t)(n_tensors + 16), nullptr, true});
    std::vector<std::unique_ptr<char[]>> owned;
    gguf_context* gout = gguf_init_empty();
    gguf_set_kv(gout, in);
    gguf_set_val_str(gout, "starling.quant.tool", "starling-layout-quant");
    gguf_set_val_str(gout, "starling.quant.level", ("layout-eval-" + a.dtype).c_str());
    gguf_set_val_str(gout, "starling.quant.calibration", packed->rounding().c_str());
    const ggml_type eval_type = a.dtype == "f16" ? GGML_TYPE_F16
                              : a.dtype == "f32" ? GGML_TYPE_F32
                                                 : GGML_TYPE_BF16;

    size_t replaced = 0, bytes_out = 0;
    for (int64_t id = 0; id < n_tensors; ++id) {
        const char* name_c = gguf_get_tensor_name(in, id);
        ggml_tensor* t = name_c ? ggml_get_tensor(ctx_in, name_c) : nullptr;
        if (!t) continue;
        const std::string name(name_c);
        const int nd = ggml_n_dims(t);
        const size_t n_el = (size_t)ggml_nelements(t);
        ggml_tensor* dst = nullptr;
        if (packed->has(name)) {
            const starling::fast::PackedTensor* pt = packed->find(name);
            if (!pt) { std::fprintf(stderr, "error: %s vanished\n", name.c_str()); return 1; }
            if (pt->K != (uint32_t)t->ne[0] ||
                pt->N != (uint32_t)(n_el / t->ne[0])) {
                std::fprintf(stderr, "error: %s: packed shape %ux%u != GGUF\n", name.c_str(), pt->N, pt->K);
                return 1;
            }
            const uint64_t cb = starling::fast::layout_code_bytes(pt->desc, pt->K);
            const uint64_t sb = starling::fast::layout_scale_bytes(pt->desc, pt->K);
            const ggml_type native = a.dtype == "auto" ? layout_ggml_type(pt->desc) : GGML_TYPE_COUNT;
            size_t dst_bytes = 0;
            if (native != GGML_TYPE_COUNT) {
                // Exact ggml blocks (same f16 scales, same codes): zero store
                // rounding and a natively-readable eval GGUF.
                dst = ggml_new_tensor(ctx_out, native, nd, t->ne);
                dst_bytes = ggml_nbytes(dst);
                owned.emplace_back(new char[dst_bytes]);
                uint8_t* out = (uint8_t*)owned.back().get();
                const size_t row_bytes = ggml_row_size(native, pt->K);
                parallel_for(pt->N, a.threads, [&](size_t r) {
                    const uint8_t* codes = pt->codes.data() + r * cb;
                    const uint8_t* scales = pt->scales.data() + r * sb;
                    const uint16_t* sup = pt->super.empty() ? nullptr : &pt->super[r];
                    const uint8_t* u8s = pt->desc.scale_dtype == starling::fast::ScaleDtype::U8Super
                                           ? scales : nullptr;
                    uint8_t* row = out + r * row_bytes;
                    const uint32_t groups = pt->K / 32;
                    for (uint32_t g = 0; g < groups; ++g) {
                        const float s = starling::fast::layout_scale_at(pt->desc, scales, sup, u8s, g);
                        const uint16_t s16 = ggml_fp32_to_fp16(s);
                        if (native == GGML_TYPE_Q8_0) {
                            row[g * 34] = (uint8_t)(s16 & 0xff);
                            row[g * 34 + 1] = (uint8_t)(s16 >> 8);
                            for (int j = 0; j < 32; ++j)
                                row[g * 34 + 2 + j] = (uint8_t)starling::fast::layout_decode_code(
                                    pt->desc, codes, g * 32 + j);
                        } else {
                            // Q4_0: d, qs[16] (byte j: low nibble = code j, high = j+16)
                            // Q4_1: d, m, qs[16]
                            const bool asym = native == GGML_TYPE_Q4_1;
                            const size_t bo = asym ? (size_t)g * 20 : (size_t)g * 18;
                            row[bo] = (uint8_t)(s16 & 0xff);
                            row[bo + 1] = (uint8_t)(s16 >> 8);
                            if (asym) {
                                const float o = starling::fast::layout_offset_at(pt->desc, scales, g);
                                const uint16_t o16 = ggml_fp32_to_fp16(o);
                                row[bo + 2] = (uint8_t)(o16 & 0xff);
                                row[bo + 3] = (uint8_t)(o16 >> 8);
                            }
                            const size_t qo = bo + (asym ? 4 : 2);
                            for (int j = 0; j < 16; ++j) {
                                const uint8_t lo = (uint8_t)(starling::fast::layout_decode_code(
                                    pt->desc, codes, g * 32 + j) & 15);
                                const uint8_t hi = (uint8_t)(starling::fast::layout_decode_code(
                                    pt->desc, codes, g * 32 + 16 + j) & 15);
                                row[qo + j] = (uint8_t)(lo | (hi << 4));
                            }
                        }
                    }
                });
            } else {
            dst = ggml_new_tensor(ctx_out, eval_type, nd, t->ne);
            const size_t dst_bytes_f = ggml_nbytes(dst);
            dst_bytes = dst_bytes_f;
            owned.emplace_back(new char[dst_bytes]);
            uint16_t* h16 = (uint16_t*)owned.back().get();
            parallel_for(pt->N, a.threads, [&](size_t r) {
                const uint8_t* codes = pt->codes.data() + r * cb;
                const uint8_t* scales = pt->scales.data() + r * sb;
                const uint16_t* sup = pt->super.empty() ? nullptr : &pt->super[r];
                const uint8_t* u8s = pt->desc.scale_dtype == starling::fast::ScaleDtype::U8Super ? scales : nullptr;
                for (uint32_t k = 0; k < pt->K; ++k) {
                    const float v = starling::fast::layout_dequant(pt->desc, codes, scales, sup, u8s, k);
                    if (eval_type == GGML_TYPE_F32)
                        ((float*)h16)[(size_t)r * pt->K + k] = v;
                    else
                        h16[(size_t)r * pt->K + k] =
                            eval_type == GGML_TYPE_F16 ? (uint16_t)ggml_fp32_to_fp16(v)
                                                       : ggml_fp32_to_bf16(v).bits;
                }
            });
            }
            ++replaced;
            bytes_out += dst_bytes;
        } else {
            dst = ggml_new_tensor(ctx_out, (ggml_type)t->type, nd, t->ne);
            const size_t src_bytes = ggml_nbytes(t);
            owned.emplace_back(new char[src_bytes]);
            std::memcpy(owned.back().get(), t->data, src_bytes);
            bytes_out += src_bytes;
        }
        ggml_set_name(dst, name_c);
        gguf_add_tensor(gout, dst);
        gguf_set_tensor_data(gout, name_c, owned.back().get());
    }
    if (replaced) gguf_set_val_str(gout, "starling.numeric_profile", "quantized");
    if (!gguf_write_to_file(gout, a.gguf_out.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "error: failed to write %s\n", a.gguf_out.c_str());
        return 1;
    }
    std::printf("%s: %zu/%lld tensors dequantized to %s, %.2f GB\n", a.gguf_out.c_str(),
                replaced, (long long)n_tensors, a.dtype.c_str(), bytes_out / 1e9);
    gguf_free(gout);
    ggml_free(ctx_out);
    gguf_free(in);
    ggml_free(ctx_in);
    return 0;
}

// ---------------------------------------------------------------------------
// verify: reload a .pack, check sizes/dequant against the source
// ---------------------------------------------------------------------------

int cmd_verify(const Args& a) {
    if (a.pack.empty()) { usage("starling-layout-quant"); return 1; }
    std::string err;
    auto packed = starling::fast::PackedWeights::load(a.pack, err);
    if (!packed) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "pack: %zu tensors, %.1f MB, rounding=%s source=%s\n", packed->size(),
                 packed->bytes() / 1e6, packed->rounding().c_str(), packed->source_hash().c_str());
    if (a.source.empty()) {
        std::printf("verify: structure OK (%zu tensors)\n", packed->size());
        return 0;
    }
    // Compare the first --rows rows of every tensor against the source.
    std::vector<SrcTensor> srcs;
    std::string st_path;
    if (a.source.size() > 5 && a.source.substr(a.source.size() - 5) == ".gguf") {
        std::fprintf(stderr, "error: verify supports safetensors sources\n");
        return 1;
    }
    st_path = a.source;
    if (!read_safetensors(a.source, &srcs, &err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::unordered_map<std::string, const SrcTensor*> by_name;
    for (const SrcTensor& t : srcs) {
        bool ok = false;
        const std::string g = gguf_name(t.src_name, &ok);
        if (ok) by_name[g] = &t;
    }
    RowReader reader(st_path);
    int bad = 0;
    packed->for_each([&](const starling::fast::PackedTensor& pt) {
        auto it = by_name.find(pt.name);
        if (it == by_name.end()) {
            std::printf("FAIL %s: not in source\n", pt.name.c_str());
            ++bad;
            return;
        }
        const SrcTensor& t = *it->second;
        if (t.N != pt.N || t.K != pt.K) {
            std::printf("FAIL %s: shape %ux%u vs source %ux%u\n", pt.name.c_str(), pt.N, pt.K, t.N, t.K);
            ++bad;
            return;
        }        const uint32_t rows = std::min(a.verify_rows, pt.N);
        std::vector<float> src;
        if (!reader.read(t, 0, rows, &src, &err)) {
            std::printf("FAIL %s: %s\n", pt.name.c_str(), err.c_str());
            ++bad;
            return;
        }
        const uint64_t cb = starling::fast::layout_code_bytes(pt.desc, pt.K);
        const uint64_t sb = starling::fast::layout_scale_bytes(pt.desc, pt.K);
        double max_rel = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* codes = pt.codes.data() + (size_t)r * cb;
            const uint8_t* scales = pt.scales.data() + (size_t)r * sb;
            const uint16_t* sup = pt.super.empty() ? nullptr : &pt.super[r];
            const uint8_t* u8s = pt.desc.scale_dtype == starling::fast::ScaleDtype::U8Super ? scales : nullptr;
            double e2 = 0, w2 = 0;
            for (uint32_t k = 0; k < pt.K; ++k) {
                const double d = src[(size_t)r * pt.K + k] -
                                 (double)starling::fast::layout_dequant(pt.desc, codes, scales, sup, u8s, k);
                e2 += d * d;
                w2 += (double)src[(size_t)r * pt.K + k] * src[(size_t)r * pt.K + k];
            }
            max_rel = std::max(max_rel, std::sqrt(e2 / (w2 + 1e-30)));
        }
        std::printf("%s %-44s %-14s first-%u-rows rel-rms %.5f\n", max_rel < 0.2 ? "ok  " : "FAIL",
                    pt.name.c_str(), starling::fast::layout_to_string(pt.desc).c_str(), rows, max_rel);
        if (max_rel >= 0.2) ++bad;
    });
    std::printf(bad ? "verify: %d FAILED\n" : "verify: all OK\n", bad);
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// cmp: quantize real tensor rows two ways, report the errors
// ---------------------------------------------------------------------------

int cmd_cmp(const Args& a) {
    if (a.source.empty() || a.tensor.empty()) { usage("starling-layout-quant"); return 1; }
    std::vector<SrcTensor> srcs;
    std::string err, st_path = a.source;
    if (!read_safetensors(a.source, &srcs, &err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const SrcTensor* hit = nullptr;
    for (const SrcTensor& t : srcs) {
        bool ok = false;
        const std::string g = gguf_name(t.src_name, &ok);
        if (ok && g == a.tensor) { hit = &t; break; }
    }
    if (!hit) { std::fprintf(stderr, "error: tensor %s not found\n", a.tensor.c_str()); return 1; }
    starling::ggml::ImatrixMap imap;
    if (!a.imatrix.empty()) {
        imap = starling::ggml::imatrix_read(a.imatrix);
        if (imap.empty()) { std::fprintf(stderr, "error: imatrix unreadable\n"); return 1; }
    }
    const float* im = nullptr;
    auto it = imap.find(a.tensor);
    if (it != imap.end() && it->second.values.size() == (size_t)hit->K)
        im = it->second.values.data();

    const uint32_t rows = std::max(1u, a.verify_rows);
    RowReader reader(st_path);
    std::vector<float> w;
    if (!reader.read(*hit, 0, rows, &w, &err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const uint32_t K = hit->K;

    LayoutDesc d;
    if (!starling::fast::layout_from_string(a.layout, &d, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    ggml_type gt = GGML_TYPE_Q4_0;
    { struct N { const char* n; ggml_type t; }; static const N tab[] = {
        {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1}, {"q8_0", GGML_TYPE_Q8_0},
        {"q6_k", GGML_TYPE_Q6_K}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K}};
      for (const auto& e : tab) if (a.ggml_type == e.n) gt = e.t; }

    auto report = [&](const std::vector<float>& dq, const char* name) {
        double we = 0, ue = 0, w2 = 0, mx = 0;
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t k = 0; k < K; ++k) {
                const double dd = w[(size_t)r * K + k] - dq[(size_t)r * K + k];
                we += im ? (double)im[k] * dd * dd : dd * dd;
                ue += dd * dd;
                w2 += (double)w[(size_t)r * K + k] * w[(size_t)r * K + k];
                mx = std::max(mx, std::fabs(dd));
            }
        std::printf("%-10s weighted-MSE %.6e unweighted %.6e rel-rms %.5f max|err| %.5f\n",
                    name, we, ue, std::sqrt(ue / w2), mx);
    };

    // ggml reference quantization
    {
        const size_t rb = ggml_row_size(gt, K);
        std::vector<uint8_t> q(rb * rows);
        ggml_quantize_init(gt);
        ggml_quantize_chunk(gt, w.data(), q.data(), 0, rows, K, im);
        std::vector<float> dq((size_t)rows * K);
        ggml_get_type_traits(gt)->to_float(q.data(), dq.data(), (int64_t)rows * K);
        report(dq, ("ggml " + a.ggml_type).c_str());
    }
    // layout quantization
    {
        const uint64_t cb = starling::fast::layout_code_bytes(d, K);
        const uint64_t sb = starling::fast::layout_scale_bytes(d, K);
        const bool sup = d.scale_dtype == starling::fast::ScaleDtype::U8Super;
        std::vector<float> dq((size_t)rows * K);
        for (uint32_t r = 0; r < rows; ++r) {
            std::vector<uint8_t> codes(cb), scales(sb);
            std::vector<uint16_t> super(sup ? 1 : 0);
            starling::fast::layout_quant_row(d, w.data() + (size_t)r * K, K, im, codes.data(),
                                             scales.data(), super.data(), sup ? scales.data() : nullptr);
            for (uint32_t k = 0; k < K; ++k)
                dq[(size_t)r * K + k] = starling::fast::layout_dequant(
                    d, codes.data(), scales.data(), super.data(), sup ? scales.data() : nullptr, k);
        }
        report(dq, a.layout.c_str());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    const std::string cmd = argv[1];
    const Args a = parse_args(argc, argv);
    if (cmd == "pack") return cmd_pack(a);
    if (cmd == "eval") return cmd_eval(a);
    if (cmd == "verify") return cmd_verify(a);
    if (cmd == "cmp") return cmd_cmp(a);
    usage(argv[0]);
    return 1;
}
