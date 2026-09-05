// starling_quantize.cpp — calibrated GGUF quantizer for Starling ASR models.
//
// Post-processes a Starling GGUF (any dtype) into a quantized one using the
// same ggml block-quantization llama.cpp uses, optionally weighted by an
// importance matrix collected on real audio (STARLING_IMATRIX via the engine,
// see cpp/runtime/imatrix.hpp). Model-agnostic: tensors are classified by
// name/shape rules, so the same tool covers parakeet today and the larger
// engines (audex, moss, qwen3-asr, ...) after their loaders allowlist the
// quantized dtypes.
//
// Usage:
//   starling-quantize --input in.gguf --output out.gguf --quant q5_k_m \
//       [--imatrix path] [--recipe path] [--shrink-f16] [--list] [--quiet]
//
// Levels (llama.cpp-style mixes; the bump group — attention value/out
// projections, FFN down-projections and the encoder->joint projection — gets
// one notch more precision):
//   q8_0   everything quantizable at Q8_0
//   q6_k   everything quantizable at Q6_K
//   q5_k_s everything quantizable at Q5_K
//   q5_k_m bump group at Q6_K, rest at Q5_K
//   q4_k_s everything quantizable at Q4_K
//   q4_k_m bump group at Q6_K, rest at Q4_K
//
// Tensors are quantization candidates when they are 2-D "*.weight" linears
// consumed via ggml_mul_mat by the engine. Kept at the source dtype:
//   - conv weights ("conv." in the name) — engine casts/convs them as F16
//   - the mel filterbank/window ("preprocessor.*")
//   - the prediction-network embedding (host-side raw F32 table)
//   - pos_bias_u/v and every 1-D tensor (norms, biases, BN statistics)
// A candidate whose row length is not a multiple of the block size (e.g. the
// 640-row joint/LSTM linears vs Q4_K's 256 block) falls back to Q8_0, and to
// the source dtype if even that does not fit.
//
// --recipe overrides the level with a line-based file for sensitivity sweeps:
//   # comment
//   default q4_k
//   self_attn\.linear_q\.weight$ q6_k
//   ^joint\. q8_0
// Patterns are ECMAScript regexes matched with std::regex_search, first match
// wins, applied on top of the same candidate gating as the named levels.
//
// --shrink-f16 additionally stores the KEPT tensors (except the embedding and
// preprocessor constants, which must stay exact F32) as F16.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "imatrix_file.hpp"

namespace {

// ---------------------------------------------------------------------------
// Levels + types
// ---------------------------------------------------------------------------

struct Bump { const char* pattern; ggml_type type; };

struct Level {
    const char* name;
    ggml_type base;
    ggml_type bump;      // type for the bump group (unused when bump == base)
    bool has_bump;
};

// The bump group: value/out attention projections, FFN down-projections, the
// encoder->joint projection. These carry the most signal per byte (matching
// llama.cpp's empirical _M mixes).
const char* kBumpRegex =
    "self_attn\\.linear_(v|out)\\.weight$"
    "|feed_forward[12]\\.linear2\\.weight$"
    "|joint\\.enc\\.weight$";

const std::vector<Level> kLevels = {
    {"q8_0",   GGML_TYPE_Q8_0,  GGML_TYPE_Q8_0,  false},
    {"q6_k",   GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K,  false},
    {"q5_k_s", GGML_TYPE_Q5_K,  GGML_TYPE_Q5_K,  false},
    {"q5_k_m", GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K,  true},
    {"q4_k_s", GGML_TYPE_Q4_K,  GGML_TYPE_Q4_K,  false},
    {"q4_k_m", GGML_TYPE_Q4_K,  GGML_TYPE_Q6_K,  true},
    {"q3_k_m", GGML_TYPE_Q3_K,  GGML_TYPE_Q5_K,  true},
    {"q2_k",   GGML_TYPE_Q2_K,  GGML_TYPE_Q4_K,  true},
    // IQ formats below ~2.6 bpw exist only as calibrated builds in practice:
    // ggml REFUSES to quantize IQ2_XXS / IQ2_XS / IQ1_S without an importance
    // matrix (ggml_quantize_requires_imatrix). IQ2_S and IQ1_M accept a null
    // matrix but are meaningfully better with one — keep --imatrix mandatory
    // for the whole IQ range here.
    {"iq2_s",   GGML_TYPE_IQ2_S,   GGML_TYPE_IQ2_S,   false},
    {"iq2_xs",  GGML_TYPE_IQ2_XS,  GGML_TYPE_IQ2_XS,  false},
    {"iq2_xxs", GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, false},
    {"iq1_m",   GGML_TYPE_IQ1_M,   GGML_TYPE_IQ1_M,   false},
    {"iq1_s",   GGML_TYPE_IQ1_S,   GGML_TYPE_IQ1_S,   false},
    {"iq4_xs",  GGML_TYPE_IQ4_XS,  GGML_TYPE_IQ4_XS,  false},
};

bool parse_type(const std::string& s, ggml_type* out) {
    struct N { const char* n; ggml_type t; };
    static const N tab[] = {
        {"q8_0", GGML_TYPE_Q8_0}, {"q6_k", GGML_TYPE_Q6_K},
        {"q5_k", GGML_TYPE_Q5_K}, {"q4_k", GGML_TYPE_Q4_K},
        {"q3_k", GGML_TYPE_Q3_K}, {"q2_k", GGML_TYPE_Q2_K},
        {"iq4_xs", GGML_TYPE_IQ4_XS}, {"iq2_s", GGML_TYPE_IQ2_S},
        {"iq2_xs", GGML_TYPE_IQ2_XS}, {"iq2_xxs", GGML_TYPE_IQ2_XXS},
        {"iq1_m", GGML_TYPE_IQ1_M}, {"iq1_s", GGML_TYPE_IQ1_S},
        {"q5_0", GGML_TYPE_Q5_0}, {"q4_0", GGML_TYPE_Q4_0},
        {"f16", GGML_TYPE_F16},   {"f32", GGML_TYPE_F32},
    };
    for (const auto& e : tab)
        if (s == e.n) { *out = e.t; return true; }
    return false;
}

const char* type_name(ggml_type t) {
    const char* n = ggml_type_name(t);
    return n ? n : "?";
}

// ---------------------------------------------------------------------------
// Candidate gating
// ---------------------------------------------------------------------------

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool is_candidate(const std::string& name, int64_t n_dims) {
    if (n_dims != 2) return false;                 // norms/biases/BN stats/convs
    const bool is_linear_weight =
        ends_with(name, ".weight") ||
        name.find("weight_ih_l") != std::string::npos ||  // LSTM input weights
        name.find("weight_hh_l") != std::string::npos;    // LSTM hidden weights
    if (!is_linear_weight) return false;           // biases, pos_bias_u/v, ...
    if (name.find("conv") != std::string::npos) return false;       // cast/conv paths
    if (name.rfind("preprocessor.", 0) == 0) return false;          // mel constants
    if (name.find("embed") != std::string::npos) return false;      // host F32 table
    return true;
}

bool shrink_eligible(const std::string& name, int64_t n_dims) {
    // Only conv WEIGHTS. They are the bulk of the kept residue (~300 MB for
    // parakeet) and the engine consumes them through F16 paths anyway
    // (pointwise convs are ggml_cast to F16, depthwise/subsampling convs take
    // F16 kernels). Everything else stays F32: 1-D biases/norms/BN stats feed
    // ggml_add/ggml_mul broadcasts which reject mixed dtypes, pos_bias_u/v
    // likewise, and the embedding table is a raw host read.
    return n_dims >= 3 && name.find("conv") != std::string::npos &&
           ends_with(name, ".weight");
}

// Largest type in {requested, q8_0} whose block size divides the row length;
// the source dtype signals "keep" if nothing fits.
ggml_type compat_type(ggml_type want, int64_t n_per_row, ggml_type src) {
    if (n_per_row % (int64_t)ggml_blck_size(want) == 0) return want;
    if (want != GGML_TYPE_Q8_0 && n_per_row % (int64_t)ggml_blck_size(GGML_TYPE_Q8_0) == 0)
        return GGML_TYPE_Q8_0;
    return src;
}

// ---------------------------------------------------------------------------
// Recipes
// ---------------------------------------------------------------------------

struct Recipe {
    std::vector<std::pair<std::regex, ggml_type>> rules;
    ggml_type def = GGML_TYPE_F32;
    bool have_default = false;
};

bool load_recipe(const std::string& path, Recipe* r) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "error: cannot open recipe %s\n", path.c_str());
        return false;
    }
    char line[1024];
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        const auto first = s.find_first_not_of(" \t");
        if (first == std::string::npos || s[first] == '#') continue;
        // Split on the last whitespace run: "<pattern> <type>".
        const auto last = s.find_last_of(" \t");
        if (last == std::string::npos) {
            std::fprintf(stderr, "error: recipe line not `<pattern> <type>`: %s\n", s.c_str());
            std::fclose(f);
            return false;
        }
        std::string pat = s.substr(first, last - first);
        std::string typ = s.substr(s.find_first_not_of(" \t", last));
        ggml_type t;
        if (!parse_type(typ, &t)) {
            std::fprintf(stderr, "error: unknown type `%s` in recipe\n", typ.c_str());
            std::fclose(f);
            return false;
        }
        if (pat == "default") {
            r->def = t;
            r->have_default = true;
        } else {
            r->rules.emplace_back(std::regex(pat), t);
        }
    }
    std::fclose(f);
    if (!r->have_default) {
        std::fprintf(stderr, "error: recipe needs a `default <type>` line\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Source tensor -> F32 rows
// ---------------------------------------------------------------------------

std::vector<float> to_f32(const ggml_tensor* t) {
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)t->data, out.data(), (int64_t)n);
    } else if (t->type == GGML_TYPE_BF16) {
        const uint16_t* src = (const uint16_t*)t->data;
        for (size_t i = 0; i < n; ++i) {
            uint32_t bits = (uint32_t)src[i] << 16;
            std::memcpy(&out[i], &bits, sizeof(float));
        }
    } else {
        std::fprintf(stderr, "error: tensor %s has unsupported source dtype %s\n",
                     t->name, type_name((ggml_type)t->type));
        out.clear();
    }
    return out;
}

struct Args {
    std::string input, output, quant = "q5_k_m", imatrix, recipe;
    bool shrink_f16 = false, list_only = false, quiet = false;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --input in.gguf --output out.gguf --quant q5_k_m\n"
        "                 [--imatrix file] [--recipe file] [--shrink-f16]\n"
        "                 [--list] [--quiet]\n",
        argv0);
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", what);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--input") args.input = next("--input");
        else if (a == "--output") args.output = next("--output");
        else if (a == "--quant") args.quant = next("--quant");
        else if (a == "--imatrix") args.imatrix = next("--imatrix");
        else if (a == "--recipe") args.recipe = next("--recipe");
        else if (a == "--shrink-f16") args.shrink_f16 = true;
        else if (a == "--list") args.list_only = true;
        else if (a == "--quiet") args.quiet = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "error: unknown arg %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }
    if (args.input.empty() || (args.output.empty() && !args.list_only)) {
        usage(argv[0]);
        return 1;
    }

    // Resolve the level / recipe.
    const Level* level = nullptr;
    for (const auto& l : kLevels)
        if (args.quant == l.name) { level = &l; break; }
    Recipe recipe;
    bool use_recipe = false;
    if (!args.recipe.empty()) {
        if (!load_recipe(args.recipe, &recipe)) return 1;
        use_recipe = true;
    } else {
        if (!level) {
            std::fprintf(stderr, "error: unknown quant level `%s` (no --recipe given)\n",
                         args.quant.c_str());
            return 1;
        }
    }
    std::unique_ptr<std::regex> bump_re;
    if (!use_recipe && level->has_bump)
        bump_re.reset(new std::regex(kBumpRegex));

    // IQ formats refuse to quantize without an importance matrix — check the
    // resolved types up front rather than deep inside ggml_quantize_chunk.
    {
        std::vector<ggml_type> used;
        if (use_recipe) {
            used.push_back(recipe.def);
            for (const auto& r : recipe.rules) used.push_back(r.second);
        } else {
            used.push_back(level->base);
            if (level->has_bump) used.push_back(level->bump);
        }
        for (ggml_type t : used) {
            if (ggml_quantize_requires_imatrix(t) && args.imatrix.empty()) {
                std::fprintf(stderr,
                    "error: %s requires an importance matrix; pass --imatrix "
                    "(collect one with STARLING_IMATRIX, see docs/quantization.md)\n",
                    type_name(t));
                return 1;
            }
        }
    }

    // imatrix (optional).
    starling::ggml::ImatrixMap imap;
    if (!args.imatrix.empty()) {
        imap = starling::ggml::imatrix_read(args.imatrix);
        if (imap.empty()) {
            std::fprintf(stderr, "error: imatrix %s is empty/unreadable\n", args.imatrix.c_str());
            return 1;
        }
        if (!args.quiet)
            std::fprintf(stderr, "imatrix: %zu tensors from %s\n",
                         imap.size(), args.imatrix.c_str());
    }

    // Open the input (mmap-backed weight context, same as ModelLoader).
    ggml_context* ctx_in = nullptr;
    gguf_init_params ip = {/*.no_alloc =*/ false, /*.ctx =*/ &ctx_in};
    gguf_context* in = gguf_init_from_file(args.input.c_str(), ip);
    if (!in) {
        std::fprintf(stderr, "error: failed to open %s\n", args.input.c_str());
        return 1;
    }

    // Output meta context: no_alloc tensors whose ->data we point at owned
    // host buffers until gguf_write_to_file.
    const int64_t n_tensors = gguf_get_n_tensors(in);
    ggml_context* ctx_out = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(n_tensors + 16),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    });
    std::vector<std::unique_ptr<char[]>> owned;  // data buffers live until write

    gguf_context* out = nullptr;
    if (!args.list_only) {
        out = gguf_init_empty();
        gguf_set_kv(out, in);
        gguf_set_val_str(out, "starling.quant.tool", "starling-quantize");
        gguf_set_val_str(out, "starling.quant.level",
                         use_recipe ? args.recipe.c_str() : args.quant.c_str());
        gguf_set_val_str(out, "starling.quant.calibration",
                         args.imatrix.empty() ? "uniform" : "imatrix");
    }

    std::unordered_set<ggml_type> qinit;
    size_t bytes_in = 0, bytes_out = 0, quantized_cnt = 0, kept_cnt = 0,
           imatrix_hits = 0, candidates = 0;

    for (int64_t id = 0; id < n_tensors; ++id) {
        const char* name_c = gguf_get_tensor_name(in, id);
        ggml_tensor* t = name_c ? ggml_get_tensor(ctx_in, name_c) : nullptr;
        if (!t) continue;
        const std::string name(name_c);
        const int n_dims = ggml_n_dims(t);
        const int64_t k = t->ne[0];  // row length (input channels)
        const size_t src_bytes = ggml_nbytes(t);
        bytes_in += src_bytes;

        // Decide the target type.
        ggml_type want = (ggml_type)t->type;
        if (is_candidate(name, n_dims)) {
            candidates++;
            if (use_recipe) {
                want = compat_type(recipe.def, k, (ggml_type)t->type);
                for (const auto& rule : recipe.rules)
                    if (std::regex_search(name, rule.first)) {
                        want = compat_type(rule.second, k, (ggml_type)t->type);
                        break;
                    }
            } else if (bump_re && std::regex_search(name, *bump_re)) {
                want = compat_type(level->bump, k, (ggml_type)t->type);
            } else {
                want = compat_type(level->base, k, (ggml_type)t->type);
            }
        } else if (args.shrink_f16 && t->type == GGML_TYPE_F32 && shrink_eligible(name, n_dims)) {
            want = GGML_TYPE_F16;
        }

        // Materialize the output tensor + data. gguf_add_tensor records the
        // name/type/shape; gguf_set_tensor_data binds the data pointer (and
        // requires the name to exist, so it must follow the add).
        ggml_tensor* dst_t = ggml_new_tensor(ctx_out, want, n_dims, t->ne);
        ggml_set_name(dst_t, name_c);
        if (out) gguf_add_tensor(out, dst_t);
        if (want != (ggml_type)t->type) {
            std::vector<float> f32 = to_f32(t);
            if (f32.empty()) return 1;
            const size_t dst_bytes = ggml_nbytes(dst_t);
            owned.emplace_back(new char[dst_bytes]);
            if (want == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t*)owned.back().get(),
                                      (int64_t)ggml_nelements(t));
            } else {
                // Quantized: weight the block search by the imatrix when the
                // entry matches this tensor's row width.
                const float* im = nullptr;
                auto it = imap.find(name);
                if (it != imap.end() && it->second.values.size() == (size_t)k) {
                    im = it->second.values.data();
                    imatrix_hits++;
                }
                if (ggml_quantize_requires_imatrix(want) && im == nullptr) {
                    // ggml_quantize_chunk GGML_ABORTs (release builds too)
                    // for the imatrix-mandatory types on a null matrix —
                    // fail cleanly instead, naming the tensor.
                    std::fprintf(stderr,
                                 "error: %s requires an imatrix entry but %s "
                                 "has none matching its row width %lld\n",
                                 type_name(want), name.c_str(), (long long)k);
                    return 1;
                }
                if (qinit.insert(want).second) ggml_quantize_init(want);
                const int64_t nrows = (int64_t)ggml_nelements(t) / (k > 0 ? k : 1);
                const size_t got = ggml_quantize_chunk(want, f32.data(), owned.back().get(),
                                                       /*start=*/0, nrows, k, im);
                if (got != dst_bytes) {
                    std::fprintf(stderr, "error: quantize %s wrote %zu/%zu bytes\n",
                                 name.c_str(), got, dst_bytes);
                    return 1;
                }
            }
            if (out) gguf_set_tensor_data(out, name_c, owned.back().get());
            bytes_out += dst_bytes;
            quantized_cnt++;
        } else {
            owned.emplace_back(new char[src_bytes]);
            std::memcpy(owned.back().get(), t->data, src_bytes);
            if (out) gguf_set_tensor_data(out, name_c, owned.back().get());
            bytes_out += src_bytes;
            kept_cnt++;
        }
        if (args.list_only) {
            std::printf("%-64s ne[0]=%-6lld %-6s -> %-6s %9zu -> %9zu bytes%s\n",
                        name.c_str(), (long long)k, type_name((ggml_type)t->type),
                        type_name(want), src_bytes,
                        (size_t)ggml_nbytes(dst_t),
                        (imap.count(name) ? " [imatrix]" : ""));
        }
    }

    if (args.list_only) {
        std::printf("\ncandidates=%zu quantized=%zu kept=%zu imatrix-covered=%zu\n",
                    candidates, quantized_cnt, kept_cnt, imatrix_hits);
        return 0;
    }

    if (!gguf_write_to_file(out, args.output.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "error: failed to write %s\n", args.output.c_str());
        return 1;
    }
    ggml_quantize_free();

    if (!args.quiet) {
        std::printf("%s: %lld tensors -> %s\n", args.output.c_str(),
                    (long long)n_tensors, args.output.c_str());
        std::printf("  level=%s calibration=%s shrink_f16=%d\n",
                    use_recipe ? "(recipe)" : args.quant.c_str(),
                    args.imatrix.empty() ? "uniform" : "imatrix",
                    args.shrink_f16 ? 1 : 0);
        std::printf("  candidates=%zu quantized/converted=%zu kept=%zu imatrix-covered=%zu\n",
                    candidates, quantized_cnt, kept_cnt, imatrix_hits);
        std::printf("  %.1f MB -> %.1f MB (%.1f%%)\n",
                    bytes_in / 1e6, bytes_out / 1e6, 100.0 * bytes_out / bytes_in);
    }

    gguf_free(out);
    ggml_free(ctx_out);
    gguf_free(in);
    ggml_free(ctx_in);
    return 0;
}
