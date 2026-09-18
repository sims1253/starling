// granite_stage_test.cpp — multi-chunk stage-attribution regression (issue
// #170). Two layers:
//
//   1. Unit checks of StageTiming (granite/stage_timing.hpp): per-chunk stage
//      durations ACCUMULATE across the long-audio loop (the old code kept
//      only the last chunk's array next to a whole-request total), the
//      one-chunk case stays exact, and the rendered log lines are pinned.
//
//   2. An end-to-end run through starling_ggml_granite_decode with a tiny
//      synthesized GGUF (zero weights, 1 encoder layer, 1 qformer layer, 1
//      LLM layer, chunk_seconds=1) — no model download, CPU-only. The test
//      captures stderr under STARLING_GRANITE_TIMING=1 and verifies the
//      emitted summaries: one per-chunk line per chunk, a whole-request line
//      whose stage aggregates cover EVERY chunk, and stage totals that
//      reconcile with the whole-request wall time within the bookkeeping
//      remainder. A tracing-off run emits no GRANITE_STAGE lines at all.
//
// Usage: ./granite_stage_test
#include "granite/stage_timing.hpp"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <cstdlib>
#define SETENV(k, v) _putenv_s(k, v)
#define UNSETENV(k) _putenv_s(k, "")
#else
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#define SETENV(k, v) setenv(k, v, 1)
#define UNSETENV(k) unsetenv(k)
#endif

extern "C" {
void* starling_ggml_granite_load(const char* gguf_path, const char** err_out);
void starling_ggml_granite_free(void* handle);
char* starling_ggml_granite_decode(void* handle, const float* pcm, int64_t n,
                                   const char** err_out);
}

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

constexpr double kRoundingMs = 1.0;  // %.1f rounding across summed lines

// --------------------------------------------------------------------------- //
// StageTiming unit checks.
// --------------------------------------------------------------------------- //
void unit_checks() {
    using starling::ggml::granite::StageTiming;
    using starling::ggml::granite::format_stage_chunk_line;
    using starling::ggml::granite::format_stage_request_line;

    const double c1[3] = {100.0, 10.0, 50.0};  // piece 160.0
    const double c2[3] = {20.0, 30.0, 40.0};   // piece 90.0
    const double c3[3] = {5.0, 15.0, 80.0};    // piece 100.0

    StageTiming st;
    st.add_chunk(c1);
    st.add_chunk(c2);
    st.add_chunk(c3);
    check(st.chunks == 3, "unit: three chunks recorded");
    check(st.total_ms[0] == 125.0 && st.total_ms[1] == 55.0 && st.total_ms[2] == 170.0,
          "unit: stage totals sum EVERY chunk (not the last one)");
    check(st.total_ms[0] != c3[0] && st.total_ms[2] != c3[2],
          "unit: totals are not the last chunk's values");
    check(st.chunk_ms[0] == c3[0] && st.chunk_ms[1] == c3[1] && st.chunk_ms[2] == c3[2],
          "unit: per-chunk slots latch the last chunk");
    check(st.chunk_total_ms() == 100.0, "unit: chunk total sums the last chunk");
    check(st.stages_total_ms() == 350.0, "unit: stages total sums all chunks");
    check(st.bookkeeping_ms(400.0) == 50.0, "unit: bookkeeping = total - stages");
    check(st.bookkeeping_ms(350.0) == 0.0, "unit: zero bookkeeping reconciles");

    StageTiming one;
    one.add_chunk(c1);
    check(one.chunks == 1 && one.total_ms[0] == 100.0 && one.total_ms[1] == 10.0 &&
              one.total_ms[2] == 50.0 && one.stages_total_ms() == 160.0,
          "unit: one-chunk behavior unchanged");

    // Pin the rendered lines (the emission in capi_granite.cpp uses these
    // exact helpers, indexing with StageTiming::chunks after add_chunk — so
    // the line renders the latched last chunk at its 1-based index).
    check(format_stage_chunk_line(st, st.chunks) ==
              "GRANITE_STAGE chunk=3 mel+enc+proj=5.0ms prompt+embeds=15.0ms "
              "gen=80.0ms piece=100.0ms",
          "unit: per-chunk line format");
    check(format_stage_request_line(st, 7.5, 400.0) ==
              "GRANITE_STAGE request chunks=3 audio=7.50s mel+enc+proj=125.0ms "
              "prompt+embeds=55.0ms gen=170.0ms stages=350.0ms bookkeeping=50.0ms "
              "total=400.0ms",
          "unit: whole-request line format");
}

// --------------------------------------------------------------------------- //
// Tiny synthesized granite GGUF (see the header comment). All-zero weights
// keep every activation exactly zero: LN(0)=0, softmax(0) uniform, argmax
// picks token 0 — deterministic output and real, measurable stage graphs.
// --------------------------------------------------------------------------- //
class TinyGraniteFixture {
public:
    std::filesystem::path path;
    ggml_context* gctx_ = nullptr;
    gguf_context* gf_ = nullptr;

    explicit TinyGraniteFixture(const std::filesystem::path& p) : path(p) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        gctx_ = ggml_init({64 << 20, nullptr, false});
        gf_ = gguf_init_empty();
        if (!gctx_ || !gf_) return;
        write_metadata();
        write_tensors();
        wrote_ = gguf_write_to_file(gf_, path.string().c_str(), /*only_meta=*/false);
    }
    ~TinyGraniteFixture() {
        if (gf_) gguf_free(gf_);
        if (gctx_) ggml_free(gctx_);
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    bool wrote() const { return wrote_; }

private:
    bool wrote_ = false;

    void kv_u32(const char* key, uint32_t v) { gguf_set_val_u32(gf_, key, v); }

    void write_metadata() {
        gguf_set_val_str(gf_, "general.architecture", "granite");
        gguf_set_val_u32(gf_, "starling.format_version", 1);
        // Encoder: 1 conformer block at the stock hidden width (the loader
        // pins enc.hidden == heads*head_dim == conv_inner/2), small ffn,
        // tiny block-local attention context.
        kv_u32("granite.enc.layers", 1);
        kv_u32("granite.enc.ffn_dim", 8);
        kv_u32("granite.enc.conv_kernel", 3);
        kv_u32("granite.enc.context_size", 4);
        kv_u32("granite.enc.output_dim", 4);
        // Projector: 2-frame windows -> 1 token each (num_queries ==
        // window/downsample), 1 qformer layer.
        kv_u32("granite.proj.window_size", 2);
        kv_u32("granite.proj.downsample_rate", 1);
        kv_u32("granite.proj.num_queries", 2);
        kv_u32("granite.proj.qformer_layers", 1);
        kv_u32("granite.proj.qformer_heads", 2);
        kv_u32("granite.proj.qformer_intermediate", 8);
        kv_u32("granite.proj.output_dim", 8);  // == llm.hidden
        // Tiny bias-free Qwen trunk (hidden == heads*head_dim == 8).
        kv_u32("granite.llm.hidden_size", 8);
        kv_u32("granite.llm.num_layers", 1);
        kv_u32("granite.llm.num_heads", 1);
        kv_u32("granite.llm.num_kv_heads", 1);
        kv_u32("granite.llm.head_dim", 8);
        kv_u32("granite.llm.intermediate_size", 8);
        kv_u32("granite.llm.vocab_size", 16);
        kv_u32("granite.llm.max_cache_len", 128);
        // Chunk policy: 1 s chunks (max_new_tokens 40 -> token-limited
        // chunk max(0.1, (40-32)/5) = 1.6 s), small decode budgets.
        kv_u32("granite.max_new_tokens", 40);
        gguf_set_val_f64(gf_, "granite.chunk_seconds", 1.0);
        kv_u32("granite.audio_token_id", 3);
        kv_u32("granite.pad_token_id", 0);
        kv_u32("granite.bos_token_id", 2);
        kv_u32("granite.eos_token_id", 2);
        const int64_t prefix[1] = {1};
        const int64_t suffix[1] = {2};
        gguf_set_arr_data(gf_, "granite.prompt_prefix", GGUF_TYPE_INT64, prefix, 1);
        gguf_set_arr_data(gf_, "granite.prompt_suffix", GGUF_TYPE_INT64, suffix, 1);
        // GPT-2 byte-level BPE table (decode-only): 16 printable tokens.
        gguf_set_val_str(gf_, "tokenizer.ggml.model", "gpt2");
        const char* tokens[16] = {"a", "b", "c", "d", "e", "f", "g", "h",
                                  "i", "j", "k", "l", "m", "n", "o", "p"};
        gguf_set_arr_str(gf_, "tokenizer.ggml.tokens", tokens, 16);
    }

    void add_tensor(const char* name, ggml_type type, std::initializer_list<int64_t> ne,
                    float value) {
        std::vector<int64_t> dims(ne);
        ggml_tensor* t = ggml_new_tensor(gctx_, type, (int) dims.size(), dims.data());
        ggml_set_name(t, name);
        if (type == GGML_TYPE_F32) {
            float* p = (float*) t->data;
            for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = value;
        } else {
            ggml_bf16_t* p = (ggml_bf16_t*) t->data;
            const ggml_bf16_t v = ggml_fp32_to_bf16(value);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = v;
        }
        gguf_add_tensor(gf_, t);
    }
    // Linear weight layout is ggml's [in, out]; norms are [n].
    void bf16(const char* name, std::initializer_list<int64_t> ne, float v = 0.0f) {
        add_tensor(name, GGML_TYPE_BF16, ne, v);
    }

    void write_tensors() {
        constexpr int64_t H = 1024;    // encoder hidden (fixed by the loader)
        constexpr int64_t C = 2048;    // enc conv_inner (== 2*hidden, fixed)
        constexpr int64_t F = 8;       // encoder/qformer ffn width
        constexpr int64_t CS = 4;      // encoder attention context
        constexpr int64_t QH = 1024;   // projector hidden
        constexpr int64_t L = 8;       // llm hidden
        constexpr int64_t V = 16;      // llm vocab

        // Mel constants (F32, shapes checked by the frontend).
        add_tensor("audio.mel_filters", GGML_TYPE_F32, {80, 257}, 0.0f);
        add_tensor("audio.mel_window", GGML_TYPE_F32, {512}, 1.0f);

        // Encoder.
        bf16("enc.input_linear.weight", {160, H});
        bf16("enc.input_linear.bias", {H});
        bf16("enc.out.weight", {H, 4});
        bf16("enc.out.bias", {4});
        bf16("enc.out_mid.weight", {4, H});
        bf16("enc.out_mid.bias", {H});
        for (const char* ff : {"ff1", "ff2"}) {
            const std::string p = std::string("enc.blk.0.") + ff;
            bf16((p + "_norm.weight").c_str(), {H}, 1.0f);
            bf16((p + "_norm.bias").c_str(), {H});
            bf16((p + "_up.weight").c_str(), {H, F});
            bf16((p + "_up.bias").c_str(), {F});
            bf16((p + "_down.weight").c_str(), {F, H});
            bf16((p + "_down.bias").c_str(), {H});
        }
        bf16("enc.blk.0.attn_norm.weight", {H}, 1.0f);
        bf16("enc.blk.0.attn_norm.bias", {H});
        bf16("enc.blk.0.attn_q.weight", {H, H});
        bf16("enc.blk.0.attn_kv.weight", {H, 2 * H});
        bf16("enc.blk.0.attn_o.weight", {H, H});
        bf16("enc.blk.0.attn_o.bias", {H});
        bf16("enc.blk.0.rel_pos_bias", {128, CS, CS});  // head_dim x ctx x ctx
        bf16("enc.blk.0.conv_norm.weight", {H}, 1.0f);
        bf16("enc.blk.0.conv_norm.bias", {H});
        bf16("enc.blk.0.conv_up.weight", {H, 2 * C});
        bf16("enc.blk.0.conv_up.bias", {2 * C});
        bf16("enc.blk.0.conv_depth.weight", {C, 3});
        bf16("enc.blk.0.bn_weight", {C}, 1.0f);
        bf16("enc.blk.0.bn_bias", {C});
        bf16("enc.blk.0.bn_mean", {C});
        bf16("enc.blk.0.bn_var", {C}, 1.0f);
        bf16("enc.blk.0.conv_down.weight", {C, H});
        bf16("enc.blk.0.conv_down.bias", {H});
        bf16("enc.blk.0.post_norm.weight", {H}, 1.0f);
        bf16("enc.blk.0.post_norm.bias", {H});

        // Projector.
        bf16("proj.query", {QH, 2});
        bf16("proj.qformer_ln.weight", {QH}, 1.0f);
        bf16("proj.qformer_ln.bias", {QH});
        for (const char* attn : {"self", "cross"}) {
            const std::string p = std::string("proj.blk.0.") + attn;
            for (const char* w : {"q", "k", "v", "out"}) {
                bf16((p + "_" + w + ".weight").c_str(), {QH, QH});
                bf16((p + "_" + w + ".bias").c_str(), {QH});
            }
            bf16((p + "_ln.weight").c_str(), {QH}, 1.0f);
            bf16((p + "_ln.bias").c_str(), {QH});
        }
        bf16("proj.blk.0.ff_up.weight", {QH, F});
        bf16("proj.blk.0.ff_up.bias", {F});
        bf16("proj.blk.0.ff_down.weight", {F, QH});
        bf16("proj.blk.0.ff_down.bias", {QH});
        bf16("proj.blk.0.ff_ln.weight", {QH}, 1.0f);
        bf16("proj.blk.0.ff_ln.bias", {QH});
        bf16("proj.out.weight", {QH, L});
        bf16("proj.out.bias", {L});

        // LLM trunk (bias-free, untied head).
        bf16("llm.embed.weight", {L, V});
        bf16("llm.lm_head.weight", {L, V});
        bf16("llm.final_norm.weight", {L}, 1.0f);
        bf16("llm.blk.0.attn_norm.weight", {L}, 1.0f);
        bf16("llm.blk.0.attn.q.weight", {L, L});
        bf16("llm.blk.0.attn.k.weight", {L, L});
        bf16("llm.blk.0.attn.v.weight", {L, L});
        bf16("llm.blk.0.attn.o.weight", {L, L});
        bf16("llm.blk.0.ffn_norm.weight", {L}, 1.0f);
        bf16("llm.blk.0.ffn.gate.weight", {L, L});
        bf16("llm.blk.0.ffn.up.weight", {L, L});
        bf16("llm.blk.0.ffn.down.weight", {L, L});
    }
};

// --------------------------------------------------------------------------- //
// stderr capture + GRANITE_STAGE line parsing.
// --------------------------------------------------------------------------- //
#ifdef _WIN32
std::string capture_stderr(const std::function<void()>& fn) {
    fn();  // no capture on Windows; the e2e layer is skipped by the caller
    return "";
}
#else
std::string capture_stderr(const std::function<void()>& fn) {
    const char* tmpl = "/tmp/granite_stage_stderr_XXXXXX";
    std::vector<char> name(tmpl, tmpl + std::strlen(tmpl) + 1);
    const int fd = mkstemp(name.data());
    if (fd < 0) return "";
    std::fflush(stderr);
    const int saved = dup(2);
    if (saved < 0) { close(fd); unlink(name.data()); return ""; }
    dup2(fd, 2);
    close(fd);
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    if (const int rfd = open(name.data(), O_RDONLY); rfd >= 0) {
        char buf[4096];
        ssize_t k;
        while ((k = read(rfd, buf, sizeof buf)) > 0) out.append(buf, (size_t) k);
        close(rfd);
    }
    unlink(name.data());
    return out;
}
#endif

std::vector<std::string> stage_lines(const std::string& log, const char* kind) {
    // kind: "chunk" -> per-chunk lines; "request" -> the request summary.
    std::vector<std::string> out;
    const std::string want = std::string("GRANITE_STAGE ") + kind;
    size_t pos = 0;
    while (pos <= log.size()) {
        size_t eol = log.find('\n', pos);
        if (eol == std::string::npos) eol = log.size();
        if (log.compare(pos, want.size(), want) == 0)
            out.push_back(log.substr(pos, eol - pos));
        if (eol == log.size()) break;
        pos = eol + 1;
    }
    return out;
}

// Parse "key=<number>" from a line (ms values carry a trailing "ms").
double field(const std::string& line, const char* key) {
    const std::string want = std::string(key) + "=";
    const size_t at = line.find(want);
    if (at == std::string::npos) return NAN;
    return std::atof(line.c_str() + at + want.size());
}

// --------------------------------------------------------------------------- //
// End-to-end stage-attribution checks through the real decode path.
// --------------------------------------------------------------------------- //
void e2e_checks(void* handle) {
    // 2.5 s of audio at chunk_seconds=1 -> chunks of 1 s, 1 s, 0.5 s (the
    // last zero-padded to a full chunk).
    const int64_t multi_n = 40000, single_n = 8000;
    std::vector<float> pcm(multi_n);
    for (int64_t i = 0; i < multi_n; ++i)
        pcm[i] = 0.2f * std::sinf((float) (2.0 * M_PI * 220.0 * i / 16000.0));

    char* text = nullptr;
    const char* err = nullptr;
    std::string log = capture_stderr([&] {
        err = nullptr;
        text = starling_ggml_granite_decode(handle, pcm.data(), multi_n, &err);
    });
    check(text != nullptr, "e2e: multi-chunk decode succeeded",
          err ? err : "");
    if (!text) return;
    check(std::string(text).find('a') != std::string::npos,
          "e2e: multi-chunk decode returned text");
    std::free(text);

    const auto chunk_lines = stage_lines(log, "chunk");
    const auto request_lines = stage_lines(log, "request");
    check(chunk_lines.size() == 3, "e2e: one summary line per chunk (3)",
          "got " + std::to_string(chunk_lines.size()));
    check(request_lines.size() == 1, "e2e: exactly one whole-request summary");
    if (chunk_lines.size() != 3 || request_lines.size() != 1) {
        std::printf("---- captured ----\n%s------------------\n", log.c_str());
        return;
    }
    for (size_t i = 0; i < 3; ++i)
        check(field(chunk_lines[i], "chunk") == (double) (i + 1),
              "e2e: chunk lines are indexed in order");

    // Every per-chunk line's stages sum to its own piece total.
    for (const std::string& l : chunk_lines)
        check(std::fabs(field(l, "piece") - (field(l, "mel+enc+proj") +
                        field(l, "prompt+embeds") + field(l, "gen"))) < 0.3,
              "e2e: chunk line piece == its three stages");

    // The request aggregates cover EVERY chunk: each stage aggregate matches
    // the sum of that stage over the per-chunk lines (the pre-fix code
    // printed only the LAST chunk's durations here).
    const std::string& req = request_lines[0];
    for (const char* stage : {"mel+enc+proj", "prompt+embeds", "gen"}) {
        double sum = 0;
        for (const std::string& l : chunk_lines) sum += field(l, stage);
        check(std::fabs(sum - field(req, stage)) < kRoundingMs,
              std::string("e2e: request ") + stage + " covers every chunk",
              "sum=" + std::to_string(sum) + " line=" +
                  std::to_string(field(req, stage)));
    }
    check(field(req, "chunks") == 3.0, "e2e: request line reports 3 chunks");

    // Reconciliation: stages == the three aggregates, bookkeeping ==
    // total - stages, and the stages fit inside the whole-request wall time.
    const double stages = field(req, "stages");
    const double total = field(req, "total");
    const double book = field(req, "bookkeeping");
    check(std::fabs(stages - (field(req, "mel+enc+proj") + field(req, "prompt+embeds") +
                              field(req, "gen"))) < 0.5,
          "e2e: request stages == sum of stage aggregates");
    check(std::fabs(book - (total - stages)) < 0.3,
          "e2e: bookkeeping == total - stages");
    check(total > 0.0 && stages <= total + 0.5,
          "e2e: stage totals fit in the whole-request wall time");
    check(book >= -0.5, "e2e: bookkeeping is non-negative",
          "bookkeeping=" + std::to_string(book));

    // One chunk stays correct: a single chunk line and aggregates equal to
    // that chunk's stages.
    std::string log1 = capture_stderr([&] {
        err = nullptr;
        text = starling_ggml_granite_decode(handle, pcm.data(), single_n, &err);
    });
    check(text != nullptr, "e2e: one-chunk decode succeeded", err ? err : "");
    if (text) std::free(text);
    const auto chunk1 = stage_lines(log1, "chunk");
    const auto req1 = stage_lines(log1, "request");
    check(chunk1.size() == 1 && req1.size() == 1,
          "e2e: one-chunk request emits one chunk line + one request line");
    if (chunk1.size() == 1 && req1.size() == 1) {
        check(field(req1[0], "chunks") == 1.0, "e2e: one-chunk request reports 1 chunk");
        for (const char* stage : {"mel+enc+proj", "prompt+embeds", "gen"})
            check(std::fabs(field(req1[0], stage) - field(chunk1[0], stage)) < kRoundingMs,
                  std::string("e2e: one-chunk ") + stage + " aggregate == the chunk");
    }

    // Tracing off: no GRANITE_STAGE output at all.
    UNSETENV("STARLING_GRANITE_TIMING");
    const std::string quiet = capture_stderr([&] {
        err = nullptr;
        text = starling_ggml_granite_decode(handle, pcm.data(), single_n, &err);
    });
    check(text != nullptr, "e2e: tracing-off decode succeeded", err ? err : "");
    if (text) std::free(text);
    check(quiet.find("GRANITE_STAGE") == std::string::npos,
          "e2e: tracing off emits no GRANITE_STAGE lines");
}

} // namespace

int main() {
    unit_checks();
#ifdef _WIN32
    std::printf("[SKIP] e2e stage-attribution checks (POSIX stderr capture)\n");
#else
    // CPU backend + timing gate, before the first load creates the backend.
    SETENV("STARLING_GGML_DEVICE", "cpu");
    SETENV("STARLING_GRANITE_TIMING", "1");
    TinyGraniteFixture fixture(std::filesystem::temp_directory_path() /
                               "granite_stage_test.gguf");
    check(fixture.wrote(), "e2e: synthesized tiny granite GGUF written");
    if (fixture.wrote()) {
        const char* err = nullptr;
        void* handle = starling_ggml_granite_load(fixture.path.string().c_str(), &err);
        check(handle != nullptr, "e2e: tiny granite model loaded", err ? err : "");
        if (handle) {
            e2e_checks(handle);
            starling_ggml_granite_free(handle);
        }
    }
#endif
    std::printf("%s\n", failures ? "GRANITE STAGE FAILED" : "GRANITE STAGE OK");
    return failures ? 1 : 0;
}
