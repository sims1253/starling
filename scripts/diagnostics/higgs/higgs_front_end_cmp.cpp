// Higgs conv front-end comparator: replicates the TWO front-ends of
// cpp/higgs/audio_encoder.cpp side by side on the SAME backend and compares
// them stage by stage. With the engine's BF16 oracle boundaries applied on
// both sides, the expected residual is +-1 bf16 ulp accumulation tie-flips
// (host conv accumulates in f64, the graph conv in f32; values landing on
// opposite sides of a bf16 rounding boundary flip by one ulp) — the engine
// tolerates this class, and the fixed CPU path produces correct transcripts.
//   host   — host_conv1d (f64 accumulation, f32 GELU) x2 + transpose (the
//            CPU / debug path);
//   graph  — ggml_conv_1d + bias + gelu_erf_bf16 + permute/cont/f32 (the
//            GPU fused-graph path).
// The engine-on-CPU produces garbage ("inaudience" loops) while the GPU path
// is byte-exact vs torch; forcing the host path on GPU reproduces the CPU
// encoder values, so the two front-ends must disagree. This harness finds
// the first disagreeing stage and the magnitude.
//
// Build (repo root): see scripts/diagnostics/vulkan/README.md for the
// include/lib/rpath pattern; link build/libstarling_ggml.so + ggml libs.
#include "higgs/loader.hpp"
#include "higgs/mel.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "lib/graph_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace starling::ggml;
using lib::bf16;
using lib::f32;

// --- faithful copy of higgs host_conv1d (audio_encoder.cpp:324) -----------
static std::vector<float> host_conv1d(const ModelLoader& ml, const std::vector<float>& in,
                                      int64_t IC, int64_t L, const std::string& wname,
                                      int stride, std::string& err) {
    std::vector<float> wf = lib::read_f32(ml, (wname + ".weight").c_str());
    if (wf.empty()) { err = wname + ".weight missing"; return {}; }
    std::vector<float> bf = lib::read_f32(ml, (wname + ".bias").c_str());
    if (bf.empty()) { err = wname + ".bias missing"; return {}; }
    const int64_t OC = (int64_t) bf.size();
    const int64_t K = 3, p = 1;
    const int64_t OL = (L + 2 * p - K) / stride + 1;
    std::vector<float> x((size_t) IC * (L + 2 * p), 0.0f);
    for (int64_t c = 0; c < IC; ++c)
        for (int64_t t = 0; t < L; ++t)
            x[(size_t) c * (L + 2 * p) + (t + p)] = in[(size_t) c * L + t];
    std::vector<float> y((size_t) OC * OL, 0.0f);
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t t = 0; t < OL; ++t) {
            double acc = (double) bf[(size_t) oc];
            for (int64_t c = 0; c < IC; ++c) {
                for (int64_t k = 0; k < K; ++k) {
                    int64_t src = t * stride + k;
                    acc += (double) wf[((size_t) oc * IC + c) * K + k] *
                           (double) x[(size_t) c * (L + 2 * p) + src];
                }
            }
            float v = (float) acc;
            v = 0.5f * v * (1.0f + std::erf(v / (float) M_SQRT2));
            y[(size_t) oc * OL + t] = v;
        }
    }
    return y;  // [OC, OL] oc-contig, GELU applied
}

static void cmp(const char* name, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { printf("%-14s SIZE %zu vs %zu\n", name, a.size(), b.size()); return; }
    double md = 0; size_t at = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > md) { md = d; at = i; }
    }
    printf("%-14s n=%-8zu maxdiff=%-12.6g at=%zu (host=%g graph=%g)%s\n",
           name, a.size(), md, at, a[at], b[at], md > 1e-2 ? "  <-- DIVERGES" : "");
}

int main() {
    std::string e;
    higgs::HiggsModel m;
    if (!m.load("models/higgs-audio-v3-bf16-exact.gguf", e)) { fprintf(stderr, "load: %s\n", e.c_str()); return 2; }
    const auto& ec = m.config.encoder;

    FILE* f = fopen("tests/fixtures/short.wav", "rb");
    if (!f) { fprintf(stderr, "no fixture\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> wav(sz);
    if (fread(wav.data(), 1, sz, f) != (size_t) sz) return 2;
    fclose(f);
    size_t off = 12;
    std::vector<float> pcm;
    while (off + 8 <= (size_t) sz) {
        unsigned clen = wav[off+4] | (wav[off+5]<<8) | (wav[off+6]<<16) | ((unsigned)wav[off+7]<<24);
        if (off + 8 + clen > (size_t) sz) return 2;
        if (!memcmp(wav.data()+off, "data", 4)) {
            for (size_t i = 0; i + 1 < clen; i += 2) {
                short s = wav[off+8+i] | (wav[off+8+i+1]<<8);
                pcm.push_back(s / 32768.0f);
            }
            break;
        }
        off += 8 + clen + (clen & 1);
    }

    higgs::MelFeatures mel;
    if (!higgs::compute_log_mel(m.config, m.loader, pcm.data(), pcm.size(), mel, e)) { fprintf(stderr, "mel: %s\n", e.c_str()); return 2; }
    const int64_t mel_T = mel.n_frames;
    const int64_t T_enc = (mel_T + 1) / 2;
    printf("mel_T=%lld T_enc=%lld d_model=%u\n", (long long) mel_T, (long long) T_enc, ec.d_model);

    // ---- host front-end (engine CPU path) ----
    std::vector<float> c1 = host_conv1d(m.loader, mel.f32, ec.num_mel_bins, mel_T, "enc.conv1", 1, e);
    if (c1.empty()) { fprintf(stderr, "conv1: %s\n", e.c_str()); return 2; }
    // BF16 oracle boundary conv1 -> conv2 (mirrors the fixed engine path and
    // the graph path's gelu_erf_bf16 store).
    for (auto& v : c1) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
    std::vector<float> c2 = host_conv1d(m.loader, c1, ec.d_model, mel_T, "enc.conv2", 2, e);
    if (c2.empty()) { fprintf(stderr, "conv2: %s\n", e.c_str()); return 2; }
    std::vector<float> layers_in((size_t) ec.d_model * T_enc);
    for (int64_t t = 0; t < T_enc; ++t)
        for (int64_t d = 0; d < (int64_t) ec.d_model; ++d)
            layers_in[(size_t) t * ec.d_model + d] = c2[(size_t) d * T_enc + t];
    // BF16 oracle boundary conv2 -> layers (same).
    for (auto& v : layers_in) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));

    // ---- graph front-end (engine GPU path), run one-shot on THIS backend ----
    std::vector<float> g_layers_in, g_c1, g_c2;
    bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
        int64_t mne[2] = {mel_T, (int64_t) ec.num_mel_bins};
        ggml_tensor* mel_in = graph_input_tensor(ctx, GGML_TYPE_F32, 2, mne,
                                                 mel.f32.data(), mel.f32.size() * sizeof(float));
        // conv1 + bias + gelu (bf16 boundary), captured pre/post gelu
        ggml_tensor* w1 = f32(ctx, lib::weight(ctx, m.loader, "enc.conv1.weight"));
        ggml_tensor* conv1 = ggml_conv_1d(ctx, w1, f32(ctx, mel_in), 1, 1, 1);
        ggml_tensor* b1 = f32(ctx, lib::weight(ctx, m.loader, "enc.conv1.bias"));
        conv1 = ggml_add(ctx, f32(ctx, conv1), ggml_reshape_2d(ctx, b1, 1, ec.d_model));
        ggml_tensor* g1 = lib::gelu_erf_bf16(ctx, conv1);
        // conv2 + bias + gelu, transpose to d-contig f32
        ggml_tensor* w2 = f32(ctx, lib::weight(ctx, m.loader, "enc.conv2.weight"));
        ggml_tensor* conv2 = ggml_conv_1d(ctx, w2, f32(ctx, g1), 2, 1, 1);
        ggml_tensor* b2 = f32(ctx, lib::weight(ctx, m.loader, "enc.conv2.bias"));
        conv2 = ggml_add(ctx, f32(ctx, conv2), ggml_reshape_2d(ctx, b2, 1, ec.d_model));
        ggml_tensor* g2 = lib::gelu_erf_bf16(ctx, conv2);
        ggml_tensor* layers = f32(ctx, ggml_cont(ctx, ggml_permute(ctx, g2, 1, 0, 2, 3)));
        capture_graph_output(f32(ctx, g1), &g_c1);      // post-gelu conv1
        capture_graph_output(f32(ctx, g2), &g_c2);      // post-gelu conv2
        return layers;
    }, g_layers_in);
    if (!ok) { fprintf(stderr, "graph front-end failed\n"); return 2; }

    // host conv pre-gelu equivalents for comparison: recompute without gelu
    // (cheap enough once) by rerunning host_conv1d's math — reuse output
    // post-gelu vs graph post-gelu as the primary comparison instead.
    cmp("conv1_postgelu", c1, g_c1);
    cmp("conv2_postgelu", c2, g_c2);
    cmp("layers_in", layers_in, g_layers_in);
    return 0;
}
