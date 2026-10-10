// Exercise real prediction/joint graphs with hand-encoded quantized weights.
// The oracle uses scalar F32 LSTM/joint math, not ggml dequantization or graphs.
#include "parakeet/prediction.hpp"
#include "parakeet/joint.hpp"
#include "parakeet/tdt.hpp"
#include "runtime/backend.hpp"
#include "runtime/cpu_repack.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace starling::ggml;
using namespace starling::ggml::parakeet;
namespace {
constexpr int H = 640, L = 2, VOCAB = 3, DURATIONS = 3;
constexpr int IQ4_VALUES[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
void check(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}
uint32_t sample(uint32_t row, uint32_t col, uint32_t salt) {
    uint32_t x = (row + 1) * 0x9e3779b9u ^ (col + salt * 97) * 0x85ebca6bu;
    x ^= x >> 16;
    x *= 0x7feb352du;
    return x ^ (x >> 15);
}
struct TempFile {
    std::filesystem::path path;
    ~TempFile() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};
struct Fixture {
    int embed_rows;
    TempFile file;
    std::unordered_map<std::string, std::vector<float>> weights;
    explicit Fixture(int rows) : embed_rows(rows) {
        file.path = std::filesystem::temp_directory_path() /
            ("starling-compact-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
             + "-" + std::to_string(rows) + ".gguf");
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({8 << 20, nullptr, false}), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> out(gguf_init_empty(), gguf_free);
        check(ctx && out, "fixture allocation");
        gguf_set_val_str(out.get(), "general.architecture", "parakeet_tdt");
        auto packed = [&](const std::string& name, int nr, ggml_type type, int salt) {
            auto* tensor = ggml_new_tensor_2d(ctx.get(), type, H, nr);
            ggml_set_name(tensor, name.c_str());
            auto& values = weights[name];
            values.resize((size_t)nr * H);
            auto* bytes = static_cast<uint8_t*>(tensor->data);
            const int stride = type == GGML_TYPE_Q8_0 ? 34 : 18;
            check(ggml_nbytes(tensor) == (size_t)nr * H / 32 * stride, "packed layout size");
            for (int r = 0; r < nr; ++r) {
                for (int b = 0; b < H / 32; ++b) {
                    // Exact powers of two: encode the IEEE half exponent directly.
                    const int exponent = (type == GGML_TYPE_Q8_0 ? -8 : -13) + (r + b) % 2;
                    const uint16_t half = static_cast<uint16_t>((15 + exponent) << 10);
                    const float scale = std::ldexp(1.0f, exponent);
                    auto* block = bytes + ((size_t)r * H / 32 + b) * stride;
                    std::memcpy(block, &half, sizeof(half));
                    std::memset(block + 2, 0, stride - 2);
                    for (int j = 0; j < 32; ++j) {
                        const auto code = sample(r, b * 32 + j, salt);
                        if (type == GGML_TYPE_Q8_0) {
                            const int q = static_cast<int>(code % 201) - 100;
                            block[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(q));
                            values[(size_t)r * H + b * 32 + j] = scale * q;
                        } else {
                            const int q = code % 16;
                            // IQ4_NL low nibbles are columns 0..15, high are 16..31.
                            block[2 + j % 16] |= static_cast<uint8_t>(q << (j < 16 ? 0 : 4));
                            values[(size_t)r * H + b * 32 + j] = scale * IQ4_VALUES[q];
                        }
                    }
                }
            }
            gguf_add_tensor(out.get(), tensor);
        };
        auto floats = [&](const std::string& name, int n) {
            auto* tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
            ggml_set_name(tensor, name.c_str());
            auto& values = weights[name];
            values.resize(n);
            for (int i = 0; i < n; ++i) values[i] = (i % 13 - 6) / 128.0f;
            // Separate token/duration winners from activation-rounding ties.
            if (name == "joint.joint_net.2.bias")
                values = {0.f, 0.4f, 0.1f, -0.2f, 0.3f, -0.1f, 0.7f};
            std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
            gguf_add_tensor(out.get(), tensor);
        };
        packed("decoder.prediction.embed.weight", rows, GGML_TYPE_Q8_0, 1);
        for (int l = 0; l < L; ++l) {
            for (const char* kind : {"ih", "hh"}) {
                const std::string suffix = std::string(kind) + "_l" + std::to_string(l);
                packed("decoder.prediction.dec_rnn.lstm.weight_" + suffix, 4 * H,
                       GGML_TYPE_IQ4_NL, 2 + l * 2 + (kind[0] == 'h'));
                floats("decoder.prediction.dec_rnn.lstm.bias_" + suffix, 4 * H);
            }
        }
        // Joint reads this shape; the encoder projection itself is supplied by the test.
        auto* enc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, H);
        ggml_set_name(enc, "joint.enc.weight");
        std::memset(enc->data, 0, ggml_nbytes(enc));
        gguf_add_tensor(out.get(), enc);
        packed("joint.pred.weight", H, GGML_TYPE_IQ4_NL, 9);
        floats("joint.pred.bias", H);
        packed("joint.joint_net.2.weight", VOCAB + 1 + DURATIONS, GGML_TYPE_IQ4_NL, 10);
        floats("joint.joint_net.2.bias", VOCAB + 1 + DURATIONS);
        check(gguf_write_to_file(out.get(), file.path.string().c_str(), false), "write fixture");
    }
};
std::vector<float> affine(const Fixture& f, const std::string& weight,
                          const std::string& bias, const std::vector<float>& x) {
    const auto& w = f.weights.at(weight);
    std::vector<float> y = f.weights.at(bias);
    for (size_t r = 0; r < y.size(); ++r) {
        float sum = 0;
        for (int c = 0; c < H; ++c) sum += w[r * H + c] * x[c];
        y[r] += sum;
    }
    return y;
}
PredState reference_step(const Fixture& f, int token, bool sos, const PredState& in) {
    std::vector<float> x(H, 0.f);
    if (!sos && token < f.embed_rows) {
        const auto& table = f.weights.at("decoder.prediction.embed.weight");
        std::copy_n(table.begin() + token * H, H, x.begin());
    }
    PredState out = in;
    for (int l = 0; l < L; ++l) {
        const std::string stem = "decoder.prediction.dec_rnn.lstm.";
        const std::string suffix = "_l" + std::to_string(l);
        auto z = affine(f, stem + "weight_ih" + suffix, stem + "bias_ih" + suffix, x);
        auto recurrent = affine(f, stem + "weight_hh" + suffix, stem + "bias_hh" + suffix, in.h[l]);
        for (int i = 0; i < 4 * H; ++i) z[i] += recurrent[i];
        const auto sigmoid = [](float v) { return 1.f / (1.f + std::exp(-v)); };
        for (int i = 0; i < H; ++i) {
            out.c[l][i] = sigmoid(z[H + i]) * in.c[l][i] + sigmoid(z[i]) * std::tanh(z[2 * H + i]);
            out.h[l][i] = sigmoid(z[3 * H + i]) * std::tanh(out.c[l][i]);
        }
        x = out.h[l];
    }
    return out;
}
float near(const std::vector<float>& actual, const std::vector<float>& expected,
           float tolerance, const std::string& name) {
    check(actual.size() == expected.size(), name + " size");
    float largest = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        check(std::isfinite(actual[i]) && error <= tolerance,
              name + " at " + std::to_string(i) + ": error " + std::to_string(error));
        largest = std::max(largest, error);
    }
    return largest;
}
void exercise(int rows) {
    Fixture fixture(rows);
    ModelLoader loader;
    check(loader.load(fixture.file.path.string().c_str()), "load fixture");
    Config cfg;
    cfg.pred_hidden = H;
    cfg.pred_rnn_layers = L;
    cfg.vocab_size = VOCAB;
    cfg.tdt_durations = {0, 1, 2};
    PredictionNet pred(loader, cfg);
    Joint joint(loader, cfg);
    auto state = pred.zero_state(), reference = state;
    float state_error = 0, logit_error = 0;
    const int tokens[] = {-1, 0, 2, 1, 3, 2, -1, 0};
    for (int step = 0; step < 8; ++step) {
        const int token = tokens[step];
        PredState next;
        std::vector<float> g;
        pred.step(token, token < 0, state, g, next);
        if (token == 0) {
            const auto wrong_row = reference_step(fixture, 1, false, reference);
            float separation = 0;
            for (int i = 0; i < H; ++i)
                separation = std::max(separation, std::abs(next.h[0][i] - wrong_row.h[0][i]));
            check(separation > 0.02f, "fixture must distinguish a wrong embedding row above tolerance");
        }
        reference = reference_step(fixture, token, token < 0, reference);
        // IQ4_NL CPU matmul also quantizes its activation operand to Q8_0.
        // Allow that rounding while comparing with an unquantized scalar oracle.
        for (int l = 0; l < L; ++l) {
            const auto where = " rows=" + std::to_string(rows) + " step=" + std::to_string(step)
                + " layer=" + std::to_string(l);
            state_error = std::max(state_error, near(next.h[l], reference.h[l], 0.002f, "hidden" + where));
            state_error = std::max(state_error, near(next.c[l], reference.c[l], 0.002f, "cell" + where));
        }
        near(g, next.h.back(), 0.f, "prediction output capture");
        state = std::move(next);
        std::vector<float> expected_embed((VOCAB + 1) * H, 0.f);
        const auto& stored_embed = fixture.weights.at("decoder.prediction.embed.weight");
        std::copy_n(stored_embed.begin(), std::min(expected_embed.size(), stored_embed.size()), expected_embed.begin());
        near(pred.embed_host(), expected_embed, 0.f, "embedding dequantization/padding");
        std::vector<float> enc(H);
        for (int i = 0; i < H; ++i) enc[i] = ((i * 7 + step * 13) % 41 - 20) / 64.f;
        auto f = affine(fixture, "joint.pred.weight", "joint.pred.bias", reference.h.back());
        for (int i = 0; i < H; ++i) f[i] = std::max(0.f, f[i] + enc[i]);
        auto expected = affine(fixture, "joint.joint_net.2.weight", "joint.joint_net.2.bias", f);
        std::vector<float> logits;
        joint.step_logits(enc.data(), g.data(), H, logits);
        logit_error = std::max(logit_error, near(logits, expected, 0.004f, "joint logits"));
        int token_argmax = -1, duration_argmax = -1;
        joint.step_argmax(enc.data(), VOCAB + 1, g.data(), H, token_argmax, duration_argmax);
        auto winner = [&](int begin, int end) {
            const int best = int(std::max_element(expected.begin() + begin, expected.begin() + end) - expected.begin());
            for (int i = begin; i < end; ++i)
                if (i != best) check(expected[best] - expected[i] > 0.008f, "argmax fixture margin");
            return best - begin;
        };
        check(token_argmax == winner(0, VOCAB + 1), "token argmax");
        check(duration_argmax == winner(VOCAB + 1, VOCAB + 1 + DURATIONS), "duration argmax");
    }
    // Decode timing (issue #357): the same id stream with and without it,
    // one frame and duration per step, each step at the frame its
    // predecessor advanced to (plus one after max_symbols steps on a frame),
    // inside the encoder output.
    {
        constexpr int T = 24, max_symbols = 3;
        std::vector<float> enc_proj((size_t)T * H);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < H; ++i)
                enc_proj[(size_t)t * H + i] = ((i * 5 + t * 11) % 37 - 18) / 48.f;
        TdtTiming timing;
        const auto ids = tdt_greedy(pred, joint, enc_proj, T, H, cfg.tdt_durations,
                                    VOCAB, max_symbols, &timing);
        check(ids == tdt_greedy(pred, joint, enc_proj, T, H, cfg.tdt_durations, VOCAB,
                                max_symbols),
              "timing leaves the decode unchanged");
        check(timing.frame.size() == ids.size() && timing.duration.size() == ids.size(),
              "one frame and duration per step");
        check(!ids.empty() && timing.frame[0] == 0, "the decode starts at frame 0");
        // Steps on one frame run until a positive duration or max_symbols
        // steps, which also moves one frame on.
        int on_frame = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            check(timing.frame[i] >= 0 && timing.frame[i] < T, "step frame inside the audio");
            check(std::find(cfg.tdt_durations.begin(), cfg.tdt_durations.end(),
                            timing.duration[i]) != cfg.tdt_durations.end(),
                  "step duration from the duration table");
            ++on_frame;
            const int next = timing.frame[i] + timing.duration[i] + (on_frame == max_symbols);
            if (i + 1 < ids.size())
                check(timing.frame[i + 1] == next, "each step reads the frame its predecessor advanced to");
            else
                check(next >= T, "the last step reaches the end");
            if (timing.duration[i] > 0 || on_frame == max_symbols) on_frame = 0;
        }
        std::printf("rows=%d: %zu decode steps over %d frames\n", rows, ids.size(), T);
    }
    // CI sets this only on runners whose IQ4_NL repack kernel is pinned by
    // cpu_repack_test; a developer CPU without that kernel still runs the
    // numerical regression with the gate enabled.
    const char* expect_repack = std::getenv("STARLING_EXPECT_ENGINE_REPACK");
    if (expect_repack && std::strcmp(expect_repack, "1") == 0) {
        check(cpu_repack::enabled(), "expected Parakeet repacking requires the gate");
        check(cpu_repack::stats().tensors > 0, "forced-on Parakeet run must repack a weight");
    }
    std::printf("rows=%d: max state error %.7f, max logit error %.7f\n", rows, state_error, logit_error);
}
} // namespace
int main() {
    try {
#ifdef _WIN32
        _putenv_s("STARLING_GGML_DEVICE", "cpu");
#else
        setenv("STARLING_GGML_DEVICE", "cpu", 1);
#endif
        check(!global_backend().is_gpu(), "test requires CPU backend");
        global_backend().set_n_threads(2);
        for (int rows : {VOCAB, VOCAB + 1, VOCAB + 2}) exercise(rows);
        std::puts("PARAKEET COMPACT ENGINE OK: Q8 rows, padding, recurrent IQ4_NL LSTM/joint");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PARAKEET COMPACT ENGINE FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
