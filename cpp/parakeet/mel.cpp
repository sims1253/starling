// mel.cpp — parakeet-tdt mel frontend (CPU FFT byte-exact + GPU DFT-matmul).
//
// Mirrors parakeet.cpp/src/mel.cpp (CPU reference) and mel_gpu.cpp (GPU path).
// The numerics are load-bearing for byte-exactness: every detail flagged in
// MEL_PIPELINE.md / mel.cpp is preserved — double-precision preemphasis, the
// window center-pad from win_length to n_fft, constant/zeros center-pad, float
// windowed frames, double-internal radix-2 FFT, double mel accumulation,
// log_guard = 2**-24, per-feature CMVN with ddof=1 + eps=1e-5, padding frames
// zeroed. Output is feat-major [n_mels, T] float32.

#include "mel.hpp"
#include "fft.hpp"

#include "runtime/graph.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace starling::ggml::parakeet {

// NeMo normalize_batch epsilon (features.py: CONSTANT = 1e-5).
static constexpr float kNormEps = 1e-5f;

// DEBUG helper: dump frame diagnostics to stderr when STARLING_MEL_DEBUG is set.
static bool mel_debug_enabled() {
    static const bool on = (std::getenv("STARLING_MEL_DEBUG") != nullptr);
    return on;
}

// --------------------------------------------------------------------------- //
// MelConstants: read the filterbank + window from the GGUF.
// --------------------------------------------------------------------------- //
void MelConstants::read_from(const ModelLoader& ml, const Config& cfg) {
    n_mels       = cfg.n_mels;
    n_fft        = cfg.n_fft;
    n_bins       = n_fft / 2 + 1;
    win_length   = cfg.win_length;
    hop_length   = cfg.hop_length;
    preemph      = cfg.preemph;
    mag_power    = cfg.mag_power;
    log_zero_guard = cfg.log_zero_guard;
    normalize    = cfg.normalize;

    // Window: stored as [win_length] f32; center-pad to [n_fft] at offset
    // (n_fft - win_length)/2 (replicates torch.stft(win_length=W, n_fft=N)
    // which zero-pads the window symmetrically).
    window.assign(n_fft, 0.0f);
    {
        ggml_tensor* wt = ml.tensor("preprocessor.featurizer.window");
        const float* wsrc = (const float*)wt->data;
        size_t off = (n_fft - win_length) / 2;
        for (uint32_t i = 0; i < win_length; ++i) window[off + i] = wsrc[i];
    }
    // Filterbank: stored ggml ne [n_bins, n_mels, 1] (numpy (1, n_mels, n_bins)
    // row-major) -> fb[m*n_bins + b]. Copy verbatim into [n_mels * n_bins].
    {
        ggml_tensor* fbt = ml.tensor("preprocessor.featurizer.fb");
        const float* fsrc = (const float*)fbt->data;
        // ggml ne[0]=n_bins, ne[1]=n_mels -> contiguous rows of n_bins, n_mels
        // rows. That's exactly fb[m*n_bins + b].
        size_t total = (size_t)n_bins * n_mels;
        filterbank.assign(fsrc, fsrc + total);
    }
}

// --------------------------------------------------------------------------- //
// MelFrontend (CPU FFT, byte-exact reference).
// --------------------------------------------------------------------------- //
void MelFrontend::compute(const float* pcm, size_t S, std::vector<float>& feats,
                          int& out_T) const {
    const uint32_t n_fft = c_.n_fft, hop = c_.hop_length, n_mels = c_.n_mels;
    const uint32_t n_bins = c_.n_bins;
    const float preemph = c_.preemph;

    // 1. Preemphasis in double; first sample unchanged.
    std::vector<double> pre(S);
    if (S > 0) {
        pre[0] = (double)pcm[0];
        for (size_t t = 1; t < S; ++t)
            pre[t] = (double)pcm[t] - (double)preemph * (double)pcm[t - 1];
    }
    // seq_len = floor(S / hop): frames whose stats count for CMVN.
    size_t seq_len = (S > 0) ? (S / hop) : 0;

    // 2. Center zero-pad: pad = n_fft/2 each side, pad_mode="constant" (zeros).
    size_t pad = n_fft / 2;
    size_t padded_len = S + n_fft;
    std::vector<double> padded(padded_len, 0.0);
    for (size_t t = 0; t < S; ++t) padded[pad + t] = pre[t];
    // T = 1 + (padded_len - n_fft) / hop.
    size_t T = 1 + (padded_len - n_fft) / hop;
    out_T = (int)T;

    feats.assign((size_t)n_mels * T, 0.0f);

    // ---- DEBUG: one-time dump of config + window/fb anchors + preemph[0:3]. ----
    if (mel_debug_enabled()) {
        std::fprintf(stderr, "\n[MEL_DEBUG] config: n_fft=%u hop=%u n_mels=%u n_bins=%u "
            "win_length=%u preemph=%.8g mag_power=%.8g log_guard=%.8g S=%zu seq_len=%zu "
            "pad=%zu padded_len=%zu T=%zu\n",
            n_fft, hop, n_mels, n_bins, c_.win_length, preemph, c_.mag_power,
            c_.log_zero_guard, S, seq_len, pad, padded_len, T);
        std::fprintf(stderr, "[MEL_DEBUG] window[0:3]=%.8g %.8g %.8g  win[54:58]=", c_.window[0], c_.window[1], c_.window[2]);
        for (int i = 54; i < 58; ++i) std::fprintf(stderr, " %.8g", c_.window[i]);
        std::fprintf(stderr, "\n[MEL_DEBUG] fb[0:3]=%.8g %.8g %.8g  fb_row0[128:131]=", c_.filterbank[0], c_.filterbank[1], c_.filterbank[2]);
        for (int b = 128; b < 131; ++b) std::fprintf(stderr, " %.8g", c_.filterbank[b]);
        std::fprintf(stderr, "\n[MEL_DEBUG] pcm[0:3]=%.8g %.8g %.8g  pre[0:3]=", pcm[0], pcm[1], pcm[2]);
        for (int i = 0; i < 3; ++i) std::fprintf(stderr, " %.8g", pre[i]);
        std::fprintf(stderr, "\n");
    }


    // 3. Per frame: window -> FFT -> power -> filterbank -> log.
    std::vector<std::complex<double>> buf(n_fft);
    std::vector<double> power(n_bins);
    for (size_t t = 0; t < T; ++t) {
        size_t start = t * hop;
        // Window (mixed precision: padded double * window float -> cast float).
        for (uint32_t i = 0; i < n_fft; ++i)
            buf[i] = std::complex<double>((double)((float)(padded[start + i] * (double)c_.window[i])), 0.0);
        // rFFT (double-internal). Round re/im to float to mirror parakeet.cpp's
        // MelKernel::frame_logmel exactly (its rfft returns float re/im, then power
        // is computed from the float values cast to double). Computing power from
        // the full-double FFT output diverges by ~1e-2 from the golden here.
        rfft(buf, n_fft);
        // Power per bin: mag = sqrt(re^2+im^2); power = mag^mag_power.
        // For mag_power==2, power = re^2+im^2 directly (skip sqrt). Match the
        // general formula anyway (sqrt then pow) for fidelity to the reference.
        for (uint32_t b = 0; b < n_bins; ++b) {
            float ref = (float)buf[b].real();
            float imf = (float)buf[b].imag();
            double re = (double)ref, im = (double)imf;
            double mag = std::sqrt(re * re + im * im);
            power[b] = std::pow(mag, (double)c_.mag_power);
        }
        // Mel projection + log (accumulate in double; log_guard = 2**-24).
        for (uint32_t m = 0; m < n_mels; ++m) {
            const float* fbrow = &c_.filterbank[(size_t)m * n_bins];
            double acc = 0.0;
            for (uint32_t b = 0; b < n_bins; ++b) acc += (double)fbrow[b] * power[b];
            feats[(size_t)m * T + t] = (float)std::log(acc + (double)c_.log_zero_guard);
        }

        // ---- DEBUG: dump frame 0 and frame 1 diagnostics to stderr. ----
        if (mel_debug_enabled() && (t == 0 || t == 1)) {
            std::fprintf(stderr, "\n[MEL_DEBUG] frame %zu\n", t);
            // (a) windowed-frame samples [0:5]
            std::fprintf(stderr, "  win_samples[0:5]:");
            for (int i = 0; i < 5; ++i)
                std::fprintf(stderr, " %.8g",
                    (double)((float)(padded[start + i] * (double)c_.window[i])));
            std::fprintf(stderr, "\n");
            // (b) FFT power spectrum bins [0:5] + max-power bin
            double pmax = -1.0; uint32_t pmax_b = 0;
            for (uint32_t b = 0; b < n_bins; ++b)
                if (power[b] > pmax) { pmax = power[b]; pmax_b = b; }
            std::fprintf(stderr, "  power[0:5]:");
            for (int b = 0; b < 5; ++b) std::fprintf(stderr, " %.8g", power[b]);
            std::fprintf(stderr, "\n  power_max: bin=%u val=%.8g\n", pmax_b, pmax);
            // (c) pre-CMVN mel bins [0:5]
            std::fprintf(stderr, "  premel[0:5]:");
            for (int m = 0; m < 5; ++m)
                std::fprintf(stderr, " %.8g", feats[(size_t)m * T + t]);
            std::fprintf(stderr, "\n");
        }
    }


    // ---- DEBUG: optional full pre-CMVN mel dump to a file (feat-major). ----
    if (mel_debug_enabled()) {
        if (const char* p = std::getenv("STARLING_MEL_DUMP_CPU")) {
            FILE* f = std::fopen(p, "wb");
            if (f) {
                uint32_t nm = n_mels; uint32_t nt = (uint32_t)T;
                std::fwrite(&nm, sizeof(uint32_t), 1, f);
                std::fwrite(&nt, sizeof(uint32_t), 1, f);
                std::fwrite(feats.data(), sizeof(float), (size_t)n_mels * T, f);
                std::fclose(f);
            }
        }
    }

    // 4. Per-feature CMVN (normalize == "per_feature"): mean/var over the first
    //    `valid = min(seq_len, T)` frames, ddof=1, eps=1e-5; zero padding frames.
    if (c_.normalize == "per_feature") {
        size_t valid = std::min(seq_len, T);
        if (valid >= 1) {
            double ddof = (valid >= 2) ? (double)(valid - 1) : 1.0;  // ddof=1, guard div-by-0
            for (uint32_t m = 0; m < n_mels; ++m) {
                float* row = &feats[(size_t)m * T];
                double mean = 0.0;
                for (size_t t = 0; t < valid; ++t) mean += row[t];
                mean /= (double)valid;
                double var = 0.0;
                for (size_t t = 0; t < valid; ++t) {
                    double d = row[t] - mean;
                    var += d * d;
                }
                var /= ddof;
                double sd = std::sqrt(var) + (double)kNormEps;
                for (size_t t = 0; t < T; ++t) {
                    if (t < valid) row[t] = (float)(((double)row[t] - mean) / sd);
                    else          row[t] = 0.0f;   // zero padding frames
                }
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// GpuMel (GPU DFT-as-matmul, cached per-T in a ReplayGraph).
// --------------------------------------------------------------------------- //
GpuMel::GpuMel(Backend& backend, const MelConstants& c)
    : backend_(backend), c_(c) {
    build_basis();
}

GpuMel::~GpuMel() = default;

void GpuMel::build_basis() {
    // DFT basis matching the CPU rfft sign convention:
    //   re[b] = sum_n x[n] * cos(2*pi*b*n/N)
    //   im[b] = -sum_n x[n] * sin(2*pi*b*n/N)
    // Stored [n_bins * n_fft], row-major (b outer, n inner), f32.
    const uint32_t N = c_.n_fft, B = c_.n_bins;
    dft_cos_.assign((size_t)B * N, 0.0f);
    dft_sin_.assign((size_t)B * N, 0.0f);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t n = 0; n < N; ++n) {
            double ang = 2.0 * M_PI * (double)b * (double)n / (double)N;
            dft_cos_[(size_t)b * N + n] = (float)std::cos(ang);
            dft_sin_[(size_t)b * N + n] = (float)(-std::sin(ang));
        }
    }
}

void GpuMel::build_or_reuse_replay(int T) {
    if (replay_ && replay_->T == T) return;
    replay_.reset();
    auto pending = std::make_unique<MelReplay>();
    auto* replay = pending.get();
    replay->T = T;
    replay->pool.alloc_f32(1);  // touch the pool
    // Host backing for the windowed frames: [n_fft * T], built per compute().
    replay->xw_host = replay->pool.alloc_f32((size_t)c_.n_fft * T);
    replay->xw_nbytes = (size_t)c_.n_fft * T * sizeof(float);

    const int64_t N = c_.n_fft, B = c_.n_bins, M = c_.n_mels, Tt = T;
    // Build the ggml graph once: xw[N,T] -> re=B@xw, im=B@xw -> power -> mel -> log.
    replay->rg = std::make_unique<ReplayGraph>(backend_,
        [this, replay, N, B, M, Tt](ggml_context* ctx) -> ggml_tensor* {
            // Inputs (registered in order; set_input feeds them per call).
            int64_t ne_xw[2]  = {N, Tt};
            ggml_tensor* xw = graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne_xw,
                replay->xw_host, replay->xw_nbytes);
            // Constant DFT bases + filterbank + log-guard scalar (GraphInputPool).
            int64_t ne_basis[2] = {N, B};
            float* cosb_h = replay->pool.alloc_f32((size_t)N * B);
            std::memcpy(cosb_h, dft_cos_.data(), (size_t)N * B * sizeof(float));
            ggml_tensor* cosb = graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne_basis,
                cosb_h, (size_t)N * B * sizeof(float));
            float* sinb_h = replay->pool.alloc_f32((size_t)N * B);
            std::memcpy(sinb_h, dft_sin_.data(), (size_t)N * B * sizeof(float));
            ggml_tensor* sinb = graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne_basis,
                sinb_h, (size_t)N * B * sizeof(float));
            // re = cosb . xw  -> [B, T]; im = sinb . xw -> [B, T]
            ggml_tensor* re = ggml_mul_mat(ctx, cosb, xw);
            ggml_tensor* im = ggml_mul_mat(ctx, sinb, xw);
            // power = re^2 + im^2 (mag_power==2 fast path; asserted by caller).
            ggml_tensor* power = ggml_add(ctx,
                ggml_mul(ctx, re, re), ggml_mul(ctx, im, im));
            // Filterbank -> feat-major mel. c_.filterbank is mel-major
            // (filterbank[m*B + b] = fb[m][b]); power is ne=[B, T] (bin-major).
            // ggml_mul_mat(a, b): A has k cols/ne0 + n rows/ne1, B has k cols/ne0 +
            // m rows/ne1, result is n cols/ne0 + m rows/ne1 with
            //   result[m, n] = sum_k B[m, k] * A[n, k].
            // We want mel[m, t] = sum_b fb[m, b] * power[t, b], laid out
            // FEAT-MAJOR (feats[m*T + t], i.e. result ne0=T, ne1=M). That means
            // n=T (a=power, ne1=T) and m=M (b=fb, ne1=M), so the arguments are
            // mul_mat(power, fb) — power as A, fb as B. (mul_mat(fb, power) would
            // yield ne0=M, ne1=T = frame-major, which is the bug this fixes: the
            // downstream CMVN / output assume feats[m*T + t] feat-major.)
            int64_t ne_fb[2] = {B, M};
            float* fb_h = replay->pool.alloc_f32((size_t)B * M);
            std::memcpy(fb_h, c_.filterbank.data(), (size_t)B * M * sizeof(float));
            ggml_tensor* fb = graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne_fb,
                fb_h, (size_t)B * M * sizeof(float));
            ggml_tensor* mel = ggml_mul_mat(ctx, power, fb);
            // log(mel + log_guard).
            float* g_h = replay->pool.alloc_f32(1); g_h[0] = c_.log_zero_guard;
            int64_t ne_g[1] = {1};
            ggml_tensor* g = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne_g,
                g_h, sizeof(float));
            ggml_tensor* lm = ggml_log(ctx, ggml_add(ctx, mel, g));
            return lm;  // [n_mels, T] feat-major
        });
    replay_ = std::move(pending);
}

void GpuMel::compute(const float* pcm, size_t S, std::vector<float>& feats,
                     int& out_T) {
    const uint32_t n_fft = c_.n_fft, hop = c_.hop_length, n_mels = c_.n_mels;
    const float preemph = c_.preemph;
    // Host preemph (double) + framing + windowing — bit-identical to MelFrontend.
    std::vector<double> pre(S);
    if (S > 0) {
        pre[0] = (double)pcm[0];
        for (size_t t = 1; t < S; ++t)
            pre[t] = (double)pcm[t] - (double)preemph * (double)pcm[t - 1];
    }
    size_t pad = n_fft / 2;
    size_t padded_len = S + n_fft;
    std::vector<double> padded(padded_len, 0.0);
    for (size_t t = 0; t < S; ++t) padded[pad + t] = pre[t];
    size_t T = 1 + (padded_len - n_fft) / hop;
    out_T = (int)T;
    size_t seq_len = (S > 0) ? (S / hop) : 0;

    build_or_reuse_replay((int)T);

    // Fill the windowed-frame host buffer: xw[t*N + i] = (float)(padded[t*hop+i] * window[i]).
    for (size_t t = 0; t < T; ++t) {
        size_t start = t * hop;
        for (uint32_t i = 0; i < n_fft; ++i)
            replay_->xw_host[t * n_fft + i] =
                (float)(padded[start + i] * (double)c_.window[i]);
    }
    // Re-upload all inputs (the persistent gallocr/ggml-cuda replay path doesn't
    // guarantee non-mel input device tensors keep contents across replays).
    for (size_t i = 0; i < replay_->rg->n_inputs(); ++i)
        replay_->rg->set_input(i, replay_->rg->input_host(i), replay_->rg->input_nbytes(i));

    // Run the graph -> [n_mels, T] feat-major.
    std::vector<float> graph_out;
    if (!replay_->rg->compute(graph_out) || graph_out.size() != (size_t)n_mels * T) {
        feats.clear(); out_T = 0; return;
    }
    feats.swap(graph_out);  // feat-major [n_mels, T]

    // ---- DEBUG: dump GPU-path windowed samples + pre-CMVN mel for frame 0/1. ----
    if (mel_debug_enabled()) {
        for (int tt = 0; tt <= 1 && tt < (int)T; ++tt) {
            const float* xw = &replay_->xw_host[(size_t)tt * n_fft];
            std::fprintf(stderr, "\n[MEL_DEBUG_GPU] frame %d\n", tt);
            std::fprintf(stderr, "  win_samples[0:5]:");
            for (int i = 0; i < 5; ++i) std::fprintf(stderr, " %.8g", xw[i]);
            std::fprintf(stderr, "\n");
            std::fprintf(stderr, "  premel[0:5]:");
            for (int m = 0; m < 5; ++m) std::fprintf(stderr, " %.8g", feats[(size_t)m * T + tt]);
            std::fprintf(stderr, "\n");
            // also DFT basis check: dump cos/sin for bin 0,1 n=0
            std::fprintf(stderr, "  dft_cos[bin0 n0..2]: %.8g %.8g %.8g  dft_sin[bin1 n0..2]: %.8g %.8g %.8g\n",
                dft_cos_[0], dft_cos_[1], dft_cos_[2],
                dft_sin_[(size_t)1 * n_fft + 0], dft_sin_[(size_t)1 * n_fft + 1], dft_sin_[(size_t)1 * n_fft + 2]);
        }
        std::fprintf(stderr, "[MEL_DEBUG_GPU] fb[0:3]=%.8g %.8g %.8g fb_row0[128:131]=%.8g %.8g %.8g\n",
            c_.filterbank[0], c_.filterbank[1], c_.filterbank[2],
            c_.filterbank[128], c_.filterbank[129], c_.filterbank[130]);
        // Optional full pre-CMVN mel dump to a file for offline numpy diff.
        if (const char* p = std::getenv("STARLING_MEL_DUMP")) {
            FILE* f = std::fopen(p, "wb");
            if (f) {
                uint32_t nm = n_mels; uint32_t nt = (uint32_t)T;
                std::fwrite(&nm, sizeof(uint32_t), 1, f);
                std::fwrite(&nt, sizeof(uint32_t), 1, f);
                std::fwrite(feats.data(), sizeof(float), (size_t)n_mels * T, f);
                std::fclose(f);
            }
        }
        // Optional dump of the windowed-frame host buffer xw[N,T] and DFT bases.
        if (const char* p = std::getenv("STARLING_MEL_DUMP_XW")) {
            FILE* f = std::fopen(p, "wb");
            if (f) {
                uint32_t nn = n_fft; uint32_t nt = (uint32_t)T; uint32_t nb = c_.n_bins;
                std::fwrite(&nn, sizeof(uint32_t), 1, f);
                std::fwrite(&nt, sizeof(uint32_t), 1, f);
                std::fwrite(&nb, sizeof(uint32_t), 1, f);
                std::fwrite(replay_->xw_host, sizeof(float), (size_t)n_fft * T, f);
                std::fwrite(dft_cos_.data(), sizeof(float), (size_t)c_.n_bins * n_fft, f);
                std::fwrite(dft_sin_.data(), sizeof(float), (size_t)c_.n_bins * n_fft, f);
                std::fclose(f);
            }
        }
    }



    // Per-feature CMVN (identical to MelFrontend; runs on host).
    if (c_.normalize == "per_feature") {
        size_t valid = std::min(seq_len, T);
        if (valid >= 1) {
            double ddof = (valid >= 2) ? (double)(valid - 1) : 1.0;
            for (uint32_t m = 0; m < n_mels; ++m) {
                float* row = &feats[(size_t)m * T];
                double mean = 0.0;
                for (size_t t = 0; t < valid; ++t) mean += row[t];
                mean /= (double)valid;
                double var = 0.0;
                for (size_t t = 0; t < valid; ++t) {
                    double d = row[t] - mean;
                    var += d * d;
                }
                var /= ddof;
                double sd = std::sqrt(var) + (double)kNormEps;
                for (size_t t = 0; t < T; ++t) {
                    if (t < valid) row[t] = (float)(((double)row[t] - mean) / sd);
                    else          row[t] = 0.0f;
                }
            }
        }
    }
}

} // namespace starling::ggml::parakeet
