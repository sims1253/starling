// mel_dump.cpp — debug aid for the mel frontend contract: run the engine's
// MelFrontend (the byte-exact CPU reference) on a WAV and write the feat-major
// [n_mels, T] float32 features to a raw file for comparison against the
// transformers feature extractor.
//
//   mel_dump <model.gguf> <audio.wav> <out.f32>
//
// Prints T and a few summary statistics; the raw file is `T * n_mels` floats.

#include "loader.hpp"
#include "mel.hpp"
#include "encoder.hpp"

#include "runtime/audio_io.hpp"
#include "runtime/graph.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <audio.wav> <out_mel.f32> [out_enc.f32]\n",
                     argv[0]);
        return 1;
    }
    starling::ggml::parakeet::ParakeetModel model;
    std::string err;
    if (!model.load(argv[1], err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    starling::ggml::parakeet::MelConstants mc;
    mc.read_from(model.loader, model.config);

    std::vector<float> pcm;
    int sr = 0;
    std::string wav_err;
    if (!starling::ggml::read_wav(argv[2], pcm, sr, wav_err)) {
        std::fprintf(stderr, "wav read failed: %s: %s\n", argv[2], wav_err.c_str());
        return 1;
    }
    std::fprintf(stderr, "wav: %zu samples @ %d Hz | fb %ux%u hop %u win %u "
                         "preemph %f mag %f norm %s guard %g\n",
                 pcm.size(), sr, mc.n_mels, mc.n_bins, mc.hop_length,
                 mc.win_length, mc.preemph, mc.mag_power, mc.normalize.c_str(),
                 mc.log_zero_guard);

    std::vector<float> feats;
    int T = 0;
    starling::ggml::parakeet::MelFrontend(mc).compute(pcm.data(), pcm.size(),
                                                      feats, T);

    // Optional encoder pass: dump the row-major [T', H] joint-projected
    // encoder output when a 4th path argument is given.
    if (argc >= 5) {
        starling::ggml::parakeet::Encoder enc(model);
        std::vector<float> enc_out;
        int Tp = 0;
        if (!enc.encode(feats, (int)mc.n_mels, T, enc_out, Tp)) {
            std::fprintf(stderr, "encoder failed\n");
            return 1;
        }
        FILE* ef = std::fopen(argv[4], "wb");
        if (!ef) { std::fprintf(stderr, "open failed: %s\n", argv[4]); return 1; }
        std::fwrite(enc_out.data(), sizeof(float), enc_out.size(), ef);
        std::fclose(ef);
        double es = 0, emn = 1e30, emx = -1e30;
        for (float v : enc_out) { es += v; if (v < emn) emn = v; if (v > emx) emx = v; }
        std::fprintf(stderr, "enc: T'=%d H=%zu mean=%.4f min=%.4f max=%.4f -> %s\n",
                     Tp, enc_out.size() / (Tp > 0 ? Tp : 1), es / enc_out.size(),
                     emn, emx, argv[4]);
    }

    FILE* f = std::fopen(argv[3], "wb");
    if (!f) { std::fprintf(stderr, "open failed: %s\n", argv[3]); return 1; }
    std::fwrite(feats.data(), sizeof(float), feats.size(), f);
    std::fclose(f);

    double sum = 0, mn = 1e30, mx = -1e30;
    for (float v : feats) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    std::fprintf(stderr, "mel: T=%d n_mels=%u mean=%.4f min=%.4f max=%.4f -> %s\n",
                 T, mc.n_mels, sum / feats.size(), mn, mx, argv[3]);
    return 0;
}
