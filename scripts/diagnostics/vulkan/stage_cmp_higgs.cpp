// Stage-by-stage comparator for the higgs ggml engine (same shape as
// stage_cmp.cpp / stage_cmp_granite.cpp). Stages: mel -> audio encoder ->
// inputs_embeds -> prefill logits -> first decoded ids. On CUDA this engine
// is byte-exact vs the Transformers golden; on CPU it produces garbage
// ("inaudience" repetition, WER 100%) — this harness locates the first bad
// stage. Works on any STARLING_GGML_DEVICE.
#include "higgs/loader.hpp"
#include "higgs/mel.hpp"
#include "higgs/audio_encoder.hpp"
#include "higgs/prompt.hpp"
#include "higgs/llm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace starling::ggml;

static void stat(const char* stage, const std::vector<float>& v) {
    if (v.empty()) { printf("%-14s EMPTY\n", stage); return; }
    double sum = 0, mx = 0; size_t nnan = 0; uint64_t mix = 1469598103934665603ull;
    for (float x : v) {
        if (std::isnan(x)) ++nnan;
        sum += x; mx = std::max(mx, (double)std::fabs(x));
        uint32_t b; memcpy(&b, &x, 4);
        mix = (mix ^ b) * 1099511628211ull;
    }
    float f0 = v[0], f1 = v.size() > 1 ? v[1] : 0.0f, f2 = v.size() > 2 ? v[2] : 0.0f;
    printf("%-14s n=%-7zu sum=%-14.4f maxabs=%-10.6g nan=%-3zu fnv=%016llx first=%g %g %g\n",
           stage, v.size(), sum, mx, nnan, (unsigned long long)mix, f0, f1, f2);
}

int main() {
    std::string e;
    higgs::HiggsModel m;
    if (!m.load("models/higgs-audio-v3-bf16-exact.gguf", e)) { fprintf(stderr, "load: %s\n", e.c_str()); return 2; }

    FILE* f = fopen("tests/fixtures/short.wav", "rb");
    if (!f) { fprintf(stderr, "no fixture wav\n"); return 2; }
    if (fseek(f, 0, SEEK_END) != 0) return 2;
    long sz = ftell(f);
    if (sz <= 12) return 2;
    if (fseek(f, 0, SEEK_SET) != 0) return 2;
    std::vector<unsigned char> wav(sz);
    if (fread(wav.data(), 1, sz, f) != (size_t)sz) return 2;
    fclose(f);
    size_t off = 12;
    std::vector<float> pcm;
    while (off + 8 <= (size_t)sz) {
        unsigned clen = wav[off+4] | (wav[off+5]<<8) | (wav[off+6]<<16) | ((unsigned)wav[off+7]<<24);
        if (off + 8 + clen > (size_t)sz) return 2;
        if (!memcmp(wav.data()+off, "data", 4)) {
            for (size_t i = 0; i + 1 < clen; i += 2) {
                short s = wav[off+8+i] | (wav[off+8+i+1]<<8);
                pcm.push_back(s / 32768.0f);
            }
            break;
        }
        off += 8 + clen + (clen & 1);
    }
    printf("pcm samples=%zu\n", pcm.size());

    higgs::MelFeatures mel;
    if (!higgs::compute_log_mel(m.config, m.loader, pcm.data(), pcm.size(), mel, e)) { fprintf(stderr, "mel: %s\n", e.c_str()); return 2; }
    stat("mel", mel.f32);

    higgs::AudioEncoding enc;
    if (!higgs::encode_audio_and_project(m, mel, enc, e)) { fprintf(stderr, "enc: %s\n", e.c_str()); return 2; }
    stat("encoder", enc.data);

    auto prompt = higgs::build_transcribe_prompt(m.config, {(int64_t)enc.n_tokens});
    higgs::InputsEmbeds ie;
    if (!higgs::build_inputs_embeds(m, prompt, enc, ie, e)) { fprintf(stderr, "embeds: %s\n", e.c_str()); return 2; }
    stat("inputs_embeds", ie.data);

    higgs::GenerateOptions op;
    higgs::GenerateResult gr;
    if (!higgs::greedy_generate(m, ie, op, gr, e)) { fprintf(stderr, "gen: %s\n", e.c_str()); return 2; }
    printf("prefill ");
    stat("logits", gr.prefill_logits);
    printf("ids(first24):");
    for (size_t i = 0; i < 24 && i < gr.ids.size(); ++i) printf(" %d", gr.ids[i]);
    printf("\n");
    return 0;
}
