// Stage-by-stage CPU-vs-Vulkan comparator for the moss ggml engine.
// Prints a summary line per pipeline stage; run twice with different
// STARLING_GGML_DEVICE and diff.
//
// Stages: mel -> audio encoder -> adapter -> inputs_embeds -> prefill logits
//         -> first decoded ids.
#include "moss/loader.hpp"
#include "moss/prompt.hpp"
#include "moss/mel.hpp"
#include "moss/audio_encoder.hpp"
#include "moss/adapter.hpp"
#include "moss/llm.hpp"
#include "moss/tokenizer.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Adapt includes if names differ; pull what exists.

using namespace starling::ggml;

static void stat(const char* stage, const std::vector<float>& v) {
    if (v.empty()) { printf("%-14s EMPTY\n", stage); return; }
    double sum = 0, mx = 0; uint64_t mix = 1469598103934665603ull;
    for (float x : v) {
        sum += x; mx = std::max(mx, (double)std::fabs(x));
        uint32_t b; memcpy(&b, &x, 4);
        mix = (mix ^ b) * 1099511628211ull;
    }
    printf("%-14s n=%-7zu sum=%-14.4f maxabs=%-10.6g fnv=%016llx first=%g %g %g\n",
           stage, v.size(), sum, mx, (unsigned long long)mix, v[0], v[1], v[2]);
}

int main() {
    std::string e;
    moss::MossModel m;
    if (!m.load("models/moss-transcribe-preview-2b-bf16-exact.gguf", e)) { fprintf(stderr, "load: %s\n", e.c_str()); return 2; }

    // 1. mel from fixture PCM
    FILE* f = fopen("tests/fixtures/short.wav", "rb");
    if (!f) { fprintf(stderr, "no fixture wav\n"); return 2; }
    // read wav via soundfile-less minimal RIFF parse: find data chunk, 16-bit mono 16k
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> wav(sz);
    fread(wav.data(), 1, sz, f); fclose(f);
    size_t off = 12;
    std::vector<float> pcm;
    while (off + 8 < (size_t)sz) {
        unsigned clen = wav[off+4] | (wav[off+5]<<8) | (wav[off+6]<<16) | ((unsigned)wav[off+7]<<24);
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

    moss::MelFeatures mel;
    if (!moss::compute_log_mel(m.config, m.loader, pcm.data(), pcm.size(), mel, e)) { fprintf(stderr, "mel: %s\n", e.c_str()); return 2; }
    stat("mel", mel.f32);

    moss::AudioEncoding enc;
    if (!moss::encode_audio(m, mel, enc, e)) { fprintf(stderr, "enc: %s\n", e.c_str()); return 2; }
    stat("encoder", enc.data);

    moss::AudioEncoding adap;
    if (!moss::apply_adapter(m, enc, adap, e)) { fprintf(stderr, "adapter: %s\n", e.c_str()); return 2; }
    stat("adapter", adap.data);

    auto prompt = moss::build_transcribe_prompt(m.config, 743);
    moss::InputsEmbeds ie;
    if (const char* ov = getenv("STAGE_CMP_EMBEDS")) {
        FILE* g = fopen(ov, "rb");
        fseek(g, 0, SEEK_END); long gs = ftell(g); fseek(g, 0, SEEK_SET);
        ie.data.resize(gs / 4);
        fread(ie.data.data(), 4, gs / 4, g); fclose(g);
        ie.n_tokens = (int64_t)prompt.ids.size(); ie.width = m.config.llm.hidden;
        stat("inputs_embeds(OVR)", ie.data);
    } else {
        if (!moss::build_inputs_embeds(m, prompt, adap, ie, e)) { fprintf(stderr, "embeds: %s\n", e.c_str()); return 2; }
        stat("inputs_embeds", ie.data);
        FILE* d = fopen("/tmp/embeds_dump.f32", "wb");
        fwrite(ie.data.data(), 4, ie.data.size(), d); fclose(d);
    }

    moss::PrefillResult pf;
    if (!moss::llm_prefill(m, ie, 2048, pf, e)) { fprintf(stderr, "prefill: %s\n", e.c_str()); return 2; }
    stat("prefill_logits", pf.logits);

    // greedy decode a few tokens
    moss::GenerateResult gr;
    moss::GenerateOptions op;
    if (!moss::greedy_generate(m, ie, op, gr, e)) { fprintf(stderr, "gen: %s\n", e.c_str()); return 2; }
    printf("ids(first16):");
    for (size_t i = 0; i < 16 && i < gr.ids.size(); ++i) printf(" %lld", (long long)gr.ids[i]);
    printf("\n");
    return 0;
}
