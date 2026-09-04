// Stage-by-stage CPU-vs-Vulkan comparator for the granite ggml engine
// (same shape as stage_cmp.cpp for moss). Stages: mel -> encoder+projector ->
// inputs_embeds -> prefill logits -> first decoded ids.
//
// STAGE_CMP_EMBEDS=<file> force-feeds the embeds (CPU run writes
// /tmp/embeds_dump_granite.f32 when unset) so the LLM stages compare on
// byte-equal inputs across backends.
#include "granite/loader.hpp"
#include "granite/prompt.hpp"
#include "granite/mel.hpp"
#include "granite/encoder.hpp"
#include "granite/llm.hpp"

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
    double sum = 0, mx = 0; uint64_t mix = 1469598103934665603ull;
    for (float x : v) {
        sum += x; mx = std::max(mx, (double)std::fabs(x));
        uint32_t b; memcpy(&b, &x, 4);
        mix = (mix ^ b) * 1099511628211ull;
    }
    float f0 = v.size() > 0 ? v[0] : 0.0f, f1 = v.size() > 1 ? v[1] : 0.0f,
          f2 = v.size() > 2 ? v[2] : 0.0f;
    printf("%-14s n=%-7zu sum=%-14.4f maxabs=%-10.6g fnv=%016llx first=%g %g %g\n",
           stage, v.size(), sum, mx, (unsigned long long)mix, f0, f1, f2);
}

int main() {
    std::string e;
    granite::GraniteModel m;
    if (!m.load("models/granite-speech-4.1-2b-bf16-exact.gguf", e)) { fprintf(stderr, "load: %s\n", e.c_str()); return 2; }

    FILE* f = fopen("tests/fixtures/short.wav", "rb");
    if (!f) { fprintf(stderr, "no fixture wav\n"); return 2; }
    if (fseek(f, 0, SEEK_END) != 0) { fprintf(stderr, "seek failed\n"); return 2; }
    long sz = ftell(f);
    if (sz <= 12) { fprintf(stderr, "bad wav size\n"); fclose(f); return 2; }
    if (fseek(f, 0, SEEK_SET) != 0) { fprintf(stderr, "seek failed\n"); fclose(f); return 2; }
    std::vector<unsigned char> wav(sz);
    if (fread(wav.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); fclose(f); return 2; }
    fclose(f);
    size_t off = 12;
    std::vector<float> pcm;
    while (off + 8 <= (size_t)sz) {
        unsigned clen = wav[off+4] | (wav[off+5]<<8) | (wav[off+6]<<16) | ((unsigned)wav[off+7]<<24);
        if (off + 8 + clen > (size_t)sz) { fprintf(stderr, "chunk overruns file\n"); return 2; }
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

    granite::MelFeatures mel;
    if (!granite::compute_log_mel(m.config, m.loader, pcm.data(), pcm.size(), mel, e)) { fprintf(stderr, "mel: %s\n", e.c_str()); return 2; }
    stat("mel", mel.f32);

    granite::AudioEmbeds enc;
    if (!granite::encode_audio_and_project(m, mel, enc, e)) { fprintf(stderr, "enc: %s\n", e.c_str()); return 2; }
    stat("encoder", enc.data);

    auto prompt = granite::build_transcribe_prompt(m.config, (int64_t)pcm.size());
    granite::InputsEmbeds ie;
    if (const char* ov = std::getenv("STAGE_CMP_EMBEDS")) {
        FILE* g = fopen(ov, "rb");
        if (!g) { fprintf(stderr, "cannot open %s\n", ov); return 2; }
        fseek(g, 0, SEEK_END); long gs = ftell(g); fseek(g, 0, SEEK_SET);
        if (gs <= 0 || gs % 4 != 0) { fprintf(stderr, "bad embeds size %ld\n", gs); fclose(g); return 2; }
        ie.data.resize(gs / 4);
        if (fread(ie.data.data(), 4, gs / 4, g) != (size_t)(gs / 4)) { fprintf(stderr, "short embeds read\n"); fclose(g); return 2; }
        fclose(g);
        ie.n_tokens = (int64_t)prompt.ids.size(); ie.width = m.config.llm.hidden;
        stat("inputs_embeds(OVR)", ie.data);
    } else {
        if (!granite::build_inputs_embeds(m, prompt, enc, ie, e)) { fprintf(stderr, "embeds: %s\n", e.c_str()); return 2; }
        stat("inputs_embeds", ie.data);
        const char* dump = std::getenv("STAGE_CMP_DUMP");
        std::string dp = dump && *dump ? dump : "/tmp/embeds_dump_granite.f32";
        FILE* d = fopen(dp.c_str(), "wb");
        if (!d) { fprintf(stderr, "cannot open %s for dump\n", dp.c_str()); return 2; }
        if (fwrite(ie.data.data(), 4, ie.data.size(), d) != ie.data.size()) { fprintf(stderr, "short dump write\n"); fclose(d); return 2; }
        fclose(d);
    }

    granite::PrefillResult pf;
    if (!granite::llm_prefill(m, ie, 640, pf, e)) { fprintf(stderr, "prefill: %s\n", e.c_str()); return 2; }
    stat("prefill_logits", pf.logits);

    granite::GenerateResult gr;
    granite::GenerateOptions op;
    if (!granite::greedy_generate(m, ie, op, gr, e)) { fprintf(stderr, "gen: %s\n", e.c_str()); return 2; }
    printf("ids(first16):");
    for (size_t i = 0; i < 16 && i < gr.ids.size(); ++i) printf(" %lld", (long long)gr.ids[i]);
    printf("\n");
    return 0;
}
