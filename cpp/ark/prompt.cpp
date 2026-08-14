// prompt.cpp — ARK prompt construction + audio-embedding scatter injection.
#include "prompt.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "lib/embed_scatter.hpp"
#include "ggml.h"
namespace starling::ggml::ark {
Prompt build_transcribe_prompt(const Config& c, int64_t mel_frames) {
    Prompt p;
    // Defaults (the empirically captured layout) if the GGUF didn't bake them.
    const std::vector<int32_t> pre = c.prompt_prefix.empty()
        ? std::vector<int32_t>{c.user_id, c.begin_audio_id} : c.prompt_prefix;
    const std::vector<int32_t> suf = c.prompt_suffix.empty()
        ? std::vector<int32_t>{c.end_audio_id, 3167, 3114, 279, 7699, 311, 1467, 13, c.assistant_id}
        : c.prompt_suffix;
    const int64_t n = audio_token_count(mel_frames, c.encoder.merge_factor);
    p.ids.reserve(pre.size() + (size_t) n + suf.size());
    p.audio_mask.reserve(p.ids.capacity());
    for (auto x : pre) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    for (int64_t i = 0; i < n; ++i) { p.ids.push_back(c.audio_token_id); p.audio_mask.push_back(1); }
    for (auto x : suf) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    return p;
}

bool build_inputs_embeds(const ArkModel& m, const Prompt& p, const AudioEncoding& a,
                         InputsEmbeds& out, std::string& err) {
    std::vector<float> emb;
    if (!lib::embed_and_scatter_audio(m.loader, m.config.llm.hidden, p.ids,
                                      p.audio_mask, a.data.data(), a.data.size(),
                                      a.width, a.n_tokens, emb, "ARK", err))
        return false;
    out.data = std::move(emb);
    out.n_tokens = (int64_t) p.ids.size();
    out.width = m.config.llm.hidden;
    return true;
}
} // namespace starling::ggml::ark
