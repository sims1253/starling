// prompt.cpp — audex prompt construction + audio-embedding scatter injection.
#include "prompt.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "lib/embed_scatter.hpp"
#include "ggml.h"
namespace starling::ggml::audex {
Prompt build_transcribe_prompt(const Config& c) {
    Prompt p;
    p.ids.reserve(c.prompt_prefix.size() + (size_t) c.sound_embedding_size +
                  c.prompt_suffix.size());
    p.audio_mask.reserve(p.ids.capacity());
    for (auto x : c.prompt_prefix) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    for (uint32_t i = 0; i < c.sound_embedding_size; ++i) {
        p.ids.push_back(c.audio_token_id);
        p.audio_mask.push_back(1);
    }
    for (auto x : c.prompt_suffix) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    return p;
}

bool build_inputs_embeds(const AudexModel& m, const Prompt& p, const AudioEmbeds& a,
                         InputsEmbeds& out, std::string& err) {
    std::vector<float> emb;
    if (!lib::embed_and_scatter_audio(m.loader, m.config.llm.hidden, p.ids,
                                      p.audio_mask, a.data.data(), a.data.size(),
                                      a.width, a.n_tokens, emb, "AUDEX", err))
        return false;
    out.data = std::move(emb);
    out.n_tokens = (int64_t) p.ids.size();
    out.width = m.config.llm.hidden;
    return true;
}
} // namespace starling::ggml::audex
