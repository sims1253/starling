// prompt.cpp — ARK prompt construction + audio-embedding scatter injection.
#include "prompt.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
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
                         InputsEmbeds& out, std::string& e) {
    if (p.ids.size() != p.audio_mask.size()) { e = "invalid ARK prompt mask"; return false; }
    size_t slots = 0;
    for (auto x : p.audio_mask) slots += x != 0;
    if (a.width != (int64_t) m.config.llm.hidden || a.data.size() % (size_t) a.width != 0) {
        e = "audio/prompt scatter size mismatch";
        return false;
    }
    // The adapter may emit fewer (long audio, mel capped) or more features than
    // there are audio slots: zero-pad or truncate to the slot count, exactly
    // like modeling_arkasr._inject_audio_embeddings.
    const int64_t sa = a.n_tokens;
    ensure_weights_realized(m.loader);
    std::vector<int32_t> ids = p.ids;
    std::vector<ggml_bf16_t> ah;
    if (sa > 0) {
        ah.resize((size_t) sa * (size_t) a.width);
        for (size_t i = 0; i < ah.size(); ++i) ah[i] = ggml_fp32_to_bf16(a.data[i]);
    }
    std::vector<float> emb;
    bool ok = run_graph([&](ggml_context* c) {
        int64_t ne[1] = {(int64_t) ids.size()};
        auto* it = graph_input_tensor(c, GGML_TYPE_I32, 1, ne, ids.data(),
                                      ids.size() * sizeof(ids[0]));
        return ggml_cast(c, ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), it),
                         GGML_TYPE_F32);
    }, emb);
    if (!ok) { e = "ARK embedding lookup failed"; return false; }
    size_t row = 0;
    for (size_t i = 0; i < p.ids.size(); ++i) {
        if (!p.audio_mask[i]) continue;
        for (size_t d = 0; d < (size_t) a.width; ++d) {
            // row indexes into the (possibly truncated) feature stream. When the
            // adapter emits fewer features than audio slots (long audio, mel
            // capped), the overflow slots must be ZEROED — matching HF
            // modeling_arkasr._inject_audio_embeddings (feat_i.new_zeros pad),
            // which overwrites the slot with a literal zero, NOT the embedded
            // audio_token_id value that the lookup left there.
            if (row < (size_t) sa)
                emb[i * (size_t) a.width + d] = ggml_bf16_to_fp32(ah[row * (size_t) a.width + d]);
            else
                emb[i * (size_t) a.width + d] = 0.0f;
        }
        ++row;
    }
    out.data = std::move(emb);
    out.n_tokens = (int64_t) p.ids.size();
    out.width = m.config.llm.hidden;
    return true;
}
} // namespace starling::ggml::ark
