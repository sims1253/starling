// embed_scatter.cpp — prompt embedding + audio scatter (see embed_scatter.hpp).
#include "embed_scatter.hpp"
#include <cstdio>

namespace starling::ggml::lib {

bool embed_and_scatter_audio(const ModelLoader& ml, int64_t hidden,
                             const std::vector<int32_t>& ids,
                             const std::vector<uint8_t>& mask,
                             const float* audio_data, size_t audio_len,
                             int64_t audio_width, int64_t audio_tokens,
                             std::vector<float>& out, const char* label,
                             std::string& err) {
    if (ids.size() != mask.size()) {
        err = std::string("invalid ") + label + " prompt mask";
        return false;
    }
    if (audio_width != hidden || audio_len % (size_t) audio_width != 0) {
        err = std::string(label) + " audio/prompt scatter size mismatch";
        return false;
    }
    const int64_t sa = audio_tokens;
    ensure_weights_realized(ml);
    std::vector<int32_t> idv = ids;
    std::vector<ggml_bf16_t> ah;
    if (sa > 0) {
        ah.resize((size_t) sa * (size_t) audio_width);
        for (size_t i = 0; i < ah.size(); ++i) ah[i] = ggml_fp32_to_bf16(audio_data[i]);
    }
    std::vector<float> emb;
    bool ok = run_graph([&](ggml_context* c) {
        int64_t ne[1] = {(int64_t) idv.size()};
        auto* it = graph_input_tensor(c, GGML_TYPE_I32, 1, ne, idv.data(),
                                      idv.size() * sizeof(idv[0]));
        return ggml_cast(c, ggml_get_rows(c, clone_weight(c, ml, "llm.embed.weight"), it),
                         GGML_TYPE_F32);
    }, emb);
    if (!ok) {
        err = std::string(label) + " embedding lookup failed";
        return false;
    }
    size_t row = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!mask[i]) continue;
        for (size_t d = 0; d < (size_t) audio_width; ++d) {
            // row indexes into the (possibly truncated) feature stream. When the
            // audio path emits fewer features than audio slots (long audio, mel
            // capped), the overflow slots must be ZEROED — matching the HF
            // zero-init-then-scatter, which overwrites the slot with a literal
            // zero, NOT the embedded audio-placeholder id the lookup left there.
            if (row < (size_t) sa)
                emb[i * (size_t) audio_width + d] =
                    ggml_bf16_to_fp32(ah[row * (size_t) audio_width + d]);
            else
                emb[i * (size_t) audio_width + d] = 0.0f;
        }
        ++row;
    }
    out = std::move(emb);
    return true;
}

} // namespace starling::ggml::lib
