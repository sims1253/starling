// prompt.cpp — Hojo inputs_embeds construction.
//
// The forward path (hojo_asr_model.HOJO_ASR.infer):
//   speech_embeds = ln_speech(bottleneck_out)            # (B, T, 2560)
//   bos_embeds = embed_tokens(bos_id=151644)             # (B, 1, 2560)
//   inputs_embeds = cat([bos_embeds, speech_embeds], dim=1)  # (B, T+1, 2560)
// NO text prompt, NO audio placeholder. The decoder is fed inputs_embeds
// directly (no input_ids expansion).
//
// ln_speech is a LayerNorm(2560) applied to the bottleneck output. The embeds
// are returned in f32 (the prefill casts them to bf16 at the boundary).
#include "prompt.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include <cstring>
#include <vector>

namespace starling::ggml::hojo {

bool build_inputs_embeds(const HojoModel& m, const BottleneckOutput& bn,
                         InputsEmbeds& out, std::string& err) {
    ensure_weights_realized(m.loader);
    const auto& lc = m.config.llm;
    const int64_t hidden = lc.hidden;
    const int64_t T = bn.n_tokens;
    if (bn.width != hidden) {
        err = "Hojo bottleneck width != llm hidden";
        return false;
    }
    // ln_speech(bottleneck) -> [hidden, T] f32.
    std::vector<float> speech_embeds;
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t bne[2] = {hidden, T};
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 2, bne,
            bn.data.data(), bn.data.size() * sizeof(float));
        ggml_tensor* y = ggml_norm(c, x, (float) 1e-5);  // ln_speech eps (default)
        y = ggml_mul(c, y, clone_weight(c, m.loader, "ln_speech.weight"));
        y = ggml_add(c, y, clone_weight(c, m.loader, "ln_speech.bias"));
        return y;  // [hidden, T] f32
    }, speech_embeds);
    if (!ok) { err = "Hojo ln_speech graph failed"; return false; }

    // Look up the bos embedding (1 token).
    std::vector<float> bos_embed;
    ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int32_t id = (int32_t) m.config.bos_token_id;
        int64_t one[1] = {1};
        ggml_tensor* id_t = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                               &id, sizeof(int32_t));
        return ggml_cast(c,
            ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), id_t),
            GGML_TYPE_F32);
    }, bos_embed);
    if (!ok || bos_embed.size() != (size_t) hidden) {
        err = "Hojo bos embed lookup failed";
        return false;
    }

    // cat([bos, speech_embeds]) -> [hidden, T+1].
    out.data.resize((size_t)(T + 1) * hidden);
    std::memcpy(out.data.data(), bos_embed.data(), hidden * sizeof(float));
    std::memcpy(out.data.data() + hidden, speech_embeds.data(),
                speech_embeds.size() * sizeof(float));
    out.n_tokens = T + 1;
    out.width = hidden;
    return true;
}
} // namespace starling::ggml::hojo
