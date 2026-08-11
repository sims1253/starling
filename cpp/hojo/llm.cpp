// llm.cpp — Hojo Qwen3-4B decoder (beam-4) on the Starling ggml runtime.
//
// Ported from higgs/llm.cpp (the freshest Qwen3-with-qk_norm). Differences from
// higgs: 36 layers / hidden 2560 / GQA 32/8 / intermediate 9728 / rope_theta
// 5e6 / vocab 151670 (config-driven, no code change), and the decode is BEAM-4
// (not greedy). The Qwen3 op order/dtype discipline (bf/ff casts, f32 RMSNorm,
// rotate-half RoPE in f32, q_norm/k_norm per head) is byte-identical to higgs.
//
// Beam search: HF-compatible (num_beams, repetition_penalty, length_penalty,
// do_sample=False). For correctness-first, each beam step runs a FRESH forward
// over [inputs_embeds_prefix + beam_tokens_so_far] and reads the last-position
// logits (no KV cache -> O(beams*steps*S^2), correct but not optimized). The
// golden's gen_ids is the winning beam; match it.
#include "llm.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace starling::ggml::hojo {
namespace {

ggml_tensor* bf(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_BF16 ? x : ggml_cast(c, x, GGML_TYPE_BF16);
}
ggml_tensor* ff(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
ggml_tensor* wb(ggml_context* c, const HojoModel& m, const std::string& n) {
    return clone_weight(c, m.loader, n.c_str());
}
// nn.Linear (Qwen3 attention has NO q/k/v/o bias).
ggml_tensor* lin(ggml_context* c, const HojoModel& m, ggml_tensor* x, const std::string& n) {
    return bf(c, ggml_mul_mat(c, wb(c, m, n), bf(c, x)));
}
ggml_tensor* addb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf(c, ggml_add(c, ff(c, a), ff(c, b)));
}
ggml_tensor* mulb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf(c, ggml_mul(c, ff(c, a), ff(c, b)));
}
// RMSNorm in f32, scaled by the (bf16) weight.
ggml_tensor* rms(ggml_context* c, const HojoModel& m, ggml_tensor* x, const std::string& n,
                 float eps) {
    ggml_tensor* y = ggml_rms_norm(c, ff(c, x), eps);
    y = bf(c, y);
    return mulb(c, y, bf(c, wb(c, m, n)));
}

std::vector<ggml_bf16_t> tobf(const std::vector<float>& x) {
    std::vector<ggml_bf16_t> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = ggml_fp32_to_bf16(x[i]);
    return r;
}

// Causal additive mask [K, S] f32: 0 where allowed, -inf beyond.
std::vector<float> build_causal_mask(int64_t S, int64_t past) {
    const int64_t K = past + S;
    std::vector<float> mask((size_t) K * S);
    const float neg = -3.3895313892515355e38f;
    for (int64_t qi = 0; qi < S; ++qi)
        for (int64_t j = 0; j < K; ++j)
            mask[(size_t) qi * K + j] = (j <= past + qi) ? 0.0f : neg;
    return mask;
}

// Precompute RoPE cos/sin for positions [0, max_pos) as f32 host tables
// (duplicated halves, matching higgs). Returned as two [D, max_pos] f32 tables.
struct RopeTables {
    std::vector<float> cos, sin;  // [D, max_pos]
    int D = 0, max_pos = 0;
};
RopeTables build_rope_tables(const LlmConfig& lc, int max_pos) {
    RopeTables r;
    r.D = (int) lc.head_dim;
    r.max_pos = max_pos;
    r.cos.assign((size_t) r.D * max_pos, 0.0f);
    r.sin.assign((size_t) r.D * max_pos, 0.0f);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < r.D / 2; ++i) {
            float inv = 1.0f / std::pow((float) lc.rope_theta, (2.0f * i) / r.D);
            float a = (float) p * inv;
            float c = std::cos(a), s = std::sin(a);
            r.cos[(size_t) p * r.D + i] = c;
            r.cos[(size_t) p * r.D + i + r.D / 2] = c;
            r.sin[(size_t) p * r.D + i] = s;
            r.sin[(size_t) p * r.D + i + r.D / 2] = s;
        }
    }
    return r;
}

// Append one Qwen3 layer over x_in [hidden, S] (inputs_embeds for this forward).
// The KV is NOT cached (each beam step is a fresh forward). cs/sn are [D, K] f32
// graph inputs (RoPE rows for positions [0,K)); mask is f32 [K, S].
ggml_tensor* append_layer(ggml_context* c, const HojoModel& m, int li,
                          ggml_tensor* x_in, int64_t S,
                          ggml_tensor* cs, ggml_tensor* sn,
                          ggml_tensor* mask) {
    const auto& lc = m.config.llm;
    const int D = lc.head_dim, H = lc.n_heads, KV = lc.n_kv_heads;
    const std::string p = "llm.blk." + std::to_string(li) + ".";
    ggml_tensor* r = x_in;
    ggml_tensor* n = rms(c, m, x_in, p + "attn_norm.weight", (float) lc.rms_norm_eps);
    ggml_tensor* q = lin(c, m, n, p + "attn.q.weight");
    ggml_tensor* k = lin(c, m, n, p + "attn.k.weight");
    ggml_tensor* v = lin(c, m, n, p + "attn.v.weight");
    q = ggml_reshape_3d(c, q, D, H, S);
    k = ggml_reshape_3d(c, k, D, KV, S);
    v = ggml_reshape_3d(c, v, D, KV, S);
    // qk_norm: RMSNorm per head over head_dim.
    q = rms(c, m, q, p + "attn.q_norm.weight", (float) lc.rms_norm_eps);
    k = rms(c, m, k, p + "attn.k_norm.weight", (float) lc.rms_norm_eps);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [D, S, H]
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));  // [D, S, KV]
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));  // [D, S, KV]
    // RoPE (rotate-half, f32).
    auto rope = [&](ggml_tensor* z, int heads) {
        ggml_tensor* lo = ggml_view_3d(c, z, D / 2, S, heads, z->nb[1], z->nb[2], 0);
        ggml_tensor* hi = ggml_view_3d(c, z, D / 2, S, heads, z->nb[1], z->nb[2],
                                       (size_t)(D / 2) * z->nb[0]);
        ggml_tensor* rot = ggml_concat(c, ggml_scale(c, ff(c, hi), -1.0f), ff(c, lo), 0);
        rot = bf(c, rot);
        return addb(c, mulb(c, z, cs), mulb(c, rot, sn));
    };
    q = rope(q, H);
    k = rope(k, KV);
    // GQA: do NOT explicitly repeat k/v. ggml_repeat has NO CUDA kernel for
    // BF16 (only F32/F16), and using it forces the scheduler onto CPU and
    // mis-sizes a pipeline buffer (the 232 GB crash). Instead, rely on
    // ggml_mul_mat's NATIVE GQA broadcast: pass k=[D,S,KV] / v=[D,S,KV] and
    // q=[D,S,H] directly; mul_mat broadcasts KV->H heads when H%KV==0.
    // Identical to higgs/llm.cpp (which is byte-exact vs the golden).
    const float scale = 1.0f / std::sqrt((float) D);
    ggml_tensor* sc = ggml_mul_mat(c, k, q);                 // [S, S, H]
    sc = bf(c, ggml_scale(c, ff(c, sc), scale));
    ggml_tensor* pr = bf(c, ggml_soft_max_ext(c, ff(c, sc), ff(c, mask), 1.0f, 0.0f));
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));  // [S, D, H]
    ggml_tensor* co = ggml_mul_mat(c, vt, pr);                       // [D, S, H]
    ggml_tensor* joined = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3)),
                                          (int64_t) D * H, S);       // [hidden, S]
    joined = bf(c, joined);
    ggml_tensor* a = lin(c, m, joined, p + "attn.o.weight");
    ggml_tensor* x = addb(c, r, a);
    r = x;
    n = rms(c, m, x, p + "ffn_norm.weight", (float) lc.rms_norm_eps);
    ggml_tensor* g = lin(c, m, n, p + "ffn.gate.weight");
    ggml_tensor* u = lin(c, m, n, p + "ffn.up.weight");
    ggml_tensor* si = bf(c, ggml_silu(c, ff(c, g)));
    ggml_tensor* z = mulb(c, si, u);
    ggml_tensor* dn = lin(c, m, z, p + "ffn.down.weight");
    x = addb(c, r, dn);
    return x;  // [hidden, S] bf16
}

// Run the Qwen3 trunk over [inputs_embeds_prefix (bf16) + token embeds for
// `extra_tokens`], return the last-position logits (f32, vocab). `prefix` is the
// f32 inputs_embeds [hidden, prefix_len]; extra_tokens are appended after.
// positions [0, prefix_len + n_extra). This is the per-beam-step forward.
bool forward_logits(const HojoModel& m, const std::vector<float>& prefix_embeds,
                    int64_t prefix_len, const std::vector<int32_t>& extra_tokens,
                    const RopeTables& rope, std::vector<float>& logits,
                    std::string& e) {
    const auto& lc = m.config.llm;
    const int64_t hidden = lc.hidden;
    const int64_t n_extra = (int64_t) extra_tokens.size();
    const int64_t S = prefix_len + n_extra;
    if (S <= 0) { e = "Hojo forward: empty sequence"; return false; }
    // Build the full inputs_embeds: prefix (bf16) + embed(extra_tokens).
    std::vector<ggml_bf16_t> xb((size_t) S * hidden);
    for (int64_t t = 0; t < prefix_len; ++t)
        for (int64_t d = 0; d < hidden; ++d)
            xb[(size_t) t * hidden + d] = ggml_fp32_to_bf16(prefix_embeds[(size_t) t * hidden + d]);
    // Look up extra token embeds.
    std::vector<ggml_bf16_t> extra_emb;
    if (n_extra > 0) {
        std::vector<float> extra_f;
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t ne[1] = {n_extra};
            ggml_tensor* it = graph_input_tensor(c, GGML_TYPE_I32, 1, ne,
                extra_tokens.data(), extra_tokens.size() * sizeof(int32_t));
            return ggml_cast(c,
                ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), it),
                GGML_TYPE_F32);
        }, extra_f);
        if (!ok) { e = "Hojo embed lookup failed"; return false; }
        extra_emb.resize(extra_f.size());
        for (size_t i = 0; i < extra_f.size(); ++i) extra_emb[i] = ggml_fp32_to_bf16(extra_f[i]);
        for (int64_t t = 0; t < n_extra; ++t)
            for (int64_t d = 0; d < hidden; ++d)
                xb[(size_t)(prefix_len + t) * hidden + d] =
                    extra_emb[(size_t) t * hidden + d];
    }
    std::vector<float> mask = build_causal_mask(S, 0);  // [S, S], past=0
    // RoPE rows for positions [0, S).
    std::vector<float> cs_host((size_t) rope.D * S), sn_host((size_t) rope.D * S);
    for (int64_t t = 0; t < S; ++t)
        for (int d = 0; d < rope.D; ++d) {
            cs_host[(size_t) t * rope.D + d] = rope.cos[(size_t) t * rope.D + d];
            sn_host[(size_t) t * rope.D + d] = rope.sin[(size_t) t * rope.D + d];
        }
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t xne[2] = {hidden, S};
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
            xb.data(), xb.size() * sizeof(ggml_bf16_t));
        int64_t rne[2] = {rope.D, S};
        ggml_tensor* cs = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
            cs_host.data(), cs_host.size() * sizeof(float));
        ggml_tensor* sn = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
            sn_host.data(), sn_host.size() * sizeof(float));
        int64_t mne[2] = {S, S};
        ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
            mask.data(), mask.size() * sizeof(float));
        for (uint32_t li = 0; li < lc.n_layers; ++li)
            x = append_layer(c, m, (int) li, x, S, cs, sn, mt);
        // Final norm + lm_head on the LAST token only.
        ggml_tensor* last = ggml_view_2d(c, x, hidden, 1, x->nb[1],
                                         (size_t)(S - 1) * x->nb[1]);
        ggml_tensor* nrm = rms(c, m, last, "llm.final_norm.weight", (float) lc.rms_norm_eps);
        return ff(c, ggml_mul_mat(c, wb(c, m, "llm.lm_head.weight"), nrm));
    }, logits);
    if (!ok) { e = "Hojo forward graph failed"; return false; }
    return true;
}

// HF repetition penalty: for each token in `input_ids`, if its logit > 0 divide
// by penalty, else multiply. Applied to logits BEFORE softmax.
void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<int32_t>& input_ids,
                              double penalty) {
    for (int32_t tok : input_ids) {
        if (tok < 0 || (size_t) tok >= logits.size()) continue;
        if (logits[(size_t) tok] > 0.0f) logits[(size_t) tok] /= (float) penalty;
        else logits[(size_t) tok] *= (float) penalty;
    }
}

// Beam hypothesis for HF beam search.
struct Beam {
    std::vector<int32_t> tokens;  // generated tokens (excl. prefix)
    float score = 0.0f;           // cumulative log-prob sum
    bool finished = false;
};

} // namespace

// ---- Greedy (num_beams==1 or diagnostic) ----
bool greedy_generate(const HojoModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    if (i.n_tokens <= 0 || i.width != (int64_t) m.config.llm.hidden) {
        e = "invalid Hojo prefill shape";
        return false;
    }
    ensure_weights_realized(m.loader);
    const int32_t eos = op.eos_token_id;
    const int32_t max_new = (int32_t) op.max_new_tokens;
    RopeTables rope = build_rope_tables(m.config.llm, (int)(i.n_tokens + max_new + 4));
    // Prefill.
    std::vector<int32_t> extra;
    if (!forward_logits(m, i.data, i.n_tokens, extra, rope, o.prefill_logits, e)) return false;
    int32_t prev = 0;
    {
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < o.prefill_logits.size(); ++k)
            if (o.prefill_logits[k] > mx) { mx = o.prefill_logits[k]; prev = (int32_t) k; }
    }
    o.ids.push_back(prev);
    if (prev == eos) { o.hit_eos = true; return true; }
    while ((int) o.ids.size() < max_new) {
        std::vector<float> lg;
        if (!forward_logits(m, i.data, i.n_tokens, o.ids, rope, lg, e)) return false;
        apply_repetition_penalty(lg, o.ids, op.repetition_penalty);
        int32_t next = 0;
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < lg.size(); ++k)
            if (lg[k] > mx) { mx = lg[k]; next = (int32_t) k; }
        o.ids.push_back(next);
        if (next == eos) { o.hit_eos = true; break; }
    }
    return true;
}

// ---- Beam search (HF-compatible, num_beams, repetition_penalty, length_penalty) ----
bool beam_generate(const HojoModel& m, const InputsEmbeds& i,
                   const GenerateOptions& op, GenerateResult& o, std::string& e) {
    if (op.num_beams <= 1 || std::getenv("STARLING_HOJO_GREEDY")) return greedy_generate(m, i, op, o, e);
    if (i.n_tokens <= 0 || i.width != (int64_t) m.config.llm.hidden) {
        e = "invalid Hojo prefill shape";
        return false;
    }
    ensure_weights_realized(m.loader);
    const int32_t eos = op.eos_token_id;
    const int32_t max_new = (int32_t) op.max_new_tokens;
    const int B = (int) op.num_beams;
    const int vocab = (int) m.config.llm.vocab;
    const float lp = (float) op.length_penalty;
    RopeTables rope = build_rope_tables(m.config.llm, (int)(i.n_tokens + max_new + 4));

    // Prefill (step 0): logits over vocab at the last prefix position.
    std::vector<int32_t> empty;
    if (!forward_logits(m, i.data, i.n_tokens, empty, rope, o.prefill_logits, e)) return false;
    std::vector<float> pf = o.prefill_logits;
    // Take top-2B candidates.
    struct Cand { int32_t tok; float score; };
    std::vector<Cand> top;
    {
        std::vector<std::pair<float, int32_t>> ranked;
        ranked.reserve(vocab);
        for (int k = 0; k < vocab; ++k) ranked.push_back({pf[k], k});
        std::partial_sort(ranked.begin(), ranked.begin() + 2 * B, ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int k = 0; k < 2 * B; ++k) top.push_back({ranked[k].second, ranked[k].first});
    }
    // log-softmax of prefill logits (HF applies log_softmax then scores).
    float pfmax = *std::max_element(pf.begin(), pf.end());
    float pfsum = 0.0f;
    for (int k = 0; k < vocab; ++k) pfsum += std::exp(pf[k] - pfmax);
    float pflogsum = std::log(pfsum) + pfmax;
    // Initialize B beams (beam 0 score = logprob, others = -inf until step 1,
    // matching HF: at step 0 only beam 0 expands, then top-2B picks the beams).
    std::vector<Beam> beams(B);
    for (int b = 0; b < B; ++b) beams[b].score = -std::numeric_limits<float>::infinity();
    // HF step-0 selection: the first token is chosen from beam 0's top-2B.
    std::vector<Beam> finished;
    {
        // The expansion score for a finished candidate uses cur_len = 1 (one
        // generated token). score = logprob / (1 ** lp).
        std::vector<std::tuple<float, int, int32_t>> ranked;  // (score, beam, tok)
        for (int k = 0; k < 2 * B; ++k) {
            float lp_ = (pf[top[k].tok] - pflogsum);  // logprob
            float sc = lp_ / std::pow(1.0f, lp);
            ranked.push_back({sc, 0, top[k].tok});
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        int picked = 0;
        for (auto& [sc, beam, tok] : ranked) {
            if (picked >= B) break;
            if (tok == eos) {
                Beam fb; fb.tokens = {tok}; fb.score = sc; fb.finished = true;
                finished.push_back(fb);
            } else {
                beams[picked].tokens = {tok};
                beams[picked].score = sc;  // cumulative (raw logprob sum)
                ++picked;
            }
        }
    }

    // Decode steps.
    for (int step = 1; step < max_new && (int) finished.size() < B; ++step) {
        // For each active beam, run a forward over [prefix + beam.tokens] and
        // get the next-token logits.
        std::vector<std::vector<float>> beam_logits(B);
        bool any_active = false;
        for (int b = 0; b < B; ++b) {
            if (beams[b].finished) continue;
            any_active = true;
            if (!forward_logits(m, i.data, i.n_tokens, beams[b].tokens, rope,
                                beam_logits[b], e)) return false;
            // Repetition penalty over the full generated sequence (HF uses
            // input_ids = the generated tokens; the prefix is inputs_embeds,
            // not input_ids, so it's excluded).
            apply_repetition_penalty(beam_logits[b], beams[b].tokens, op.repetition_penalty);
        }
        if (!any_active) break;
        // log-softmax each beam's logits, accumulate candidate scores.
        // cur_len = step + 1 (number of generated tokens AFTER this step).
        const float cur_len = (float)(step + 1);
        std::vector<std::tuple<float, int, int32_t>> ranked;  // (cmp_score, beam, tok)
        ranked.reserve((size_t) B * vocab);
        for (int b = 0; b < B; ++b) {
            if (beams[b].finished) continue;
            const auto& lg = beam_logits[b];
            float lmax = *std::max_element(lg.begin(), lg.end());
            float lsum = 0.0f;
            for (int k = 0; k < vocab; ++k) lsum += std::exp(lg[k] - lmax);
            float llogsum = std::log(lsum) + lmax;
            for (int k = 0; k < vocab; ++k) {
                float logp = lg[k] - llogsum;
                float raw = beams[b].score + logp;  // cumulative logprob
                float cmp = raw / std::pow(cur_len, lp);
                ranked.push_back({cmp, b, (int32_t) k});
            }
        }
        std::partial_sort(ranked.begin(), ranked.begin() + 2 * B, ranked.end(),
                          [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        // Select next-beam tokens (2B candidates -> B active + EOS finishes).
        std::vector<Beam> next_beams(B);
        for (int b = 0; b < B; ++b) next_beams[b].score = -std::numeric_limits<float>::infinity();
        int picked = 0;
        for (auto& [cmp, b, tok] : ranked) {
            if (picked >= B && (int) finished.size() >= B) break;
            if (tok == eos) {
                Beam fb = beams[b];
                fb.tokens.push_back(tok);
                fb.score = beams[b].score;  // raw cumulative (no +logp for eos)
                fb.finished = true;
                // HF comparison score for finished: raw / cur_len**lp.
                finished.push_back(fb);
                // finished comparison uses cmp of the eos candidate.
                // (We carry cmp implicitly via insertion order; re-sort below.)
                if ((int) finished.size() <= B) finished.back().score = beams[b].score;
                continue;
            }
            if (picked >= B) continue;
            next_beams[picked].tokens = beams[b].tokens;
            next_beams[picked].tokens.push_back(tok);
            // cumulative raw score (HF stores raw; cmp uses length-normalized).
            next_beams[picked].score = beams[b].score;
            // We'll add the logp on the NEXT step's expansion; but HF adds it
            // to beam_scores at selection. Store raw = beam_score + logp.
            // Recompute logp for this token to store the cumulative.
            // (cheaper: re-derive from beam_logits.)
            // Use the cmp-derived raw = cmp * cur_len**lp (avoids re-softmax).
            next_beams[picked].score = cmp * std::pow(cur_len, lp);
            ++picked;
        }
        // If we didn't fill B beams (lots of EOS), pad with the best remaining.
        beams = std::move(next_beams);
        if (picked == 0 && (int) finished.size() >= B) break;
    }

    // Pick the best finished hypothesis (length-normalized score); fall back to
    // the best active beam if none finished.
    if (finished.empty()) {
        int best = 0;
        for (int b = 1; b < B; ++b)
            if (beams[b].score > beams[best].score) best = b;
        o.ids = beams[best].tokens;
        o.hit_eos = false;
    } else {
        int best = 0;
        // Compare finished by their length-normalized score.
        auto norm_score = [&](const Beam& bm) {
            float len = (float) bm.tokens.size();
            return bm.score / std::pow(len, lp);
        };
        for (int k = 1; k < (int) finished.size(); ++k)
            if (norm_score(finished[k]) > norm_score(finished[best])) best = k;
        o.ids = finished[best].tokens;
        o.hit_eos = true;
    }
    return true;
}
} // namespace starling::ggml::hojo
