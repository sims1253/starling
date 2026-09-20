// greedy_termination_test.cpp — S03 termination-contract regression test.
//
// Contract under test (cpp/lib/qwen_decode.hpp):
//   1. A first-token (prefill-argmax) EOS stops generation for SINGLE-stop
//      decoders (moss/ark/granite/qwen3) exactly as it always did for
//      dual-stop ones (higgs/s1) — the historical eos2-gate that kept
//      single-stop engines decoding past a first-token EOS is retired.
//   2. stop_reason distinguishes WHY generation ended: kEos (truthful
//      completion) vs kBudgetExhausted (max_new_tokens ran out first — a
//      truncation that must never be reported as complete success).
//   3. The debug/probe path obeys the same contract as the release path.
//
// Layers:
//   * Pure predicate checks of lib::generation_stops_on (no model needed):
//     primary EOS, configured/unconfigured secondary, the -1 sentinel.
//   * Engine-level checks through granite::greedy_generate — granite IS a
//     single-stop engine on the shared lib stack — using the tiny synthesized
//     zero-weight granite GGUF (tiny_granite_fixture.hpp): every activation is
//     exactly zero, so the argmax is deterministically token 0. Passing
//     eos_token_id=0 makes the PREFILL token itself the EOS; eos_token_id=5
//     (never emitted) forces the budget to exhaust. CPU-only, no downloads.
//   * A link/null-safety check of the C-side completion entry
//     starling_ggml_moss_last_completion (capi_moss.cpp).
//
// Model-gated (NOT covered here): the GPU-only K-step branch (same predicate,
// exercised only with a device backend), a real MOSS GGUF end-to-end run, and
// the dual-stop engines' own wrappers.
//
// Usage: ./greedy_termination_test   (exit 0 = pass, 1 = assertion failure,
//                                     2 = fixture/model infrastructure failure)

#include "granite/loader.hpp"
#include "granite/llm.hpp"
#include "lib/qwen_decode.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "tiny_granite_fixture.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

extern "C" {
// Internal entry defined in cpp/moss/capi_moss.cpp (parakeet-_pub precedent:
// not in the public starling_ggml.h; direct consumers declare it themselves).
int starling_ggml_moss_last_completion(void * handle);
}

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

// --------------------------------------------------------------------------- //
// Pure predicate checks — the whole stop policy, no model.
// --------------------------------------------------------------------------- //
void predicate_checks() {
    using starling::ggml::lib::GenerateParams;
    using starling::ggml::lib::generation_stops_on;

    // Single-stop (moss-style): the primary EOS always stops.
    const GenerateParams single{8, 128, 0};
    check(generation_stops_on(0, single),
          "predicate: primary EOS stops (single-stop)");
    check(!generation_stops_on(1, single) && !generation_stops_on(15, single),
          "predicate: non-stop tokens do not stop (single-stop)");

    // The -1 "never stop" sentinel matches no real token id (the
    // moss_kstep_oob_test full-decode trick must keep working).
    const GenerateParams none{8, 128, -1};
    bool any = false;
    for (int32_t t = 0; t < 16; ++t) any |= generation_stops_on(t, none);
    check(!any, "predicate: eos=-1 sentinel stops nothing");

    // Dual-stop (higgs/s1-style): primary and CONFIGURED secondary both stop.
    const GenerateParams dual{8, 128, 5, 7};
    check(generation_stops_on(5, dual) && generation_stops_on(7, dual),
          "predicate: primary and secondary both stop (dual-stop)");
    check(!generation_stops_on(6, dual),
          "predicate: other tokens do not stop (dual-stop)");
    const GenerateParams uncfg{8, 128, 5, -1};
    check(generation_stops_on(5, uncfg) && !generation_stops_on(7, uncfg),
          "predicate: unconfigured secondary (-1) stops nothing extra");
}

// --------------------------------------------------------------------------- //
// C-entry sanity — links, and a null handle reports "no completion".
// --------------------------------------------------------------------------- //
void c_entry_checks() {
    check(starling_ggml_moss_last_completion(nullptr) == 0,
          "C entry: null handle reports COMPLETION_NONE (0)");
}

// --------------------------------------------------------------------------- //
// Engine-level checks: real greedy_generate on a real (tiny, zero-weight)
// single-stop granite model. The deterministic argmax is token 0.
// --------------------------------------------------------------------------- //
using starling::ggml::granite::GenerateOptions;
using starling::ggml::granite::GenerateResult;
using starling::ggml::granite::GraniteModel;
using starling::ggml::granite::InputsEmbeds;
using starling::ggml::lib::GenStopReason;

// One greedy run with fixed-shaped zero inputs (hidden=8, S=4). Returns false
// only for infrastructure failures (load/graph errors).
bool run_gen(GraniteModel& m, int32_t eos, int32_t budget, GenerateResult& out,
             std::string& detail) {
    InputsEmbeds in;
    in.width = (int64_t) m.config.llm.hidden;
    in.n_tokens = 4;
    in.data.assign((size_t) in.n_tokens * (size_t) in.width, 0.0f);
    GenerateOptions op;
    op.max_new_tokens = budget;
    op.max_cache_len = (int32_t) m.config.llm.max_cache;
    op.eos_token_id = eos;
    std::string e;
    if (!starling::ggml::granite::greedy_generate(m, in, op, out, e)) {
        detail = "greedy_generate failed: " + e;
        return false;
    }
    return true;
}

std::string ids_detail(const GenerateResult& r) {
    std::string d = "reason=" +
        std::string(r.stop_reason == GenStopReason::kEos ? "eos" : "budget") +
        " ids=[";
    for (size_t i = 0; i < r.ids.size(); ++i) {
        if (i) d += ",";
        d += std::to_string(r.ids[i]);
    }
    return d + "]";
}

void engine_checks(GraniteModel& m) {
    // Precondition: the zero-weight fixture's deterministic argmax is 0.
    // If this ever fails the fixture's determinism argument broke, not the
    // termination contract — the details line says which.
    {
        GenerateResult r;
        std::string d;
        check(run_gen(m, /*eos=*/5, /*budget=*/1, r, d) && r.ids.size() == 1 &&
                  r.ids[0] == 0,
              "engine: zero-weight fixture argmax is token 0",
              d.empty() ? ids_detail(r) : d);
    }

    // (a) First-token EOS, single-stop engine: eos==0 == the prefill argmax.
    //     Must terminate with reason=eos and NOT generate further tokens.
    //     (Pre-S03 behavior: 8 tokens, no stop — this is the exact retired
    //     regression.)
    {
        GenerateResult r;
        std::string d;
        check(run_gen(m, /*eos=*/0, /*budget=*/8, r, d) &&
                  r.stop_reason == GenStopReason::kEos && r.ids.size() == 1 &&
                  r.ids[0] == 0,
              "engine: prefill EOS at token 1 -> reason=eos, 1 token, no further decode",
              d.empty() ? ids_detail(r) : d);
    }

    // (b) Budget exhaustion: eos=5 is never emitted; the budget runs out.
    //     Must be kBudgetExhausted — distinguishable from success-by-EOS —
    //     with the full budget of tokens (non-stop decode parity).
    {
        GenerateResult r;
        std::string d;
        check(run_gen(m, /*eos=*/5, /*budget=*/8, r, d) &&
                  r.stop_reason == GenStopReason::kBudgetExhausted &&
                  r.ids.size() == 8,
              "engine: no stop token -> reason=budget_exhausted, full budget emitted",
              d.empty() ? ids_detail(r) : d);
    }

    // (c) max_new_tokens=1 boundaries: EOS still EOS; otherwise budget.
    {
        GenerateResult r;
        std::string d;
        check(run_gen(m, /*eos=*/0, /*budget=*/1, r, d) &&
                  r.stop_reason == GenStopReason::kEos && r.ids.size() == 1,
              "engine: one-token budget with prefill EOS -> reason=eos",
              d.empty() ? ids_detail(r) : d);
    }
    {
        GenerateResult r;
        std::string d;
        check(run_gen(m, /*eos=*/5, /*budget=*/1, r, d) &&
                  r.stop_reason == GenStopReason::kBudgetExhausted &&
                  r.ids.size() == 1,
              "engine: one-token budget without EOS -> reason=budget_exhausted",
              d.empty() ? ids_detail(r) : d);
    }

    // (d) Debug-path parity: STARLING_GRANITE_DUMP_LAYERS routes greedy onto
    //     the per-layer probe path; the SAME contract must hold there
    //     (pre-S03 the debug path had no initial stop check at all and would
    //     have emitted the full budget).
    {
        const std::string dump_prefix =
            (std::filesystem::temp_directory_path() / "s03_greedy_term_layer").string();
        ::setenv("STARLING_GRANITE_DUMP_LAYERS", dump_prefix.c_str(), 1);
        GenerateResult r;
        std::string d;
        const bool ok = run_gen(m, /*eos=*/0, /*budget=*/8, r, d);
        ::unsetenv("STARLING_GRANITE_DUMP_LAYERS");
        std::error_code ignored;
        for (int li = 0; li < 8; ++li)
            std::filesystem::remove(
                std::filesystem::path(dump_prefix + "_" + std::to_string(li) + ".f32"),
                ignored);
        check(ok && r.stop_reason == GenStopReason::kEos && r.ids.size() == 1,
              "engine: debug/probe path honors prefill EOS identically",
              d.empty() ? ids_detail(r) : d);
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    predicate_checks();
    c_entry_checks();

    TinyGraniteFixture fixture(std::filesystem::temp_directory_path() /
                               "s03_greedy_term_tiny_granite.gguf");
    if (!fixture.wrote()) {
        std::fprintf(stderr, "greedy_termination_test: fixture GGUF not written\n");
        return 2;
    }
    GraniteModel m;
    std::string e;
    if (!m.load(fixture.path.string().c_str(), e)) {
        std::fprintf(stderr, "greedy_termination_test: load: %s\n", e.c_str());
        return 2;
    }
    engine_checks(m);
    starling::ggml::shutdown_backend();

    std::printf("%s\n", failures ? "GREEDY TERMINATION FAILED" : "GREEDY TERMINATION OK");
    return failures ? 1 : 0;
}
