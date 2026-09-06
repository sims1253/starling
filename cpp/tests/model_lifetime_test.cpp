// Tiny, deterministic weights exercise model isolation and lifetime without downloads.
#include "runtime/model_loader.hpp"
#include "runtime/backend.hpp"
#include "runtime/lru_cache.hpp"
#include "runtime/graph.hpp"
#include "lib/qwen_decode.hpp"
#include "include/starling_ggml.h"
#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace starling::ggml;
namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Fixture {
    std::filesystem::path path;
    Fixture(int winner) {
        path = std::filesystem::temp_directory_path() /
            ("starling-lifetime-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(winner) + ".gguf");
        auto* ctx = ggml_init({1 << 20, nullptr, false});
        auto* gguf = gguf_init_empty();
        auto* embed = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, 32);
        ggml_set_name(embed, "llm.embed.weight");
        for (int row = 0; row < 32; ++row)
            for (int col = 0; col < 32; ++col)
                static_cast<ggml_bf16_t*>(embed->data)[row * 32 + col] = ggml_fp32_to_bf16(row == winner ? 1.f : 0.f);
        auto* norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        ggml_set_name(norm, "llm.final_norm.weight");
        for (int i = 0; i < 32; ++i) static_cast<float*>(norm->data)[i] = 1.f;
        gguf_add_tensor(gguf, embed);
        gguf_add_tensor(gguf, norm);
        // One real transformer layer keeps position/mask/KV inputs connected.
        // Zero projections make both residual branches identities.
        for (const char* name : {"attn.q", "attn.k", "attn.v", "attn.o", "ffn.gate", "ffn.up", "ffn.down"}) {
            auto* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, 32);
            const std::string key = std::string("llm.blk.0.") + name + ".weight";
            ggml_set_name(tensor, key.c_str());
            for (int i = 0; i < 32 * 32; ++i) static_cast<ggml_bf16_t*>(tensor->data)[i] = ggml_fp32_to_bf16(0.f);
            gguf_add_tensor(gguf, tensor);
        }
        for (const char* name : {"attn_norm", "attn.q_norm", "attn.k_norm", "ffn_norm"}) {
            auto* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
            const std::string key = std::string("llm.blk.0.") + name + ".weight";
            ggml_set_name(tensor, key.c_str());
            for (int i = 0; i < 32; ++i) static_cast<float*>(tensor->data)[i] = 1.f;
            gguf_add_tensor(gguf, tensor);
        }
        const bool wrote = gguf_write_to_file(gguf, path.string().c_str(), false);
        gguf_free(gguf);
        ggml_free(ctx);
        check(wrote, "write fixture");
    }
    ~Fixture() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};
const lib::QwenDecodeSpec spec{false, "STARLING_LIFETIME_TEST", "test", "test"};
void prefill(ModelLoader& loader, int winner) {
    lib::QwenDecodeCtx model{spec, loader, {1, 32, 1, 1, 32, 16, 10000.f, 1e-6f}};
    lib::InputsEmbeds inputs{std::vector<float>(64, 1.f), 2, 32};
    lib::PrefillResult result;
    std::string error;
    check(lib::llm_prefill(model, inputs, 16, result, error), error.c_str());
    check(result.first_token == winner, "prefill reused another model's weights");
    if (global_backend().is_gpu()) check(lib::prefill_replay_cache_size(loader) == 1, "per-model prefill cache missing");
}
struct CacheA {
    std::vector<int>& order;
    explicit CacheA(std::vector<int>& out) : order(out) {}
    ~CacheA() { order.push_back(1); }
};
struct CacheB {
    std::vector<int>& order;
    explicit CacheB(std::vector<int>& out) : order(out) {}
    ~CacheB() { order.push_back(2); }
};
}

int main() {
    try {
        Fixture a(3), b(7);
        // Backend creation before the first loader also tests exit-registration order.
        global_backend();
        {
            ModelLoader empty;
            bool threw = false;
            try { ensure_weights_realized(empty); } catch (const std::runtime_error&) { threw = true; }
            check(threw, "weight realization failure must propagate");
        }
        // A failed graph build must release its context and restore the input
        // registration scope before the next request reaches this thread.
        auto broken_build = [](ggml_context* ctx) -> ggml_tensor* {
            auto* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            add_graph_root(t);
            throw std::runtime_error("injected graph build failure");
        };
        for (int mode = 0; mode < 2; ++mode) {
            bool threw = false;
            try {
                if (mode) { ReplayGraph graph(global_backend(), broken_build); }
                else { std::vector<float> out; global_backend().compute(broken_build, out); }
            } catch (const std::runtime_error&) { threw = true; }
            check(threw, "graph build error did not propagate");
            add_graph_root(nullptr); // no active scope: must be a no-op
        }
        LruCache<int, int> retry_cache(2);
        try { retry_cache.get_or_init(1, [](int&) { throw std::runtime_error("init"); }); }
        catch (const std::runtime_error&) {}
        check(retry_cache.size() == 0, "failed cache entry survived initialization");
        check(*retry_cache.get_or_init(1, [](int& value) { value = 42; }) == 42, "cache retry reused partial entry");

        starling_ggml_load(static_cast<starling_ggml_model>(0), nullptr);
        const std::string caller_error = starling_ggml_last_error(nullptr);
        const char* worker_backend_name = nullptr;
        std::thread worker([&] {
            starling_ggml_load(STARLING_GGML_PARAKEET_TDT, nullptr);
            worker_backend_name = starling_ggml_backend_name();
        });
        worker.join();
        check(caller_error == starling_ggml_last_error(nullptr), "another thread overwrote last error");
        check(worker_backend_name && std::string(worker_backend_name) == starling_ggml_backend_name(), "backend name outlived its calling thread");
        for (int cycle = 0; cycle < 12; ++cycle) {
            ModelLoader first;
            check(first.load(a.path.string().c_str()), "load A");
            first.add_tensor_alias("embedding-alias", "llm.embed.weight");
            prefill(first, 3);
            check(first.tensor("embedding-alias") == first.tensor("llm.embed.weight"), "realization lost tensor alias identity");
            {
                ModelLoader second;
                check(second.load(b.path.string().c_str()), "load B");
                prefill(second, 7);
                prefill(first, 3);
            }
            prefill(first, 3);
            ModelLoader reloaded;
            check(reloaded.load(b.path.string().c_str()), "reload B");
            prefill(reloaded, 7);
            prefill(first, 3);
        }
        // Both pending model resources and shutdown itself are safe to free twice.
        std::vector<int> order;
        ModelLoader alive;
        check(alive.load(a.path.string().c_str()), "load final A");
        prefill(alive, 3);
        alive.cache<CacheA>() = std::make_unique<CacheA>(order);
        alive.cache<CacheB>() = std::make_unique<CacheB>(order);
        shutdown_backend();
        shutdown_backend();
        check((order == std::vector<int>{2, 1}), "cache dependency destruction order");
        check(!alive.find_cache<CacheA>(), "shutdown retained model cache");
        check(alive.tensor("llm.embed.weight")->buffer == nullptr, "shutdown retained weight buffer");
        check(starling_ggml_load(STARLING_GGML_PARAKEET_TDT, a.path.string().c_str()) == nullptr, "load after terminal shutdown");
        check(std::string(starling_ggml_last_error(nullptr)).find("shut down") != std::string::npos, "shutdown error crossed C ABI");
        std::puts("MODEL LIFETIME OK: A/B/A, unload/reload, aliases, errors, ordered shutdown");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "MODEL LIFETIME FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
