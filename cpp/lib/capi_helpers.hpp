#pragma once

#include "runtime/backend.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace starling::ggml::lib {

template<class Model, class Tokenizer> struct EngineContext {
    std::unique_ptr<Model> model;
    Tokenizer tokenizer;
    std::string err;
};

// The error must outlive an unsuccessful load, whose context is destroyed.
// Each engine's instantiation keeps its own thread-local error storage.
template<class Context>
void* load_engine(const char* path, const char* label, const char** error) {
    static thread_local std::string load_error;
    try {
        if (!path || !*path)
            throw std::runtime_error(std::string("null or empty ") + label + " GGUF path");
        auto ctx = std::make_unique<Context>();
        using Model = typename decltype(ctx->model)::element_type;
        ctx->model = std::make_unique<Model>();
        if (!ctx->model->load(path, ctx->err) ||
            !ctx->tokenizer.load(ctx->model->loader, ctx->model->config, ctx->err))
            throw std::runtime_error(ctx->err);
        ensure_weights_realized(ctx->model->loader);
        if (error) *error = nullptr;
        return ctx.release();
    } catch (const std::exception& e) {
        load_error = e.what();
    } catch (...) {
        load_error = std::string("unknown exception loading ") + label + " model";
    }
    if (error) *error = load_error.c_str();
    return nullptr;
}

// String literals have static lifetime. Do not bind them to std::string,
// whose temporary buffer would die before the caller reads the error.
inline void report(const char** out, const char* message) {
    if (out) *out = message;
}
inline void report(const char** out, const std::string& message) {
    if (out) *out = message.c_str();
}

} // namespace starling::ggml::lib
