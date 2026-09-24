// engine_select.hpp — which engine backs a model load.
//
// STARLING_ENGINE=auto (default in builds with the fast engines): use the
// model-specialized fast engine when it supports the model file and a Vulkan
// device is available, otherwise the general ggml engine.
// STARLING_ENGINE=fast: require the fast engine (the load fails otherwise).
// STARLING_ENGINE=ggml: always use the ggml engine.

#pragma once

#include <cstdlib>
#include <cstring>

namespace starling::fast {

enum class EngineChoice { Ggml, Fast, Auto };

inline EngineChoice engine_choice() {
    const char* v = std::getenv("STARLING_ENGINE");
    if (!v || !*v || std::strcmp(v, "auto") == 0) return EngineChoice::Auto;
    if (std::strcmp(v, "fast") == 0) return EngineChoice::Fast;
    return EngineChoice::Ggml;
}

} // namespace starling::fast
