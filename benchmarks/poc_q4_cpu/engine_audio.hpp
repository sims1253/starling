#pragma once

#include "engine_gguf.hpp"
#include "engine_math.hpp"

#include <cstddef>
#include <vector>

namespace direct {

struct Mel {
    Buffer values; // feature-major [mels, frames]
    int frames=0;
    int valid=0;
};

// One previous window, keyed by absolute sample position and checked against
// the overlapping PCM bytes. Stores raw log-mel before window-wide CMVN.
struct MelCache {
    uint64_t first_sample=0;
    std::vector<float> pcm;
    Buffer raw;
    int frames=0;
    size_t reused_frames=0;
    void clear() { pcm.clear(); raw.clear(); frames=0; reused_frames=0; }
};

Mel compute_mel(const Model& model,const float* pcm,size_t samples,
                MelCache* cache=nullptr,uint64_t first_sample=0);
Buffer subsample(const Model& model,Kernels& kernels,const Mel& mel,
                 int& frames,int& valid);
std::vector<float> positions(int frames,int dim);

} // namespace direct
