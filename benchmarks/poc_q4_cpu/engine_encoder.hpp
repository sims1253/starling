#pragma once

#include "engine_gguf.hpp"
#include "engine_math.hpp"
#include "engine_audio.hpp"

#include <vector>
#include <array>

namespace direct {

class Encoder {
public:
    Encoder(const Model& model,Kernels& kernels):model_(model),kernels_(kernels) {}
    Buffer run(const float* pcm,size_t samples,int& frames);
    Buffer run_window(const float* pcm,size_t samples,uint64_t first_sample,int& frames);
    size_t last_reused_mel_frames() const { return mel_cache_.reused_frames; }
    const std::array<double,10>& stage_seconds() const { return stage_seconds_; }
    void reset_stream() { mel_cache_.clear(); }
private:
    Buffer layer(const Buffer& x,int t,int valid,int index,const Buffer& pos);
    Buffer attention(const Buffer& x,int t,int valid,int index,const std::string& prefix,const Buffer& pos);
    Buffer conv(const Buffer& x,int t,int valid,const std::string& prefix);
    const Model& model_;
    Kernels& kernels_;
    int cached_frames_ = -1;
    std::vector<Buffer> pos_cache_;
    MelCache mel_cache_;
    std::array<double,10> stage_seconds_{};
};

} // namespace direct
