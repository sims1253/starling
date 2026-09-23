#pragma once

#include "engine_gguf.hpp"
#include "engine_math.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace direct {

class Decoder {
public:
    Decoder(const Model& model,Kernels& kernels):model_(model),kernels_(kernels) {}
    std::vector<int32_t> decode(const Buffer& encoder,int frames);
    std::string text(const std::vector<int32_t>& ids) const;
private:
    struct State { std::vector<Buffer> h,c; };
    Buffer predict(int token,bool sos,const State& in,State& out);
    void joint(const float* enc,const Buffer& pred,int& token,int& duration);
    const Model& model_;
    Kernels& kernels_;
};

} // namespace direct
