#pragma once

#include "engine_gguf.hpp"
#include "independent_q4k.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace direct {

using Buffer = std::vector<float>; // time-major: [time, channels]

class Kernels {
public:
    explicit Kernels(int threads, bool pack_q4=false)
        :threads_(threads),pack_q4_(pack_q4) {}
    Buffer linear(const Tensor& weight, const Buffer& input, int batch,
                  const Tensor* bias=nullptr);
    int threads() const { return threads_; }
    const std::array<double,16>& seconds_by_type() const { return seconds_by_type_; }
private:
    int threads_;
    bool pack_q4_;
    std::unordered_map<const Tensor*, std::unique_ptr<independent_q4k::Matrix>> q4_;
    std::array<double,16> seconds_by_type_{};
};

Buffer norm(const Buffer& x,int batch,int dim,const Tensor& scale,const Tensor& bias);
void add_inplace(Buffer& a,const Buffer& b,float factor=1.0f);
void silu_inplace(Buffer& x,int threads);
void relu_inplace(Buffer& x);
float dot(const float* a,const float* b,int n);

} // namespace direct
