#include "independent_q4k.hpp"
#if defined(POC_REFERENCE_NUMERICS) && (!defined(__AVX2__) || !defined(__FMA__))
#error "POC_REFERENCE_NUMERICS requires AVX2 and FMA"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

namespace independent_q4k {
namespace {

float half_to_float(uint16_t bits) {
#if defined(__F16C__)
    return _cvtsh_ss(bits);
#else
    const bool negative = (bits & 0x8000u) != 0;
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    float value;
    if (exponent == 0) value = std::ldexp(float(fraction), -24);
    else if (exponent == 31) value = fraction ? NAN : INFINITY;
    else value = std::ldexp(float(1024 + fraction), exponent - 25);
    return negative ? -value : value;
#endif
}

uint16_t read_u16(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

void quantize(const float* input, ActivationBlock& out) {
    float largest = 0.0f;
    for (int i = 0; i < block_width; ++i)
        if (std::fabs(input[i]) > std::fabs(largest)) largest = input[i];
    if (largest == 0.0f) {
        out.scale = 0.0f;
        std::memset(out.values, 0, sizeof(out.values));
        std::memset(out.sums, 0, sizeof(out.sums));
        return;
    }
    const float inverse = -127.0f / largest;
    out.scale = 1.0f / inverse;
#if defined(__AVX2__)
    const __m256 gain=_mm256_set1_ps(inverse);
    alignas(32) int32_t rounded[8];
    for(int i=0;i<block_width;i+=8) {
        __m256 values=_mm256_mul_ps(_mm256_loadu_ps(input+i),gain);
        _mm256_store_si256(reinterpret_cast<__m256i*>(rounded),
                          _mm256_cvtps_epi32(values));
        for(int j=0;j<8;++j)
            out.values[i+j]=static_cast<int8_t>(std::min(127,rounded[j]));
    }
#else
    for (int i = 0; i < block_width; ++i) {
        const int rounded = static_cast<int>(std::nearbyint(input[i] * inverse));
        out.values[i] = static_cast<int8_t>(std::min(127, rounded));
    }
#endif
    for (int group = 0; group < 8; ++group) {
        int sum = 0;
        for (int i = 0; i < 32; ++i) sum += out.values[group * 32 + i];
        out.sums[group] = static_cast<int16_t>(sum);
    }
}

int dot_scaled_block(const PackedBlock& weight, const ActivationBlock& input) {
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    for (int group = 0; group < 8; ++group) {
        const __m256i a = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weight.values + group * 32));
        const __m256i b = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input.values + group * 32));
        const __m256i pairs = _mm256_maddubs_epi16(a, b);
        const __m256i scaled = _mm256_madd_epi16(
            pairs, _mm256_set1_epi16(weight.group_scale[group]));
        acc = _mm256_add_epi32(acc, scaled);
    }
    const __m128i lo = _mm256_castsi256_si128(acc);
    const __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    return _mm_cvtsi128_si32(sum);
#elif defined(__ARM_FEATURE_DOTPROD)
    int sum=0;
    for(int group=0;group<8;++group) {
        int32x4_t acc=vdupq_n_s32(0);
        for(int i=0;i<32;i+=16)
            acc=vdotq_s32(acc,
                vld1q_s8(reinterpret_cast<const int8_t*>(weight.values+group*32+i)),
                vld1q_s8(input.values+group*32+i));
        sum+=int(weight.group_scale[group])*vaddvq_s32(acc);
    }
    return sum;
#else
    int sum = 0;
    for (int group = 0; group < 8; ++group)
        for (int i = 0; i < 32; ++i)
            sum += int(weight.group_scale[group]) * int(weight.values[group * 32 + i]) *
                   int(input.values[group * 32 + i]);
    return sum;
#endif
}

float raw_dot_block(const uint8_t* raw, const RawMetadata& meta,
                    const ActivationBlock& input) {
    const uint8_t* quants = raw + 16;
    int min_dot = 0;
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
#else
    int scaled_dot = 0;
#endif
    for (int group = 0; group < 8; group += 2) {
        const int sl = meta.group_scale[group], sh = meta.group_scale[group + 1];
        min_dot += int(meta.group_min[group])*int(input.sums[group]);
        min_dot += int(meta.group_min[group+1])*int(input.sums[group+1]);
#if defined(__AVX2__)
        const __m256i packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(quants + group * 16));
        const __m256i mask = _mm256_set1_epi8(15);
        const __m256i low = _mm256_and_si256(packed, mask);
        const __m256i high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        const __m256i xl = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input.values + group * 32));
        const __m256i xh = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input.values + (group + 1) * 32));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
            _mm256_maddubs_epi16(low, xl), _mm256_set1_epi16(sl)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
            _mm256_maddubs_epi16(high, xh), _mm256_set1_epi16(sh)));
#elif defined(__ARM_FEATURE_DOTPROD)
        const uint8x16_t mask=vdupq_n_u8(15);
        int32x4_t low_acc=vdupq_n_s32(0),high_acc=vdupq_n_s32(0);
        for(int i=0;i<32;i+=16) {
            const uint8x16_t packed=vld1q_u8(quants+group*16+i);
            const int8x16_t low=vreinterpretq_s8_u8(vandq_u8(packed,mask));
            const int8x16_t high=vreinterpretq_s8_u8(vshrq_n_u8(packed,4));
            low_acc=vdotq_s32(low_acc,low,vld1q_s8(input.values+group*32+i));
            high_acc=vdotq_s32(high_acc,high,vld1q_s8(input.values+(group+1)*32+i));
        }
        scaled_dot+=sl*vaddvq_s32(low_acc)+sh*vaddvq_s32(high_acc);
#else
        for (int i = 0; i < 32; ++i) {
            const uint8_t q = quants[group * 16 + i];
            scaled_dot += sl * int(q & 15) * int(input.values[group * 32 + i]);
            scaled_dot += sh * int(q >> 4) * int(input.values[(group + 1) * 32 + i]);
        }
#endif
    }
#if defined(__AVX2__)
    const __m128i lo = _mm256_castsi256_si128(acc);
    const __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    const int scaled_dot = _mm_cvtsi128_si32(sum);
#endif
    return input.scale * (meta.scale * scaled_dot - meta.min_scale * min_dot);
}

#if defined(POC_REFERENCE_NUMERICS) && defined(__AVX2__) && defined(__FMA__)
float raw_dot_row_ref(const uint8_t* row,const RawMetadata* meta,
                      const ActivationBlock* input,int64_t blocks) {
    const __m256i mask=_mm256_set1_epi8(15);
    __m256 acc=_mm256_setzero_ps();
    __m128 min_acc=_mm_setzero_ps();
    for(int64_t block=0;block<blocks;++block) {
        const RawMetadata& m=meta[block];
        const ActivationBlock& x=input[block];
        const uint8_t* quants=row+block*weight_block_bytes+16;
        __m256i ints=_mm256_setzero_si256();
        int min_part[4]={};
        for(int group=0;group<8;group+=2) {
            const __m256i packed=_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(quants+group*16));
            const __m256i low=_mm256_and_si256(packed,mask);
            const __m256i high=_mm256_and_si256(_mm256_srli_epi16(packed,4),mask);
            const __m256i xl=_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x.values+group*32));
            const __m256i xh=_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x.values+(group+1)*32));
            const __m256i sl=_mm256_set1_epi16(m.group_scale[group]);
            const __m256i sh=_mm256_set1_epi16(m.group_scale[group+1]);
            const __m256i lo=_mm256_madd_epi16(_mm256_maddubs_epi16(low,xl),sl);
            const __m256i hi=_mm256_madd_epi16(_mm256_maddubs_epi16(high,xh),sh);
            ints=_mm256_add_epi32(ints,_mm256_add_epi32(lo,hi));
            min_part[group/2]=int(m.group_min[group])*int(x.sums[group])+
                              int(m.group_min[group+1])*int(x.sums[group+1]);
        }
        const float d=x.scale*m.scale;
        const float dmin=-x.scale*m.min_scale;
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(ints),acc);
        const __m128i mini=_mm_loadu_si128(reinterpret_cast<const __m128i*>(min_part));
        min_acc=_mm_fmadd_ps(_mm_set1_ps(dmin),_mm_cvtepi32_ps(mini),min_acc);
    }
    __m128 sum=_mm_add_ps(_mm256_extractf128_ps(acc,1),_mm256_castps256_ps128(acc));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
    min_acc=_mm_add_ps(min_acc,_mm_movehl_ps(min_acc,min_acc));
    min_acc=_mm_add_ss(min_acc,_mm_movehdup_ps(min_acc));
    return _mm_cvtss_f32(sum)+_mm_cvtss_f32(min_acc);
}
#endif

#if defined(__AVX2__)
void raw_dot_block4(const uint8_t* raw,const RawMetadata& meta,
                    const ActivationBlock& x0,const ActivationBlock& x1,
                    const ActivationBlock& x2,const ActivationBlock& x3,float* totals) {
    __m256i acc0=_mm256_setzero_si256(),acc1=_mm256_setzero_si256();
    __m256i acc2=_mm256_setzero_si256(),acc3=_mm256_setzero_si256();
    int min0=0,min1=0,min2=0,min3=0;
    const uint8_t* quants=raw+16;
    const __m256i mask=_mm256_set1_epi8(15);
    for(int group=0;group<8;group+=2) {
        const __m256i packed=_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(quants+group*16));
        const __m256i low=_mm256_and_si256(packed,mask);
        const __m256i high=_mm256_and_si256(_mm256_srli_epi16(packed,4),mask);
        const __m256i sl=_mm256_set1_epi16(meta.group_scale[group]);
        const __m256i sh=_mm256_set1_epi16(meta.group_scale[group+1]);
#define APPLY_LANE(N) do { \
            min##N+=int(meta.group_min[group])*int(x##N.sums[group]); \
            min##N+=int(meta.group_min[group+1])*int(x##N.sums[group+1]); \
            const __m256i xl=_mm256_loadu_si256( \
                reinterpret_cast<const __m256i*>(x##N.values+group*32)); \
            const __m256i xh=_mm256_loadu_si256( \
                reinterpret_cast<const __m256i*>(x##N.values+(group+1)*32)); \
            acc##N=_mm256_add_epi32(acc##N,_mm256_madd_epi16( \
                _mm256_maddubs_epi16(low,xl),sl)); \
            acc##N=_mm256_add_epi32(acc##N,_mm256_madd_epi16( \
                _mm256_maddubs_epi16(high,xh),sh)); \
        } while(0)
        APPLY_LANE(0);APPLY_LANE(1);APPLY_LANE(2);APPLY_LANE(3);
#undef APPLY_LANE
    }
    auto finish=[&](const __m256i& acc,int min_dot,const ActivationBlock& x) {
        __m128i sum=_mm_add_epi32(_mm256_castsi256_si128(acc),
                                  _mm256_extracti128_si256(acc,1));
        sum=_mm_add_epi32(sum,_mm_srli_si128(sum,8));
        sum=_mm_add_epi32(sum,_mm_srli_si128(sum,4));
        const int scaled_dot=_mm_cvtsi128_si32(sum);
        return x.scale*(meta.scale*scaled_dot-meta.min_scale*min_dot);
    };
    totals[0]+=finish(acc0,min0,x0);
    totals[1]+=finish(acc1,min1,x1);
    totals[2]+=finish(acc2,min2,x2);
    totals[3]+=finish(acc3,min3,x3);
}

#endif

} // namespace

Matrix::Matrix(const void* bytes, size_t size, int64_t cols, int64_t rows, bool prepack)
    : cols_(cols), rows_(rows), blocks_per_row_(cols / block_width),
      raw_(static_cast<const uint8_t*>(bytes)) {
    if (!bytes || cols <= 0 || rows <= 0 || cols % block_width ||
        size != size_t(rows * blocks_per_row_) * weight_block_bytes)
        throw std::invalid_argument("invalid Q4_K matrix shape or byte count");
    if (prepack) blocks_.resize(size_t(rows * blocks_per_row_));
    else metadata_.resize(size_t(rows * blocks_per_row_));
    const auto* source = static_cast<const uint8_t*>(bytes);
    const size_t nblocks = size_t(rows * blocks_per_row_);
    for (size_t block = 0; block < nblocks; ++block) {
        const uint8_t* raw = source + block * weight_block_bytes;
        RawMetadata decoded{};
        decoded.scale = half_to_float(read_u16(raw));
        decoded.min_scale = half_to_float(read_u16(raw + 2));
        const uint8_t* scales = raw + 4;
        const uint8_t* quants = raw + 16;
        for (int group = 0; group < 8; ++group) {
            if (group < 4) {
                decoded.group_scale[group] = scales[group] & 63;
                decoded.group_min[group] = scales[group + 4] & 63;
            } else {
                decoded.group_scale[group] = (scales[group + 4] & 15) |
                                            ((scales[group - 4] >> 6) << 4);
                decoded.group_min[group] = (scales[group + 4] >> 4) |
                                          ((scales[group] >> 6) << 4);
            }
        }
        if (!prepack) {
            metadata_[block] = decoded;
            continue;
        }
        PackedBlock& packed = blocks_[block];
        packed.scale = decoded.scale;
        packed.min_scale = decoded.min_scale;
        std::memcpy(packed.group_scale, decoded.group_scale, 8);
        std::memcpy(packed.group_min, decoded.group_min, 8);
        for (int group = 0; group < 8; group += 2) {
            const uint8_t* source_group = quants + group * 16;
            for (int i = 0; i < 32; ++i) {
                packed.values[group * 32 + i] = source_group[i] & 15;
                packed.values[(group + 1) * 32 + i] = source_group[i] >> 4;
            }
        }
    }
}

void Matrix::run(const float* input, int batch, float* output, int threads,
                 bool raw) {
    if (!input || !output || batch <= 0 || threads <= 0)
        throw std::invalid_argument("invalid Q4_K matmul input");
    if ((raw && metadata_.empty()) || (!raw && blocks_.empty()))
        throw std::invalid_argument("Q4_K matrix was not prepared for the selected kernel");
    activations_.resize(size_t(batch * blocks_per_row_));
    for (int m = 0; m < batch; ++m)
        for (int64_t block = 0; block < blocks_per_row_; ++block)
            quantize(input + size_t(m) * cols_ + block * block_width,
                     activations_[size_t(m * blocks_per_row_ + block)]);

    if (raw) {
#if defined(POC_REFERENCE_NUMERICS) && defined(__AVX2__) && defined(__FMA__)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
        for(int64_t row=0;row<rows_;++row) {
            for(int m=0;m<batch;++m)
                output[size_t(m)*rows_+row]=raw_dot_row_ref(
                    raw_+size_t(row*blocks_per_row_)*weight_block_bytes,
                    metadata_.data()+size_t(row*blocks_per_row_),
                    activations_.data()+size_t(m*blocks_per_row_),blocks_per_row_);
        }
        return;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
        for (int64_t row = 0; row < rows_; ++row) {
            int m=0;
#if defined(__AVX2__)
            for(;m+4<=batch;m+=4) {
                float results[4]={0,0,0,0};
                for(int64_t block=0;block<blocks_per_row_;++block) {
                    const size_t index=size_t(row*blocks_per_row_+block);
                    raw_dot_block4(raw_+index*weight_block_bytes,metadata_[index],
                        activations_[size_t((m+0)*blocks_per_row_+block)],
                        activations_[size_t((m+1)*blocks_per_row_+block)],
                        activations_[size_t((m+2)*blocks_per_row_+block)],
                        activations_[size_t((m+3)*blocks_per_row_+block)],results);
                }
                for(int lane=0;lane<4;++lane)
                    output[size_t(m+lane)*rows_+row]=results[lane];
            }
#endif
            for (;m<batch;++m) {
                float result = 0.0f;
                for (int64_t block = 0; block < blocks_per_row_; ++block) {
                    const size_t index = size_t(row * blocks_per_row_ + block);
                    const ActivationBlock& x = activations_[size_t(m * blocks_per_row_ + block)];
                    result += raw_dot_block(raw_ + index * weight_block_bytes,
                                            metadata_[index], x);
                }
                output[size_t(m) * rows_ + row] = result;
            }
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (int64_t row = 0; row < rows_; ++row) {
        for (int m = 0; m < batch; ++m) {
            float result = 0.0f;
            for (int64_t block = 0; block < blocks_per_row_; ++block) {
                const PackedBlock& w = blocks_[size_t(row * blocks_per_row_ + block)];
                const ActivationBlock& x = activations_[size_t(m * blocks_per_row_ + block)];
                const int32_t scaled_dot = dot_scaled_block(w, x);
                int32_t min_dot=0;
                for(int group=0;group<8;++group)
                    min_dot+=int(w.group_min[group])*int(x.sums[group]);
                result += x.scale * (w.scale * scaled_dot - w.min_scale * min_dot);
            }
            output[size_t(m) * rows_ + row] = result;
        }
    }
}

} // namespace independent_q4k
