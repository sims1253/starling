#include "engine_math.hpp"
#if defined(POC_REFERENCE_NUMERICS) && (!defined(__AVX2__) || !defined(__FMA__))
#error "POC_REFERENCE_NUMERICS requires AVX2 and FMA"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace direct {
namespace {
#if defined(__AVX2__) && defined(__FMA__) && defined(POC_REFERENCE_NUMERICS)
// Same F32 exp polynomial as the CPU reference's AVX2 SiLU. The quantized
// projection after SiLU can change integer codes after a one-ulp drift.
__m256 exp_compatible(__m256 x) {
    const __m256 r=_mm256_set1_ps(0x1.8p23f);
    const __m256 z=_mm256_fmadd_ps(x,_mm256_set1_ps(0x1.715476p+0f),r);
    const __m256 n=_mm256_sub_ps(z,r);
    const __m256 b=_mm256_fnmadd_ps(n,_mm256_set1_ps(0x1.7f7d1cp-20f),
                    _mm256_fnmadd_ps(n,_mm256_set1_ps(0x1.62e4p-1f),x));
    const __m256i e=_mm256_slli_epi32(_mm256_castps_si256(z),23);
    const __m256 k=_mm256_castsi256_ps(
        _mm256_add_epi32(e,_mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c=_mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.f),n),_mm256_set1_ps(126),_CMP_GT_OQ));
    const __m256 u=_mm256_mul_ps(b,b);
    const __m256 j=_mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f),b,
                             _mm256_set1_ps(0x1.573e2ep-5f)),u,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f),b,
                             _mm256_set1_ps(0x1.fffdb6p-2f))),
        u,_mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f),b));
    if(!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
        return _mm256_fmadd_ps(j,k,k);
    const __m256i g=_mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n,_mm256_setzero_ps(),_CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u));
    const __m256 s1=_mm256_castsi256_ps(_mm256_add_epi32(g,_mm256_set1_epi32(0x7f000000u)));
    const __m256 s2=_mm256_castsi256_ps(_mm256_sub_epi32(e,g));
    const __m256i d=_mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.f),n),_mm256_set1_ps(192),_CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d),_mm256_mul_ps(s1,s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2,j,s2),s1)),
                _mm256_andnot_ps(_mm256_castsi256_ps(c),_mm256_fmadd_ps(k,j,k)))));
}
#endif
struct Q8Block { float scale; int8_t q[256]; int16_t sums[16]; };
void quantize_q8(const float* x,int n,Q8Block& b) {
    float largest=0;
    for(int i=0;i<n;++i) if(std::fabs(x[i])>std::fabs(largest)) largest=x[i];
    if(largest==0) { b.scale=0; std::memset(b.q,0,size_t(n)); std::memset(b.sums,0,sizeof(b.sums)); return; }
    const float inverse=-127.0f/largest;
    b.scale=1.0f/inverse;
#if defined(__AVX2__)
    const __m256 gain=_mm256_set1_ps(inverse);
    alignas(32) int32_t rounded[8];
    for(int i=0;i<n;i+=8) {
        __m256 values=_mm256_mul_ps(_mm256_loadu_ps(x+i),gain);
        _mm256_store_si256(reinterpret_cast<__m256i*>(rounded),
                          _mm256_cvtps_epi32(values));
        for(int j=0;j<8;++j) b.q[i+j]=int8_t(std::min(127,rounded[j]));
    }
#else
    for(int i=0;i<n;++i) b.q[i]=int8_t(std::min(127,int(std::nearbyint(x[i]*inverse))));
#endif
    for(int g=0;g<n/16;++g) {
        int sum=0; for(int i=0;i<16;++i) sum+=b.q[g*16+i];
        b.sums[g]=int16_t(sum);
    }
}
int signed_dot_16(const int8_t* a,const int8_t* b) {
#if defined(__AVX2__)
    __m256i av=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
    __m256i bv=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
    __m256i prod=_mm256_madd_epi16(av,bv);
    __m128i lo=_mm256_castsi256_si128(prod),hi=_mm256_extracti128_si256(prod,1);
    __m128i sum=_mm_add_epi32(lo,hi);
    sum=_mm_add_epi32(sum,_mm_srli_si128(sum,8));
    sum=_mm_add_epi32(sum,_mm_srli_si128(sum,4));
    return _mm_cvtsi128_si32(sum);
#elif defined(__ARM_FEATURE_DOTPROD)
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0),vld1q_s8(a),vld1q_s8(b)));
#else
    int s=0; for(int i=0;i<16;++i) s+=int(a[i])*int(b[i]); return s;
#endif
}
void decode_q6(const uint8_t* p,uint8_t* out) {
    const uint8_t *ql=p,*qh=p+128;
    for(int n=0;n<256;n+=128) for(int l=0;l<32;++l) {
        int off=n/128*64,hi=n/128*32;
        out[n+l]    =uint8_t((ql[off+l]&15)|(((qh[hi+l]>>0)&3)<<4));
        out[n+l+32] =uint8_t((ql[off+l+32]&15)|(((qh[hi+l]>>2)&3)<<4));
        out[n+l+64] =uint8_t((ql[off+l]>>4)|(((qh[hi+l]>>4)&3)<<4));
        out[n+l+96] =uint8_t((ql[off+l+32]>>4)|(((qh[hi+l]>>6)&3)<<4));
    }
}
[[maybe_unused]] int q6_dot(const uint8_t* weights,const Q8Block& a
#if defined(__AVX2__)
           ,const __m256i* scale_pairs,const __m256i& scales16
#else
           ,const int8_t* scales
#endif
           ) {
#if defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256();
    for(int g=0;g<16;g+=2) {
        __m256i w=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights+g*16));
        __m256i x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q+g*16));
        __m256i pairs=_mm256_maddubs_epi16(w,x);
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(pairs,scale_pairs[g/2]));
    }
    __m256i sums=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.sums));
    __m256i correction=_mm256_madd_epi16(sums,scales16);
    acc=_mm256_sub_epi32(acc,_mm256_slli_epi32(correction,5));
    __m128i sum=_mm_add_epi32(_mm256_castsi256_si128(acc),_mm256_extracti128_si256(acc,1));
    sum=_mm_add_epi32(sum,_mm_srli_si128(sum,8));
    sum=_mm_add_epi32(sum,_mm_srli_si128(sum,4));
    return _mm_cvtsi128_si32(sum);
#elif defined(__ARM_FEATURE_DOTPROD)
    int total=0;
    for(int g=0;g<16;++g) {
        const int dot=vaddvq_s32(vdotq_s32(vdupq_n_s32(0),
            vld1q_s8(reinterpret_cast<const int8_t*>(weights+g*16)),
            vld1q_s8(a.q+g*16)));
        total+=int(scales[g])*(dot-32*int(a.sums[g]));
    }
    return total;
#else
    int total=0,correction=0;
    for(int g=0;g<16;++g) correction+=int(scales[g])*int(a.sums[g]);
    for(int g=0;g<16;++g) for(int i=0;i<16;++i)
        total+=int(scales[g])*int(weights[g*16+i])*int(a.q[g*16+i]);
    return total-32*correction;
#endif
}
#if defined(POC_REFERENCE_NUMERICS) && defined(__AVX2__) && defined(__FMA__)
__m256i q6_dot_lanes(const uint8_t* weights,const Q8Block& a,
                     const __m256i* scale_pairs,const __m256i& scales16) {
    __m256i acc=_mm256_setzero_si256();
    for(int g=0;g<16;g+=2) {
        const __m256i w=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights+g*16));
        const __m256i x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q+g*16));
        const __m256i pairs=_mm256_maddubs_epi16(w,x);
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(pairs,scale_pairs[g/2]));
    }
    const __m256i sums=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.sums));
    const __m256i correction=_mm256_madd_epi16(sums,scales16);
    return _mm256_sub_epi32(acc,_mm256_slli_epi32(correction,5));
}
#endif
void q6_linear(const Tensor& w,const Buffer& x,int batch,Buffer& y,int threads) {
    const int cols=int(w.cols()),rows=int(w.rows()),blocks=cols/256;
    std::vector<Q8Block> acts(size_t(batch)*blocks);
    for(int b=0;b<blocks;++b) for(int m=0;m<batch;++m)
        quantize_q8(x.data()+size_t(m)*cols+b*256,256,acts[size_t(b)*batch+m]);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for(int row=0;row<rows;++row) {
#if defined(POC_REFERENCE_NUMERICS) && defined(__AVX2__) && defined(__FMA__)
        std::vector<float> acc_lanes(size_t(batch)*8,0.0f);
#endif
        for(int block=0;block<blocks;++block) {
            const uint8_t* p=w.data+(size_t(row)*blocks+block)*210;
            alignas(32) uint8_t weights[256]; decode_q6(p,weights);
            const int8_t* scales=reinterpret_cast<const int8_t*>(p+192);
            const float d=half(uint16_t(p[208])|(uint16_t(p[209])<<8));
#if defined(__AVX2__)
            __m256i scale_pairs[8];
            for(int g=0;g<16;g+=2)
                scale_pairs[g/2]=_mm256_set_m128i(
                    _mm_set1_epi16(scales[g+1]),_mm_set1_epi16(scales[g]));
            __m256i scales16=_mm256_cvtepi8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(scales)));
#endif
            for(int m=0;m<batch;++m) {
                const auto& a=acts[size_t(block)*batch+m];
#if defined(POC_REFERENCE_NUMERICS) && defined(__AVX2__) && defined(__FMA__)
                const __m256i total=q6_dot_lanes(weights,a,scale_pairs,scales16);
                float* lane=acc_lanes.data()+size_t(m)*8;
                const __m256 prior=_mm256_loadu_ps(lane);
                const __m256 next=_mm256_fmadd_ps(_mm256_set1_ps(a.scale*d),
                                                   _mm256_cvtepi32_ps(total),prior);
                _mm256_storeu_ps(lane,next);
#else
                int total=q6_dot(weights,a
#if defined(__AVX2__)
                                 ,scale_pairs,scales16
#else
                                 ,scales
#endif
                                 );
                y[size_t(m)*rows+row]+=(d*a.scale)*total;
#endif
            }
        }
#if defined(POC_REFERENCE_NUMERICS) && defined(__AVX2__) && defined(__FMA__)
        for(int m=0;m<batch;++m) {
            const __m256 v=_mm256_loadu_ps(acc_lanes.data()+size_t(m)*8);
            __m128 sum=_mm_add_ps(_mm256_extractf128_ps(v,1),_mm256_castps256_ps128(v));
            sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
            sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
            y[size_t(m)*rows+row]=_mm_cvtss_f32(sum);
        }
#endif
    }
}
void q8_linear(const Tensor& w,const Buffer& x,int batch,Buffer& y,int threads) {
    const int cols=int(w.cols()),rows=int(w.rows()),blocks=cols/32;
    std::vector<Q8Block> acts(size_t(batch)*blocks);
    for(int m=0;m<batch;++m) for(int b=0;b<blocks;++b)
        quantize_q8(x.data()+size_t(m)*cols+b*32,32,acts[size_t(m)*blocks+b]);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for(int row=0;row<rows;++row) for(int m=0;m<batch;++m) {
        float total=0;
        for(int block=0;block<blocks;++block) {
            const uint8_t* p=w.data+(size_t(row)*blocks+block)*34;
            float d=half(uint16_t(p[0])|(uint16_t(p[1])<<8));
            const auto& a=acts[size_t(m)*blocks+block];
            const auto* q=reinterpret_cast<const int8_t*>(p+2);
            int integer=signed_dot_16(q,a.q)+signed_dot_16(q+16,a.q+16);
            total+=(d*a.scale)*integer;
        }
        y[size_t(m)*rows+row]=total;
    }
}
} // namespace


float dot(const float* a,const float* b,int n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
    int i=0;
    for(;i+16<=n;i+=16) {
        acc0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),acc0);
        acc1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),acc1);
    }
    alignas(32) float v[8];
    _mm256_store_ps(v,_mm256_add_ps(acc0,acc1));
    float s=v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
    for(;i<n;++i) s+=a[i]*b[i];
    return s;
#elif defined(__aarch64__)
    float32x4_t acc0=vdupq_n_f32(0),acc1=vdupq_n_f32(0);
    int i=0;
    for(;i+8<=n;i+=8) {
        acc0=vfmaq_f32(acc0,vld1q_f32(a+i),vld1q_f32(b+i));
        acc1=vfmaq_f32(acc1,vld1q_f32(a+i+4),vld1q_f32(b+i+4));
    }
    float s=vaddvq_f32(vaddq_f32(acc0,acc1));
    for(;i<n;++i) s+=a[i]*b[i];
    return s;
#else
    float s=0; for(int i=0;i<n;++i) s+=a[i]*b[i]; return s;
#endif
}

void dot2x2(const float* w0,const float* w1,const float* x0,const float* x1,
            int n,float* out) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a00=_mm256_setzero_ps(),a01=_mm256_setzero_ps();
    __m256 a10=_mm256_setzero_ps(),a11=_mm256_setzero_ps();
    __m256 b00=_mm256_setzero_ps(),b01=_mm256_setzero_ps();
    __m256 b10=_mm256_setzero_ps(),b11=_mm256_setzero_ps();
    int i=0;
    for(;i+16<=n;i+=16) {
        const __m256 wa=_mm256_loadu_ps(w0+i),wb=_mm256_loadu_ps(w0+i+8);
        const __m256 wc=_mm256_loadu_ps(w1+i),wd=_mm256_loadu_ps(w1+i+8);
        const __m256 xa=_mm256_loadu_ps(x0+i),xb=_mm256_loadu_ps(x0+i+8);
        const __m256 xc=_mm256_loadu_ps(x1+i),xd=_mm256_loadu_ps(x1+i+8);
        a00=_mm256_fmadd_ps(wa,xa,a00);a01=_mm256_fmadd_ps(wb,xb,a01);
        a10=_mm256_fmadd_ps(wa,xc,a10);a11=_mm256_fmadd_ps(wb,xd,a11);
        b00=_mm256_fmadd_ps(wc,xa,b00);b01=_mm256_fmadd_ps(wd,xb,b01);
        b10=_mm256_fmadd_ps(wc,xc,b10);b11=_mm256_fmadd_ps(wd,xd,b11);
    }
    auto reduce=[](__m256 a,__m256 b) {
        alignas(32) float v[8];_mm256_store_ps(v,_mm256_add_ps(a,b));
        return v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
    };
    out[0]=reduce(a00,a01);out[1]=reduce(a10,a11);
    out[2]=reduce(b00,b01);out[3]=reduce(b10,b11);
    for(;i<n;++i) {
        out[0]+=w0[i]*x0[i];out[1]+=w0[i]*x1[i];
        out[2]+=w1[i]*x0[i];out[3]+=w1[i]*x1[i];
    }
#elif defined(__aarch64__)
    float32x4_t a00=vdupq_n_f32(0),a01=vdupq_n_f32(0);
    float32x4_t a10=vdupq_n_f32(0),a11=vdupq_n_f32(0);
    float32x4_t b00=vdupq_n_f32(0),b01=vdupq_n_f32(0);
    float32x4_t b10=vdupq_n_f32(0),b11=vdupq_n_f32(0);
    int i=0;
    for(;i+8<=n;i+=8) {
        const float32x4_t wa=vld1q_f32(w0+i),wb=vld1q_f32(w0+i+4);
        const float32x4_t wc=vld1q_f32(w1+i),wd=vld1q_f32(w1+i+4);
        const float32x4_t xa=vld1q_f32(x0+i),xb=vld1q_f32(x0+i+4);
        const float32x4_t xc=vld1q_f32(x1+i),xd=vld1q_f32(x1+i+4);
        a00=vfmaq_f32(a00,wa,xa);a01=vfmaq_f32(a01,wb,xb);
        a10=vfmaq_f32(a10,wa,xc);a11=vfmaq_f32(a11,wb,xd);
        b00=vfmaq_f32(b00,wc,xa);b01=vfmaq_f32(b01,wd,xb);
        b10=vfmaq_f32(b10,wc,xc);b11=vfmaq_f32(b11,wd,xd);
    }
    out[0]=vaddvq_f32(vaddq_f32(a00,a01));
    out[1]=vaddvq_f32(vaddq_f32(a10,a11));
    out[2]=vaddvq_f32(vaddq_f32(b00,b01));
    out[3]=vaddvq_f32(vaddq_f32(b10,b11));
    for(;i<n;++i) {
        out[0]+=w0[i]*x0[i];out[1]+=w0[i]*x1[i];
        out[2]+=w1[i]*x0[i];out[3]+=w1[i]*x1[i];
    }
#else
    out[0]=dot(w0,x0,n);out[1]=dot(w0,x1,n);
    out[2]=dot(w1,x0,n);out[3]=dot(w1,x1,n);
#endif
}

Buffer Kernels::linear(const Tensor& w,const Buffer& x,int batch,const Tensor* bias) {
    const auto begin=std::chrono::steady_clock::now();
    const int cols=int(w.cols()), rows=int(w.rows());
    if(batch<1 || x.size()!=size_t(batch)*cols) throw std::runtime_error("linear shape mismatch");
    Buffer y(size_t(batch)*rows);
    if(w.type==12) {
        auto& m=q4_[&w];
        if(!m) m=std::make_unique<independent_q4k::Matrix>(
            w.data,size_t(rows)*size_t(cols/256)*144,cols,rows,pack_q4_);
        m->run(x.data(),batch,y.data(),threads_,!pack_q4_);
    } else if(w.type==14) {
        q6_linear(w,x,batch,y,threads_);
    } else if(w.type==8) {
        q8_linear(w,x,batch,y,threads_);
    } else if(w.type==0 && rows>=2 && batch>=2) {
        const float* weights=reinterpret_cast<const float*>(w.data);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
        for(int pair=0;pair<rows/2;++pair) {
            const int r=2*pair;
            const float* w0=weights+size_t(r)*cols,*w1=w0+cols;
            int b=0;
            for(;b+2<=batch;b+=2) {
                float values[4];
                dot2x2(w0,w1,x.data()+size_t(b)*cols,
                       x.data()+size_t(b+1)*cols,cols,values);
                y[size_t(b)*rows+r]=values[0];
                y[size_t(b+1)*rows+r]=values[1];
                y[size_t(b)*rows+r+1]=values[2];
                y[size_t(b+1)*rows+r+1]=values[3];
            }
            if(b<batch) {
                const float* a=x.data()+size_t(b)*cols;
                y[size_t(b)*rows+r]=dot(w0,a,cols);
                y[size_t(b)*rows+r+1]=dot(w1,a,cols);
            }
        }
        if(rows&1) {
            const int r=rows-1;const float* wr=weights+size_t(r)*cols;
            for(int b=0;b<batch;++b)
                y[size_t(b)*rows+r]=dot(wr,x.data()+size_t(b)*cols,cols);
        }
    } else if(w.type==0 || w.type==1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
        {
            Buffer unpacked;
            if(w.type!=0) unpacked.resize(cols);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for(int r=0;r<rows;++r) {
                const float* wr;
                if(w.type==0) wr=reinterpret_cast<const float*>(w.data)+size_t(r)*cols;
                else { dequant_row(w,r,unpacked.data()); wr=unpacked.data(); }
                for(int b=0;b<batch;++b)
                    y[size_t(b)*rows+r]=dot(wr,x.data()+size_t(b)*cols,cols);
            }
        }
    } else throw std::runtime_error("unsupported linear weight type");
    if(bias) {
        if(bias->cols()!=rows || bias->type!=0) throw std::runtime_error("linear bias mismatch");
        const float* bp=reinterpret_cast<const float*>(bias->data);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) if(batch*rows>32768)
#endif
        for(size_t i=0;i<y.size();++i) y[i]+=bp[i%rows];
    }
    if(w.type<seconds_by_type_.size())
        seconds_by_type_[w.type]+=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-begin).count();
    return y;
}


Buffer norm(const Buffer& x,int batch,int dim,const Tensor& scale,const Tensor& bias) {
    if(x.size()!=size_t(batch)*dim || scale.cols()!=dim || bias.cols()!=dim)
        throw std::runtime_error("norm shape mismatch");
    Buffer y(x.size());
    const float* g=reinterpret_cast<const float*>(scale.data);
    const float* b=reinterpret_cast<const float*>(bias.data);
    for(int t=0;t<batch;++t) {
        const float* row=x.data()+size_t(t)*dim;
#if defined(POC_REFERENCE_NUMERICS) && defined(__AVX2__)
        {
        double sum=0;
        for(int d=0;d<dim;++d) sum+=double(row[d]);
        const float mean=float(sum)/dim;
        double variance=0;
        int d=0;
        float* out=y.data()+size_t(t)*dim;
        for(;d+8<=dim;d+=8) {
            const __m256 centered=_mm256_sub_ps(_mm256_loadu_ps(row+d),_mm256_set1_ps(mean));
            _mm256_storeu_ps(out+d,centered);
            const __m256 sq=_mm256_mul_ps(centered,centered);
            __m128 reduced=_mm_add_ps(_mm256_extractf128_ps(sq,1),
                                       _mm256_castps256_ps128(sq));
            reduced=_mm_add_ps(reduced,_mm_movehl_ps(reduced,reduced));
            reduced=_mm_add_ss(reduced,_mm_movehdup_ps(reduced));
            variance+=double(_mm_cvtss_f32(reduced));
        }
        for(;d<dim;++d) {
            const float v=row[d]-mean;out[d]=v;variance+=double(v*v);
        }
        const float inv=1.0f/std::sqrt(float(variance/dim)+1e-5f);
        for(int i=0;i<dim;++i) out[i]*=inv;
        for(int i=0;i<dim;++i) out[i]*=g[i];
        for(int i=0;i<dim;++i) out[i]+=b[i];
        continue;
        }
#endif
        double mean=0; for(int d=0;d<dim;++d) mean+=row[d]; mean/=dim;
        double var=0; for(int d=0;d<dim;++d) { double v=row[d]-mean; var+=v*v; } var/=dim;
        float inv=1.0f/std::sqrt(float(var)+1e-5f);
        float* out=y.data()+size_t(t)*dim;
        for(int d=0;d<dim;++d) out[d]=(row[d]-float(mean))*inv*g[d]+b[d];
    }
    return y;
}
void add_inplace(Buffer& a,const Buffer& b,float f) {
    if(a.size()!=b.size()) throw std::runtime_error("add shape mismatch");
    for(size_t i=0;i<a.size();++i) a[i]+=f*b[i];
}
void silu_inplace(Buffer& x,int threads) {
#if defined(__AVX2__) && defined(__FMA__) && defined(POC_REFERENCE_NUMERICS)
    const size_t blocks=x.size()/8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads>1 && blocks>4096)
#endif
    for(size_t i=0;i<blocks;++i) {
        __m256 v=_mm256_loadu_ps(x.data()+8*i);
        __m256 neg=_mm256_sub_ps(_mm256_setzero_ps(),v);
        __m256 denom=_mm256_add_ps(_mm256_set1_ps(1.0f),exp_compatible(neg));
        _mm256_storeu_ps(x.data()+8*i,_mm256_div_ps(v,denom));
    }
    for(size_t i=8*blocks;i<x.size();++i) x[i]=x[i]/(1.0f+std::exp(-x[i]));
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads>1 && x.size()>32768)
#endif
    for(size_t i=0;i<x.size();++i) x[i]=x[i]/(1.0f+std::exp(-x[i]));
#endif
}
void relu_inplace(Buffer& x) { for(float& v:x) v=std::max(0.0f,v); }

} // namespace direct
