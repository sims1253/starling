#include "engine_encoder.hpp"
#include "engine_audio.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace direct {
namespace {
float sigmoid(float x) { return 1.0f/(1.0f+std::exp(-x)); }
} // namespace

Buffer Encoder::attention(const Buffer& x,int T,int valid,int index,const std::string& p,const Buffer& pe) {
    const int D=int(model_.integer("parakeet.encoder.d_model",1024));
    const int H=int(model_.integer("parakeet.encoder.n_heads",8));
    const int dk=D/H,P=2*T-1;
    auto project=[&](const std::string& n,const Buffer& a,int batch) {
        return kernels_.linear(model_.at(p+n+".weight"),a,batch,model_.find(p+n+".bias"));
    };
    const Buffer q=project("linear_q",x,T);
    const Buffer k=project("linear_k",x,T);
    const Buffer v=project("linear_v",x,T);
    Buffer ph_local;
    const Buffer* ph_ptr=nullptr;
    if(index<int(pos_cache_.size()) && !pos_cache_[index].empty()) ph_ptr=&pos_cache_[index];
    else {
        ph_local=project("linear_pos",pe,P);
        if(index<int(pos_cache_.size())) {
            pos_cache_[index]=std::move(ph_local);
            ph_ptr=&pos_cache_[index];
        } else ph_ptr=&ph_local;
    }
    const Buffer& ph=*ph_ptr;
    const float* bu=reinterpret_cast<const float*>(model_.at(p+"pos_bias_u").data);
    const float* bv=reinterpret_cast<const float*>(model_.at(p+"pos_bias_v").data);
    Buffer context(size_t(T)*D,0.0f);
    const float inv=1.0f/std::sqrt(float(dk));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(kernels_.threads())
#endif
    for(int h=0;h<H;++h) {
        Buffer qu(dk),qv(dk),scores(T);
        for(int qi=0;qi<T;++qi) {
            if(qi>=valid) continue;
            const float* qr=q.data()+size_t(qi)*D+h*dk;
            for(int d=0;d<dk;++d) { qu[d]=qr[d]+bu[h*dk+d]; qv[d]=qr[d]+bv[h*dk+d]; }
            float mx=-INFINITY;
            for(int kj=0;kj<valid;++kj) {
                const int rel=T-1+kj-qi;
                float s=(dot(qu.data(),k.data()+size_t(kj)*D+h*dk,dk)
                       +dot(qv.data(),ph.data()+size_t(rel)*D+h*dk,dk))*inv;
                scores[kj]=s; mx=std::max(mx,s);
            }
            float denom=0;
            for(int kj=0;kj<valid;++kj) { scores[kj]=std::exp(scores[kj]-mx); denom+=scores[kj]; }
            for(int kj=0;kj<valid;++kj) scores[kj]/=denom;
            float* dst=context.data()+size_t(qi)*D+h*dk;
#if defined(__AVX2__) && defined(__FMA__)
            int d=0;
            for(;d+32<=dk;d+=32) {
                __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
                for(int kj=0;kj<valid;++kj) {
                    const float* vr=v.data()+size_t(kj)*D+h*dk+d;
                    __m256 scale=_mm256_set1_ps(scores[kj]);
                    a0=_mm256_fmadd_ps(scale,_mm256_loadu_ps(vr),a0);
                    a1=_mm256_fmadd_ps(scale,_mm256_loadu_ps(vr+8),a1);
                    a2=_mm256_fmadd_ps(scale,_mm256_loadu_ps(vr+16),a2);
                    a3=_mm256_fmadd_ps(scale,_mm256_loadu_ps(vr+24),a3);
                }
                _mm256_storeu_ps(dst+d,a0);_mm256_storeu_ps(dst+d+8,a1);
                _mm256_storeu_ps(dst+d+16,a2);_mm256_storeu_ps(dst+d+24,a3);
            }
            for(;d<dk;++d) for(int kj=0;kj<valid;++kj)
                dst[d]+=scores[kj]*v[size_t(kj)*D+h*dk+d];
#else
            for(int kj=0;kj<valid;++kj) {
                float a=scores[kj];
                const float* vr=v.data()+size_t(kj)*D+h*dk;
                for(int d=0;d<dk;++d) dst[d]+=a*vr[d];
            }
#endif
        }
    }
    return project("linear_out",context,T);
}

Buffer Encoder::conv(const Buffer& x,int T,int valid,const std::string& p) {
    const int D=int(model_.integer("parakeet.encoder.d_model",1024));
    const int K=int(model_.integer("parakeet.encoder.conv_kernel",9));
    Tensor pw1w=model_.at(p+"pointwise_conv1.weight");
    pw1w.rank=2; pw1w.shape={D,2*D,1,1};
    Buffer pw1=kernels_.linear(pw1w,x,T,
                               model_.find(p+"pointwise_conv1.bias"));
    Buffer gate(size_t(T)*D);
    for(int t=0;t<T;++t) for(int c=0;c<D;++c)
        gate[size_t(t)*D+c]=t<valid?pw1[size_t(t)*2*D+c]*sigmoid(pw1[size_t(t)*2*D+D+c]):0.0f;
    const Tensor& dw=model_.at(p+"depthwise_conv.weight");
    const Tensor* db=model_.find(p+"depthwise_conv.bias");
    const float* w=reinterpret_cast<const float*>(dw.data);
    const float* gamma=reinterpret_cast<const float*>(model_.at(p+"batch_norm.weight").data);
    const float* beta=reinterpret_cast<const float*>(model_.at(p+"batch_norm.bias").data);
    const float* mean=reinterpret_cast<const float*>(model_.at(p+"batch_norm.running_mean").data);
    const float* var=reinterpret_cast<const float*>(model_.at(p+"batch_norm.running_var").data);
    Buffer y(size_t(T)*D);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(kernels_.threads())
#endif
    for(int c=0;c<D;++c) {
        float scale=gamma[c]/std::sqrt(var[c]+1e-5f);
        float shift=beta[c]-mean[c]*scale;
        float b=db?scalar(*db,c):0;
        for(int t=0;t<T;++t) {
            float v=b;
            for(int j=0;j<K;++j) {
                int ti=t+j-K/2;
                if(ti>=0&&ti<T) v+=w[size_t(c)*K+j]*gate[size_t(ti)*D+c];
            }
            v=v*scale+shift;
            y[size_t(t)*D+c]=v*sigmoid(v);
        }
    }
    Tensor pw2w=model_.at(p+"pointwise_conv2.weight");
    pw2w.rank=2; pw2w.shape={D,D,1,1};
    return kernels_.linear(pw2w,y,T,
                           model_.find(p+"pointwise_conv2.bias"));
}

Buffer Encoder::layer(const Buffer& x,int T,int valid,int index,const Buffer& pos) {
    using Clock=std::chrono::steady_clock;
    auto elapsed=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
    const int D=int(model_.integer("parakeet.encoder.d_model",1024));
    const std::string p="encoder.layers."+std::to_string(index)+".";
    auto ln=[&](const Buffer& a,const std::string& n) {
        return norm(a,T,D,model_.at(p+n+".weight"),model_.at(p+n+".bias"));
    };
    auto ff=[&](const Buffer& a,const std::string& n) {
        std::string s=p+n+".";
        Buffer h=kernels_.linear(model_.at(s+"linear1.weight"),a,T,
                                  model_.find(s+"linear1.bias"));
        auto activation_start=Clock::now();
        silu_inplace(h,kernels_.threads());
        stage_seconds_[9]+=elapsed(activation_start,Clock::now());
        return kernels_.linear(model_.at(s+"linear2.weight"),h,T,
                               model_.find(s+"linear2.bias"));
    };
    Buffer r=x;
    auto s=Clock::now();
    add_inplace(r,ff(ln(r,"norm_feed_forward1"),"feed_forward1"),0.5f);
    auto e=Clock::now(); stage_seconds_[3]+=elapsed(s,e); s=e;
    add_inplace(r,attention(ln(r,"norm_self_att"),T,valid,index,p+"self_attn.",pos));
    e=Clock::now(); stage_seconds_[4]+=elapsed(s,e); s=e;
    add_inplace(r,conv(ln(r,"norm_conv"),T,valid,p+"conv."));
    e=Clock::now(); stage_seconds_[5]+=elapsed(s,e); s=e;
    add_inplace(r,ff(ln(r,"norm_feed_forward2"),"feed_forward2"),0.5f);
    e=Clock::now(); stage_seconds_[6]+=elapsed(s,e); s=e;
    auto result=ln(r,"norm_out");
    e=Clock::now();stage_seconds_[7]+=elapsed(s,e);
    return result;
}

Buffer Encoder::run(const float* pcm,size_t samples,int& frames) {
    reset_stream();
    return run_window(pcm,samples,0,frames);
}

Buffer Encoder::run_window(const float* pcm,size_t samples,uint64_t first_sample,int& frames) {
    using Clock=std::chrono::steady_clock;
    auto elapsed=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
    auto s=Clock::now();
    Mel mel=compute_mel(model_,pcm,samples,&mel_cache_,first_sample);
    auto e=Clock::now();stage_seconds_[0]+=elapsed(s,e);s=e;
    int valid=0;
    Buffer x=subsample(model_,kernels_,mel,frames,valid);
    e=Clock::now();stage_seconds_[1]+=elapsed(s,e);s=e;
    const int D=int(model_.integer("parakeet.encoder.d_model",1024));
    Buffer pos=positions(frames,D);
    e=Clock::now();stage_seconds_[2]+=elapsed(s,e);
    const int layers=int(model_.integer("parakeet.encoder.n_layers",24));
    const size_t pos_bytes=size_t(layers)*size_t(2*frames-1)*D*sizeof(float);
    if(cached_frames_!=frames) {
        pos_cache_.clear();
        if(pos_bytes<=32u*1024*1024) pos_cache_.resize(layers);
        cached_frames_=frames;
    }
    for(int i=0;i<layers;++i) {
        x=layer(x,frames,valid,i,pos);
    }
    s=Clock::now();
    auto out=kernels_.linear(model_.at("joint.enc.weight"),x,frames,
                             &model_.at("joint.enc.bias"));
    e=Clock::now();stage_seconds_[8]+=elapsed(s,e);
    return out;
}

} // namespace direct
