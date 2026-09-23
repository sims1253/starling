#include "engine_audio.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace direct {
namespace {
#if defined(POC_REFERENCE_NUMERICS)
float round_f16(float x) {
#if defined(__F16C__)
    return _cvtsh_ss(_cvtss_sh(x,0));
#else
    return float(static_cast<_Float16>(x));
#endif
}
#endif
void fft(std::vector<std::complex<double>>& a) {
    const int n=int(a.size());
    std::vector<double> re(n),im(n);
    for(int i=0;i<n;++i) { re[i]=a[i].real(); im[i]=a[i].imag(); }
    for(int i=1,j=0;i<n;++i) {
        int bit=n>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j) { std::swap(re[i],re[j]); std::swap(im[i],im[j]); }
    }
    for(int len=2;len<=n;len<<=1) {
        double ang=-2.0*3.14159265358979323846/len;
        double wr=std::cos(ang),wi=std::sin(ang);
        for(int i=0;i<n;i+=len) {
            double cr=1,ci=0;
            for(int k=0;k<len/2;++k) {
                int u=i+k,v=u+len/2;
                double tr=cr*re[v]-ci*im[v],ti=cr*im[v]+ci*re[v];
                re[v]=re[u]-tr; im[v]=im[u]-ti;
                re[u]+=tr; im[u]+=ti;
                double nr=cr*wr-ci*wi;
                ci=cr*wi+ci*wr; cr=nr;
            }
        }
    }
    for(int i=0;i<n;++i) a[i]={re[i],im[i]};
}

struct Grid {
    Buffer a;
    int t=0,f=0,c=0;
    float& at(int ti,int fi,int ci) { return a[(size_t(ti)*f+fi)*c+ci]; }
    float at(int ti,int fi,int ci) const { return a[(size_t(ti)*f+fi)*c+ci]; }
};
Grid depthwise(const Grid& x,const Tensor& w,const Tensor& bias,int stride,int threads) {
    Grid y; y.t=(x.t+1)/stride; y.f=(x.f+1)/stride; y.c=x.c;
    y.a.resize(size_t(y.t)*y.f*y.c);
    const int positions=y.t*y.f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads>1 && positions>256)
#endif
    for(int pos=0;pos<positions;++pos) {
        const int t=pos/y.f,f=pos%y.f;
        for(int c=0;c<x.c;++c) {
            float v=scalar(bias,c);
            for(int kt=0;kt<3;++kt) for(int kf=0;kf<3;++kf) {
                int ti=t*stride+kt-1,fi=f*stride+kf-1;
                if(ti>=0&&ti<x.t&&fi>=0&&fi<x.f)
                    v+=x.at(ti,fi,c)*scalar(w,size_t(c)*9+kt*3+kf);
            }
            y.at(t,f,c)=v;
        }
    }
    return y;
}

Grid pointwise(const Grid& x,const Tensor& w,const Tensor& bias,int threads) {
    if(w.rank!=4 || w.shape[0]!=1 || w.shape[1]!=1 || w.shape[2]!=x.c)
        throw std::runtime_error("pointwise convolution shape mismatch");
    Grid y; y.t=x.t; y.f=x.f; y.c=int(w.shape[3]);
    y.a.resize(size_t(y.t)*y.f*y.c);
    const float* wp=reinterpret_cast<const float*>(w.data);
    const float* bp=reinterpret_cast<const float*>(bias.data);
#if defined(__AVX2__) && defined(__FMA__)
    if(x.c==256 && y.c==256) {
        // Two positions by two output channels. All four dot products share
        // their input/weight vector loads and keep the original dot reduction
        // order so the model's marginal token decisions do not move.
        const int positions=x.t*x.f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for(int tile=0;tile<positions/2;++tile) {
            const int p=tile*2;
            const float* x0=x.a.data()+size_t(p)*256;
            const float* x1=x0+256;
            float* y0=y.a.data()+size_t(p)*256;
            float* y1=y0+256;
            for(int c=0;c<256;c+=2) {
                const float* w0=wp+size_t(c)*256;
                const float* w1=w0+256;
                __m256 a00=_mm256_setzero_ps(),b00=_mm256_setzero_ps();
                __m256 a01=_mm256_setzero_ps(),b01=_mm256_setzero_ps();
                __m256 a10=_mm256_setzero_ps(),b10=_mm256_setzero_ps();
                __m256 a11=_mm256_setzero_ps(),b11=_mm256_setzero_ps();
                for(int k=0;k<256;k+=16) {
                    __m256 x0a=_mm256_loadu_ps(x0+k),x0b=_mm256_loadu_ps(x0+k+8);
                    __m256 x1a=_mm256_loadu_ps(x1+k),x1b=_mm256_loadu_ps(x1+k+8);
                    __m256 w0a=_mm256_loadu_ps(w0+k),w0b=_mm256_loadu_ps(w0+k+8);
                    __m256 w1a=_mm256_loadu_ps(w1+k),w1b=_mm256_loadu_ps(w1+k+8);
                    a00=_mm256_fmadd_ps(x0a,w0a,a00);b00=_mm256_fmadd_ps(x0b,w0b,b00);
                    a01=_mm256_fmadd_ps(x0a,w1a,a01);b01=_mm256_fmadd_ps(x0b,w1b,b01);
                    a10=_mm256_fmadd_ps(x1a,w0a,a10);b10=_mm256_fmadd_ps(x1b,w0b,b10);
                    a11=_mm256_fmadd_ps(x1a,w1a,a11);b11=_mm256_fmadd_ps(x1b,w1b,b11);
                }
                auto reduce=[](__m256 a,__m256 b) {
                    alignas(32) float lanes[8];
                    _mm256_store_ps(lanes,_mm256_add_ps(a,b));
                    return lanes[0]+lanes[1]+lanes[2]+lanes[3]+
                           lanes[4]+lanes[5]+lanes[6]+lanes[7];
                };
                y0[c]=std::max(0.0f,reduce(a00,b00)+bp[c]);
                y0[c+1]=std::max(0.0f,reduce(a01,b01)+bp[c+1]);
                y1[c]=std::max(0.0f,reduce(a10,b10)+bp[c]);
                y1[c+1]=std::max(0.0f,reduce(a11,b11)+bp[c+1]);
            }
        }
        if(positions&1) {
            const int p=positions-1;
            const float* a=x.a.data()+size_t(p)*256;
            float* out=y.a.data()+size_t(p)*256;
            for(int c=0;c<256;++c)
                out[c]=std::max(0.0f,dot(wp+size_t(c)*256,a,256)+bp[c]);
        }
        return y;
    }
#endif
    constexpr int tile=16;
    const int positions=x.t*x.f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for(int block=0;block<(positions+tile-1)/tile;++block) {
        const int begin=block*tile,end=std::min(begin+tile,positions);
        for(int c=0;c<y.c;++c) {
            const float* wr=wp+size_t(c)*x.c;
            const float bias_value=bp[c];
            for(int p=begin;p<end;++p) {
                const float* a=x.a.data()+size_t(p)*x.c;
                y.a[size_t(p)*y.c+c]=std::max(0.0f,dot(wr,a,x.c)+bias_value);
            }
        }
    }
    return y;
}

} // namespace

Mel compute_mel(const Model& model,const float* pcm,size_t samples,
                MelCache* cache,uint64_t first_sample) {
    const int nfft=int(model.integer("parakeet.preprocessor.n_fft",512));
    const int win=int(model.integer("parakeet.preprocessor.win_length",400));
    const int hop=int(model.integer("parakeet.preprocessor.hop_length",160));
    const int nm=int(model.integer("parakeet.preprocessor.n_mels",128));
    const float preemph=float(model.real("parakeet.preprocessor.preemph",0.97));
    const float power=float(model.real("parakeet.preprocessor.mag_power",2));
    const double guard=model.real("parakeet.preprocessor.log_zero_guard",5.9604645e-08);
    const int bins=nfft/2+1;
    const Tensor& window_t=model.at("preprocessor.featurizer.window");
    const Tensor& fb=model.at("preprocessor.featurizer.fb");
    if(window_t.cols()!=win || fb.cols()!=bins || fb.rows()!=nm)
        throw std::runtime_error("mel tensor shape mismatch");
    std::vector<float> window(nfft,0);
    for(int i=0;i<win;++i) window[(nfft-win)/2+i]=scalar(window_t,i);
    std::vector<double> pre(samples),padded(samples+nfft,0);
    if(samples) pre[0]=pcm[0];
    for(size_t i=1;i<samples;++i) pre[i]=double(pcm[i])-double(preemph)*double(pcm[i-1]);
    for(size_t i=0;i<samples;++i) padded[nfft/2+i]=pre[i];
    Mel result; result.frames=int(samples/hop)+1; result.valid=int(samples/hop);
    result.values.resize(size_t(result.frames)*nm);
    std::vector<uint8_t> reused(size_t(result.frames),0);
    if(cache) {
        cache->reused_frames=0;
        if(!cache->pcm.empty() && !cache->raw.empty() && cache->frames>0 &&
           first_sample<=UINT64_MAX-samples &&
           cache->first_sample<=UINT64_MAX-cache->pcm.size() &&
           cache->raw.size()==size_t(cache->frames)*nm) {
            uint64_t overlap_begin=std::max(first_sample,cache->first_sample);
            uint64_t overlap_end=std::min(first_sample+samples,
                                          cache->first_sample+cache->pcm.size());
            if(overlap_end>overlap_begin &&
               std::memcmp(pcm+(overlap_begin-first_sample),
                           cache->pcm.data()+(overlap_begin-cache->first_sample),
                           size_t(overlap_end-overlap_begin)*sizeof(float))==0) {
                for(int t=0;t<result.frames;++t) {
                    uint64_t center=first_sample+uint64_t(t)*hop;
                    if(center<cache->first_sample || (center-cache->first_sample)%hop) continue;
                    uint64_t old_t=(center-cache->first_sample)/hop;
                    if(old_t>=uint64_t(cache->frames) || center<uint64_t(nfft/2+1)) continue;
                    uint64_t source_begin=center-uint64_t(nfft/2+1);
                    uint64_t source_end=center+uint64_t(nfft/2);
                    if(source_begin<overlap_begin || source_end>overlap_end) continue;
                    for(int m=0;m<nm;++m)
                        result.values[size_t(m)*result.frames+t]=
                            cache->raw[size_t(m)*cache->frames+old_t];
                    reused[size_t(t)]=1;
                    ++cache->reused_frames;
                }
            }
        }
    }
    std::vector<std::complex<double>> buf(nfft);
    std::vector<double> spectrum(bins);
    const float* filter=reinterpret_cast<const float*>(fb.data);
    for(int t=0;t<result.frames;++t) {
        if(reused[size_t(t)]) continue;
        const int start=t*hop;
        for(int i=0;i<nfft;++i)
            buf[i]={double(float(padded[size_t(start+i)]*double(window[i]))),0};
        fft(buf);
        for(int i=0;i<bins;++i) {
            double re=float(buf[i].real()),im=float(buf[i].imag());
            const double magnitude_squared=re*re+im*im;
            spectrum[i]=power==2.0f ? magnitude_squared
                                     : std::pow(std::sqrt(magnitude_squared),double(power));
        }
        for(int m=0;m<nm;++m) {
            double v=0;
            for(int b=0;b<bins;++b) v+=double(filter[size_t(m)*bins+b])*spectrum[b];
            result.values[size_t(m)*result.frames+t]=float(std::log(v+guard));
        }
    }
    if(cache) {
        // Keep one bounded window. Longer batch jobs have no useful overlap
        // state and should not pin memory in a stream session.
        if(samples<=30u*16000u) {
            cache->first_sample=first_sample;
            cache->pcm.assign(pcm,pcm+samples);
            cache->raw=result.values;
            cache->frames=result.frames;
        } else cache->clear();
    }
    const int valid=std::min(result.valid,result.frames);
    if(valid>=1) for(int m=0;m<nm;++m) {
        float* row=result.values.data()+size_t(m)*result.frames;
        double mean=0;
        for(int t=0;t<valid;++t) mean+=row[t];
        mean/=valid;
        double var=0;
        for(int t=0;t<valid;++t) {double d=row[t]-mean; var+=d*d;}
        var/=valid>=2?valid-1:1;
        double sd=std::sqrt(var)+1e-5;
        for(int t=0;t<result.frames;++t) row[t]=t<valid?float((row[t]-mean)/sd):0;
    }
    return result;
}

Buffer subsample(const Model& model,Kernels& kernels,const Mel& mel,int& frames,int& valid) {
    const int C=int(model.integer("parakeet.encoder.subsampling_conv_channels",256));
    const int F=int(model.integer("parakeet.preprocessor.n_mels",128));
    Grid x; x.t=mel.frames; x.f=F; x.c=1;
    x.a.resize(size_t(x.t)*F);
    for(int t=0;t<x.t;++t) for(int f=0;f<F;++f)
        x.at(t,f,0)=mel.values[size_t(f)*mel.frames+t];
#if defined(POC_REFERENCE_NUMERICS)
    for(float& v:x.a) v=round_f16(v);
#endif
    const std::string p="encoder.pre_encode.";
    const Tensor& w0=model.at(p+"conv.0.weight"),&b0=model.at(p+"conv.0.bias");
    Grid y; y.t=(x.t+1)/2; y.f=(x.f+1)/2; y.c=C;
    y.a.resize(size_t(y.t)*y.f*C);
    const float* w=reinterpret_cast<const float*>(w0.data);
#if defined(POC_REFERENCE_NUMERICS)
    Buffer rounded_w(w,w+size_t(C)*9);
    for(float& v:rounded_w) v=round_f16(v);
    w=rounded_w.data();
#endif
    const float* b=reinterpret_cast<const float*>(b0.data);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(kernels.threads())
#endif
    for(int pos=0;pos<y.t*y.f;++pos) {
        const int t=pos/y.f,f=pos%y.f;
        float patch[9];bool present[9];
        for(int kt=0;kt<3;++kt) for(int kf=0;kf<3;++kf) {
            int ti=t*2+kt-1,fi=f*2+kf-1;
            const int j=kt*3+kf;
            present[j]=ti>=0&&ti<x.t&&fi>=0&&fi<x.f;
            if(present[j]) patch[j]=x.at(ti,fi,0);
        }
        for(int c=0;c<C;++c) {
            float v=b[c];
            for(int j=0;j<9;++j) if(present[j])
                v+=w[size_t(c)*9+j]*patch[j];
            y.at(t,f,c)=std::max(0.0f,v);
        }
    }
    for(int stage: {2,5}) {
        Grid d=depthwise(y,model.at(p+"conv."+std::to_string(stage)+".weight"),
                         model.at(p+"conv."+std::to_string(stage)+".bias"),2,
                         kernels.threads());
        y=pointwise(d,model.at(p+"conv."+std::to_string(stage+1)+".weight"),
                      model.at(p+"conv."+std::to_string(stage+1)+".bias"),kernels.threads());
    }
    frames=y.t; valid=mel.valid;
    for(int i=0;i<3;++i) valid=(valid+1)/2;
    valid=std::min(valid,frames);
    Buffer flat(size_t(frames)*C*y.f);
    for(int t=0;t<frames;++t) for(int c=0;c<C;++c) for(int f=0;f<y.f;++f)
        flat[(size_t(t)*C+c)*y.f+f]=t<valid?y.at(t,f,c):0.0f;
    auto out=kernels.linear(model.at(p+"out.weight"),flat,frames,&model.at(p+"out.bias"));
    return out;
}

std::vector<float> positions(int frames,int dim) {
    std::vector<float> out(size_t(2*frames-1)*dim);
    const double factor=-std::log(10000.0)/dim;
    for(int p=0;p<2*frames-1;++p) for(int i=0;i<dim/2;++i) {
        double arg=double(frames-1-p)*std::exp(double(2*i)*factor);
        out[size_t(p)*dim+2*i]=float(std::sin(arg));
        out[size_t(p)*dim+2*i+1]=float(std::cos(arg));
    }
    return out;
}

} // namespace direct
