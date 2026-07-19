#include "mel.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "pocketfft_hdronly.h"
#include "ggml-backend.h"


namespace starling::ggml::moss {
namespace {
size_t reflect_index(int64_t i,size_t n) {
    if(n<=1)return 0; const int64_t period=2*(int64_t)n-2; i%=period; if(i<0)i+=period; return (size_t)(i<(int64_t)n?i:period-i);
}
}

bool compute_log_mel(const Config& cfg,const ModelLoader& ml,const float* pcm,size_t S,MelFeatures& out,std::string& err){
    const auto& c=cfg.frontend; const size_t N=c.n_fft,H=c.hop_length,M=c.n_mels,B=N/2+1;
    if(!pcm&&S){err="null PCM input";return false;} if(S<2){err="MOSS reflect padding requires at least 2 PCM samples";return false;}
    auto* wt=ml.tensor("audio.mel_window"); auto* ft=ml.tensor("audio.mel_filters");
    if(!wt||!ft||wt->type!=GGML_TYPE_F32||ft->type!=GGML_TYPE_F32){err="MOSS mel constants missing or not F32";return false;}
    if((size_t)ggml_nelements(wt)!=N||(size_t)ggml_nelements(ft)!=M*B){err="MOSS mel constant shape mismatch";return false;}
    std::vector<float> window_host, bank_host;
    const float* window=(const float*)wt->data; const float* bank=(const float*)ft->data;
    // Weight realization repoints loader tensors to device tensors. Mel stays
    // on the host, so read constants back when a prior encoder call realized
    // the model instead of dereferencing device/null data on fixture 2.
    if(wt->buffer){window_host.resize(N);ggml_backend_tensor_get(wt,window_host.data(),0,N*sizeof(float));window=window_host.data();}
    if(ft->buffer){bank_host.resize(M*B);ggml_backend_tensor_get(ft,bank_host.data(),0,M*B*sizeof(float));bank=bank_host.data();}
    const size_t fullT=S/H+1, T=fullT-1; std::vector<float> logmel(M*fullT); std::vector<double> powers(B*fullT); std::vector<double> mel64(M*fullT); std::vector<double> frame(N); std::vector<std::complex<double>> z(B);
    for(size_t t=0;t<fullT;++t){
        const int64_t start=(int64_t)(t*H)-(int64_t)(N/2);
        for(size_t i=0;i<N;++i)frame[i]=(double)pcm[reflect_index(start+(int64_t)i,S)]*(double)window[i];
        pocketfft::r2c({N},{sizeof(double)},{sizeof(std::complex<double>)},0,true,frame.data(),z.data(),1.0);
        for(size_t b=0;b<B;++b){
            // NumPy stores the RFFT as complex64, computes abs(complex64) in
            // float32, then squares that float32 for power=2. Preserve both
            // rounding boundaries rather than widening magnitude/power to f64.
            const float re=(float)z[b].real(), im=(float)z[b].imag();
            const float mag=std::hypot(re,im);
            const float power=mag*mag;
            powers[b*fullT+t]=(double)power;
        }
    }
    for(size_t m=0;m<M;++m)for(size_t t=0;t<fullT;++t){double a=0; for(size_t b=0;b<B;++b)a+=(double)bank[b*M+m]*powers[b*fullT+t]; mel64[m*fullT+t]=a;}
    for(size_t m=0;m<M;++m)for(size_t t=0;t<fullT;++t) logmel[m*fullT+t]=(float)std::log10(std::max(mel64[m*fullT+t],(double)c.mel_floor));
    float mx=-std::numeric_limits<float>::infinity(); for(float v:logmel) mx=std::max(mx,v);
    out.n_mels=M; out.n_frames=T; out.f32.resize(M*T); out.data.resize(M*T);
    for(size_t m=0;m<M;++m)for(size_t t=0;t<T;++t){float v=std::max(logmel[m*fullT+t],mx-c.dynamic_range); v=(v+c.normalization_offset)/c.normalization_divisor; size_t i=m*T+t; out.f32[i]=v; out.data[i]=ggml_fp32_to_bf16(v);}
    if(const char* p=std::getenv("STARLING_MEL_DUMP")){if(FILE* f=std::fopen(p,"wb")){std::fwrite(out.f32.data(),sizeof(float),out.f32.size(),f);std::fclose(f);}}
    return true;
}
} // namespace starling::ggml::moss
