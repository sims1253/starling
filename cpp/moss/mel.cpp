#include "mel.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>
#include "lib/pocketfft_hdronly.h"
#include "lib/threads.hpp"
#include "ggml-backend.h"


namespace starling::ggml::moss {
namespace {
size_t reflect_index(int64_t i,size_t n) {
    if(n<=1)return 0; const int64_t period=2*(int64_t)n-2; i%=period; if(i<0)i+=period; return (size_t)(i<(int64_t)n?i:period-i);
}

// Worker count for the mel frontend.
//   STARLING_MEL_THREADS unset -> min(hardware_concurrency(), 16)
//   STARLING_MEL_THREADS=N     -> N (>=1); N==1 forces the serial path.
// Parallelizing the per-frame / per-(m,t) loop nests changes neither any
// per-element floating-point operation nor any inner accumulation order
// (the filterbank's inner sum stays b=0..B-1), so the output is bit-identical
// to the serial path. pocketfft::r2c is reentrant: its plan cache is mutex-
// guarded and every call uses its own scratch, so concurrent calls from
// multiple std::threads are safe.
// Run body(tid, lo, hi) over [0, total) in contiguous disjoint chunks, one per
// std::thread. Serial path (nthr<=1): a single call body(0, 0, total), no
// threads spawned. tid is the chunk index (0..spawned-1) so callers can fold
// per-chunk partial results into a deterministic order-sensitive reduction.
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
    const size_t fullT=S/H+1, T=fullT-1; std::vector<float> logmel(M*fullT); std::vector<double> powers(B*fullT); std::vector<double> mel64(M*fullT); const size_t nthr=lib::mel_thread_count();
    // Transpose the filterbank to bank_t[m*B+b] (== bank[b*M+m], contiguous per
    // m) and lay out powers as powers[t*B+b] (contiguous per frame). Loop 2 then
    // reads both operands contiguously in b instead of strided (bank stride M,
    // powers stride fullT), turning the filterbank into cache-friendly dot
    // products. Same values, same b=0..B-1 accumulation order -> bit-identical.
    std::vector<float> bank_t(M*B); for(size_t m=0;m<M;++m) for(size_t b=0;b<B;++b) bank_t[m*B+b]=bank[b*M+m];
    // Loop 1: per frame reflect-pad + window + r2c FFT + power. Frames are
    // fully independent; frame[] and z[] were shared across iterations in the
    // serial path, so each thread now owns a private copy.
    lib::parallel_for(nthr, fullT, [&](size_t /*tid*/, size_t lo, size_t hi){
        std::vector<double> frame(N); std::vector<std::complex<double>> z(B);
        for(size_t t=lo;t<hi;++t){
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
                powers[t*B+b]=(double)power;
            }
        }
    });
    // Loop 2: mel filterbank. Each (m,t) cell is an independent dot product;
    // the inner accumulation stays exactly b=0..B-1, and cells write disjoint
    // mel64[] entries, so the parallel path is bit-identical to the serial one.
    lib::parallel_for(nthr, M*fullT, [&](size_t /*tid*/, size_t lo, size_t hi){
        for(size_t idx=lo;idx<hi;++idx){
            const size_t m=idx/fullT, t=idx%fullT; double a=0;
            const float* fb=&bank_t[m*B]; const double* pw=&powers[t*B];
            for(size_t b=0;b<B;++b)a+=(double)fb[b]*pw[b];
            mel64[m*fullT+t]=a;
        }
    });
    // Loop 3a: log10, plus a per-chunk max for a deterministic global-max
    // reduction (per-chunk maxes combined in chunk order; max is order-
    // insensitive, but keep the reduction deterministic regardless).
    std::vector<float> chunk_max(nthr, -std::numeric_limits<float>::infinity());
    lib::parallel_for(nthr, M*fullT, [&](size_t tid, size_t lo, size_t hi){
        float cm=-std::numeric_limits<float>::infinity();
        for(size_t idx=lo;idx<hi;++idx){
            float v=(float)std::log10(std::max(mel64[idx],(double)c.mel_floor));
            logmel[idx]=v; cm=std::max(cm,v);
        }
        chunk_max[tid]=cm;
    });
    float mx=-std::numeric_limits<float>::infinity(); for(size_t i=0;i<nthr;++i) mx=std::max(mx,chunk_max[i]);
    out.n_mels=M; out.n_frames=T; out.f32.resize(M*T); out.data.resize(M*T);
    // Loop 3b: clamp + normalize + bf16. Each (m,t) output is independent.
    lib::parallel_for(nthr, M*T, [&](size_t /*tid*/, size_t lo, size_t hi){
        for(size_t idx=lo;idx<hi;++idx){
            const size_t m=idx/T, t=idx%T;
            float v=std::max(logmel[m*fullT+t],mx-c.dynamic_range); v=(v+c.normalization_offset)/c.normalization_divisor;
            const size_t i=m*T+t; out.f32[i]=v; out.data[i]=ggml_fp32_to_bf16(v);
        }
    });
    if(const char* p=std::getenv("STARLING_MEL_DUMP")){if(FILE* f=std::fopen(p,"wb")){std::fwrite(out.f32.data(),sizeof(float),out.f32.size(),f);std::fclose(f);}}
    return true;
}
} // namespace starling::ggml::moss
