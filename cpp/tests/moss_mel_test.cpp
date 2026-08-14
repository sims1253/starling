#include "golden_io.hpp"
#include "moss/loader.hpp"
#include "moss/mel.hpp"
#include "runtime/audio_io.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace starling::ggml;
int main(int argc,char**argv){std::setvbuf(stdout,nullptr,_IONBF,0);std::string root=argc>1?argv[1]:"."; moss::MossModel model;std::string e;if(!model.load((root+"/models/moss-transcribe-preview-2b-bf16-exact.gguf").c_str(),e)){std::fprintf(stderr,"load: %s\n",e.c_str());return 2;}bool all=true;for(const char* name:{"short","medium","long"}){std::vector<float> pcm,gold;int sr=0;if(!read_wav((root+"/tests/fixtures/"+name+".wav").c_str(),pcm,sr,e)||sr!=16000){std::fprintf(stderr,"%s wav: %s sr=%d\n",name,e.c_str(),sr);return 2;}moss::MelFeatures got;if(!moss::compute_log_mel(model.config,model.loader,pcm.data(),pcm.size(),got,e)){std::fprintf(stderr,"%s mel: %s\n",name,e.c_str());return 2;}if(!test::read_f32(root+"/golden/raw/moss_"+name+"_mel.f32",gold,e)){std::fprintf(stderr,"%s\n",e.c_str());return 2;}size_t mism=0;double mx=0;bool ulp_ok=true,finite=true;double worst_excess=0;
 size_t n=std::min(gold.size(),got.data.size());
 for(size_t i=0;i<n;++i){
  float q=ggml_bf16_to_fp32(got.data[i]);
  bool fin=std::isfinite((double)q)&&std::isfinite((double)gold[i]);
  if(!fin)finite=false;
  if(q!=gold[i])++mism;
  if(!fin)continue; // non-finite is already a hard fail; skip abs/ULP bookkeeping
  double diff=std::fabs((double)q-(double)gold[i]);
  mx=std::max(mx,diff);
  if(diff>0.0){
   // Per-bin one-ULP tolerance: bf16 spacing is
   // exponent-dependent, so a single global absolute constant both accepts
   // >1 ULP in small-magnitude bins and rejects a legit one-ULP bin in a
   // larger binade. Measure the one-bf16-step gap at THIS reference's
   // magnitude: the reference is an exact bf16 value, so round it to bf16
   // bits and take the adjacent bf16 value one step away from zero.
   uint16_t rb=ggml_fp32_to_bf16(gold[i]).bits;
   float adj=ggml_bf16_to_fp32(ggml_bf16_t{(uint16_t)(rb+1)});
   double ulp=std::isfinite((double)adj)?std::fabs((double)adj-(double)gold[i]):std::fabs((double)gold[i]);
   double tol=1.5*ulp; // cleanly separates 1 ULP (accept) from 2 ULP (reject) within a binade
   if(diff>tol){ulp_ok=false;worst_excess=std::max(worst_excess,diff-tol);}
  }
 }
 mism+=gold.size()>n?gold.size()-n:got.data.size()-n;
 // Token-exact contract: the reference mel is the
 // processor's bf16 values; our f32 log-mel re-rounds a handful of boundary
 // bins (observed <= 40/951680, each exactly one bf16 ULP). Gate: at most 64
 // mismatching bins, each within one bf16 ULP measured at that bin's own
 // magnitude (per-bin, not a single global absolute constant), and no
 // non-finite output.
 bool ok=mism<=64&&ulp_ok&&finite;all&=ok;std::printf("%-6s %s shape=[%lld,%lld] mismatches=%zu/%zu max_abs=%.9g ulp_ok=%d finite=%d worst_ulp_excess=%.3g\n",name,ok?"PASS":"FAIL",(long long)got.n_mels,(long long)got.n_frames,mism,std::max(gold.size(),got.data.size()),mx,(int)ulp_ok,(int)finite,worst_excess);}return all?0:1;}
