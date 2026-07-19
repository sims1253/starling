#include "golden_io.hpp"
#include "moss/adapter.hpp"
#include "moss/audio_encoder.hpp"
#include "moss/loader.hpp"
#include "moss/mel.hpp"
#include "runtime/audio_io.hpp"
#include "runtime/graph.hpp"
#include "runtime/backend.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace starling::ggml;
struct Stats{size_t equal=0,n=0;double max_abs=0;bool size_match=true,finite=true;};
// Hardened compare: exact size match (truncation/empty output must not pass)
// and explicit non-finite detection. std::max silently drops NaN diffs
// ((finite < NaN) is false, so a NaN value never raises max_abs); track
// finiteness explicitly so a NaN/Inf output is a hard failure.
static Stats compare(const std::vector<float>&a,const std::vector<float>&b){
  Stats s;s.size_match=(a.size()==b.size());s.n=a.size();
  size_t n=std::min(a.size(),b.size());
  for(size_t i=0;i<n;++i){
    s.equal+=a[i]==b[i];
    double d=std::abs((double)a[i]-(double)b[i]);
    if(std::isfinite(d))s.max_abs=std::max(s.max_abs,d);else s.finite=false;
  }
  for(float v:a)if(!std::isfinite((double)v)){s.finite=false;break;}
  for(float v:b)if(!std::isfinite((double)v)){s.finite=false;break;}
  return s;
}
int main(int argc,char**argv){std::setvbuf(stdout,nullptr,_IONBF,0);std::string root=argc>1?argv[1]:".";moss::MossModel model;std::string e;if(!model.load((root+"/models/moss-transcribe-preview-2b-bf16-exact.gguf").c_str(),e)){std::fprintf(stderr,"load: %s\n",e.c_str());return 2;}bool all=true;std::printf("device fixture stage              bitwise       max-abs\n");for(const char*name:{"short","medium","long"}){std::vector<float>pcm;int sr=0;if(!read_wav((root+"/tests/fixtures/"+name+".wav").c_str(),pcm,sr,e)||sr!=16000){std::fprintf(stderr,"wav: %s\n",e.c_str());return 2;}moss::MelFeatures mel;if(!moss::compute_log_mel(model.config,model.loader,pcm.data(),pcm.size(),mel,e)){std::fprintf(stderr,"mel: %s\n",e.c_str());return 2;}moss::AudioEncoding enc,ad;if(!moss::encode_audio(model,mel,enc,e)||!moss::apply_adapter(model,enc,ad,e)){std::fprintf(stderr,"%s: %s\n",name,e.c_str());return 2;}for(auto item:{std::pair<const char*,const std::vector<float>*>("encoder_hidden",&enc.data),{"audio_embeds",&ad.data}}){std::vector<float>gold;if(!test::read_f32(root+"/golden/raw/moss_"+name+"_"+item.first+".f32",gold,e)){std::fprintf(stderr,"%s\n",e.c_str());return 2;}Stats s=compare(*item.second,gold);double pct=s.n?100.0*s.equal/s.n:100;
    // Token-exact contract (docs/ggml-moss-goldens.md): bitwise-vs-eager is
    // infeasible after 32 attention layers of bf16 ULP compounding; what
    // matters is audio_embeds accuracy (LLM ids/text gate verifies the
    // downstream). Tolerances = ~1.5x observed on the eager golden path.
    double tol=std::string(item.first)=="encoder_hidden"?0.02:0.001;
    if(!s.size_match)std::fprintf(stderr,"%s %s: SIZE MISMATCH got=%zu gold=%zu\n",name,item.first,item.second->size(),gold.size());
    if(!s.finite)std::fprintf(stderr,"%s %s: NON-FINITE value present\n",name,item.first);
    bool pass=s.size_match&&s.finite&&s.max_abs<=tol;all&=pass;std::printf("%-6s %-7s %-18s %9.5f%%  %.9g%s\n",global_backend().device_name(),name,item.first,pct,s.max_abs,pass?"":" FAIL");}}
 shutdown_backend();return all?0:1;}
