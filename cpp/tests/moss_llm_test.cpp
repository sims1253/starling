#include "golden_io.hpp"
#include "moss/llm.hpp"
#include "moss/prompt.hpp"
#include "moss/tokenizer.hpp"
#include "runtime/graph.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
using namespace starling::ggml;
struct Stats{size_t eq=0,n=0;double maxabs=0;bool size_match=true,finite=true;};
// Hardened compare: exact size match against the loaded golden (its width is
// the ground truth -- logits must be exactly the documented 151936 vocab) and
// explicit non-finite detection (std::max silently drops NaN diffs).
static Stats cmp(const std::vector<float>&a,const std::vector<float>&b){
  Stats s;s.size_match=(a.size()==b.size());s.n=a.size();
  size_t n=std::min(a.size(),b.size());
  for(size_t i=0;i<n;++i){
    s.eq+=a[i]==b[i];
    double d=std::abs((double)a[i]-(double)b[i]);
    if(std::isfinite(d))s.maxabs=std::max(s.maxabs,d);else s.finite=false;
  }
  for(float v:a)if(!std::isfinite((double)v)){s.finite=false;break;}
  for(float v:b)if(!std::isfinite((double)v)){s.finite=false;break;}
  return s;
}
static std::vector<int> top5(const std::vector<float>&x){std::vector<int>v(x.size());for(size_t i=0;i<v.size();++i)v[i]=i;std::partial_sort(v.begin(),v.begin()+std::min<size_t>(5,v.size()),v.end(),[&](int a,int b){return x[a]>x[b]||(x[a]==x[b]&&a<b);});v.resize(std::min<size_t>(5,v.size()));return v;}
static std::string text(const std::string&p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char**argv){std::setvbuf(stdout,nullptr,_IONBF,0);std::string root=argc>1?argv[1]:".";std::string only=argc>2?argv[2]:"";moss::MossModel m;std::string e;if(!m.load((root+"/models/moss-transcribe-preview-2b-bf16-exact.gguf").c_str(),e)){std::fprintf(stderr,"load: %s\n",e.c_str());return 2;}moss::Tokenizer tok;if(!tok.load(m.loader,m.config,e)){std::fprintf(stderr,"tokenizer: %s\n",e.c_str());return 2;}bool all=true;std::printf("fixture embeds-bitwise embeds-maxabs prefill-bitwise prefill-maxabs top5 argmax ids text\n");for(const char*n:{"short","medium","long"}){if(!only.empty()&&only!=n)continue;std::string base=root+"/golden/raw/moss_"+n+"_";std::vector<int64_t> pi,gi;std::vector<float> ga,ge,glog;if(!test::read_i64(base+"prompt_ids.i64",pi,e)||!test::read_i64(base+"ids.i64",gi,e)||!test::read_f32(base+"audio_embeds.f32",ga,e)||!test::read_f32(base+"inputs_embeds.f32",ge,e)||!test::read_f32(base+"prefill_logits.f32",glog,e)){std::fprintf(stderr,"%s\n",e.c_str());return 2;}
 auto p=moss::build_transcribe_prompt(m.config,(std::string(n)=="short"?743:(std::string(n)=="medium"?2230:7435)));bool prompt=pi.size()==p.ids.size();for(size_t i=0;prompt&&i<pi.size();++i)prompt=pi[i]==p.ids[i];moss::AudioEncoding a;a.data=ga;a.n_tokens=ga.size()/m.config.llm.hidden;a.width=m.config.llm.hidden;moss::InputsEmbeds in;if(!moss::build_inputs_embeds(m,p,a,in,e)){std::fprintf(stderr,"embed: %s\n",e.c_str());return 2;}auto es=cmp(in.data,ge);
 // LLM gates intentionally start from the independent golden merged embedding,
 // so decoder failures are not hidden by the known encoder/adapter gate.
 in.data=ge;in.n_tokens=ge.size()/m.config.llm.hidden;in.width=m.config.llm.hidden;moss::PrefillResult staged;if(!moss::llm_prefill(m,in,2048,staged,e)){std::fprintf(stderr,"prefill %s: %s\n",n,e.c_str());return 2;}auto ls=cmp(staged.logits,glog);auto st1=top5(staged.logits),st2=top5(glog);std::printf("%-7s %7.3f%% %.7g %7.3f%% %.7g %s %s pending pending STAGED\n",n,es.n?100.0*es.eq/es.n:100,es.maxabs,ls.n?100.0*ls.eq/ls.n:100,ls.maxabs,st1==st2?"yes":"NO",(!st1.empty()&&!st2.empty()&&st1[0]==st2[0])?"yes":"NO");
 moss::GenerateResult gr;moss::GenerateOptions op;if(!moss::greedy_generate(m,in,op,gr,e)){std::fprintf(stderr,"generate %s: %s\n",n,e.c_str());return 2;}ls=cmp(gr.prefill_logits,glog);auto t1=top5(gr.prefill_logits),t2=top5(glog);bool ids=gr.ids.size()==gi.size();for(size_t i=0;ids&&i<gi.size();++i)ids=gr.ids[i]==gi[i];std::string got=tok.decode(gr.ids,true),want=text(root+"/golden/moss_"+n+"_text.txt");bool tx=got==want;
 // Gate: token-exact (the golden path's bf16 ULP compounding makes a strict
 // bitwise-logits gate infeasible). Require
 // bitwise embeds, exact argmax, a generous logits max-abs bound
 // (observed <= 3.75), and exact ids/text.
 // top5 (ordered, and even set equality under exact-value ties) is
 // informational only: bf16 ties reorder by index tie-break.
 bool pass=prompt&&es.size_match&&es.finite&&es.eq==es.n&&ls.size_match&&ls.finite&&ls.maxabs<=8.0&&!t1.empty()&&t1[0]==t2[0]&&ids&&tx;all&=pass;
 if(!pass)std::fprintf(stderr,"  [fail-components] prompt=%d embeds=%d(eSz=%d,eFin=%d) logitSz=%d logitFin=%d maxabs=%.4g argmax=%d ids=%d text=%d\n",(int)prompt,(int)(es.size_match&&es.eq==es.n),(int)es.size_match,(int)es.finite,(int)ls.size_match,(int)ls.finite,ls.maxabs,(int)(!t1.empty()&&!t2.empty()&&t1[0]==t2[0]),(int)ids,(int)tx);std::printf("%-7s %7.3f%% %.7g %7.3f%% %.7g %s %s %s %s%s\n",n,es.n?100.0*es.eq/es.n:100,es.maxabs,ls.n?100.0*ls.eq/ls.n:100,ls.maxabs,t1==t2?"yes":"NO",(!t1.empty()&&!t2.empty()&&t1[0]==t2[0])?"yes":"NO",ids?"yes":"NO",tx?"yes":"NO",pass?"":" FAIL");}
 shutdown_backend();return all?0:1;}
