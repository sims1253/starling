#include "prompt.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
namespace starling::ggml::moss {
Prompt build_transcribe_prompt(const Config& c,int64_t frames){
 Prompt p; const std::vector<int32_t> pre=c.prompt_prefix.empty()?std::vector<int32_t>{151644,872,198,151669}:c.prompt_prefix;
 const std::vector<int32_t> suf=c.prompt_suffix.empty()?std::vector<int32_t>{151670,151645,198,151644,77091,198}:c.prompt_suffix;
 const auto n=audio_token_length(frames); p.ids.reserve(pre.size()+n+suf.size());p.audio_mask.reserve(p.ids.capacity());
 for(auto x:pre){p.ids.push_back(x);p.audio_mask.push_back(0);}for(int64_t i=0;i<n;++i){p.ids.push_back(c.audio_placeholder_id);p.audio_mask.push_back(1);}for(auto x:suf){p.ids.push_back(x);p.audio_mask.push_back(0);}return p;
}
bool build_inputs_embeds(const MossModel&m,const Prompt&p,const AudioEncoding&a,InputsEmbeds&o,std::string&e){
 if(p.ids.size()!=p.audio_mask.size()){e="invalid MOSS prompt mask";return false;}size_t slots=0;for(auto x:p.audio_mask)slots+=x!=0;
 if(a.n_tokens!=(int64_t)slots||a.width!=(int64_t)m.config.llm.hidden||a.data.size()!=slots*a.width){e="audio/prompt scatter size mismatch";return false;}
 ensure_weights_realized(m.loader);std::vector<int32_t> ids=p.ids;std::vector<ggml_bf16_t> ah(a.data.size());for(size_t i=0;i<ah.size();++i)ah[i]=ggml_fp32_to_bf16(a.data[i]);
 std::vector<float> emb;bool ok=run_graph([&](ggml_context*c){int64_t ne[1]={(int64_t)ids.size()};auto*it=graph_input_tensor(c,GGML_TYPE_I32,1,ne,ids.data(),ids.size()*sizeof(ids[0]));return ggml_cast(c,ggml_get_rows(c,clone_weight(c,m.loader,"llm.embed.weight"),it),GGML_TYPE_F32);},emb);
 if(!ok){e="MOSS embedding lookup failed";return false;}size_t row=0;for(size_t i=0;i<p.ids.size();++i)if(p.audio_mask[i]){for(size_t d=0;d<a.width;++d)emb[i*a.width+d]=ggml_bf16_to_fp32(ah[row*a.width+d]);++row;}
 o.data=std::move(emb);o.n_tokens=p.ids.size();o.width=m.config.llm.hidden;return true;
}
}
