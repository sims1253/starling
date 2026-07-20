#include "adapter.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace starling::ggml::moss {
bool apply_adapter(const MossModel& model,const AudioEncoding& in,AudioEncoding& out,std::string& err){
    // Must happen before run_graph() takes the global backend mutex.
    ensure_weights_realized(model.loader);
 if(in.n_tokens<=0||in.width!=(int64_t)model.config.adapter_input||in.data.size()!=(size_t)in.n_tokens*in.width){err="invalid MOSS adapter input";return false;}
 std::vector<ggml_bf16_t> host(in.data.size());for(size_t i=0;i<host.size();++i)host[i]=ggml_fp32_to_bf16(in.data[i]);
 bool ok=run_graph([&](ggml_context*c){int64_t ne[2]={in.width,in.n_tokens};auto*x=graph_input_tensor(c,GGML_TYPE_BF16,2,ne,host.data(),host.size()*sizeof(host[0]));
  auto lin=[&](const char*n,ggml_tensor*z){return ggml_cast(c,ggml_mul_mat(c,clone_weight(c,model.loader,n),z),GGML_TYPE_BF16);};
  auto*g=lin("adapter.gate.weight",x);auto*u=lin("adapter.up.weight",x);
  // ATen boundaries: BF16 gate -> F32 SiLU -> BF16, then one BF16 multiply
  // result before the down projection. Generic ggml elementwise is F32-only.
  auto*a=ggml_cast(c,ggml_silu(c,ggml_cast(c,g,GGML_TYPE_F32)),GGML_TYPE_BF16);
  auto*z=ggml_cast(c,ggml_mul(c,ggml_cast(c,a,GGML_TYPE_F32),ggml_cast(c,u,GGML_TYPE_F32)),GGML_TYPE_BF16);
  return ggml_cast(c,lin("adapter.down.weight",z),GGML_TYPE_F32);
 },out.data);
 if(!ok){err="MOSS adapter graph execution failed";return false;}out.n_tokens=in.n_tokens;out.width=model.config.adapter_output;
 const char*p=std::getenv("STARLING_MOSS_DEBUG");if(p&&std::strcmp(p,"1")==0){double mx=0;for(float x:out.data)mx=std::max(mx,std::abs((double)x));std::fprintf(stderr,"MOSS_DEBUG %-12s elements=%zu max_abs_value=%.9g (compare performed by test)\n","post-adapter",out.data.size(),mx);}return true;
}
} // namespace starling::ggml::moss
