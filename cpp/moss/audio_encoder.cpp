#include "audio_encoder.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace starling::ggml::moss {
namespace {

ggml_tensor* weight(ggml_context* c, const ModelLoader& ml, const std::string& n) {
    return clone_weight(c, ml, n.c_str());
}
ggml_tensor* bf16(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_BF16 ? x : ggml_cast(c, x, GGML_TYPE_BF16);
}
ggml_tensor* f32(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
// nn.Linear in the BF16 oracle: the GEMM and bias constitute one operation and
// expose a BF16 tensor. ggml GEMM exposes F32, so round at that boundary.
ggml_tensor* linear(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                    const std::string& n, bool bias) {
    ggml_tensor* y = ggml_mul_mat(c, weight(c, ml, n + ".weight"), bf16(c, x));
    if (bias) y = ggml_add(c, f32(c, y), f32(c, weight(c, ml, n + ".bias")));
    return bf16(c, y);
}
ggml_tensor* conv2d_bf16(ggml_context* c, ggml_tensor* kernel, ggml_tensor* input) {
    // Use ggml's canonical Conv2d builder. Its im2col kernels do not accept a
    // BF16 destination on CUDA, so present the BF16 values as F32 and round the
    // convolution output immediately back to the oracle's BF16 boundary.
    return bf16(c, ggml_conv_2d(c, f32(c, kernel), f32(c, input),
                                2, 2, 1, 1, 1, 1));
}

ggml_tensor* exact_gelu(ggml_context* c, ggml_tensor* x) {
    // ggml_gelu is the tanh approximation. GELU_ERF is the required
    // approximate="none" path. Generic elementwise kernels are F32, then we
    // immediately restore the ATen BF16 output boundary.
    return bf16(c, ggml_gelu_erf(c, f32(c, x)));
}
ggml_tensor* add_bf16(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf16(c, ggml_add(c, f32(c, a), f32(c, b)));
}
ggml_tensor* layer_norm(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    // PyTorch ordinary LayerNorm: F32 reduction and affine, one final BF16
    // store. Do not use the fused NORM+MUL+ADD patch: retaining its F32 result
    // through the following GEMM violates the explicit cast policy.
    ggml_tensor* y = ggml_norm(c, f32(c, x), eps);
    y = ggml_mul(c, y, f32(c, weight(c, ml, n + ".weight")));
    y = ggml_add(c, y, f32(c, weight(c, ml, n + ".bias")));
    return bf16(c, y);
}

bool debug_enabled() {
    const char* p = std::getenv("STARLING_MOSS_DEBUG");
    return p && std::strcmp(p, "1") == 0;
}

} // namespace

bool encode_audio(const MossModel& model, const MelFeatures& mel,
                  AudioEncoding& out, std::string& err) {
    // Weight realization may call global_backend(); do it before run_graph(),
    // which holds the non-recursive global backend mutex while building.
    // Realizing lazily from clone_weight() inside the build lambda deadlocks.
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    if (mel.n_mels != 128 || mel.n_frames <= 0 ||
        mel.data.size() != (size_t)mel.n_mels * mel.n_frames) {
        err = "invalid MOSS mel shape/data"; return false;
    }
    // Unit assertions demanded by the contract; these also protect the subtle
    // remainder-zero behavior from future simplification.
    assert(audio_token_length(743) == 97);
    assert(audio_token_length(2230) == 290);
    assert(audio_token_length(7435) == 967);

    const int64_t T = mel.n_frames;
    const int C = (int)((T + 99) / 100);
    const int tail = (int)(T % 100 ? T % 100 : 100);
    const int P = C == 1 ? tail : 100; // longest piece
    const int M = (int)audio_token_length(P);
    const int A = (int)audio_token_length(T);

    // ggml Conv2d input is [W,H,C,N]. W=time and H=frequency. Chunk storage is
    // chunk-major, then frequency-major mel is transposed into contiguous W.
    std::vector<ggml_bf16_t> chunks((size_t)C * 128 * P, ggml_fp32_to_bf16(0));
    std::vector<int32_t> valid; valid.reserve(A);
    for (int ci = 0; ci < C; ++ci) {
        const int len = (ci == C-1) ? tail : 100;
        const int ai = (int)audio_token_length(len);
        for (int f = 0; f < 128; ++f)
            for (int t = 0; t < len; ++t)
                chunks[((size_t)ci*128 + f)*P + t] =
                    mel.data[(size_t)f*T + ci*100 + t];
        for (int t = 0; t < ai; ++t) valid.push_back(ci*M + t);
    }
    if ((int)valid.size() != A) { err = "MOSS packed length invariant failed"; return false; }

    // Use the converter-captured Torch F32 sinusoid table.


    std::vector<float> dbg_conv, dbg_l0, dbg_l31, dbg_post;
    const bool debug = debug_enabled();
    bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
        int64_t ine[4] = {P,128,1,C};
        ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_BF16, 4, ine,
            chunks.data(), chunks.size()*sizeof(chunks[0]));
        int channels[3] = {480,480,480};
        for (int i=0;i<3;++i) {
            const std::string n="enc.conv"+std::to_string(i+1);
            ggml_tensor* cw=weight(ctx,model.loader,n+".weight");
            x=conv2d_bf16(ctx,cw,x);
            x=ggml_add(ctx,f32(ctx,x),ggml_reshape_4d(ctx,f32(ctx,weight(ctx,model.loader,n+".bias")),1,1,channels[i],1));
            x=exact_gelu(ctx,bf16(ctx,x)); // conv boundary, then exact GELU boundary
        }
        // [M,16,480,C] -> contiguous [16,480,M,C], flatten each time row.
        x=ggml_cont(ctx,ggml_permute(ctx,x,2,0,1,3));
        x=ggml_reshape_2d(ctx,x,16*480,(int64_t)M*C);
        x=linear(ctx,model.loader,x,"enc.conv_out",false);

        ggml_tensor* pet=weight(ctx,model.loader,"enc.positional_embedding");
        ggml_tensor* pe=ggml_view_2d(ctx,pet,ec.d_model,M,pet->nb[1],0);
        x=add_bf16(ctx,x,bf16(ctx,pe));
        int64_t vne[1]={A};
        ggml_tensor* vi=graph_input_tensor(ctx,GGML_TYPE_I32,1,vne,valid.data(),valid.size()*sizeof(int32_t));
        x=bf16(ctx,ggml_get_rows(ctx,x,vi));
        if(debug) capture_graph_output(f32(ctx,x),&dbg_conv);

        const int W=M*((int)ec.n_window_infer/100);
        const int H=(int)ec.n_heads, D=(int)ec.head_dim;
        const float scale=1.0f/std::sqrt((float)D);
        for(int li=0;li<(int)ec.n_layers;++li) {
            const std::string pre="enc.blk."+std::to_string(li)+".";
            ggml_tensor* r=x;
            ggml_tensor* n=layer_norm(ctx,model.loader,x,pre+"attn_norm",ec.layer_norm_eps);
            ggml_tensor* q=linear(ctx,model.loader,n,pre+"attn.q",true);
            ggml_tensor* k=linear(ctx,model.loader,n,pre+"attn.k",true);
            ggml_tensor* v=linear(ctx,model.loader,n,pre+"attn.v",true);
            ggml_tensor* joined=nullptr;
            for(int begin=0;begin<A;begin+=W) {
                const int S=std::min(W,A-begin);
                auto window=[&](ggml_tensor* z) {
                    ggml_tensor* vw=ggml_view_2d(ctx,z,ec.d_model,S,z->nb[1],(size_t)begin*z->nb[1]);
                    vw=ggml_reshape_3d(ctx,vw,D,H,S);
                    return ggml_cont(ctx,ggml_permute(ctx,vw,0,2,1,3)); // [D,S,H]
                };
                ggml_tensor *qw=window(q),*kw=window(k),*vw=window(v);
                // BF16 QK GEMM result and BF16 scalar multiply boundaries,
                // followed by F32 softmax and BF16 probabilities.
                ggml_tensor* scores=bf16(ctx,ggml_mul_mat(ctx,kw,qw));
                scores=bf16(ctx,ggml_scale(ctx,f32(ctx,scores),scale));
                ggml_tensor* prob=ggml_soft_max_ext(ctx,f32(ctx,scores),nullptr,1.0f,0.0f);
                prob=bf16(ctx,prob);
                ggml_tensor* vt=ggml_cont(ctx,ggml_permute(ctx,vw,1,0,2,3)); // [S,D,H]
                ggml_tensor* co=bf16(ctx,ggml_mul_mat(ctx,vt,prob)); // [D,S,H]
                co=ggml_cont(ctx,ggml_permute(ctx,co,0,2,1,3));
                co=ggml_reshape_2d(ctx,co,ec.d_model,S);
                joined=joined?bf16(ctx,ggml_concat(ctx,f32(ctx,joined),f32(ctx,co),1)):co;
            }
            ggml_tensor* a=linear(ctx,model.loader,joined,pre+"attn.o",true);
            x=add_bf16(ctx,r,a);
            r=x;
            n=layer_norm(ctx,model.loader,x,pre+"ffn_norm",ec.layer_norm_eps);
            ggml_tensor* h=linear(ctx,model.loader,n,pre+"ffn.fc1",true);
            h=exact_gelu(ctx,h);
            h=linear(ctx,model.loader,h,pre+"ffn.fc2",true);
            x=add_bf16(ctx,r,h);
            if(debug&&li==0)capture_graph_output(f32(ctx,x),&dbg_l0);
            if(debug&&li==31)capture_graph_output(f32(ctx,x),&dbg_l31);
        }
        x=layer_norm(ctx,model.loader,x,"enc.ln_post",ec.layer_norm_eps);
        if(debug)capture_graph_output(f32(ctx,x),&dbg_post);
        x=linear(ctx,model.loader,x,"enc.proj1",true);
        x=exact_gelu(ctx,x);
        return f32(ctx,linear(ctx,model.loader,x,"enc.proj2",true));
    },out.data);
    if(!ok){err="MOSS encoder graph execution failed";return false;}
    out.n_tokens=A; out.width=ec.output_dim;
    if(debug) {
        auto report=[](const char* n,const std::vector<float>& v){double mx=0;for(float x:v)mx=std::max(mx,std::abs((double)x));std::fprintf(stderr,"MOSS_DEBUG %-12s elements=%zu max_abs_value=%.9g (stage golden unavailable)\n",n,v.size(),mx);};
        report("post-conv",dbg_conv);report("post-layer0",dbg_l0);report("post-layer31",dbg_l31);report("post-ln_post",dbg_post);
    }
    return true;
}
} // namespace starling::ggml::moss
