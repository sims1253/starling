#include "engine_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace direct {
namespace {
float sigmoid(float x) { return 1.0f/(1.0f+std::exp(-x)); }
int argmax(const float* x,int n) {
    int best=0; for(int i=1;i<n;++i) if(x[i]>x[best]) best=i; return best;
}
} // namespace

Buffer Decoder::predict(int token,bool sos,const State& in,State& out) {
    const int H=int(model_.integer("parakeet.decoder.pred_hidden",640));
    const int L=int(model_.integer("parakeet.decoder.pred_rnn_layers",2));
    Buffer x(H,0.0f);
    if(!sos) {
        const Tensor& emb=model_.at("decoder.prediction.embed.weight");
        if(token<0 || token>=emb.rows()) throw std::runtime_error("decoder token out of range");
        for(int i=0;i<H;++i) x[i]=scalar(emb,size_t(token)*H+i);
    }
    out.h.resize(L);out.c.resize(L);
    const std::string p="decoder.prediction.dec_rnn.lstm.";
    for(int l=0;l<L;++l) {
        const std::string s="_l"+std::to_string(l);
        Buffer z=kernels_.linear(model_.at(p+"weight_ih"+s),x,1,
                                  &model_.at(p+"bias_ih"+s));
        Buffer zh=kernels_.linear(model_.at(p+"weight_hh"+s),in.h[l],1,
                                   &model_.at(p+"bias_hh"+s));
        add_inplace(z,zh);
        out.h[l].resize(H);out.c[l].resize(H);
        for(int i=0;i<H;++i) {
            float ig=sigmoid(z[i]),fg=sigmoid(z[H+i]);
            float gg=std::tanh(z[2*H+i]),og=sigmoid(z[3*H+i]);
            out.c[l][i]=fg*in.c[l][i]+ig*gg;
            out.h[l][i]=og*std::tanh(out.c[l][i]);
        }
        x=out.h[l];
    }
    return x;
}

void Decoder::joint(const float* enc,const Buffer& pred,int& token,int& duration) {
    const int H=int(model_.at("joint.pred.weight").rows());
    Buffer p=kernels_.linear(model_.at("joint.pred.weight"),pred,1,
                              &model_.at("joint.pred.bias"));
    for(int i=0;i<H;++i) p[i]=std::max(0.0f,p[i]+enc[i]);
    Buffer logits=kernels_.linear(model_.at("joint.joint_net.2.weight"),p,1,
                                   &model_.at("joint.joint_net.2.bias"));
    const int tokens=int(model_.integer("parakeet.vocab_size",8193))+1;
    token=argmax(logits.data(),tokens);
    duration=argmax(logits.data()+tokens,int(logits.size())-tokens);
}

std::vector<int32_t> Decoder::decode(const Buffer& encoder,int frames) {
    const int H=int(model_.at("joint.enc.weight").rows());
    if(encoder.size()!=size_t(frames)*H) throw std::runtime_error("encoder output shape mismatch");
    const int hidden=int(model_.integer("parakeet.decoder.pred_hidden",640));
    const int layers=int(model_.integer("parakeet.decoder.pred_rnn_layers",2));
    const int blank=int(model_.integer("parakeet.blank_id",8192));
    const int max_symbols=int(model_.integer("parakeet.decoding.max_symbols",10));
    auto durations=model_.integers("parakeet.tdt.durations");
    if(durations.empty()) throw std::runtime_error("GGUF missing TDT duration table");
    State committed; committed.h.resize(layers,Buffer(hidden,0));
    committed.c.resize(layers,Buffer(hidden,0));
    State candidate;
    Buffer pred;
    bool have_pred=false,emitted=false;
    int last=blank,t=0;
    std::vector<int32_t> ids;
    while(t<frames) {
        int added=0,skip=0;
        bool loop=true;
        while(loop && added<max_symbols) {
            if(!have_pred) {
                pred=predict(last,!emitted,committed,candidate);
                have_pred=true;
            }
            int token,duration;
            joint(encoder.data()+size_t(t)*H,pred,token,duration);
            if(duration<0||duration>=int(durations.size())) throw std::runtime_error("duration index out of range");
            skip=int(durations[size_t(duration)]);
            ids.push_back(token);
            if(token!=blank) {
                last=token; committed=std::move(candidate); emitted=true; have_pred=false;
            }
            ++added;t+=skip;loop=(skip==0);
        }
        if(skip==0) skip=1;
        if(added==max_symbols) ++t;
    }
    return ids;
}

std::string Decoder::text(const std::vector<int32_t>& ids) const {
    auto pieces=model_.strings("parakeet.tokenizer.pieces");
    std::string raw;
    for(int32_t id:ids) if(id>=0 && size_t(id)<pieces.size()) raw+=pieces[size_t(id)];
    std::string out;
    for(size_t i=0;i<raw.size();) {
        if(i+3<=raw.size() && (uint8_t)raw[i]==0xe2 &&
           (uint8_t)raw[i+1]==0x96 && (uint8_t)raw[i+2]==0x81) {
            out+=' '; i+=3;
        } else out+=raw[i++];
    }
    if(!out.empty()&&out[0]==' ') out.erase(0,1);
    return out;
}

} // namespace direct
