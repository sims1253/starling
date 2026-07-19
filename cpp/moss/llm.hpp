#pragma once
#include "prompt.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::moss {
struct LayerKvCache { std::vector<ggml_bf16_t> k,v; };
struct LlmState { std::vector<LayerKvCache> layers; int64_t length=0; };
struct PrefillResult { std::vector<float> logits; int32_t first_token=-1; LlmState state; };
struct GenerateOptions { int32_t max_new_tokens=200,max_cache_len=2048,eos_token_id=151645; };
struct GenerateResult { std::vector<int32_t> ids; bool hit_eos=false; std::vector<float> prefill_logits; };
bool llm_prefill(const MossModel&,const InputsEmbeds&,int32_t max_cache_len,PrefillResult&,std::string&);
bool greedy_generate(const MossModel&,const InputsEmbeds&,const GenerateOptions&,GenerateResult&,std::string&);
}
