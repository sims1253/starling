// pos_enc.hpp — FastConformer relative positional encoding table.
//
// Mirrors NeMo's RelPositionalEncoding (multi_head_attention.py) as Starling-
// authored first-party code: for an input of length T (here T' = the subsampled
// encoder length), build pos_emb over 2T-1 relative positions running from
// +(T-1) DOWN TO -(T-1). Each row holds the Transformer-XL sinusoid
//   div_term[i] = exp(2i * -(log(10000)/d_model))
//   pe[p, 2i]   = sin(position(p) * div_term[i])
//   pe[p, 2i+1] = cos(position(p) * div_term[i])
// with position(0) = T-1 ... position(2T-2) = -(T-1).
//
// Computed host-side in double and shipped as a graph input (the same table is
// shared across all 24 conformer layers in the encoder). The relpos attention
// folds pe into the q*k^T scores via the bd skew; there is no explicit x += pe.

#pragma once

#include <vector>

namespace starling::ggml::parakeet {

// Relative positional encoding table, row-major [2T-1, d_model] (d_model
// fastest), i.e. out[p*d_model + c]. T is the encoder length T'. d_model is the
// model dim (1024 for parakeet-tdt-0.6b-v3).
void rel_pos_encoding(int T, int d_model, std::vector<float>& out);

} // namespace starling::ggml::parakeet
