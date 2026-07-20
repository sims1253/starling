// tdt_multistep.hpp — K-step CUDA-graph-captured TDT greedy decode (Wave C).
//
// Starling port of parakeet.cpp's tdt_multistep.{hpp,cpp}: capture K consecutive
// TDT decode steps (prediction LSTM step + joint + argmax + duration-advance +
// blank-skip chaining) into ONE ReplayGraph and sync to host ONCE per K steps
// instead of once per step. T-aware K (16 for T<=512, 96 for long). Termination
// check on the host after each replay from a single readback.
//
// ADAPTATION vs parakeet.cpp: the reference returns the NON-BLANK hyp (it drops
// blanks). Starling's golden id stream INCLUDES blanks (see tdt.cpp's header),
// so this port emits EVERY token (including blanks) -- the host scatter always
// appends the step's token. (Verified: blank+dur0 -- the one case where the
// in-graph "force dur=1" would change the blank count vs the serial loop's
// max_symbols guard -- does NOT occur on the short/medium/long goldens, max
// blank run is 2; so emitting all tokens reproduces the serial loop's id stream
// byte-for-byte. See plans/wave-c-parakeet-gpu-decode.md req #1.)
//
// Returns std::nullopt when the multistep path is unavailable (CPU backend, or
// graph capture failure) so the caller (tdt_greedy) falls back to the serial
// loop. A returned (possibly empty) vector is a successful decode.

#pragma once

#include "joint.hpp"
#include "prediction.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace starling::ggml::parakeet {

// K-step CUDA-graph-captured TDT greedy decode. See the file header.
//
//   pred:      prediction net (LSTM).
//   joint:     joint net.
//   enc_proj:  precomputed joint.enc projection over all T frames, row-major
//              [T, joint_hidden] (enc_proj[t*H + h], frame t contiguous). This
//              IS the encoder Phase 1b output (verified byte-exact vs golden).
//   T:         number of encoder frames.
//   durations: TDT duration classes (e.g. [0,1,2,3,4]).
//   blank_id:  blank token id (= vocab_size).
//   max_symbols: cap on consecutive dur-0 emits at a single frame.
//
// Returns the emitted token id stream INCLUDING blanks (matches the serial
// tdt_greedy output byte-for-byte) on success, or std::nullopt if the multistep
// path is unavailable (CPU / capture failure) -- the caller falls back.
std::optional<std::vector<int32_t>> tdt_greedy_multistep(
    const PredictionNet& pred, const Joint& joint,
    const std::vector<float>& enc_proj, int T,
    const std::vector<int32_t>& durations,
    int blank_id, int max_symbols);

// Drop every cached K-step decode graph (each holds a ReplayGraph that owns a
// device buffer + a captured CUDA graph). Registered with
// register_decode_cache_clearer() so shutdown_backend() frees these WHILE the
// CUDA driver is still alive (rather than by static destruction at process
// exit, which runs after the driver's own atexit handler and aborts). Safe to
// call when no graph was ever built (no-op) and idempotent.
void clear_kstep_cache();

} // namespace starling::ggml::parakeet
