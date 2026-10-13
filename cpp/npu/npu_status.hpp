// Copyright 2026 The Starling Authors. Apache-2.0.
//
// Why a TPU piece of a loaded model runs (or does not): the engine reports
// each piece's state here. Problems the user can fix (an artifact compiled
// for older TPU firmware or by an older exporter) are logged once per process
// (logcat tag "StarlingNpu" on Android, and stderr), and every state is
// written to status-<piece>-<hash16>.txt in each existing STARLING_NPU_DIR
// directory: scripts/npu/npu_compile_all.sh reads those files to compile
// exactly what the loaded models need ("missing" / "stale").
#pragma once

#include <cstdint>
#include <string>

namespace starling::npu {

enum class PieceState { Ok, Missing, Stale, Error };

// piece: "qwen3-asr-enc".
// model_hash: the artifact's model hash (names the artifact and the status
// file); identity: the checkpoint identity the compile script checks the HF
// checkpoint against (encoder_contract.hpp).
// model: what the compile script needs to pick the checkpoint, as
// space-separated key=value pairs (e.g. "arch=qwen3 hd=1024").
void report_piece(const std::string& piece, uint64_t model_hash, uint64_t identity, const std::string& model,
                  PieceState state, const std::string& message);

// Whether this device has the TPU runtime, without loading it: the firmware
// lists libedgetpu_litert.so among its public vendor libraries.
bool tpu_present();

// STARLING_NPU_DIR, else STARLING_FAST_CACHE_DIR (':'-separated).
std::string npu_dirs();

}  // namespace starling::npu
