// Copyright 2026 The Starling Authors. Apache-2.0 (see tpu_sb.hpp for the
// origin of the TPU runtime interface).
//
// Minimal "compile once, load fast" client for the Pixel TPU on top of the
// phone's own firmware: on-device compile of a .tflite into TPU bytecode
// (EdgeTpuCompiler*), a small artifact file around that bytecode, and
// execution of the loaded artifact through the SouthBound (thr*) API with
// AHardwareBuffer-backed I/O.
#pragma once

#include "artifact_format.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct AHardwareBuffer;

namespace starling::npu {

struct SouthBound;

// dlopen()s libedgetpu_litert.so by soname (public vendor library) and calls
// thrInitialize(). Idempotent; returns nullptr and sets err on failure.
const SouthBound* load_southbound(std::string& err);
std::string vendor_version();

// Compiles a whole .tflite (single subgraph, fully supported) on the device.
bool compile_tflite(const void* tflite, size_t size, std::vector<uint8_t>& bytecode, std::string& err);

// Artifact: a 4 KiB header followed by the bytecode (artifact_format.hpp).
bool write_artifact(const std::string& path, const ArtifactInfo& info, const std::vector<uint8_t>& bytecode,
                    std::string& err);
// Reads and validates the header (bytecode within the file, see
// parse_artifact_header).
bool read_artifact_info(const std::string& path, ArtifactInfo& info, std::string& err);

class TpuModel {
public:
    TpuModel() = default;
    ~TpuModel();
    TpuModel(const TpuModel&) = delete;
    TpuModel& operator=(const TpuModel&) = delete;

    bool open(const std::string& artifact_path, std::string& err);   // validates, then mmaps the artifact
    // Host access to the I/O buffers (AHardwareBuffer BLOBs, also importable
    // into Vulkan as external memory).
    bool write_input(size_t i, const void* data, size_t bytes, std::string& err);
    bool read_output(size_t i, void* data, size_t bytes, std::string& err);
    AHardwareBuffer* output_ahwb(size_t i) const { return out_[i].ahwb; }
    bool run(std::string& err);   // synchronous: prepare, invoke, wait
    const ArtifactInfo& info() const { return info_; }

private:
    struct Buf {
        AHardwareBuffer* ahwb = nullptr;
        uint64_t handle = 0;
        size_t bytes = 0;
    };
    bool alloc(Buf& b, size_t bytes, std::string& err);
    void close();

    const SouthBound* sb_ = nullptr;
    ArtifactInfo info_;
    void* ctx_ = nullptr;    // ThrContext*
    void* graph_ = nullptr;  // ThrGraph*
    void* ictx_ = nullptr;   // ThrInvocationContext*
    uint64_t sq_ = 0;
    bool sq_loaded_ = false;
    void* map_ = nullptr;
    size_t map_len_ = 0;
    int fd_ = -1;
    std::vector<Buf> in_, out_;
    std::vector<std::string> edge_names_;
};

}  // namespace starling::npu
