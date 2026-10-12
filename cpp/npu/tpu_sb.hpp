// Copyright 2024 Google LLC.
// Copyright 2026 The Starling Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// The C interface of the Pixel's on-device TPU runtime as reached through the
// public firmware library libedgetpu_litert.so (listed in
// /vendor/etc/public.libraries.txt): the "SouthBound" (thr*) execution API and
// the on-device-compile (EdgeTpuCompiler*) API. Only the declarations this
// client calls are reproduced, with their ABI as used by LiteRT's Apache-2.0
// glue (litert/vendors/google_tensor/dispatch/sb_api.h and
// litert/vendors/google_tensor/edgetpu_compiler_api.h). Nothing here is part of
// the Google Tensor SDK; everything resolves at run time from the phone's OS.
#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

typedef struct ThrContext ThrContext;
typedef struct ThrGraph ThrGraph;
typedef struct ThrInvocationContext ThrInvocationContext;
typedef uint64_t ThrSqContainerHandle;
typedef uint64_t ThrBufferHandle;
typedef const char* ThrEdgeId;
typedef const char* ThrNodeId;

enum ThrStatus : int { kThrStatusSuccess = 0, kThrStatusFail = -1 };
enum ThrEdgeType : int { kThrEdgeTypeTensor = 1 };
enum ThrNodeType : int { kThrNodeTypeNpu = 2 };
enum ThrSqContainerType : int { kThrSqContainerTypeMlModel = 2 };
enum ThrBufferType : int { kThrBufferTypeHostMemory = 0, kThrBufferTypeAHardwareBuffer = 1 };

typedef struct EdgeTpuCompilerContext EdgeTpuCompilerContext;

}  // extern "C"

namespace starling::npu {

// Function table resolved from libedgetpu_litert.so. Optional entries may be
// null on older firmware.
struct SouthBound {
    ThrStatus (*initialize)();
    const char* (*get_vendor_api_version)();
    ThrContext* (*context_create)();
    ThrStatus (*context_delete)(ThrContext*);
    ThrStatus (*set_attr_str)(ThrContext*, const char*, const char*);
    ThrGraph* (*graph_create)(ThrContext*);
    ThrStatus (*graph_delete)(ThrGraph*);
    ThrStatus (*graph_add_edge)(ThrGraph*, ThrEdgeId, ThrEdgeType);
    ThrStatus (*graph_add_sq_node)(ThrGraph*, ThrNodeId, ThrNodeType);
    ThrStatus (*graph_connect_node_input)(ThrGraph*, ThrNodeId, ThrEdgeId);
    ThrStatus (*graph_connect_node_output)(ThrGraph*, ThrNodeId, ThrEdgeId);
    ThrStatus (*graph_set_input_edge)(ThrGraph*, ThrEdgeId);
    ThrStatus (*graph_set_output_edge)(ThrGraph*, ThrEdgeId);
    ThrStatus (*graph_assign_sq)(ThrGraph*, ThrNodeId, ThrSqContainerHandle, const char*);
    ThrStatus (*load_sq_container)(ThrContext*, ThrSqContainerType, const void*, size_t, ThrSqContainerHandle*);
    ThrStatus (*load_sq_container_fd)(ThrContext*, ThrSqContainerType, int, size_t, bool, ThrSqContainerHandle*);
    ThrStatus (*load_sq_container_fd_with_offset)(ThrContext*, ThrSqContainerType, int, size_t, size_t, bool,
                                                  ThrSqContainerHandle*);
    ThrStatus (*unload_sq_container)(ThrContext*, ThrSqContainerHandle);
    ThrStatus (*register_buffer_with_offset)(ThrContext*, ThrBufferType, void*, size_t, size_t, ThrBufferHandle*);
    ThrStatus (*unregister_buffer)(ThrContext*, ThrBufferHandle);
    ThrInvocationContext* (*invocation_context_get)(ThrGraph*, ThrContext*);
    ThrStatus (*invocation_context_delete)(ThrGraph*, ThrInvocationContext*);
    ThrStatus (*invocation_context_attach_buffer)(ThrInvocationContext*, ThrContext*, ThrEdgeId, ThrBufferHandle);
    ThrStatus (*invocation_context_prepare_for_invoke)(ThrInvocationContext*, bool);
    ThrStatus (*invocation_context_invoke_once)(ThrInvocationContext*);
    ThrStatus (*invocation_context_wait)(ThrInvocationContext*);

    int (*compiler_create)(EdgeTpuCompilerContext**);
    int (*compiler_compile_flatbuffer)(EdgeTpuCompilerContext*, const char*, size_t, const char*, size_t, char***,
                                       size_t**, size_t*, char**);
    void (*compiler_free_compiled_code)(EdgeTpuCompilerContext*, char**, size_t*, size_t);
    void (*compiler_free_error_message)(EdgeTpuCompilerContext*, char*);
    void (*compiler_destroy)(EdgeTpuCompilerContext*);
};

}  // namespace starling::npu
