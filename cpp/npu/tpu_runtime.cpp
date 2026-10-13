// Copyright 2026 The Starling Authors.
// Portions derived from LiteRT's litert/vendors/google_tensor glue
// (adapter_odc.cc, dispatch/litert_dispatch_device_context.cc,
// dispatch/litert_dispatch_graph.cc, dispatch/litert_dispatch_invocation_context.cc,
// dispatch/sb_api_late_binding.cc), Copyright 2024-2025 Google LLC.
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
#include "tpu_runtime.hpp"

#include "encoder_contract.hpp"

#include "tpu_sb.hpp"

#include <android/hardware_buffer.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <mutex>

namespace starling::npu {

namespace {

SouthBound g_sb;
bool g_loaded = false;
std::string g_load_err;
std::once_flag g_once;

template <typename F>
bool sym(void* h, const char* name, F& out, bool required, std::string& missing) {
    out = reinterpret_cast<F>(dlsym(h, name));
    if (!out && required) missing += std::string(missing.empty() ? "" : ", ") + name;
    return out != nullptr;
}

void do_load() {
    // By soname: the linker namespace resolves it through
    // /vendor/etc/public.libraries.txt, which is what makes it reachable from
    // an untrusted app (an absolute /vendor path is not).
    void* h = dlopen("libedgetpu_litert.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        g_load_err = std::string("dlopen libedgetpu_litert.so: ") + dlerror();
        return;
    }
    std::string miss;
    SouthBound& s = g_sb;
    sym(h, "thrInitialize", s.initialize, true, miss);
    sym(h, "thrGetVendorApiVersion", s.get_vendor_api_version, false, miss);
    sym(h, "thrContextCreate", s.context_create, true, miss);
    sym(h, "thrContextDelete", s.context_delete, true, miss);
    sym(h, "thrVendorSetSystemAttributeStr", s.set_attr_str, false, miss);
    sym(h, "thrGraphCreate", s.graph_create, true, miss);
    sym(h, "thrGraphDelete", s.graph_delete, true, miss);
    sym(h, "thrGraphAddEdge", s.graph_add_edge, true, miss);
    sym(h, "thrGraphAddSqNode", s.graph_add_sq_node, true, miss);
    sym(h, "thrGraphConnectNodeInput", s.graph_connect_node_input, true, miss);
    sym(h, "thrGraphConnectNodeOutput", s.graph_connect_node_output, true, miss);
    sym(h, "thrGraphSetInputEdge", s.graph_set_input_edge, true, miss);
    sym(h, "thrGraphSetOutputEdge", s.graph_set_output_edge, true, miss);
    sym(h, "thrGraphAssignSq", s.graph_assign_sq, true, miss);
    sym(h, "thrLoadSqContainer", s.load_sq_container, true, miss);
    sym(h, "thrUnloadSqContainer", s.unload_sq_container, true, miss);
    sym(h, "thrRegisterBufferWithOffset", s.register_buffer_with_offset, true, miss);
    sym(h, "thrUnregisterBuffer", s.unregister_buffer, true, miss);
    sym(h, "thrInvocationContextGet", s.invocation_context_get, true, miss);
    sym(h, "thrInvocationContextDelete", s.invocation_context_delete, true, miss);
    sym(h, "thrInvocationContextAttachBuffer", s.invocation_context_attach_buffer, true, miss);
    sym(h, "thrInvocationContextPrepareForInvoke", s.invocation_context_prepare_for_invoke, true, miss);
    sym(h, "thrInvocationContextInvokeOnce", s.invocation_context_invoke_once, true, miss);
    sym(h, "thrInvocationContextWait", s.invocation_context_wait, true, miss);
    // On-device compiler (optional: only needed to build an artifact).
    sym(h, "EdgeTpuCompilerCreate", s.compiler_create, false, miss);
    sym(h, "EdgeTpuCompilerCompileFlatbuffer", s.compiler_compile_flatbuffer, false, miss);
    sym(h, "EdgeTpuCompilerFreeCompiledCode", s.compiler_free_compiled_code, false, miss);
    sym(h, "EdgeTpuCompilerFreeErrorMessage", s.compiler_free_error_message, false, miss);
    sym(h, "EdgeTpuCompilerDestroy", s.compiler_destroy, false, miss);
    if (!miss.empty()) {
        g_load_err = "libedgetpu_litert.so lacks: " + miss;
        return;
    }
    if (s.initialize() != kThrStatusSuccess) {
        g_load_err = "thrInitialize failed";
        return;
    }
    g_loaded = true;
}

const char* edge_name(std::vector<std::string>& names, size_t i) {
    return names[i].c_str();
}

// Reads the header of the open artifact `fd` and checks it against the size
// of that same file.
bool read_artifact_fd(int fd, const std::string& path, ArtifactInfo& info, std::string& err) {
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        err = "cannot stat " + path;
        return false;
    }
    ArtifactHeader h{};
    const ssize_t got = ::pread(fd, &h, sizeof h, 0);
    if (!parse_artifact_header(&h, got > 0 ? (size_t)got : 0, (uint64_t)st.st_size, info, err)) {
        err += ": " + path;
        return false;
    }
    return true;
}

}  // namespace

const SouthBound* load_southbound(std::string& err) {
    std::call_once(g_once, do_load);
    if (!g_loaded) {
        err = g_load_err;
        return nullptr;
    }
    return &g_sb;
}

std::string vendor_version() {
    std::string e;
    const SouthBound* s = load_southbound(e);
    if (!s || !s->get_vendor_api_version) return "";
    const char* v = s->get_vendor_api_version();
    return v ? v : "";
}

uint64_t fnv1a(const void* data, size_t size) { return fnv1a_update(kFnvBasis, data, size); }

bool compile_tflite(const void* tflite, size_t size, std::vector<uint8_t>& bytecode, std::string& err) {
    const SouthBound* s = load_southbound(err);
    if (!s) return false;
    if (!s->compiler_create || !s->compiler_compile_flatbuffer) {
        err = "this firmware has no on-device compiler (EdgeTpuCompiler*)";
        return false;
    }
    EdgeTpuCompilerContext* cc = nullptr;
    if (s->compiler_create(&cc) != 0 || !cc) {
        err = "EdgeTpuCompilerCreate failed";
        return false;
    }
    // Default EdgeTpuCompilerOptions: an empty serialized proto (what LiteRT's
    // ODC adapter sends today; it does not forward any options yet).
    static const char kNoOptions[1] = {0};
    char** codes = nullptr;
    size_t* sizes = nullptr;
    size_t n = 0;
    char* msg = nullptr;
    int rc = s->compiler_compile_flatbuffer(cc, static_cast<const char*>(tflite), size, kNoOptions, 0, &codes,
                                            &sizes, &n, &msg);
    bool ok = rc == 0 && n == 1 && codes && sizes;
    if (!ok) {
        err = "EdgeTpuCompilerCompileFlatbuffer rc=" + std::to_string(rc) + " bytecodes=" + std::to_string(n) +
              (msg ? std::string(": ") + msg : std::string());
    } else {
        bytecode.assign(reinterpret_cast<uint8_t*>(codes[0]), reinterpret_cast<uint8_t*>(codes[0]) + sizes[0]);
    }
    if (msg && s->compiler_free_error_message) s->compiler_free_error_message(cc, msg);
    if (codes && s->compiler_free_compiled_code) s->compiler_free_compiled_code(cc, codes, sizes, n);
    if (s->compiler_destroy) s->compiler_destroy(cc);
    return ok;
}

bool write_artifact(const std::string& path, const ArtifactInfo& info, const std::vector<uint8_t>& bytecode,
                    std::string& err) {
    if (info.in_bytes.size() > 64 || info.out_bytes.size() > 64 || info.function.size() >= 64 ||
        info.key.size() >= 256 || bytecode.empty()) {
        err = "artifact: too many I/Os, function name or key too long, or no bytecode";
        return false;
    }
    ArtifactHeader h{};
    std::memcpy(h.magic, kArtifactMagic, sizeof h.magic);
    h.n_in = (uint32_t)info.in_bytes.size();
    h.n_out = (uint32_t)info.out_bytes.size();
    for (size_t i = 0; i < info.in_bytes.size(); ++i) h.in_bytes[i] = info.in_bytes[i];
    for (size_t i = 0; i < info.out_bytes.size(); ++i) h.out_bytes[i] = info.out_bytes[i];
    std::snprintf(h.function, sizeof h.function, "%s", info.function.c_str());
    std::snprintf(h.vendor, sizeof h.vendor, "%s", info.vendor.c_str());
    std::snprintf(h.key, sizeof h.key, "%s", info.key.c_str());
    h.model_hash = info.model_hash;
    h.tflite_hash = info.tflite_hash;
    h.bytecode_offset = kArtifactHeaderBytes;
    h.bytecode_size = bytecode.size();
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        err = "cannot write " + tmp;
        return false;
    }
    std::vector<uint8_t> head(kArtifactHeaderBytes, 0);
    std::memcpy(head.data(), &h, sizeof h);
    bool ok = std::fwrite(head.data(), 1, head.size(), f) == head.size() &&
              std::fwrite(bytecode.data(), 1, bytecode.size(), f) == bytecode.size();
    ok = (std::fflush(f) == 0) && ok;
    if (ok) ::fsync(fileno(f));
    std::fclose(f);
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "write failed: " + path;
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool read_artifact_info(const std::string& path, ArtifactInfo& info, std::string& err) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        err = "cannot open " + path;
        return false;
    }
    const bool ok = read_artifact_fd(fd, path, info, err);
    ::close(fd);
    return ok;
}

TpuModel::~TpuModel() { close(); }

void TpuModel::close() {
    const SouthBound* s = sb_;
    if (!s) return;
    auto* ctx = static_cast<ThrContext*>(ctx_);
    auto* graph = static_cast<ThrGraph*>(graph_);
    if (ictx_) s->invocation_context_delete(graph, static_cast<ThrInvocationContext*>(ictx_));
    ictx_ = nullptr;
    for (auto* v : {&in_, &out_}) {
        for (Buf& b : *v) {
            if (b.handle) s->unregister_buffer(ctx, b.handle);
            if (b.ahwb) AHardwareBuffer_release(b.ahwb);
        }
        v->clear();
    }
    if (graph) s->graph_delete(graph);
    graph_ = nullptr;
    if (sq_loaded_) s->unload_sq_container(ctx, sq_);
    sq_loaded_ = false;
    if (ctx) s->context_delete(ctx);
    ctx_ = nullptr;
    if (map_) munmap(map_, map_len_);
    map_ = nullptr;
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

bool TpuModel::alloc(Buf& b, size_t bytes, std::string& err) {
    AHardwareBuffer_Desc d{};
    d.width = (uint32_t)bytes;
    d.height = 1;
    d.layers = 1;
    d.format = AHARDWAREBUFFER_FORMAT_BLOB;
    d.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
              AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    if (AHardwareBuffer_allocate(&d, &b.ahwb) != 0) {
        d.usage &= ~AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;   // GPU usage is optional
        if (AHardwareBuffer_allocate(&d, &b.ahwb) != 0) {
            err = "AHardwareBuffer_allocate(" + std::to_string(bytes) + ") failed";
            return false;
        }
    }
    b.bytes = bytes;
    if (sb_->register_buffer_with_offset(static_cast<ThrContext*>(ctx_), kThrBufferTypeAHardwareBuffer, b.ahwb, 0,
                                         bytes, &b.handle) != kThrStatusSuccess) {
        err = "thrRegisterBufferWithOffset(AHWB) failed";
        return false;
    }
    return true;
}

bool TpuModel::open(const std::string& path, std::string& err) {
    close();
    sb_ = load_southbound(err);
    if (!sb_) return false;
    const SouthBound* s = sb_;
    // Header and size come from the one fd that gets mapped, so a truncated
    // (or swapped) file is rejected here instead of faulting in the firmware.
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        err = "cannot open " + path;
        return false;
    }
    fd_ = fd;
    if (!read_artifact_fd(fd, path, info_, err)) return false;
    // The bytecode stays in the mapped file (no copy).
    map_len_ = (size_t)(info_.bytecode_offset + info_.bytecode_size);
    map_ = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        err = "mmap failed";
        return false;
    }

    auto* ctx = s->context_create();
    if (!ctx) {
        err = "thrContextCreate failed";
        return false;
    }
    ctx_ = ctx;
    // Same context attributes as LiteRT's dispatch device context.
    if (s->set_attr_str) {
        s->set_attr_str(ctx, "edgetpu_use_tpu_tachyon", "1");
        s->set_attr_str(ctx, "sq_container_load_method", "dmabuf");
    }

    ThrSqContainerHandle sq = 0;
    const ThrStatus st =
        s->load_sq_container(ctx, kThrSqContainerTypeMlModel, static_cast<const uint8_t*>(map_) + info_.bytecode_offset,
                             info_.bytecode_size, &sq);
    if (st != kThrStatusSuccess) {
        err = "loading the SQ container failed";
        return false;
    }
    sq_ = sq;
    sq_loaded_ = true;

    // One NPU node, positional edges: inputs edge_0..n-1, outputs after them
    // (the layout LiteRT's CreateFromBytecode builds).
    auto* graph = s->graph_create(ctx);
    if (!graph) {
        err = "thrGraphCreate failed";
        return false;
    }
    graph_ = graph;
    static const char* kNode = "node_0";
    if (s->graph_add_sq_node(graph, kNode, kThrNodeTypeNpu) != kThrStatusSuccess ||
        s->graph_assign_sq(graph, kNode, sq, info_.function.empty() ? nullptr : info_.function.c_str()) !=
            kThrStatusSuccess) {
        err = "graph node / assign SQ failed (function '" + info_.function + "')";
        return false;
    }
    const size_t n_in = info_.in_bytes.size(), n_out = info_.out_bytes.size();
    edge_names_.clear();
    for (size_t i = 0; i < n_in + n_out; ++i) edge_names_.push_back("edge_" + std::to_string(i));
    for (size_t i = 0; i < n_in + n_out; ++i) {
        const char* e = edge_name(edge_names_, i);
        const bool is_in = i < n_in;
        if (s->graph_add_edge(graph, e, kThrEdgeTypeTensor) != kThrStatusSuccess ||
            (is_in ? s->graph_connect_node_input(graph, kNode, e) : s->graph_connect_node_output(graph, kNode, e)) !=
                kThrStatusSuccess ||
            (is_in ? s->graph_set_input_edge(graph, e) : s->graph_set_output_edge(graph, e)) != kThrStatusSuccess) {
            err = "graph edge setup failed";
            return false;
        }
    }
    auto* ic = s->invocation_context_get(graph, ctx);
    if (!ic) {
        err = "thrInvocationContextGet failed";
        return false;
    }
    ictx_ = ic;
    in_.resize(n_in);
    out_.resize(n_out);
    for (size_t i = 0; i < n_in; ++i)
        if (!alloc(in_[i], info_.in_bytes[i], err)) return false;
    for (size_t i = 0; i < n_out; ++i)
        if (!alloc(out_[i], info_.out_bytes[i], err)) return false;
    for (size_t i = 0; i < n_in + n_out; ++i) {
        const Buf& b = i < n_in ? in_[i] : out_[i - n_in];
        if (s->invocation_context_attach_buffer(ic, ctx, edge_name(edge_names_, i), b.handle) != kThrStatusSuccess) {
            err = "attach buffer to " + edge_names_[i] + " failed";
            return false;
        }
    }
    return true;
}

bool TpuModel::write_input(size_t i, const void* data, size_t bytes, std::string& err) {
    if (i >= in_.size() || bytes > in_[i].bytes) {
        err = "write_input: bad index/size";
        return false;
    }
    void* p = nullptr;
    if (AHardwareBuffer_lock(in_[i].ahwb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &p) != 0) {
        err = "AHardwareBuffer_lock failed";
        return false;
    }
    std::memcpy(p, data, bytes);
    AHardwareBuffer_unlock(in_[i].ahwb, nullptr);
    return true;
}

bool TpuModel::read_output(size_t i, void* data, size_t bytes, std::string& err) {
    if (i >= out_.size() || bytes > out_[i].bytes) {
        err = "read_output: bad index/size";
        return false;
    }
    void* p = nullptr;
    if (AHardwareBuffer_lock(out_[i].ahwb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, &p) != 0) {
        err = "AHardwareBuffer_lock failed";
        return false;
    }
    std::memcpy(data, p, bytes);
    AHardwareBuffer_unlock(out_[i].ahwb, nullptr);
    return true;
}

bool TpuModel::run(std::string& err) {
    auto* ic = static_cast<ThrInvocationContext*>(ictx_);
    if (!ic) {
        err = "model not open";
        return false;
    }
    if (sb_->invocation_context_prepare_for_invoke(ic, false) != kThrStatusSuccess) {
        err = "thrInvocationContextPrepareForInvoke failed";
        return false;
    }
    if (sb_->invocation_context_invoke_once(ic) != kThrStatusSuccess) {
        err = "thrInvocationContextInvokeOnce failed";
        return false;
    }
    if (sb_->invocation_context_wait(ic) != kThrStatusSuccess) {
        err = "thrInvocationContextWait failed";
        return false;
    }
    return true;
}

}  // namespace starling::npu
