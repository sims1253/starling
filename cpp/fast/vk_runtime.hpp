// vk_runtime.hpp — the minimal Vulkan compute runtime behind the fast engines.
//
// The fast engines (cpp/fast/) are model-specialized: every kernel is written
// for the exact shapes of one model family, and the host records whole
// forward passes into command buffers that are replayed without per-op host
// work. This runtime is deliberately small: it loads libvulkan at run time
// (no link-time dependency, so a library built with the fast engine still
// loads on machines without a Vulkan driver), picks one compute queue,
// allocates buffers, builds specialized compute pipelines from the SPIR-V
// blobs compiled at build time, and records/replays command buffers.
//
// Portability rules (the target GPUs include mobile drivers of uneven
// quality): shaders use only Vulkan 1.0 core features (16-bit values are
// packed/unpacked from 32-bit words, no 8/16-bit storage, no subgroup ops);
// every storage-buffer binding stays below maxStorageBufferRange; tuning knobs
// are specialization constants, so the same SPIR-V runs everywhere.

#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace starling::fast::vk {

// Every entry point the runtime uses, resolved at load time.
#define STARLING_VK_INSTANCE_FNS(X)            \
    X(vkDestroyInstance)                       \
    X(vkEnumeratePhysicalDevices)              \
    X(vkGetPhysicalDeviceProperties)           \
    X(vkGetPhysicalDeviceProperties2)          \
    X(vkGetPhysicalDeviceFeatures2)            \
    X(vkGetPhysicalDeviceQueueFamilyProperties)\
    X(vkGetPhysicalDeviceMemoryProperties)     \
    X(vkGetPhysicalDeviceMemoryProperties2)    \
    X(vkEnumerateDeviceExtensionProperties)    \
    X(vkCreateDevice)                          \
    X(vkGetDeviceProcAddr)

#define STARLING_VK_DEVICE_FNS(X)              \
    X(vkDestroyDevice)                         \
    X(vkGetDeviceQueue)                        \
    X(vkCreateBuffer)                          \
    X(vkDestroyBuffer)                         \
    X(vkGetBufferMemoryRequirements)           \
    X(vkAllocateMemory)                        \
    X(vkFreeMemory)                            \
    X(vkBindBufferMemory)                      \
    X(vkMapMemory)                             \
    X(vkUnmapMemory)                           \
    X(vkFlushMappedMemoryRanges)               \
    X(vkInvalidateMappedMemoryRanges)          \
    X(vkCreateShaderModule)                    \
    X(vkDestroyShaderModule)                   \
    X(vkCreateDescriptorSetLayout)             \
    X(vkDestroyDescriptorSetLayout)            \
    X(vkCreatePipelineLayout)                  \
    X(vkDestroyPipelineLayout)                 \
    X(vkCreateComputePipelines)                \
    X(vkDestroyPipeline)                       \
    X(vkCreatePipelineCache)                   \
    X(vkDestroyPipelineCache)                  \
    X(vkGetPipelineCacheData)                  \
    X(vkCreateDescriptorPool)                  \
    X(vkDestroyDescriptorPool)                 \
    X(vkAllocateDescriptorSets)                \
    X(vkUpdateDescriptorSets)                  \
    X(vkCreateCommandPool)                     \
    X(vkDestroyCommandPool)                    \
    X(vkAllocateCommandBuffers)                \
    X(vkFreeCommandBuffers)                    \
    X(vkBeginCommandBuffer)                    \
    X(vkEndCommandBuffer)                      \
    X(vkCmdBindPipeline)                       \
    X(vkCmdBindDescriptorSets)                 \
    X(vkCmdPushConstants)                      \
    X(vkCmdDispatch)                           \
    X(vkCmdPipelineBarrier)                    \
    X(vkCmdCopyBuffer)                         \
    X(vkCmdFillBuffer)                         \
    X(vkCmdResetQueryPool)                     \
    X(vkCmdWriteTimestamp)                     \
    X(vkCreateQueryPool)                       \
    X(vkDestroyQueryPool)                      \
    X(vkGetQueryPoolResults)                   \
    X(vkCreateFence)                           \
    X(vkDestroyFence)                          \
    X(vkWaitForFences)                         \
    X(vkResetFences)                           \
    X(vkQueueSubmit)                           \
    X(vkDeviceWaitIdle)

struct Fns {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;
#define STARLING_VK_DECL(name) PFN_##name name = nullptr;
    STARLING_VK_INSTANCE_FNS(STARLING_VK_DECL)
    STARLING_VK_DEVICE_FNS(STARLING_VK_DECL)
#undef STARLING_VK_DECL
};

// What the engines need to know about the device to pick kernel variants.
struct DeviceInfo {
    std::string name;
    uint32_t vendor_id = 0, device_id = 0, driver_version = 0, api_version = 0;
    bool discrete = false;
    bool uma = false;                    // a device-local memory type is host-visible
    bool uma_cached = false;             // ... and CPU-cached (mapped writes are fast)
    uint32_t subgroup_size = 0;
    uint32_t max_shared_bytes = 0;       // maxComputeSharedMemorySize
    uint32_t max_wg_invocations = 0;
    uint64_t max_storage_range = 0;      // maxStorageBufferRange
    uint64_t min_storage_align = 0;      // minStorageBufferOffsetAlignment
    uint64_t max_alloc = 0;              // maxMemoryAllocationSize (0 = unknown)
    double timestamp_period_ns = 0.0;    // 0 = timestamps unsupported
    bool f16 = false;                    // shaderFloat16 enabled on the device
    bool int_dot = false;                // shaderIntegerDotProduct enabled (packed 4x8 dots)
};

class Context;

// A device buffer. Memory is bound 1:1 (the engines allocate few, large
// buffers). `host` is non-null when the memory is host-visible and mapped.
struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* host = nullptr;
    bool coherent = false;
    Context* ctx = nullptr;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept;
    ~Buffer() { release(); }
    void release();
    explicit operator bool() const { return buf != VK_NULL_HANDLE; }
};

enum class Mem {
    Device,    // device-local; host access via staging (mapped when UMA)
    DeviceOnly,// device-local, never mapped: bulk weights (staged through
               // cached memory — mapped UMA memory can be uncached for the CPU)
    Upload,    // host-visible write-combined staging
    Readback,  // host-visible, cached when available
};

// A storage-buffer range bound to one kernel binding.
struct Ref {
    const Buffer* buf = nullptr;
    VkDeviceSize off = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
    Ref() = default;
    Ref(const Buffer& b, VkDeviceSize o = 0, VkDeviceSize r = VK_WHOLE_SIZE)
        : buf(&b), off(o), range(r) {}
};

// One embedded SPIR-V module (see the generated fast_shaders.cpp).
struct ShaderBlob {
    const char* name;
    const uint32_t* words;
    size_t n_words;
    uint32_t n_bindings;
};
const ShaderBlob* find_shader(const char* name);

// A specialized compute pipeline. All kernels share one layout shape: N
// storage-buffer bindings in set 0 plus a 128-byte push-constant block.
struct Pipeline {
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    uint32_t n_bindings = 0;
    std::string label;
};

constexpr uint32_t kPushBytes = 128;

// A recorded command buffer that can be submitted repeatedly. Descriptor
// sets are allocated from the recording's own pool, so a recording stays
// valid (and replayable) until it is destroyed.
class Recording {
public:
    explicit Recording(Context& ctx);
    ~Recording();
    Recording(const Recording&) = delete;
    Recording& operator=(const Recording&) = delete;

    void begin();
    void end();
    // Close the current command buffer and continue recording in a new one.
    // Segments are submitted back to back (one fence at the end): each GPU
    // job stays short, which keeps driver watchdogs (desktop TDR, mobile GPU
    // hang detection) quiet and lets the compositor interleave its work.
    void split();

    // Record one dispatch. `push` holds up to kPushBytes of push constants.
    void dispatch(const Pipeline& p, const std::vector<Ref>& bindings,
                  const void* push, size_t push_bytes,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
    // Profiling label for the following dispatches (defaults to the shader).
    void label(const char* l) { label_ = l ? l : ""; }
    // Compute->compute (and transfer) memory barrier.
    void barrier();
    void copy(const Buffer& src, VkDeviceSize soff, const Buffer& dst,
              VkDeviceSize doff, VkDeviceSize bytes);
    void fill(const Buffer& dst, VkDeviceSize off, VkDeviceSize bytes, uint32_t value);

    // Submit and block until the GPU finishes.
    bool submit_and_wait(std::string& err);

    size_t n_dispatches() const { return n_dispatch_; }
    // Per-label GPU time of the last submission (profiling builds only;
    // enabled by STARLING_FAST_PROFILE=1 before recording).
    void report_profile(const char* title) const;

private:
    Context& ctx_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cb_ = VK_NULL_HANDLE;          // segment being recorded
    std::vector<VkCommandBuffer> segs_;             // all segments, in order
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> dpools_;
    uint32_t dpool_left_ = 0;
    std::string fail_;   // first recording error (descriptor allocation); reported by submit_and_wait
    size_t n_dispatch_ = 0;
    bool profile_ = false;
    VkQueryPool qpool_ = VK_NULL_HANDLE;
    uint32_t n_queries_ = 0, max_queries_ = 0;
    std::vector<std::string> q_labels_;
    std::string label_;
    VkDescriptorSet alloc_set(const Pipeline& p);
};

class Context {
public:
    // The process-wide context, created on first use. Returns nullptr (with
    // `err`) when no usable Vulkan device exists. STARLING_FAST_DEVICE=<n>
    // selects a physical device by index.
    static Context* get(std::string& err);
    ~Context();

    const DeviceInfo& info() const { return info_; }
    const Fns& fn() const { return fn_; }
    VkDevice device() const { return dev_; }

    bool create_buffer(Buffer& out, VkDeviceSize bytes, Mem kind, std::string& err);
    // Host <-> device copies (staged unless the buffer is mapped).
    bool upload(Buffer& dst, VkDeviceSize off, const void* src, size_t bytes, std::string& err);
    bool download(const Buffer& src, VkDeviceSize off, void* dst, size_t bytes, std::string& err);

    // Pipeline for `shader` specialized with `spec` (constant_id i = spec[i]).
    // Cached; the pointer stays valid for the context's lifetime.
    const Pipeline* pipeline(const char* shader, const std::vector<uint32_t>& spec,
                             std::string& err);

    // #325 driver-degradation guards. wedged(): a previous GPU operation
    // failed in a way that indicates the driver is in a bad state (OOM at a
    // fence, device lost, timeout); everything after fails fast with `why`.
    // A fresh (< 15 min) wedge marker from a previous process also counts,
    // so a restart loop cannot deepen the wedge. check_memory_budget(): with
    // VK_EXT_memory_budget, refuse a load cleanly when the device-local
    // heaps cannot fit `need` (+ margin).
    bool wedged() const { return wedged_; }
    const std::string& wedged_why() const { return wedged_why_; }
    void mark_wedged(const std::string& why);
    bool check_memory_budget(uint64_t need, std::string& err);

    std::mutex& queue_mutex() { return queue_mu_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return qfam_; }
    bool profiling() const { return profile_; }

private:
    Context() = default;
    bool init(std::string& err);
    bool ensure_staging(VkDeviceSize bytes, std::string& err);
    int find_memory(uint32_t type_bits, VkMemoryPropertyFlags want,
                    VkMemoryPropertyFlags avoid) const;

    Fns fn_;
    void* lib_ = nullptr;
    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkPhysicalDeviceMemoryProperties memprops_{};
    VkPipelineCache pcache_ = VK_NULL_HANDLE;
    DeviceInfo info_;
    bool profile_ = false;

    std::mutex queue_mu_;
    std::mutex pipe_mu_;
    std::map<std::string, std::unique_ptr<Pipeline>> pipes_;
    std::map<uint32_t, std::pair<VkDescriptorSetLayout, VkPipelineLayout>> layouts_;
    std::map<std::string, VkShaderModule> modules_;

    // Transfer helpers (staging buffer + a one-shot command buffer).
    Buffer staging_;
    VkCommandPool xfer_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer xfer_cb_ = VK_NULL_HANDLE;
    VkFence xfer_fence_ = VK_NULL_HANDLE;
    std::string pcache_path_;
    std::string wedge_path_;
    bool wedged_ = false;
    bool wrote_marker_ = false;
    std::string wedged_why_;
    bool mem_budget_ = false;

    friend struct Buffer;
    friend class Recording;
};

} // namespace starling::fast::vk
