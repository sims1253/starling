// vk_runtime.cpp — see vk_runtime.hpp.

#include "vk_runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace starling::fast::vk {

namespace {

void* open_vulkan_library() {
#if defined(_WIN32)
    return (void*)LoadLibraryA("vulkan-1.dll");
#elif defined(__APPLE__)
    const char* names[] = {"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"};
    for (const char* n : names)
        if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
    return nullptr;
#else
    const char* names[] = {"libvulkan.so.1", "libvulkan.so"};
    for (const char* n : names)
        if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
    return nullptr;
#endif
}

void* lib_symbol(void* lib, const char* name) {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

std::string vk_err(const char* what, VkResult r) {
    return std::string(what) + " failed (VkResult " + std::to_string((int)r) + ")";
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] && v[0] != '0';
}

} // namespace

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        release();
        buf = o.buf; mem = o.mem; size = o.size; host = o.host;
        coherent = o.coherent; ctx = o.ctx;
        o.buf = VK_NULL_HANDLE; o.mem = VK_NULL_HANDLE; o.size = 0;
        o.host = nullptr; o.ctx = nullptr;
    }
    return *this;
}

void Buffer::release() {
    if (!ctx) return;
    const Fns& f = ctx->fn_;
    if (host) f.vkUnmapMemory(ctx->dev_, mem);
    if (buf) f.vkDestroyBuffer(ctx->dev_, buf, nullptr);
    if (mem) f.vkFreeMemory(ctx->dev_, mem, nullptr);
    buf = VK_NULL_HANDLE; mem = VK_NULL_HANDLE; host = nullptr; size = 0; ctx = nullptr;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

Context* Context::get(std::string& err) {
    static std::mutex mu;
    static std::unique_ptr<Context> ctx;
    static bool tried = false;
    static std::string init_err;
    std::lock_guard<std::mutex> lk(mu);
    if (!tried) {
        tried = true;
        std::unique_ptr<Context> c(new Context());
        if (c->init(init_err)) ctx = std::move(c);
    }
    if (!ctx) err = init_err.empty() ? "Vulkan unavailable" : init_err;
    return ctx.get();
}

bool Context::init(std::string& err) {
    lib_ = open_vulkan_library();
    if (!lib_) { err = "Vulkan loader library not found"; return false; }
    fn_.vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)lib_symbol(lib_, "vkGetInstanceProcAddr");
    if (!fn_.vkGetInstanceProcAddr) { err = "vkGetInstanceProcAddr missing"; return false; }
    fn_.vkCreateInstance =
        (PFN_vkCreateInstance)fn_.vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    fn_.vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)
        fn_.vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
    uint32_t inst_ver = VK_API_VERSION_1_0;
    if (fn_.vkEnumerateInstanceVersion) fn_.vkEnumerateInstanceVersion(&inst_ver);
    if (inst_ver < VK_API_VERSION_1_1) { err = "Vulkan 1.1 instance required"; return false; }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "starling-fast";
    app.pEngineName = "starling-fast";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkResult r = fn_.vkCreateInstance(&ici, nullptr, &inst_);
    if (r != VK_SUCCESS) { err = vk_err("vkCreateInstance", r); return false; }
#define STARLING_VK_LOAD_I(name)                                                   \
    fn_.name = (PFN_##name)fn_.vkGetInstanceProcAddr(inst_, #name);                \
    if (!fn_.name) { err = "missing Vulkan entry point " #name; return false; }
    STARLING_VK_INSTANCE_FNS(STARLING_VK_LOAD_I)
#undef STARLING_VK_LOAD_I

    uint32_t n_phys = 0;
    fn_.vkEnumeratePhysicalDevices(inst_, &n_phys, nullptr);
    if (n_phys == 0) { err = "no Vulkan physical devices"; return false; }
    std::vector<VkPhysicalDevice> phys(n_phys);
    fn_.vkEnumeratePhysicalDevices(inst_, &n_phys, phys.data());

    // Device choice: explicit index, else the first discrete GPU, else the
    // first integrated GPU, else device 0. CPU (software) devices are only
    // used when explicitly requested: they are never faster than the CPU path.
    int pick = -1;
    if (const char* e = std::getenv("STARLING_FAST_DEVICE")) {
        int i = std::atoi(e);
        if (i >= 0 && i < (int)n_phys) pick = i;
    }
    if (pick < 0) {
        int best_score = -1;
        for (uint32_t i = 0; i < n_phys; ++i) {
            VkPhysicalDeviceProperties p;
            fn_.vkGetPhysicalDeviceProperties(phys[i], &p);
            if (p.apiVersion < VK_API_VERSION_1_1) continue;
            int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
                      : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                      : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 1 : -1;
            if (score > best_score) { best_score = score; pick = (int)i; }
        }
        if (pick < 0) { err = "no Vulkan 1.1 GPU found"; return false; }
    }
    phys_ = phys[(size_t)pick];

    VkPhysicalDeviceSubgroupProperties sgp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    p2.pNext = &sgp;
    sgp.pNext = &m3;
    fn_.vkGetPhysicalDeviceProperties2(phys_, &p2);
    const VkPhysicalDeviceProperties& props = p2.properties;
    info_.name = props.deviceName;
    info_.vendor_id = props.vendorID;
    info_.device_id = props.deviceID;
    info_.driver_version = props.driverVersion;
    info_.api_version = props.apiVersion;
    info_.discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    info_.subgroup_size = sgp.subgroupSize;
    info_.max_shared_bytes = props.limits.maxComputeSharedMemorySize;
    info_.max_wg_invocations = props.limits.maxComputeWorkGroupInvocations;
    info_.max_storage_range = props.limits.maxStorageBufferRange;
    info_.min_storage_align = props.limits.minStorageBufferOffsetAlignment;
    info_.max_alloc = m3.maxMemoryAllocationSize;

    uint32_t n_qf = 0;
    fn_.vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n_qf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n_qf);
    fn_.vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n_qf, qf.data());
    int qpick = -1;
    // Prefer a dedicated compute family where one exists (desktop GPUs: it
    // runs beside the compositor instead of stalling it), else the universal
    // queue every conformant driver exposes (mobile GPUs).
    if (!env_on("STARLING_FAST_UNIVERSAL_QUEUE"))
        for (uint32_t i = 0; i < n_qf; ++i)
            if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                qpick = (int)i; break;
            }
    if (qpick < 0)
        for (uint32_t i = 0; i < n_qf; ++i)
            if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                qpick = (int)i; break;
            }
    if (qpick < 0)
        for (uint32_t i = 0; i < n_qf; ++i)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qpick = (int)i; break; }
    if (qpick < 0) { err = "no Vulkan compute queue"; return false; }
    qfam_ = (uint32_t)qpick;
    if (qf[qfam_].timestampValidBits > 0 && props.limits.timestampPeriod > 0)
        info_.timestamp_period_ns = props.limits.timestampPeriod;

    // Optional features: f16 arithmetic (packed-f16 GEMM variants).
    std::vector<const char*> exts;
    {
        uint32_t n_ext = 0;
        fn_.vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n_ext, nullptr);
        std::vector<VkExtensionProperties> ep(n_ext);
        fn_.vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n_ext, ep.data());
        auto has_ext = [&](const char* n) {
            for (auto& e : ep) if (std::strcmp(e.extensionName, n) == 0) return true;
            return false;
        };
        const bool core12 = props.apiVersion >= VK_API_VERSION_1_2;
        if (core12 || has_ext(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
            VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &f16;
            fn_.vkGetPhysicalDeviceFeatures2(phys_, &f2);
            info_.f16 = f16.shaderFloat16 == VK_TRUE && !env_on("STARLING_FAST_NO_F16");
            if (info_.f16 && !core12) exts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        }
        // Integer dot products (probe only so far). The instance targets
        // Vulkan 1.1, so this goes through the KHR extension even on 1.3.
        if (has_ext(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
            VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR idf{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &idf;
            fn_.vkGetPhysicalDeviceFeatures2(phys_, &f2);
            info_.int_dot = idf.shaderIntegerDotProduct == VK_TRUE;
            if (info_.int_dot) exts.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        }
        if (env_on("STARLING_FAST_VERBOSE")) {
            std::fprintf(stderr, "[fast-vk] device %04x:%04x driver %08x api %08x\n", info_.vendor_id,
                         info_.device_id, info_.driver_version, info_.api_version);
            for (auto& e : ep)
                std::fprintf(stderr, "[fast-vk] ext %s %u\n", e.extensionName, e.specVersion);
            VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &sg;
            fn_.vkGetPhysicalDeviceProperties2(phys_, &p2);
            std::fprintf(stderr, "[fast-vk] subgroup size %u ops %08x stages %08x\n", sg.subgroupSize,
                         sg.supportedOperations, sg.supportedStages);
            std::fprintf(stderr, "[fast-vk] f16 %d int_dot %d\n", info_.f16, info_.int_dot);
        }
    }
    VkPhysicalDeviceShaderFloat16Int8Features f16_on{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16_on.shaderFloat16 = info_.f16 ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR idot_on{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR};
    idot_on.shaderIntegerDotProduct = VK_TRUE;


    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)exts.size();
    dci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
    void* feat_chain = nullptr;
    if (info_.f16) {
        f16_on.pNext = feat_chain;
        feat_chain = &f16_on;
    }
    if (info_.int_dot) {
        idot_on.pNext = feat_chain;
        feat_chain = &idot_on;
    }
    dci.pNext = feat_chain;
    r = fn_.vkCreateDevice(phys_, &dci, nullptr, &dev_);
    if (r != VK_SUCCESS) { err = vk_err("vkCreateDevice", r); return false; }
#define STARLING_VK_LOAD_D(name)                                                   \
    fn_.name = (PFN_##name)fn_.vkGetDeviceProcAddr(dev_, #name);                   \
    if (!fn_.name) { err = "missing Vulkan device entry point " #name; return false; }
    STARLING_VK_DEVICE_FNS(STARLING_VK_LOAD_D)
#undef STARLING_VK_LOAD_D
    fn_.vkGetDeviceQueue(dev_, qfam_, 0, &queue_);

    fn_.vkGetPhysicalDeviceMemoryProperties(phys_, &memprops_);
    for (uint32_t i = 0; i < memprops_.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = memprops_.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            memprops_.memoryHeaps[memprops_.memoryTypes[i].heapIndex].size > (512ull << 20))
        {
            info_.uma = !info_.discrete;
            if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) info_.uma_cached = info_.uma;
        }
    }

    // Pipeline cache: persisted next to other caches so the second process
    // start skips driver shader compilation (hundreds of ms on mobile).
    std::vector<char> cache_blob;
    if (const char* d = std::getenv("STARLING_FAST_CACHE_DIR")) {
        char tag[64];
        std::snprintf(tag, sizeof tag, "/starling-fast-%08x-%08x.pcache",
                      info_.vendor_id, info_.device_id);
        pcache_path_ = std::string(d) + tag;
        std::ifstream in(pcache_path_, std::ios::binary);
        if (in) cache_blob.assign(std::istreambuf_iterator<char>(in), {});
    }
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pci.initialDataSize = cache_blob.size();
    pci.pInitialData = cache_blob.empty() ? nullptr : cache_blob.data();
    if (fn_.vkCreatePipelineCache(dev_, &pci, nullptr, &pcache_) != VK_SUCCESS) {
        pci.initialDataSize = 0; pci.pInitialData = nullptr;
        fn_.vkCreatePipelineCache(dev_, &pci, nullptr, &pcache_);
    }

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = qfam_;
    r = fn_.vkCreateCommandPool(dev_, &cpci, nullptr, &xfer_pool_);
    if (r != VK_SUCCESS) { err = vk_err("vkCreateCommandPool", r); return false; }
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = xfer_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    r = fn_.vkAllocateCommandBuffers(dev_, &cbai, &xfer_cb_);
    if (r != VK_SUCCESS) { err = vk_err("vkAllocateCommandBuffers", r); return false; }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    r = fn_.vkCreateFence(dev_, &fci, nullptr, &xfer_fence_);
    if (r != VK_SUCCESS) { err = vk_err("vkCreateFence", r); return false; }

    profile_ = env_on("STARLING_FAST_PROFILE") && info_.timestamp_period_ns > 0;
    if (env_on("STARLING_FAST_VERBOSE"))
        std::fprintf(stderr,
            "[fast] device '%s' vendor=%04x api=%u.%u subgroup=%u shared=%u "
            "max_storage=%llu uma=%d%s f16=%d queue_family=%u\n",
            info_.name.c_str(), info_.vendor_id, VK_VERSION_MAJOR(info_.api_version),
            VK_VERSION_MINOR(info_.api_version), info_.subgroup_size,
            info_.max_shared_bytes, (unsigned long long)info_.max_storage_range,
            info_.uma ? 1 : 0, info_.uma_cached ? "(cached)" : "", info_.f16 ? 1 : 0, qfam_);
    return true;
}

Context::~Context() {
    if (!dev_) {   // init failed after creating the instance
        if (inst_) fn_.vkDestroyInstance(inst_, nullptr);
        return;
    }
    fn_.vkDeviceWaitIdle(dev_);
    if (pcache_ && !pcache_path_.empty()) {
        size_t n = 0;
        if (fn_.vkGetPipelineCacheData(dev_, pcache_, &n, nullptr) == VK_SUCCESS && n) {
            std::vector<char> blob(n);
            if (fn_.vkGetPipelineCacheData(dev_, pcache_, &n, blob.data()) == VK_SUCCESS) {
                std::ofstream out(pcache_path_, std::ios::binary);
                out.write(blob.data(), (std::streamsize)n);
            }
        }
    }
    staging_.release();
    for (auto& kv : pipes_) fn_.vkDestroyPipeline(dev_, kv.second->pipe, nullptr);
    for (auto& kv : layouts_) {
        fn_.vkDestroyPipelineLayout(dev_, kv.second.second, nullptr);
        fn_.vkDestroyDescriptorSetLayout(dev_, kv.second.first, nullptr);
    }
    for (auto& kv : modules_) fn_.vkDestroyShaderModule(dev_, kv.second, nullptr);
    if (pcache_) fn_.vkDestroyPipelineCache(dev_, pcache_, nullptr);
    if (xfer_fence_) fn_.vkDestroyFence(dev_, xfer_fence_, nullptr);
    if (xfer_pool_) fn_.vkDestroyCommandPool(dev_, xfer_pool_, nullptr);
    fn_.vkDestroyDevice(dev_, nullptr);
    if (inst_) fn_.vkDestroyInstance(inst_, nullptr);
    // The loader library stays mapped: other Vulkan users in the process
    // (e.g. ggml's backend) may share it.
}

int Context::find_memory(uint32_t type_bits, VkMemoryPropertyFlags want,
                         VkMemoryPropertyFlags avoid) const {
    for (uint32_t i = 0; i < memprops_.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = memprops_.memoryTypes[i].propertyFlags;
        if ((f & want) == want && !(f & avoid)) return (int)i;
    }
    return -1;
}

bool Context::create_buffer(Buffer& out, VkDeviceSize bytes, Mem kind, std::string& err) {
    out.release();
    bytes = std::max<VkDeviceSize>(bytes, 16);
    bytes = (bytes + 15) & ~VkDeviceSize(15);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = fn_.vkCreateBuffer(dev_, &bci, nullptr, &out.buf);
    if (r != VK_SUCCESS) { err = vk_err("vkCreateBuffer", r); return false; }
    out.ctx = this;
    out.size = bytes;
    VkMemoryRequirements req;
    fn_.vkGetBufferMemoryRequirements(dev_, out.buf, &req);

    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    int mt = -1;
    switch (kind) {
    case Mem::Device:
        // UMA: device-local memory that is also host-visible lets uploads be
        // plain memcpy (no staging round trip, no second copy of the weights).
        if (info_.uma_cached) mt = find_memory(req.memoryTypeBits, DL | HV | HC | CA, 0);
        if (mt < 0 && info_.uma) mt = find_memory(req.memoryTypeBits, DL | HV | HC, 0);
        if (mt < 0) mt = find_memory(req.memoryTypeBits, DL, 0);
        break;
    case Mem::DeviceOnly:
        mt = find_memory(req.memoryTypeBits, DL, HV);
        if (mt < 0) mt = find_memory(req.memoryTypeBits, DL, 0);
        break;
    case Mem::Upload:
        mt = find_memory(req.memoryTypeBits, HV | HC, CA);
        if (mt < 0) mt = find_memory(req.memoryTypeBits, HV | HC, 0);
        break;
    case Mem::Readback:
        mt = find_memory(req.memoryTypeBits, HV | HC | CA, 0);
        if (mt < 0) mt = find_memory(req.memoryTypeBits, HV | CA, 0);
        if (mt < 0) mt = find_memory(req.memoryTypeBits, HV | HC, 0);
        break;
    }
    if (mt < 0) mt = find_memory(req.memoryTypeBits, 0, 0);
    if (mt < 0) { err = "no suitable Vulkan memory type"; out.release(); return false; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = (uint32_t)mt;
    r = fn_.vkAllocateMemory(dev_, &mai, nullptr, &out.mem);
    if (r != VK_SUCCESS) {
        err = vk_err("vkAllocateMemory", r) + " (" + std::to_string(req.size >> 20) + " MiB)";
        out.release();
        return false;
    }
    fn_.vkBindBufferMemory(dev_, out.buf, out.mem, 0);
    VkMemoryPropertyFlags f = memprops_.memoryTypes[mt].propertyFlags;
    if ((f & HV) && kind != Mem::DeviceOnly) {
        r = fn_.vkMapMemory(dev_, out.mem, 0, VK_WHOLE_SIZE, 0, &out.host);
        if (r != VK_SUCCESS) out.host = nullptr;
        out.coherent = (f & HC) != 0;
    }
    return true;
}

bool Context::ensure_staging(VkDeviceSize bytes, std::string& err) {
    if (staging_ && staging_.size >= bytes) return true;
    VkDeviceSize want = std::max<VkDeviceSize>(bytes, 64ull << 20);
    if (!create_buffer(staging_, want, Mem::Readback, err)) return false;
    if (!staging_.host) { staging_.release(); err = "staging buffer is not host-mappable"; return false; }
    return true;
}

bool Context::upload(Buffer& dst, VkDeviceSize off, const void* src, size_t bytes, std::string& err) {
    if (bytes == 0) return true;
    if (off + bytes > dst.size) { err = "upload out of range"; return false; }
    if (dst.host) {
        std::memcpy((char*)dst.host + off, src, bytes);
        if (!dst.coherent) {
            VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            mr.memory = dst.mem; mr.offset = 0; mr.size = VK_WHOLE_SIZE;
            fn_.vkFlushMappedMemoryRanges(dev_, 1, &mr);
        }
        return true;
    }
    std::lock_guard<std::mutex> lk(queue_mu_);
    const VkDeviceSize chunk = 64ull << 20;
    if (!ensure_staging(std::min<VkDeviceSize>(bytes, chunk), err)) return false;
    size_t done = 0;
    while (done < bytes) {
        size_t n = (size_t)std::min<VkDeviceSize>(bytes - done, staging_.size);
        std::memcpy(staging_.host, (const char*)src + done, n);
        if (!staging_.coherent) {
            VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            mr.memory = staging_.mem; mr.offset = 0; mr.size = VK_WHOLE_SIZE;
            fn_.vkFlushMappedMemoryRanges(dev_, 1, &mr);
        }
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        fn_.vkBeginCommandBuffer(xfer_cb_, &bi);
        VkBufferCopy c{0, off + done, n};
        fn_.vkCmdCopyBuffer(xfer_cb_, staging_.buf, dst.buf, 1, &c);
        fn_.vkEndCommandBuffer(xfer_cb_);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &xfer_cb_;
        VkResult r = fn_.vkQueueSubmit(queue_, 1, &si, xfer_fence_);
        if (r != VK_SUCCESS) { err = vk_err("vkQueueSubmit(upload)", r); return false; }
        fn_.vkWaitForFences(dev_, 1, &xfer_fence_, VK_TRUE, UINT64_MAX);
        fn_.vkResetFences(dev_, 1, &xfer_fence_);
        done += n;
    }
    return true;
}

bool Context::download(const Buffer& src, VkDeviceSize off, void* dst, size_t bytes, std::string& err) {
    if (bytes == 0) return true;
    if (off + bytes > src.size) { err = "download out of range"; return false; }
    if (src.host) {
        if (!src.coherent) {
            VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            mr.memory = src.mem; mr.offset = 0; mr.size = VK_WHOLE_SIZE;
            fn_.vkInvalidateMappedMemoryRanges(dev_, 1, &mr);
        }
        std::memcpy(dst, (const char*)src.host + off, bytes);
        return true;
    }
    std::lock_guard<std::mutex> lk(queue_mu_);
    const VkDeviceSize chunk = 64ull << 20;
    if (!ensure_staging(std::min<VkDeviceSize>(bytes, chunk), err)) return false;
    size_t done = 0;
    while (done < bytes) {
        size_t n = (size_t)std::min<VkDeviceSize>(bytes - done, staging_.size);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        fn_.vkBeginCommandBuffer(xfer_cb_, &bi);
        VkBufferCopy c{off + done, 0, n};
        fn_.vkCmdCopyBuffer(xfer_cb_, src.buf, staging_.buf, 1, &c);
        fn_.vkEndCommandBuffer(xfer_cb_);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &xfer_cb_;
        VkResult r = fn_.vkQueueSubmit(queue_, 1, &si, xfer_fence_);
        if (r != VK_SUCCESS) { err = vk_err("vkQueueSubmit(download)", r); return false; }
        fn_.vkWaitForFences(dev_, 1, &xfer_fence_, VK_TRUE, UINT64_MAX);
        fn_.vkResetFences(dev_, 1, &xfer_fence_);
        if (!staging_.coherent) {
            VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            mr.memory = staging_.mem; mr.offset = 0; mr.size = VK_WHOLE_SIZE;
            fn_.vkInvalidateMappedMemoryRanges(dev_, 1, &mr);
        }
        std::memcpy((char*)dst + done, staging_.host, n);
        done += n;
    }
    return true;
}

const Pipeline* Context::pipeline(const char* shader, const std::vector<uint32_t>& spec,
                                  std::string& err) {
    std::string key = shader;
    for (uint32_t v : spec) key += ":" + std::to_string(v);
    std::lock_guard<std::mutex> lk(pipe_mu_);
    auto it = pipes_.find(key);
    if (it != pipes_.end()) return it->second.get();

    const ShaderBlob* blob = find_shader(shader);
    if (!blob) { err = std::string("unknown fast-engine shader ") + shader; return nullptr; }
    VkShaderModule mod;
    auto mit = modules_.find(shader);
    if (mit != modules_.end()) {
        mod = mit->second;
    } else {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = blob->n_words * 4;
        smci.pCode = blob->words;
        VkResult r = fn_.vkCreateShaderModule(dev_, &smci, nullptr, &mod);
        if (r != VK_SUCCESS) { err = vk_err("vkCreateShaderModule", r); return nullptr; }
        modules_[shader] = mod;
    }
    auto& lay = layouts_[blob->n_bindings];
    if (!lay.first) {
        std::vector<VkDescriptorSetLayoutBinding> b(blob->n_bindings);
        for (uint32_t i = 0; i < blob->n_bindings; ++i) {
            b[i] = {};
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = blob->n_bindings;
        dci.pBindings = b.data();
        VkResult r = fn_.vkCreateDescriptorSetLayout(dev_, &dci, nullptr, &lay.first);
        if (r != VK_SUCCESS) { err = vk_err("vkCreateDescriptorSetLayout", r); return nullptr; }
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &lay.first;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        r = fn_.vkCreatePipelineLayout(dev_, &plci, nullptr, &lay.second);
        if (r != VK_SUCCESS) { err = vk_err("vkCreatePipelineLayout", r); return nullptr; }
    }
    std::vector<VkSpecializationMapEntry> me(spec.size());
    for (size_t i = 0; i < spec.size(); ++i) me[i] = {(uint32_t)i, (uint32_t)(i * 4), 4};
    VkSpecializationInfo si{};
    si.mapEntryCount = (uint32_t)me.size();
    si.pMapEntries = me.data();
    si.dataSize = spec.size() * 4;
    si.pData = spec.data();
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    cpci.layout = lay.second;
    auto p = std::make_unique<Pipeline>();
    VkResult r = fn_.vkCreateComputePipelines(dev_, pcache_, 1, &cpci, nullptr, &p->pipe);
    if (r != VK_SUCCESS) { err = vk_err("vkCreateComputePipelines", r) + " for " + key; return nullptr; }
    p->layout = lay.second;
    p->dsl = lay.first;
    p->n_bindings = blob->n_bindings;
    p->label = shader;
    const Pipeline* raw = p.get();
    pipes_[key] = std::move(p);
    return raw;
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

Recording::Recording(Context& ctx) : ctx_(ctx) {
    const Fns& f = ctx_.fn_;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = ctx_.qfam_;
    VkResult r = f.vkCreateCommandPool(ctx_.dev_, &cpci, nullptr, &pool_);
    if (r != VK_SUCCESS) { pool_ = VK_NULL_HANDLE; fail_ = vk_err("vkCreateCommandPool", r); }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    r = f.vkCreateFence(ctx_.dev_, &fci, nullptr, &fence_);
    if (r != VK_SUCCESS) { fence_ = VK_NULL_HANDLE; if (fail_.empty()) fail_ = vk_err("vkCreateFence", r); }
    profile_ = ctx_.profile_;
}

Recording::~Recording() {
    const Fns& f = ctx_.fn_;
    for (VkDescriptorPool p : dpools_) f.vkDestroyDescriptorPool(ctx_.dev_, p, nullptr);
    if (qpool_) f.vkDestroyQueryPool(ctx_.dev_, qpool_, nullptr);
    if (fence_) f.vkDestroyFence(ctx_.dev_, fence_, nullptr);
    if (pool_) f.vkDestroyCommandPool(ctx_.dev_, pool_, nullptr);
}

void Recording::begin() {
    const Fns& f = ctx_.fn_;
    if (!pool_) return;   // fail_ is set; submit_and_wait reports it
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    f.vkAllocateCommandBuffers(ctx_.dev_, &cbai, &cb_);
    segs_.push_back(cb_);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    f.vkBeginCommandBuffer(cb_, &bi);
    if (profile_) {
        max_queries_ = 4096;
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = max_queries_;
        if (!qpool_ && f.vkCreateQueryPool(ctx_.dev_, &qci, nullptr, &qpool_) != VK_SUCCESS) {
            qpool_ = VK_NULL_HANDLE;
            profile_ = false;   // profiling is best-effort
            return;
        }
        f.vkCmdResetQueryPool(cb_, qpool_, 0, max_queries_);
        n_queries_ = 0;
        q_labels_.clear();
        f.vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool_, n_queries_++);
        q_labels_.push_back("<start>");
    }
}

void Recording::end() {
    if (cb_) ctx_.fn_.vkEndCommandBuffer(cb_);
}

void Recording::split() {
    const Fns& f = ctx_.fn_;
    if (!cb_) return;
    f.vkEndCommandBuffer(cb_);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    f.vkAllocateCommandBuffers(ctx_.dev_, &cbai, &cb_);
    segs_.push_back(cb_);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    f.vkBeginCommandBuffer(cb_, &bi);
    // Order this segment after everything submitted before it.
    barrier();
}

VkDescriptorSet Recording::alloc_set(const Pipeline& p) {
    const Fns& f = ctx_.fn_;
    const uint32_t kSets = 256;
    // Two attempts: the current pool, then a fresh one (a pool can run out of
    // descriptors before sets when layouts have many bindings).
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (dpools_.empty() || dpool_left_ == 0) {
            VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSets * 16};
            VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpci.maxSets = kSets;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &ps;
            VkDescriptorPool dp = VK_NULL_HANDLE;
            VkResult r = f.vkCreateDescriptorPool(ctx_.dev_, &dpci, nullptr, &dp);
            if (r != VK_SUCCESS) {
                if (fail_.empty()) fail_ = vk_err("vkCreateDescriptorPool", r);
                return VK_NULL_HANDLE;
            }
            dpools_.push_back(dp);
            dpool_left_ = kSets;
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpools_.back();
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &p.dsl;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkResult r = f.vkAllocateDescriptorSets(ctx_.dev_, &ai, &set);
        if (r == VK_SUCCESS) {
            --dpool_left_;
            return set;
        }
        dpool_left_ = 0;
        if (attempt == 1 && fail_.empty()) fail_ = vk_err("vkAllocateDescriptorSets", r);
    }
    return VK_NULL_HANDLE;
}

void Recording::dispatch(const Pipeline& p, const std::vector<Ref>& bindings,
                         const void* push, size_t push_bytes,
                         uint32_t gx, uint32_t gy, uint32_t gz) {
    const Fns& f = ctx_.fn_;
    if (!cb_) return;
    if (bindings.empty() && p.n_bindings) {
        if (fail_.empty()) fail_ = std::string("dispatch without bindings: ") + p.label;
        return;
    }
    VkDescriptorSet set = alloc_set(p);
    if (!set) return;   // fail_ is set; submit_and_wait reports it
    std::vector<VkDescriptorBufferInfo> infos(p.n_bindings);
    std::vector<VkWriteDescriptorSet> writes(p.n_bindings);
    for (uint32_t i = 0; i < p.n_bindings; ++i) {
        const Ref& r = i < bindings.size() ? bindings[i] : bindings.back();
        infos[i].buffer = r.buf->buf;
        infos[i].offset = r.off;
        VkDeviceSize range = r.range == VK_WHOLE_SIZE ? r.buf->size - r.off : r.range;
        range = std::min<VkDeviceSize>(range, ctx_.info_.max_storage_range);
        infos[i].range = range;
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    f.vkUpdateDescriptorSets(ctx_.dev_, p.n_bindings, writes.data(), 0, nullptr);
    f.vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    f.vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    uint8_t pc[kPushBytes] = {};
    std::memcpy(pc, push, std::min<size_t>(push_bytes, kPushBytes));
    f.vkCmdPushConstants(cb_, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pc);
    if (gx && gy && gz) f.vkCmdDispatch(cb_, gx, gy, gz);
    ++n_dispatch_;
    if (profile_ && n_queries_ < max_queries_) {
        f.vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool_, n_queries_++);
        q_labels_.push_back(label_.empty() ? p.label : label_);
    }
}

void Recording::barrier() {
    if (!cb_) return;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    ctx_.fn_.vkCmdPipelineBarrier(cb_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Recording::copy(const Buffer& src, VkDeviceSize soff, const Buffer& dst,
                     VkDeviceSize doff, VkDeviceSize bytes) {
    if (!cb_) return;
    VkBufferCopy c{soff, doff, bytes};
    ctx_.fn_.vkCmdCopyBuffer(cb_, src.buf, dst.buf, 1, &c);
}

void Recording::fill(const Buffer& dst, VkDeviceSize off, VkDeviceSize bytes, uint32_t value) {
    if (!cb_) return;
    ctx_.fn_.vkCmdFillBuffer(cb_, dst.buf, off, bytes, value);
}

bool Recording::submit_and_wait(std::string& err) {
    const Fns& f = ctx_.fn_;
    if (!fail_.empty()) { err = "recording failed: " + fail_; return false; }
    if (segs_.empty()) { err = "recording is empty (begin() not called)"; return false; }
    std::lock_guard<std::mutex> lk(ctx_.queue_mu_);
    VkResult r = VK_SUCCESS;
    for (size_t i = 0; i < segs_.size(); ++i) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &segs_[i];
        r = f.vkQueueSubmit(ctx_.queue_, 1, &si, i + 1 == segs_.size() ? fence_ : VK_NULL_HANDLE);
        if (r != VK_SUCCESS) {
            // Earlier segments may be in flight: drain before reporting.
            f.vkDeviceWaitIdle(ctx_.dev_);
            err = vk_err("vkQueueSubmit", r);
            return false;
        }
    }
    // Bounded wait: a lost device must surface as an error, not a hang.
    r = f.vkWaitForFences(ctx_.dev_, 1, &fence_, VK_TRUE, 120ull * 1000 * 1000 * 1000);
    if (r != VK_SUCCESS) {
        // The fence may still be pending (timeout): drain the queue before
        // it is reset or the recording's buffers are freed.
        f.vkDeviceWaitIdle(ctx_.dev_);
        f.vkResetFences(ctx_.dev_, 1, &fence_);
        err = vk_err("vkWaitForFences", r);
        return false;
    }
    f.vkResetFences(ctx_.dev_, 1, &fence_);
    return true;
}

void Recording::report_profile(const char* title) const {
    if (!profile_ || n_queries_ < 2) return;
    const Fns& f = ctx_.fn_;
    std::vector<uint64_t> ts(n_queries_);
    if (f.vkGetQueryPoolResults(ctx_.dev_, qpool_, 0, n_queries_, ts.size() * 8, ts.data(), 8,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return;
    const double ns = ctx_.info_.timestamp_period_ns;
    std::map<std::string, std::pair<double, int>> agg;
    for (uint32_t i = 1; i < n_queries_; ++i) {
        auto& a = agg[q_labels_[i]];
        a.first += (double)(ts[i] - ts[i - 1]) * ns * 1e-6;
        a.second += 1;
    }
    const double total = (double)(ts[n_queries_ - 1] - ts[0]) * ns * 1e-6;
    std::vector<std::pair<double, std::string>> rows;
    for (auto& kv : agg) rows.push_back({kv.second.first, kv.first});
    std::sort(rows.rbegin(), rows.rend());
    std::fprintf(stderr, "[fast-profile] %s: %.3f ms GPU, %zu dispatches\n", title, total, n_dispatch_);
    for (auto& row : rows)
        std::fprintf(stderr, "  %-28s %9.3f ms  %5.1f%%  x%d\n", row.second.c_str(), row.first,
                     100.0 * row.first / std::max(total, 1e-9), agg[row.second].second);
}

} // namespace starling::fast::vk
