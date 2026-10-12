// Times fc.comp (ML Drift's FC kernel ported to Vulkan) at a GEMM shape and
// checks 64 outputs against a CPU reference.
// vk_fc_probe M K N [reps=20]   (K % 8 == 0, N % 16 == 0)
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "fc_spv.h"   // fc_<F16>_<SL>_<ATEX>[]

struct Var { const char* name; int f16, sl, atex; const uint32_t* spv; size_t sz; };
static const struct Var VARS[] = {
    {"f32  SL1 tex ", 0, 1, 1, fc_0_1_1, sizeof fc_0_1_1},
    {"f16  SL1 tex ", 1, 1, 1, fc_1_1_1, sizeof fc_1_1_1},
    {"f16  SL1 ssbo", 1, 1, 0, fc_1_1_0, sizeof fc_1_1_0},
    {"f16  SL2 tex ", 1, 2, 1, fc_1_2_1, sizeof fc_1_2_1},
    {"f16  SL4 tex ", 1, 4, 1, fc_1_4_1, sizeof fc_1_4_1},
    {"f32  SL4 tex ", 0, 4, 1, fc_0_4_1, sizeof fc_0_4_1},
};

#define VK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, "%s -> %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static uint16_t f2h(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
  if (e < 113) return (uint16_t)s;                 // flush tiny to zero (inputs are |x| <= 1)
  uint32_t h = s | ((e - 112) << 10) | (m >> 13);
  if (m & 0x1000) h++;                             // round half up
  return (uint16_t)h;
}
static float h2f(uint16_t h) {
  uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
  uint32_t b = e ? s | ((e + 112) << 23) | (m << 13) : s;
  float f; memcpy(&f, &b, 4); return f;
}

static VkDevice dev;
static VkPhysicalDevice pd;
static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  fprintf(stderr, "no memory type\n"); exit(1);
}
typedef struct { VkBuffer b; VkDeviceMemory m; void* p; } Buf;
static Buf mkbuf(VkDeviceSize n, VkBufferUsageFlags u) {
  Buf r;
  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, u};
  VK(vkCreateBuffer(dev, &bci, NULL, &r.b));
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, r.b, &mr);
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
                              find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  VK(vkAllocateMemory(dev, &mai, NULL, &r.m));
  VK(vkBindBufferMemory(dev, r.b, r.m, 0));
  VK(vkMapMemory(dev, r.m, 0, n, 0, &r.p));
  return r;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s M K N [reps]\n", argv[0]); return 2; }
  uint32_t M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), reps = argc > 4 ? atoi(argv[4]) : 20;
  VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vk_fc_probe", 1, NULL, 0, VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai};
  VkInstance inst; VK(vkCreateInstance(&ici, NULL, &inst));
  uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &pd);
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
  uint32_t nq = 8; VkQueueFamilyProperties qf[8];
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
  uint32_t qfi = 0; while (!(qf[qfi].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++qfi;
  float prio = 1;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfi, 1, &prio};
  VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures sgx = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES, NULL, VK_TRUE};
  VkPhysicalDevice16BitStorageFeatures s16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &sgx, VK_TRUE};
  VkPhysicalDeviceShaderFloat16Int8Features f16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, &s16, VK_TRUE};
  VkPhysicalDeviceFeatures2 ff = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f16};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &ff, 0, 1, &qci};
  VK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue q; vkGetDeviceQueue(dev, qfi, 0, &q);
  printf("%s: M=%u K=%u N=%u, %u reps\n", pp.deviceName, M, K, N, reps);

  // Inputs: A f16 PHWC4, reference weights wf[k][n] (f16-rounded).
  srand(7);
  float* af = malloc(sizeof(float) * M * K);
  float* wf = malloc(sizeof(float) * K * N);
  for (size_t i = 0; i < (size_t)M * K; ++i) af[i] = h2f(f2h((rand() % 2001) / 1000.0f - 1.0f));
  for (size_t i = 0; i < (size_t)K * N; ++i) wf[i] = h2f(f2h(((rand() % 2001) / 1000.0f - 1.0f) * 0.05f));
  Buf A = mkbuf((VkDeviceSize)M * K * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
  uint16_t* ah = A.p;
  for (uint32_t s = 0; s < K / 4; ++s)
    for (uint32_t m = 0; m < M; ++m)
      for (int c = 0; c < 4; ++c) ah[((size_t)s * M + m) * 4 + c] = f2h(af[(size_t)m * K + 4 * s + c]);
  Buf C = mkbuf((VkDeviceSize)M * N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  VkBufferViewCreateInfo bvci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, A.b, VK_FORMAT_R16G16B16A16_SFLOAT, 0, VK_WHOLE_SIZE};
  VkBufferView av; VK(vkCreateBufferView(dev, &bvci, NULL, &av));

  VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi};
  VkCommandPool cp; VK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
  VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
  VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2};
  VkQueryPool qp; VK(vkCreateQueryPool(dev, &qpci, NULL, &qp));
  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence; VK(vkCreateFence(dev, &fci, NULL, &fence));

  for (size_t vi = 0; vi < sizeof VARS / sizeof VARS[0]; ++vi) {
    const struct Var* v = &VARS[vi];
    uint32_t SL = v->sl, KS = K / 4, groups = N / 4 / SL;
    // W[g][s/2][(h*SL + j)*4 + c] = weights of input channel 4(s+h)+c for output channels 4(g*SL+j)..+3
    Buf W = mkbuf((VkDeviceSize)K * N * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    uint16_t* wh = W.p;
    for (uint32_t g = 0; g < groups; ++g)
      for (uint32_t s = 0; s < KS; s += 2)
        for (uint32_t h = 0; h < 2; ++h)
          for (uint32_t j = 0; j < SL; ++j)
            for (uint32_t c = 0; c < 4; ++c)
              for (uint32_t o = 0; o < 4; ++o) {
                size_t lane = (h * SL + j) * 4 + c;
                size_t idx = (((size_t)g * (KS / 2) + s / 2) * (8 * SL) + lane) * 4 + o;
                wh[idx] = f2h(wf[(size_t)(4 * (s + h) + c) * N + 4 * (g * SL + j) + o]);
              }
    VkDescriptorSetLayoutBinding bd[3] = {
        {0, v->atex ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, bd};
    VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(dev, &dl, NULL, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
    VkPipelineLayout pl; VK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 2, ps};
    VkDescriptorPool dp; VK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
    VkDescriptorSet ds; VK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo bi0 = {A.b, 0, VK_WHOLE_SIZE}, bi1 = {W.b, 0, VK_WHOLE_SIZE}, bi2 = {C.b, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[3] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, bd[0].descriptorType, NULL, v->atex ? NULL : &bi0, v->atex ? &av : NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi1},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi2}};
    vkUpdateDescriptorSets(dev, 3, wr, 0, NULL);
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, v->sz, v->spv};
    VkShaderModule sm; VK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}, pl};
    VkPipeline pipe; VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    double best = 1e30;
    for (int r = 0; r < 5; ++r) {   // the first rounds ramp the GPU clock
      VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      VK(vkBeginCommandBuffer(cb, &cbbi));
      vkCmdResetQueryPool(cb, qp, 0, 2);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
      uint32_t pc[3] = {M, K, N};
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
      VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
      for (uint32_t i = 0; i < reps; ++i) {
        vkCmdDispatch(cb, groups, (M + 127) / 128, 1);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
      }
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
      VK(vkEndCommandBuffer(cb));
      VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
      VK(vkQueueSubmit(q, 1, &si, fence));
      VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));
      VK(vkResetFences(dev, 1, &fence));
      uint64_t ts[2];
      VK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      double ms = (ts[1] - ts[0]) * pp.limits.timestampPeriod * 1e-6 / reps;
      if (r >= 2 && ms < best) best = ms;
    }
    double maxerr = 0;
    const float* c = C.p;
    for (int t = 0; t < 64; ++t) {
      uint32_t m = rand() % M, o = rand() % N;
      double ref = 0, mag = 0;
      for (uint32_t k = 0; k < K; ++k) { ref += (double)af[(size_t)m * K + k] * wf[(size_t)k * N + o]; mag += fabs((double)af[(size_t)m * K + k] * wf[(size_t)k * N + o]); }
      double e = fabs(ref - c[(size_t)m * N + o]) / (mag > 1e-6 ? mag : 1e-6);
      if (e > maxerr) maxerr = e;
    }
    printf("%s  %7.3f ms  %6.1f GFLOPS  (rel err %.1e)\n", v->name, best, 2.0 * M * K * N / best / 1e6, maxerr);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkUnmapMemory(dev, W.m); vkDestroyBuffer(dev, W.b, NULL); vkFreeMemory(dev, W.m, NULL);
  }
  vkDeviceWaitIdle(dev);
  return 0;
}
