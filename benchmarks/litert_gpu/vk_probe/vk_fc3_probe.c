// Times fc3.comp (subgroup-broadcast GEMM on cpp/fast's operand layouts) and
// checks 64 outputs per variant against a CPU reference.
// vk_fc3_probe M K N [reps=20] [kind filter: w4|w8|f16]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "fc3_spv.h"   // FC3_VARIANTS: {name, bkind, nb, rm, atex, words, bytes}

struct Var { const char* name; int bkind, nb, rm, atex; const uint32_t* spv; size_t sz; };
static const struct Var VARS[] = { FC3_VARIANTS };

#define VK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, "%s -> %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static uint16_t f2h(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
  if (e < 113) return (uint16_t)s;
  uint32_t h = s | ((e - 112) << 10) | (m >> 13);
  if (m & 0x1000) h++;
  return (uint16_t)h;
}
static float h2f(uint16_t h) {
  uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
  uint32_t b = e ? s | ((e + 112) << 23) | (m << 13) : s;
  float f; memcpy(&f, &b, 4); return f;
}

static VkDevice dev;
static VkPhysicalDevice pd;
typedef struct { VkBuffer b; VkDeviceMemory m; void* p; VkDeviceSize n; } Buf;
static Buf mkbuf(VkDeviceSize n, VkBufferUsageFlags u) {
  Buf r; r.n = n;
  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, u};
  VK(vkCreateBuffer(dev, &bci, NULL, &r.b));
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, r.b, &mr);
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  uint32_t want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, t = 0;
  while (!((mr.memoryTypeBits >> t) & 1) || (mp.memoryTypes[t].propertyFlags & want) != want) ++t;
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, t};
  VK(vkAllocateMemory(dev, &mai, NULL, &r.m));
  VK(vkBindBufferMemory(dev, r.b, r.m, 0));
  VK(vkMapMemory(dev, r.m, 0, n, 0, &r.p));
  return r;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s M K N [reps] [w4|w8|f16]\n", argv[0]); return 2; }
  uint32_t M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), reps = argc > 4 ? atoi(argv[4]) : 20;
  int only = -1;
  if (argc > 5) only = !strcmp(argv[5], "w4") ? 0 : !strcmp(argv[5], "w8") ? 1 : 2;
  VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vk_fc3_probe", 1, NULL, 0, VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai};
  VkInstance inst; VK(vkCreateInstance(&ici, NULL, &inst));
  uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &pd);
  VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 pp2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg};
  vkGetPhysicalDeviceProperties2(pd, &pp2);
  VkPhysicalDeviceProperties pp = pp2.properties;
  uint32_t nq = 8; VkQueueFamilyProperties qf[8];
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
  uint32_t qfi = 0; while (!(qf[qfi].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++qfi;
  float prio = 1;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfi, 1, &prio};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci};
  VK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue q; vkGetDeviceQueue(dev, qfi, 0, &q);
  const uint32_t WG = sg.subgroupSize;
  printf("%s: subgroup %u, maxTexelBufferElements %u, minTexelBufferOffsetAlignment %llu | M=%u K=%u N=%u\n",
         pp.deviceName, WG, pp.limits.maxTexelBufferElements,
         (unsigned long long)pp.limits.minTexelBufferOffsetAlignment, M, K, N);

  srand(7);
  const uint32_t G = K / 32;
  float* af = malloc(sizeof(float) * M * K);
  for (size_t i = 0; i < (size_t)M * K; ++i) af[i] = h2f(f2h((rand() % 2001) / 1000.0f - 1.0f));
  Buf A = mkbuf((VkDeviceSize)M * K * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
  uint16_t* ah = A.p;
  for (size_t i = 0; i < (size_t)M * K; ++i) ah[i] = f2h(af[i]);
  // Weights in each layout + the dequantized reference wf[n][k].
  float* wf[3]; Buf Bq[3], Bs[3];
  for (int b = 0; b < 3; ++b) {
    wf[b] = malloc(sizeof(float) * N * K);
    if (b == 0) {        // W4: 4 words of nibbles + packHalf2x16(scale, offset) per 32-group
      Bq[b] = mkbuf((VkDeviceSize)N * G * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      Bs[b] = mkbuf((VkDeviceSize)N * G * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      uint32_t* qw = Bq[b].p; uint32_t* sw = Bs[b].p;
      for (uint32_t i = 0; i < N * G; ++i) {
        float s = h2f(f2h(0.002f + 0.004f * (rand() % 1000) / 1000.0f)), o = h2f(f2h(-8 * s));
        sw[i] = f2h(s) | ((uint32_t)f2h(o) << 16);
        for (int w = 0; w < 4; ++w) {
          uint32_t v = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
          qw[i * 4 + w] = v;
          for (int t = 0; t < 8; ++t) wf[b][(size_t)(i / G) * K + (i % G) * 32 + w * 8 + t] = s * ((v >> (4 * t)) & 15) + o;
        }
      }
    } else if (b == 1) { // W8: 8 words of int8 + packHalf2x16(s_lo, s_hi)
      Bq[b] = mkbuf((VkDeviceSize)N * G * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      Bs[b] = mkbuf((VkDeviceSize)N * G * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      uint32_t* qw = Bq[b].p; uint32_t* sw = Bs[b].p;
      for (uint32_t i = 0; i < N * G; ++i) {
        float s0 = h2f(f2h(0.0002f + 0.0004f * (rand() % 1000) / 1000.0f)), s1 = h2f(f2h(0.0002f + 0.0004f * (rand() % 1000) / 1000.0f));
        sw[i] = f2h(s0) | ((uint32_t)f2h(s1) << 16);
        for (int w = 0; w < 8; ++w) {
          uint32_t v = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
          qw[i * 8 + w] = v;
          for (int t = 0; t < 4; ++t) wf[b][(size_t)(i / G) * K + (i % G) * 32 + w * 4 + t] = (w < 4 ? s0 : s1) * (float)(int8_t)(v >> (8 * t));
        }
      }
    } else {             // F16 [N][K]
      Bq[b] = mkbuf((VkDeviceSize)N * K * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      Bs[b] = mkbuf(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      uint16_t* h = Bq[b].p;
      for (size_t i = 0; i < (size_t)N * K; ++i) { float v = h2f(f2h(((rand() % 2001) / 1000.0f - 1.0f) * 0.05f)); h[i] = f2h(v); wf[b][i] = v; }
    }
  }
  Buf C = mkbuf((VkDeviceSize)M * N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  const uint32_t texels = M * K / 8;
  const int tex_ok = texels <= pp.limits.maxTexelBufferElements;
  VkBufferView av = VK_NULL_HANDLE;
  if (tex_ok) {
    VkBufferViewCreateInfo bvci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, A.b, VK_FORMAT_R32G32B32A32_UINT, 0, VK_WHOLE_SIZE};
    VK(vkCreateBufferView(dev, &bvci, NULL, &av));
  }

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
    if (only >= 0 && v->bkind != only) continue;
    if (v->atex && !tex_ok) { printf("%-22s  (A exceeds maxTexelBufferElements)\n", v->name); continue; }
    VkDescriptorSetLayoutBinding bd[4] = {
        {0, v->atex ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 4, bd};
    VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(dev, &dl, NULL, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
    VkPipelineLayout pl; VK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 2, ps};
    VkDescriptorPool dp; VK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
    VkDescriptorSet ds; VK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    int b = v->bkind;
    VkDescriptorBufferInfo bi0 = {A.b, 0, VK_WHOLE_SIZE}, bi1 = {Bq[b].b, 0, VK_WHOLE_SIZE}, bi2 = {Bs[b].b, 0, VK_WHOLE_SIZE}, bi3 = {C.b, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[4] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, bd[0].descriptorType, NULL, v->atex ? NULL : &bi0, v->atex ? &av : NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi1},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi2},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi3}};
    vkUpdateDescriptorSets(dev, 4, wr, 0, NULL);
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, v->sz, v->spv};
    VkShaderModule sm; VK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo spi = {1, &me, 4, &WG};
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", &spi}, pl};
    VkPipeline pipe; VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
      VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      VK(vkBeginCommandBuffer(cb, &cbbi));
      vkCmdResetQueryPool(cb, qp, 0, 2);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
      uint32_t pc[4] = {M, N, K, K};   // fc3.comp: M, N, K, lda
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
      VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
      for (uint32_t i = 0; i < reps; ++i) {
        vkCmdDispatch(cb, (N + v->nb - 1) / v->nb, (M + WG * v->rm - 1) / (WG * v->rm), 1);
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
      for (uint32_t k = 0; k < K; ++k) { double x = (double)af[(size_t)m * K + k] * wf[b][(size_t)o * K + k]; ref += x; mag += fabs(x); }
      double e = fabs(ref - c[(size_t)m * N + o]) / (mag > 1e-6 ? mag : 1e-6);
      if (e > maxerr) maxerr = e;
    }
    memset(C.p, 0, C.n);
    printf("%-22s  %7.3f ms  %6.1f GFLOPS  (rel err %.1e)\n", v->name, best, 2.0 * M * K * N / best / 1e6, maxerr);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  }
  vkDeviceWaitIdle(dev);
  return 0;
}
