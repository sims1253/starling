// Vulkan twin of cl_alu_probe.c: runs alu.comp's modes (built by build.sh into
// alu_spv.h, one SPIR-V blob per MODE) and prints GFLOPS from timestamps.
// vk_alu_probe [groups=2048] [iters=4096]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "alu_spv.h"   // const uint32_t alu_mode<N>[] for N = 0..6

static const char* NAMES[] = {"f32 mad", "f32 fma", "f16v4 mad", "f16v4 fma", "f16v2 fma", "bc32", "bc16"};
static const uint32_t* SPV[] = {alu_mode0, alu_mode1, alu_mode2, alu_mode3, alu_mode4, alu_mode5, alu_mode6};
static const size_t SPV_SZ[] = {sizeof alu_mode0, sizeof alu_mode1, sizeof alu_mode2, sizeof alu_mode3,
                                sizeof alu_mode4, sizeof alu_mode5, sizeof alu_mode6};

#define VK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, "%s -> %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
  uint32_t groups = argc > 1 ? atoi(argv[1]) : 2048, iters = argc > 2 ? atoi(argv[2]) : 4096;
  VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vk_alu_probe", 1, NULL, 0, VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai};
  VkInstance inst;
  VK(vkCreateInstance(&ici, NULL, &inst));
  uint32_t n = 1;
  VkPhysicalDevice pd;
  vkEnumeratePhysicalDevices(inst, &n, &pd);
  VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg};
  vkGetPhysicalDeviceProperties2(pd, &pp);
  VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures sgx = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES};
  VkPhysicalDeviceShaderFloat16Int8Features f16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, &sgx};
  VkPhysicalDeviceFeatures2 ff = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f16};
  vkGetPhysicalDeviceFeatures2(pd, &ff);
  printf("device %s, api %u.%u, subgroup %u (ops 0x%x stages 0x%x), shaderFloat16 %u, subgroupExtendedTypes %u, ts %.3f ns\n",
         pp.properties.deviceName, VK_VERSION_MAJOR(pp.properties.apiVersion), VK_VERSION_MINOR(pp.properties.apiVersion),
         sg.subgroupSize, sg.supportedOperations, sg.supportedStages, f16.shaderFloat16,
         sgx.shaderSubgroupExtendedTypes, pp.properties.limits.timestampPeriod);

  uint32_t nq = 8;
  VkQueueFamilyProperties qf[8];
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
  uint32_t qfi = 0;
  while (qfi < nq && !(qf[qfi].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++qfi;
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfi, 1, &prio};
  VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures sgx_on = sgx;
  sgx_on.pNext = NULL;
  VkPhysicalDeviceShaderFloat16Int8Features f16_on = f16;
  f16_on.pNext = &sgx_on;
  VkPhysicalDeviceFeatures2 ff_on = ff;
  ff_on.pNext = &f16_on;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &ff_on, 0, 1, &qci};
  VkDevice dev;
  VK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, qfi, 0, &q);

  VkDeviceSize bytes = (VkDeviceSize)groups * 128 * 4;
  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
  VkBuffer buf;
  VK(vkCreateBuffer(dev, &bci, NULL, &buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buf, &mr);
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  uint32_t mt = 0;
  while (!(mr.memoryTypeBits & (1u << mt))) ++mt;
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt};
  VkDeviceMemory mem;
  VK(vkAllocateMemory(dev, &mai, NULL, &mem));
  VK(vkBindBufferMemory(dev, buf, mem, 0));

  VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b};
  VkDescriptorSetLayout dsl;
  VK(vkCreateDescriptorSetLayout(dev, &dl, NULL, &dsl));
  VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
  VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
  VkPipelineLayout pl;
  VK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
  VkDescriptorPool dp;
  VK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
  VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
  VkDescriptorSet ds;
  VK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo dbi = {buf, 0, bytes};
  VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi};
  vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

  VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi};
  VkCommandPool cp;
  VK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
  VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
  VkCommandBuffer cb;
  VK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2};
  VkQueryPool qp;
  VK(vkCreateQueryPool(dev, &qpci, NULL, &qp));
  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK(vkCreateFence(dev, &fci, NULL, &fence));

  for (int m = 0; m < 7; ++m) {
    if ((m >= 2 && m <= 4 && !f16.shaderFloat16) || (m == 6 && !(f16.shaderFloat16 && sgx.shaderSubgroupExtendedTypes))) {
      printf("%-10s unsupported\n", NAMES[m]);
      continue;
    }
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, SPV_SZ[m], SPV[m]};
    VkShaderModule sm;
    VK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                       {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                                        VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"},
                                       pl};
    VkPipeline pipe;
    VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    double best = 1e30;
    for (int r = 0; r < 6; ++r) {   // the first runs warm the clock
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      VK(vkBeginCommandBuffer(cb, &bi));
      vkCmdResetQueryPool(cb, qp, 0, 2);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
      struct { uint32_t iters; float seed; } pc = {iters, 0.999f};
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pc);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
      vkCmdDispatch(cb, groups, 1, 1);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
      VK(vkEndCommandBuffer(cb));
      VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
      VK(vkQueueSubmit(q, 1, &si, fence));
      VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));
      VK(vkResetFences(dev, 1, &fence));
      uint64_t ts[2];
      VK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      double ms = (ts[1] - ts[0]) * pp.properties.limits.timestampPeriod * 1e-6;
      if (r >= 2 && ms < best) best = ms;
    }
    // 8 vec4 (mode 4: vec2) multiply-adds per lane per iteration in every mode.
    double width = m == 4 ? 2 : 4;
    double flops = 2.0 * width * 8 * (double)iters * groups * 128;
    printf("%-10s %8.3f ms  %7.1f GFLOPS\n", NAMES[m], best, flops / best / 1e6);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
  }
  vkDeviceWaitIdle(dev);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  return 0;
}
