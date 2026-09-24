               OpCapability Shader
               OpCapability DotProduct
               OpCapability DotProductInput4x8BitPacked
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %bin %bout
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %rt ArrayStride 4
               OpDecorate %blk Block
               OpMemberDecorate %blk 0 Offset 0
               OpDecorate %bin DescriptorSet 0
               OpDecorate %bin Binding 0
               OpDecorate %bout DescriptorSet 0
               OpDecorate %bout Binding 1
               OpDecorate %gid BuiltIn GlobalInvocationId
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
         %v3u = OpTypeVector %uint 3
         %fn = OpTypeFunction %void
         %rt = OpTypeRuntimeArray %uint
        %blk = OpTypeStruct %rt
       %ptrb = OpTypePointer StorageBuffer %blk
       %ptru = OpTypePointer StorageBuffer %uint
       %ptrv = OpTypePointer Input %v3u
        %c0 = OpConstant %uint 0
        %c1 = OpConstant %uint 1
        %c2 = OpConstant %uint 2
        %c3 = OpConstant %uint 3
        %gid = OpVariable %ptrv Input
        %bin = OpVariable %ptrb StorageBuffer
       %bout = OpVariable %ptrb StorageBuffer
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %g3 = OpLoad %v3u %gid
         %ix = OpCompositeExtract %uint %g3 0
        %pit = OpAccessChain %ptru %bin %c0 %c0
         %pa = OpAccessChain %ptru %bin %c0 %c1
         %pb = OpAccessChain %ptru %bin %c0 %c2
         %ps = OpAccessChain %ptru %bin %c0 %c3
         %va = OpLoad %uint %pa
         %vb = OpLoad %uint %pb
         %vs = OpLoad %uint %ps
         %vit = OpLoad %uint %pit
         %dd = OpSDot %uint %va %vb PackedVectorFormat4x8Bit
         %v0 = OpBitwiseXor %uint %vs %dd
               OpBranch %loop
        %loop = OpLabel
          %k = OpPhi %uint %c0 %entry %nk %cont
          %v = OpPhi %uint %v0 %entry %nv %cont
        %cmp = OpSLessThan %bool %k %vit
               OpLoopMerge %merge %cont None
               OpBranchConditional %cmp %body %merge
        %body = OpLabel
         %nv = OpSDot %uint %v %vb PackedVectorFormat4x8Bit
         %nk = OpIAdd %uint %k %c1
               OpBranch %cont
        %cont = OpLabel
               OpBranch %loop
       %merge = OpLabel
         %po = OpAccessChain %ptru %bout %c0 %ix
               OpStore %po %v
               OpReturn
               OpFunctionEnd
