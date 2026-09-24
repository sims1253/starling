# fast.cmake — build rules for the model-specialized fast engines (cpp/fast).
#
# Included by backends/native/CMakeLists.txt when STARLING_FAST=ON. Compiles
# every compute shader variant to SPIR-V with glslc (the host's, or the NDK's
# shader-tools copy when cross-compiling for Android), embeds the blobs into a
# generated C++ table, and adds the engine sources to starling_ggml_core.
# libvulkan is loaded at run time, so only the Vulkan headers are needed here.

set(STARLING_FAST_DIR ${STARLING_REPO_ROOT}/cpp/fast)

# ---- tools / headers --------------------------------------------------------
set(_glslc_hints "")
if(DEFINED ENV{VULKAN_SDK})
  list(APPEND _glslc_hints "$ENV{VULKAN_SDK}/bin")
endif()
# Host glslc first (SPIR-V is built on the host regardless of target); the
# NDK's older shader-tools copy is the fallback for hosts without one.
find_program(STARLING_GLSLC NAMES glslc HINTS ${_glslc_hints} NO_CMAKE_FIND_ROOT_PATH)
if(NOT STARLING_GLSLC AND ANDROID_NDK)
  file(GLOB _ndk_glslc "${ANDROID_NDK}/shader-tools/*/glslc")
  foreach(_g IN LISTS _ndk_glslc)
    get_filename_component(_d "${_g}" DIRECTORY)
    list(APPEND _glslc_hints "${_d}")
  endforeach()
  find_program(STARLING_GLSLC NAMES glslc HINTS ${_glslc_hints} NO_CMAKE_FIND_ROOT_PATH)
endif()
if(NOT STARLING_GLSLC)
  message(FATAL_ERROR "STARLING_FAST=ON needs glslc (Vulkan SDK / shaderc, or the Android NDK's shader-tools)")
endif()

if(NOT ANDROID)
  find_path(STARLING_FAST_VK_INCLUDE vulkan/vulkan.h
            HINTS "$ENV{VULKAN_SDK}/include")
  if(NOT STARLING_FAST_VK_INCLUDE)
    message(FATAL_ERROR "STARLING_FAST=ON needs the Vulkan headers (vulkan/vulkan.h)")
  endif()
endif()

# ---- shader variants ----------------------------------------------------------
# name | source | defines (";"-free, comma separated) | binding count
set(STARLING_FAST_SHADERS
  "gemm_w4|gemm.comp|B_W4|7"
  "gemm_w8|gemm.comp|B_W8|7"
  "gemm_f16|gemm.comp|B_F16|7"
  "gemm_f16t|gemm.comp|B_F16T|7"
  "gemm_w4_h|gemm.comp|B_W4,F16MATH|7"
  "gemm_w8_h|gemm.comp|B_W8,F16MATH|7"
  "gemm_f16_h|gemm.comp|B_F16,F16MATH|7"
  "gemm_f16t_h|gemm.comp|B_F16T,F16MATH|7"
  "gemm_conv|gemm.comp|B_F16,A_CONV|7"
  "gemm_conv_h|gemm.comp|B_F16,A_CONV,F16MATH|7"
  "gemm_coop_w4|gemm_coop.comp|B_W4|7|--target-env=vulkan1.3"
  "gemm_coop_w8|gemm_coop.comp|B_W8|7|--target-env=vulkan1.3"
  "gemm_coop_f16|gemm_coop.comp|B_F16|7|--target-env=vulkan1.3"
  "gemm_coop_f16t|gemm_coop.comp|B_F16T|7|--target-env=vulkan1.3"
  "coop_probe|coop_probe.comp||3|--target-env=vulkan1.3"
  "alu_probe|alu_probe.comp||1"
  "gemv_w4|gemv.comp|W_W4|7"
  "gemv_w8|gemv.comp|W_W8|7"
  "gemv_f16|gemv.comp|W_F16|7"
  "attn_decode|attn_decode.comp||8"
  "rope_kv|rope_kv.comp||7"
  "embed_rows|embed_rows.comp|W_W4|10"
  "embed_rows_w8|embed_rows.comp|W_W8|10"
  "embed_rows_f16|embed_rows.comp|W_F16|10"
  "dec_next|dec_next.comp|W_W8|11"
  "dec_next_w4|dec_next.comp|W_W4|11"
  "dec_next_f16|dec_next.comp|W_F16|11"
  "norm|norm.comp||6"
  "softmax|softmax.comp||3"
  "pk_conv|pk_conv.comp||5"
  "idot_probe|idot_probe.spv||2"
)

set(_spv_dir ${CMAKE_CURRENT_BINARY_DIR}/fast_spv)
file(MAKE_DIRECTORY ${_spv_dir})
file(GLOB _fast_glsl_includes ${STARLING_FAST_DIR}/shaders/*.glsl)
set(_spv_files "")
set(_embed_list "")
foreach(_entry IN LISTS STARLING_FAST_SHADERS)
  string(REPLACE "|" ";" _parts "${_entry}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _src)
  list(GET _parts 2 _defs)
  list(GET _parts 3 _nb)
  set(_extra "")
  list(LENGTH _parts _nparts)
  if(_nparts GREATER 4)
    list(GET _parts 4 _extra)
  endif()
  set(_def_args "")
  if(_defs)
    string(REPLACE "," ";" _deflist "${_defs}")
    foreach(_d IN LISTS _deflist)
      list(APPEND _def_args "-D${_d}")
    endforeach()
  endif()
  set(_out ${_spv_dir}/${_name}.spv)
  if(_src MATCHES "\\.spv$")
    # Prebuilt SPIR-V (e.g. hand-assembled probes glslc cannot express).
    add_custom_command(
      OUTPUT ${_out}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${STARLING_FAST_DIR}/shaders/${_src} ${_out}
      DEPENDS ${STARLING_FAST_DIR}/shaders/${_src}
      COMMENT "spv ${_name}")
  else()
  add_custom_command(
    OUTPUT ${_out}
    COMMAND ${STARLING_GLSLC} --target-env=vulkan1.1 -O ${_def_args} ${_extra}
            -I ${STARLING_FAST_DIR}/shaders
            ${STARLING_FAST_DIR}/shaders/${_src} -o ${_out}
    DEPENDS ${STARLING_FAST_DIR}/shaders/${_src} ${_fast_glsl_includes}
    COMMENT "glslc ${_name}" VERBATIM)
  endif()
  list(APPEND _spv_files ${_out})
  list(APPEND _embed_list "${_name}@${_out}@${_nb}")
endforeach()

string(REPLACE ";" "|" _embed_arg "${_embed_list}")
set(_embed_cpp ${CMAKE_CURRENT_BINARY_DIR}/fast_shaders.cpp)
add_custom_command(
  OUTPUT ${_embed_cpp}
  COMMAND ${CMAKE_COMMAND} "-DENTRIES=${_embed_arg}" -DOUT=${_embed_cpp}
          -P ${STARLING_FAST_DIR}/embed_spv.cmake
  DEPENDS ${_spv_files} ${STARLING_FAST_DIR}/embed_spv.cmake
  COMMENT "embed fast-engine SPIR-V"
  VERBATIM)

file(GLOB STARLING_FAST_CPP ${STARLING_FAST_DIR}/*.cpp)
target_sources(starling_ggml_core PRIVATE ${STARLING_FAST_CPP} ${_embed_cpp})
target_compile_definitions(starling_ggml_core PUBLIC STARLING_HAVE_FAST=1)
target_include_directories(starling_ggml_core PUBLIC ${STARLING_FAST_DIR})
if(STARLING_FAST_VK_INCLUDE)
  target_include_directories(starling_ggml_core PUBLIC ${STARLING_FAST_VK_INCLUDE})
endif()
target_link_libraries(starling_ggml_core PUBLIC ${CMAKE_DL_LIBS})
