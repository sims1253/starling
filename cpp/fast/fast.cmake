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
if(ANDROID_NDK)
  file(GLOB _ndk_glslc "${ANDROID_NDK}/shader-tools/*/glslc")
  foreach(_g IN LISTS _ndk_glslc)
    get_filename_component(_d "${_g}" DIRECTORY)
    list(APPEND _glslc_hints "${_d}")
  endforeach()
endif()
if(DEFINED ENV{VULKAN_SDK})
  list(APPEND _glslc_hints "$ENV{VULKAN_SDK}/bin")
endif()
find_program(STARLING_GLSLC NAMES glslc HINTS ${_glslc_hints} NO_CMAKE_FIND_ROOT_PATH)
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
  "norm|norm.comp||6"
  "softmax|softmax.comp||3"
  "pk_conv|pk_conv.comp||5"
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
  set(_def_args "")
  if(_defs)
    string(REPLACE "," ";" _deflist "${_defs}")
    foreach(_d IN LISTS _deflist)
      list(APPEND _def_args "-D${_d}")
    endforeach()
  endif()
  set(_out ${_spv_dir}/${_name}.spv)
  add_custom_command(
    OUTPUT ${_out}
    COMMAND ${STARLING_GLSLC} --target-env=vulkan1.1 -O ${_def_args}
            -I ${STARLING_FAST_DIR}/shaders
            ${STARLING_FAST_DIR}/shaders/${_src} -o ${_out}
    DEPENDS ${STARLING_FAST_DIR}/shaders/${_src} ${_fast_glsl_includes}
    COMMENT "glslc ${_name}"
    VERBATIM)
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
