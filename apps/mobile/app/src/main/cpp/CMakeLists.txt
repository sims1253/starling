# JNI bridge for the on-device Starling engine. Configures the repository's
# native engine as a subproject of the Android build and links it into a
# single libstarling_jni.so loaded by the app.
#
# Path note: backends/native/CMakeLists.txt derives the repository root from
# its own location, so the engine, ggml submodule, and patch script all
# resolve correctly regardless of where this file sits.
cmake_minimum_required(VERSION 3.24)  # FetchContent OVERRIDE_FIND_PACKAGE
project(starling_mobile_native LANGUAGES C CXX)

# Locate the repository root by walking up to the directory that contains the
# native engine, instead of counting parent levels by hand.
get_filename_component(_search_dir "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
while(NOT EXISTS "${_search_dir}/backends/native/CMakeLists.txt")
    get_filename_component(_parent "${_search_dir}/.." ABSOLUTE)
    if(_parent STREQUAL _search_dir)
        message(FATAL_ERROR "starling repository root not found above ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    set(_search_dir "${_parent}")
endwhile()
set(STARLING_NATIVE_DIR "${_search_dir}/backends/native")
set(STARLING_REPO_ROOT "${_search_dir}")

# Distributable-artifact settings (see backends/native/CMakeLists.txt):
# host-specific codegen is invalid under the NDK toolchain.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(STARLING_GGML_TESTS OFF CACHE BOOL "" FORCE)
set(STARLING_QUANTIZE OFF CACHE BOOL "" FORCE)
set(STARLING_SERVE OFF CACHE BOOL "" FORCE)

# Without GGML_NATIVE, ggml's arm64 baseline is plain armv8-a: no dot-product
# or FP16 vector kernels, several times slower for quantized models. Every
# mainstream arm64 phone core since 2018 has both; NativeSupport.kt refuses
# the on-device engine on older CPUs before this library is ever loaded.
# Gradle passes STARLING_ANDROID_ARM_ARCH (see app/build.gradle.kts), e.g.
# the +i8mm variant whose 2-row SMMLA kernels speed up Q4_K/Q6_K/Q8_0.
set(STARLING_ANDROID_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "arm64 -march for ggml")
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "${STARLING_ANDROID_ARM_ARCH}" CACHE STRING "" FORCE)
endif()

# Optional Vulkan GPU backend (Gradle: -PstarlingVulkan=true). The NDK ships
# libvulkan and glslc but not the C++ headers (vulkan.hpp) ggml-vulkan needs,
# so a pinned Vulkan-Headers release supplies both vulkan.h and vulkan.hpp
# (they must come from the same version). ggml loads Vulkan entry points
# dynamically, so the minSdk libvulkan stub is enough to link; at runtime the
# device must offer Vulkan 1.2, else ggml registers no GPU and the CPU runs.
option(STARLING_ANDROID_VULKAN "Build the ggml Vulkan backend into the app" OFF)
if(STARLING_ANDROID_VULKAN)
    include(FetchContent)
    FetchContent_Declare(vulkan_headers
        URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.357.tar.gz
        URL_HASH SHA256=7dc0dbcf1d49dd3d7da3761c251c6097dfbaac475321a4a8a99269d3d5abecdc
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    # ggml-vulkan also find_package()s SPIRV-Headers in CONFIG mode; serve
    # it from the matching SDK release instead of a host installation.
    FetchContent_Declare(SPIRV-Headers
        URL https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz
        URL_HASH SHA256=4d703067a7e06331ccb37bdfed3f9b7879cc61969a2689ae95c95db34a47ff07
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        OVERRIDE_FIND_PACKAGE)
    set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
    set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(vulkan_headers SPIRV-Headers)
    set(Vulkan_INCLUDE_DIR "${vulkan_headers_SOURCE_DIR}/include" CACHE PATH "" FORCE)
    # find_program does not search the NDK in a cross build; point at its glslc.
    set(Vulkan_GLSLC_EXECUTABLE "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}/glslc" CACHE FILEPATH "" FORCE)
    if(NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
        message(FATAL_ERROR "glslc not found at ${Vulkan_GLSLC_EXECUTABLE}")
    endif()
    set(STARLING_GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(${STARLING_NATIVE_DIR} starling_native)

if(STARLING_ANDROID_VULKAN)
    # The pinned headers must win over the NDK sysroot's older vulkan.h.
    target_include_directories(ggml-vulkan BEFORE PRIVATE
        "${vulkan_headers_SOURCE_DIR}/include" "${spirv-headers_SOURCE_DIR}/include")
endif()

add_library(starling_jni SHARED starling_jni.cpp)
target_link_libraries(starling_jni PRIVATE starling_ggml)
