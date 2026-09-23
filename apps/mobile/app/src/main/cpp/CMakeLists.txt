# JNI bridge for the on-device Starling engine. Configures the repository's
# native engine as a subproject of the Android build and links it into a
# single libstarling_jni.so loaded by the app.
#
# Path note: backends/native/CMakeLists.txt derives the repository root from
# its own location, so the engine, ggml submodule, and patch script all
# resolve correctly regardless of where this file sits.
cmake_minimum_required(VERSION 3.22)
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

add_subdirectory(${STARLING_NATIVE_DIR} starling_native)

add_library(starling_jni SHARED starling_jni.cpp)
target_link_libraries(starling_jni PRIVATE starling_ggml)
