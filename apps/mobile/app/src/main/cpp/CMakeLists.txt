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

add_subdirectory(${STARLING_NATIVE_DIR} starling_native)

add_library(starling_jni SHARED starling_jni.cpp)
target_link_libraries(starling_jni PRIVATE starling_ggml)
