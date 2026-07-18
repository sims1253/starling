# Starling ggml engine — optional C++ sibling to the pure-Python Starling package.
#
# This builds libstarling_ggml.{so,dylib,dll}, the in-process ggml/CUDA inference
# engine for parakeet-tdt and moss, driven from Python via ctypes
# (src/starling/_ggml/_native.py). It is OPTIONAL: the pure-Python Starling
# package (hatchling, src/starling) keeps working when this C++ is absent — the
# Python side discovers the built .so via _native.available() and silently skips
# the ggml engine if it isn't found.
#
# Build (CUDA, the primary path on NVIDIA):
#   cmake -B build -DSTARLING_GGML_CUDA=ON -DSTARLING_GGML_SHARED=ON
#   cmake --build build -j
#
# Other backends:
#   -DSTARLING_GGML_METAL=ON   (macOS, -DGGML_METAL_EMBED_LIBRARY=ON recommended)
#   -DSTARLING_GGML_VULKAN=ON  (Intel/AMD/ARM)
#   -DSTARLING_GGML_HIP=ON     (AMD ROCm)
#   (unset all -> CPU; verified byte-exact, the portable fallback)
#
# See docs/ggml-engine.md for the full build matrix + cross-platform notes.

cmake_minimum_required(VERSION 3.18)
project(starling_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------------------------------- #
# Options. We expose STARLING_GGML_<BACKEND> and FORCE-set the upstream
# GGML_<BACKEND> cache vars before add_subdirectory(third_party/ggml), mirroring
# parakeet.cpp's pattern (either spelling works at the cmake -D line).
# --------------------------------------------------------------------------- #
option(STARLING_GGML_SHARED "Build libstarling_ggml as shared (for ctypes)" OFF)
option(STARLING_GGML_CUDA   "Forward GGML_CUDA (NVIDIA)" OFF)
option(STARLING_GGML_METAL  "Forward GGML_METAL (Apple)" OFF)
option(STARLING_GGML_VULKAN "Forward GGML_VULKAN (Intel/AMD/ARM)" OFF)
option(STARLING_GGML_HIP    "Forward GGML_HIP (AMD ROCm)" OFF)

set(GGML_CUDA    ${STARLING_GGML_CUDA}    CACHE BOOL "" FORCE)
set(GGML_METAL   ${STARLING_GGML_METAL}   CACHE BOOL "" FORCE)
set(GGML_VULKAN  ${STARLING_GGML_VULKAN}  CACHE BOOL "" FORCE)
set(GGML_HIP     ${STARLING_GGML_HIP}     CACHE BOOL "" FORCE)

# CUDA graphs: ggml defaults this OFF; on CUDA builds it captures/replays the
# compute graph (the per-shape ReplayGraph mechanism the runtime relies on for
# the encoder + K-step decode). Force it on whenever CUDA is on. The runtime
# kill-switch GGML_CUDA_DISABLE_GRAPHS=1 still works at process start.
if(STARLING_GGML_CUDA)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()

# -march=native + tinyBLAS SGEMM: free speedups for local/dev builds. Force ON
# unless the caller overrides (CI/release sets GGML_NATIVE=OFF / passes it
# explicitly for portable distributable binaries).
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

# --------------------------------------------------------------------------- #
# Apply the in-tree ggml patches BEFORE configuring the submodule so the patched
# source is what gets compiled. Idempotent: already-applied patches are skipped.
# (Carried from parakeet.cpp scripts/apply_ggml_patches.sh.)
#
# NOTE for Windows CI: invoke `bash scripts/apply_ggml_patches.sh` explicitly
# under Git Bash first — CMake's find_program(bash) can pick up the stub
# WSL bash.exe in System32 and silently skip the patches (parakeet.cpp's
# documented landmine). A non-zero patch result here is a WARNING, not fatal.
# --------------------------------------------------------------------------- #
find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh")
  execute_process(
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _patch_result
    OUTPUT_VARIABLE _patch_output
    ERROR_VARIABLE _patch_output)
  if(NOT _patch_result EQUAL 0)
    message(WARNING "ggml patch apply returned ${_patch_result}:\n${_patch_output}")
  endif()
endif()

# ggml as a subproject. GGML_STANDALONE auto-detects OFF here, so its
# tests/examples are skipped and the consumer links one target: `ggml`.
add_subdirectory(third_party/ggml)

# --------------------------------------------------------------------------- #
# libstarling_ggml: the Starling-owned engine. Currently a stub (runtime +
# C-API shell come in the next commits); parakeet + moss model sources are
# added as they're written.
# --------------------------------------------------------------------------- #
file(GLOB STARLING_GGML_RUNTIME_CPP ${CMAKE_SOURCE_DIR}/cpp/runtime/*.cpp)
file(GLOB STARLING_GGML_PARAKEET_CPP ${CMAKE_SOURCE_DIR}/cpp/parakeet/*.cpp)
file(GLOB STARLING_GGML_MOSS_CPP ${CMAKE_SOURCE_DIR}/cpp/moss/*.cpp)
set(STARLING_GGML_CPP
    ${STARLING_GGML_RUNTIME_CPP}
    ${STARLING_GGML_PARAKEET_CPP}
    ${STARLING_GGML_MOSS_CPP}
    ${CMAKE_SOURCE_DIR}/cpp/capi.cpp)

if(STARLING_GGML_SHARED)
  add_library(starling_ggml SHARED ${STARLING_GGML_CPP})
  # The C-API header uses plain extern "C" with no __declspec(dllexport), so
  # without this MSVC produces a DLL with zero exported symbols (no import .lib,
  # useless for ctypes). Auto-exports everything. No effect on Linux/macOS.
  set_target_properties(starling_ggml PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(starling_ggml STATIC ${STARLING_GGML_CPP})
endif()

target_include_directories(starling_ggml PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp/include
    PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/runtime
    ${CMAKE_SOURCE_DIR}/cpp/parakeet
    ${CMAKE_SOURCE_DIR}/cpp/moss
    ${CMAKE_SOURCE_DIR}/third_party             # for dr_wav.h
    ${CMAKE_SOURCE_DIR}/third_party/ggml/src    # for ggml-impl.h (cgraph uid field)
    ${CMAKE_SOURCE_DIR}/third_party/ggml/include) # ggml.h etc.
target_compile_definitions(starling_ggml PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_link_libraries(starling_ggml PUBLIC ggml)

# Output the .so next to the build dir root so _native.py's default search path
# (build/libstarling_ggml.so) finds it without env-var help during development.
set_target_properties(starling_ggml PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
