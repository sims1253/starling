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
option(STARLING_GGML_TESTS  "Build Starling C++ stage tests" ON)

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

# GGML_NATIVE and GGML_LLAMAFILE default ON for best local performance
# (-march=native, tinyBLAS SGEMM tuned to this host). When building a
# DISTRIBUTABLE artifact (.so / wheel for other machines), pass
# -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF to avoid host-specific codegen.
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
# documented landmine). On Windows a non-zero patch result stays a WARNING
# (that same bash.exe environment issue can cause spurious failures); on
# Unix it is FATAL_ERROR — we refuse to silently ship an unpatched ggml.
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
    if(WIN32)
      message(WARNING "ggml patch apply returned ${_patch_result} (skipped on Windows):\n${_patch_output}")
    else()
      message(FATAL_ERROR "ggml patch apply failed (rc=${_patch_result}). Refusing to build an unpatched ggml:\n${_patch_output}")
    endif()
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
file(GLOB STARLING_GGML_LIB_CPP ${CMAKE_SOURCE_DIR}/cpp/lib/*.cpp)
file(GLOB STARLING_GGML_PARAKEET_CPP ${CMAKE_SOURCE_DIR}/cpp/parakeet/*.cpp)
file(GLOB STARLING_GGML_MOSS_CPP ${CMAKE_SOURCE_DIR}/cpp/moss/*.cpp)
file(GLOB STARLING_GGML_ARK_CPP ${CMAKE_SOURCE_DIR}/cpp/ark/*.cpp)
file(GLOB STARLING_GGML_HIGGS_CPP ${CMAKE_SOURCE_DIR}/cpp/higgs/*.cpp)
file(GLOB STARLING_GGML_HOJO_CPP ${CMAKE_SOURCE_DIR}/cpp/hojo/*.cpp)
set(STARLING_GGML_CPP
    ${STARLING_GGML_RUNTIME_CPP}
    ${STARLING_GGML_LIB_CPP}
    ${STARLING_GGML_PARAKEET_CPP}
    ${STARLING_GGML_MOSS_CPP}
    ${STARLING_GGML_ARK_CPP}
    ${STARLING_GGML_HIGGS_CPP}
    ${STARLING_GGML_HOJO_CPP}
    ${CMAKE_SOURCE_DIR}/cpp/capi.cpp)

add_library(starling_ggml_core OBJECT ${STARLING_GGML_CPP})
set_target_properties(starling_ggml_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(starling_ggml_core PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp/include ${CMAKE_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/cpp/runtime ${CMAKE_SOURCE_DIR}/cpp/parakeet
    ${CMAKE_SOURCE_DIR}/cpp/moss ${CMAKE_SOURCE_DIR}/cpp/ark ${CMAKE_SOURCE_DIR}/cpp/higgs
    ${CMAKE_SOURCE_DIR}/cpp/hojo
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/ggml/src ${CMAKE_SOURCE_DIR}/third_party/ggml/include)
target_compile_definitions(starling_ggml_core PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_link_libraries(starling_ggml_core PUBLIC ggml)

if(STARLING_GGML_SHARED)
  add_library(starling_ggml SHARED $<TARGET_OBJECTS:starling_ggml_core>)
  set_target_properties(starling_ggml PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(starling_ggml STATIC $<TARGET_OBJECTS:starling_ggml_core>)
endif()
target_include_directories(starling_ggml PUBLIC ${CMAKE_SOURCE_DIR}/cpp/include)
target_link_libraries(starling_ggml PUBLIC ggml)

if(STARLING_GGML_TESTS)
  set(STARLING_GGML_TEST_SUPPORT_CPP ${CMAKE_SOURCE_DIR}/cpp/tests/golden_io.cpp)
  file(GLOB STARLING_GGML_TEST_MAIN_CPP ${CMAKE_SOURCE_DIR}/cpp/tests/*_test.cpp)
  foreach(test_src IN LISTS STARLING_GGML_TEST_MAIN_CPP)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} ${STARLING_GGML_TEST_SUPPORT_CPP} $<TARGET_OBJECTS:starling_ggml_core>)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp ${CMAKE_SOURCE_DIR}/cpp/tests
        ${CMAKE_SOURCE_DIR}/third_party ${CMAKE_SOURCE_DIR}/third_party/ggml/include)
    target_link_libraries(${test_name} PRIVATE ggml)

    # The stream_session_test needs the serve sources (server.cpp,
    # stream_session.cpp, audio.cpp) + cpp-httplib's header path.
    if(test_name STREQUAL "stream_session_test")
      target_sources(${test_name} PRIVATE
          ${CMAKE_SOURCE_DIR}/cpp/serve/server.cpp
          ${CMAKE_SOURCE_DIR}/cpp/serve/stream_session.cpp
          ${CMAKE_SOURCE_DIR}/cpp/serve/audio.cpp)
      target_include_directories(${test_name} PRIVATE
          ${CMAKE_SOURCE_DIR}/cpp/include
          ${CMAKE_SOURCE_DIR}/cpp/serve)
      find_package(Threads REQUIRED)
      target_link_libraries(${test_name} PRIVATE Threads::Threads)
    endif()

    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach()
endif()

# Output the .so next to the build dir root so _native.py's default search path
# (build/libstarling_ggml.so) finds it without env-var help during development.
set_target_properties(starling_ggml PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


# --------------------------------------------------------------------------- #
# starling-serve: the native HTTP/WebSocket serving binary.
#
# A self-contained executable that wraps libstarling_ggml behind the same
# HTTP/WS API as the Python `starling.server`, so freestyle's TypeScript
# supervisor is a drop-in replacement:
#
#   spawn("starling-serve", ["--model", slug, "--gguf", path, "--port", "8181"])
#
# Build (CPU-only for development / CI smoke tests):
#   cmake -B build -DSTARLING_SERVE=ON
#   cmake --build build -j --target starling-serve
#
# Build (CUDA, the freestyle production path):
#   cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON
#   cmake --build build -j --target starling-serve
#
# Build (Vulkan, the universal Intel/AMD/ARM path):
#   cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_VULKAN=ON
#   cmake --build build -j --target starling-serve
#
# The binary links statically against starling_ggml_core (no separate .so to
# ship). On Windows it has no Python / VC++ runtime dependency (static CRT
# recommended via CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded).
# --------------------------------------------------------------------------- #
option(STARLING_SERVE "Build the starling-serve native HTTP/WS binary" OFF)

# Starling serve version (bumped with the GGML ABI; checked by --version).
set(STARLING_SERVE_VERSION "0.1.0" CACHE STRING "starling-serve version string")

if(STARLING_SERVE)
  set(STARLING_SERVE_CPP
      ${CMAKE_SOURCE_DIR}/cpp/serve/main.cpp
      ${CMAKE_SOURCE_DIR}/cpp/serve/server.cpp
      ${CMAKE_SOURCE_DIR}/cpp/serve/stream_session.cpp
      ${CMAKE_SOURCE_DIR}/cpp/serve/audio.cpp)

  add_executable(starling-serve
      ${STARLING_SERVE_CPP}
      $<TARGET_OBJECTS:starling_ggml_core>)

  target_include_directories(starling-serve PRIVATE
      ${CMAKE_SOURCE_DIR}/cpp/include
      ${CMAKE_SOURCE_DIR}/cpp/serve
      ${CMAKE_SOURCE_DIR}/cpp
      ${CMAKE_SOURCE_DIR}/third_party
      ${CMAKE_SOURCE_DIR}/third_party/ggml/include)

  # Version string baked into the binary for --version.
  target_compile_definitions(starling-serve PRIVATE
      STARLING_SERVE_VERSION="${STARLING_SERVE_VERSION}")

  # cpp-httplib needs threads on Unix.
  find_package(Threads REQUIRED)
  target_link_libraries(starling-serve PRIVATE ggml Threads::Threads)

  # Static CRT on Windows (no VC++ redistributable dependency).
  if(WIN32 AND MSVC)
    set_property(TARGET starling-serve PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()

  set_target_properties(starling-serve PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
