# Starling's native engine, serving API, and quantizer.
# CPU builds need no Python runtime or accelerator SDK. The optional shared
# library also supports existing Python ctypes consumers in src/starling/_ggml.
#
# Build from the repository root:
#   cmake -B build -DSTARLING_SERVE=ON -DSTARLING_QUANTIZE=ON
#   cmake --build build -j --target starling-serve starling-quantize
#
# Other backends:
#   -DSTARLING_GGML_METAL=ON   (macOS, -DGGML_METAL_EMBED_LIBRARY=ON recommended)
#   -DSTARLING_GGML_VULKAN=ON  (Intel/AMD/ARM)
#   -DSTARLING_GGML_HIP=ON     (AMD ROCm)
#   -DSTARLING_GGML_CUDA=ON    (optional NVIDIA accelerator)
#   (all OFF -> CPU)
#
# See docs/ggml-engine.md for the full build matrix + cross-platform notes.

cmake_minimum_required(VERSION 3.18)
project(starling_ggml LANGUAGES C CXX)

get_filename_component(STARLING_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------------------------------- #
# Options. We expose STARLING_GGML_<BACKEND> and FORCE-set the upstream
# GGML_<BACKEND> cache vars before configuring ggml. Set the STARLING-prefixed
# option at the command line; it owns the upstream option's value.
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
# Validate/apply the complete patch series before configuring ggml.
# Prefer Git Bash on Windows over the System32 WSL launcher. An explicit
# -DBASH_EXECUTABLE remains available for nonstandard installations.
if(WIN32 AND NOT BASH_EXECUTABLE)
  find_package(Git REQUIRED)
  get_filename_component(_git_bin "${GIT_EXECUTABLE}" DIRECTORY)
  find_program(BASH_EXECUTABLE bash
    PATHS "${_git_bin}/../bin" "${_git_bin}/../usr/bin"
    NO_DEFAULT_PATH)
endif()
find_program(BASH_EXECUTABLE bash REQUIRED)
execute_process(
  COMMAND "${BASH_EXECUTABLE}" "${STARLING_REPO_ROOT}/scripts/apply_ggml_patches.sh"
  WORKING_DIRECTORY "${STARLING_REPO_ROOT}"
  RESULT_VARIABLE _patch_result
  OUTPUT_VARIABLE _patch_output
  ERROR_VARIABLE _patch_error)
if(NOT _patch_result EQUAL 0)
  message(FATAL_ERROR "ggml patch validation failed:\n${_patch_output}${_patch_error}")
endif()

# ggml as a subproject. GGML_STANDALONE auto-detects OFF here, so its
# tests/examples are skipped and the consumer links one target: `ggml`.
add_subdirectory("${STARLING_REPO_ROOT}/third_party/ggml" "${CMAKE_BINARY_DIR}/ggml")

# --------------------------------------------------------------------------- #
# libstarling_ggml: the Starling-owned engine (runtime + flat C API + the
# per-model implementations under cpp/<model>/).
# --------------------------------------------------------------------------- #
file(GLOB STARLING_GGML_RUNTIME_CPP ${STARLING_REPO_ROOT}/cpp/runtime/*.cpp)
file(GLOB STARLING_GGML_LIB_CPP ${STARLING_REPO_ROOT}/cpp/lib/*.cpp)
file(GLOB STARLING_GGML_PARAKEET_CPP ${STARLING_REPO_ROOT}/cpp/parakeet/*.cpp)
file(GLOB STARLING_GGML_MOSS_CPP ${STARLING_REPO_ROOT}/cpp/moss/*.cpp)
file(GLOB STARLING_GGML_ARK_CPP ${STARLING_REPO_ROOT}/cpp/ark/*.cpp)
file(GLOB STARLING_GGML_HIGGS_CPP ${STARLING_REPO_ROOT}/cpp/higgs/*.cpp)
file(GLOB STARLING_GGML_HOJO_CPP ${STARLING_REPO_ROOT}/cpp/hojo/*.cpp)
file(GLOB STARLING_GGML_GRANITE_CPP ${STARLING_REPO_ROOT}/cpp/granite/*.cpp)
file(GLOB STARLING_GGML_QWEN3_CPP ${STARLING_REPO_ROOT}/cpp/qwen3/*.cpp)
file(GLOB STARLING_GGML_S1_CPP ${STARLING_REPO_ROOT}/cpp/s1/*.cpp)
file(GLOB STARLING_GGML_AUDEX_CPP ${STARLING_REPO_ROOT}/cpp/audex/*.cpp)
set(STARLING_GGML_CPP
    ${STARLING_GGML_RUNTIME_CPP}
    ${STARLING_GGML_LIB_CPP}
    ${STARLING_GGML_PARAKEET_CPP}
    ${STARLING_GGML_MOSS_CPP}
    ${STARLING_GGML_ARK_CPP}
    ${STARLING_GGML_HIGGS_CPP}
    ${STARLING_GGML_HOJO_CPP}
    ${STARLING_GGML_GRANITE_CPP}
    ${STARLING_GGML_QWEN3_CPP}
    ${STARLING_GGML_S1_CPP}
    ${STARLING_GGML_AUDEX_CPP}
    ${STARLING_REPO_ROOT}/cpp/capi.cpp)

add_library(starling_ggml_core OBJECT ${STARLING_GGML_CPP})
set_target_properties(starling_ggml_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(starling_ggml_core PUBLIC
    ${STARLING_REPO_ROOT}/cpp/include ${STARLING_REPO_ROOT}/cpp
    ${STARLING_REPO_ROOT}/cpp/runtime ${STARLING_REPO_ROOT}/cpp/parakeet
    ${STARLING_REPO_ROOT}/cpp/moss ${STARLING_REPO_ROOT}/cpp/ark ${STARLING_REPO_ROOT}/cpp/higgs
    ${STARLING_REPO_ROOT}/cpp/hojo ${STARLING_REPO_ROOT}/cpp/granite
    ${STARLING_REPO_ROOT}/third_party
    ${STARLING_REPO_ROOT}/third_party/ggml/src ${STARLING_REPO_ROOT}/third_party/ggml/include)
target_compile_definitions(starling_ggml_core PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_link_libraries(starling_ggml_core PUBLIC ggml)

if(STARLING_GGML_SHARED)
  add_library(starling_ggml SHARED $<TARGET_OBJECTS:starling_ggml_core>)
  set_target_properties(starling_ggml PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(starling_ggml STATIC $<TARGET_OBJECTS:starling_ggml_core>)
endif()
set_target_properties(starling_ggml PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_include_directories(starling_ggml PUBLIC ${STARLING_REPO_ROOT}/cpp/include)
target_link_libraries(starling_ggml PUBLIC ggml)

if(STARLING_GGML_TESTS)
  set(STARLING_GGML_TEST_SUPPORT_CPP ${STARLING_REPO_ROOT}/cpp/tests/golden_io.cpp)
  file(GLOB STARLING_GGML_TEST_MAIN_CPP ${STARLING_REPO_ROOT}/cpp/tests/*_test.cpp)
  foreach(test_src IN LISTS STARLING_GGML_TEST_MAIN_CPP)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} ${STARLING_GGML_TEST_SUPPORT_CPP} $<TARGET_OBJECTS:starling_ggml_core>)
    target_include_directories(${test_name} PRIVATE
        ${STARLING_REPO_ROOT}/cpp ${STARLING_REPO_ROOT}/cpp/tests
        ${STARLING_REPO_ROOT}/third_party ${STARLING_REPO_ROOT}/third_party/ggml/include)
    target_link_libraries(${test_name} PRIVATE ggml)

    # The stream_session_test needs the serve sources (server.cpp,
    # stream_session.cpp, audio.cpp) + cpp-httplib's header path.
    if(test_name STREQUAL "stream_session_test")
      target_sources(${test_name} PRIVATE
          ${STARLING_REPO_ROOT}/cpp/serve/server.cpp
          ${STARLING_REPO_ROOT}/cpp/serve/stream_session.cpp
          ${STARLING_REPO_ROOT}/cpp/serve/audio.cpp)
      target_include_directories(${test_name} PRIVATE
          ${STARLING_REPO_ROOT}/cpp/include
          ${STARLING_REPO_ROOT}/cpp/serve)
      find_package(Threads REQUIRED)
      target_link_libraries(${test_name} PRIVATE Threads::Threads)
    endif()

    # The audio_parser_test covers the pure WAV/multipart parsers in
    # cpp/serve/audio.cpp (issue #12 hardening) — no model, CPU-safe.
    if(test_name STREQUAL "audio_parser_test")
      target_sources(${test_name} PRIVATE
          ${STARLING_REPO_ROOT}/cpp/serve/audio.cpp)
      target_include_directories(${test_name} PRIVATE
          ${STARLING_REPO_ROOT}/cpp/serve)
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
# starling-quantize: the calibrated-quantization CLI for Starling GGUFs.
#
# Post-processes an f32/f16/bf16 GGUF into a quantized one (k-quants + q8_0),
# optionally weighted by an importance matrix collected on calibration audio
# (STARLING_IMATRIX via the engine). See docs/quantization.md.
option(STARLING_QUANTIZE "Build the starling-quantize GGUF quantizer CLI" ON)
if(STARLING_QUANTIZE)
  add_executable(starling-quantize ${STARLING_REPO_ROOT}/cpp/tools/starling_quantize.cpp)
  target_include_directories(starling-quantize PRIVATE
      ${STARLING_REPO_ROOT}/cpp
      ${STARLING_REPO_ROOT}/cpp/runtime
      ${STARLING_REPO_ROOT}/third_party/ggml/include
      ${STARLING_REPO_ROOT}/third_party/ggml/src)
  target_link_libraries(starling-quantize PRIVATE ggml)
  set_target_properties(starling-quantize PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # mel_dump: debug aid — dump the engine mel features of a wav (see
  # docs/quantization.md and the converter's mel contract).
  add_executable(mel_dump ${STARLING_REPO_ROOT}/cpp/tools/mel_dump.cpp
      $<TARGET_OBJECTS:starling_ggml_core>)
  target_include_directories(mel_dump PRIVATE
      ${STARLING_REPO_ROOT}/cpp
      ${STARLING_REPO_ROOT}/cpp/runtime ${STARLING_REPO_ROOT}/cpp/parakeet
      ${STARLING_REPO_ROOT}/third_party/ggml/include)
  target_link_libraries(mel_dump PRIVATE ggml)
  set_target_properties(mel_dump PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

# --------------------------------------------------------------------------- #
# starling-serve: the native HTTP/WebSocket serving binary.
#
# A self-contained executable that wraps libstarling_ggml behind the same
# HTTP/WS endpoint surface as the Python `starling.server` (client-visible
# differences are documented in docs/native-serving.md):
#
#   spawn("starling-serve", ["--model", slug, "--gguf", path, "--port", "8181"])
#
# Build (CPU-only for development / CI smoke tests):
#   cmake -B build -DSTARLING_SERVE=ON
#   cmake --build build -j --target starling-serve
#
# Build (optional CUDA acceleration on NVIDIA):
#   cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON
#   cmake --build build -j --target starling-serve
#
# Build (Vulkan, the universal Intel/AMD/ARM path):
#   cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_VULKAN=ON
#   cmake --build build -j --target starling-serve
#
# The binary links statically against starling_ggml_core (no separate .so to
# ship). On Windows, pass -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded to use
# the static CRT consistently for the executable, engine, and ggml targets.
# --------------------------------------------------------------------------- #
option(STARLING_SERVE "Build the starling-serve native HTTP/WS binary" OFF)

# Starling serve version (bumped with the GGML ABI; checked by --version).
set(STARLING_SERVE_VERSION "0.1.0" CACHE STRING "starling-serve version string")

if(STARLING_SERVE)
  set(STARLING_SERVE_CPP
      ${STARLING_REPO_ROOT}/cpp/serve/main.cpp
      ${STARLING_REPO_ROOT}/cpp/serve/server.cpp
      ${STARLING_REPO_ROOT}/cpp/serve/stream_session.cpp
      ${STARLING_REPO_ROOT}/cpp/serve/audio.cpp)

  add_executable(starling-serve
      ${STARLING_SERVE_CPP}
      $<TARGET_OBJECTS:starling_ggml_core>)

  target_include_directories(starling-serve PRIVATE
      ${STARLING_REPO_ROOT}/cpp/include
      ${STARLING_REPO_ROOT}/cpp/serve
      ${STARLING_REPO_ROOT}/cpp
      ${STARLING_REPO_ROOT}/third_party
      ${STARLING_REPO_ROOT}/third_party/ggml/include)

  # Version string baked into the binary for --version.
  target_compile_definitions(starling-serve PRIVATE
      STARLING_SERVE_VERSION="${STARLING_SERVE_VERSION}")

  # cpp-httplib needs threads on Unix.
  find_package(Threads REQUIRED)
  target_link_libraries(starling-serve PRIVATE ggml Threads::Threads)

  set_target_properties(starling-serve PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

# An explicit test target exercises real HTTP handling with a deterministic
# engine double. It needs no model files and is never shipped as starling-serve.
if(STARLING_GGML_TESTS)
  find_package(Threads REQUIRED)
  add_executable(starling-serve-contract-fixture EXCLUDE_FROM_ALL
    ${STARLING_REPO_ROOT}/cpp/serve/main.cpp
    ${STARLING_REPO_ROOT}/cpp/serve/server.cpp
    ${STARLING_REPO_ROOT}/cpp/serve/stream_session.cpp
    ${STARLING_REPO_ROOT}/cpp/serve/audio.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/fixture_engine.cpp)
  target_include_directories(starling-serve-contract-fixture PRIVATE
    ${STARLING_REPO_ROOT}/cpp/include ${STARLING_REPO_ROOT}/cpp
    ${STARLING_REPO_ROOT}/cpp/serve ${STARLING_REPO_ROOT}/third_party)
  target_link_libraries(starling-serve-contract-fixture PRIVATE Threads::Threads)
  set_target_properties(starling-serve-contract-fixture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
