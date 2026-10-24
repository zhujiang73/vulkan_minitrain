cmake_minimum_required(VERSION 3.20)
project(vulkan-minitrain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama_train: llama.cpp with this project's training patches merged into the source.
# Lives in src/llama_train (pruned upstream tree: ggml with vulkan/cuda/cpu backends,
# src, common, vendor). Local changes are listed in src/llama_train/LOCAL-CHANGES.md.
# Point -DLLAMA_TRAIN_DIR=<path> at a plain llama.cpp checkout to build against that.
set(LLAMA_TRAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/llama_train"
    CACHE PATH "llama.cpp source tree (llama_train)")

if (NOT EXISTS "${LLAMA_TRAIN_DIR}/include/llama.h")
    message(FATAL_ERROR "LLAMA_TRAIN_DIR is not a llama.cpp tree: ${LLAMA_TRAIN_DIR}")
endif()

# Defaults for the llama.cpp build. A plain set(... CACHE ...) does not override a
# value given on the command line, so -DGGML_CUDA=ON / -DGGML_VULKAN=OFF still work.
# Tests, tools, examples and server stay off: this project only needs the libraries.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "llama: build common utils library")
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "llama: build tests")
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "llama: build tools")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples")
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "llama: build server")
set(GGML_VULKAN          ON  CACHE BOOL "ggml: use Vulkan")
set(GGML_CUDA            OFF CACHE BOOL "ggml: use CUDA")
set(BUILD_SHARED_LIBS    ON  CACHE BOOL "build shared libraries")

add_subdirectory("${LLAMA_TRAIN_DIR}" "${CMAKE_BINARY_DIR}/llama_train")

add_executable(vulkan-minitrain src/minitrain.cpp)

# llama-common provides common_tokenize() and common_opt_dataset_init().
target_link_libraries(vulkan-minitrain PRIVATE llama-common llama)

# Keep the executable next to the backend DLLs so it runs from the build tree.
set_target_properties(vulkan-minitrain PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")

if (MSVC)
    target_compile_options(vulkan-minitrain PRIVATE /utf-8)
endif()
