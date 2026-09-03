cmake_minimum_required(VERSION 3.25)

project(amoeba VERSION 0.1.0 LANGUAGES CXX)

if(APPLE)
    set(defaultGameArenaSdkRoot "$ENV{HOME}/.local")
    set(defaultMlxRoot "/opt/homebrew")
else()
    set(defaultGameArenaSdkRoot "/opt/arena")
    set(defaultMlxRoot "/opt/mlx")
endif()

set(GAME_ARENA_SDK_ROOT "${defaultGameArenaSdkRoot}" CACHE PATH "Prefix where the Game Arena C SDK is installed")
find_package(arena 1.0 CONFIG QUIET HINTS "${GAME_ARENA_SDK_ROOT}")
if(NOT arena_FOUND)
    find_package(arena 0.1 CONFIG REQUIRED HINTS "${GAME_ARENA_SDK_ROOT}")
endif()

set(MLX_ROOT "${defaultMlxRoot}" CACHE PATH "Prefix where MLX is installed")
if(NOT APPLE)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(CUDAToolkit REQUIRED)
    find_package(CUDNN REQUIRED)
endif()

option(AMOEBA_BUILD_MLX_FROM_SOURCE "Build the pinned MLX dependency from source" ON)
option(AMOEBA_ENABLE_CUDA "Build MLX with its CUDA backend on Linux" ON)
option(AMOEBA_BIG_GPU_TRAINING
       "Use the high-throughput self-play and training profile for a large CUDA GPU"
       OFF)

if(AMOEBA_BUILD_MLX_FROM_SOURCE)
    include(FetchContent)
    set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
    if(NOT APPLE)
        set(MLX_BUILD_METAL OFF CACHE BOOL "" FORCE)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND AMOEBA_ENABLE_CUDA)
        set(MLX_BUILD_CUDA ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        mlx
        GIT_REPOSITORY https://github.com/ml-explore/mlx.git
        GIT_TAG v0.32.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(mlx)
else()
    find_package(MLX 0.32 CONFIG REQUIRED HINTS "${MLX_ROOT}")
endif()

# One executable selects online play or training from its command line.
set(AMOEBA_SOURCES
    amoeba/amoeba.cpp
    amoeba/mcts.cpp
    amoeba/network.cpp
    amoeba/transformer_network.cpp
    amoeba/hex_ray_lite_network.cpp
    amoeba/training.cpp
)

add_executable(amoeba_bot amoeba/main.cpp ${AMOEBA_SOURCES})
target_compile_features(amoeba_bot PRIVATE cxx_std_23)
target_link_libraries(amoeba_bot PRIVATE arena::arena mlx)
target_compile_definitions(amoeba_bot PRIVATE
    AMOEBA_BIG_GPU_TRAINING=$<BOOL:${AMOEBA_BIG_GPU_TRAINING}>)

if(NOT APPLE)
    target_link_libraries(amoeba_bot PRIVATE CUDA::cudart)
endif()
