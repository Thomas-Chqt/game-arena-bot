cmake_minimum_required(VERSION 3.22)

project(amoeba VERSION 0.1.0 LANGUAGES CXX)

set(GAME_ARENA_SDK_ROOT "$ENV{HOME}/.local" CACHE PATH "Prefix where the Game Arena C SDK is installed")
find_package(arena 0.1.0 CONFIG REQUIRED HINTS "${GAME_ARENA_SDK_ROOT}")

set(MLX_ROOT "/opt/homebrew" CACHE PATH "Prefix where MLX is installed")
find_package(MLX 0.32 CONFIG REQUIRED HINTS "${MLX_ROOT}")

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
