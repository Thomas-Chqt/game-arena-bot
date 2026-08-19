cmake_minimum_required(VERSION 3.22)

project(game_arena_bots VERSION 0.1.0 LANGUAGES CXX)

set(GAME_ARENA_SDK_ROOT "$ENV{HOME}/.local" CACHE PATH "Prefix where the Game Arena C SDK is installed")
find_package(arena 0.6 CONFIG REQUIRED HINTS "${GAME_ARENA_SDK_ROOT}")

set(MLX_ROOT "/opt/homebrew" CACHE PATH "Prefix where MLX is installed")
find_package(MLX 0.32 CONFIG REQUIRED HINTS "${MLX_ROOT}")

add_subdirectory(games/amoeba)
add_subdirectory(amoeba_bot_1)
